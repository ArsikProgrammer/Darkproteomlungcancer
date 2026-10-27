"""Build a darkness annotation from AlphaFold pLDDT in the local tar archive.

Darkness definition: pLDDT < 70 (low confidence / likely disordered).
"""
import re
from pathlib import Path
import argparse
import gzip
import json
import tarfile
import urllib.request
import pandas as pd


UNIPROT_URL = (
    "https://rest.uniprot.org/uniprotkb/search?query=organism_id:9606%20AND%20reviewed:true"
    "&format=json&fields=accession,gene_names&size=500"
)


def fetch_uniprot_mapping(cache_path: Path):
    if cache_path.exists():
        return pd.read_csv(cache_path)
    rows = []
    url = UNIPROT_URL
    while url:
        with urllib.request.urlopen(url, timeout=60) as r:
            data = json.load(r)
            link = r.headers.get("Link", "")
        for e in data.get("results", []):
            acc = e.get("primaryAccession")
            for g in e.get("genes", []):
                name = g.get("geneName", {}).get("value")
                if acc and name:
                    rows.append({"accession": acc, "gene": name})
        m = re.search(r"<([^>]+)>", link)
        url = m.group(1) if (m and 'rel="next"' in link) else None
    df = pd.DataFrame(rows).drop_duplicates()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def parse_pdb_plddt(pdb_text: str):
    res = {}
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        if atom_name != "CA":
            continue
        try:
            resnum = int(line[22:26])
            bfactor = float(line[60:66])
        except ValueError:
            continue
        if resnum not in res:
            res[resnum] = bfactor
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mutations", nargs="+", required=True)
    parser.add_argument("--tar", default="data/raw/UP000005640_9606_HUMAN_v6.tar")
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=70.0)
    parser.add_argument("--min_coverage", type=float, default=0.9)


    args = parser.parse_args()

    genes = set()
    for m in args.mutations:
        df = pd.read_csv(m, low_memory=False)
        genes.update(df["Hugo_Symbol"].dropna().astype(str).tolist())
    print("Unique genes needed:", len(genes))

    cache = Path("data/processed/_uniprot_gene_accession.csv")
    mapping = fetch_uniprot_mapping(cache)
    mapping = mapping[mapping["gene"].isin(genes)]
    needed_acc = set(mapping["accession"].tolist())
    print("Mapped genes:", mapping["gene"].nunique(), "accessions:", len(needed_acc))

    acc_plddt = {}
    tar_path = Path(args.tar)
    truncated = False
    with tar_path.open("rb") as f:
        tf = tarfile.open(fileobj=f, mode="r:")
        processed = 0
        try:
            while True:
                member = tf.next()
                if member is None:
                    break
                if not member.name.endswith(".pdb.gz"):
                    continue
                acc = member.name.split("-")[1]
                if acc not in needed_acc:
                    continue
                try:
                    data = tf.extractfile(member).read()
                    pdb = gzip.decompress(data).decode("utf-8", "ignore")
                except Exception:
                    continue
                plddt = parse_pdb_plddt(pdb)
                if plddt:
                    acc_plddt[acc] = plddt
                    processed += 1
                    if processed % 100 == 0:
                        print("Parsed proteins:", processed)
        except Exception as e:
            truncated = True
            print("Tar stream ended:", type(e).__name__, str(e)[:200])

    print("Total parsed proteins:", len(acc_plddt))
    if truncated:
        raise RuntimeError("AlphaFold tar archive is truncated; coverage is incomplete")
    coverage = len(acc_plddt) / len(needed_acc) if needed_acc else 1.0
    print("Protein coverage:", round(coverage, 4))
    if coverage < args.min_coverage:
        raise RuntimeError(f"AlphaFold coverage {coverage:.3f} below required {args.min_coverage}")


    rows = []
    for m in args.mutations:
        mut = pd.read_csv(m, low_memory=False)
        for _, r in mut[["Hugo_Symbol", "Protein_position"]].drop_duplicates().iterrows():
            gene = str(r["Hugo_Symbol"])
            pos = r["Protein_position"]
            try:
                pos = int(pos)
            except (ValueError, TypeError):
                continue
            accs = mapping[mapping["gene"] == gene]["accession"].tolist()
            plddt_val = None
            for acc in accs:
                if acc in acc_plddt and pos in acc_plddt[acc]:
                    plddt_val = acc_plddt[acc][pos]
                    break
            if plddt_val is None:
                continue
            rows.append({
                "Hugo_Symbol": gene,
                "Protein_position": pos,
                "is_dark": int(plddt_val < args.threshold),
                "plddt": plddt_val,
                "source": "alphafold_plddt",
            })

    out = pd.DataFrame(rows).drop_duplicates()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print("Darkness annotation rows:", len(out), "dark:", int(out.is_dark.sum()) if len(out) else 0)
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
