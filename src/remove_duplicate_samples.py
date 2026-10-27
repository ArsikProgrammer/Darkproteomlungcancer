"""Remove duplicate samples across cohorts using overlap coefficient >=0.5."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

def load(name):
    df = pd.read_csv(PROC / name, low_memory=False)
    df["key"] = (df["Chromosome"].astype(str) + ":" + df["Start_Position"].astype(str)
                 + ":" + df["Reference_Allele"].astype(str) + ":" + df["Tumor_Seq_Allele2"].astype(str))
    return df

def overlap_pairs(a, b):
    ag = a.groupby("Tumor_Sample_Barcode")["key"].apply(set)
    bg = b.groupby("Tumor_Sample_Barcode")["key"].apply(set)
    out = []
    for sa, va in ag.items():
        for sb, vb in bg.items():
            denom = min(len(va), len(vb))
            if denom == 0:
                continue
            coef = len(va & vb) / denom
            if coef >= 0.5:
                out.append((sa, sb, len(va & vb), len(va), len(vb), coef))
    return out

def decide_kept(sa, sb, na, nb, cohort_a, cohort_b):
    # George (WGS) > WES; if equal keep George
    if cohort_a == "George" and cohort_b != "George":
        return "George", cohort_a, sa, cohort_b, sb
    if cohort_b == "George" and cohort_a != "George":
        return "George", cohort_b, sb, cohort_a, sa
    if na >= nb:
        return cohort_a, cohort_a, sa, cohort_b, sb
    return cohort_b, cohort_b, sb, cohort_a, sa

def main():
    george = load("george_coding_snps.csv")
    jhu = load("sclc_jhu_coding_snps.csv")
    clcgp = load("sclc_clcgp_coding_snps.csv")

    cohorts = {"George": george, "JHU": jhu, "CLCGP": clcgp}
    pairs = []
    pairs += [("George", "JHU", *p) for p in overlap_pairs(george, jhu)]
    pairs += [("George", "CLCGP", *p) for p in overlap_pairs(george, clcgp)]
    pairs += [("JHU", "CLCGP", *p) for p in overlap_pairs(jhu, clcgp)]

    removed = set()
    log_rows = []
    for cohort_a, cohort_b, sa, sb, inter, na, nb, coef in pairs:
        kept_cohort, kept_name, kept_sample, removed_cohort, removed_sample = decide_kept(
            sa, sb, na, nb, cohort_a, cohort_b)
        removed.add((removed_cohort, removed_sample))
        log_rows.append({
            "cohort_a": cohort_a, "sample_a": sa,
            "cohort_b": cohort_b, "sample_b": sb,
            "overlap_coeff": coef,
            "kept": kept_cohort, "removed": f"{removed_cohort} {removed_sample}",
        })

    # within-cohort duplicates
    for cohort, df in cohorts.items():
        g = df.groupby("Tumor_Sample_Barcode")["key"].apply(set)
        names = list(g.index)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                va, vb = g[names[i]], g[names[j]]
                denom = min(len(va), len(vb))
                if denom == 0:
                    continue
                coef = len(va & vb) / denom
                if coef >= 0.5:
                    if len(va) >= len(vb):
                        removed.add((cohort, names[j]))
                        kept_sample = names[i]
                    else:
                        removed.add((cohort, names[i]))
                        kept_sample = names[j]
                    log_rows.append({
                        "cohort_a": cohort, "sample_a": names[i],
                        "cohort_b": cohort, "sample_b": names[j],
                        "overlap_coeff": coef,
                        "kept": f"{cohort} {kept_sample}",
                        "removed": f"{cohort} {names[j] if len(va) >= len(vb) else names[i]}",
                    })

    pd.DataFrame(log_rows).to_csv(PROC / "duplicate_removal_log_full.csv", index=False)
    removed_samples = {r.split()[-1] for r in pd.read_csv(PROC / "duplicate_removal_log_full.csv")["removed"].astype(str)}
    print("Removed samples:", len(removed_samples))

    # filter context counts
    for inp, outp in [("tumor_context_counts.csv", "tumor_context_counts_dedup.csv"),
                      ("sample_features.csv", "sample_features_dedup.csv")]:
        p = PROC / inp
        if not p.exists():
            continue
        df = pd.read_csv(p, low_memory=False)
        before = df["Tumor_Sample_Barcode"].nunique() if "Tumor_Sample_Barcode" in df.columns else len(df)
        df2 = df[~df["Tumor_Sample_Barcode"].isin(removed_samples)].copy()
        after = df2["Tumor_Sample_Barcode"].nunique() if "Tumor_Sample_Barcode" in df2.columns else len(df2)
        df2.to_csv(PROC / outp, index=False)
        print(f"{inp}: samples {before} -> {after}, rows {len(df)} -> {len(df2)}")

    print("Saved duplicate_removal_log_full.csv with", len(log_rows), "pairs")


if __name__ == "__main__":
    main()
