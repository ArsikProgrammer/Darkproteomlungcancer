"""Annotate coding SNPs with a darkness label from an external darkness track."""
from pathlib import Path
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mutations", required=True, help="coding SNPs CSV")
    parser.add_argument("--darkness", required=True, help="darkness annotation CSV: Hugo_Symbol,Protein_position,is_dark")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    mut = pd.read_csv(args.mutations, low_memory=False)
    dark = pd.read_csv(args.darkness, low_memory=False)

    required = {"Hugo_Symbol", "Protein_position", "is_dark"}
    if not required.issubset(dark.columns):
        raise ValueError(f"Darkness file must contain columns: {required}")

    dark = dark[["Hugo_Symbol", "Protein_position", "is_dark"]].drop_duplicates()
    dark["Protein_position"] = pd.to_numeric(dark["Protein_position"], errors="coerce")
    mut["Protein_position"] = pd.to_numeric(mut["Protein_position"], errors="coerce")

    out = mut.merge(dark, on=["Hugo_Symbol", "Protein_position"], how="left", validate="many_to_one")
    out["is_dark"] = out["is_dark"]  # keep unknown as NaN; do not treat missing pLDDT as ordered
    out["has_plddt"] = out["is_dark"].notna()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print("Rows:", len(out), "with pLDDT:", int(out["has_plddt"].sum()),
          "missing:", int((~out["has_plddt"]).sum()))

    print("Saved:", args.output)


if __name__ == "__main__":
    main()
