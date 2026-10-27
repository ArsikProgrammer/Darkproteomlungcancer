"""Compute per-sample O/E for mutations in IDR and fit a model."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mutations", required=True, help="annotated mutations with context_96 and is_dark")
    parser.add_argument("--background", required=True, help="CSV: context_96,dark_fraction")
    parser.add_argument("--signatures", required=True, help="tumor signature activities")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    mut = pd.read_csv(args.mutations, low_memory=False)
    bg = pd.read_csv(args.background)
    sig = pd.read_csv(args.signatures, low_memory=False)

    if "context_96" not in mut.columns or "is_dark" not in mut.columns:
        raise ValueError("Mutations must contain context_96 and is_dark")

    bg_map = dict(zip(bg["context_96"], bg["dark_fraction"]))
    if bg["dark_fraction"].isna().any() or ((bg["dark_fraction"] <= 0) | (bg["dark_fraction"] >= 1)).any():
        raise ValueError("background dark_fraction must be in (0,1) for all contexts")


    # per-mutation expected probability of being dark
    mut["p_dark_bg"] = mut["context_96"].map(bg_map).fillna(bg["dark_fraction"].mean())

    agg = (
        mut.groupby("Tumor_Sample_Barcode")
        .agg(
            n_mutations=("is_dark", "size"),
            O=("is_dark", "sum"),
            E=("p_dark_bg", "sum"),
        )
        .reset_index()
    )
    agg["OE"] = agg["O"] / agg["E"].replace(0, np.nan)
    agg["log_OE"] = np.log(agg["OE"].replace(0, np.nan))

    out = agg.merge(sig, on="Tumor_Sample_Barcode", how="left", validate="one_to_one")

    # fit model: log_OE ~ SBS4 + total_mutations + dataset
    df = out.dropna(subset=["log_OE", "SBS4", "total_mutations"])
    if "dataset" in df.columns:
        dummies = pd.get_dummies(df["dataset"], prefix="dataset", drop_first=True, dtype=float)
        df = pd.concat([df, dummies], axis=1)
    predictors = ["SBS4", "total_mutations"] + [c for c in df.columns if c.startswith("dataset_")]
    X = sm.add_constant(df[predictors].astype(float))
    y = df["log_OE"].astype(float)
    model = sm.OLS(y, X).fit()
    print(model.summary().as_text()[:1200])

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print("Samples:", len(out))
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
