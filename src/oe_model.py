"""Binomial GEE model for O/E dark mutations (clustered by sample)."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mutations", required=True)
    parser.add_argument("--background", required=True)
    parser.add_argument("--signatures", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    mut = pd.read_csv(args.mutations, low_memory=False)
    bg = pd.read_csv(args.background)
    sig = pd.read_csv(args.signatures, low_memory=False)

    bg_map = dict(zip(bg["context_96"], bg["dark_fraction"]))
    if bg["dark_fraction"].isna().any() or ((bg["dark_fraction"] <= 0) | (bg["dark_fraction"] >= 1)).any():
        raise ValueError("background dark_fraction must be in (0,1) for all contexts")

    mut["p_dark_bg"] = mut["context_96"].map(bg_map).fillna(bg["dark_fraction"].mean())
    mut["offset"] = np.log(mut["p_dark_bg"] / (1 - mut["p_dark_bg"]))

    sig_cols = ["Tumor_Sample_Barcode", "SBS4", "total_mutations"]
    if "dataset" in sig.columns:
        sig_cols.append("dataset")
    mut = mut.merge(sig[sig_cols], on="Tumor_Sample_Barcode", how="left", validate="many_to_one")

    # standardize SBS4
    mut["SBS4_std"] = (mut["SBS4"] - mut["SBS4"].mean()) / mut["SBS4"].std()
    mut["TMB_std"] = (mut["total_mutations"] - mut["total_mutations"].mean()) / mut["total_mutations"].std()

    # design matrix
    X = pd.DataFrame({
        "const": 1.0,
        "SBS4_std": mut["SBS4_std"],
        "TMB_std": mut["TMB_std"],
    })
    if "dataset" in mut.columns:
        dummies = pd.get_dummies(mut["dataset"], prefix="dataset", drop_first=True, dtype=float)
        X = pd.concat([X, dummies], axis=1)

    y = mut["is_dark"].astype(int)
    offset = mut["offset"].values
    groups = mut["Tumor_Sample_Barcode"]

    gee = sm.GEE(y, X, groups=groups, family=Binomial(), cov_struct=Exchangeable(), offset=offset)
    res = gee.fit()
    print(res.summary().as_text()[:1500])

    # save coefficients
    out = pd.DataFrame({
        "term": res.params.index,
        "coef": res.params.values,
        "se": res.bse.values,
        "pvalue": res.pvalues.values,
    })
    # OR for SBS4
    if "SBS4_std" in res.params.index:
        beta = res.params["SBS4_std"]
        se = res.bse["SBS4_std"]
        or_val = np.exp(beta)
        or_low = np.exp(beta - 1.96 * se)
        or_high = np.exp(beta + 1.96 * se)
        print(f"\nSBS4: OR per 1 SD = {or_val:.4f} (95% CI: {or_low:.4f}, {or_high:.4f})")
        out = pd.concat([out, pd.DataFrame([{
            "term": "SBS4_OR", "coef": or_val, "se": np.nan, "pvalue": res.pvalues["SBS4_std"]
        }])], ignore_index=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print("Saved:", args.output)

if __name__ == "__main__":
    main()
