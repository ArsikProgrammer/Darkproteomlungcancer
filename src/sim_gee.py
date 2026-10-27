"""Simulation to assess GEE bias and coverage."""
import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable

def run_one(mut_base, sig_map, std_map, rng, beta, bg_map):
    mut = mut_base.copy()
    mut["Start_Position"] = pd.to_numeric(mut["Start_Position"], errors="coerce")
    ctx = pd.read_csv("data/processed/george_coding_snps_context.csv", low_memory=False)
    ctx["Start_Position"] = pd.to_numeric(ctx["Start_Position"], errors="coerce")
    mut = mut.merge(ctx[["Tumor_Sample_Barcode", "Start_Position", "context_96"]], on=["Tumor_Sample_Barcode", "Start_Position"], how="left")
    mut = mut.dropna(subset=["context_96"])
    probs = []
    for _, r in mut.iterrows():
        sbs4_std = std_map.get(r["Tumor_Sample_Barcode"], 0.0)
        p_bg = bg_map.get(r["context_96"], 0.24)
        p = 1 / (1 + np.exp(-(np.log(p_bg/(1-p_bg)) + beta * sbs4_std)))
        probs.append(p)
    mut["is_dark"] = rng.binomial(1, probs).astype(int)
    mut["p_dark_bg"] = mut["context_96"].map(bg_map).fillna(0.24)
    mut["offset"] = np.log(mut["p_dark_bg"] / (1 - mut["p_dark_bg"]))
    mut["SBS4_std"] = mut["Tumor_Sample_Barcode"].map(std_map).fillna(0)
    mut["TMB_std"] = 0.0  # placeholder
    X = pd.DataFrame({"const": 1.0, "SBS4_std": mut["SBS4_std"]})
    y = mut["is_dark"].astype(int)
    groups = mut["Tumor_Sample_Barcode"]
    try:
        res = sm.GEE(y, X, groups=groups, family=Binomial(), cov_struct=Exchangeable(), offset=mut["offset"].values).fit()
        beta_hat = res.params["SBS4_std"]
        se = res.bse["SBS4_std"]
        ci_low = beta_hat - 1.96 * se
        ci_high = beta_hat + 1.96 * se
        pval = res.pvalues["SBS4_std"]
        return beta_hat, ci_low, ci_high, pval
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--odds_ratio", type=float, default=1.5)
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    mut_base = pd.read_csv("data/processed/george_coding_snps.csv", low_memory=False)
    sig = pd.read_csv("data/processed/tumor_signature_activities.csv")
    sig_map = dict(zip(sig.Tumor_Sample_Barcode, sig.SBS4))
    vals = pd.Series(list(sig_map.values()))
    std = (vals - vals.mean()) / vals.std()
    std_map = dict(zip(sig.Tumor_Sample_Barcode, std))

    contexts = pd.read_csv("data/processed/george_coding_snps_context.csv", low_memory=False).context_96.unique()
    bg = pd.DataFrame({"context_96": contexts, "dark_fraction": np.random.default_rng(7).uniform(0.15, 0.35, size=len(contexts))})
    bg_map = dict(zip(bg.context_96, bg.dark_fraction))

    beta_true = np.log(args.odds_ratio)
    rng = np.random.default_rng(args.seed)
    betas = []
    cover = 0
    sig_count = 0
    for i in range(args.sims):
        out = run_one(mut_base, sig_map, std_map, rng, beta_true, bg_map)
        if out is None:
            continue
        beta_hat, ci_low, ci_high, pval = out
        betas.append(beta_hat)
        if ci_low <= beta_true <= ci_high:
            cover += 1
        if pval < 0.05:
            sig_count += 1
    betas = np.array(betas)
    print(f"OR={args.odds_ratio}, sims={len(betas)}")
    print(f"Mean beta_hat: {betas.mean():.4f} (true {beta_true:.4f})")
    print(f"Mean OR_hat: {np.exp(betas.mean()):.4f}")
    print(f"Coverage: {cover/len(betas):.3f}")
    if args.sims < 200:
        print("WARNING: fewer than 200 sims; power/coverage estimates are unstable")

    print(f"Mean SE (approx): {betas.std():.4f}")

if __name__ == "__main__":
    main()
