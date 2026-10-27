"""Power simulation for Dark Proteome × SBS4 effect (corrected)."""
import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/sample_features_50_annotated.csv")
    parser.add_argument("--odds_ratio", type=float, default=1.2)
    parser.add_argument("--sims", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--coverage", type=float, default=1.0)
    parser.add_argument("--random_effects", action="store_true", default=False)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = df.dropna(subset=["dark_fraction", "SBS4", "n_mutations"])
    sbs4_std = (df["SBS4"] - df["SBS4"].mean()) / df["SBS4"].std()
    base = df["dark_fraction"].mean()
    n = len(df)
    beta = np.log(args.odds_ratio)
    rng = np.random.default_rng(args.seed)

    sample_sd = df["dark_fraction"].std()
    cohort_sd = 0.02

    sig = 0
    var_fracs = []
    for _ in range(args.sims):
        if args.random_effects:
            sample_re = rng.normal(0, sample_sd, size=n)
            cohort_re = rng.normal(0, cohort_sd, size=n)
            linpred = np.log(base/(1-base)) + beta * sbs4_std.values + sample_re + cohort_re
        else:
            linpred = np.log(base/(1-base)) + beta * sbs4_std.values
        p = 1 / (1 + np.exp(-linpred))
        n_mut = (df["n_mutations"].values * args.coverage).astype(int)
        n_mut = np.clip(n_mut, 1, None)
        n_dark = rng.binomial(n_mut, p)
        frac = n_dark / n_mut
        var_fracs.append(frac.var())
        X = sm.add_constant(sbs4_std.values)
        w = n_mut.astype(float)
        try:
            model = sm.WLS(frac, X, weights=w).fit()
            if model.pvalues[1] < 0.05:
                sig += 1
        except Exception:
            continue
    power = sig / args.sims
    print(f"OR={args.odds_ratio}, sims={args.sims}, n={n}, coverage={args.coverage}, RE={args.random_effects}")
    print(f"Power (p<0.05): {power:.3f}")
    print(f"Base dark fraction: {base:.3f}")
    print(f"Mean simulated frac variance: {np.mean(var_fracs):.4f}")
    if power >= 0.999:
        print("WARNING: power is suspiciously 1.000; check simulation design and coverage")


if __name__ == "__main__":
    main()
