"""Refit signatures with a small, lung-relevant set + bootstrap CIs."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
COUNTS = ROOT / "data" / "processed" / "tumor_context_counts.csv"
MATRIX = ROOT / "data" / "raw" / "COSMIC_v3.3.1_SBS_GRCh37.txt"
SIGS = ["SBS1", "SBS2", "SBS4", "SBS5", "SBS13", "SBS17a", "SBS40"]


def fit(vec, A):
    coef, _ = nnls(A, vec)
    total = vec.sum()
    act = dict(zip(SIGS, coef))
    frac = act["SBS4"] / total if total else 0.0
    return act, frac


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=50)
    parser.add_argument("--counts", default="data/processed/tumor_context_counts.csv")
    parser.add_argument("--output", default="data/processed/tumor_signature_activities_refit.csv")

    args = parser.parse_args()

    counts = pd.read_csv(args.counts)

    mat = pd.read_csv(MATRIX, sep="\t").set_index("Type")
    mat = mat[SIGS]
    contexts = list(mat.index)
    A = mat.to_numpy(dtype=float)

    pivot = counts.pivot_table(index="Tumor_Sample_Barcode", columns="context_96", values="count", aggfunc="sum", fill_value=0)
    pivot = pivot.reindex(columns=contexts, fill_value=0)
    meta = counts.drop_duplicates("Tumor_Sample_Barcode").set_index("Tumor_Sample_Barcode")

    rows = []
    rng = np.random.default_rng(42)
    for sample, vec in pivot.iterrows():
        b = vec.to_numpy(dtype=float)
        act, frac = fit(b, A)
        total = float(b.sum())
        # bootstrap
        boot_fracs = []
        for _ in range(args.bootstrap):
            if total <= 0:
                break
            # multinomial resample of contexts
            probs = b / total if total > 0 else np.ones(len(b)) / len(b)
            resampled = rng.multinomial(int(total), probs)
            _, bf = fit(resampled.astype(float), A)
            boot_fracs.append(bf)
        ci_low = float(np.percentile(boot_fracs, 2.5)) if boot_fracs else float("nan")
        ci_high = float(np.percentile(boot_fracs, 97.5)) if boot_fracs else float("nan")
        rows.append({
            "Tumor_Sample_Barcode": sample,
            "dataset": meta.loc[sample, "dataset"],
            "total_mutations": total,
            "SBS4": act["SBS4"],
            "SBS4_fraction": frac,
            "SBS4_fraction_ci_low": ci_low,
            "SBS4_fraction_ci_high": ci_high,
            **{s: act[s] for s in SIGS if s != "SBS4"},
        })

    out = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print("Samples:", len(out))
    print("SBS4 fraction median:", out.SBS4_fraction.median())
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
