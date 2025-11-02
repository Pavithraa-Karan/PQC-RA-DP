import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def analyze(df, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    summary = []
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    summary.append({"type":"n_rows", "value": len(df)})
    summary.append({"type":"n_numeric", "value": len(num_cols)})
    summary.append({"type":"n_categorical", "value": len(cat_cols)})
    pd.DataFrame(summary).to_csv(outdir / "dataset_summary.csv", index=False)
    # basic stats
    if num_cols:
        df[num_cols].describe().to_csv(outdir / "numeric_describe.csv")
    if cat_cols:
        cat_stats = {c: df[c].value_counts(dropna=False).to_dict() for c in cat_cols}
        import json
        with open(outdir / "categorical_value_counts.json", "w") as f:
            json.dump(cat_stats, f, default=str, indent=2)
    # missing
    df.isna().sum().to_frame("missing_count").to_csv(outdir / "missing_counts.csv")

def synth_with_sdv(df, n_samples, rng):
    try:
        from sdv.tabular import GaussianCopula
    except Exception as e:
        raise RuntimeError("SDV not available. Install with: pip install sdv") from e
    model = GaussianCopula(random_state=rng)
    model.fit(df)
    samples = model.sample(n_samples)
    # preserve column order and dtypes where possible
    return samples[df.columns]

def synth_independent(df, n_samples, rng):
    out = pd.DataFrame(index=range(n_samples))
    rng = np.random.default_rng(rng)
    for c in df.columns:
        col = df[c]
        if pd.api.types.is_numeric_dtype(col):
            # choose parametric vs non-parametric by unimodality test (simple skew check)
            if col.dropna().nunique() <= 10:
                # discrete numeric -> sample with observed frequency
                probs = col.value_counts(normalize=True, dropna=False)
                out[c] = rng.choice(probs.index.to_list(), size=n_samples, p=probs.values)
            else:
                # if highly skewed, try log-transform sampling
                skew = col.dropna().skew()
                if abs(skew) < 1.0:
                    mu, sd = col.dropna().mean(), col.dropna().std()
                    out[c] = rng.normal(loc=mu, scale=sd, size=n_samples)
                else:
                    from scipy.stats import gaussian_kde
                    data = col.dropna().values
                    kde = gaussian_kde(data)
                    samp = kde.resample(n_samples).reshape(-1)
                    out[c] = samp
        else:
            probs = col.value_counts(normalize=True, dropna=False)
            # Laplace smoothing for rare categories
            labels = probs.index.to_list()
            p = probs.values + 1e-6
            p = p / p.sum()
            out[c] = rng.choice(labels, size=n_samples, p=p)
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", default="/workspaces/PQC-RA-DP/SYNTH_DATA.xlsx")
    p.add_argument("--sheet", "-s", default=None)
    p.add_argument("--n", "-n", type=int, default=2000)
    p.add_argument("--out", "-o", default="/workspaces/PQC-RA-DP/SYNTH_DATA_generated.xlsx")
    p.add_argument("--analysis", default="/workspaces/PQC-RA-DP/analysis_outputs")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        print("Input file not found:", input_path)
        sys.exit(1)
    df = pd.read_excel(input_path, sheet_name=args.sheet)
    analyze(df, Path(args.analysis))
    # try joint model first (recommended in literature to preserve correlations)
    try:
        synth = synth_with_sdv(df, args.n, args.seed)
        method = "sdv.GaussianCopula"
    except Exception as e:
        print("SDV modelling failed or not available, falling back to independent-column sampling. Error:", e)
        synth = synth_independent(df, args.n, args.seed)
        method = "independent_per_column"
    # Post-process: cast types similar to original where safe
    for c in df.columns:
        if pd.api.types.is_integer_dtype(df[c].dropna()):
            # round floats to nearest integer if original was integer
            if pd.api.types.is_integer_dtype(df[c]) or (np.allclose(df[c].dropna() % 1, 0)):
                try:
                    synth[c] = synth[c].round().astype('Int64')
                except Exception:
                    pass
    # Save
    outp = Path(args.out)
    synth.to_excel(outp, index=False)
    synth.to_csv(outp.with_suffix(".csv"), index=False)
    print(f"Saved synthetic data ({method}) to {outp} and CSV companion.")

if __name__ == "__main__":
    main()