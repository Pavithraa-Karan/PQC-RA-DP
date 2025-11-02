import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression

def analyze(df, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    # Accept either a DataFrame or a dict of DataFrames (multiple sheets)
    if isinstance(df, dict):
        for name, subdf in df.items():
            with open(outdir / f"pandas_info_{name}.txt", "w") as f:
                subdf.info(buf=f)
            subdf.describe(include="all").to_csv(outdir / f"describe_{name}.csv")
            subdf.isna().sum().to_frame("missing_count").to_csv(outdir / f"missing_counts_{name}.csv")
    else:
        with open(outdir / "pandas_info.txt", "w") as f:
            df.info(buf=f)
        df.describe(include="all").to_csv(outdir / "describe.csv")
        df.isna().sum().to_frame("missing_count").to_csv(outdir / "missing_counts.csv")

def empirical_quantile_map(col):
    # returns sorted values and their cumulative probs (for interpolation)
    arr = col.dropna().values
    if arr.size == 0:
        return np.array([]), np.array([])
    sorted_vals = np.sort(arr)
    # use plotting positions to avoid 0/1
    n = sorted_vals.size
    probs = (np.arange(1, n + 1) - 0.5) / n
    return sorted_vals, probs

def col_to_normal_scores(col):
    # Rank-based normal scores (PIT using ranks)
    n = len(col)
    ranks = rankdata(col, method="average")  # ranks 1..n including ties
    u = ranks / (n + 1.0)
    return norm.ppf(u)

def transform_df_to_latent(df, numeric_cols):
    # For numeric cols: map observed values to normal scores (z)
    Z = np.zeros((len(df), len(numeric_cols)))
    for i, c in enumerate(numeric_cols):
        col = df[c].copy()
        # replace nan with random draws from empirical distribution before ranking
        missing_mask = col.isna()
        if missing_mask.any():
            observed = col.dropna().values
            if observed.size == 0:
                # all missing -> fill zeros
                col = col.fillna(0.0)
            else:
                # fill missing by sampling observed values (won't affect ranks much)
                col.loc[missing_mask] = np.random.choice(observed, size=missing_mask.sum(), replace=True)
        Z[:, i] = col_to_normal_scores(col.values)
    return Z

def fit_covariance(Z):
    # Ledoit-Wolf shrinkage for stable covariance estimation
    lw = LedoitWolf().fit(Z)
    cov = lw.covariance_
    mean = lw.location_
    return mean, cov

def sample_latent(mean, cov, n_samples, rng):
    return np.random.default_rng(rng).multivariate_normal(mean=mean, cov=cov, size=n_samples)

def latent_to_marginal_values(z_samples, original_cols_info):
    # Map latent normals -> uniform via CDF -> original values via empirical quantile interpolation
    n, d = z_samples.shape
    out_df = pd.DataFrame(index=range(n))
    for i, (col_name, info) in enumerate(original_cols_info.items()):
        sorted_vals = info["sorted_vals"]
        probs = info["probs"]
        if sorted_vals.size == 0:
            # column had all missing; produce all-nan
            out_df[col_name] = pd.NA
            continue
        u = norm.cdf(z_samples[:, i])
        # interp u in probs to get values (handle edge cases)
        # for reproducibility, clamp u to (min(probs), max(probs))
        u_clamped = np.clip(u, probs.min(), probs.max())
        # use numpy.interp on probs->sorted_vals
        vals = np.interp(u_clamped, probs, sorted_vals)
        # if original values were integer-like, round and cast to Int64
        if info["is_int_like"]:
            vals = np.round(vals).astype("int")
        out_df[col_name] = vals
    return out_df

def fit_categorical_conditionals(df, cat_cols, latent_Z):
    # Fit multinomial logistic regression for each categorical col using latent Z as predictors
    cat_models = {}
    for c in cat_cols:
        y = df[c].astype("object").fillna("___MISSING___")
        # if only one category, store frequencies to sample later
        if y.nunique() <= 1:
            freqs = y.value_counts(normalize=True).to_dict()
            cat_models[c] = ("freq_only", freqs)
            continue
        clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000)
        # if latent_Z has zero columns (no numeric), use intercept-only by fitting on zeros
        X = latent_Z if latent_Z.shape[1] > 0 else np.zeros((len(y), 1))
        try:
            clf.fit(X, y)
            cat_models[c] = ("model", clf)
        except Exception:
            # fallback to empirical frequencies
            freqs = y.value_counts(normalize=True).to_dict()
            cat_models[c] = ("freq_only", freqs)
    return cat_models

def sample_categories(cat_models, latent_samples, rng):
    rng = np.random.default_rng(rng)
    n = latent_samples.shape[0]
    cat_df = pd.DataFrame(index=range(n))
    for c, m in cat_models.items():
        tag = m[0]
        if tag == "freq_only":
            labels = list(m[1].keys())
            probs = np.array(list(m[1].values()))
            cat_df[c] = rng.choice(labels, size=n, p=probs)
        else:
            clf = m[1]
            Xs = latent_samples if latent_samples.shape[1] > 0 else np.zeros((n, 1))
            probs = clf.predict_proba(Xs)
            classes = clf.classes_
            picks = []
            for rowp in probs:
                picks.append(rng.choice(classes, p=rowp))
            cat_df[c] = picks
    # replace placeholder for missing back to actual missing marker
    cat_df = cat_df.replace("___MISSING___", pd.NA)
    return cat_df

def preserve_missingness(synth_df, original_missing_frac, rng):
    # original_missing_frac: dict col->fraction_missing
    rng = np.random.default_rng(rng)
    for c, frac in original_missing_frac.items():
        if frac <= 0:
            continue
        mask = rng.random(size=len(synth_df)) < frac
        synth_df.loc[mask, c] = pd.NA
    return synth_df

def generate_synthetic(
    df, n_samples=2000, rng_seed=42, analysis_out=Path("/workspaces/PQC-RA-DP/analysis_outputs")
):
    analysis_out = Path(analysis_out)
    analyze(df, analysis_out)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.columns.difference(numeric_cols).tolist()
    # collect marginals info for numeric columns
    original_cols_info = {}
    original_missing_frac = {}
    for c in numeric_cols:
        sorted_vals, probs = empirical_quantile_map(df[c])
        is_int_like = False
        if df[c].dropna().size > 0:
            # consider int-like if all observed values are near integers
            is_int_like = np.allclose(df[c].dropna() % 1, 0, atol=1e-8)
        original_cols_info[c] = {"sorted_vals": sorted_vals, "probs": probs, "is_int_like": is_int_like}
        original_missing_frac[c] = df[c].isna().mean()
    # if no numeric columns, can't fit copula: fallback to sampling categoricals independently
    if len(numeric_cols) == 0:
        # sample categoricals by empirical frequencies
        freqs = {c: df[c].fillna("__MISSING__").value_counts(normalize=True).to_dict() for c in cat_cols}
        rng = np.random.default_rng(rng_seed)
        out = pd.DataFrame(index=range(n_samples))
        for c, f in freqs.items():
            labels = list(f.keys())
            probs = np.array(list(f.values()))
            out[c] = rng.choice(labels, size=n_samples, p=probs)
        # restore missing
        out = out.replace("__MISSING__", pd.NA)
        out.to_excel("/workspaces/PQC-RA-DP/SYNTH_DATA_copula_generated.xlsx", index=False)
        return out

    # transform training numeric columns to latent normals
    Z = transform_df_to_latent(df[numeric_cols], numeric_cols)
    mean, cov = fit_covariance(Z)
    # sample latent normals for synthetic set
    Z_synth = sample_latent(mean, cov, n_samples, rng_seed)

    # prepare marginal info mapping for numeric cols
    # original_cols_info must be in same order as numeric_cols
    ordered_info = {c: original_cols_info[c] for c in numeric_cols}
    synth_numeric = latent_to_marginal_values(Z_synth, ordered_info)

    # preserve missingness for numeric columns
    for c in numeric_cols:
        original_missing_frac[c] = df[c].isna().mean()
    synth_numeric = preserve_missingness(synth_numeric, original_missing_frac, rng_seed + 1)

    # Fit categorical conditionals on latent Z (from training)
    latent_for_cat = Z  # use training latent representation as predictors
    cat_models = fit_categorical_conditionals(df[cat_cols], cat_cols, latent_for_cat)
    synth_cats = sample_categories(cat_models, Z_synth, rng_seed + 2)

    # combine
    synth = pd.concat([synth_numeric.reset_index(drop=True), synth_cats.reset_index(drop=True)], axis=1)
    # try cast columns to similar dtypes where safe
    for c in numeric_cols:
        if original_cols_info[c]["is_int_like"]:
            synth[c] = synth[c].astype("Int64")
    out_path_xlsx = Path("/workspaces/PQC-RA-DP/SYNTH_DATA_copula_generated.xlsx")
    out_path_csv = out_path_xlsx.with_suffix(".csv")
    synth.to_excel(out_path_xlsx, index=False)
    synth.to_csv(out_path_csv, index=False)
    return synth

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", default="/workspaces/PQC-RA-DP/SYNTH_DATA.xlsx")
    p.add_argument("--sheet", "-s", default=None)
    p.add_argument("--n", "-n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    path = Path(args.input)
    if not path.exists():
        print("Input not found:", path)
        return
    raw = pd.read_excel(path, sheet_name=args.sheet)
    # pd.read_excel(..., sheet_name=None) or sheet_name omitted may return a dict of DataFrames
    if isinstance(raw, dict):
        if args.sheet is None:
            if len(raw) == 1:
                sheet_name = next(iter(raw))
                df = raw[sheet_name]
                print(f"Using sheet '{sheet_name}'")
            else:
                # choose the first sheet by default but inform the user
                sheet_name = list(raw.keys())[0]
                df = raw[sheet_name]
                print(f"Multiple sheets found; using first sheet '{sheet_name}'. To select a sheet, pass --sheet NAME")
        else:
            # user requested a specific sheet name/index
            df = raw.get(args.sheet)
            if df is None:
                print(f"Requested sheet '{args.sheet}' not found. Available sheets: {list(raw.keys())}")
                return
    else:
        df = raw
    synth = generate_synthetic(df, n_samples=args.n, rng_seed=args.seed)
    print("Saved synthetic data to /workspaces/PQC-RA-DP/SYNTH_DATA_copula_generated.xlsx")

if __name__ == "__main__":
    main()