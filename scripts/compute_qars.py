import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd

# Heuristic mappings for sensitivity labels
SENS_MAP = {
    "critical": 1.0,
    "high": 0.75,
    "moderate": 0.5,
    "medium": 0.5,
    "low": 0.25,
    "public": 0.1,
    "pii": 0.9,
    "regulated": 0.95,
}

RSA_KEYWORDS = ["rsa", "ecc", "dh", "diffie", "dsa", "ecdsa"]

def find_column(df, patterns):
    for p in patterns:
        for c in df.columns:
            if re.search(p, c, flags=re.I):
                return c
    return None

def col_numeric_or_none(df, col):
    if col is None:
        return None
    try:
        vals = pd.to_numeric(df[col], errors="coerce")
        return vals
    except Exception:
        return None

def infer_X(df):
    c = find_column(df, [r"(?<!_)\bX\b", r"shelf", r"lifetime", r"confidential", r"retention", r"years"])
    vals = col_numeric_or_none(df, c)
    if vals is None:
        return pd.Series(1.0, index=df.index)  # default 1 year
    # clamp to reasonable range
    return vals.fillna(vals.median()).clip(lower=0.0)

def infer_Y(df):
    c = find_column(df, [r"(?<!_)\bY\b", r"migrat", r"upgrade", r"time_to", r"mitigation"])
    vals = col_numeric_or_none(df, c)
    if vals is not None:
        return vals.fillna(vals.median()).clip(lower=0.0)
    # fallback: infer from legacy/complexity flags
    legacy_col = find_column(df, [r"legacy", r"end_of_life", r"eol", r"deprecated"])
    if legacy_col:
        mask = df[legacy_col].astype(str).str.lower().isin(["true", "yes", "1", "y"])
        y = pd.Series(1.0, index=df.index)
        y[mask] = 5.0
        return y
    return pd.Series(1.0, index=df.index)

def infer_Z(df, global_default=None):
    c = find_column(df, [r"(?<!_)\bZ\b", r"quantum", r"crqc", r"horizon"])
    vals = col_numeric_or_none(df, c)
    if vals is not None:
        # if empty, fallback
        return vals.fillna(vals.median()).clip(lower=0.1)
    if global_default is not None:
        return pd.Series(float(global_default), index=df.index)
    return pd.Series(12.0, index=df.index)  # default threat horizon ~12 years

def infer_D(df):
    # sensitivity label or numeric severity
    c = find_column(df, [r"sensit", r"classif", r"data_class", r"impact", r"D\b"])
    if c:
        vals = df[c].astype(str).fillna("").str.lower()
        # numeric?
        numeric = pd.to_numeric(df[c], errors="coerce")
        if numeric.notna().any():
            num = numeric.fillna(numeric.median()).clip(0.0, 1.0)
            # if it looks like 1-5 scale, normalize
            if num.max() > 1.5:
                num = (num - num.min()) / (num.max() - num.min())
            return num
        # map labels
        mapped = vals.map(lambda v: SENS_MAP.get(v.strip(), np.nan))
        # look for keywords in other columns
    # fallback: check PII/regulatory flags
    pii_col = find_column(df, [r"pii", r"personal", r"gdpr", r"hipaa", r"ssn"])
    if pii_col:
        m = df[pii_col].astype(str).str.lower().isin(["true", "yes", "1", "y"])
        s = pd.Series(0.25, index=df.index)
        s[m] = 0.95
        return s
    return pd.Series(0.5, index=df.index)  # default moderate

def infer_v(df):
    # cryptographic visibility: 1 if RSA/ECC/DH in algorithm or crypto column, 0 if PQC mentioned
    c = find_column(df, [r"algor", r"crypto", r"cipher", r"algorithm"])
    v = pd.Series(0.5, index=df.index)  # unknown -> 0.5
    if c:
        s = df[c].astype(str).str.lower().fillna("")
        has_rsa = s.str.contains("|".join(RSA_KEYWORDS))
        has_pqc = s.str.contains(r"pqc|post-quantum|kyber|ntru|sphincs|dilithium")
        v = pd.Series(0.0, index=df.index)
        v[has_rsa & ~has_pqc] = 1.0
        v[has_pqc & ~has_rsa] = 0.0
        v[has_rsa & has_pqc] = 0.5  # hybrid
    else:
        # fallback: if any column indicates 'uses_asymmetric' or 'public_key'
        c2 = find_column(df, [r"asym", r"public_key", r"pubkey", r"key_type"])
        if c2:
            s2 = df[c2].astype(str).str.lower().fillna("")
            v = pd.Series(0.0, index=df.index)
            v[s2.str.contains("true|yes|rsa|ecc|dh|public")] = 1.0
    return v

def infer_q(df):
    # harvestability: combine internet exposure, cloud, public endpoints into [0,1]
    features = []
    for pat in [r"internet", r"expos", r"public", r"cloud", r"endpoint", r"api"]:
        c = find_column(df, [pat])
        if c:
            col = df[c].astype(str).str.lower()
            # map booleans/qualitative to 0/1
            mapped = col.map(lambda v: 1.0 if re.search(r"true|yes|1|public|internet|cloud", v) else 0.0)
            features.append(mapped.astype(float))
    if features:
        arr = pd.concat(features, axis=1).fillna(0.0)
        # average signals
        return arr.mean(axis=1).clip(0.0, 1.0)
    # fallback: try numeric 'exposure' columns
    c = find_column(df, [r"exposur", r"q_score", r"harvest"])
    vals = col_numeric_or_none(df, c)
    if vals is not None:
        v = vals.fillna(vals.median())
        if v.max() > 1.5:
            v = (v - v.min()) / (v.max() - v.min())
        return v.clip(0.0, 1.0)
    return pd.Series(0.5, index=df.index)

def map_sensitivity_score(s_series):
    # if already in [0,1], pass through; else map known labels
    if s_series.dtype.kind in "fiu" and (s_series.max() <= 1.0 and s_series.min() >= 0.0):
        return s_series.clip(0.0, 1.0)
    # if series of strings:
    s = s_series.astype(str).fillna("").str.lower()
    mapped = s.map(lambda v: SENS_MAP.get(v.strip(), np.nan))
    # fill NaN with 0.5
    return mapped.fillna(0.5)

def compute_qars(df, X, Y, Z, S, v, q, wT, wS, wE, alpha):
    # ensure numeric series and align indices
    X = X.astype(float)
    Y = Y.astype(float)
    Z = Z.astype(float).replace(0, 0.1)
    r = (X + Y) / Z
    T = 1.0 / (1.0 + np.exp(-alpha * (r - 1.0)))
    S = map_sensitivity_score(S)
    E = (v * q).astype(float)
    # final weighted score
    QARS = (wT * T) + (wS * S) + (wE * E)
    return T.clip(0, 1), S.clip(0, 1), E.clip(0, 1), QARS.clip(0, 1)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", default="/workspaces/PQC-RA-DP/SYNTH_DATA_ctgan_generated.csv")
    p.add_argument("--output", "-o", default="/workspaces/PQC-RA-DP/SYNTH_DATA_qars_scored.csv")
    p.add_argument("--Z", type=float, default=None, help="Global Z (threat horizon) if dataset lacks one")
    p.add_argument("--alpha", type=float, default=6.0, help="Steepness alpha for logistic timeline map")
    p.add_argument("--wT", type=float, default=1/3, help="Weight for Timeline")
    p.add_argument("--wS", type=float, default=1/3, help="Weight for Sensitivity")
    p.add_argument("--wE", type=float, default=1/3, help="Weight for Exposure")
    p.add_argument("--top", type=int, default=10, help="Print top-N highest QARS assets")
    args = p.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print("Input file not found:", inp)
        return

    df = pd.read_csv(inp)

    # infer components
    X = infer_X(df)
    Y = infer_Y(df)
    Z = infer_Z(df, global_default=args.Z)
    S_col = find_column(df, [r"sensit", r"classif", r"impact", r"D\b"])
    S = df[S_col] if S_col is not None else infer_D(df)
    v = infer_v(df)
    q = infer_q(df)

    # compute QARS
    T, S_mapped, E, QARS = compute_qars(df, X, Y, Z, S, v, q, args.wT, args.wS, args.wE, args.alpha)

    out = df.copy()
    out["_QARS_T"] = T
    out["_QARS_S"] = S_mapped
    out["_QARS_E"] = E
    out["_QARS_score"] = QARS
    out["_QARS_weight_T"] = args.wT
    out["_QARS_weight_S"] = args.wS
    out["_QARS_weight_E"] = args.wE
    out["_QARS_alpha"] = args.alpha
    # save
    out.to_csv(args.output, index=False)
    print(f"Saved scored dataset to {args.output}")

    # print top N
    topn = out.sort_values("_QARS_score", ascending=False).head(args.top)
    cols_to_show = [c for c in ["_QARS_score", "_QARS_T", "_QARS_S", "_QARS_E"] if c in out.columns]
    print(f"Top {args.top} by QARS:")
    print(topn[cols_to_show].to_string(index=False))

if __name__ == "__main__":
    main()