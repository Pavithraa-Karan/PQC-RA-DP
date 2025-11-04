import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def read_input(path, sheet=None):
    raw = pd.read_excel(path, sheet_name=sheet)
    if isinstance(raw, dict):
        if sheet is None:
            # choose first sheet by default
            sheet_name = list(raw.keys())[0]
            print(f"Multiple sheets found; using first sheet '{sheet_name}'. To select a sheet, pass --sheet NAME")
            return raw[sheet_name]
        else:
            df = raw.get(sheet)
            if df is None:
                raise ValueError(f"Sheet '{sheet}' not found. Available: {list(raw.keys())}")
            return df
    return raw

def detect_column_types(df):
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    int_like = []
    for c in num_cols:
        arr = df[c].dropna()
        if arr.size and np.allclose(arr % 1, 0, atol=1e-8):
            int_like.append(c)
    return cat_cols, num_cols, int_like

def prepare_for_ctgan(df, cat_cols):
    dfp = df.copy()
    # CTGAN requires no NaN; use placeholders:
    # - categorical: special token
    # - numeric: median imputation (we will restore missingness later)
    missing_info = {c: dfp[c].isna().mean() for c in dfp.columns}
    for c in dfp.columns:
        if c in cat_cols:
            dfp[c] = dfp[c].astype(object).fillna("___MISSING_CAT___")
        else:
            # numeric
            med = dfp[c].median(skipna=True)
            if pd.isna(med):
                med = 0.0
            dfp[c] = dfp[c].fillna(med)
    return dfp, missing_info

def postprocess_synthetic(synth, original_df, int_like, missing_info, rng_seed=42):
    # round integer-like columns and cast to Int64
    for c in int_like:
        if c in synth.columns:
            synth[c] = np.round(synth[c]).astype("Int64")
    # restore missingness by per-column fraction (random mask)
    rng = np.random.default_rng(rng_seed)
    for c, frac in missing_info.items():
        if frac > 0 and c in synth.columns:
            mask = rng.random(size=len(synth)) < frac
            synth.loc[mask, c] = pd.NA
    # attempt to cast categorical placeholders back to NaN
    for c in synth.select_dtypes(include=["object", "category"]).columns:
        synth[c] = synth[c].replace("___MISSING_CAT___", pd.NA)
    # reorder columns to match original
    synth = synth[original_df.columns.tolist()]
    return synth

def _set_seed(seed):
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # torch not available; continue
        pass

def build_sdv_metadata(df, discrete_columns):
    try:
        from sdv.metadata import SingleTableMetadata
    except Exception as e:
        raise RuntimeError("sdv.metadata not available. Install sdv: pip install sdv") from e
    metadata = SingleTableMetadata()
    # detect_from_dataframe accepts a positional DataFrame in some sdv versions
    try:
        # prefer positional call to avoid keyword compatibility issues
        metadata.detect_from_dataframe(df)
    except TypeError:
        # fallback to older/newer signatures that may accept a keyword
        metadata.detect_from_dataframe(dataframe=df)
    # ensure declared discrete columns are categorical where possible
    for col in discrete_columns:
        try:
            metadata.update_column(col, sdtype='categorical')
        except Exception:
            # ignore if update_column signature differs between SDV versions
            try:
                # some versions expect a dict of metadata for the column
                col_meta = metadata.get_column_metadata(col)
                col_meta['sdtype'] = 'categorical'
                metadata.update_column(col, col_meta)
            except Exception:
                pass
    return metadata

def generate_with_ctgan(df, n_samples=2000, epochs=300, seed=42):
    try:
        from sdv.single_table import CTGANSynthesizer
    except Exception as e:
        raise RuntimeError("sdv not installed. Install with: pip install sdv") from e

    cat_cols, num_cols, int_like = detect_column_types(df)
    dfp, missing_info = prepare_for_ctgan(df, cat_cols)

    discrete_columns = cat_cols.copy()

    # build SDV metadata required by CTGANSynthesizer
    metadata = build_sdv_metadata(dfp, discrete_columns)

    # set RNG seeds for reproducibility (Python, NumPy, Torch)
    _set_seed(seed)

    # instantiate CTGAN with metadata (metadata is required)
    model = CTGANSynthesizer(metadata)

    # fit the model (metadata already marks discrete columns)
    model.fit(dfp)

    # sampling will follow the RNG state we set above
    synth = model.sample(n_samples)

    synth = postprocess_synthetic(synth, df, int_like, missing_info, rng_seed=seed + 1)
    return synth

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", default="/workspaces/PQC-RA-DP/SYNTH_DATA.xlsx")
    p.add_argument("--sheet", "-s", default=None)
    p.add_argument("--n", "-n", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    path = Path(args.input)
    if not path.exists():
        print("Input not found:", path)
        return

    df = read_input(path, sheet=args.sheet)
    synth = generate_with_ctgan(df, n_samples=args.n, epochs=args.epochs, seed=args.seed)

    out_xlsx = Path("/workspaces/PQC-RA-DP/SYNTH_DATA_ctgan_generated.xlsx")
    out_csv = out_xlsx.with_suffix(".csv")
    synth.to_excel(out_xlsx, index=False)
    synth.to_csv(out_csv, index=False)
    print(f"Saved CTGAN synthetic data to {out_xlsx} and {out_csv}")

if __name__ == "__main__":
    main()