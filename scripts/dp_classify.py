import argparse
import io
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from diffprivlib.models import LogisticRegression

def prepare_df(path_or_text):
    """Load CSV from file path or string content into a DataFrame."""
    # load CSV file path or string content
    if "\n" in path_or_text and "," in path_or_text.splitlines()[0]:
        df = pd.read_csv(io.StringIO(path_or_text))
    else:
        df = pd.read_csv(path_or_text)
    return df

def build_features(df):
    """Extract candidate numeric and categorical features from the scored QARS DataFrame."""
    # pick candidate numeric and categorical features available in scored CSV
    num_cols = []
    for c in ("X_years","Y_years","Z_years","q_derived","T","S","E"):
        if c in df.columns:
            num_cols.append(c)
    cat_cols = []
    for c in ("Algo","App_type","Application","Vendor","Sector","vendor pqc compliant","third party used"):
        if c in df.columns:
            cat_cols.append(c)
    # fallback: use columns starting with 'w' for weights
    for c in df.columns:
        if c.lower() in ("wt","ws","we","wT","wS","wE") or c.lower().startswith("w"):
            if c not in num_cols and c not in cat_cols:
                num_cols.append(c)
    X_num = df[num_cols].astype(float).fillna(0.0) if num_cols else pd.DataFrame(index=df.index)
    X_cat = df[cat_cols].fillna("unknown") if cat_cols else pd.DataFrame(index=df.index)
    return X_num, X_cat, num_cols, cat_cols

def derive_label(df, label_col="QARS_category", threshold=0.85):
    """Derive the multi-class label (0=Low, 1=Medium, 2=High, 3=Critical)."""
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not present. Cannot train multi-class classifier.")

    # We expect QARS_category to be present from batch scoring
    # Map categories to numerical labels: Low=0, Medium=1, High=2, Critical=3
    category_to_int = {
        "low": 0,
        "medium": 1,
        "high": 2,
        "critical": 3
    }
    
    y = df[label_col].astype(str).str.lower().map(category_to_int)
    
    # Fill NaN (e.g., if a category name is unexpected) with a safe default like 'Low' (0)
    y = y.fillna(0).astype(int)

    # Note: We must return only the numpy array of labels for training
    return y.values

def main(argv=None):
    ap = argparse.ArgumentParser(prog="dp_classify.py")
    # Changed default label-col to QARS_category for multi-class
    ap.add_argument("--csv", required=True, help="Path to scored CSV (from batch scoring)")
    ap.add_argument("--label-col", default="QARS_category", help="Label column (must be QARS_category for multi-class)")
    ap.add_argument("--epsilon", type=float, default=1.0, help="DP epsilon")
    ap.add_argument("--delta", type=float, default=1e-5, help="DP delta")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--save", default="dp_qars_model.joblib")
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args(argv)

    df = prepare_df(args.csv)
    X_num, X_cat, num_cols, cat_cols = build_features(df)
    if X_num.empty and X_cat.empty:
        raise SystemExit("No usable features found in CSV. Ensure scored CSV includes numeric fields (X_years, q_derived, T, S, E, etc.) or categorical fields (Algo, Sector).")

    # Use the categorical label for multi-class classification
    y = derive_label(df, label_col=args.label_col)

    # ColumnTransformer pipeline
    transformers = []
    if not X_num.empty:
        transformers.append(("num", StandardScaler(), num_cols))
    if not X_cat.empty:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols))
    preproc = ColumnTransformer(transformers, remainder="drop")

    # DP logistic regression (multi-class)
    # The diffprivlib LogisticRegression model handles multi-class classification automatically
    dp_clf = LogisticRegression(epsilon=args.epsilon, delta=args.delta, data_norm=10.0, max_iter=1000, tol=1e-6)

    pipe = Pipeline([
        ("preproc", preproc),
        ("clf", dp_clf)
    ])

    X = pd.concat([X_num, X_cat], axis=1) if (not X_num.empty and not X_cat.empty) else (X_num if not X_num.empty else X_cat)
    X = X.fillna(0.0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state, stratify=y)

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    
    # AUC is typically reported per-class or macro/micro average for multi-class
    try:
        y_prob = pipe.predict_proba(X_test)
        # Macro-average AUC
        auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
    except Exception:
        y_prob = None
        auc = None

    print("Classification report (Multi-class: 0=Low, 1=Medium, 2=High, 3=Critical):\n", classification_report(y_test, y_pred))
    if auc is not None:
        print("ROC AUC (Macro-Average, One-vs-Rest):", auc)

    joblib.dump(pipe, args.save)
    print("Saved pipeline to", args.save)

if __name__ == "__main__":
    main()
