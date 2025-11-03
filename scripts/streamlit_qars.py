"""
Streamlit app for QARS (Quantum-Adjusted Risk Score)

This provides an interactive UI to compute QARS using inputs for X,Y,Z, sensitivity,
algorithm, harvestability and sector presets. It imports the model helpers from
`qars_model.py`.

Run with:
  streamlit run scripts/streamlit_qars.py
"""
from __future__ import annotations

import streamlit as st
import io
from typing import Dict, Any

from qars_model import (
    compute_qars,
    qars_category,
    sector_profiles,
    SENSITIVITY_MAP,
    batch_score_csv_string,
    score_row_from_csv_row,
    compute_dynamic_weights_from_row,
    load_threat_feed,
    _derive_q_from_row,
    _parse_float_field,
    _parse_text_field,
    _parse_yesno_field,
)

st.set_page_config(page_title="QARS - Quantum-Adjusted Risk Score", layout="wide", initial_sidebar_state="expanded")

st.title("Quantum-Adjusted Risk Score (QARS)")
st.markdown("Interactive demo of the QARS model with dynamic, threat-aware weighting and exposure modeling. "
            "Use the controls in the sidebar to explore how timeline (T), sensitivity (S) and exposure (E) "
            "combine into an actionable portfolio view.")

# sidebar controls
# App mode selection appears before global calibration so user chooses mode first
mode = st.sidebar.radio("App mode", ["Single asset demo", "Portfolio dashboard (CSV)", "Classifier (CSV)"], index=0)
st.sidebar.header("Global / Calibration")
# threat feed controls
st.sidebar.markdown("### Threat feed")
threat_source = st.sidebar.selectbox("Threat feed source", ["Built-in (none)", "Upload JSON", "Env var path"], index=0)
# persist threat feed in session_state so demo button and uploads persist
if "threat_feed_data" not in st.session_state:
    # try to preload default file if present
    try:
        st.session_state["threat_feed_data"] = load_threat_feed()
    except Exception:
        st.session_state["threat_feed_data"] = {}
threat_feed_data = st.session_state["threat_feed_data"]
if threat_source == "Upload JSON":
    tf_upload = st.sidebar.file_uploader("Upload threat_feed.json", type=["json"])
    if tf_upload:
        try:
            import json as _json
            uploaded = _json.load(tf_upload)
            st.session_state["threat_feed_data"] = uploaded
            threat_feed_data = uploaded
            st.sidebar.success("Loaded uploaded threat feed into session")
        except Exception:
            st.sidebar.error("Failed to parse uploaded threat feed")
elif threat_source == "Env var path":
    st.sidebar.text("Using QARS_THREAT_FEED environment path if set")

# one-click demo: load example high-severity scenario into session_state
if st.sidebar.button("Load example high-severity demo"):
    demo_feed = {
        "global": 0.3,
        "Finance": 1.0,
        "Cloud": 0.9,
        "IoT/Embedded": 0.95,
        "Healthcare": 1.0
    }
    st.session_state["threat_feed_data"] = demo_feed
    threat_feed_data = demo_feed
    st.sidebar.success("Example high-severity threat feed loaded")

# sector preset selection (keeps existing behaviour)
sector = st.sidebar.selectbox("Sector preset", list(sector_profiles.keys()))
manual_weights = sector_profiles.get(sector, sector_profiles["Default"])

st.sidebar.markdown("### Manual weights (optional)")
# sliders use session_state keys so we can reset them without calling experimental_rerun
wT_manual = st.sidebar.slider("Manual wT", 0.0, 1.0, float(manual_weights["wT"]), key="wT_manual")
wS_manual = st.sidebar.slider("Manual wS", 0.0, 1.0, float(manual_weights["wS"]), key="wS_manual")
wE_manual = st.sidebar.slider("Manual wE", 0.0, 1.0, float(manual_weights["wE"]), key="wE_manual")
use_manual_weights = st.sidebar.checkbox("Use manual weights instead of dynamic", value=False, key="use_manual_weights")

st.sidebar.markdown("### Timeline scaling")
linear_t = st.sidebar.checkbox("Use linear timeline scaling (min(1,r))", value=False, key="linear_t")
alpha = st.sidebar.slider("Logistic steepness alpha", 1.0, 40.0, 10.0, key="alpha")

# Asset / Mode selection (mode selected in the sidebar above)
# mode already defined in the sidebar earlier
if mode == "Single asset demo":
    # Asset input panel (single-asset demo)
    st.header("Asset parameters (single asset demo)")
    col1, col2, col3 = st.columns(3)
    with col1:
        X = st.number_input("Confidentiality duration X (years)", min_value=0.0, value=15.0, step=1.0)
        Y = st.number_input("Migration time Y (years)", min_value=0.0, value=1.0, step=1.0)
    with col2:
        Z = st.number_input("Projected CRQC horizon Z (years)", min_value=0.0, value=12.0, step=1.0)
        q = st.slider("Harvestability q (0-1)", 0.0, 1.0, 0.3)
    with col3:
        sensitivity = st.selectbox("Data sensitivity", list(SENSITIVITY_MAP.keys()), index=2)
        algorithm = st.text_input("Algorithm (e.g., RSA, ECC, PQC)", value="RSA")

    st.markdown("---")

    # compute dynamic weights for current single-asset inputs (demonstration)
    current_row: Dict[str, Any] = {
        "Algo": algorithm,
        "Data Sensisitivity": sensitivity,
        "data trans": "internet" if "http" in algorithm.lower() else "",
        "App_type": "public",
        "vendor pqc compliant": "No",
        "third party used": "Yes",
        "vendor supply time": "2",
        "data shelf life": f"{X}",
        "Frequency": "10",
        "Sector": sector,
    }
    # decide threat_feed to pass
    tf = threat_feed_data if threat_feed_data else load_threat_feed()

    dynamic_w = compute_dynamic_weights_from_row(current_row, sector=sector if sector else None, threat_feed=tf)

    # choose weights to pass to compute_qars
    if use_manual_weights:
        # normalize manual
        total = (wT_manual + wS_manual + wE_manual) or 1.0
        weights_in = {"wT": wT_manual / total, "wS": wS_manual / total, "wE": wE_manual / total}
    else:
        weights_in = dynamic_w

    score, breakdown = compute_qars(X=X, Y=Y, Z=Z, sensitivity=sensitivity, algorithm=algorithm, q=q, weights=weights_in, alpha=alpha, linear_t=linear_t)
    category = qars_category(score)

    # Results / visualization panel (improved)
    st.subheader(f"QARS = {score:.3f} — {category}")
    left, right = st.columns([2, 1])
    with left:
        st.markdown("### Score breakdown")
        st.metric("Timeline (T)", f"{breakdown['T']:.3f}")
        st.metric("Sensitivity (S)", f"{breakdown['S']:.3f}")
        st.metric("Exposure (E)", f"{breakdown['E']:.3f}")
        st.markdown("#### Weights used")
        col_w1, col_w2, col_w3 = st.columns(3)
        col_w1.metric("wT", f"{weights_in['wT']:.3f}")
        col_w2.metric("wS", f"{weights_in['wS']:.3f}")
        col_w3.metric("wE", f"{weights_in['wE']:.3f}")

        # visual comparison: manual vs dynamic (if not using manual)
        with st.expander("Compare manual vs dynamic weights"):
            st.write("Dynamic weights (computed):", {k: round(v, 4) for k, v in dynamic_w.items()})
            st.write("Manual weights (from sliders):", {"wT": round(wT_manual, 4), "wS": round(wS_manual, 4), "wE": round(wE_manual, 4)})
            # show quick sensitivity: how much would score change if we switched to manual?
            total_manual = (wT_manual + wS_manual + wE_manual) or 1.0
            manual_norm = {"wT": wT_manual/total_manual, "wS": wS_manual/total_manual, "wE": wE_manual/total_manual}
            score_manual, _ = compute_qars(X=X, Y=Y, Z=Z, sensitivity=sensitivity, algorithm=algorithm, q=q, weights=manual_norm, alpha=alpha, linear_t=linear_t)
            st.write("Score with manual weights:", round(score_manual, 4), " (Δ = {:.3f})".format(score_manual - score))

    with right:
        st.markdown("### Actionable insight")
        if category == "Critical":
            st.error("Critical: immediate PQC migration or mitigations required.")
        elif category == "High":
            st.warning("High: prioritize in near-term migration roadmap.")
        elif category == "Medium":
            st.info("Medium: plan in standard cycles.")
        else:
            st.success("Low: monitor periodically.")

        # Provide prioritized recommendation based on dynamic weights
        if weights_in["wT"] > weights_in["wS"]:
            st.markdown("- Timeline urgency dominates sensitivity → accelerate migration planning.")
        if weights_in["wE"] > 0.4:
            st.markdown("- Exposure is significant → prioritize deployment of hybrid/PQC or reduce harvestability.")
        st.markdown("---")
        st.markdown("Switch to 'Portfolio dashboard (CSV)' mode to analyze many assets at once.")

elif mode == "Portfolio dashboard (CSV)":
    # Portfolio dashboard mode (CSV upload only)
    st.header("Portfolio Dashboard (upload CSV to use)")
    st.markdown("Upload a CSV with headers such as: Asset, Frequency, Algo, data shelf life, migration, vendor supply time, Data Sensisitivity, vendor pqc compliant, third party used, data trans(algo), App_type, Sector, Z")
    uploaded = st.file_uploader("Upload CSV file for batch scoring", type=["csv"])
    if uploaded:
        try:
            raw = uploaded.getvalue().decode("utf-8")
        except Exception:
            raw = uploaded.getvalue().decode("latin-1")
        out_csv = batch_score_csv_string(raw, weights=None, alpha=alpha, linear_t=linear_t)  # let batch scorer compute dynamic weights
        if not out_csv:
            st.error("No rows parsed from uploaded CSV.")
        else:
            import pandas as pd
            df = pd.read_csv(io.StringIO(out_csv))
            st.success(f"Scored {len(df)} assets")

            # Helper: find or create an Asset column for display (case-insensitive)
            def _find_or_make_asset_col(df):
                for c in df.columns:
                    if c and str(c).strip().lower() in ("asset", "name", "asset name", "asset_name", "identifier", "id"):
                        return c
                # no obvious asset column -> create one from index
                asset_names = [str(n) if str(n).strip() else f"asset_{i}" for i, n in enumerate(df.index)]
                df.insert(0, "Asset", asset_names)
                return "Asset"

            asset_col = _find_or_make_asset_col(df)
            # ensure QARS exists as numeric column
            if "QARS" not in df.columns:
                df["QARS"] = 0.0
            # coerce QARS to float where possible
            try:
                df["QARS"] = df["QARS"].astype(float)
            except Exception:
                df["QARS"] = df["QARS"].apply(lambda v: float(v) if pd.notna(v) and str(v).strip() != "" else 0.0)

            # ---------------------------
            # Executive-style Portfolio Dashboard
            # ---------------------------
            import altair as alt
            import numpy as np

            # summary metrics
            df["QARS_f"] = df["QARS"].astype(float)
            portfolio_avg_qars = float(df["QARS_f"].mean()) if len(df) else 0.0
            critical_count = int((df["QARS_f"] >= 0.85).sum())
            pct_critical = 100.0 * critical_count / max(1, len(df))

            # compute dynamic weights per row (using threat feed in session or loader)
            threat_feed_for_dashboard = threat_feed_data if threat_feed_data else load_threat_feed()
            wT_list, wS_list, wE_list = [], [], []
            for _, row in df.iterrows():
                row_dict = row.to_dict()
                sec = _parse_text_field(row_dict, ["Sector", "sector", "Industry", "industry"], default=None)
                w = compute_dynamic_weights_from_row(row_dict, sector=sec, threat_feed=threat_feed_for_dashboard)
                wT_list.append(w.get("wT", 0.0))
                wS_list.append(w.get("wS", 0.0))
                wE_list.append(w.get("wE", 0.0))

            avg_wT = float(np.mean(wT_list)) if wT_list else 0.0
            avg_wS = float(np.mean(wS_list)) if wS_list else 0.0
            avg_wE = float(np.mean(wE_list)) if wE_list else 0.0

            # Top row: gauge + dynamic weight stacked bar + quick cards
            st.subheader("Executive Summary")
            g1, g2, g3 = st.columns([1.2, 2, 1])
            with g1:
                st.markdown("**Portfolio QARS**")
                # simple gauge: use progress + numeric
                prog = min(1.0, max(0.0, portfolio_avg_qars))
                st.progress(prog)
                st.metric("Avg QARS", f"{portfolio_avg_qars:.3f}", delta=f"{pct_critical:.1f}% critical")
                st.write(f"Critical assets: {critical_count} / {len(df)}")
            # with g2:
            #     st.markdown("**Dynamic Weight Shift (portfolio average)**")
            #     st.write("Visualization removed per request. Portfolio averages for ω_T, ω_S, ω_E are shown in the Strategic Snapshot to the right.")
            # with g3:
                # st.markdown("**Strategic Snapshot**")
                # st.metric("Avg QARS", f"{portfolio_avg_qars:.3f}")
                # st.metric("Critical assets", f"{critical_count}")
                # st.metric("% Critical", f"{pct_critical:.1f}%")

            st.markdown("---")

            # Top priority assets
            st.subheader("Top Priority Assets (by QARS)")
            display_cols = [asset_col, "QARS"]
            # include migration cost if exists
            mc_key = next((c for c in df.columns if c.strip().lower() in ("migration cost", "migration_cost", "migrationcost")), None)
            if mc_key:
                display_cols.append(mc_key)
            # include weight exposure to explain priority
            df["_wE"] = wE_list
            display_cols.append("_wE")
            topn = df.sort_values(by="QARS_f", ascending=False)[display_cols].head(10).copy()
            topn = topn.rename(columns={asset_col: "Asset", "_wE": "wE"})
            st.table(topn)

            # Download prioritized CSV
            buf = io.StringIO()
            topn.to_csv(buf, index=False)
            st.download_button("Download top assets CSV", data=buf.getvalue(), file_name="qars_top_assets.csv", mime="text/csv")

            st.markdown("---")
            st.subheader("Portfolio Details (first 50 rows)")
            st.write(df.head(50))
            st.download_button("Download scored CSV", data=out_csv, file_name="qars_scored.csv", mime="text/csv")

elif mode == "Classifier (CSV)":
    # Classifier app mode: train DP classifier on scored CSV and allow adaptive updates
    st.header("Classifier (train DP classifier + adaptive learner)")
    st.markdown("Upload a scored CSV (output of Portfolio dashboard / batch scoring). The app will train a differentially-private binary classifier (critical vs non-critical) and expose an adaptive learner you can update with new instances.")

    import joblib
    import numpy as np
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
    from diffprivlib.models import LogisticRegression as DPLogisticRegression

    uploaded = st.file_uploader("Upload scored CSV for classifier training", type=["csv"])
    if uploaded:
        try:
            raw = uploaded.getvalue().decode("utf-8")
        except Exception:
            raw = uploaded.getvalue().decode("latin-1")

        df = dp_classify.prepare_df(raw)
        X_num, X_cat, num_cols, cat_cols = dp_classify.build_features(df)
        y = dp_classify.derive_label(df)

        if X_num.empty and X_cat.empty:
            st.error("No usable features found in CSV. Ensure scored CSV includes numeric fields (X_years, q_derived, T, S, E, etc.) or categorical fields (Algo, Sector).")
        else:
            st.sidebar.markdown("### DP training parameters")
            eps = st.sidebar.slider("DP epsilon", 0.01, 10.0, 1.0)
            delta = st.sidebar.number_input("DP delta", value=1e-5, format="%.0e")
            test_size = st.sidebar.slider("Test size fraction", 0.05, 0.5, 0.2)
            save_name = st.sidebar.text_input("Save model filename", value="models/dp_qars_model.joblib")

            # build preprocessing
            transformers = []
            if not X_num.empty:
                transformers.append(("num", StandardScaler(), num_cols))
            if not X_cat.empty:
                transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols))
            preproc = ColumnTransformer(transformers, remainder="drop")

            # DP classifier pipeline
            dp_clf = DPLogisticRegression(epsilon=eps, delta=delta, data_norm=10.0, max_iter=1000)
            dp_pipe = Pipeline([("preproc", preproc), ("clf", dp_clf)])

            # Adaptive (SGD) pipeline for online updates
            sgd_clf = SGDClassifier(loss="log_loss", learning_rate="optimal", max_iter=1000, random_state=42)
            adaptive_pipe = Pipeline([("preproc", preproc), ("clf", sgd_clf)])

            # Prepare X and split
            X_all = pd.concat([X_num, X_cat], axis=1) if (not X_num.empty and not X_cat.empty) else (X_num if not X_num.empty else X_cat)
            X_all = X_all.fillna(0.0)
            X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=test_size, random_state=42, stratify=y)

            # Train DP model
            with st.spinner("Training DP classifier..."):
                dp_pipe.fit(X_train, y_train)
            y_pred = dp_pipe.predict(X_test)
            st.subheader("DP classifier evaluation")
            st.text(classification_report(y_test, y_pred))
            try:
                y_prob = dp_pipe.predict_proba(X_test)[:, 1]
                st.write("ROC AUC:", roc_auc_score(y_test, y_prob))
            except Exception:
                pass

            # Train adaptive SGD model on same data (for online updates)
            adaptive_pipe.fit(X_train, y_train)
            st.subheader("Adaptive model initial accuracy")
            st.write("Accuracy:", accuracy_score(y_test, adaptive_pipe.predict(X_test)))

            # Save both models
            import os
            os.makedirs(os.path.dirname(save_name), exist_ok=True)
            joblib.dump({"dp": dp_pipe, "adaptive": adaptive_pipe, "num_cols": num_cols, "cat_cols": cat_cols}, save_name)
            st.success(f"Saved DP + adaptive pipelines to {save_name}")

            # --- Adaptive update UI ---
            st.markdown("---")
            st.subheader("Adaptive update (single new instance)")
            st.markdown("Provide a single-row CSV (same columns as scored CSV) or paste a CSV row. You must supply the true label for the new instance (Critical=1, Non-critical=0).")
            new_in = st.text_area("Paste single-row CSV (header + row) or leave empty to sample one from the uploaded data")
            use_sample = False
            if new_in.strip() == "":
                if st.button("Use a random sample from training data for update"):
                    new_row = df.sample(1, random_state=99)
                    use_sample = True
                else:
                    new_row = None
            else:
                try:
                    new_row = pd.read_csv(io.StringIO(new_in))
                except Exception as e:
                    st.error("Failed to parse pasted CSV row: " + str(e))
                    new_row = None

            if new_row is not None:
                st.write("New instance preview:")
                st.write(new_row.head(1))
                # derive new label (or ask user)
                if "QARS_category" in new_row.columns or "QARS" in new_row.columns:
                    suggested = int(dp_classify.derive_label(new_row, label_col="QARS_category")[0]) if "QARS_category" in new_row.columns else int(dp_classify.derive_label(new_row)[0])
                else:
                    suggested = 1
                true_label = st.selectbox("True label for new instance (0=non-critical, 1=critical)", options=[0,1], index=suggested)

                # transform new instance and perform partial_fit on adaptive classifier
                try:
                    # align columns before building features
                    Xn_num, Xn_cat, _, _ = dp_classify.build_features(pd.concat([df.head(0), new_row], ignore_index=True))
                    Xn = pd.concat([Xn_num, Xn_cat], axis=1) if (not Xn_num.empty and not Xn_cat.empty) else (Xn_num if not Xn_num.empty else Xn_cat)
                    Xn = Xn.fillna(0.0).iloc[-1:]
                    # transform via preproc
                    preproc = adaptive_pipe.named_steps["preproc"]
                    Xn_trans = preproc.transform(Xn)
                    clf = adaptive_pipe.named_steps["clf"]
                    # partial_fit (ensure classes provided)
                    clf.partial_fit(Xn_trans, np.array([true_label]), classes=np.array([0,1]))

                    # add small Gaussian noise to weights for DP-like protection
                    def _add_noise_sgd(clf_obj, sigma=0.01):
                        if hasattr(clf_obj, "coef_"):
                            noise = np.random.normal(0, sigma, clf_obj.coef_.shape)
                            clf_obj.coef_ += noise
                        if hasattr(clf_obj, "intercept_"):
                            clf_obj.intercept_ += np.random.normal(0, sigma, clf_obj.intercept_.shape)
                        return clf_obj

                    sigma = st.sidebar.slider("Adaptive noise sigma", 0.0, 0.5, 0.05)
                    _add_noise_sgd(clf, sigma=sigma)

                    # evaluate adaptive model after update
                    y_pred_adapt = adaptive_pipe.predict(X_test)
                    st.write("Adaptive model accuracy after update:", accuracy_score(y_test, y_pred_adapt))

                    # save updated adaptive model back to disk
                    joblib.dump({"dp": dp_pipe, "adaptive": adaptive_pipe, "num_cols": num_cols, "cat_cols": cat_cols}, save_name)
                    st.success("Adaptive model updated and saved.")
                except Exception as e:
                    st.error("Failed to perform adaptive update: " + str(e))

    else:
        st.info("Upload a scored CSV to train the classifier.")
