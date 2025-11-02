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
mode = st.sidebar.radio("App mode", ["Single asset demo", "Portfolio dashboard (CSV)"], index=0)
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

# Asset / Mode selection (mode selected in sidebar above)
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

else:
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

            # Dashboard sections removed per request.
            # Provide a simple preview and download so users can still inspect scored rows.
            st.subheader("Scored assets preview")
            st.write(df.head(50))
            st.download_button("Download scored CSV", data=out_csv, file_name="qars_scored.csv", mime="text/csv")
