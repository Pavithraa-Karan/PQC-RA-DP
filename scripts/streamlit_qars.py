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
from qars_model import (
    compute_qars,
    qars_category,
    sector_profiles,
    SENSITIVITY_MAP,
    batch_score_csv_string,
    score_row_from_csv_row,
)


st.set_page_config(page_title="QARS - Quantum-Adjusted Risk Score", layout="wide")

st.title("Quantum-Adjusted Risk Score (QARS)")
st.sidebar.header("Input Parameters")

# Sector presets
sector = st.sidebar.selectbox("Sector preset", list(sector_profiles.keys()))
weights = sector_profiles.get(sector, sector_profiles["Default"]) if sector else sector_profiles["Default"]

st.sidebar.markdown("**Calibration / Weights**")
wT = st.sidebar.slider("Weight: Timeline (wT)", 0.0, 1.0, float(weights["wT"]))
wS = st.sidebar.slider("Weight: Sensitivity (wS)", 0.0, 1.0, float(weights["wS"]))
wE = st.sidebar.slider("Weight: Exposure (wE)", 0.0, 1.0, float(weights["wE"]))

st.sidebar.markdown("**Timeline scaling**")
linear_t = st.sidebar.checkbox("Use linear timeline scaling (min(1,r))", value=False)
alpha = st.sidebar.slider("Logistic steepness alpha", 1.0, 40.0, 10.0)

st.header("Asset parameters")
col1, col2, col3 = st.columns(3)
with col1:
    X = st.number_input("Confidentiality duration X (years)", min_value=0.0, value=15.0)
    Y = st.number_input("Migration time Y (years)", min_value=0.0, value=1.0)
with col2:
    Z = st.number_input("Projected CRQC horizon Z (years)", min_value=0.0, value=12.0)
    q = st.slider("Harvestability q (0-1)", 0.0, 1.0, 0.3)
with col3:
    sensitivity = st.selectbox("Data sensitivity", list(SENSITIVITY_MAP.keys()), index=2)
    algorithm = st.text_input("Algorithm (e.g., RSA, ECC, PQC)", value="RSA")

st.markdown("---")

weights_dict = {"wT": wT, "wS": wS, "wE": wE}

score, breakdown = compute_qars(X=X, Y=Y, Z=Z, sensitivity=sensitivity, algorithm=algorithm, q=q, weights=weights_dict, alpha=alpha, linear_t=linear_t)
category = qars_category(score)

col_a, col_b = st.columns([2, 1])
with col_a:
    st.subheader(f"QARS = {score:.3f} — {category}")
    st.write("Breakdown:")
    st.write(breakdown)
    st.markdown("#### Interpretation")
    if category == "Critical":
        st.error("Critical: immediate PQC migration or compensating controls required.")
    elif category == "High":
        st.warning("High: prioritize for near-term migration planning.")
    elif category == "Medium":
        st.info("Medium: schedule in normal upgrade cycles.")
    else:
        st.success("Low: standard maintenance is sufficient.")

with col_b:
    st.subheader("What affects this score?")
    st.metric("Timeline (T)", f"{breakdown['T']:.3f}")
    st.metric("Sensitivity (S)", f"{breakdown['S']:.3f}")
    st.metric("Exposure (E)", f"{breakdown['E']:.3f}")
    st.markdown("---")
    st.write("Weights used:")
    st.write({"wT": f"{breakdown['wT']:.3f}", "wS": f"{breakdown['wS']:.3f}", "wE": f"{breakdown['wE']:.3f}"})

st.markdown("---")
st.markdown("### Quick examples")
if st.button("Load example assets"):
    exA = dict(X=15, Y=1, Z=12, sensitivity="High", algorithm="RSA", q=0.3)
    exB = dict(X=1, Y=2, Z=12, sensitivity="Low", algorithm="RSA", q=0.8)
    sA, bA = compute_qars(**exA, weights=weights_dict, alpha=alpha, linear_t=linear_t)
    sB, bB = compute_qars(**exB, weights=weights_dict, alpha=alpha, linear_t=linear_t)
    st.write("Asset A:", {"score": round(sA, 3), "cat": qars_category(sA)})
    st.write("Breakdown A:", bA)
    st.write("Asset B:", {"score": round(sB, 3), "cat": qars_category(sB)})
    st.write("Breakdown B:", bB)


st.markdown("---")
st.header("Batch scoring (CSV)")
st.markdown("Upload a CSV with headers such as: Frequency, key size, data self life, migration, vendor supply time, Algo, App_type, Application, Data Sensisitivity, arch flexibility, data trans(algo), is third party quantum safe, third party used, vendor pqc compliant.")
uploaded = st.file_uploader("Upload CSV file for batch scoring", type=["csv"])
if uploaded is not None:
    try:
        raw = uploaded.getvalue().decode("utf-8")
    except Exception:
        raw = uploaded.getvalue().decode("latin-1")
    out_csv = batch_score_csv_string(raw, weights=weights_dict, alpha=alpha, linear_t=linear_t)
    if out_csv:
        st.success("Scoring complete — preview below")
        # show first 50 lines as a table
        import csv as _csv
        from io import StringIO

        rdr = _csv.DictReader(StringIO(out_csv))
        rows = list(rdr)
        if rows:
            # display first 50
            st.write(rows[:50])
        st.download_button("Download scored CSV", data=out_csv, file_name="qars_scored.csv", mime="text/csv")
    else:
        st.warning("No rows found in uploaded CSV or parsing failed.")
