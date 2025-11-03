# QARS (Quantum-Adjusted Risk Score) Streamlit prototype

This workspace contains a lightweight implementation of the QARS model described
in the user's supplied article. It includes a pure-Python model module and a Streamlit
app to interactively compute and visualise scores.

Files added:
- `scripts/qars_model.py` — QARS model functions (compute_qars, helpers, presets).
- `scripts/streamlit_qars.py` — Streamlit UI to input parameters and view the score.
- `requirements.txt` — runtime dependency (Streamlit).

Quick start
1. (Optional) Create a virtual environment and activate it.
2. Install requirements:
```
pip install -r requirements.txt
```
3. Run the Streamlit app:
```
streamlit run scripts/streamlit_qars.py
```

Notes
- The model is intentionally simple and focuses on the core QARS equations from the
  article. It is ready for extension (persistence, richer exposure modelling, sector
  calibration UI, CSV import/export).

Further improvements (suggested):
- Add unit tests for the model functions.
- Add CSV import/export of asset inventories and batch scoring.
- Save scenarios and calibration presets.
