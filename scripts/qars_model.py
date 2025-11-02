"""
QARS model implementation

Provides functions to compute the Quantum-Adjusted Risk Score (QARS) described
in the user's article. The implementation is intentionally lightweight and has no
external run-time dependencies beyond the Python standard library.

Usage:
  from qars_model import compute_qars, sector_profiles
  params = dict(X=15, Y=1, Z=12, sensitivity='High', algorithm='RSA', q=0.3)
  score, breakdown = compute_qars(**params)

The Streamlit app (`scripts/streamlit_qars.py`) imports and uses these helpers.
"""
from __future__ import annotations

import math
import csv
import io
from typing import Dict, Tuple, List

# Sensitivity mapping (default)
SENSITIVITY_MAP = {
    "Low": 0.25,
    "Moderate": 0.5,
    "High": 0.75,
    "Critical": 1.0,
}

# Default sector weight presets
sector_profiles = {
    "Default": {"wT": 1/3, "wS": 1/3, "wE": 1/3},
    "Finance": {"wT": 0.4, "wS": 0.4, "wE": 0.2},
    "IoT/Embedded": {"wT": 0.5, "wS": 0.2, "wE": 0.3},
    "Cloud": {"wT": 0.3, "wS": 0.2, "wE": 0.5},
}

# Algorithm families considered classically breakable by CRQC (simplified)
BREAKABLE_ALGOS = {"RSA", "ECC", "DH", "DSA"}


def _safe_logistic(x: float, alpha: float = 10.0, x0: float = 1.0) -> float:
    """Numerically stable logistic mapping into (0,1).

    Parameters
    - x: input
    - alpha: steepness
    - x0: mid-point where logistic equals 0.5
    """
    # avoid overflow for large negative/positive values
    z = alpha * (x - x0)
    if z >= 0:
        try:
            return 1.0 / (1.0 + math.exp(-z))
        except OverflowError:
            return 1.0
    else:
        try:
            ez = math.exp(z)
            return ez / (1.0 + ez)
        except OverflowError:
            return 0.0


def ftime(r: float, alpha: float = 10.0, linear: bool = False) -> float:
    """Timeline scaling function mapping r = (X+Y)/Z into [0,1].

    If linear=True uses min(1, r) for interpretability; otherwise logistic with
    midpoint at r=1 and steepness alpha.
    """
    if r <= 0:
        return 0.0
    if linear:
        return min(1.0, r)
    return _safe_logistic(r, alpha=alpha, x0=1.0)


def fsens(s_label: str) -> float:
    """Map a sensitivity label to numeric score in [0,1]."""
    return float(SENSITIVITY_MAP.get(s_label, 0.5))


def fexpos(v: int, q: float) -> float:
    """Exposure scaling: simple product of cryptographic visibility and harvestability.

    v: 0 or 1 (or a float 0..1 in future refined versions)
    q: harvestability in [0,1]
    """
    v = 1.0 if v else 0.0
    q = max(0.0, min(1.0, float(q)))
    return v * q


def compute_T(X: float, Y: float, Z: float, alpha: float = 10.0, linear: bool = False) -> float:
    """Compute timeline factor T(a).

    Handles edge cases: if Z <= 0 (no estimate) we treat threat horizon as far away
    and set a conservative small T unless X+Y is large.
    """
    # avoid division by zero
    if Z is None or Z <= 0:
        # if no horizon is given, interpret as low immediate risk but allow large X+Y
        r = (X + Y) / 1e6
    else:
        r = (X + Y) / float(Z)
    return float(ftime(r, alpha=alpha, linear=linear))


def compute_S(sensitivity_label: str) -> float:
    return float(fsens(sensitivity_label))


def compute_E(algorithm: str, q: float) -> float:
    """Backward-compatible compute_E: if only algorithm+q provided, fall back to
    the original binary visibility * q behaviour. New code should use
    compute_E_from_attributes for a richer exposure model.
    """
    v = 1 if (algorithm and algorithm.upper() in BREAKABLE_ALGOS) else 0
    return float(fexpos(v, q))


def compute_E_from_attributes(
    algorithm: str,
    key_size: float | None = None,
    data_trans: str | None = None,
    app_type: str | None = None,
    frequency: float | None = None,
    data_shelf_life: float | None = None,
    third_party_used: bool = False,
    third_party_qsafe: bool | None = None,
    arch_flex: str | None = None,
    vendor_pqc_compliant: bool | None = None,
    wC: float = 0.5,
    wO: float = 0.5,
) -> float:
    """Compute refined Exposure E(a) from detailed attributes.

    Returns a float in [0,1].
    """
    alg = (algorithm or "").strip().upper()

    # Detect PQC/hybrid
    is_pqc = False
    if "PQC" in alg or "HYBRID" in alg:
        is_pqc = True

    # E_Alg
    if is_pqc:
        E_Alg = 0.0
    elif alg in BREAKABLE_ALGOS:
        E_Alg = 1.0
    else:
        # handle AES with key size
        if alg.startswith("AES") or "AES" in alg:
            try:
                ks = float(key_size) if key_size is not None else 128.0
            except Exception:
                ks = 128.0
            E_Alg = 0.05 if ks >= 256 else 0.3
        else:
            # unknown algorithm -> moderate-high
            E_Alg = 0.7

    # E_FS (Agility/Forward Secrecy)
    af = (arch_flex or "").strip().lower()
    if any(k in af for k in ("pfs", "forward", "hybrid", "pqc", "agile")):
        E_FS = 0.2
    elif vendor_pqc_compliant:
        E_FS = 0.3
    else:
        E_FS = 1.0

    # If algorithm is PQC, treat cryptographic vulnerability as zero regardless of FS
    if is_pqc:
        E_Crypto = 0.0
    else:
        E_Crypto = max(E_Alg, E_FS)

    # E_Ops: E_Int, E_Vol, E_3P
    dt = (data_trans or "").strip().lower()
    at = (app_type or "").strip().lower()
    if any(k in at for k in ("public", "web", "api", "cloud")) or any(k in dt for k in ("internet", "http", "https", "tls")):
        E_Int = 1.0
    elif any(k in at for k in ("internal", "intranet", "air", "air-gap", "airgap")) or "internal" in dt:
        E_Int = 0.1
    else:
        E_Int = 0.5

    # E_Vol from frequency and data shelf life
    try:
        fq = float(frequency) if frequency is not None else 0.0
    except Exception:
        fq = 0.0
    try:
        shelf = float(data_shelf_life) if data_shelf_life is not None else 0.0
    except Exception:
        shelf = 0.0
    # normalize
    norm_freq = min(1.0, math.log1p(fq) / math.log1p(1000.0)) if fq > 0 else 0.0
    norm_shelf = min(1.0, shelf / 30.0) if shelf > 0 else 0.0
    avg_norm = (norm_freq + norm_shelf) / 2.0
    E_Vol = 0.2 + 0.8 * avg_norm

    # E_3P
    if not third_party_used:
        E_3P = 0.1
    else:
        if third_party_qsafe is None:
            # unknown -> medium
            E_3P = 0.6
        elif third_party_qsafe:
            E_3P = 0.4
        else:
            E_3P = 1.0

    E_Ops = (E_Int + E_Vol + E_3P) / 3.0

    # Combine: multiplicative model as requested
    E_final = E_Crypto * E_Ops
    return float(max(0.0, min(1.0, E_final)))


def compute_qars(
    X: float,
    Y: float,
    Z: float,
    sensitivity: str = "Moderate",
    algorithm: str = "RSA",
    q: float = 0.5,
    weights: Dict[str, float] | None = None,
    alpha: float = 10.0,
    linear_t: bool = False,
) -> Tuple[float, Dict[str, float]]:
    """Compute QARS and return (score, breakdown).

    Inputs:
    - X: confidentiality duration (years)
    - Y: migration time (years)
    - Z: projected time until adversary CRQC (years)
    - sensitivity: one of the sensitivity labels in SENSITIVITY_MAP
    - algorithm: string naming the crypto primitive (e.g., 'RSA', 'PQC')
    - q: harvestability [0,1]
    - weights: dict with keys 'wT','wS','wE' summing to 1; if None use equal
    - alpha, linear_t: parameters for timeline scaling

    Returns:
    - score in [0,1]
    - breakdown dict with T,S,E and used weights
    """
    # Coerce numeric inputs and guard negatives
    X = max(0.0, float(X))
    Y = max(0.0, float(Y))
    Z = float(Z) if Z is not None else 0.0
    q = max(0.0, min(1.0, float(q)))

    if weights is None:
        wT = wS = wE = 1.0 / 3.0
    else:
        wT = float(weights.get("wT", 0.0))
        wS = float(weights.get("wS", 0.0))
        wE = float(weights.get("wE", 0.0))
        total = wT + wS + wE
        if total <= 0:
            wT = wS = wE = 1.0 / 3.0
        else:
            wT /= total
            wS /= total
            wE /= total

    T = compute_T(X, Y, Z, alpha=alpha, linear=linear_t)
    S = compute_S(sensitivity)
    E = compute_E(algorithm, q)

    score = wT * T + wS * S + wE * E
    breakdown = {
        "T": T,
        "S": S,
        "E": E,
        "wT": wT,
        "wS": wS,
        "wE": wE,
    }
    return float(score), breakdown


def qars_category(score: float, bands: Dict[str, float] | None = None) -> str:
    """Map score in [0,1] to a qualitative band.

    Default bands: low <0.30, medium 0.30–0.60, high >0.60, critical >0.85.
    """
    s = max(0.0, min(1.0, float(score)))
    if bands is None:
        # thresholds are lower bounds for categories (exclusive of previous)
        bands = {"critical": 0.85, "high": 0.60, "medium": 0.30, "low": 0.0}
    if s >= bands["critical"]:
        return "Critical"
    if s >= bands["high"]:
        return "High"
    if s >= bands["medium"]:
        return "Medium"
    return "Low"


### CSV batch scoring helpers


def _find_key(row: Dict[str, str], candidates: List[str]) -> str | None:
    """Find a key in row matching any candidate (case-insensitive, strip).

    Returns the matched key name or None.
    """
    # some csv.DictReader variants may include None as a key when rows have
    # more fields than headers; skip None keys defensively
    norm = {k.strip().lower(): k for k in row.keys() if k is not None}
    for c in candidates:
        k = c.strip().lower()
        if k in norm:
            return norm[k]
    return None


def _parse_float_field(row: Dict[str, str], candidates: List[str], default: float = 0.0) -> float:
    key = _find_key(row, candidates)
    if not key:
        return default
    val = row.get(key, "")
    if val is None:
        return default
    val = str(val).strip()
    if val == "":
        return default
    # try to extract number
    try:
        # remove non-digit chars except dot and minus
        cleaned = "".join(ch for ch in val if (ch.isdigit() or ch in ".-"))
        return float(cleaned)
    except Exception:
        return default


def _parse_yesno_field(row: Dict[str, str], candidates: List[str]) -> bool:
    key = _find_key(row, candidates)
    if not key:
        return False
    v = row.get(key, "")
    if v is None:
        return False
    v = str(v).strip().lower()
    return v in ("yes", "y", "true", "1", "t")


def _parse_text_field(row: Dict[str, str], candidates: List[str], default: str = "") -> str:
    key = _find_key(row, candidates)
    if not key:
        return default
    v = row.get(key, "")
    if v is None:
        return default
    return str(v).strip()


def _derive_q_from_row(row: Dict[str, str]) -> float:
    """Heuristic to compute harvestability q in [0,1] from available attributes.

    Rules (simple heuristic): base 0.5; increase for public transport, public apps,
    third-party non-PQC, vendor non-compliant, etc.
    """
    q = 0.5
    data_trans = _parse_text_field(row, ["data trans(algo)", "data trans", "data transmission", "data trans"], "").lower()
    app_type = _parse_text_field(row, ["app_type", "app type", "App_type", "Application type", "Application"], "").lower()
    if any(k in data_trans for k in ("internet", "http", "https", "tls", "tcp", "udp", "public")):
        q += 0.25
    if any(k in app_type for k in ("public", "web", "api", "cloud")):
        q += 0.15
    # third-party quantum safe
    third_safe = _parse_yesno_field(row, ["is third party quantum safe", "is third party quantum safe?", "third party quantum safe"]) 
    third_used = _parse_yesno_field(row, ["third party used", "third_party_used", "third party used?"])
    if not third_safe and third_used:
        q += 0.15
    # vendor pqc compliant
    vendor_ok = _parse_yesno_field(row, ["vendor pqc compliant", "vendor_pqc_compliant", "vendor pqc"])
    if not vendor_ok and third_used:
        q += 0.15
    return min(1.0, q)


def score_row_from_csv_row(row: Dict[str, str], weights: Dict[str, float] | None = None, alpha: float = 10.0, linear_t: bool = False) -> Dict[str, str]:
    """Map CSV row to QARS inputs, compute score and return augmented row dict.

    Mapping assumptions (inferred from provided headers):
    - X (confidentiality years) <- 'data self life' or 'data shelf life' numeric
    - Y (migration years) <- 'migration' numeric; if 'vendor supply time' present it's added to migration
    - Z (threat horizon) <- try 'Frequency' if numeric or mapped categories; otherwise default 12
    - sensitivity <- 'Data Sensisitivity' mapping to labels
    - algorithm <- 'Algo'
    - q derived heuristically from transmission, app type, third-party columns
    """
    # Defaults
    X = _parse_float_field(row, ["data self life", "data shelf life", "data selflife", "data_shelf_life"], default=1.0)
    Y = _parse_float_field(row, ["migration", "migration time"], default=1.0)
    vendor_supply = _parse_float_field(row, ["vendor supply time", "vendor_supply_time"], default=0.0)
    Y = max(0.0, Y + vendor_supply)

    # Z: projected CRQC horizon is not provided in the CSV by default.
    # Do not use 'Frequency' as Z (Frequency is usage count). Instead, allow an
    # explicit 'Z'/'projected' column; otherwise default to 12 years.
    Z = _parse_float_field(row, ["Z", "projected crqc", "projected crqc horizon", "projected_crqc", "projected_z"], default=12.0)

    # Frequency: how often the algorithm is used for this asset; affects harvestability q
    frequency = _parse_float_field(row, ["Frequency", "frequency", "freq"], default=0.0)

    algorithm = _parse_text_field(row, ["Algo", "algo", "Algorithm"], default="RSA")
    sensitivity = _parse_text_field(row, ["Data Sensisitivity", "data sensisitivity", "data sensitivity", "Data Sensitivity"], default="Moderate")

    # derive q from row and include frequency influence
    q = _derive_q_from_row(row)
    # scale up q a bit if the algorithm is used many times (harvestability increases)
    try:
        if frequency and float(frequency) > 0:
            # normalize frequency impact: use log-scale to avoid runaway values
            fq = float(frequency)
            q = min(1.0, q + min(0.3, math.log1p(fq) / 10.0))
    except Exception:
        pass

    score, breakdown = compute_qars(X=X, Y=Y, Z=Z, sensitivity=sensitivity, algorithm=algorithm, q=q, weights=weights, alpha=alpha, linear_t=linear_t)

    out = dict(row)  # copy original fields
    out.update({
        "QARS": f"{score:.6f}",
        "QARS_category": qars_category(score),
        "T": f"{breakdown['T']:.6f}",
        "S": f"{breakdown['S']:.6f}",
        "E": f"{breakdown['E']:.6f}",
        "q_derived": f"{q:.3f}",
        "X_years": f"{X}",
        "Y_years": f"{Y}",
        "Z_years": f"{Z}",
    })
    return out


def batch_score_csv_string(csv_text: str, weights: Dict[str, float] | None = None, alpha: float = 10.0, linear_t: bool = False) -> str:
    """Read CSV text, score rows, and return CSV text with appended QARS columns."""
    input_io = io.StringIO(csv_text)
    reader = csv.DictReader(input_io)
    rows = list(reader)
    if not rows:
        return ""
    out_rows = []
    for r in rows:
        out_rows.append(score_row_from_csv_row(r, weights=weights, alpha=alpha, linear_t=linear_t))
    # write out
    out_io = io.StringIO()
    fieldnames = list(out_rows[0].keys())
    writer = csv.DictWriter(out_io, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for r in out_rows:
        writer.writerow(r)
    return out_io.getvalue()


if __name__ == "__main__":
    # simple demo when invoked directly
    demo = dict(X=15, Y=1, Z=12, sensitivity="High", algorithm="RSA", q=0.3)
    score, breakdown = compute_qars(**demo)
    cat = qars_category(score)
    print(f"QARS={score:.3f} ({cat})")
    print("Breakdown:", breakdown)
