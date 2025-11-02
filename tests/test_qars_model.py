import io
import csv
import sys
import os

# Ensure repository root is on sys.path so we can import scripts package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts import qars_model as m


def test_compute_qars_returns_valid_range_and_breakdown():
    score, breakdown = m.compute_qars(X=15, Y=1, Z=12, sensitivity="High", algorithm="RSA", q=0.3)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    assert set(breakdown.keys()) >= {"T", "S", "E", "wT", "wS", "wE"}


def test_qars_category_thresholds():
    assert m.qars_category(0.0) == "Low"
    assert m.qars_category(0.3) == "Medium"
    assert m.qars_category(0.6) == "High"
    assert m.qars_category(0.9) == "Critical"


def test_score_row_from_csv_row_has_expected_columns():
    row = {
        "Frequency": "12",
        "key size": "2048",
        "data self life": "15",
        "migration": "1",
        "vendor supply time": "0.5",
        "Algo": "RSA",
        "App_type": "web",
        "Application": "ServiceA",
        "Data Sensisitivity": "High",
        "arch flexibility": "low",
        "data trans(algo)": "internet",
        "is third party quantum safe": "No",
        "third party used": "Yes",
        "vendor pqc compliant": "No",
    }
    out = m.score_row_from_csv_row(row)
    # check augmented fields
    assert "QARS" in out
    assert "QARS_category" in out
    assert "T" in out and "S" in out and "E" in out
    assert float(out["q_derived"]) <= 1.0


def test_batch_score_csv_string_outputs_rows_and_headers():
    sample_csv = (
        "Frequency,key size,data self life,migration,vendor supply time,Algo,App_type,"
        "Application,Data Sensisitivity,arch flexibility,data trans(algo),is third party quantum safe,"
        "third party used,vendor pqc compliant\n"
        "12,2048,15,1,0.5,RSA,web,ServiceA,High,low,internet,No,Yes,No\n"
    )
    out_csv = m.batch_score_csv_string(sample_csv)
    assert out_csv
    # parse output CSV and check fields
    rdr = csv.DictReader(io.StringIO(out_csv))
    rows = list(rdr)
    assert len(rows) == 1
    r0 = rows[0]
    assert "QARS" in r0 and "QARS_category" in r0
    assert float(r0["T"]) >= 0.0 and float(r0["T"]) <= 1.0
