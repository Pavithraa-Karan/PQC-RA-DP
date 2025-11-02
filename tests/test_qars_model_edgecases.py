import io
import csv
import sys
import os

# ensure repo root in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts import qars_model as m


def test_missing_fields_row_does_not_crash_and_has_qars():
    # Row missing many of the expected headers
    row = {"Algo": "RSA"}
    out = m.score_row_from_csv_row(row)
    assert "QARS" in out
    assert out["QARS_category"] in ("Low", "Medium", "High", "Critical")


def test_unusual_header_names_matching_is_case_insensitive_and_strips_spaces():
    csv_text = (
        " Frequency , Key Size , DATA SELF LIFE , migration , vendor supply time , Algo , App_type , Application , Data Sensisitivity , data trans(algo) , third party used , vendor pqc compliant\n"
        "soon,2048,30,2,1.0,RSA,WEB,ServiceX,CRITICAL,INTERNET,Yes,No\n"
    )
    out = m.batch_score_csv_string(csv_text)
    assert out
    rdr = csv.DictReader(io.StringIO(out))
    rows = list(rdr)
    assert len(rows) == 1
    r0 = rows[0]
    assert r0["QARS"]


def test_pqc_algorithm_results_in_zero_exposure():
    # Use algorithm 'PQC' (assumed non-breakable) with high q; E should be 0
    score, breakdown = m.compute_qars(X=5, Y=1, Z=8, sensitivity="High", algorithm="PQC", q=1.0)
    assert breakdown["E"] == 0.0


def test_linear_timeline_mapping_hits_one_when_r_greater_than_one():
    # X+Y=20, Z=12 -> r=20/12 > 1; with linear_t True, T should be min(1,r) => 1.0
    score_lin, breakdown_lin = m.compute_qars(X=15, Y=5, Z=12, sensitivity="Moderate", algorithm="RSA", q=0.5, linear_t=True)
    assert abs(breakdown_lin["T"] - 1.0) < 1e-9
    # logistic shouldn't necessarily be exactly 1.0
    score_log, breakdown_log = m.compute_qars(X=15, Y=5, Z=12, sensitivity="Moderate", algorithm="RSA", q=0.5, linear_t=False)
    assert breakdown_log["T"] <= 1.0
    assert breakdown_log["T"] > 0.9  # logistic with default alpha should be high but <1
