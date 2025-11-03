import os
import sys
import json

# ensure repo root in path so tests can import scripts package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts import qars_model as m


def make_compliant_row():
    return {
        "Asset": "compliant_asset",
        "Algo": "PQC-HYBRID",
        "Data Sensisitivity": "High",
        "data trans": "tls",
        "App_type": "internal",
        "vendor pqc compliant": "Yes",
        "third party used": "No",
        "vendor supply time": "1",
        "migration": "1",
        "patch cadence": "monthly",
        "backup": "replicated",
        "Owner": "Security Team",
    }


def make_noncompliant_row():
    return {
        "Asset": "noncompliant_asset",
        "Algo": "RSA-2048",
        "Data Sensisitivity": "",
        "data trans": "http",
        "App_type": "public",
        "vendor pqc compliant": "No",
        "third party used": "Yes",
        "vendor supply time": "12",
        "migration": "10",
        "patch cadence": "",
        "backup": "",
        "Owner": "",
    }


def test_check_compliance_for_compliant_row():
    row = make_compliant_row()
    res = m.check_compliance_for_row(row, standards=["NIST", "ISO", "NCSC"], threshold=0.8)
    assert isinstance(res, dict)
    assert "score" in res and 0.0 <= res["score"] <= 1.0
    assert res["compliant"] is True


def test_check_compliance_for_noncompliant_row():
    row = make_noncompliant_row()
    res = m.check_compliance_for_row(row, standards=["NIST", "ISO", "NCSC"], threshold=0.8)
    assert isinstance(res, dict)
    assert res["compliant"] is False
    # At least one control should be False
    assert any(not v for v in res["details"].values())


def test_flag_non_compliant_assets_list():
    rows = [make_compliant_row(), make_noncompliant_row()]
    flagged = m.flag_non_compliant_assets(rows, standards=["NIST", "ISO", "NCSC"], threshold=0.8)
    # flagged should contain only the noncompliant asset
    assert isinstance(flagged, list)
    ids = [f["asset"] for f in flagged]
    assert "noncompliant_asset" in ids
    assert "compliant_asset" not in ids