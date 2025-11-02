import json
import os
import sys

# ensure repo root in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts import qars_model as m


def test_threat_increases_timeline_weight():
    """When sector threat severity is high, wT should increase relative to the sector prior."""
    row = {}
    sector = "Finance"
    base_wT = m.sector_profiles["Finance"]["wT"]
    threat_feed = {"Finance": 1.0}
    weights = m.compute_dynamic_weights_from_row(row, sector=sector, threat_feed=threat_feed)
    assert abs(weights["wT"] + weights["wS"] + weights["wE"] - 1.0) < 1e-9
    assert weights["wT"] > base_wT


def test_vendor_agility_reduces_vas_and_increases_wE():
    """Poor vendor agility + global threat pressure should increase wE above the default."""
    row = {
        "vendor pqc compliant": "No",
        "third party used": "Yes",
        "vendor supply time": "10",  # long supply time
        "Algo": "RSA",
        "data trans": "internet",
        "App_type": "public",
    }
    threat_feed = {"global": 0.8}
    base_wE = m.sector_profiles["Default"]["wE"]
    weights = m.compute_dynamic_weights_from_row(row, sector=None, threat_feed=threat_feed, critical_supply_time=5.0)
    assert abs(weights["wT"] + weights["wS"] + weights["wE"] - 1.0) < 1e-9
    assert weights["wE"] > base_wE


def test_load_threat_feed_reads_file(tmp_path):
    """load_threat_feed should read JSON and normalize numeric values to floats 0..1."""
    data = {"Finance": 0.75, "global": 0.2}
    p = tmp_path / "threat_feed.json"
    p.write_text(json.dumps(data))
    res = m.load_threat_feed(path=str(p))
    assert "Finance" in res
    assert abs(res["Finance"] - 0.75) < 1e-9
    assert res["global"] == 0.2