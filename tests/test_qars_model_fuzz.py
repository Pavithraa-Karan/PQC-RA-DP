import io
import csv
import random
import string
import sys
import os

# ensure repo root on path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts import qars_model as m


def test_header_variants_and_pqc_algo():
    # Headers with spaces, caps, underscores and trailing spaces
    headers = [
        " Frequency",
        "Key_Size",
        "DATA SELF life",
        "migration ",
        "vendor supply time",
        "Algo",
        "App Type",
        "Application",
        "Data Sensisitivity",
        "data trans(algo)",
        "is third party quantum safe",
        "third party used",
        "vendor pqc compliant",
    ]
    header_line = ",".join(headers) + "\n"
    # Use PQC (non-breakable) to check E==0 in output
    data_line = "soon,2048,30,2,1.0,PQC,public,ServiceX,Critical,https,No,Yes,Yes\n"
    out = m.batch_score_csv_string(header_line + data_line)
    assert out
    rdr = csv.DictReader(io.StringIO(out))
    rows = list(rdr)
    assert len(rows) == 1
    r0 = rows[0]
    assert r0["QARS"]
    # E should be 0 for PQC algorithm
    assert float(r0["E"]) == 0.0


def test_malformed_csv_missing_commas_and_blank_lines():
    # Missing some commas in rows and blank lines
    csv_text = "Frequency,key size,data self life\n12,2048\n\n,,\n"
    out = m.batch_score_csv_string(csv_text)
    # Should return a CSV string (possibly only header)
    assert isinstance(out, str)
    # Should not raise and should be parseable by csv.DictReader
    rdr = csv.DictReader(io.StringIO(out))
    _ = list(rdr)  # just ensure parsing works


def test_malformed_csv_extra_columns():
    # Some rows have extra columns; ensure function doesn't crash
    csv_text = "Frequency,key size,data self life\n12,2048,15,unexpected_extra\n"
    out = m.batch_score_csv_string(csv_text)
    assert isinstance(out, str)
    rdr = csv.DictReader(io.StringIO(out))
    rows = list(rdr)
    assert len(rows) == 1


def test_random_fuzz_rows_no_crash():
    random.seed(42)
    # Create random rows with random keys/values and exercise score_row_from_csv_row
    for _ in range(200):
        # random small set of keys
        keys = [
            "".join(random.choices(string.ascii_letters + " _", k=random.randint(3, 12))).strip()
            for _ in range(random.randint(3, 8))
        ]
        row = {k: "".join(random.choices(string.printable, k=random.randint(0, 20))) for k in keys}
        # occasionally include some known keys
        if random.random() < 0.5:
            row["Algo"] = random.choice(["RSA", "ECC", "PQC", "DH"])
        if random.random() < 0.5:
            row["data self life"] = str(random.randint(0, 50))
        if random.random() < 0.5:
            row["migration"] = str(random.uniform(0, 5))
        # Must not raise
        out = m.score_row_from_csv_row(row)
        assert "QARS" in out
        assert out["QARS_category"] in ("Low", "Medium", "High", "Critical")


def test_high_volume_long_values_fast():
    # Construct a CSV with many rows and very long fields to ensure performance/robustness
    header = "Frequency,key size,data self life,migration,Algo,data trans(algo)\n"
    lines = [header]
    for i in range(150):
        long_field = "x" * (i % 50 + 1)
        lines.append(f"12,2048,15,1,RSA,{long_field}\n")
    out = m.batch_score_csv_string("".join(lines))
    assert out
    rdr = csv.DictReader(io.StringIO(out))
    rows = list(rdr)
    assert len(rows) == 150
