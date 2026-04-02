"""Pre-train data validation tests.

No torch or hydra imports — runs in < 1 second with no model loading.
Reads DATA_PROCESSED env var (defaults to tests/fixtures) to locate CSVs.
"""

import csv
import os
from pathlib import Path

DATA_PROCESSED = Path(os.environ.get("DATA_PROCESSED", "tests/fixtures"))


def test_train_csv_exists():
    assert (DATA_PROCESSED / "train.csv").exists(), (
        f"train.csv not found at {DATA_PROCESSED / 'train.csv'}"
    )


def test_val_csv_exists():
    assert (DATA_PROCESSED / "val.csv").exists(), (
        f"val.csv not found at {DATA_PROCESSED / 'val.csv'}"
    )


def test_csv_schema():
    with open(DATA_PROCESSED / "train.csv", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = [name.strip().lower() for name in (reader.fieldnames or [])]
    assert "source" in fieldnames, f"'source' column missing, got: {fieldnames}"
    assert "target" in fieldnames, f"'target' column missing, got: {fieldnames}"


def test_csv_min_rows():
    with open(DATA_PROCESSED / "train.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 1, f"train.csv has {len(rows)} data rows, expected at least 1"
