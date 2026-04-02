"""Post-train artifact and quality gate tests.

Run after CI training completes. Reads TRAIN_LOSS_THRESHOLD env var
(default 10.0) for the quality gate assertion.
"""

import glob
import json
import math
import os

TRAIN_LOSS_THRESHOLD = float(os.environ.get("TRAIN_LOSS_THRESHOLD", "10.0"))


def test_metrics_json_exists():
    assert os.path.exists("metrics.json"), (
        "metrics.json not found in working directory — did training complete?"
    )


def test_metrics_keys():
    with open("metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)
    assert "train/loss" in metrics, (
        f"'train/loss' key missing from metrics.json. Keys found: {list(metrics.keys())}"
    )
    value = float(metrics["train/loss"])
    assert math.isfinite(value), f"train/loss is not finite: {value}"


def test_quality_gate_train_loss():
    with open("metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)
    loss = float(metrics["train/loss"])
    assert loss <= TRAIN_LOSS_THRESHOLD, (
        f"Quality Gate failed: train/loss={loss:.4f} > threshold={TRAIN_LOSS_THRESHOLD:.2f}"
    )


def test_checkpoint_exists():
    checkpoints = glob.glob("runs/ci/**/*.pth", recursive=True)
    assert len(checkpoints) >= 1, (
        "No .pth checkpoint found under runs/ci/ — did training save a final model?"
    )
