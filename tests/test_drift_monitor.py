"""Tests for the PSI drift monitor. Run: uv run pytest -q"""
import pandas as pd
import pytest

from src.drift_monitor import PSI_MODERATE_SHIFT, PSI_NO_SHIFT, check_drift, save_reference
from src.synth_data import generate


@pytest.fixture(scope="module", autouse=True)
def reference():
    save_reference(n=20_000, seed=42)


def test_no_drift_on_fresh_undrifted_sample():
    same_distribution = generate(3_000, seed=123)
    report = check_drift(same_distribution)
    assert report["max_psi"] < PSI_NO_SHIFT
    assert report["overall_verdict"] == "stable"
    assert report["action"] == "none"


def test_detects_major_drift():
    shifted = generate(3_000, seed=456)
    shifted["monthly_income"] = shifted["monthly_income"] * 0.5   # simulate a recession
    report = check_drift(shifted)
    assert report["per_feature"]["monthly_income"]["psi"] > PSI_MODERATE_SHIFT
    assert report["overall_verdict"] == "major_shift_retrain"
    assert "retrain" in report["action"]


def test_unaffected_features_stay_stable_during_partial_drift():
    """Only monthly_income shifts — other features should not be falsely flagged."""
    shifted = generate(3_000, seed=456)
    shifted["monthly_income"] = shifted["monthly_income"] * 0.5
    report = check_drift(shifted)
    assert report["per_feature"]["age"]["psi"] < PSI_NO_SHIFT
    assert report["per_feature"]["num_existing_loans"]["psi"] < PSI_NO_SHIFT


def test_missing_reference_raises(tmp_path, monkeypatch):
    import src.drift_monitor as dm
    monkeypatch.setattr(dm, "REFERENCE_PATH", tmp_path / "does_not_exist.json")
    with pytest.raises(FileNotFoundError):
        check_drift(pd.DataFrame({"age": [30]}))
