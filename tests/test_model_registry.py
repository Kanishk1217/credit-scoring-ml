"""Unit tests for the model registry builder. Run: uv run pytest -q"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from build_model_registry import MODEL_DIRS, build  # noqa: E402


def test_build_registers_all_known_models():
    """Registered models are a subset of MODEL_DIRS -- a dir can be listed ahead of actually
    being trained (e.g. models_lendingclub before its first training run), so this only asserts
    the models that DO exist on disk get picked up, not that every listed dir must exist yet."""
    reg = build(registered_at="2026-01-01T00:00:00+00:00")
    assert set(reg["models"].keys()) <= set(MODEL_DIRS)
    assert {"models", "models_real", "models_real_rich"} <= set(reg["models"].keys())


def test_build_fingerprints_are_stable_and_distinct():
    reg = build(registered_at="2026-01-01T00:00:00+00:00")
    fingerprints = [e["fingerprint"] for e in reg["models"].values()]
    assert all(f.startswith("sha256:") for f in fingerprints)
    assert len(set(fingerprints)) == len(fingerprints), "each trained model should fingerprint distinctly"


def test_build_is_deterministic_given_same_timestamp():
    reg_a = build(registered_at="2026-01-01T00:00:00+00:00")
    reg_b = build(registered_at="2026-01-01T00:00:00+00:00")
    assert reg_a == reg_b


def test_build_records_real_metrics_not_placeholders():
    reg = build(registered_at="2026-01-01T00:00:00+00:00")
    for name, entry in reg["models"].items():
        assert entry["metrics"].get("test_auc") is not None, f"{name} missing test_auc"
        assert 0.5 < entry["metrics"]["test_auc"] <= 1.0
