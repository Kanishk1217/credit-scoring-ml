"""Model registry: catalogs every trained model directory (models/, models_real/,
models_real_rich/, ...) with its provenance -- data source, metrics, a content fingerprint of
the actual trained artifacts, and the git commit that produced it -- so whatever the API is
currently serving can always be traced back to how it was built and verified.

The fingerprint hashes the trained artifact files themselves (xgb + fusion weights), not the raw
training data -- a strong proxy for "this exact model came from this exact training run," since
any change to features, data, or hyperparameters changes the resulting weights.

Run:  uv run python src/build_model_registry.py
"""
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "model_registry.json"
MODEL_DIRS = ["models", "models_real", "models_real_rich", "models_lendingclub"]
# Two config-file names are supported: "hybrid_config.json" for the XGBoost+LSTM hybrid models,
# "model_config.json" for plain-tabular models with no sequence branch (e.g. models_lendingclub) --
# naming a non-hybrid model's config "hybrid_config.json" would mislead a future reader.
CONFIG_FILENAMES = ["hybrid_config.json", "model_config.json"]


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _find_config(model_dir: Path) -> Path | None:
    for name in CONFIG_FILENAMES:
        p = model_dir / name
        if p.exists():
            return p
    return None


def _fingerprint(model_dir: Path) -> str | None:
    """Hash whichever trained-artifact files (.joblib / .npz) actually exist in this model
    directory -- a hybrid model has an xgboost file plus a fusion-net file, a plain tabular model
    (no sequence branch) has only the xgboost file. Returns None only if neither type is present."""
    artifacts = sorted(model_dir.glob("*.joblib")) + sorted(model_dir.glob("*.npz"))
    if not artifacts:
        return None
    h = hashlib.sha256()
    for f in artifacts:
        h.update(f.read_bytes())
    return f"sha256:{h.hexdigest()[:16]}"


def build(registered_at: str | None = None) -> dict:
    """registered_at is injectable for tests; defaults to real wall-clock time."""
    registered_at = registered_at or datetime.now(UTC).isoformat(timespec="seconds")
    commit = _git_commit()
    registry: dict = {"generated_at": registered_at, "git_commit": commit, "models": {}}

    for name in MODEL_DIRS:
        model_dir = ROOT / name
        cfg_path = _find_config(model_dir)
        if cfg_path is None:
            continue
        cfg = json.loads(cfg_path.read_text())
        registry["models"][name] = {
            "data_source": cfg.get("data_source", "unknown"),
            "n_total": cfg.get("n_total"),
            "static_feature_count": len(cfg.get("static_cols", [])),
            "seq_len": cfg.get("seq_len"),
            "thresholds": cfg.get("thresholds"),
            "metrics": cfg.get("metrics", {}),
            "fairness_audit": {
                k: v.get("demographic_parity_gap") if isinstance(v, dict) else None
                for k, v in cfg.get("fairness_audit", {}).items()
            },
            "fingerprint": _fingerprint(model_dir),
            "registered_at": registered_at,
            "git_commit": commit,
        }
    return registry


if __name__ == "__main__":
    reg = build()
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2) + "\n")
    print(f"registered {len(reg['models'])} model(s) -> {REGISTRY_PATH}")
    for name, entry in reg["models"].items():
        print(f"  {name}: {entry['fingerprint']}  auc={entry['metrics'].get('test_auc')}  "
              f"data_source={entry['data_source'][:60]}")
