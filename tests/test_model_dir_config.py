"""Verifies the CREDIT_MODEL_DIR toggle actually swaps which trained model api/app.py serves.

Run in a subprocess (not imported in-process) because api.config.Settings is instantiated once
at import time -- other test modules in this suite already import api.app with the default
"models" directory, so a same-process env-var change wouldn't take effect.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_PROBE = """
import json
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
from fastapi.testclient import TestClient
from api.app import app
with TestClient(app) as client:
    print("PROBE_RESULT:" + json.dumps(client.get("/").json()))
"""


def _health_via_subprocess(model_dir: str) -> dict:
    env = {**os.environ, "CREDIT_MODEL_DIR": model_dir, "CREDIT_API_KEYS": "testkey123"}
    result = subprocess.run(
        [sys.executable, "-c", _PROBE], cwd=ROOT, env=env,
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    line = next(ln for ln in result.stdout.splitlines() if ln.startswith("PROBE_RESULT:"))
    return json.loads(line.removeprefix("PROBE_RESULT:"))


def test_default_model_dir_is_synthetic():
    body = _health_via_subprocess("models")
    assert body["model_dir"] == "models"
    assert body["test_auc"] > 0.85


def test_real_model_dir_serves_real_data_model():
    body = _health_via_subprocess("models_real")
    assert body["model_dir"] == "models_real"
    assert "REAL Home Credit" in body["data_source"]
    assert body["test_auc"] < 0.7   # honest, lower AUC of the real 7-feature model, not synthetic


_DOCS_PROBE = """
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
from fastapi.testclient import TestClient
from api.app import app
with TestClient(app) as client:
    print("PROBE_RESULT:" + str(client.get("/docs").status_code))
"""


def _docs_status_via_subprocess(environment: str) -> int:
    env = {**os.environ, "CREDIT_ENVIRONMENT": environment, "CREDIT_API_KEYS": "testkey123"}
    result = subprocess.run(
        [sys.executable, "-c", _DOCS_PROBE], cwd=ROOT, env=env,
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    line = next(ln for ln in result.stdout.splitlines() if ln.startswith("PROBE_RESULT:"))
    return int(line.removeprefix("PROBE_RESULT:"))


def test_docs_disabled_in_production_even_if_enable_docs_left_default():
    """Defense in depth: CREDIT_ENABLE_DOCS defaults true, but production must never expose
    interactive docs even if that flag is forgotten in the deploy config."""
    assert _docs_status_via_subprocess("production") == 404


def test_docs_enabled_in_development():
    assert _docs_status_via_subprocess("development") == 200
