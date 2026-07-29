"""Unit tests for the scoring business logic (calibration, decisions, pricing, explainability),
independent of the HTTP layer. Run: uv run pytest -q
"""
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import joblib
import numpy as np
import pytest

from api import scoring
from api.infer_numpy import NumpyHybrid

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def loaded():
    cfg = json.loads((ROOT / "models" / "hybrid_config.json").read_text())
    xgb = joblib.load(ROOT / "models" / "hybrid_xgb.joblib")
    net = NumpyHybrid(ROOT / "models" / "hybrid_fusion.npz", cfg["lstm_hidden"])
    return cfg, xgb, net


def test_calibration_reduces_systematic_bias(loaded):
    """Regression test for the exact bug that shipped: uncalibrated PD was ~2x the true rate.
    On a fresh synthetic sample (different seed than training), calibrated mean PD must track
    the actual default rate much more closely than the raw score does."""
    from src.synth_data import generate
    df = generate(20_000, seed=999)   # different seed = genuinely held-out data
    cfg, xgb, net = loaded
    static = df[cfg["static_cols"]].values.astype("float32")
    xgb_scores = xgb.predict_proba(static)[:, 1]
    seq_cols = [f"pay_{m}" for m in range(cfg["seq_len"])]
    seq = (df[seq_cols].values.astype("float32") - cfg["seq_mean"]) / cfg["seq_std"]

    raw = np.array([net.predict(seq[i], xgb_scores[i]) for i in range(len(df))])
    cal = np.array([scoring.calibrate(cfg, p) for p in raw])
    actual_rate = df["default"].mean()

    raw_error = abs(raw.mean() - actual_rate)
    cal_error = abs(cal.mean() - actual_rate)
    assert cal_error < raw_error, "calibration should shrink the gap to the true default rate"
    assert cal_error < 0.03, f"calibrated mean PD {cal.mean():.3f} too far from actual {actual_rate:.3f}"


def test_decide_threshold_ordering(loaded):
    cfg, _, _ = loaded
    assert cfg["thresholds"]["approve"] < cfg["thresholds"]["decline"]
    below = cfg["thresholds"]["approve"] / 2
    mid = (cfg["thresholds"]["approve"] + cfg["thresholds"]["decline"]) / 2
    above = min(0.99, cfg["thresholds"]["decline"] * 1.5)
    assert scoring.decide(cfg, below)[0] == "approve"
    assert scoring.decide(cfg, mid)[0] == "review"
    assert scoring.decide(cfg, above)[0] == "decline"


def test_collateral_widens_decline_threshold(loaded):
    cfg, _, _ = loaded
    _, _, decline_unsecured = scoring.decide(cfg, 0.5, has_collateral=False)
    _, _, decline_secured = scoring.decide(cfg, 0.5, has_collateral=True)
    assert decline_secured > decline_unsecured


def test_pricing_rate_increases_with_risk():
    low = scoring.price_loan(0.02, 60000, 20000, 300000, "approve")
    high = scoring.price_loan(0.30, 60000, 20000, 300000, "review")
    assert high["interest_rate_pct"] > low["interest_rate_pct"]
    assert scoring.RATE_FLOOR_PCT <= low["interest_rate_pct"] <= scoring.RATE_CAP_PCT
    assert scoring.RATE_FLOOR_PCT <= high["interest_rate_pct"] <= scoring.RATE_CAP_PCT


def test_pricing_declined_gets_zero_offer():
    r = scoring.price_loan(0.5, 60000, 20000, 300000, "decline")
    assert r["max_loan_amount"] == 0 and r["monthly_emi"] == 0


def test_pricing_collateral_never_hurts():
    unsecured = scoring.price_loan(0.15, 50000, 30000, 250000, "approve", has_collateral=False)
    secured = scoring.price_loan(0.15, 50000, 30000, 250000, "approve", has_collateral=True)
    assert secured["interest_rate_pct"] <= unsecured["interest_rate_pct"]


def test_explain_returns_ranked_factors(loaded):
    cfg, xgb, net = loaded
    static_row = scoring.build_static_row(cfg["static_cols"], 29, 30000, 150000, 120000, 1.5, 4)
    xgb_score = float(xgb.predict_proba(static_row)[0, 1])
    seq_std = [(v - cfg["seq_mean"]) / cfg["seq_std"] for v in [0]*6 + [1, 1, 2, 2, 3, 3]]
    factors = scoring.explain(cfg, xgb, cfg["static_cols"], static_row, xgb_score, net, seq_std)
    assert len(factors) > 0
    impacts = [abs(f["impact"]) for f in factors]
    assert impacts == sorted(impacts, reverse=True), "factors must be ranked by impact magnitude"
    assert all(f["direction"] in {"increases_risk", "decreases_risk"} for f in factors)
    # a deteriorating recent payment history should be the (or a) top driver of risk
    assert any(f["factor"] == "Recent payment history" for f in factors)


def test_no_protected_attributes_in_static_cols(loaded):
    """The single most important invariant from the fairness fix."""
    cfg, _, _ = loaded
    forbidden = {"sex", "gender", "marriage", "education", "SEX", "EDUCATION", "MARRIAGE"}
    assert not (forbidden & set(cfg["static_cols"]))
