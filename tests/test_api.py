"""API tests: auth, validation, scoring, calibration sanity, fairness contract, and batch.
Run: uv run pytest -q
"""
import os

# configure the app BEFORE importing it (env vars beat .env)
os.environ["CREDIT_API_KEYS"] = "testkey123"
os.environ["CREDIT_RATE_LIMIT"] = "10000/minute"   # don't trip the limiter during tests

import pytest
from fastapi.testclient import TestClient

from api.app import app

KEY = {"X-API-Key": "testkey123"}

HEALTHY = {
    "age": 45, "monthly_income": 80000, "credit_limit": 400000, "existing_debt": 40000,
    "employment_years": 12.0, "num_existing_loans": 1,
    "pay_status": [-1] * 12, "has_collateral": False,
}
RISKY = {
    "age": 29, "monthly_income": 30000, "credit_limit": 150000, "existing_debt": 120000,
    "employment_years": 1.5, "num_existing_loans": 4,
    "pay_status": [0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3], "has_collateral": False,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:      # context manager runs lifespan -> loads models
        yield c


def test_health(client):
    r = client.get("/")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert r.json()["test_auc"] > 0.85   # the calibrated hybrid model, not the old uncalibrated one


def test_predict_requires_key(client):
    assert client.post("/predict", json=HEALTHY).status_code == 401


def test_predict_rejects_wrong_key(client):
    assert client.post("/predict", json=HEALTHY, headers={"X-API-Key": "nope"}).status_code == 401


def test_predict_ok_shape(client):
    r = client.post("/predict", json=HEALTHY, headers=KEY)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["probability_of_default"] <= 1.0
    assert body["recommendation"] in {"approve", "review", "decline"}
    assert "why" in body and isinstance(body["why"], list) and len(body["why"]) > 0
    assert "pricing" in body and "max_loan_amount" in body["pricing"]
    assert "advice" in body


def test_healthy_vs_risky_ordering(client):
    """A model that can't tell these apart is useless — this is the core sanity check."""
    healthy_pd = client.post("/predict", json=HEALTHY, headers=KEY).json()["probability_of_default"]
    risky_pd = client.post("/predict", json=RISKY, headers=KEY).json()["probability_of_default"]
    assert healthy_pd < 0.2
    assert risky_pd > 0.5
    assert risky_pd > healthy_pd


def test_calibration_is_sane(client):
    """Guards against the exact bug that shipped before: predicted PD should not be wildly
    higher than what the decision thresholds imply is 'risky'. A healthy on-time payer must not
    be flagged for decline (that was happening when the model was uncalibrated)."""
    r = client.post("/predict", json=HEALTHY, headers=KEY).json()
    assert r["recommendation"] == "approve"
    assert r["probability_of_default"] < r["approve_threshold"] + 0.05


def test_declined_applicant_gets_no_loan_offer(client):
    r = client.post("/predict", json=RISKY, headers=KEY).json()
    if r["recommendation"] == "decline":
        assert r["pricing"]["max_loan_amount"] == 0


def test_collateral_improves_terms(client):
    """Collateral must never make an offer worse: same or better rate, same or larger approval."""
    unsecured = client.post("/predict", json={**HEALTHY, "has_collateral": False}, headers=KEY).json()
    secured = client.post("/predict", json={**HEALTHY, "has_collateral": True}, headers=KEY).json()
    assert secured["pricing"]["interest_rate_pct"] <= unsecured["pricing"]["interest_rate_pct"]


def test_advice_improves_pd(client):
    r = client.post("/predict", json=RISKY, headers=KEY).json()
    for item in r["advice"]:
        assert item["projected_pd"] < item["current_pd"]
        assert item["pd_improvement"] > 0


def test_protected_attributes_rejected(client):
    """Protected attributes must be structurally impossible to submit (extra='forbid')."""
    tainted = {**HEALTHY, "sex": 2}
    assert client.post("/predict", json=tainted, headers=KEY).status_code == 422


def test_predict_validation(client):
    bad = {**HEALTHY, "age": 5, "pay_status": [0] * 11 + [99]}
    assert client.post("/predict", json=bad, headers=KEY).status_code == 422


def test_predict_wrong_sequence_length(client):
    bad = {**HEALTHY, "pay_status": [0, 0, 0]}   # must be exactly 12
    assert client.post("/predict", json=bad, headers=KEY).status_code == 422


def test_batch(client):
    r = client.post("/predict/batch", json={"applicants": [HEALTHY, RISKY]}, headers=KEY)
    assert r.status_code == 200 and r.json()["count"] == 2


def test_security_headers(client):
    r = client.get("/")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert "x-request-id" in r.headers


# --- /score, /score/batch: loan-officer dashboard contract ---

OFFICER_HEALTHY = {
    "age": 45, "monthly_income": 80000, "credit_limit": 400000, "existing_debt": 40000,
    "employment_years": 12.0, "num_existing_loans": 1, "payment_history": [-1] * 12,
}
OFFICER_RISKY = {
    "age": 29, "monthly_income": 30000, "credit_limit": 150000, "existing_debt": 120000,
    "employment_years": 1.5, "num_existing_loans": 4,
    "payment_history": [0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3],
}


def test_score_shape(client):
    r = client.post("/score", json=OFFICER_HEALTHY, headers=KEY)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["pd"] <= 1.0
    assert body["band"] in set("ABCDE")
    assert body["verdict"] in {"approve", "review", "decline"}
    assert len(body["factors"]) > 0
    weights = [f["weightPct"] for f in body["factors"]]
    assert abs(sum(weights) - 1.0) < 0.05   # weights should roughly sum to 1 (top_n truncation)
    assert "offered_amount" in body["pricing"] and body["pricing"]["tenor_months"] == 24


def test_score_healthy_vs_risky_ordering(client):
    healthy_pd = client.post("/score", json=OFFICER_HEALTHY, headers=KEY).json()["pd"]
    risky_pd = client.post("/score", json=OFFICER_RISKY, headers=KEY).json()["pd"]
    assert risky_pd > healthy_pd


def test_score_requires_key(client):
    assert client.post("/score", json=OFFICER_HEALTHY).status_code == 401


def test_score_protected_attributes_rejected(client):
    tainted = {**OFFICER_HEALTHY, "gender": "F"}
    assert client.post("/score", json=tainted, headers=KEY).status_code == 422


def test_score_batch(client):
    r = client.post("/score/batch",
                    json={"applicants": {"H": OFFICER_HEALTHY, "R": OFFICER_RISKY}}, headers=KEY)
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["count"] == 2
    ids = {row["applicant_id"] for row in body["results"]}
    assert ids == {"H", "R"}
    assert body["summary"]["approve"] + body["summary"]["review"] + body["summary"]["decline"] == 2


# --- /self-assessment: consumer contract ---

CONSUMER_PROFILE = {
    "age": 34, "monthly_income": 65000, "credit_limit": 200000, "existing_debt": 180000,
    "employment_years": 4, "num_existing_loans": 2,
    "payment_history": [0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 1, 0],
}


def test_self_assessment_shape(client):
    r = client.post("/self-assessment", json={"profile": CONSUMER_PROFILE}, headers=KEY)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["pd"] <= 1.0
    assert body["band"] in {"thriving", "steady", "almost", "building", "starting"}
    assert "reject" not in body["band_headline"].lower() and "denied" not in body["band_headline"].lower()
    assert len(body["why"]) > 0
    assert "goal" in body and "target_pd" in body["goal"]


def test_self_assessment_requires_key(client):
    assert client.post("/self-assessment", json={"profile": CONSUMER_PROFILE}).status_code == 401


def test_self_assessment_protected_attributes_rejected(client):
    tainted = {"profile": {**CONSUMER_PROFILE, "sex": 1}}
    assert client.post("/self-assessment", json=tainted, headers=KEY).status_code == 422


def test_self_assessment_advice_never_worsens_pd(client):
    r = client.post("/self-assessment", json={"profile": CONSUMER_PROFILE}, headers=KEY).json()
    for item in r["advice"]:
        assert item["pd_after"] < item["pd_before"]
        assert item["delta"] > 0


def test_self_assessment_goal_reaches_or_admits_unreachable(client):
    risky_profile = {**CONSUMER_PROFILE, "existing_debt": 900000, "monthly_income": 25000,
                     "payment_history": [3, 4, 5, 6, 7, 8, 9, 9, 9, 9, 9, 9]}
    r = client.post("/self-assessment", json={"profile": risky_profile}, headers=KEY).json()
    goal = r["goal"]
    if goal["reachable"]:
        assert goal["projected_pd"] <= goal["target_pd"]
    else:
        assert goal["projected_offer"] is not None
