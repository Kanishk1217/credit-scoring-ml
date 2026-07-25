"""API tests: auth, validation, scoring, and batch. Run: uv run pytest -q"""
import os

# configure the app BEFORE importing it (env vars beat .env)
os.environ["CREDIT_API_KEYS"] = "testkey123"
os.environ["CREDIT_RATE_LIMIT"] = "10000/minute"   # don't trip the limiter during tests

import pytest
from fastapi.testclient import TestClient

from api.app import app

KEY = {"X-API-Key": "testkey123"}
APPLICANT = {
    "limit_bal": 120000, "sex": 2, "education": 2, "marriage": 1, "age": 30,
    "bill_amt": [80000, 82000, 85000, 88000, 90000, 92000],
    "pay_amt": [3000, 3000, 2500, 2000, 1500, 1000],
    "pay_status": [0, 0, -1, -1, 2, 2],
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:      # context manager runs lifespan -> loads models
        yield c


def test_health(client):
    r = client.get("/")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_requires_key(client):
    assert client.post("/predict", json=APPLICANT).status_code == 401


def test_predict_rejects_wrong_key(client):
    assert client.post("/predict", json=APPLICANT, headers={"X-API-Key": "nope"}).status_code == 401


def test_predict_ok(client):
    r = client.post("/predict", json=APPLICANT, headers=KEY)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["probability_of_default"] <= 1.0
    assert body["recommendation"] in {"approve", "review", "decline"}


def test_predict_validation(client):
    bad = {**APPLICANT, "age": 5, "pay_status": [0, 0, 0, 0, 0, 99]}
    assert client.post("/predict", json=bad, headers=KEY).status_code == 422


def test_batch(client):
    r = client.post("/predict/batch", json={"applicants": [APPLICANT, APPLICANT]}, headers=KEY)
    assert r.status_code == 200 and r.json()["count"] == 2


def test_security_headers(client):
    r = client.get("/")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert "x-request-id" in r.headers
