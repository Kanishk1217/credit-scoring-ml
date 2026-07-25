"""FastAPI service for the hybrid credit-scoring model.

Loads the saved model files once at startup, then serves probability-of-default predictions
for new applicants over HTTP.

Run locally:
    uv run uvicorn api.app:app --reload --port 8000
Then open http://localhost:8000/docs for an interactive form.
"""
# --- OpenMP fix: MUST be before importing xgboost/torch, and xgboost MUST come before torch ---
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
from pathlib import Path

import joblib
import numpy as np
import xgboost          # noqa: F401  (import order matters — before torch)
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hybrid_model import Hybrid

# ---- load the trained model FILES once, at startup ----
CFG = json.loads((ROOT / "models" / "hybrid_config.json").read_text())
XGB = joblib.load(ROOT / "models" / "hybrid_xgb.joblib")
NET = Hybrid(CFG["lstm_hidden"])
NET.load_state_dict(torch.load(ROOT / "models" / "hybrid_fusion.pt"))
NET.eval()

app = FastAPI(title="Credit Scoring — Hybrid Model", version="1.0")


class Applicant(BaseModel):
    """One loan applicant. Static facts + a 6-month payment-status sequence (Apr..Sep)."""
    limit_bal: float = Field(..., example=200000)
    sex: int = Field(..., example=2)
    education: int = Field(..., example=2)
    marriage: int = Field(..., example=1)
    age: int = Field(..., example=35)
    bill_amt: list[float] = Field(..., min_length=6, max_length=6, example=[50000, 48000, 47000, 46000, 45000, 44000])
    pay_amt: list[float] = Field(..., min_length=6, max_length=6, example=[2000, 2000, 2000, 2000, 2000, 2000])
    pay_status: list[int] = Field(..., min_length=6, max_length=6,
                                  description="payment status Apr->Sep; <=0 on time, 1..9 months late",
                                  example=[0, 0, -1, -1, 2, 2])


def predict_pd(a: Applicant) -> float:
    """Raw applicant -> preprocess -> both models -> probability of default."""
    # static branch: assemble features in the exact training order, then XGBoost score
    static = np.array([[a.limit_bal, a.sex, a.education, a.marriage, a.age,
                        *a.bill_amt, *a.pay_amt]], dtype="float32")
    score = XGB.predict_proba(static)[:, 1]

    # temporal branch: standardize the sequence the same way training did, shape (1, 6, 1)
    seq = (np.array([a.pay_status], dtype="float32") - CFG["seq_mean"]) / CFG["seq_std"]
    seq_t = torch.tensor(seq).unsqueeze(-1)
    score_t = torch.tensor(score, dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        pd_ = torch.sigmoid(NET(seq_t, score_t)).item()
    return pd_


@app.get("/")
def health():
    return {"status": "ok", "model": "hybrid XGBoost + LSTM"}


@app.post("/predict")
def predict(applicant: Applicant):
    pd_ = predict_pd(applicant)
    # the business threshold is a lender decision, not a model output (Session 3 lesson)
    decision = "decline" if pd_ >= 0.5 else "review" if pd_ >= 0.2 else "approve"
    return {
        "probability_of_default": round(pd_, 4),
        "recommendation": decision,
        "threshold_note": "approve < 0.20 <= review < 0.50 <= decline (lender-set)",
    }
