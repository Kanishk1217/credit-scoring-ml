"""Hardened FastAPI service for the hybrid credit-scoring model.

Production concerns covered here:
- API-key auth (constant-time compare), per-key rate limiting
- strict input validation, locked-down CORS, security headers
- request-ID + latency logging, and an audit log of every credit decision
- single and batch scoring
- non-leaking error handling; secrets from env only

Run locally:
    uv run uvicorn api.app:app --port 8077
    # docs: http://localhost:8077/docs   (send header  X-API-Key: <your key>)
"""
# --- OpenMP fix: MUST be before importing xgboost/torch, and xgboost MUST come before torch ---
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import logging
import secrets
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
import numpy as np
import torch
import xgboost  # noqa: F401  (import order matters — before torch)
from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from api.config import settings
from hybrid_model import Hybrid

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("credit-api")
audit = logging.getLogger("credit-audit")   # every decision is logged here (regulatory trail)

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["cfg"] = json.loads((ROOT / "models" / "hybrid_config.json").read_text())
    STATE["xgb"] = joblib.load(ROOT / "models" / "hybrid_xgb.joblib")
    net = Hybrid(STATE["cfg"]["lstm_hidden"])
    net.load_state_dict(torch.load(ROOT / "models" / "hybrid_fusion.pt"))
    net.eval()
    STATE["net"] = net
    if not settings.api_key_set:
        logger.warning("No API keys configured (CREDIT_API_KEYS). Scoring endpoints will return 503.")
    logger.info("models loaded; %d API key(s) configured", len(settings.api_key_set))
    yield
    STATE.clear()


def rate_key(request: Request) -> str:
    return request.headers.get("X-API-Key") or get_remote_address(request)


limiter = Limiter(key_func=rate_key, default_limits=[settings.rate_limit])

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs" if settings.enable_docs else None,
    redoc_url=None,
)
app.state.limiter = limiter

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,   # exact list, never "*"
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)


@app.middleware("http")
async def observability(request: Request, call_next):
    rid = uuid.uuid4().hex[:12]
    start = time.perf_counter()
    response = await call_next(request)
    dur_ms = (time.perf_counter() - start) * 1000
    logger.info("rid=%s %s %s -> %s %.1fms", rid, request.method, request.url.path,
                response.status_code, dur_ms)
    response.headers["X-Request-ID"] = rid
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    return response


@app.exception_handler(RateLimitExceeded)
async def ratelimit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"detail": "rate limit exceeded"})


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception):
    logger.exception("unhandled error")
    return JSONResponse(status_code=500, content={"detail": "internal server error"})


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: Annotated[str | None, Security(api_key_header)]) -> str:
    valid = settings.api_key_set
    if not valid:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "API not configured")
    if not key or not any(secrets.compare_digest(key, k) for k in valid):   # constant-time
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing API key")
    return key


ApiKey = Annotated[str, Depends(require_api_key)]


class Applicant(BaseModel):
    limit_bal: float = Field(..., ge=0, le=1e9, examples=[200000])
    sex: int = Field(..., ge=1, le=2, examples=[2])
    education: int = Field(..., ge=0, le=6, examples=[2])
    marriage: int = Field(..., ge=0, le=3, examples=[1])
    age: int = Field(..., ge=18, le=120, examples=[35])
    bill_amt: list[float] = Field(..., min_length=6, max_length=6)
    pay_amt: list[float] = Field(..., min_length=6, max_length=6)
    pay_status: list[int] = Field(..., min_length=6, max_length=6,
                                  description="Apr->Sep; <=0 on time, 1..9 months late")

    @field_validator("pay_status")
    @classmethod
    def _status_range(cls, v: list[int]) -> list[int]:
        if any(s < -2 or s > 9 for s in v):
            raise ValueError("pay_status values must be between -2 and 9")
        return v

    @field_validator("bill_amt", "pay_amt")
    @classmethod
    def _amount_sane(cls, v: list[float]) -> list[float]:
        if any(abs(x) > 1e9 for x in v):
            raise ValueError("amounts out of range")
        return v


class PredictionResponse(BaseModel):
    probability_of_default: float
    recommendation: str
    model_version: str


class BatchRequest(BaseModel):
    applicants: list[Applicant] = Field(..., min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    results: list[PredictionResponse]
    count: int


def _score(a: Applicant) -> PredictionResponse:
    cfg, xgb, net = STATE["cfg"], STATE["xgb"], STATE["net"]
    static = np.array([[a.limit_bal, a.sex, a.education, a.marriage, a.age,
                        *a.bill_amt, *a.pay_amt]], dtype="float32")
    score = xgb.predict_proba(static)[:, 1]
    seq = (np.array([a.pay_status], dtype="float32") - cfg["seq_mean"]) / cfg["seq_std"]
    with torch.no_grad():
        pd_ = torch.sigmoid(
            net(torch.tensor(seq).unsqueeze(-1), torch.tensor(score, dtype=torch.float32).unsqueeze(1))
        ).item()
    rec = "decline" if pd_ >= 0.5 else "review" if pd_ >= 0.2 else "approve"
    audit.info("decision pd=%.4f recommendation=%s model=%s", pd_, rec, settings.app_version)
    return PredictionResponse(probability_of_default=round(pd_, 4), recommendation=rec,
                              model_version=settings.app_version)


@app.get("/", tags=["health"])
def health():
    return {"status": "ok", "model": "hybrid XGBoost + LSTM",
            "version": settings.app_version, "models_loaded": bool(STATE)}


@app.post("/predict", response_model=PredictionResponse, tags=["scoring"])
@limiter.limit(settings.rate_limit)
def predict(request: Request, applicant: Applicant, _key: ApiKey):
    return _score(applicant)


@app.post("/predict/batch", response_model=BatchResponse, tags=["scoring"])
@limiter.limit(settings.rate_limit)
def predict_batch(request: Request, batch: BatchRequest, _key: ApiKey):
    results = [_score(a) for a in batch.applicants]
    return BatchResponse(results=results, count=len(results))
