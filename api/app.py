"""Hardened FastAPI service for the hybrid credit-scoring model.

Model: XGBoost (static features) + LSTM (12-month payment sequence), fused, then ISOTONIC
CALIBRATED so the output is a real probability of default, not just a ranking. Decisions use
COST-BASED thresholds (fit on held-out data) instead of an arbitrary 0.5 cutoff. No protected
attributes (sex/marriage/education) are used for scoring.

Production concerns covered here:
- API-key auth (constant-time compare), per-key rate limiting
- strict input validation, locked-down CORS, security headers
- request-ID + latency logging, and an audit log of every credit decision
- single and batch scoring, explainability (why), pricing, and improvement advice
- non-leaking error handling; secrets from env only

Run locally:
    uv run uvicorn api.app:app --port 8077
    # docs: http://localhost:8077/docs   (send header  X-API-Key: <your key>)
"""
# limit XGBoost threads (kind to small free instances); no torch here, so no OpenMP clash
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import logging
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
import xgboost  # noqa: F401  (needed to unpickle the saved XGBoost model)
from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api import scoring
from api.config import settings
from api.infer_numpy import NumpyHybrid

ROOT = Path(__file__).resolve().parents[1]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("credit-api")
audit = logging.getLogger("credit-audit")   # every decision is logged here (regulatory trail)

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = ROOT / settings.model_dir
    STATE["cfg"] = json.loads((model_dir / "hybrid_config.json").read_text())
    STATE["xgb"] = joblib.load(model_dir / "hybrid_xgb.joblib")
    STATE["net"] = NumpyHybrid(model_dir / "hybrid_fusion.npz", STATE["cfg"]["lstm_hidden"])
    if not settings.api_key_set:
        logger.warning("No API keys configured (CREDIT_API_KEYS). Scoring endpoints will return 503.")
    logger.info("models loaded from %s; %d API key(s) configured; test AUC=%s",
                settings.model_dir, len(settings.api_key_set),
                STATE["cfg"].get("metrics", {}).get("test_auc"))
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
    """No sex/marriage/education fields — those are protected attributes and are never scored.
    extra="forbid" so a client literally cannot smuggle a protected attribute into the request."""
    model_config = {"extra": "forbid"}

    age: int = Field(..., ge=18, le=100, examples=[34])
    monthly_income: float = Field(..., gt=0, le=1e8, examples=[45000])
    credit_limit: float = Field(..., ge=0, le=1e9, examples=[300000])
    existing_debt: float = Field(..., ge=0, le=1e9, examples=[80000])
    employment_years: float = Field(..., ge=0, le=60, examples=[5.0])
    num_existing_loans: int = Field(..., ge=0, le=50, examples=[2])
    pay_status: list[int] = Field(..., min_length=12, max_length=12,
                                  description="12 months, oldest to newest; <=0 on time, 1..9 months late")
    has_collateral: bool = Field(False, description="secured loan — improves rate and approval cutoff")
    requested_amount: float | None = Field(None, ge=0, le=1e9)

    @field_validator("pay_status")
    @classmethod
    def _status_range(cls, v: list[int]) -> list[int]:
        if any(s < -2 or s > 9 for s in v):
            raise ValueError("pay_status values must be between -2 and 9")
        return v


class Factor(BaseModel):
    factor: str
    impact: float
    direction: str


class Advice(BaseModel):
    scenario: str
    current_pd: float
    projected_pd: float
    pd_improvement: float


class Pricing(BaseModel):
    max_loan_amount: int
    interest_rate_pct: float
    tenure_months: int
    monthly_emi: int
    note: str


class AssessmentResponse(BaseModel):
    probability_of_default: float
    recommendation: str
    approve_threshold: float
    decline_threshold: float
    why: list[Factor]
    pricing: Pricing
    advice: list[Advice]
    model_version: str


class BatchRequest(BaseModel):
    applicants: list[Applicant] = Field(..., min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    results: list[AssessmentResponse]
    count: int


def _score(a: Applicant) -> AssessmentResponse:
    cfg, xgb, net = STATE["cfg"], STATE["xgb"], STATE["net"]
    static_cols = cfg["static_cols"]

    static_row = scoring.build_static_row(static_cols, a.age, a.monthly_income, a.credit_limit,
                                          a.existing_debt, a.employment_years, a.num_existing_loans)
    xgb_score = float(xgb.predict_proba(static_row)[0, 1])
    seq_std = [(float(v) - cfg["seq_mean"]) / cfg["seq_std"] for v in a.pay_status]
    raw_pd = net.predict(seq_std, xgb_score)
    pd_ = scoring.calibrate(cfg, raw_pd)

    decision, approve_t, decline_t = scoring.decide(cfg, pd_, a.has_collateral)

    why = scoring.explain(cfg, xgb, static_cols, static_row, xgb_score, net, seq_std)

    pricing = scoring.price_loan(pd_, a.monthly_income, a.existing_debt, a.credit_limit,
                                 decision, a.has_collateral, a.requested_amount)

    advice = scoring.advice(cfg, xgb, static_cols, a.age, a.monthly_income, a.credit_limit,
                            a.existing_debt, a.employment_years, a.num_existing_loans,
                            [float(v) for v in a.pay_status], net, pd_)

    audit.info("decision pd=%.4f recommendation=%s collateral=%s model=%s",
              pd_, decision, a.has_collateral, settings.app_version)

    return AssessmentResponse(
        probability_of_default=round(pd_, 4),
        recommendation=decision,
        approve_threshold=round(approve_t, 4),
        decline_threshold=round(decline_t, 4),
        why=[Factor(**f) for f in why],
        pricing=Pricing(**pricing),
        advice=[Advice(**adv) for adv in advice],
        model_version=settings.app_version,
    )


@app.get("/", tags=["health"])
def health():
    return {"status": "ok", "model": "hybrid XGBoost + LSTM (calibrated)",
            "model_dir": settings.model_dir,
            "data_source": STATE.get("cfg", {}).get("data_source"),
            "version": settings.app_version, "models_loaded": bool(STATE),
            "test_auc": STATE.get("cfg", {}).get("metrics", {}).get("test_auc")}


@app.post("/predict", response_model=AssessmentResponse, tags=["scoring"])
@limiter.limit(settings.rate_limit)
def predict(request: Request, applicant: Applicant, _key: ApiKey):
    return _score(applicant)


@app.post("/predict/batch", response_model=BatchResponse, tags=["scoring"])
@limiter.limit(settings.rate_limit)
def predict_batch(request: Request, batch: BatchRequest, _key: ApiKey):
    results = [_score(a) for a in batch.applicants]
    return BatchResponse(results=results, count=len(results))
