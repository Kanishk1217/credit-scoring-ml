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

from api import advice_engine, scoring
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

    registry_path = ROOT / "model_registry.json"
    if registry_path.exists():
        registry = json.loads(registry_path.read_text())
        STATE["registry_entry"] = registry.get("models", {}).get(settings.model_dir)
    else:
        STATE["registry_entry"] = None
        logger.warning("model_registry.json not found; run `uv run python src/build_model_registry.py`")

    if not settings.api_key_set:
        logger.warning("No API keys configured (CREDIT_API_KEYS). Scoring endpoints will return 503.")
    logger.info("models loaded from %s; %d API key(s) configured; test AUC=%s; fingerprint=%s",
                settings.model_dir, len(settings.api_key_set),
                STATE["cfg"].get("metrics", {}).get("test_auc"),
                (STATE["registry_entry"] or {}).get("fingerprint"))
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
    override_reason: str | None = None


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

    override_reason = scoring.check_hard_override(
        a.monthly_income, a.existing_debt, a.credit_limit, [float(v) for v in a.pay_status])
    if override_reason:
        decision = "decline"

    why = scoring.explain(cfg, xgb, static_cols, static_row, xgb_score, net, seq_std)

    pricing = scoring.price_loan(pd_, a.monthly_income, a.existing_debt, a.credit_limit,
                                 decision, a.has_collateral, a.requested_amount)

    advice = scoring.advice(cfg, xgb, static_cols, a.age, a.monthly_income, a.credit_limit,
                            a.existing_debt, a.employment_years, a.num_existing_loans,
                            [float(v) for v in a.pay_status], net, pd_)

    audit.info("decision pd=%.4f recommendation=%s collateral=%s override=%s model=%s",
              pd_, decision, a.has_collateral, override_reason, settings.app_version)

    return AssessmentResponse(
        probability_of_default=round(pd_, 4),
        recommendation=decision,
        approve_threshold=round(approve_t, 4),
        decline_threshold=round(decline_t, 4),
        why=[Factor(**f) for f in why],
        pricing=Pricing(**pricing),
        advice=[Advice(**adv) for adv in advice],
        model_version=settings.app_version,
        override_reason=scoring.HARD_OVERRIDE_REASONS.get(override_reason) if override_reason else None,
    )


@app.get("/", tags=["health"])
def health():
    registry_entry = STATE.get("registry_entry")
    return {"status": "ok", "model": "hybrid XGBoost + LSTM (calibrated)",
            "model_dir": settings.model_dir,
            "data_source": STATE.get("cfg", {}).get("data_source"),
            "version": settings.app_version, "models_loaded": bool(STATE),
            "test_auc": STATE.get("cfg", {}).get("metrics", {}).get("test_auc"),
            "fingerprint": (registry_entry or {}).get("fingerprint"),
            "registered_at": (registry_entry or {}).get("registered_at"),
            "trained_at_commit": (registry_entry or {}).get("git_commit")}


@app.get("/model-registry", tags=["health"])
def model_registry(_key: ApiKey):
    """Full provenance catalog of every trained model directory -- data source, metrics,
    fairness audit, and a content fingerprint of the actual trained artifacts. Requires a key
    (unlike `/`) since it exposes fairness-audit gaps and thresholds for internal-only models."""
    registry_path = ROOT / "model_registry.json"
    if not registry_path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "model_registry.json not found")
    return json.loads(registry_path.read_text())


@app.post("/predict", response_model=AssessmentResponse, tags=["scoring"])
@limiter.limit(settings.rate_limit)
def predict(request: Request, applicant: Applicant, _key: ApiKey):
    return _score(applicant)


@app.post("/predict/batch", response_model=BatchResponse, tags=["scoring"])
@limiter.limit(settings.rate_limit)
def predict_batch(request: Request, batch: BatchRequest, _key: ApiKey):
    results = [_score(a) for a in batch.applicants]
    return BatchResponse(results=results, count=len(results))


# --- /score, /score/batch: loan-officer dashboard contract (see docs/specs/dashboard-ux.md) ---
# Same underlying model and scoring.py decision logic as /predict; only the request/response
# shape differs, matching the dashboard's ApplicantInput/ScoreResult types field-for-field.

OFFICER_TENOR_MONTHS = 24


class OfficerApplicantInput(BaseModel):
    model_config = {"extra": "forbid"}

    age: int = Field(..., ge=18, le=100)
    monthly_income: float = Field(..., gt=0, le=1e8)
    credit_limit: float = Field(..., ge=0, le=1e9)
    existing_debt: float = Field(..., ge=0, le=1e9)
    employment_years: float = Field(..., ge=0, le=60)
    num_existing_loans: int = Field(..., ge=0, le=50)
    payment_history: list[int] = Field(..., min_length=12, max_length=12)

    @field_validator("payment_history")
    @classmethod
    def _status_range(cls, v: list[int]) -> list[int]:
        if any(s < -2 or s > 9 for s in v):
            raise ValueError("payment_history values must be between -2 and 9")
        return v


class OfficerFactor(BaseModel):
    feature: str
    label: str
    value: str
    contribution: float
    direction: str
    weightPct: float


class OfficerPricing(BaseModel):
    offered_amount: int
    apr: float
    emi: int
    tenor_months: int


class OfficerScoreResult(BaseModel):
    pd: float
    band: str
    verdict: str
    factors: list[OfficerFactor]
    pricing: OfficerPricing
    override_reason: str | None = None


class OfficerBatchRow(OfficerScoreResult):
    applicant_id: str


class OfficerBatchSummary(BaseModel):
    count: int
    approve: int
    review: int
    decline: int
    avg_pd: float
    median_pd: float
    band_dist: dict[str, int]
    total_offered_exposure: float


class OfficerBatchRequest(BaseModel):
    applicants: dict[str, OfficerApplicantInput] = Field(..., min_length=1, max_length=5000)


class OfficerBatchResponse(BaseModel):
    results: list[OfficerBatchRow]
    summary: OfficerBatchSummary


def _format_factor_value(col: str, a: OfficerApplicantInput) -> str:
    if col == "age":
        return f"{a.age}"
    if col == "monthly_income":
        return f"₹{a.monthly_income:,.0f}"
    if col == "credit_limit":
        return f"₹{a.credit_limit:,.0f}"
    if col == "existing_debt":
        return f"₹{a.existing_debt:,.0f}"
    if col == "debt_to_income":
        dti = a.existing_debt / (a.monthly_income * 12 + 1)
        return f"{dti:.2f}"
    if col == "employment_years":
        return f"{a.employment_years:g} yrs"
    if col == "num_existing_loans":
        return f"{a.num_existing_loans}"
    if col == "payment_history":
        late = sum(1 for v in a.payment_history if v > 0)
        return f"{late} late months" if late != 1 else "1 late month"
    return ""


def _score_officer(a: OfficerApplicantInput) -> OfficerScoreResult:
    cfg, xgb, net = STATE["cfg"], STATE["xgb"], STATE["net"]
    static_cols = cfg["static_cols"]

    static_row = scoring.build_static_row(static_cols, a.age, a.monthly_income, a.credit_limit,
                                          a.existing_debt, a.employment_years, a.num_existing_loans)
    xgb_score = float(xgb.predict_proba(static_row)[0, 1])
    seq_std = [(float(v) - cfg["seq_mean"]) / cfg["seq_std"] for v in a.payment_history]
    raw_pd = net.predict(seq_std, xgb_score)
    pd_ = scoring.calibrate(cfg, raw_pd)

    decision, _approve_t, _decline_t = scoring.decide(cfg, pd_)

    override_reason = scoring.check_hard_override(
        a.monthly_income, a.existing_debt, a.credit_limit, [float(v) for v in a.payment_history])
    if override_reason:
        decision = "decline"

    raw_factors = scoring.explain(cfg, xgb, static_cols, static_row, xgb_score, net, seq_std)
    total = sum(abs(f["impact"]) for f in raw_factors) or 1.0
    factors = [
        OfficerFactor(
            feature=f["col"],
            label=f["factor"],
            value=_format_factor_value(f["col"], a),
            contribution=f["impact"],
            direction="raises" if f["direction"] == "increases_risk" else "lowers",
            weightPct=round(abs(f["impact"]) / total, 4),
        )
        for f in raw_factors
    ]

    pricing = scoring.price_loan(pd_, a.monthly_income, a.existing_debt, a.credit_limit,
                                 decision, tenure_months=OFFICER_TENOR_MONTHS)

    audit.info("officer decision pd=%.4f verdict=%s override=%s model=%s",
              pd_, decision, override_reason, settings.app_version)

    return OfficerScoreResult(
        pd=round(pd_, 4), band=scoring.band(pd_), verdict=decision, factors=factors,
        pricing=OfficerPricing(
            offered_amount=pricing["max_loan_amount"],
            apr=round(pricing["interest_rate_pct"] / 100, 4),
            emi=pricing["monthly_emi"],
            tenor_months=pricing["tenure_months"],
        ),
        override_reason=scoring.HARD_OVERRIDE_REASONS.get(override_reason) if override_reason else None,
    )


@app.post("/score", response_model=OfficerScoreResult, tags=["officer"])
@limiter.limit(settings.rate_limit)
def score(request: Request, applicant: OfficerApplicantInput, _key: ApiKey):
    return _score_officer(applicant)


@app.post("/score/batch", response_model=OfficerBatchResponse, tags=["officer"])
@limiter.limit(settings.rate_limit)
def score_batch(request: Request, batch: OfficerBatchRequest, _key: ApiKey):
    rows = [
        OfficerBatchRow(applicant_id=aid, **_score_officer(a).model_dump())
        for aid, a in batch.applicants.items()
    ]
    counts = {"approve": 0, "review": 0, "decline": 0}
    band_dist = {b: 0 for b in "ABCDE"}
    exposure = 0.0
    for r in rows:
        counts[r.verdict] += 1
        band_dist[r.band] += 1
        if r.verdict != "decline":
            exposure += r.pricing.offered_amount
    pds = sorted(r.pd for r in rows)
    n = len(pds)
    median = pds[n // 2] if n % 2 else (pds[n // 2 - 1] + pds[n // 2]) / 2
    summary = OfficerBatchSummary(
        count=n, approve=counts["approve"], review=counts["review"], decline=counts["decline"],
        avg_pd=round(sum(pds) / n, 4), median_pd=round(median, 4),
        band_dist=band_dist, total_offered_exposure=exposure,
    )
    return OfficerBatchResponse(results=rows, summary=summary)


# --- /self-assessment: consumer self-service contract (see docs/specs/consumer-ux.md) ---
# Same model + scoring.py decision logic, wrapped in warm ("readiness", not "reject") framing,
# plus an advice/goal engine that re-scores real what-if profiles through the actual model
# rather than guessing -- every delta shown to the user came from an actual forward pass.

class ConsumerProfile(BaseModel):
    model_config = {"extra": "forbid"}

    age: int = Field(..., ge=18, le=80)
    monthly_income: float = Field(..., gt=0, le=1e8)
    credit_limit: float = Field(..., ge=0, le=1e9)
    existing_debt: float = Field(..., ge=0, le=1e9)
    employment_years: float = Field(..., ge=0, le=50)
    num_existing_loans: int = Field(..., ge=0, le=20)
    payment_history: list[int] = Field(..., min_length=12, max_length=12)

    @field_validator("payment_history")
    @classmethod
    def _status_range(cls, v: list[int]) -> list[int]:
        if any(s < -2 or s > 9 for s in v):
            raise ValueError("payment_history values must be between -2 and 9")
        return v


class SelfAssessmentRequest(BaseModel):
    model_config = {"extra": "forbid"}
    profile: ConsumerProfile


class ConsumerOffer(BaseModel):
    qualifies: bool
    secured: bool
    max_amount: int
    apr: float
    tenure_months: int
    monthly_emi: int


class ConsumerWhyFactor(BaseModel):
    feature: str
    label: str
    impact: float
    direction: str
    detail: str


class ConsumerAdvice(BaseModel):
    id: str
    title: str
    pd_before: float
    pd_after: float
    delta: float
    effort: str
    horizon_months: int
    cost_inr: float | None
    unlocks: ConsumerOffer


class ConsumerGoal(BaseModel):
    target_pd: float
    reachable: bool
    steps: list[str]
    projected_pd: float
    projected_offer: ConsumerOffer


class AssessResponse(BaseModel):
    pd: float
    band: str
    band_headline: str
    threshold: float
    offer_now: ConsumerOffer
    why: list[ConsumerWhyFactor]
    advice: list[ConsumerAdvice]
    goal: ConsumerGoal
    note: str | None = None


@app.post("/self-assessment", response_model=AssessResponse, tags=["consumer"])
@limiter.limit(settings.rate_limit)
def self_assessment(request: Request, body: SelfAssessmentRequest, _key: ApiKey):
    cfg, xgb, net = STATE["cfg"], STATE["xgb"], STATE["net"]
    profile = body.profile.model_dump()
    threshold = cfg["thresholds"]["decline"]

    pd_ = advice_engine.score_profile(cfg, xgb, net, profile)
    band_name, headline = advice_engine.band(pd_, threshold)
    offer_now = advice_engine.price_offer(pd_, threshold, profile["monthly_income"], profile["existing_debt"])
    why = advice_engine.why_factors(cfg, xgb, net, profile)
    advice_items = advice_engine.build_advice(cfg, xgb, net, profile, pd_, threshold)
    goal = advice_engine.build_goal(cfg, xgb, net, profile, advice_items, pd_, threshold)

    override_reason = scoring.check_hard_override(
        profile["monthly_income"], profile["existing_debt"], profile["credit_limit"],
        [float(v) for v in profile["payment_history"]])
    note = None
    if override_reason:
        offer_now = {**offer_now, "qualifies": False, "secured": True}
        note = "This profile needs a closer, manual look before any offer is final."

    audit.info("self-assessment pd=%.4f band=%s override=%s model=%s",
              pd_, band_name, override_reason, settings.app_version)

    return AssessResponse(
        pd=round(pd_, 4), band=band_name, band_headline=headline, threshold=round(threshold, 4),
        offer_now=ConsumerOffer(**offer_now),
        why=[ConsumerWhyFactor(**w) for w in why],
        advice=[ConsumerAdvice(**{k: v for k, v in a.items() if k != "_mutate"}) for a in advice_items],
        goal=ConsumerGoal(**goal),
        note=note,
    )
