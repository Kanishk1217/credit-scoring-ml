---
title: Credit Scoring API
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# Credit Scoring ML System

A credit-scoring system that estimates a loan applicant's **probability of default**, built as a
step-by-step learning project that goes from classical machine learning to deep learning, and ends with
a hardened, deployable API. The headline model is a **hybrid** that fuses XGBoost (static features) with
an LSTM (payment-history sequence).

## Highlights
- **Full ML-to-DL journey** in 9 documented notebooks: EDA, Weight of Evidence, a logistic-regression
  scorecard, XGBoost / LightGBM / CatBoost with calibration, multi-table feature engineering, a neural
  net from scratch, an LSTM, and the fused hybrid.
- **Hybrid model** (XGBoost + LSTM) reaches **0.892 AUC** on synthetic data, is **isotonic-calibrated**
  (predicted PD tracks the true default rate, not ~2x it), and uses **cost-based decision thresholds**
  instead of an arbitrary 0.5 cutoff.
- **No protected attributes.** Sex/marriage/education are never scored — enforced at the API schema
  level (`extra="forbid"`), not just by convention.
- **Explainability, pricing, and advice**: every decision ships with ranked "why" factors (XGBoost
  SHAP + a payment-history counterfactual), risk-based loan pricing (amount + interest rate), and
  what-if advice ("6 months on-time payments would drop your risk from X% to Y%").
- **Production API** (FastAPI): API-key auth, rate limiting, input validation, security headers, audit
  logging, single and batch scoring, a test suite, Docker, and CI.

## Model performance (held-out test)
| Model | Test AUC | Notes |
|---|---|---|
| Logistic-regression scorecard (German Credit) | 0.80 | learning notebook |
| XGBoost (Give Me Some Credit) | 0.869 | learning notebook |
| LSTM on payment sequences (Taiwan, 6mo) | 0.737 | learning notebook |
| Hybrid: XGBoost + LSTM (Taiwan, 6mo, uncalibrated) | 0.775 | superseded — see below |
| **Hybrid: XGBoost + LSTM (synthetic, 12mo, calibrated) — SERVED MODEL** | **0.892** | see below |

**Why the served model changed.** The Taiwan-trained hybrid (0.775 AUC) scored SEX/EDUCATION/
MARRIAGE as inputs (illegal for real lending in most jurisdictions) and shipped **without
calibration** — its mean predicted PD was ~43% against an actual default rate of ~22%, roughly
double the truth, which made the fixed 0.2/0.5 decision thresholds nonsensical (91% of applicants
would have been flagged for review or decline). The served model now trains on a larger synthetic
book (150k rows, 12 months of history, no protected attributes) and is isotonic-calibrated with
cost-based thresholds fit on held-out data. Full before/after numbers, the reliability table, and
the fairness audit are in the model card.

See [docs/model_card.md](docs/model_card.md) for intended use, limitations, and fairness notes.

## Architecture (hybrid, served by the API)
```
   static features ──> XGBoost ───────────────> risk score ┐
                                                            concat ──> MLP ──> probability of default
   payment sequence ──> LSTM ──> temporal embedding ───────┘
```

## Quickstart

```bash
# 1. environment (uses uv: https://docs.astral.sh/uv/)
uv sync

# 2. (optional) retrain the model from scratch — one script, torch used for training only
uv run python src/train_synth_model.py
#   -> models/hybrid_xgb.joblib, hybrid_fusion.npz (torch-free weights), hybrid_config.json

# 3. configure and run the API
cp .env.example .env        # then set CREDIT_API_KEYS to your own key(s)
uv run uvicorn api.app:app --port 8077
#   interactive docs: http://localhost:8077/docs  (send header X-API-Key: <your key>)
```

### Scoring an applicant
```bash
curl -X POST http://localhost:8077/predict \
  -H "Content-Type: application/json" -H "X-API-Key: <your key>" \
  -d '{"age":30,"monthly_income":45000,"credit_limit":250000,"existing_debt":90000,
       "employment_years":3.5,"num_existing_loans":2,
       "pay_status":[0,0,0,0,0,0,0,0,-1,1,2,2]}'
# -> {"probability_of_default":0.5188,"recommendation":"decline", "why":[...],
#     "pricing":{"max_loan_amount":0,...}, "advice":[...], "model_version":"1.0.0"}
```
`pay_status` is **12 months**, oldest to newest; `<= 0` means paid on time, `1..9` means that many
months late. No sex/marriage/education fields exist in the schema (`extra="forbid"` rejects them).
Add `"has_collateral": true` for a secured loan (better rate, wider approval cutoff). The response
includes ranked **why** factors, a **pricing** offer, and **advice** (ranked what-if improvements).
A `/predict/batch` endpoint scores up to 1,000 applicants at once.

## API security and operations
- **Auth:** every scoring request needs a valid `X-API-Key` (constant-time comparison).
- **Rate limiting:** per key, configurable (default 30/minute).
- **Validation:** strict Pydantic bounds on every field; bad input returns 422.
- **Hardening:** locked-down CORS, security headers, non-leaking error responses.
- **Observability:** request-ID + latency logs, and an audit log of every credit decision.
- **Config:** all settings and secrets come from environment variables (see `.env.example`).

## Testing, lint, CI
```bash
uv run pytest -q          # API tests: auth, validation, scoring, batch
uv run ruff check .       # lint
```
GitHub Actions runs both on every push (`.github/workflows/ci.yml`).

## Deployment
Backend: containerized (`Dockerfile`, torch-free at serve time — fits a free 512MB host) and
deployed to Render (`render.yaml`). Frontend: `frontend/` deploys to Cloudflare Pages, whose
`functions/api/[[path]].ts` proxies to the backend and injects the API key server-side so the
browser never holds it. Set `CREDIT_API_KEYS` (backend) and `BACKEND_URL`/`API_KEY` (frontend) as
secrets in each platform's dashboard; never commit them.

## Project structure
```
api/          FastAPI app, config, and scoring business logic (the deployable service)
src/          model definitions, training scripts (train_synth_model.py), data loaders
models/       saved model artifacts (xgb, fusion net weights, config incl. calibration/thresholds)
notebooks/    01-09: the full learning journey, EDA -> hybrid
frontend/     Vite + React homepage and live demo
tests/        API + scoring unit tests
docs/         learning log, model card, and feature design specs
data/         datasets (git-ignored)
```

## Datasets
**Served model:** a synthetic lender book (`src/synth_data.py`) — 150k borrowers, 12 months of
payment history, financial features only (no protected attributes). **Learning notebooks (01-09):**
German Credit (UCI), Give Me Some Credit (Kaggle), Home Credit Default Risk (Kaggle), Taiwan Credit
Card (UCI). All public. Data files are not committed.

## Limitations
This is a learning and demonstration project. The served model is trained on **synthetic** data
(the schema mirrors a real lender's, so swapping in real data is a loader change, not a redesign) —
it is **not** validated for real lending decisions. Calibration, cost-based thresholds, and a
fairness audit are already implemented (see the model card), but before any real use the model
would need retraining on the lender's own real population and ongoing drift monitoring.

## Roadmap
- Loan-officer dashboard (single applicant + CSV batch)
- Consumer self-assessment page (check your own risk, get improvement advice)
- Self-serve API-key sign-up
- Real Indian credit-bureau data as a design-partner pilot
- Drift monitoring (PSI) on incoming traffic
