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
- **Hybrid model** (XGBoost + LSTM) reaches **0.775 AUC**, beating either component alone.
- **Production API** (FastAPI): API-key auth, rate limiting, input validation, security headers, audit
  logging, single and batch scoring, a test suite, Docker, and CI.

## Model performance (held-out test)
| Model | Test AUC |
|---|---|
| Logistic-regression scorecard (German Credit) | 0.80 |
| XGBoost (Give Me Some Credit) | 0.869 |
| LSTM on payment sequences (Taiwan) | 0.737 |
| **Hybrid: XGBoost + LSTM (Taiwan)** | **0.775** |

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

# 2. (optional) rebuild the model files from data — two steps, separate processes
uv run python src/make_hybrid_features.py   # XGBoost branch -> models/hybrid_xgb.joblib
uv run python src/train_fusion.py           # fusion net     -> models/hybrid_fusion.pt

# 3. configure and run the API
cp .env.example .env        # then set CREDIT_API_KEYS to your own key(s)
uv run uvicorn api.app:app --port 8077
#   interactive docs: http://localhost:8077/docs  (send header X-API-Key: <your key>)
```

### Scoring an applicant
```bash
curl -X POST http://localhost:8077/predict \
  -H "Content-Type: application/json" -H "X-API-Key: <your key>" \
  -d '{"limit_bal":120000,"sex":2,"education":2,"marriage":1,"age":30,
       "bill_amt":[80000,82000,85000,88000,90000,92000],
       "pay_amt":[3000,3000,2500,2000,1500,1000],
       "pay_status":[0,0,-1,-1,2,2]}'
# -> {"probability_of_default":0.7758,"recommendation":"decline","model_version":"1.0.0"}
```
`pay_status` is 6 months (Apr to Sep); `<= 0` means paid on time, `1..9` means that many months late.
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
Containerized (`Dockerfile`) and ready for Render (`render.yaml`). Set `CREDIT_API_KEYS` and
`CREDIT_ALLOWED_ORIGINS` as secrets in the platform dashboard; never commit them.

## Project structure
```
api/          FastAPI app + config (the deployable service)
src/          model definition, training scripts, data loaders
models/        saved model artifacts (xgb, fusion net, config)
notebooks/    01-09: the full learning journey, EDA -> hybrid
tests/         API test suite
docs/          learning log + model card
data/          datasets (git-ignored)
```

## Datasets
German Credit (UCI), Give Me Some Credit (Kaggle), Home Credit Default Risk (Kaggle), Taiwan Credit
Card (UCI). All public. Data files are not committed.

## Limitations
This is a learning and demonstration project. The models are trained on public research datasets and
are **not** validated for real lending decisions. Before any real use they would need retraining on the
lender's own population, probability calibration, a fairness audit, and drift monitoring. See the model
card for detail.

## Roadmap
- Per-applicant explanations (SHAP) in the API response
- Fairness audit (demographic parity, equalized odds)
- Drift monitoring (PSI) on incoming traffic
- Frontend dashboard for loan officers
- Risk-based pricing (recommended amount and interest rate)
