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
- **Tested cross-population generalization, honestly**: the real Home Credit model scores AUC
  0.51 (no signal) on a different real lending population (US LendingClub) — and a generic
  onboarding script (`src/train_new_market.py`) that trains a population-specific model instead
  recovers real signal (AUC 0.66) on the exact same people. See "Multi-market models" below.

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

## Multi-market models: real data, not just synthetic

Beyond the synthetic served model above, this project also trains and validates models on real,
independently-sourced lending data — and treats "does this generalize to a different real
population" as a question to actually test, not assume.

| Model | Data | Test AUC | 5-fold CV AUC | Notes |
|---|---|---|---|---|
| `models_real` | Real Home Credit, 7 self-reportable features | 0.665 | 0.6650 ± 0.0030 | intended for public API / `/advisor` |
| `models_real_rich` | Real Home Credit, 38 features (+ bureau history) | 0.756 | 0.7554 ± 0.0047 | not yet servable — needs a bureau-data lookup |
| `models_lendingclub` | Real historical US LendingClub loans, 34 features | 0.661 | 0.6577 ± 0.0060 | trained via the generic onboarding script below |

**The honest, load-bearing finding**: `models_real` was tested against LendingClub's real,
independent population and scored **AUC 0.5079 — no better than a coin flip.** Quantile-matching
the income/debt scale to Home Credit's distribution only moved it to 0.5446, proving the failure
wasn't a units mismatch — the actual relationship between financial features and default risk
differs by population. Training a model on that population's own real outcomes recovered real
signal: re-scoring the *same* 300 real applicants through `models_lendingclub` moved AUC from
0.5079 to 0.7460. There is no universal credit-risk model; every real lender uses population-
specific models for exactly this reason.

### `src/train_new_market.py` — generic new-market onboarding

One script, not a scattered pile of one-off files: feature engineering (with an explicit data-
leakage allowlist/blocklist and an assertion that guards it) → training (XGBoost, early-stopped
on a validation split) → isotonic calibration → cost-based thresholds → fairness audit → 5-fold
cross-validation → confusion matrix → model registry entry, in one run.

```bash
uv run python src/train_new_market.py data/raw/lendingclub/loan.csv --market lendingclub
```

Full reasoning (why no LSTM branch for this market, the leakage discipline, a real two-digit-year
date bug found and fixed, fairness-proxy caveats) is in
[docs/model_creation_summary.md](docs/model_creation_summary.md), with Mermaid workflow diagrams.
The executed walkthrough with plots (confusion matrix, ROC, calibration, CV boxplot) is
[notebooks/10_new_market_onboarding.ipynb](notebooks/10_new_market_onboarding.ipynb).

Every model — synthetic, real, and multi-market — is catalogued in `model_registry.json` with a
content fingerprint (sha256 of the actual trained artifacts, not the config) and honest metrics;
`GET /model-registry` (API-key protected) returns the full catalog live.

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
uv run python src/train_home_credit_models.py --variant synthetic
#   -> models/hybrid_xgb.joblib, hybrid_fusion.npz (torch-free weights), hybrid_config.json
#   --variant real | real_rich trains the real-Home-Credit variants the same way

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
secrets in each platform's dashboard; never commit them. **The two must match** — a mismatch
surfaces as "invalid or missing API key" (401) on every scoring request.

Accounts (`/officer` and `/advisor` both require sign-in): set `VITE_SUPABASE_URL` and
`VITE_SUPABASE_ANON_KEY` as Cloudflare Pages environment variables (safe to expose client-side —
see `frontend/.env.example` for the full setup, including enabling Google sign-in).

## Project structure
```
api/          FastAPI app, config, and scoring business logic (the deployable service)
src/          model definitions; training scripts (train_home_credit_models.py covers all 3
              Home Credit-family models, train_new_market.py for a new real-world market), data loaders
models/, models_real/, models_real_rich/, models_lendingclub/
              saved model artifacts, one directory per model (registry: model_registry.json)
notebooks/    01-09: the full learning journey, EDA -> hybrid; 10: new-market onboarding (LendingClub)
frontend/     Vite + React homepage, live demo, /officer and /advisor dashboards (Supabase-gated)
tests/        API + scoring unit tests
docs/         learning log, model card, model creation summary, and feature design specs
data/         datasets (git-ignored)
```

## Datasets
**Served (synthetic) model:** a synthetic lender book (`src/synth_data.py`) — 150k borrowers, 12
months of payment history, financial features only (no protected attributes). **Real models:**
Home Credit Default Risk (Kaggle, `models_real`/`models_real_rich`) and historical LendingClub
loans (`models_lendingclub`) — see "Multi-market models" above. **Learning notebooks (01-09):**
German Credit (UCI), Give Me Some Credit (Kaggle), Home Credit Default Risk (Kaggle), Taiwan Credit
Card (UCI). All public. Data files are not committed.

## Limitations
This is a learning and demonstration project. The served model is trained on **synthetic** data
(the schema mirrors a real lender's, so swapping in real data is a loader change, not a redesign) —
it is **not** validated for real lending decisions. Calibration, cost-based thresholds, and a
fairness audit are already implemented (see the model card), but before any real use the model
would need retraining on the lender's own real population and ongoing drift monitoring.

## Model verification
Beyond the metrics above, the model has been checked for:
- **CV stability**: 5-fold CV, AUC 0.884-0.891 (std 0.0024) — not a lucky split (`src/cross_validate.py`).
- **Prevalence sensitivity**: AUC/recall are stable across natural vs artificially-balanced
  evaluation sets; precision and calibration are not (expected — calibration is population-specific).
- **Monotonicity**: risk moves in the economically sensible direction for every static feature
  (more debt → more risk, more income → less risk, etc.), with no reversals.
- **Drift monitoring**: `src/drift_monitor.py` computes the Population Stability Index against a
  saved reference distribution and flags moderate/major population shift (e.g. would catch a real
  deployment population drifting away from the one the model was calibrated on).

Full numbers: `reports/model_report_card.md`.

## Roadmap
- ~~Loan-officer dashboard (single applicant + CSV batch)~~ — done, `/officer`, Supabase-gated
- ~~Consumer self-assessment page~~ — done, `/advisor`, Supabase-gated
- ~~Prove/disprove cross-population generalization~~ — done: `models_real` fails on a different
  real population (AUC 0.51); `models_lendingclub` proves a population-specific model recovers
  real signal (AUC 0.66) on the same data
- Wire `models_real_rich` and `models_lendingclub` into live serving (needs a new request schema
  and feature-row builder per model, not just a config flip — see `docs/model_creation_summary.md`)
- Apply the validated per-group threshold fairness mitigation to production scoring (currently
  tested but not shipped — see `reports/real_data_model_report.md`)
- Self-serve API-key sign-up (distinct from the Supabase dashboard login already built)
- A repeat run of `train_new_market.py` against a real Indian credit-bureau dataset, if one
  becomes available as a design-partner pilot
