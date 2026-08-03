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

**The production model — the one actually running behind the live API and the website — is
`models_real`, trained on 167,115 real Home Credit applicants (7 self-reportable features).** It is
served with `CREDIT_MODEL_DIR=models_real`. A separate hybrid trained on a synthetic book (`models/`)
still exists in the repo as the original learning artifact and is what the test suite's default
assumptions are built around, but it is not what the deployed site scores real people with.

## Highlights
- **Full ML-to-DL journey** in 9 documented notebooks: EDA, Weight of Evidence, a logistic-regression
  scorecard, XGBoost / LightGBM / CatBoost with calibration, multi-table feature engineering, a neural
  net from scratch, an LSTM, and the fused hybrid.
- **Hybrid model** (XGBoost + LSTM), same architecture for every data source: **0.6654 AUC** on the
  real Home Credit data it's actually served on (0.892 on the synthetic learning dataset — see the
  table below for why the honest real number is lower). Both are **isotonic-calibrated** (predicted PD
  tracks the true default rate) and use a **recall-target decision threshold** — see "Decision
  threshold" below for what that means and the real trade-off numbers.
- **No protected attributes.** Sex/marriage/education are never scored — enforced at the API schema
  level (`extra="forbid"`), not just by convention.
- **Explainability, pricing, and advice**: every decision ships with ranked "why" factors (XGBoost
  SHAP + a payment-history counterfactual), risk-based loan pricing (amount + interest rate), and
  what-if advice ("6 months on-time payments would drop your risk from X% to Y%").
- **Production API** (FastAPI): API-key auth, rate limiting, input validation, security headers, audit
  logging, single and batch scoring, a test suite, Docker, and CI.
- **Tested cross-population generalization, honestly**: `models_real` scores AUC 0.51 (no signal) on
  a different real lending population (US LendingClub) — and a generic onboarding script
  (`src/train_new_market.py`) that trains a population-specific model instead recovers real signal
  (AUC 0.66) on the exact same people. See "Multi-market models" below.

## Model performance (held-out test)
| Model | Test AUC | Notes |
|---|---|---|
| Logistic-regression scorecard (German Credit) | 0.80 | learning notebook |
| XGBoost (Give Me Some Credit) | 0.869 | learning notebook |
| LSTM on payment sequences (Taiwan, 6mo) | 0.737 | learning notebook |
| Hybrid: XGBoost + LSTM (Taiwan, 6mo, uncalibrated) | 0.775 | superseded — see below |
| Hybrid: XGBoost + LSTM (synthetic, 12mo, calibrated) | 0.892 | learning artifact, still in the test suite, not what's deployed |
| **Hybrid: XGBoost + LSTM (real Home Credit, 12mo, calibrated) — SERVED MODEL** | **0.6654** | what the live site actually scores you with |

**Why the real number is lower, and that's the honest one to trust.** The synthetic book was
generated so its two branches (financial snapshot, payment trajectory) carry clean, complementary
signal — real applicants are messier. 0.6654 AUC on 167,115 real Home Credit applicants (8.1% actual
default rate) is the number that reflects what this model can really tell about a real person, and
it's the number quoted everywhere else in this README. Full before/after history of the original
Taiwan→synthetic retraining (protected attributes, missing calibration) is in the model card.

See [docs/model_card.md](docs/model_card.md) (synthetic learning model) and
[reports/real_data_model_report.md](reports/real_data_model_report.md) (`models_real`, the served
model) for intended use, limitations, and fairness notes.

## Decision threshold — what it does, and the real numbers behind it

The model outputs one number per applicant: a calibrated probability of default (PD). That number
alone isn't a decision — the **threshold** is the cutoff that turns it into approve / review /
decline. Moving the threshold doesn't change the model or require retraining anything; it only moves
where the line falls on the same calibrated score, which changes who ends up on which side of it.

`models_real` ships a **recall-target threshold**, `decline_threshold = 0.075`, chosen to catch
**65% of real defaulters** — deliberately, not as a leftover default. That choice was measured
against the alternative (a much higher cutoff, e.g. 0.15) on the same held-out test set:

| | `decline_threshold = 0.075` (shipped) | `decline_threshold = 0.15` (considered, not shipped) |
|---|---|---|
| Defaulters caught (recall) | 62.5% | 14.6% |
| Of declines, actually defaulters (precision) | 12.6% | 19.2% |
| Overall accuracy | 61.8% | 88.1% |
| Of all applicants declined | 40.2% | 6.2% |
| Good (would-repay) applicants wrongly declined | 38.2% | 5.4% |
| Good applicants wrongly declined per real defaulter caught | 6.9 | 4.2 |

The lower threshold catches roughly 2 in 3 defaulters at the cost of declining 4 in 10 applicants
overall, most of whom would have repaid. The higher threshold declines far fewer people overall and
wrongly rejects far fewer good applicants, but only catches about 1 in 7 defaulters. There is no
setting that avoids this trade-off — only a choice of which cost to accept. The 0.075 threshold is
shipped because catching defaulters was weighted higher than minimizing false declines; this is a
recorded business choice, not a hidden default, and it's re-derivable any time by rerunning
`src/train_home_credit_models.py --variant real` against the current calibrated model.

## Multi-market models: real data, not just synthetic

Beyond the synthetic learning model, this project trains and validates models on real,
independently-sourced lending data — and treats "does this generalize to a different real
population" as a question to actually test, not assume.

| Model | Data | Test AUC | 5-fold CV AUC | Notes |
|---|---|---|---|---|
| **`models_real`** | Real Home Credit, 7 self-reportable features | **0.665** | 0.6650 ± 0.0030 | **served — production model behind the API and `/advisor`** |
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

# 2. (optional) retrain a model from scratch — one script, torch used for training only
uv run python src/train_home_credit_models.py --variant real
#   -> models_real/hybrid_xgb.joblib, hybrid_fusion.npz (torch-free weights), hybrid_config.json
#   --variant synthetic | real_rich trains the other variants the same way

# 3. configure and run the API
cp .env.example .env        # then set CREDIT_API_KEYS to your own key(s)
#   CREDIT_MODEL_DIR defaults to "models" (synthetic) locally so the test suite's assumptions hold;
#   set CREDIT_MODEL_DIR=models_real in .env to run the same model production actually serves
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
# with CREDIT_MODEL_DIR=models_real (the served model), this returns:
# -> {"probability_of_default":0.2303,"recommendation":"decline","approve_threshold":0.037,
#     "decline_threshold":0.075,"why":[...], "pricing":{"max_loan_amount":0,...},
#     "advice":[...], "model_version":"1.0.0"}
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
**Served (production) model:** real Home Credit Default Risk data (Kaggle) — 167,115 real applicants
across `application_train` + `bureau` + `installments_payments`, 7 self-reportable features
(`models_real`). **Also trained, not served:** `models_real_rich` (38 features + bureau history) and
`models_lendingclub` (historical US LendingClub loans) — see "Multi-market models" above. **Synthetic
learning model** (`models/`, not served): a generated lender book (`src/synth_data.py`) — 150k
borrowers, 12 months of payment history, financial features only. **Learning notebooks (01-09):**
German Credit (UCI), Give Me Some Credit (Kaggle), Home Credit Default Risk (Kaggle), Taiwan Credit
Card (UCI). All public. Data files are not committed.

## Limitations
The served model (`models_real`) is trained on real historical Home Credit applications, not live
production outcomes from this specific lender — calibration, the recall-target threshold, and a
fairness audit are already implemented (see `reports/real_data_model_report.md`), but before any real
lending use it would need validation against the actual lender's own population and ongoing drift
monitoring. It has also been directly tested against an independent real population (US
LendingClub) and found not to generalize (AUC 0.51) — see "Multi-market models" above; a new market
needs its own model, not a reused one. The synthetic hybrid (`models/`) is a learning artifact only,
never intended for real decisions.

## Model verification
Beyond the metrics above, `models_real` has been checked for:
- **CV stability**: 5-fold CV, AUC 0.662-0.6703 (mean 0.665, std 0.003) — not a lucky split.
- **Fairness audit**: gender demographic-parity gap 0.0617 (equalized-odds gap 0.0569), region
  demographic-parity gap 0.0481 (equalized-odds gap 0.0297) — `region`/`gender` are illustrative
  proxies only, never used as scoring inputs.
- **Monotonicity**: risk moves in the economically sensible direction for every static feature
  (more debt → more risk, more income → less risk, etc.), with no reversals.
- **Drift monitoring**: `src/drift_monitor.py` computes the Population Stability Index against a
  saved reference distribution and flags moderate/major population shift.

Full numbers: `reports/real_data_model_report.md` (`models_real`) and `reports/model_report_card.md`
(the synthetic learning model).

## Roadmap
- ~~Loan-officer dashboard (single applicant + CSV batch)~~ — done, `/officer`, Supabase-gated
- ~~Consumer self-assessment page~~ — done, `/advisor`, Supabase-gated
- ~~Prove/disprove cross-population generalization~~ — done: `models_real` fails on a different
  real population (AUC 0.51); `models_lendingclub` proves a population-specific model recovers
  real signal (AUC 0.66) on the same data
- ~~Serve a real-data model, not just the synthetic one~~ — done: production runs
  `CREDIT_MODEL_DIR=models_real`
- Wire `models_real_rich` and `models_lendingclub` into live serving (needs a new request schema
  and feature-row builder per model, not just a config flip — see `docs/model_creation_summary.md`)
- Apply the validated per-group threshold fairness mitigation to production scoring (currently
  tested but not shipped — see `reports/real_data_model_report.md`)
- Self-serve API-key sign-up (distinct from the Supabase dashboard login already built)
- A repeat run of `train_new_market.py` against a real Indian credit-bureau dataset, if one
  becomes available as a design-partner pilot
