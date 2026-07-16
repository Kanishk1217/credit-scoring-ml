# Credit Scoring ML System

A finance ML + deep learning project: predict loan default (probability of default, PD)
using classical ML and deep learning, with explainability, a fairness audit, and a
production API. Built as a step-by-step learning project.

## Models (target end state)
1. **Traditional** — XGBoost / LightGBM / CatBoost scorecard on static financial features.
2. **LSTM sequence** — temporal default risk from 12 months of payment behaviour.
3. **Hybrid** — GBDT static score + LSTM temporal embedding fused in an MLP (headline model).
4. **Alternative data** — thin-file scoring for borrowers with no credit history.

Plus SHAP/LIME explainability, a Fairlearn fairness audit, a FastAPI serving layer,
and a Streamlit loan-officer dashboard.

## Environment
Managed with [uv](https://docs.astral.sh/uv/). Python 3.12.

```bash
# one-time: install uv (already done on this machine)
# curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync                 # install/refresh dependencies from uv.lock
uv run jupyter lab      # launch Jupyter for the notebooks
uv run python <file>    # run any script inside the project env
```

## Layout
```
data/raw/         # original downloaded datasets (git-ignored)
data/processed/   # cleaned / feature-engineered data (git-ignored)
notebooks/        # learning + EDA notebooks, numbered by order
src/              # reusable pipeline code (loaders, features, training)
models/           # saved model artifacts (git-ignored)
api/              # FastAPI serving app
reports/          # figures, metrics, case study
docs/             # notes, theory writeups
```

## Datasets
| Dataset | Source | Role |
|---|---|---|
| German Credit | UCI (no login) | Fundamentals, first scorecard |
| Give Me Some Credit | Kaggle | First large-scale model |
| Home Credit Default Risk | Kaggle | Primary 7-table dataset |
| Taiwan Credit Card | UCI | Cross-dataset validation |
| LendingClub | Kaggle | Alternative-data / thin-file phase |

## Progress
- [x] Phase 1.1 — environment setup (uv, Python 3.12, classical stack)
- [ ] Phase 1.4 — EDA on German Credit + Give Me Some Credit
