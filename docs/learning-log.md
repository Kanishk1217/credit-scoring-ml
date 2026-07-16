# Learning Log

A running, plain-language record of what we covered each session. Newest at the bottom.
Read this to refresh before a new session, or months later to remember why things are the way
they are. Each entry: what we did, the concepts, and anything to remember.

---

## Session 1 — 2026-07-16 — Setup + first EDA (Phase 1, tasks 1 & 4)

### What we did
- Installed **uv** (fast Python package/environment manager) and created the project at
  `/Users/kanishk_pansari/Desktop/Credit Ml` with Python 3.12.
- Installed the classical-ML stack only (numpy, pandas, scikit-learn, matplotlib, seaborn,
  jupyter, kaggle). Deep-learning libraries come later, in their own phase.
- Set up the repo layout (`data/`, `notebooks/`, `src/`, `models/`, `api/`, `reports/`, `docs/`)
  and git.
- Downloaded **German Credit** (UCI, 1000 borrowers, 20 features) — no login needed.
- Built and verified `notebooks/01_german_credit_eda.ipynb`.

### Concepts learned
- **A dataset is often coded.** German Credit uses codes like `A11`, `A34`. Step one of any ML
  project is decoding raw values into readable, correctly-typed columns.
- **Target convention in credit scoring:** we predict *default*. We made a `default` column where
  **1 = defaulted (bad)**, **0 = repaid (good)**. "1 = the event you care about" is the standard.
- **Class imbalance and the accuracy trap.** German Credit is 700 repaid / 300 defaulted. A model
  that blindly predicts "everyone repays" gets **70% accuracy** while catching zero defaulters, so
  it's useless to a lender. This is why we'll judge models with **AUC, KS, Gini** (which measure how
  well a model *separates* good from bad) instead of raw accuracy.
- **Skewed numeric features.** `credit_amount` is right-skewed (many small loans, a few huge ones).
  Skew can matter for some models/encodings; a log transform is a common fix. Revisit in feature eng.
- **Default rate by category = the core credit-scoring EDA move.** For each category of a feature,
  what fraction defaulted? Big spread across categories = predictive feature. Example we saw:
  checking status `< 0 DM` defaults **49%** vs `no account` only **12%**. That gap is signal.
- Noticed a real data quirk: in `credit_history`, the "critical/other credits" group defaults *less*
  than the "all credits paid" group — a reminder that features don't always behave the way intuition
  expects, and that's worth investigating rather than ignoring.

### Key facts / commands to remember
- Run anything in the project env: `cd "/Users/kanishk_pansari/Desktop/Credit Ml" && uv run <cmd>`.
- Launch notebooks: `uv run jupyter lab`.
- German Credit split: 700 good / 300 bad. Approve-everyone baseline accuracy = 70%.

### Next session
- **Weight of Evidence (WoE)** and **Information Value (IV):** turn default-rate gaps into
  model-ready numbers and rank features by predictive power (Phase 1, task 6).
- Download **Give Me Some Credit** from Kaggle (need `kaggle.json` token) and rerun EDA at ~250x scale.

---

## Session 2 — 2026-07-16 — Weight of Evidence & Information Value (Phase 1, task 6)

### What we did
- Graduated the German Credit schema + loading out of notebook 01 into `src/data_loader.py`
  (`load_german_credit()`), so notebooks share one loader. First use of the "notebook → src/" pattern.
- Built `notebooks/02_woe_iv.ipynb`: WoE by hand for one feature, then a reusable `woe_iv()`
  function, then IV ranking of every feature, numeric binning, and the WoE transform.

### Concepts learned
- **Weight of Evidence (WoE)** converts a category into a single risk number:
  `WoE = ln(good_share / bad_share)` where good = repaid, bad = defaulted. **Positive = safer than
  average, negative = riskier, ~0 = uninformative.** WoE moves *opposite* to the default rate.
  Confirmed: `no account` (12% default) → WoE +1.17; `< 0 DM` (49% default) → WoE -0.82.
- **Zero-cell problem:** if a category has 0 bads (or 0 goods), the log is infinite. Fixed by adding
  a small `+0.5` to each count before computing distributions.
- **Information Value (IV)** = `sum((good_share - bad_share) * WoE)` scores a whole feature.
  Rule of thumb: <0.02 useless · 0.02-0.1 weak · 0.1-0.3 medium · 0.3-0.5 strong · >0.5 suspicious.
- **German Credit IV ranking:** checking_status 0.66 (dominant, "suspiciously strong"),
  credit_history 0.29, savings_status 0.19, purpose 0.17, property 0.11 are the useful ones;
  job (0.009) and telephone (0.006) are useless. IV *is* feature selection.
- **Numeric features must be binned first** (e.g. `pd.qcut(col, q=5)` = quantile bins) before WoE.
  age IV = 0.068, credit_amount IV = 0.093 — both weak-ish alone.
- **WoE transform** = replace a column's categories with their WoE numbers, producing a model-ready
  numeric column. This is the direct input to the logistic-regression scorecard.

### Key facts / commands to remember
- Shared loader: `from src.data_loader import load_german_credit`. Notebooks add repo root to
  `sys.path` first: `sys.path.insert(0, os.path.abspath(".."))`.
- Higher WoE = safer. Higher IV = more predictive (but very high IV can signal leakage).

### Next session
- **Logistic-regression scorecard** on German Credit: WoE-encode all features, fit logistic
  regression, convert coefficients into integer point scores (industry scorecard format) — Phase 2,
  task 1.
- Then download **Give Me Some Credit** from Kaggle (`kaggle.json`) and scale up.

---

## Session 3 — 2026-07-16 — Logistic-regression scorecard (Phase 2, task 1)

### What we did
- Built `notebooks/03_logistic_scorecard.ipynb`: full WoE -> logistic regression -> integer points
  scorecard on German Credit, with proper train/test discipline and real metrics.

### Concepts learned
- **A scorecard** is the deployable artifact: points per attribute, summed to a total, mapped to PD.
  Logistic regression is the engine because it is linear in log-odds and fully interpretable
  (regulator-friendly). Model: `ln(PD / (1-PD)) = b0 + sum(bi * WoE_i)`.
- **Data leakage, and how to avoid it:** split into train/test FIRST, then learn WoE maps and numeric
  bin edges from **train only**, and apply them to test. Computing WoE on the full data before
  splitting inflates AUC dishonestly. This is the #1 beginner mistake.
- **Stratified split** (`stratify=y`) preserves the 30% default rate in both halves — matters with
  imbalance.
- **Metrics that replace accuracy:** AUC (prob. model ranks a random bad above a random good;
  0.5 = coin flip), Gini = 2*AUC - 1, KS = max gap between good/bad cumulative score distributions.
  Our scorecard: **AUC 0.80, Gini 0.61, KS 0.58** on test — above the 0.75 target.
- **Coefficient signs:** since higher WoE = safer, coefficients came out mostly negative
  (17/20) — higher WoE lowers PD. A built-in sanity check.
- **Points formula:** `Factor = pdo/ln(2)`, `Offset = base_score - Factor*ln(base_odds)`,
  `points_i = (Offset - Factor*intercept)/n - Factor * coef_i * WoE_i`. Higher points = safer. We
  anchored `base_score=500` at the **portfolio's own good:bad odds (2.33)** with `pdo=40`, rather
  than the FICO-style "600 at 50:1" (which assumes a ~2% default book and left everyone off-scale).
  Sanity: `no account` scored 51 pts, `< 0 DM` only 9 — matches risk ordering.
- **The cutoff is a business lever, not a model output.** At cutoff 480 we approve 65% of applicants
  with a 13% default rate among approved, while the rejected pool defaults 61%. Raise the cutoff →
  approve fewer, safer. The model just supplies the score; the lender picks the line.

### Key facts / commands to remember
- Always split before encoding. Learn WoE/bins on train, `.map(...).fillna(0)` on test (unseen
  category -> neutral WoE 0).
- Gini = 2*AUC - 1. Higher score = lower PD (corr was -0.97).

### Next session
- **Give Me Some Credit** (250k borrowers) from Kaggle: set up `kaggle.json`, download, EDA at scale.
- Then **XGBoost** — the gradient-boosted model that will beat this scorecard on AUC and become the
  benchmark deep learning has to beat (Phase 2, tasks 2-3).
