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

---

## Session 4 — 2026-07-16 — Give Me Some Credit + XGBoost (Phase 2, tasks 2 & 5-part)

### What we did
- Set up the Kaggle API token (`~/.kaggle/access_token`). Competition download 403'd (rules not
  accepted), so used the public mirror `brycecf/give-me-some-credit-dataset` instead.
- Installed `xgboost` + `optuna`. **macOS gotcha:** xgboost needs OpenMP — fixed with
  `brew install libomp` (LightGBM will need it too, already installed now).
- Built `notebooks/04_give_me_some_credit_xgboost.ipynb`: EDA + cleaning + XGBoost + LR baseline +
  Optuna tuning on 150k borrowers.

### Concepts learned
- **Real messy data vs clean:** 150k rows, 10 features, target `SeriousDlqin2yrs` (90+ days late).
  **6.7% default rate** — so "predict everyone repays" scores **93.3% accuracy** and is useless. The
  extreme version of the Phase 1 accuracy-trap lesson.
- **Missing data:** MonthlyIncome ~20% missing, NumberOfDependents ~2.6%. Added a
  `MonthlyIncome_missing` flag (missingness itself can be signal).
- **Data-quality landmines:** age==0 (impossible), sentinel codes 96/98 in the past-due columns
  (not real counts), and wild ratio outliers (DebtRatio max 329k). Cleaned only the genuine errors
  (-> NaN); left the outliers alone.
- **Gradient boosting:** builds trees one at a time, each trained to fix the current ensemble's
  errors. Captures interactions automatically. Key knobs: n_estimators, learning_rate, max_depth,
  subsample/colsample_bytree, scale_pos_weight, early_stopping.
- **XGBoost handles NaN natively** (learns which way to send missing at each split) — so we passed
  gaps in directly, no imputation. And **trees ignore monotonic outliers** (they split on rank), so
  huge values don't distort like they would a linear model. Two reasons XGBoost suits messy tabular.
- **Imbalance fix:** `scale_pos_weight = n_neg / n_pos` (= 13.96 here) up-weights rare defaulters.
- **3-way split** (train/valid/test, 60/20/20): valid drives early stopping, test stays untouched
  for the honest final score.
- **Results:** XGBoost AUC **0.869** vs Logistic Regression **0.820** on the same data — boosting
  wins by ~0.05 AUC. Top feature: RevolvingUtilizationOfUnsecuredLines.
- **Optuna** does smart hyperparameter search (optimizing on validation, never test). Lesson: tuning
  gave only a marginal gain here because sensible defaults were already strong.

### Key facts / commands to remember
- Kaggle token at `~/.kaggle/access_token`. Download data: `uv run kaggle datasets download -d <ref> -p data/raw --unzip`. Data files are git-ignored.
- xgboost/lightgbm on this Mac need `brew install libomp` (done).
- Give Me Some Credit: use `cs-training.csv` only (`cs-test.csv` is unlabeled). 150k rows, 6.7% default.
- XGBoost early stopping: `early_stopping_rounds` in constructor, `eval_set=[(X_val,y_val)]` in fit.

### Next session
- **LightGBM + CatBoost** on the same data, head-to-head with XGBoost (AUC/KS/Gini) — Phase 2 task 3.
- **Probability calibration** (Platt / isotonic): make the output a true PD, not just a good ranking.

---

## Session 5 — 2026-07-17 — Deep-dive review: understanding the XGBoost notebook (teaching-style reset)

Not new material — user said the hand-off style wasn't teaching them. Switched to showing real data
and output live in chat and explaining code line by line. See [[feedback-credit-scoring-teaching-style]].

### What we actually understood (with real data shown)
- **Seeing the data:** compared one real repayer vs one real defaulter side by side. Same age (40),
  but the defaulter earned 5x MORE income — income misled; the giveaway was payment history (defaulter
  late 3x/1x/3x, repayer never). Lesson: "signal" = the columns where good/bad actually differ.
- **Cleaning / NaN:** saw the value_counts of a past-due column decline smoothly 0->13 then jump to
  fake sentinel codes 96 (5x) and 98 (264x). Those aren't counts. `clean()` sets age==0 and the 96/98
  to `np.nan` (NaN = "honestly unknown", NOT 0 which means "never late"). Added `MonthlyIncome_missing`
  flag (missingness can be signal). Why NaN is fine: **XGBoost learns a default branch direction for
  missing values at each split** and handles them natively; logistic regression CAN'T (arithmetic
  breaks on NaN) so it needed `SimpleImputer(median)`.
- **XGBoost call = two jobs:** `XGBClassifier(...)` only configures (fills a settings form; params:
  n_estimators=ceiling, learning_rate=nudge size, max_depth, subsample/colsample, scale_pos_weight,
  early_stopping_rounds). `.fit(..., eval_set=[(X_val,y_val)])` trains tree-by-tree. Watched the live
  log: val AUC 0.805 (1 tree) -> 0.862 (50) -> plateau ~0.865; asked for 1000 trees but early stopping
  kept only 153. eval_set is the validation data it watches; those AUCs are on data not trained on.
- **predict_proba:** returns TWO numbers per person [P(repay), P(default)] summing to 1; `[:, 1]`
  = "all rows, column 1" = P(default). Scored our two borrowers: repayer PD 9.5% (didn't default),
  defaulter PD 90% (did). `predict` would hard-threshold at 0.5; `predict_proba` keeps the probability
  so the LENDER picks the cutoff (ties to Session 3).
- **Optuna:** `objective(trial)` builds+trains a model with settings Optuna proposes via
  `trial.suggest_int/float(name, lo, hi)` and RETURNS val AUC (the number to maximize). `create_study
  (direction="maximize")` + `study.optimize(objective, n_trials=N)` runs it N times, learning from past
  trials. Watched 10 trials: all landed 0.8616-0.8651 (tiny spread -> tuning barely helps here).
  Optimizes on validation, never test.

### Next session
- New material, same live style: **LightGBM + CatBoost** head-to-head with XGBoost, then **probability
  calibration**.

---

## Session 6 — 2026-07-17 — LightGBM + CatBoost + calibration (Phase 2, tasks 3 & 5), live style

Done interactively in chat (real output shown, explained as we went). Also did a hands-on exercise
first: user ranked 3 made-up applicants by default risk and reasoned correctly (income doesn't save a
bad payer; maxed-out utilization is a big risk signal even without past misses).

### Concepts learned
- **LightGBM vs CatBoost vs XGBoost** (same data, same split): XGBoost 0.8686, CatBoost **0.8689**
  (marginally best, strong out-of-the-box as advertised), LightGBM **0.8411** untuned.
- **"Defaults aren't destiny":** LightGBM lagged only because its settings were borrowed from XGBoost.
  Its growth is leaf-wise; key brakes are `num_leaves` (fewer=simpler) and `min_child_samples`
  (higher=more regularized). Optuna-tuned it jumped 0.8411 -> **0.8644**, into the pack. Trial with
  num_leaves=234 overfit (0.814); winners used num_leaves~30 + min_child_samples=200. Lesson: benchmark
  all three on YOUR data; newer/faster != automatically better.
- **All three boosters handle NaN natively** — the cleaning carried over unchanged.
- **Calibration** (the big one): boosting models RANK well but their raw probabilities can be
  dishonest. Here `scale_pos_weight`~14 (used to fight imbalance) inflated every predicted PD 2-10x.
  Showed it with a reliability table: model said 26% where reality was 2.6%; said 86% where reality
  was 37%. AUC didn't care (ranking fine) — that's why you must check calibration separately.
- **The fix:** fit a calibrator on a holdout (we used the validation set) with
  `CalibratedClassifierCV(FrozenEstimator(model), method=...)`, evaluate on untouched test.
  **Platt/sigmoid** = smooth S-curve (safer on small data); **isotonic** = free-form monotonic (best
  with lots of data). After calibration both columns sat right on "really happened" (isotonic: says
  6.9% / real 6.7%; says 36.3% / real 37.1%). **AUC unchanged (0.8686 all three)** — calibration
  rescales numbers without reordering anyone. Real trade-off learned: scale_pos_weight buys ranking at
  the cost of honest probabilities; calibration buys the honesty back for free.

### Not yet done from Phase 2
- SMOTE/ADASYN oversampling (we only did scale_pos_weight / class_weight for imbalance).
- A consolidated model-comparison notebook (this session's work is not yet saved as a .ipynb).
- Home Credit (7-table multi-table feature engineering) — the big Phase 2 finale / bridge to DL.

### Next session (options)
- Consolidate Session 5-6 live work into `notebooks/05_model_comparison_calibration.ipynb`.
- OR start **Home Credit** multi-table feature engineering.
- OR begin **Phase 3 deep learning** (PyTorch, MLP from scratch).

---

## Phase 2 CLOSED — 2026-07-19

Notebook 05 built, executed, committed. User chose to end Phase 2 here (leaving Home Credit for later).

**What Phase 1 + 2 delivered (notebooks 01-05):**
- Full classical credit-scoring pipeline on tabular data, learned end to end.
- German Credit: EDA, WoE/IV, logistic-regression scorecard (AUC 0.80).
- Give Me Some Credit: cleaning messy data, XGBoost (0.869) beating logistic regression (0.820),
  Optuna, the depth/overfitting lesson.
- Three-way booster comparison (XGBoost ~= CatBoost 0.869; LightGBM 0.841 untuned -> 0.864 tuned).
- Probability calibration (Platt/isotonic) to turn good rankings into honest PDs.
- Recurring principles internalized: accuracy lies on imbalanced data (use AUC/Gini/KS); never leak
  (learn encodings/bins/tuning/calibration on train only); model gives a number, business sets the
  threshold; ranking != calibrated probability; honesty over hype.

**Consciously DEFERRED (can revisit anytime):**
- SMOTE/ADASYN oversampling vs scale_pos_weight (small topic).
- Home Credit 7-table aggregation (data is downloaded at data/raw/home_credit/, 307k applicants,
  7 tables, ~2.5GB). This is the "junior->senior" feature-engineering skill when we want it.

### Next: Phase 3 — Deep Learning
- PyTorch fundamentals: tensors, autograd, computation graph.
- Build an MLP from scratch, full training loop, on data the user already knows (Give Me Some Credit).
- Honest comparison: does a neural net beat XGBoost on static tabular features? (Usually not, that's
  the point, and it motivates why LSTMs on payment SEQUENCES are where DL actually wins.)
- Keep the live teaching style: show data/output in chat, explain line by line, small steps.

---

## Session 7 — 2026-07-19 — Home Credit multi-table feature engineering (Phase 2 finale), live style

User chose "Home Credit first, then DL", then "one more round to watch it compound." Notebook 06 built
and executed. Done live in chat with real output.

### Concepts learned
- **The 7-table / one-to-many problem:** main table `application_train` (307k applicants, 1 row each,
  8.1% default). Child tables have MANY rows per applicant (bureau: 1.7M rows = past loans at other
  lenders, ~5-6 each). XGBoost needs 1 row per person, so we must aggregate.
- **The core skill = `groupby -> flag -> agg -> merge`:**
  - `groupby("SK_ID_CURR")` buckets each applicant's child rows.
  - `.agg(name=(col, func))` collapses each bucket to summary numbers (count/sum/mean/max/min).
  - To aggregate a CATEGORY, make a 0/1 flag first (e.g. is_active = CREDIT_ACTIVE=="Active") then sum.
  - `merge(..., how="left")` attaches features to the main table, keeping all 307k applicants; no
    history -> NaN (XGBoost handles natively). 44,020 applicants had no bureau history.
- Traced applicant 100047: 5 bureau rows collapsed to 1 row of features (loan_count 5, active 3,
  total_debt ~3.22M); they defaulted.
- **Feature engineering COMPOUNDS (the payoff test):** application numerics only AUC 0.7542;
  +6 simple bureau feats 0.7560; +17 richer bureau feats 0.7577; +16 previous_application feats 0.7652.
  Total +0.011 from 2 of 6 child tables. previous_application (own history with this lender) lifted
  more than bureau. Path to brief's 0.78+ = all 6 tables, hundreds of features. Technique is simple;
  THOROUGHNESS is the skill (junior vs senior).
- Honest note: single-table lift is small (app table already has strong EXT_SOURCE_* bureau-score
  columns); the value is cumulative across tables.

### Phase 2 truly complete now (notebooks 01-06).
### Next: Phase 3 deep learning (PyTorch, MLP from scratch, MLP-vs-XGBoost honest test).
