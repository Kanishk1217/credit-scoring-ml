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

---

## Session 8 — 2026-07-25 — Deep Learning foundations: neuron → MLP (Phase 3), live style

Built from the ground up, live in chat. Notebook 07 saved. Used the pytorch-patterns skill.
User also banked a FUTURE idea: risk-based pricing (loan amount + interest rate, not just approve/
reject) — deferred until after DL. Recorded in project memory.

### Concepts learned (deeply, from scratch)
- **A neuron = logistic regression:** weighted sum (z = w·x + b) then sigmoid squash to a probability.
  Weights=coefficients, bias=intercept, sigmoid=the PD map from the scorecard. Built one neuron with
  made-up then real numbers.
- **How a network learns = one 5-step loop:** forward (predict) -> loss -> gradient -> update -> repeat.
- **Loss = binary cross-entropy:** rewards confident-correct (tiny loss), punishes confident-wrong
  (big loss). Untrained neuron predicts ~0.5, loss ~0.693 = -ln(0.5) = the starting line.
- **Ranking vs loss (callback to calibration):** an untrained neuron with tiny random weights had
  AUC 0.81 (lucky ranking) but loss 0.69 (useless probabilities). We train on LOSS because it cares
  about the actual probability values, not just order.
- **Gradient descent (the one new idea):** gradient = slope of loss vs each weight; step OPPOSITE
  (downhill); learning_rate = step size. For a neuron: grad_w = mean(error × input), error = p - y.
  Built it from scratch in numpy: loss 0.69 -> 0.60 (1 step) -> 0.20 (converged). Our weights matched
  sklearn LogisticRegression EXACTLY ([0.752,-0.25,0.411], bias -3.131). We built LR from nothing.
- **PyTorch = the same loop, mechanized:** nn.Linear(3,1)=one neuron, BCEWithLogitsLoss=sigmoid+CE,
  `loss.backward()` = AUTOGRAD (computes grad_w automatically) = **backpropagation** (chain rule
  through layers). PyTorch neuron matched from-scratch exactly.
- **Hidden layers + ReLU = a real MLP:** stacking Linear layers alone stays linear; ReLU (max(0,x))
  adds the kink that lets it learn curves/interactions. Built Linear->ReLU->Linear->ReLU->Linear.
- **MLPs need impute + standardize** (can't eat NaN or raw scales), unlike XGBoost.
- **HONEST FINALE:** on Give Me Some Credit static features, XGBoost **0.869** beats MLP **0.837** by
  ~0.03. Trees win on tabular data (mixed scales, NaN, thresholds, interactions, little tuning). This
  is the field consensus. **The lesson & project thesis:** use DL where data has structure a tree
  can't see (images, text, SEQUENCES). Motivates the LSTM on 12-month payment sequences next.

### Env gotcha logged: torch + xgboost segfault in same process (OpenMP). Run in separate processes.

### Next: LSTM/GRU on payment sequences — where deep learning finally wins. Then the hybrid model.

---

## Session 9 — 2026-07-25 — LSTMs & sequences (Phase 3), live style

Notebook 08 saved. Downloaded Taiwan Credit Card (data/raw/taiwan_credit/UCI_Credit_Card.csv, 30k
accounts, 6 months of payment status per account, 22% default). Built from the ground up.

### Concepts learned
- **Why sequences need a different model:** the 6 monthly PAY columns carry signal in their ORDER
  (improving vs deteriorating), which XGBoost/MLP treat as unordered columns and can't fully see.
  Showed 2 borrowers with the SAME payment values in different order -> opposite outcomes.
- **Recurrent memory (hand-built tiny RNN):** keep a running hidden state h, update each step
  `h=tanh(wx*x + wh*h)`. The final memory summarizes the trajectory, weighting recent months more.
  Demo: improving borrower final memory -0.77 (safe), deteriorating +0.86 (risky); REVERSING the
  same values flipped it to -0.92. Order is the signal; memory captures it.
- **LSTM = robust RNN:** adds a cell state + 3 gates (forget/input/output = sigmoid 0-1 dials) to hold
  long-range info and avoid vanishing gradients.
- **Built an LSTM in PyTorch:** `nn.LSTM(input_size, hidden_size, batch_first=True)`; input shape
  `(batch, timesteps, features)` = (N, 6, 1); take the FINAL hidden state `hn[-1]` -> Linear -> logit.
  Tiny model (4,513 params), trained 25 epochs.
- **HONEST result:** LSTM test AUC **0.737** vs XGBoost on the same 6 PAY cols (static) **0.741** —
  XGBoost still edges it. Why: 6 months is SHORT (most signal is in the single most-recent month, which
  the tree exploits) and LSTMs are data-hungry. **Sequential data alone doesn't make the LSTM win; the
  sequences must be long/rich enough that trajectory beats the latest snapshot.**
- Where the LSTM pays off: longer/richer sequences (Home Credit installments, 12+ months, multi-feature)
  and especially the HYBRID (XGBoost static + LSTM temporal embedding) = project headline model.

### Next: the HYBRID model (fuse XGBoost static score + LSTM temporal embedding in a final MLP).

---

## Session 10 — 2026-07-25 — The HYBRID model (Phase 3-4 headline), live style

Notebook 09 + src/make_hybrid_features.py saved. The culmination: fuse static ML + deep learning.

### Concepts learned
- **Hybrid architecture:** static features -> XGBoost -> risk score (scalar); payment sequence ->
  LSTM -> 32-dim temporal embedding; **concatenate** [score, embedding] -> small MLP -> PD.
- **Why fuse:** the two branches carry COMPLEMENTARY info. XGBoost sees the static snapshot (limit,
  age, bill/pay amounts); the LSTM sees the payment TRAJECTORY. Neither alone has both.
- **Leakage care (ties to earlier lessons):** XGBoost scores for TRAIN rows must be out-of-fold
  (`cross_val_predict`, 5-fold) so the fusion MLP never trains on a score that saw its own label.
  Val/test scores from the model fit on full train.
- **Two-process pattern (OpenMP):** src/make_hybrid_features.py (xgboost) writes
  data/processed/hybrid_feats.npz; notebook 09 (torch) loads it and trains the fusion. Never import
  torch + xgboost in one process.
- **RESULT (Taiwan test AUC):** XGBoost-static 0.7298, LSTM-sequence 0.7365, **HYBRID 0.7750**
  (+0.0385 over best single, and beats the 0.741 XGBoost-on-PAY-cols from nb08 — best model in the
  project so far). The hybrid genuinely won.
- **The project's central thesis, proven by hand:** static ML and deep learning are COMPLEMENTARY,
  not competitors. Use trees for tables, LSTMs for sequences, and FUSE them.

### Deep-learning arc complete (notebooks 07-09): neuron -> MLP -> LSTM -> hybrid, all from scratch.

### Remaining project phases (not yet done):
- Home Credit installments: longer/richer sequences to grow the LSTM/hybrid edge.
- Alternative-data / thin-file model (Model 4).
- Explainability (SHAP/LIME), fairness audit (Fairlearn) — Phase 5.
- FastAPI serving + Streamlit dashboard + deployment — Phase 6.
- FUTURE (user idea): risk-based pricing (loan amount + interest rate). See project memory.

---

## Session 11 — 2026-07-25 — Deployment: serving the hybrid as a live API (Phase 6 start)

User was confused about "what to do with a trained model after train/test/accuracy." This session
resolved it concretely. Deployed the HYBRID (user's choice, the harder one).

### The key concept (finally made concrete)
- **A trained model is just a FILE.** Training's whole output is that file. Everything after is
  loading the file and calling it on new data. We saved: models/hybrid_xgb.joblib (XGBoost),
  models/hybrid_fusion.pt (LSTM+fusion), models/hybrid_config.json (preprocessing recipe).
- **The deployment lifecycle:** finalize -> SAVE to file -> bundle preprocessing -> wrap in an API
  (load file once at startup) -> new raw input hits endpoint -> preprocess -> predict -> return PD ->
  monitor for drift.

### What we built
- src/hybrid_model.py: shared Hybrid nn.Module (so training + serving use the same architecture).
- src/make_hybrid_features.py: xgboost branch, saves model + config + OOF features (run first).
- src/train_fusion.py: torch branch, trains + saves the fusion net (run second).
- api/app.py: FastAPI. Loads the 3 files once, `/predict` endpoint: raw applicant JSON -> preprocess
  (static features in training order -> XGBoost score; standardize sequence) -> fusion net -> PD +
  approve/review/decline recommendation (thresholds = lender decision, Session 3 lesson).
- Tested LIVE: deteriorating applicant PD 0.776 (decline), healthy applicant PD 0.204 (review).

### Engineering lessons
- **OpenMP segfault SOLVED for serving:** set OMP_NUM_THREADS=1 + KMP_DUPLICATE_LIB_OK=TRUE at top of
  app.py BEFORE imports, and import xgboost BEFORE torch. Then one process serves both. (Training
  notebooks stay torch-only.)
- Port 8000 was taken by another of the user's apps -> used 8077. `uvicorn api.app:app --port 8077`,
  interactive docs at /docs.

### Next: push to GitHub, then deploy to Render (public URL). Model-artifact strategy needed (models/
are git-ignored; either commit the small files or regenerate on deploy).

---

## Session 12 — 2026-07-25 — Production hardening + GitHub (Phase 6), used fastapi-patterns skill

User: "don't just do security/auth/rate-limiting, fill the whole pipeline and make it perfect." Built
the full production backend.

### What we built (beyond the basic API)
- **Security:** API-key auth (`X-API-Key`, constant-time `secrets.compare_digest`), per-key rate
  limiting (slowapi), strict Pydantic validation (field bounds + custom validators), locked-down CORS
  (exact origins), security headers (nosniff/DENY/no-referrer/no-store), non-leaking error handlers.
  All secrets from env via pydantic-settings (env_prefix CREDIT_). .env gitignored, .env.example committed.
- **Ops:** request-ID + latency logging middleware, audit log of every decision (regulatory trail),
  `/predict/batch` (up to 1000 applicants).
- **Quality:** pytest suite (tests/test_api.py: health, auth 401, validation 422, scoring, batch,
  headers — 7 tests pass), ruff lint (config in pyproject, notebooks excluded), GitHub Actions CI
  (.github/workflows/ci.yml: uv sync + ruff + pytest on push).
- **Deploy:** Dockerfile (+ .dockerignore) and render.yaml. Committed the 3 small hybrid model files
  (removed from gitignore) so CI + deploy have them.
- **Docs:** docs/model_card.md (intended use, perf, limitations, fairness), rewritten README.md.
- Config split: api/config.py (Settings), api/app.py (app). api/__init__.py added.

### GitHub
- gh had TWO accounts (active niharagility21, plus Kanishk1217). Switched active -> Kanishk1217.
- Kanishk1217's keyring (fine-grained) token could NOT create repos; user supplied a classic PAT
  (full scopes incl repo+workflow). **Repo: https://github.com/Kanishk1217/credit-scoring-ml (private).**
- NOTE: active gh account is now Kanishk1217 (was niharagility21). Pushes to this repo need Kanishk1217
  active. User should ROTATE the pasted PAT.

### Next: UI/UX discussion + frontend (loan-officer dashboard). Then Render deploy for a public URL.
User wants to think through functionality + how banks/creditors would use it.

---

## Session 13 — 2026-07-25 — Production pipeline direction + synthetic data (stage 1)

User asked how to overcome the "not for real lending" limitation and how real applicants get used.
Explained: scoring new people already works (that's the API); the gap is TRUST, which needs a
lender's own outcome data + retraining + calibration + fairness + shadow-pilot + drift monitoring.
User chose direction: **"full production pipeline on simulated-at-scale data"** — build the real
machinery so swapping sim data for a real lender's book is a one-line change.

Also: user unhappy with the homepage design ("does not look good, use better components"). Chose to
CONNECT CHROME so I can iterate visually with shadcn/ui. **Frontend is PAUSED until Chrome is
connected** (extension currently not connected — can't screenshot). Plan: rebuild with shadcn/ui.

### Production-pipeline roadmap (agreed)
1. Synthetic data at scale (DONE this session) 2. Config-driven training pipeline 3. SHAP
explainability -> API + frontend 4. Fairness audit 5. Drift monitoring (PSI) + retrain trigger
6. Model registry/versioning.

### Stage 1 built: src/synth_data.py
- generate(n, seed, months=12) -> realistic lender book: static financial features + 12-month payment
  sequence + default. Schema mirrors real data (data-source-agnostic pipeline).
- Design: TWO semi-independent latents — `fin` (financial risk, observable from static) and `beh`
  (payment discipline, only revealed by the sequence). Sequence generated with momentum. Default
  depends on BOTH -> static & sequence are COMPLEMENTARY.
- Verified (100k rows, 22.4% default): static AUC 0.729, sequence 0.834, static+sequence **0.886**.
  Hybrid genuinely beats both, like real credit data.

### Next: stage 2 — config-driven training pipeline (data -> features -> hybrid -> calibrate ->
version), runnable on sim data now and real data later by swapping the loader.

---

## Session 14 — 2026-07-29 — Model reliability fix: calibration, protected attrs, fairness audit

User pasted a review of the DEPLOYED (Taiwan) model flagging: protected attributes scored (sex/
education/marriage — illegal for real lending), no calibration, thin data (30k/6mo), no explanations.
User: "why is model accuracy only 47%... data is very imbalanced... understand everything and make
this model more reliable." Investigated directly rather than guessing.

### The real bug, proven on the deployed model (held-out test set)
- Actual default rate 22.1%, but MEAN PREDICTED PD was 43.4% — ~2x inflated. Root cause:
  `scale_pos_weight` (used to fight class imbalance during training) systematically inflates
  probabilities, and the deployed model was never recalibrated afterward (we'd taught this exact
  failure mode as a lesson back in Session 6 but never applied the fix to what was actually live).
- Using the shipped hardcoded thresholds (decline>=.5, review>=.2): 91% of ALL applicants got
  flagged (8.7% approve / 61.8% review / 29.5% decline) despite true default rate being 22%.
  AUC itself was fine (0.776) — ranking wasn't the problem, the raw probabilities were.
- User decisions (confirmed): KEEP age as a feature (legal, predictive, will be fairness-audited);
  do NOT use SMOTE/resampling (breaks calibration, doesn't help AUC — class weights + calibration +
  cost-based thresholds is the correct fix); cost ratio 5:1 (approving a defaulter judged 5x worse
  than declining a good customer).

### What we built (full rewrite of the served model + API)
- **src/train_synth_model.py**: single pipeline — 150k synthetic rows (12mo, no protected attrs) ->
  4-way split (train/val/cal/test) -> XGBoost (OOF scores, no leakage) -> LSTM+fusion (PyTorch,
  training only) -> **isotonic calibration fit on `cal`** -> **cost-based thresholds (5:1) fit on
  `cal`** -> **fairness audit on `test`** (gender/region, held out of scoring) -> torch-free export
  (NumPy weights, verified match to 1.19e-07) -> honest metrics/thresholds/fairness saved into
  hybrid_config.json.
- **Result: AUC 0.775 -> 0.892** (bigger, richer data). Calibration: mean predicted PD 34.9% ->
  **22.0%** (actual 22.4%) — reliability table matches within a few points at every decile (was 2x
  off before). Decision mix at cost-optimal thresholds: 45.7% approve / 15.1% review / 39.2% decline
  (sane, vs old 8.7%/61.8%/29.5%).
- **Fairness audit**: gender demographic-parity gap 0.000, region gap 0.015 — both near zero.
  Documented honestly: this is EXPECTED on synthetic data (generator doesn't make gender/region
  causally affect risk) — proves the audit methodology works, not a real-population guarantee.
- **api/scoring.py** (new module, pure functions, business logic separated from routing):
  `calibrate()` (np.interp over the isotonic grid), `decide()` (cost-based thresholds, collateral
  widens the decline cutoff via the Bayes-optimal ratio (1+C)/(1+C*(1-recovery))), `explain()`
  (XGBoost `pred_contribs` SHAP for static features, converted to probability-space via the local
  sigmoid derivative p*(1-p); payment-history counterfactual for the sequence branch — actual vs
  all-on-time), `price_loan()` (risk-based rate + affordability-capped amount, collateral improves
  terms), `advice()` (re-scores what-if scenarios: 6mo on-time, debt -25%/-50%, income +20%, ranked
  by PD improvement).
- **api/app.py**: new `Applicant` schema — age, monthly_income, credit_limit, existing_debt,
  employment_years, num_existing_loans, pay_status (12 months), has_collateral, requested_amount.
  **`model_config = {"extra": "forbid"}`** — protected attributes are structurally impossible to
  submit, not just unused. `/predict` and `/predict/batch` now return probability_of_default,
  recommendation, approve/decline_threshold, why (ranked factors), pricing, advice.
- **frontend/src/LiveDemo.tsx**: rebuilt for 12 months (was 6), new fields, added a collateral
  toggle that shows the pricing engine live (rate/amount change in the demo).
- **Verified end to end**: risky applicant (escalating lateness, high DTI) -> PD 96.6%, decline,
  top factor "Recent payment history"; healthy applicant -> PD 0.9%, approve, offer ₹400k @ 11.37%;
  collateral confirmed to flip decisions in the borderline PD window (0.095: review -> approve) and
  never worsen terms.
- **Tests**: tests/test_scoring.py (new, unit tests on scoring.py incl. a regression test that
  calibration must shrink the gap to true default rate on fresh held-out data — guards against this
  exact bug recurring) + tests/test_api.py rewritten for new schema (healthy-vs-risky ordering,
  calibration sanity, collateral never hurts, advice always improves PD, protected attrs rejected
  with 422). 22/22 tests pass, ruff clean.
- Cleaned up stale models/hybrid_fusion.pt (old Taiwan torch artifact, no longer produced).
- Recovered 4 of 7 design specs (pricing-engine, dashboard-ux, consumer-ux, advice-engine) from a
  workflow that hit an org spend limit mid-run; saved to docs/specs/ for the next phase (3 specs —
  explainability, provisioning, frontend-arch — did not complete and were designed fresh here instead
  where needed).

### Next: the paused feature build — loan-officer dashboard, consumer self-assessment page, and
self-serve API-key sign-up (specs cached in docs/specs/). Model reliability work is done and tested.

## Session 15 — 2026-07-30 — Real-data AUC gap, fairness mitigation, and the full 5-phase build

User pushback mid-session, verbatim: "So we caught 1 of 7 defaulters and was the data any good to
start with... focus on model first all depends on it." Then: "Just do all 5 [phases]... focus on
model data cross validation test train... go in loop till everything is perfect... test test
refine test verify go in this loop and dont stop." Executed all 5 phases in order without stopping.

### Phase 1 — closed the real-data AUC gap, three rounds of feature engineering
- Base real model (7 features, application_train + bureau + installments): AUC 0.6655.
- +bureau_balance/POS/credit-card/installments aggregates (20 feat): AUC 0.6898.
- +previous_application, deeper bureau_balance (35 feat): AUC 0.7092.
- +EXT_SOURCE_1/2/3 (38 feat, largest single jump): **AUC 0.7564**. EXT_SOURCE is Home Credit's own
  external bureau-like risk score, already in application_train.csv, standalone AUC 0.66-0.68 each —
  left as real NaN (56%/0.2%/20% missing) since XGBoost handles missing values natively; imputing a
  strong predictor risks manufacturing a fake signal.
- Verified with proper 5-fold CV (full retrain per fold, no leakage): AUC mean 0.7560, std 0.0044 —
  genuinely stable, not a lucky split. Recall improved 0.137 -> 0.361 at the 5:1 cost threshold.
- For reference, Home Credit's Kaggle winners hit ~0.79-0.80 with hundreds of features + stacking;
  0.756 with 38 features and one model family closes ~90% of that gap.

### The fairness cost of EXT_SOURCE, root-caused and mitigated with real numbers
- Adding EXT_SOURCE grew the fairness gaps 3-8x (gender demographic-parity 0.0055 -> 0.0454, region
  0.023 -> 0.1026). Root cause, checked directly: EXT_SOURCE_1 genuinely differs by gender in the
  raw data (mean 0.546 women vs 0.407 men) and by region — and actual default rates differ the same
  direction (women 7.0% vs men 10.1%), so part of the gap is the model correctly learning a real
  base-rate difference, not fabricating one.
- Isolated the part NOT explained by real risk difference via equalized-odds gap (conditions on
  actual outcome): still substantial, 0.0344 gender / 0.0775 region.
- Mitigation tested (no retraining — reused saved artifacts): per-group decline thresholds
  equalizing each group's TPR-on-good-borrowers to the population rate. Cut both gaps ~77% for
  under 1.5 points of overall recall. Validated finding, not yet wired into production scoring.

### Model promotion path — decided and made configurable, not hardcoded
- `api/app.py` now reads `settings.model_dir` (env `CREDIT_MODEL_DIR`) instead of a hardcoded path.
  `models/` (synthetic) stays default to protect the existing synthetic-calibration regression test;
  `models_real/` (7-feat, honest, self-reportable) is the intended public API / consumer-page model;
  `models_real_rich/` (38-feat, needs bureau/EXT_SOURCE data) is internal-loan-officer-only, since a
  new self-service applicant can't supply those fields themselves.

### Phase 2 — loan-officer dashboard at /officer
- Single-applicant scoring (form + 12-month payment grid + decision panel with gauge/why-list/
  pricing/actions) and CSV batch (drag-drop, validation, summary tiles, sortable table with
  per-row drill-down), per docs/specs/dashboard-ux.md. New backend `/score` + `/score/batch`
  reusing scoring.py's decision logic, different response shape.
- Verified with real Playwright (not just typecheck) against the live dev servers — found and fixed
  two real bugs this way: the gauge's "breakeven" tick label overlapped the zone labels below it,
  and the batch summary's band histogram always highlighted band E instead of the tallest bar.

### Phase 3 — consumer self-assessment page at /advisor
- Manual entry + CSV upload converging on one Profile payload; warm band framing (thriving/steady/
  almost/building/starting — never "reject"/"denied"); new `api/advice_engine.py` with an offer-
  pricing formula, WHY factors, and an advice/goal engine that RE-SCORES real what-if profiles
  through the actual model (on-time streaks, debt paydown, limit increase) rather than estimating
  deltas — every number shown to the user came from a real forward pass, verified by hand: ontime
  3/6/12-month scenarios differentiate correctly on the real model (0.127->0.121->0.090->0.058)
  even though the bundled synthetic demo model was flat at one specific test point (a genuine model
  characteristic, confirmed by direct comparison, not a code bug).

### Phase 4 — hard-override policy layer
- `scoring.check_hard_override()` force-declines three extreme/unambiguous cases regardless of
  model output: severe recent delinquency (3+ months at 6+ months past due in the trailing 6),
  debt > 3x annual income, and an implausible credit-limit-to-income ratio. Wired into all three
  serving surfaces so the guard applies everywhere, not just one endpoint. Both frontends surface
  the override reason transparently rather than silently overriding.
- Caught a real test-scenario bug during Playwright verification (not a code bug): placed severe
  lateness in the middle of a 12-month array instead of the trailing 6-month window the check
  actually examines — confirmed by moving it into the recent window and re-testing.

### Phase 5 — model registry
- `src/build_model_registry.py` catalogs every trained model directory into `model_registry.json`:
  data source, metrics, fairness gaps, and a sha256 **fingerprint of the actual trained artifact
  files** (not the raw data) — changes iff the model itself changed. `GET /` now reports the
  fingerprint/commit of whatever's currently loaded; new `GET /model-registry` returns the full
  catalog. Answers "what is this API serving right now and how was it built" without tribal
  knowledge.

### Process note
Every phase was lint-checked (ruff + oxlint), test-checked (pytest, 54/54 passing by the end,
tsc -b clean), and — for the two UI phases — actually exercised in a real headless Chromium via
Playwright (installed mid-session since the Claude-in-Chrome extension wasn't connected), not just
type-checked. That's how both real UI bugs above were caught. Every phase committed and pushed to
GitHub individually rather than batched at the end.

### Next: the fairness mitigation (per-group thresholds) is validated but not yet wired into
production scoring — that's the highest-leverage remaining reliability gap. After that: self-serve
API-key sign-up (spec not yet written) and deciding whether/how to combine gender+region group
thresholds (tested independently so far, not simultaneously).

## Session 16 — 2026-07-30 (later same day) — Production fixes, a new-market onboarding
pipeline, and consolidating the Home Credit training scripts

Same day as Session 15, separate sitting. User reported three things broken on the live
deployed site (screenshots): the homepage live demo showed a dev-only "start the API on :8077"
message, `/officer` returned 401 "unauthorized", `/advisor` returned "invalid or missing API
key". Root cause for all three: the same Cloudflare `API_KEY` secret / Render `CREDIT_API_KEYS`
mismatch. Fixed: generated a new shared key, set it via `wrangler pages secret put` (I have
Cloudflare CLI access, not Render dashboard access — user pasted the Render side themselves).
Also fixed along the way: `/docs` wasn't actually disabled in production (added a real
`is_production` check, previously dead code); the Cloudflare Pages Function proxy silently
dropped every response header except `content-type`; `/model-registry` had no auth requirement
while every other endpoint did; Dockerfile only shipped `models/` to Render, not the registry
file or the other two model dirs.

### User asked: "how good is the model, honestly — find me real lender data to test it on"
Explicitly rejected reusing Home Credit/Taiwan/German/Give-Me-Some-Credit ("we have used these
datasets ALL THE TIME") — wanted something genuinely untouched. Found and downloaded historical
LendingClub data (39,717 real US loans, resolved outcomes) via web search, a completely
independent source. Built a balanced 300-row test batch (honesty ledger: age is NOT in US
lending data, so it was imputed and flagged; the 12-month payment sequence was approximated from
real aggregate delinquency fields, not a literal feed — everything else, including the ground
truth, is real). Result: **AUC 0.5079** on `models_real` — no signal at all. User then asked for
60-70% "accuracy" — had to correct this: on this imbalanced data, a naive "approve everyone"
baseline already gets 91.9% accuracy, so a 60-70% accuracy target is actually WORSE than doing
nothing. What they actually wanted was 60-70% *recall* (catch rate) — computed the real
threshold-sweep table and let them pick an operating point from real numbers, not a vague
promise. Applied a recall-target (~65%) threshold policy to `models_real`/`models_real_rich` at
their explicit choice, with the real cost shown up front: auto-approve rate collapsed from 64%
to 14% of applicants.

### The generalization question — plan mode, adversarial critique, then build
User then asked "what would it take to fix that" (the 0.51 AUC) and, after I showed the AUC was
still ~0.51 even after quantile-matching income/debt scale to Home Credit's distribution (proving
it wasn't a units problem), asked for a *generic* onboarding pipeline instead of one-off scripts,
critiqued and planned first. Used `EnterPlanMode` + a `Plan`-type agent to adversarially review my
own draft plan before writing any code — the critique agent caught real things I'd gotten wrong:
no `recall_target` precedent actually existed in the training scripts (it was a manual JSON
patch, not tested code — dropped a planned dual-policy flag as a result), a two-digit-year date
bug in `earliest_cr_line` (confirmed 90 rows parsed to a date after loan issuance), a missing
`loan_status` filter assertion, and that the registry's fingerprint function would hard-crash
existing tests once a non-hybrid model dir was added (it was — fixed before it happened, not
after). Built `src/train_new_market.py` (one file: engineer → train → calibrate → CV → fairness
→ registry), trained `models_lendingclub` on it (plain XGBoost, deliberately no LSTM — this
market has no real payment-sequence data, so building a fake one would inject fake signal, not
real). Result: AUC 0.6608 (CV-stable 0.6577±0.0060), and the SAME 300 real applicants that broke
`models_real` scored **AUC 0.7460** through the new model — the concrete proof that population-
specific models are the actual fix, not a "better" universal architecture.

### "Combine everything into one file" — pushed back on the notebooks, agreed on the scripts
User asked to merge all 10 notebooks + all training scripts into one file "since that's more
optimized." Pushed back explicitly: notebooks are 10 different concepts each with real rendered
output, not duplicated code — merging would delete their reference value, not optimize anything.
Agreed the 5 Home Credit training scripts (train_synth_model.py, train_real_data_model.py,
train_real_rich_model.py, cross_validate_real.py, fairness_mitigation_real_rich.py) genuinely
were ~90% duplicated structure across 3 variants and were worth consolidating. User said yes to
the narrowed scope. Built `src/train_home_credit_models.py`, retrained all three variants, and
**verified every model's artifact fingerprint was byte-for-byte identical to before** — proof the
consolidation changed zero model behavior. In the process, found a real, unrelated bug: when the
recall-target thresholds were applied to `models_real`/`models_real_rich` earlier this session,
the config's `fairness_audit` and `metrics` sections were hand-patched for `thresholds` but never
recomputed — they'd been silently describing the OLD cost-based decision boundary ever since.
Honest current fairness numbers: models_real gender gap 0.0055 → **0.0617**, models_real_rich
0.0454 → **0.0827** (same model, same test set, only the active threshold differs — verified
directly by re-scoring at both thresholds). `models_real` also got 5-fold CV for the first time
(it never had one before). Deleted the 5 superseded scripts; nothing outside that cluster
imported them (verified via grep before deleting, not assumed).

### Next: wire the validated fairness mitigation and one of the un-served models
(`models_real_rich` or `models_lendingclub`) into actual production scoring — both are trained,
validated, and registered, neither is servable through the current API without a new request
schema. Re-verify the fairness-mitigation's ~77% relative-reduction claim against the newly
corrected (larger) baseline gaps before relying on it.
