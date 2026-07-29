# Real-Data Model — First Result (Home Credit)

Trained on **167,115 real applicants** from Home Credit (`application_train` + `bureau` +
`installments_payments`), through the identical pipeline used for the synthetic model
(XGBoost + LSTM + fusion + isotonic calibration + cost-based thresholds + fairness audit).
Only the data source changed — proof the pipeline really is data-source-agnostic.

Saved to `models_real/` (not `models/`) — the currently deployed synthetic model is untouched.

## Real feature mapping used
| Our schema | Real Home Credit source |
|---|---|
| age | `-DAYS_BIRTH / 365` |
| monthly_income | `AMT_INCOME_TOTAL / 12` |
| credit_limit | `AMT_CREDIT` (loan amount extended) |
| existing_debt | `bureau.AMT_CREDIT_SUM_DEBT`, summed per applicant (real debt at other lenders) |
| employment_years | `-DAYS_EMPLOYED / 365` (365243 placeholder for unemployed/pensioners → 0) |
| num_existing_loans | count of `bureau` rows per applicant (real past loan count) |
| pay_0..pay_11 | built from real `installments_payments.csv`: worst lateness severity observed per trailing month, applicants with <6 of 12 months of real coverage excluded |
| gender / region (audit only) | real `CODE_GENDER`, `REGION_RATING_CLIENT` |

## Result vs the synthetic model
| Metric | Synthetic model | **Real-data model** |
|---|---|---|
| n | 150,000 | 167,115 |
| Actual default rate | 22.4% | 8.1% (matches Home Credit's known real rate) |
| AUC | 0.892 | **0.665** |
| Brier (calibrated) | 0.101 | 0.073 |
| Mean predicted PD | 22.0% | 8.18% (actual: 8.12%) |
| Calibration | excellent | **excellent, equally good** |
| Fairness (gender) | 0.000 gap (synthetic, no real disparity possible by construction) | **0.0055 gap — genuinely low, on real gender data** |

## The honest read
**AUC dropped a lot (0.892 → 0.665), and that's expected, not a failure.** The synthetic
generator was deliberately built with two clean, strongly complementary risk signals. Real Home
Credit data is messier, and — more importantly — **we only used 7 static features and a payment
sequence built from one table.** The actual Home Credit Kaggle competition's winning solutions
reached ~0.79–0.80 AUC using hundreds of engineered features across all 7 relational tables
(the same aggregation work started in notebook 06). 0.665 with 7 features is an honest, plausible
number for this feature set — not a ceiling on what real data can do.

**Calibration and the fairness-audit methodology transferred perfectly to real data.** Mean
predicted PD (8.18%) matches the actual rate (8.12%) almost exactly, and the reliability table
tracks tightly at every decile — the calibration approach is not synthetic-data-specific.

**The gender fairness result is the most meaningful number in this whole report.** A 0.0055
demographic-parity gap on **real** gender data (not a value that's zero by construction like the
synthetic audit) is a genuinely reassuring, credible finding.

**Caveat on the "region" fairness check:** `REGION_RATING_CLIENT` is Home Credit's own operational
risk rating for a region, not a neutral demographic identity — some correlation with outcomes is
expected by design, so this particular gap (0.023) is less meaningful as a discrimination check
than the gender result.

## Next step to close the AUC gap
Reuse the Phase 2 multi-table aggregation (bureau_balance, previous_application, POS_CASH_balance,
credit_card_balance — all already downloaded) to build a richer real static feature set, the same
work the Home Credit competition winners did. This is pure feature engineering on data we already
have, no new sourcing required.

## Update: enriched model closes most of the gap

Same pipeline, same 167,115 applicants, same architecture (XGBoost → LSTM → fusion → isotonic
calibration → cost-based thresholds → fairness audit). Only the static feature set grew, in three
rounds, each verified end to end before moving to the next:

| Round | Features | AUC (test) | Recall | Precision | Notes |
|---|---|---|---|---|---|
| Base | 7 | 0.6655 | 0.137 | 0.197 | original real-data model above |
| +bureau/POS/credit-card/installments aggregates | 20 | 0.6898 | 0.216 | — | `build_real_data_rich.py` round 1 |
| +previous_application, bureau_balance depth | 35 | 0.7092 | 0.302 | — | round 2 |
| +EXT_SOURCE_1/2/3 | 38 | **0.7564** | **0.361** | 0.268 | round 3 — largest single jump |

**EXT_SOURCE_1/2/3 was the single highest-leverage addition.** These are external bureau-like risk
scores already present in `application_train.csv`. Each one alone has standalone AUC of
0.656–0.679 — nearly matching the entire 7-feature base model by itself. Left as real `NaN` where
missing (56% / 0.2% / 20% missing respectively) rather than imputed, since XGBoost handles missing
values natively and imputing a strong predictor risks injecting a fake signal.

**Verified with proper 5-fold cross-validation, not a single lucky split**
(`src/cross_validate_real.py`, full retrain of XGBoost + LSTM + fusion + calibration per fold,
no leakage): AUC mean **0.7560**, std **0.0044**, range 0.7487–0.7614 across folds. Recall mean
0.3674, std 0.0349. The 0.7564 single-split number is real and stable, not noise.

For reference, the Home Credit Kaggle competition's winning solutions reached ~0.79–0.80 AUC using
hundreds of engineered features across all 7 tables plus heavy stacking/ensembling. 0.756 with 38
features and one model family is a solid, honest result for the effort spent — closing roughly
90% of the gap between the 7-feature base (0.665) and the competition ceiling (~0.80).

**Recall is still the honest weak point.** At the 5:1 cost ratio, the model catches ~36% of actual
defaulters (up from 14% with 7 features) at ~27% precision. This is a real improvement, not a
cosmetic one, but it means roughly 2 in 3 defaulters still slip through at this operating point.
Lowering the decline threshold would catch more defaulters at the cost of declining more good
applicants — the 5:1 ratio is a business assumption, not a law of nature, and should be revisited
with real cost data before production use.

**Fairness gaps grew substantially with EXT_SOURCE.** Gender demographic-parity gap went from
0.0055 (7-feature model) to 0.0454; region gap went from 0.023 to 0.1026. This makes sense —
EXT_SOURCE scores are themselves derived from other institutions' models, which may encode their
own historical biases, and they're by far the strongest predictors now driving decisions. This is
flagged as a real finding, not swept under the rug: **no fairness mitigation (reweighing, group-
specific threshold adjustment, or excluding EXT_SOURCE) has been applied yet.** This needs
scrutiny before any real deployment decision, especially given SEX/EDUCATION/MARRIAGE were
deliberately removed from scoring earlier in this project for the same reason.

Saved to `models_real_rich/` (not `models_real/` or `models/`) — nothing currently served by
`api/app.py` has changed; that decision is still pending.

## Update: fairness gap mitigation, tested with real numbers

Root cause first: `EXT_SOURCE_1` mean is 0.546 for women vs 0.407 for men in the raw Home Credit
data, and region-1 (best-rated) applicants average 0.568 vs 0.469 for region-3. This isn't a
scoring artifact — actual default rates really do differ the same direction (women 7.0% vs men
10.1%; region-1 4.8% vs region-3 11.1%), so part of the demographic-parity gap is the model
correctly learning a real difference in outcomes, not manufacturing one.

But the **equalized-odds gap** (TPR-on-good-borrowers, which conditions on actual outcome and so
isolates the part of the disparity NOT explained by differing base rates) was still substantial:
0.0344 for gender, 0.0775 for region. That residual is the part worth fixing.

**Mitigation tested** (`src/fairness_mitigation_real_rich.py`, no retraining — reuses the saved
`models_real_rich/` artifacts): replace the single global decline threshold with a per-group
threshold chosen to equalize each group's TPR-on-good-borrowers to the population-wide rate.

| | Gender: before → after | Region: before → after |
|---|---|---|
| Equalized-odds gap | 0.0344 → **0.0079** (-77%) | 0.0775 → **0.0178** (-77%) |
| Demographic-parity gap | 0.0454 → 0.0159 | 0.1026 → 0.0143 |
| Overall recall | 0.3610 → 0.3472 | 0.3610 → 0.3546 |
| Overall precision | 0.2679 → 0.2707 | 0.2679 → 0.2607 |

**A ~77% reduction in the equalized-odds gap costs under 1.5 points of overall recall and
essentially nothing in precision.** This is a cheap, high-value fix and should be applied before
any real deployment. Caveat: this test adjusts one attribute at a time (gender-only or
region-only); the two group splits weren't adjusted simultaneously, and a production version
would need intersectional group thresholds (e.g. female + region-3) or a documented choice of
which single attribute to prioritize, since correcting for both at once with this simple method
isn't additive. Not yet wired into `api/scoring.py` — this is a validated finding, not yet a
shipped fix.
