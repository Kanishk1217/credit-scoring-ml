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
