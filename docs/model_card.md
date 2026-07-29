# Model Card — Hybrid Credit Scoring Model

## Overview
A hybrid model that estimates an applicant's **probability of default (PD)** by combining two views
of the borrower:
- an **XGBoost** model over static financial features, and
- an **LSTM** over the last 12 months of the payment-status sequence (the trajectory of on-time vs
  late payments).

The two are fused: the XGBoost score and the LSTM's temporal embedding are concatenated and passed
through a small MLP. The raw output is then **isotonic-calibrated** so the number is a genuine
probability, and the approve/review/decline call is made with **cost-based thresholds** fit on
held-out data.

## Change history — why the model was retrained (2026-07)
The first deployed version was trained on the Taiwan Credit Card dataset (30k rows, 6 months of
history) and had three real problems, found by direct testing against the live API:

1. **It scored SEX, EDUCATION, and MARRIAGE** — protected attributes that are illegal to use for
   credit decisions in most jurisdictions.
2. **It had no probability calibration.** On a held-out test set, the mean predicted PD was **43.4%**
   against an actual default rate of **22.1%** — roughly double the truth. Using the shipped 0.2/0.5
   decision thresholds, this meant **91% of all applicants** (approve 8.7% / review 61.8% / decline
   29.5%) would have been flagged for review or decline, regardless of their real risk.
3. **Six months of history is thin** for a temporal model, and 30k rows is a small training set.

The current model fixes all three: it is trained on a **larger synthetic book (150k rows, 12 months
of history)** that contains no protected attributes, and its output is **calibrated** with
**cost-based thresholds** instead of an arbitrary 0.5 cutoff.

## Intended use
- **Use:** a decision-support signal for consumer-credit underwriting. Output is a calibrated
  probability, an approve/review/decline call, ranked "why" factors, a loan-pricing offer, and
  ranked improvement advice.
- **Users:** lenders, credit teams, risk analysts, integrating via the REST API; consumers checking
  their own risk and how to improve it.
- **Not for:** automated final decisions without human review, employment or insurance decisions, or
  any use outside consumer credit.

## Training data
- **Synthetic lender book** (`src/synth_data.py`): 150,000 simulated borrowers with 7 static
  financial features (age, monthly income, credit limit, existing debt, debt-to-income, employment
  years, number of existing loans) and 12 months of payment-status history. The generator is
  designed so the static and sequence branches carry **complementary** signal (financial risk vs.
  payment discipline), matching the structure of real credit data. The schema mirrors what a real
  lender would provide, so swapping in real data is a loader change, not a redesign.
- **No protected attributes are used for scoring.** `gender` and `region` exist in the generator
  *only* for the fairness audit below; the API schema uses `extra="forbid"`, so it is structurally
  impossible to submit sex/marriage/education to the scoring endpoint.

## Performance (held-out test set, 22,512 synthetic borrowers, 22.4% actual default rate)
| Metric | Value |
|---|---|
| AUC (ranking quality) | **0.892** |
| Brier score (uncalibrated) | 0.122 |
| Brier score (calibrated) | **0.101** |
| Mean predicted PD (uncalibrated) | 34.9% |
| Mean predicted PD (calibrated) | **22.0%** (actual: 22.4%) |

**Reliability** (calibrated PD by decile, test set — predicted should track actual):

| Decile | Predicted | Actual |
|---|---|---|
| 1 (safest) | 0.7% | 0.6% |
| 4 | 4.7% | 5.6% |
| 7 | 19.4% | 18.7% |
| 10 (riskiest) | 83.6% | 84.8% |

Every decile is within a few points of the true rate — this is what a genuinely calibrated model
looks like, in contrast to the previous version's 2x-inflated output.

**Decision mix at the cost-optimal thresholds (5:1 cost ratio — approving a defaulter is judged 5x
worse than declining a good customer):** approve 45.7% / review 15.1% / decline 39.2%, against a
true default rate of 22.4%. (Compare to the previous model's 8.7% / 61.8% / 29.5%.)

## Explainability
Each decision includes a **ranked "why"** list:
- **Static factors** use XGBoost's exact SHAP contributions (`pred_contribs`), converted to an
  approximate probability-scale impact via the local sigmoid derivative.
- **Payment history** uses a counterfactual: the fused model's output with the applicant's actual
  12-month sequence vs. with every month standardized as on-time. The difference is the sequence's
  contribution, on the same probability scale as the static factors.

This is a first-order approximation for combining two different attribution methods onto one scale,
not an exact decomposition — it is presented as a ranked, directional explanation, not a precise
percentage breakdown.

## Pricing and advice
- **Risk-based pricing:** interest rate scales with calibrated PD (base rate + risk premium); max
  loan amount is capped by affordability (a fraction of disposable income) and the credit limit.
  Collateral lowers the rate and widens the approval cutoff (verified: it never makes an offer worse,
  and can flip a borderline decision from review/decline to approve).
- **Advice:** a small set of what-if scenarios (recent on-time payments, reduced debt, higher income)
  are re-scored and ranked by projected PD improvement, so an applicant can see concretely what would
  change their outcome.

## Fairness audit
Run on the held-out test set, across `gender` (2 groups) and `region` (5 groups) — attributes that
are **never used for scoring**, held out specifically for this audit:

| Attribute | Demographic parity gap | Equalized odds gap |
|---|---|---|
| Gender | 0.000 | 0.009 |
| Region | 0.015 | 0.026 |

Both gaps are near zero. **Caveat:** this is expected on synthetic data, because the generator does
not make gender/region causally influence risk — this audit demonstrates the *methodology* (the code
path a real fairness audit would run), not a guarantee about a real population. A real deployment
must re-run this audit on the lender's actual population and protected attributes, where genuine
disparities are possible even without those attributes being direct model inputs (e.g. via proxy
correlation with income or region).

## Limitations
- Trained on synthetic data; **would require retraining on the lender's own population** before real
  use, and ongoing drift monitoring (e.g. PSI).
- The explainability "impact" numbers are an approximation for ranking and direction, not an exact
  decomposition of the calibrated probability.
- No causal claims: the model finds correlations, not causes of default.
- Pricing formulas (base rate, risk premium, affordability cap) are illustrative business-policy
  defaults, not calibrated to any real lender's cost of capital or risk appetite.

## Maintenance
- Retrain when population drift is detected (PSI > 0.2) or performance degrades.
- Version every model artifact; the API returns `model_version` with each prediction, and the health
  endpoint (`GET /`) reports the current model's test AUC.
