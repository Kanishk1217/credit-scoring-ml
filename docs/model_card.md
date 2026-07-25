# Model Card — Hybrid Credit Scoring Model

## Overview
A hybrid model that estimates an applicant's **probability of default (PD)** by combining two views of
the borrower:
- an **XGBoost** model over static features (credit limit, age, bill and payment amounts), and
- an **LSTM** over the recent monthly **payment-status sequence** (the trajectory of on-time vs late).

The two are fused: the XGBoost score and the LSTM's temporal embedding are concatenated and passed
through a small MLP that outputs the PD.

## Intended use
- **Use:** a decision-support signal for consumer-credit underwriting. Output is a probability plus a
  suggested action (approve / review / decline) under lender-set thresholds.
- **Users:** lenders, credit teams, risk analysts, integrating via the REST API.
- **Not for:** automated final decisions without human review, employment or insurance decisions, or any
  use outside consumer credit. The suggested action is advisory; the threshold is the lender's policy.

## Training data
- **Taiwan Credit Card default dataset** (UCI / Kaggle): 30,000 accounts, 6 months of payment history.
  Public, anonymized, research dataset. **It is not representative of any specific current lending
  population**, and is used here to demonstrate the modeling approach.

## Performance (held-out test set)
| Model | Test AUC |
|---|---|
| XGBoost (static features only) | 0.730 |
| LSTM (payment sequence only) | 0.737 |
| **Hybrid (fused)** | **0.775** |

AUC measures ranking quality (separating defaulters from non-defaulters). The hybrid beats either
branch alone because the branches carry complementary information (static snapshot vs payment
trajectory).

## Limitations
- Trained on a single public dataset from one market and period; **would require retraining on the
  lender's own population before real use**, and ongoing drift monitoring (e.g. PSI).
- Six months of history is short; the temporal component's advantage grows with longer sequences.
- Probabilities are calibrated only approximately; calibrate against the deployment population before
  using PD for pricing or capital.
- No causal claims: the model finds correlations, not causes of default.

## Fairness and ethics
- The training data contains demographic fields (sex, age, marriage, education). **A fairness audit
  (demographic parity, equalized odds across groups) must be run before deployment**, and protected
  attributes reconsidered, to avoid discriminatory outcomes. This audit is planned and not yet done.
- Decisions affecting people must remain explainable and contestable; per-applicant explanation
  (SHAP) is planned.

## Maintenance
- Retrain when population drift is detected (PSI > 0.2) or performance degrades.
- Version every model artifact; the API returns `model_version` with each prediction.
