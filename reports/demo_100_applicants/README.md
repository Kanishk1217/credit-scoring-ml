# Demo: 100 applicants through the deployed model

From the "bank teller uploads a batch of 100 people" walkthrough. Reproducible with
`src/synth_data.generate(100, seed=7777)` — a fresh sample never seen during training, validation,
calibration, or testing.

## Files
- **`input_upload.csv`** — the batch exactly as a bank teller would upload it: one row per
  applicant, the 6 static financial fields plus 12 months of payment status (`pay_1` oldest ..
  `pay_12` most recent). This is the real input schema the API accepts.
- **`model_results.csv`** — the model's output for each applicant (probability of default,
  recommendation, pricing offer, top explanatory factor), scored through the actual deployed
  `/predict/batch` endpoint. Includes one extra column,
  `actual_default_SIMULATED_GROUND_TRUTH_ONLY`, which is **only available here because the data is
  simulated** — a real bank would not know this outcome until the loan matured months later. It
  exists purely so we can grade the model's accuracy on this walkthrough.
- **`summary.json`** — the confusion matrix and metrics computed from the two files above.

## Headline result
Of 23 real defaulters in this batch, the model correctly flagged 20 (87% recall/catch rate).
Raw accuracy (66%) is actually *lower* than a naive "always approve" baseline (77%) — which is
exactly why accuracy is the wrong metric to judge this model on: the naive baseline catches zero
defaulters. See the main model card (`docs/model_card.md`) for the full discussion.
