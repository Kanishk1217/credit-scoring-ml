# Demo: 5,000 fresh applicants for the live web app

A brand-new synthetic batch — `src.synth_data.generate(5000, seed=9191)` — a seed never used
for training, validation, calibration, testing, or the `demo_100_applicants` batch. The model
has not seen these applicants.

## Files
- **`input_upload.csv`** — 5,000 rows, exactly the schema the live API/web app accepts:
  `applicant_id` + the 6 static financial fields + 12 months of payment status (`pay_0` oldest
  .. `pay_11` most recent). No outcome column, no demographic columns — safe to upload as-is.
- **`ground_truth_HOLD_OUT.csv`** — `applicant_id` + `actual_default` only. Deliberately kept in
  a separate file so it can't leak into the upload. Use it after scoring `input_upload.csv`
  through the web app to check the model's predictions against what actually happened, the same
  way `demo_100_applicants/model_results.csv` did.

Regenerate with:
```python
from src.synth_data import generate
df = generate(5000, seed=9191)
```
