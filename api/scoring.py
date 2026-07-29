"""Scoring business logic: calibration, cost-based decisions, explainability, pricing, advice.

Kept separate from app.py (routing/security) so the actual credit-decision logic is in one
reviewable, testable place. All functions are pure (no I/O), given the loaded model objects.
"""
from __future__ import annotations

import numpy as np
import xgboost

HUMAN_LABELS = {
    "age": "Age",
    "monthly_income": "Monthly income",
    "credit_limit": "Credit limit",
    "existing_debt": "Existing debt",
    "debt_to_income": "Debt-to-income ratio",
    "employment_years": "Employment history",
    "num_existing_loans": "Number of existing loans",
}

# --- pricing constants (risk-based pricing engine) ---
COST_RATIO = 5.0                 # cost(approve a defaulter) / cost(decline a good customer)
RECOVERY_RATE = 0.5              # fraction of principal recovered on a secured default
BASE_RATE_PCT = 11.0             # annual %, unsecured base rate
RISK_PREMIUM_PCT = 40.0          # percentage points added per unit of PD
SECURED_BASE_DISCOUNT_PCT = 3.0  # rate discount when collateral is pledged
RATE_FLOOR_PCT = 10.5
RATE_CAP_PCT = 24.0
DTI_EMI_CAP = 0.40               # new EMI capped at 40% of disposable income
EXISTING_DEBT_SERVICE_RATE = 0.03  # ~monthly obligation on existing debt balance
DEFAULT_TENURE_MONTHS = 36
# how much collateral widens the decline cutoff, derived from the 1/(1+C) Bayes-optimal
# threshold ratio: (1+C) / (1+C*(1-recovery))
SECURED_THRESHOLD_MULTIPLIER = (1 + COST_RATIO) / (1 + COST_RATIO * (1 - RECOVERY_RATE))


def build_static_row(static_cols: list[str], age, monthly_income, credit_limit,
                     existing_debt, employment_years, num_existing_loans) -> np.ndarray:
    """Assemble the static feature vector in the exact column order the model was trained on,
    computing debt_to_income the same way training data did."""
    dti = existing_debt / (monthly_income * 12 + 1)
    values = {
        "age": age, "monthly_income": monthly_income, "credit_limit": credit_limit,
        "existing_debt": existing_debt, "debt_to_income": dti,
        "employment_years": employment_years, "num_existing_loans": num_existing_loans,
    }
    return np.array([[values[c] for c in static_cols]], dtype="float32")


def calibrate(cfg: dict, raw_pd: float) -> float:
    """Map an uncalibrated model output to a calibrated probability via the fitted isotonic curve
    (stored as a lookup grid, applied with linear interpolation — no sklearn needed at serve time)."""
    grid_x = cfg["calibration"]["grid_x"]
    grid_y = cfg["calibration"]["grid_y"]
    return float(np.interp(raw_pd, grid_x, grid_y))


def decide(cfg: dict, calibrated_pd: float, has_collateral: bool = False) -> tuple[str, float, float]:
    """Cost-based approve/review/decline using thresholds fit on held-out data (not an
    arbitrary 0.5/0.2 split). Collateral widens the decline cutoff because a secured loan's
    expected loss is lower (partial recovery on default)."""
    approve_t = cfg["thresholds"]["approve"]
    decline_t = cfg["thresholds"]["decline"]
    if has_collateral:
        decline_t = min(0.95, decline_t * SECURED_THRESHOLD_MULTIPLIER)
        approve_t = min(decline_t * 0.5, approve_t * SECURED_THRESHOLD_MULTIPLIER)
    if calibrated_pd >= decline_t:
        return "decline", approve_t, decline_t
    if calibrated_pd >= approve_t:
        return "review", approve_t, decline_t
    return "approve", approve_t, decline_t


def explain(cfg: dict, xgb_model, static_cols: list[str], static_row: np.ndarray,
            xgb_score: float, net, seq_std: list[float], top_n: int = 5) -> list[dict]:
    """Rank human-readable factors driving the decision.

    Static branch: exact XGBoost SHAP contributions (pred_contribs, in margin/log-odds units),
    converted to an approximate probability-space impact via the local sigmoid derivative
    p*(1-p) (a first-order linearization — standard way to read margin contributions as
    probability deltas near the operating point).

    Sequence branch: a counterfactual — the fused model's output with the applicant's ACTUAL
    (standardized) payment history vs. with every month standardized as if it were on-time.
    The difference is the payment history's contribution, on the same probability scale.
    """
    booster = xgb_model.get_booster()
    dm = xgboost.DMatrix(static_row, feature_names=static_cols)
    contribs = booster.predict(dm, pred_contribs=True)[0]        # (n_features + 1,), last = bias
    deriv = xgb_score * (1 - xgb_score)                           # d(sigmoid)/d(margin) at this point

    factors = []
    for name, c in zip(static_cols, contribs[:-1], strict=False):
        impact = float(c) * deriv
        if abs(impact) < 1e-4:
            continue
        factors.append({
            "col": name,
            "factor": HUMAN_LABELS.get(name, name),
            "impact": round(impact, 4),
            "direction": "increases_risk" if impact > 0 else "decreases_risk",
        })

    # sequence counterfactual: "-1" (on-time) standardized the same way the real sequence was
    on_time_std = [(-1.0 - cfg["seq_mean"]) / cfg["seq_std"]] * len(seq_std)
    factors.append(_sequence_factor(net, seq_std, on_time_std, xgb_score))

    factors.sort(key=lambda f: abs(f["impact"]), reverse=True)
    return factors[:top_n]


def _sequence_factor(net, seq_std_actual: list[float], seq_std_ontime: list[float],
                     xgb_score: float) -> dict:
    pd_actual = net.predict(seq_std_actual, xgb_score)
    pd_ontime = net.predict(seq_std_ontime, xgb_score)
    impact = pd_actual - pd_ontime
    return {
        "col": "payment_history",
        "factor": "Recent payment history",
        "impact": round(float(impact), 4),
        "direction": "increases_risk" if impact > 0 else "decreases_risk",
    }


def band(pd: float) -> str:
    """Coarse risk grade (A best - E worst), independent of the approve/review/decline action
    thresholds -- a descriptive bucket for dashboards/reporting, not a decision boundary."""
    if pd <= 0.03:
        return "A"
    if pd <= 0.07:
        return "B"
    if pd <= 0.15:
        return "C"
    if pd <= 0.30:
        return "D"
    return "E"


def price_loan(calibrated_pd: float, monthly_income: float, existing_debt: float,
              credit_limit: float, decision: str, has_collateral: bool = False,
              requested_amount: float | None = None,
              tenure_months: int = DEFAULT_TENURE_MONTHS) -> dict:
    """Risk-based pricing: interest rate scales with PD, max amount from affordability capped
    by the credit limit. Declined applicants get a zero offer."""
    pd_ = max(0.0, min(1.0, calibrated_pd))
    premium_mult = (1 - RECOVERY_RATE) if has_collateral else 1.0
    base = BASE_RATE_PCT - (SECURED_BASE_DISCOUNT_PCT if has_collateral else 0.0)
    rate_pct = max(RATE_FLOOR_PCT, min(RATE_CAP_PCT, base + RISK_PREMIUM_PCT * pd_ * premium_mult))

    disposable = monthly_income - existing_debt * EXISTING_DEBT_SERVICE_RATE
    max_emi = DTI_EMI_CAP * disposable
    r = rate_pct / 100 / 12
    n = tenure_months

    if decision == "decline" or max_emi <= 0:
        max_loan, emi = 0, 0
    else:
        principal = max_emi * (1 - (1 + r) ** -n) / r if r > 0 else max_emi * n
        max_loan = int(min(principal, credit_limit) // 1000 * 1000)
        emi = int(round(max_loan * r / (1 - (1 + r) ** -n))) if r > 0 and max_loan > 0 else int(max_emi)

    note_parts = []
    if decision == "decline":
        note_parts.append("Declined applicants are not offered a loan amount.")
    elif has_collateral:
        note_parts.append("Collateral improved the rate and the approval cutoff.")
    if requested_amount and max_loan and requested_amount > max_loan:
        note_parts.append("Requested amount exceeds what affordability supports at this tenure.")

    return {
        "max_loan_amount": max_loan,
        "interest_rate_pct": round(rate_pct, 2),
        "tenure_months": tenure_months,
        "monthly_emi": emi,
        "note": " ".join(note_parts) or "Standard offer.",
    }


# --- advice: ranked what-if scenarios showing how specific changes move the risk ---
ADVICE_SCENARIOS = [
    ("6 months of on-time payments", "seq_ontime_6"),
    ("Reduce existing debt by 25%", "debt_25"),
    ("Reduce existing debt by 50%", "debt_50"),
    ("Increase monthly income by 20%", "income_20"),
]


def advice(cfg: dict, xgb_model, static_cols: list[str], age, monthly_income, credit_limit,
          existing_debt, employment_years, num_existing_loans, seq_raw: list[float],
          net, current_pd: float, top_n: int = 3) -> list[dict]:
    """Re-score a handful of what-if scenarios and rank them by projected PD improvement."""
    seq_mean, seq_std_ = cfg["seq_mean"], cfg["seq_std"]

    def score_with(debt=existing_debt, income=monthly_income, seq=seq_raw):
        row = build_static_row(static_cols, age, income, credit_limit, debt,
                               employment_years, num_existing_loans)
        s = float(xgb_model.predict_proba(row)[0, 1])
        seq_std = [(v - seq_mean) / seq_std_ for v in seq]
        raw = net.predict(seq_std, s)
        return calibrate(cfg, raw)

    n_late_recent = sum(1 for v in seq_raw[-6:] if v > 0)
    results = []
    if n_late_recent > 0:
        seq_ontime = seq_raw[:-6] + [-1.0] * 6
        results.append(("6 months of on-time payments", score_with(seq=seq_ontime)))
    if existing_debt > 0:
        results.append(("Reduce existing debt by 25%", score_with(debt=existing_debt * 0.75)))
        results.append(("Reduce existing debt by 50%", score_with(debt=existing_debt * 0.5)))
    results.append(("Increase monthly income by 20%", score_with(income=monthly_income * 1.2)))

    out = []
    for label, projected in results:
        delta = projected - current_pd
        if delta >= -1e-4:      # only show scenarios that actually help
            continue
        out.append({
            "scenario": label,
            "current_pd": round(current_pd, 4),
            "projected_pd": round(projected, 4),
            "pd_improvement": round(-delta, 4),
        })
    out.sort(key=lambda a: a["pd_improvement"], reverse=True)
    return out[:top_n]
