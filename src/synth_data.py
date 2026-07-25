"""Synthetic lender dataset generator.

Produces a realistic credit "book": static financial features + a 12-month payment-status
sequence + a default outcome, with realistic structure so a model can genuinely learn signal:

- static features drive a latent risk propensity (debt-to-income, income, age, employment),
- the monthly payment sequence is generated with momentum (being late raises the chance of
  staying late), so trajectories (improving vs deteriorating) are real,
- default depends on both the latent risk AND the recent payment trajectory, plus noise.

The schema mirrors what a real lender would provide, so the rest of the pipeline is
data-source-agnostic: swap this generator for a real CSV loader and nothing else changes.

Usage:
    from src.synth_data import generate
    df = generate(n=100_000, seed=42)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS = 12
STATIC_COLS = ["age", "monthly_income", "credit_limit", "existing_debt",
               "debt_to_income", "employment_years", "num_existing_loans"]
DEMO_COLS = ["gender", "region"]          # for the fairness audit, not for scoring
SEQ_COLS = [f"pay_{m}" for m in range(MONTHS)]   # pay_0 oldest .. pay_11 most recent


def generate(n: int = 100_000, seed: int = 42, months: int = MONTHS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ---- static features ----
    age = rng.integers(21, 70, n)
    log_income = rng.normal(10.7, 0.5, n)             # ~ monthly income (log scale)
    income = np.exp(log_income)
    credit_limit = income * rng.uniform(1.5, 6.0, n)
    existing_debt = credit_limit * rng.beta(2, 3, n)
    dti = existing_debt / (income * 12 + 1)           # debt-to-income
    employment_years = rng.gamma(2.0, 3.0, n).clip(0, 40)
    num_loans = rng.poisson(2, n)
    gender = rng.integers(0, 2, n)
    region = rng.integers(0, 5, n)

    # ---- two semi-independent risk factors (so static & sequence are complementary) ----
    # financial risk: fully observable from the static features
    fin = (2.0 * dti - 0.5 * (log_income - 10.7) - 0.015 * (age - 45)
           - 0.04 * employment_years + 0.08 * num_loans)
    fin = (fin - fin.mean()) / fin.std()                        # standardize ~ N(0,1)
    # behavioral risk: payment discipline, a separate trait NOT in the static columns
    beh = rng.normal(0, 1, n)

    # ---- payment sequence driven mainly by behavioral risk (reveals `beh`, not `fin`) ----
    base = -1.3 + 1.5 * beh + 0.35 * fin + rng.normal(0, 0.3, n)
    pay = np.zeros((n, months), dtype=int)
    momentum = np.zeros(n)
    for m in range(months):
        p_late = 1.0 / (1.0 + np.exp(-(base + 1.3 * momentum)))
        late = rng.random(n) < p_late
        severity = np.where(late, rng.integers(1, 5, n), 0)      # 1..4 months late
        on_time = rng.choice([-1, 0], n)                         # paid vs revolving
        pay[:, m] = np.where(late, severity, on_time)
        momentum = np.minimum(severity, 3) / 3.0

    # ---- default depends on BOTH: financial risk (static) + recent behavior (sequence) ----
    late_amt = np.clip(pay, 0, None)
    recent_late = late_amt[:, -4:].sum(axis=1)                   # lateness last 4 months
    trend = late_amt[:, -4:].sum(1) - late_amt[:, :4].sum(1)     # deteriorating > 0
    default_logit = (-3.25
                     + 1.15 * fin                               # static reveals this
                     + 0.32 * recent_late                       # sequence reveals this
                     + 0.16 * trend
                     + rng.normal(0, 0.6, n))
    p_default = 1.0 / (1.0 + np.exp(-default_logit))
    default = (rng.random(n) < p_default).astype(int)

    df = pd.DataFrame({
        "age": age,
        "monthly_income": income.round(0),
        "credit_limit": credit_limit.round(0),
        "existing_debt": existing_debt.round(0),
        "debt_to_income": dti.round(4),
        "employment_years": employment_years.round(1),
        "num_existing_loans": num_loans,
        "gender": gender,
        "region": region,
        "default": default,
    })
    for m in range(months):
        df[f"pay_{m}"] = pay[:, m]
    return df


if __name__ == "__main__":
    df = generate(100_000, seed=42)
    print("shape:", df.shape, "| default rate:", round(df["default"].mean(), 4))
    print(df.head(3).to_string())
