"""Build a REAL training dataset from Home Credit (application_train + bureau + installments),
in the exact schema our pipeline already expects (STATIC_COLS + 12-month SEQ_COLS + DEMO_COLS +
target), so it's a drop-in replacement for src/synth_data.generate().

Real column mapping:
  age               <- -DAYS_BIRTH / 365
  monthly_income    <- AMT_INCOME_TOTAL / 12
  credit_limit      <- AMT_CREDIT (the loan amount Home Credit extended)
  existing_debt     <- bureau.AMT_CREDIT_SUM_DEBT, summed per applicant (real past debt at OTHER lenders)
  debt_to_income    <- existing_debt / (monthly_income*12 + 1)
  employment_years  <- -DAYS_EMPLOYED / 365 (365243 placeholder for unemployed/pensioners -> NaN -> 0)
  num_existing_loans<- bureau.SK_ID_BUREAU, counted per applicant (real past loan count)
  pay_0..pay_11     <- built from installments_payments.csv: for each of the trailing 12 months,
                       the worst (max) lateness severity observed, 0 = on time, capped at 9.
  gender/region     <- CODE_GENDER, REGION_RATING_CLIENT (real; audit-only, never scored)
  default           <- TARGET

Only applicants with >=6 of the 12 months covered by real installment records are kept, so the
sequence reflects real observed behavior rather than mostly-filled defaults.

Usage:
    from src.build_real_data import build
    df = build()   # same columns as synth_data.generate(), but every row is a real applicant
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HC = ROOT / "data" / "raw" / "home_credit"

STATIC_COLS = ["age", "monthly_income", "credit_limit", "existing_debt",
              "debt_to_income", "employment_years", "num_existing_loans"]
DEMO_COLS = ["gender", "region"]
MONTHS = 12
SEQ_COLS = [f"pay_{m}" for m in range(MONTHS)]
MIN_MONTHS_COVERED = 6


def _build_sequences() -> pd.DataFrame:
    """One row per SK_ID_CURR: pay_0 (oldest of the trailing 12) .. pay_11 (most recent)."""
    inst = pd.read_csv(HC / "installments_payments.csv",
                       usecols=["SK_ID_CURR", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT"])
    inst = inst.dropna(subset=["DAYS_ENTRY_PAYMENT"])   # a few unpaid instalments have no payment date

    days_late = inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]
    # bucket real days-late into the same 0-9 severity scale used everywhere else in the project
    # (0 = on time/early; roughly one severity step per ~30 days late; capped at 9)
    severity = np.clip(np.ceil(days_late.clip(lower=0) / 30).astype(int), 0, 9)
    month_idx = (-inst["DAYS_INSTALMENT"] // 30).astype(int)   # 0 = most recent month, 11 = oldest kept

    recent = pd.DataFrame({
        "SK_ID_CURR": inst["SK_ID_CURR"].values,
        "month_idx": month_idx.values,
        "severity": severity.values,
    })
    recent = recent[recent["month_idx"] < MONTHS]

    coverage = recent.groupby("SK_ID_CURR")["month_idx"].nunique()
    keep_ids = coverage[coverage >= MIN_MONTHS_COVERED].index

    # worst (max) severity per applicant per month bucket
    worst = recent.groupby(["SK_ID_CURR", "month_idx"])["severity"].max().unstack(fill_value=0)
    worst = worst.reindex(columns=range(MONTHS), fill_value=0)   # ensure all 12 columns exist
    worst = worst.reindex(index=keep_ids, fill_value=0)          # months with no instalment -> on-time

    # month_idx 0 = most recent -> that should be pay_11 (our SEQ_COLS convention: pay_0 oldest)
    worst = worst[list(range(MONTHS))[::-1]]
    worst.columns = SEQ_COLS
    return worst.reset_index()


def _build_static_and_target() -> pd.DataFrame:
    app = pd.read_csv(HC / "application_train.csv",
                      usecols=["SK_ID_CURR", "TARGET", "DAYS_BIRTH", "AMT_INCOME_TOTAL",
                               "AMT_CREDIT", "DAYS_EMPLOYED", "CODE_GENDER", "REGION_RATING_CLIENT"])
    app["age"] = -app["DAYS_BIRTH"] / 365.0
    app["monthly_income"] = app["AMT_INCOME_TOTAL"] / 12.0
    app["credit_limit"] = app["AMT_CREDIT"]
    days_emp = app["DAYS_EMPLOYED"].replace(365243, np.nan)   # known Home Credit placeholder
    app["employment_years"] = (-days_emp / 365.0).fillna(0.0).clip(lower=0)
    app["gender"] = app["CODE_GENDER"]
    app["region"] = app["REGION_RATING_CLIENT"]
    app["default"] = app["TARGET"]
    return app[["SK_ID_CURR", "default", "age", "monthly_income", "credit_limit",
               "employment_years", "gender", "region"]]


def _build_bureau_aggregates() -> pd.DataFrame:
    bureau = pd.read_csv(HC / "bureau.csv", usecols=["SK_ID_CURR", "SK_ID_BUREAU", "AMT_CREDIT_SUM_DEBT"])
    agg = bureau.groupby("SK_ID_CURR").agg(
        num_existing_loans=("SK_ID_BUREAU", "count"),
        existing_debt=("AMT_CREDIT_SUM_DEBT", "sum"),
    )
    agg["existing_debt"] = agg["existing_debt"].fillna(0).clip(lower=0)
    return agg.reset_index()


def build() -> pd.DataFrame:
    static = _build_static_and_target()
    bureau = _build_bureau_aggregates()
    seq = _build_sequences()

    df = static.merge(bureau, on="SK_ID_CURR", how="left").merge(seq, on="SK_ID_CURR", how="inner")
    df["num_existing_loans"] = df["num_existing_loans"].fillna(0).astype(int)
    df["existing_debt"] = df["existing_debt"].fillna(0.0)
    df["debt_to_income"] = df["existing_debt"] / (df["monthly_income"] * 12 + 1)

    # a handful of rows have non-positive income (data errors) -- drop, can't compute debt_to_income sanely
    df = df[df["monthly_income"] > 0].copy()
    df = df[df["gender"].isin(["M", "F"])].copy()   # drop the rare 'XNA' placeholder gender

    keep = ["SK_ID_CURR", "default"] + STATIC_COLS + DEMO_COLS + SEQ_COLS
    return df[keep].reset_index(drop=True)


if __name__ == "__main__":
    df = build()
    print(f"shape: {df.shape}")
    print(f"default rate: {df['default'].mean():.4f}")
    print(f"\nstatic feature summary:\n{df[STATIC_COLS].describe().to_string()}")
    print(f"\ngender counts:\n{df['gender'].value_counts().to_string()}")
    print(f"\nsequence sample (first 3 rows):\n{df[SEQ_COLS].head(3).to_string()}")
    out = ROOT / "data" / "processed" / "home_credit_real.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nsaved -> {out}")
