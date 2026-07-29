"""Enriched real-data feature set: extends build_real_data.py with deep aggregates from all
remaining Home Credit tables (bureau, bureau_balance, previous_application, POS_CASH_balance,
credit_card_balance, installments_payments) -- all already downloaded. Same aggregation pattern
proven in Phase 2's notebook 06, extended in depth (recency, ratios, totals, not just count/sum)
since that depth is what actually separates a basic feature set from a competitive one.

IMPORTANT product distinction: many of these features (previous-application approval history,
monthly bureau/POS/credit-card delinquency, credit-card utilization, lifetime installment
history) are things a LENDER'S OWN SYSTEMS already know about an existing/returning customer --
not something a brand-new applicant would type into a form. This enriched feature set is for an
INTERNAL underwriting tool (the loan-officer dashboard, which pulls an applicant's bureau/account
history automatically, exactly like a real underwriting system does), not the public self-service
API, which keeps the simpler 7-field schema from build_real_data.py.

Usage:
    from src.build_real_data_rich import build_rich
    df = build_rich()   # base build_real_data.build() columns + ~26 additional real features
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_real_data import (  # noqa: E402
    DEMO_COLS,
    HC,
    SEQ_COLS,
    STATIC_COLS,
    _build_sequences,
    _build_static_and_target,
)

RICH_COLS = [
    # external bureau-like risk scores Home Credit itself computed (NOT self-reportable --
    # a lender would pull these from their own systems / a bureau API, exactly like the
    # credit-bureau-API path discussed for a real deployment). By far the single strongest
    # individual predictors in this dataset (each alone: standalone AUC ~0.66-0.68).
    "ext_source_1", "ext_source_2", "ext_source_3",
    # bureau (deepened beyond count + total debt)
    "bureau_active_count", "bureau_max_credit_sum", "bureau_overdue_amt_sum",
    "bureau_days_since_last_credit",
    # bureau_balance (monthly delinquency at other lenders)
    "bureau_bal_max_dpd_severity", "bureau_bal_num_bad_months",
    # previous_application (this lender's own history with the applicant)
    "prev_app_count", "prev_approval_rate", "prev_refusal_rate",
    "prev_mean_amt_credit", "prev_max_amt_credit", "prev_mean_annuity",
    "prev_days_since_last_decision",
    # POS_CASH_balance (point-of-sale/cash loan monthly status)
    "pos_max_dpd", "pos_mean_dpd", "pos_num_dpd_months", "pos_count", "pos_completed_ratio",
    # credit_card_balance (revolving utilization + delinquency)
    "cc_mean_utilization", "cc_max_utilization", "cc_max_dpd", "cc_num_dpd_months",
    "cc_mean_drawings", "cc_count",
    # installments_payments -- LIFETIME summary stats (separate from the 12-month sequence)
    "inst_total_count", "inst_mean_days_late", "inst_pct_late", "inst_total_shortfall",
]

# bureau_balance STATUS: '0'..'5' = increasing DPD severity (5 = 120+/write-off), 'C'=closed, 'X'=unknown
_STATUS_SEVERITY = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "C": 0, "X": 0}


def _ext_sources() -> pd.DataFrame:
    """External bureau-like risk scores, straight from application_train.csv. Left as real NaN
    where missing (56% missing for source 1, 0.2% for source 2, 20% for source 3) -- XGBoost
    handles missing values natively, so no imputation needed or wanted."""
    app = pd.read_csv(HC / "application_train.csv",
                      usecols=["SK_ID_CURR", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"])
    app.columns = ["SK_ID_CURR", "ext_source_1", "ext_source_2", "ext_source_3"]
    return app


def _bureau_aggregates_deep() -> pd.DataFrame:
    bureau = pd.read_csv(HC / "bureau.csv", usecols=[
        "SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "AMT_CREDIT_SUM",
        "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_OVERDUE", "DAYS_CREDIT"])
    bureau["is_active"] = (bureau["CREDIT_ACTIVE"] == "Active").astype(int)
    agg = bureau.groupby("SK_ID_CURR").agg(
        num_existing_loans=("SK_ID_BUREAU", "count"),
        existing_debt=("AMT_CREDIT_SUM_DEBT", "sum"),
        bureau_active_count=("is_active", "sum"),
        bureau_max_credit_sum=("AMT_CREDIT_SUM", "max"),
        bureau_overdue_amt_sum=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        bureau_days_since_last_credit=("DAYS_CREDIT", "max"),   # DAYS_CREDIT is negative; max = most recent
    )
    agg["bureau_days_since_last_credit"] = -agg["bureau_days_since_last_credit"]
    return agg.reset_index()


def _bureau_balance_aggregates() -> pd.DataFrame:
    bb = pd.read_csv(HC / "bureau_balance.csv", usecols=["SK_ID_BUREAU", "STATUS"])
    bureau_ids = pd.read_csv(HC / "bureau.csv", usecols=["SK_ID_BUREAU", "SK_ID_CURR"])
    bb = bb.merge(bureau_ids, on="SK_ID_BUREAU", how="inner")
    bb["severity"] = bb["STATUS"].map(_STATUS_SEVERITY).fillna(0)
    agg = bb.groupby("SK_ID_CURR").agg(
        bureau_bal_max_dpd_severity=("severity", "max"),
        bureau_bal_num_bad_months=("severity", lambda s: int((s > 0).sum())),
    )
    return agg.reset_index()


def _previous_application_aggregates() -> pd.DataFrame:
    prev = pd.read_csv(HC / "previous_application.csv", usecols=[
        "SK_ID_CURR", "NAME_CONTRACT_STATUS", "AMT_CREDIT", "AMT_ANNUITY", "DAYS_DECISION"])
    prev["is_approved"] = (prev["NAME_CONTRACT_STATUS"] == "Approved").astype(int)
    prev["is_refused"] = (prev["NAME_CONTRACT_STATUS"] == "Refused").astype(int)
    agg = prev.groupby("SK_ID_CURR").agg(
        prev_app_count=("NAME_CONTRACT_STATUS", "count"),
        prev_approval_rate=("is_approved", "mean"),
        prev_refusal_rate=("is_refused", "mean"),
        prev_mean_amt_credit=("AMT_CREDIT", "mean"),
        prev_max_amt_credit=("AMT_CREDIT", "max"),
        prev_mean_annuity=("AMT_ANNUITY", "mean"),
        prev_days_since_last_decision=("DAYS_DECISION", "max"),   # negative; max = most recent
    )
    agg["prev_days_since_last_decision"] = -agg["prev_days_since_last_decision"]
    return agg.reset_index()


def _pos_cash_aggregates() -> pd.DataFrame:
    pos = pd.read_csv(HC / "POS_CASH_balance.csv",
                      usecols=["SK_ID_CURR", "SK_DPD", "NAME_CONTRACT_STATUS"])
    pos["is_completed"] = (pos["NAME_CONTRACT_STATUS"] == "Completed").astype(int)
    agg = pos.groupby("SK_ID_CURR").agg(
        pos_max_dpd=("SK_DPD", "max"),
        pos_mean_dpd=("SK_DPD", "mean"),
        pos_num_dpd_months=("SK_DPD", lambda s: int((s > 0).sum())),
        pos_count=("SK_DPD", "count"),
        pos_completed_ratio=("is_completed", "mean"),
    )
    return agg.reset_index()


def _credit_card_aggregates() -> pd.DataFrame:
    cc = pd.read_csv(HC / "credit_card_balance.csv", usecols=[
        "SK_ID_CURR", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL", "SK_DPD", "AMT_DRAWINGS_CURRENT"])
    cc["utilization"] = (cc["AMT_BALANCE"] / cc["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)).clip(0, 5)
    agg = cc.groupby("SK_ID_CURR").agg(
        cc_mean_utilization=("utilization", "mean"),
        cc_max_utilization=("utilization", "max"),
        cc_max_dpd=("SK_DPD", "max"),
        cc_num_dpd_months=("SK_DPD", lambda s: int((s > 0).sum())),
        cc_mean_drawings=("AMT_DRAWINGS_CURRENT", "mean"),
        cc_count=("SK_DPD", "count"),
    )
    for c in ["cc_mean_utilization", "cc_max_utilization"]:
        agg[c] = agg[c].fillna(0)
    return agg.reset_index()


def _installments_lifetime_aggregates() -> pd.DataFrame:
    """LIFETIME summary stats from every installment ever, distinct from the trailing-12-month
    sequence -- e.g. total shortfall (how much money, in total, has this person ever underpaid)."""
    inst = pd.read_csv(HC / "installments_payments.csv", usecols=[
        "SK_ID_CURR", "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT", "AMT_PAYMENT"])
    inst = inst.dropna(subset=["DAYS_ENTRY_PAYMENT"])
    days_late = (inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]).clip(lower=0)
    shortfall = (inst["AMT_INSTALMENT"] - inst["AMT_PAYMENT"]).clip(lower=0)
    tmp = pd.DataFrame({
        "SK_ID_CURR": inst["SK_ID_CURR"].values,
        "days_late": days_late.values,
        "is_late": (days_late.values > 0).astype(int),
        "shortfall": shortfall.values,
    })
    agg = tmp.groupby("SK_ID_CURR").agg(
        inst_total_count=("is_late", "count"),
        inst_mean_days_late=("days_late", "mean"),
        inst_pct_late=("is_late", "mean"),
        inst_total_shortfall=("shortfall", "sum"),
    )
    return agg.reset_index()


def build_rich() -> pd.DataFrame:
    """base static features (deepened bureau aggregation included) + sequence + ~26 enriched
    real features across all remaining Home Credit tables."""
    static = _build_static_and_target()
    bureau = _bureau_aggregates_deep()
    seq = _build_sequences()
    df = static.merge(bureau, on="SK_ID_CURR", how="left").merge(seq, on="SK_ID_CURR", how="inner")
    df["num_existing_loans"] = df["num_existing_loans"].fillna(0).astype(int)
    df["existing_debt"] = df["existing_debt"].fillna(0.0)
    df["debt_to_income"] = df["existing_debt"] / (df["monthly_income"] * 12 + 1)
    df = df[df["monthly_income"] > 0].copy()
    df = df[df["gender"].isin(["M", "F"])].copy()

    for agg_df in [_ext_sources(), _bureau_balance_aggregates(), _previous_application_aggregates(),
                  _pos_cash_aggregates(), _credit_card_aggregates(),
                  _installments_lifetime_aggregates()]:
        df = df.merge(agg_df, on="SK_ID_CURR", how="left")

    # ext_source_* are left as real NaN on purpose (XGBoost handles missing values natively;
    # filling would misleadingly imply "worst score" (0) or "typical score" (median))
    no_fill = {"ext_source_1", "ext_source_2", "ext_source_3",
              "prev_mean_amt_credit", "prev_max_amt_credit", "prev_mean_annuity",
              "prev_days_since_last_decision", "bureau_days_since_last_credit",
              "inst_mean_days_late"}
    zero_fill = [c for c in RICH_COLS if c not in no_fill]
    for c in zero_fill:
        df[c] = df[c].fillna(0)
    # amounts/recency: fill with the column median rather than 0 (0 would misleadingly imply
    # "no credit" or "just happened")
    for c in ["prev_mean_amt_credit", "prev_max_amt_credit", "prev_mean_annuity",
             "prev_days_since_last_decision", "bureau_days_since_last_credit", "inst_mean_days_late"]:
        df[c] = df[c].fillna(df[c].median())

    keep = ["SK_ID_CURR", "default"] + STATIC_COLS + RICH_COLS + DEMO_COLS + SEQ_COLS
    return df[keep].reset_index(drop=True)


if __name__ == "__main__":
    df = build_rich()
    print(f"shape: {df.shape}")
    print(f"default rate: {df['default'].mean():.4f}")
    print(f"\nrich feature summary:\n{df[RICH_COLS].describe().to_string()}")
    out = "data/processed/home_credit_real_rich.csv"
    df.to_csv(out, index=False)
    print(f"\nsaved -> {out}")
