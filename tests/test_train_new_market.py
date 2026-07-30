"""Unit tests for the new-market onboarding pipeline (src/train_new_market.py).

These are fast, synthetic-input tests of specific correctness properties (leakage guard, date
wraparound fix, resolved-status filtering) -- NOT a full pipeline run, which needs the real
39,717-row LendingClub file and is exercised manually via
`uv run python src/train_new_market.py data/raw/lendingclub/loan.csv`.
"""
import sys
from io import StringIO
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import train_new_market as tnm  # noqa: E402


def _tiny_raw_csv(rows: str) -> pd.DataFrame:
    header = ("id,member_id,loan_amnt,annual_inc,dti,open_acc,revol_bal,revol_util,total_acc,"
              "delinq_2yrs,inq_last_6mths,pub_rec,emp_length,home_ownership,verification_status,"
              "purpose,application_type,addr_state,issue_d,earliest_cr_line,loan_status")
    return pd.read_csv(StringIO(header + "\n" + rows))


def test_load_and_filter_drops_unresolved_statuses(tmp_path):
    """Current/Late/In Grace Period loans are unresolved -- must never be silently treated as
    the good class (that would mislabel censored loans as confirmed-good)."""
    csv_text = (
        "id,member_id,loan_amnt,annual_inc,dti,open_acc,revol_bal,revol_util,total_acc,"
        "delinq_2yrs,inq_last_6mths,pub_rec,emp_length,home_ownership,verification_status,"
        "purpose,application_type,addr_state,issue_d,earliest_cr_line,loan_status\n"
        "1,1,10000,50000,15.0,5,2000,20%,10,0,0,0,5 years,RENT,Verified,car,INDIVIDUAL,CA,"
        "Jan-11,Jan-05,Fully Paid\n"
        "2,2,15000,60000,20.0,8,5000,40%,15,1,1,0,10+ years,MORTGAGE,Not Verified,"
        "debt_consolidation,INDIVIDUAL,NY,Feb-11,Mar-02,Charged Off\n"
        "3,3,8000,40000,10.0,3,1000,10%,8,0,0,0,2 years,OWN,Source Verified,medical,"
        "INDIVIDUAL,TX,Mar-11,Jun-08,Current\n"
    )
    p = tmp_path / "tiny.csv"
    p.write_text(csv_text)
    df = tnm.load_and_filter(p)
    assert len(df) == 2
    assert set(df["loan_status"]) == {"Fully Paid", "Charged Off"}
    assert df.loc[df["loan_status"] == "Charged Off", "default"].iloc[0] == 1
    assert df.loc[df["loan_status"] == "Fully Paid", "default"].iloc[0] == 0


def test_two_digit_year_wraparound_is_fixed():
    """"Sep-68"-style two-digit years must not parse to 2068 -- a credit line can't postdate the
    loan that used it as underwriting history. This is the exact bug found in the real data."""
    df = pd.DataFrame({
        "issue_d": ["Jan-11", "Jan-11"],
        "earliest_cr_line": ["Sep-68", "Apr-99"],  # first one wraps naively, second doesn't
    })
    fixed = tnm._fix_two_digit_year_dates(df)
    assert (fixed["_earliest_cr_line"] <= fixed["_issue_d"]).all()
    assert fixed["_earliest_cr_line"].iloc[0].year == 1968


def test_engineer_features_leakage_guard_raises(monkeypatch):
    """If a post-origination/outcome-adjacent column were ever added to the numeric passthrough
    list by mistake, engineer_features must refuse to build a feature matrix, not silently leak."""
    df = _tiny_raw_csv(
        "1,1,10000,50000,15.0,5,2000,20%,10,0,0,0,5 years,RENT,Verified,car,INDIVIDUAL,CA,"
        "Jan-11,Jan-05,Fully Paid\n"
    )
    df["default"] = 0
    df["total_pymnt"] = 9999.0  # a real leakage column, deliberately injected into the source
    monkeypatch.setattr(tnm, "LC_NUMERIC_PASSTHROUGH", [*tnm.LC_NUMERIC_PASSTHROUGH, "total_pymnt"])
    with pytest.raises(AssertionError, match="leakage guard"):
        tnm.engineer_features(df)


def test_engineer_features_keeps_missing_as_nan_not_zero():
    """emp_length 'n/a' and missing revol_util must stay NaN for XGBoost's native missing-value
    handling -- coercing to 0 would misleadingly mean 'no job history' / 'zero utilization'."""
    df = _tiny_raw_csv(
        "1,1,10000,50000,15.0,5,2000,,10,0,0,0,n/a,RENT,Verified,car,INDIVIDUAL,CA,"
        "Jan-11,Jan-05,Fully Paid\n"
    )
    df["default"] = 0
    out, feature_cols, _ = tnm.engineer_features(df)
    assert pd.isna(out["employment_years"].iloc[0])
    assert pd.isna(out["revol_util"].iloc[0])


def test_engineer_features_never_includes_fairness_column_as_a_feature():
    df = _tiny_raw_csv(
        "1,1,10000,50000,15.0,5,2000,20%,10,0,0,0,5 years,RENT,Verified,car,INDIVIDUAL,CA,"
        "Jan-11,Jan-05,Fully Paid\n"
    )
    df["default"] = 0
    out, feature_cols, fairness_col = tnm.engineer_features(df)
    assert fairness_col == "region"
    assert "region" not in feature_cols
    assert "addr_state" not in feature_cols


def test_evaluate_confusion_matrix_matches_manual_count():
    import numpy as np
    y_true = np.array([1, 1, 0, 0, 1])
    calibrated_pd = np.array([0.9, 0.1, 0.8, 0.05, 0.6])
    result = tnm.evaluate(y_true, calibrated_pd, decline_t=0.5)
    cm = result["confusion_matrix"]
    # pd>=0.5: idx0(y1)->TP, idx1(y1)->FN, idx2(y0)->FP, idx3(y0)->TN, idx4(y1)->TP
    assert cm == {"true_positive": 2, "false_positive": 1, "true_negative": 1, "false_negative": 1}
    assert result["test_recall"] == pytest.approx(2 / 3, abs=1e-4)
