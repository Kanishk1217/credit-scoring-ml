"""One-file onboarding pipeline for training a new-market credit-risk model from real,
independently-sourced lending data: feature engineering -> training -> calibration ->
cost-based thresholds -> fairness audit -> 5-fold cross-validation -> confusion matrix ->
model registry entry, in one run.

Concrete first instance: historical LendingClub (US) loans, 2007-2011, resolved outcomes only.

This mirrors src/train_home_credit_models.py, which does the equivalent consolidation for the
three Home Credit-family models (synthetic/real/real_rich) -- both replace what used to be
several scattered per-model scripts with one file per data-source family that a future market or
model can extend, not clone.

Architecture note -- read this before assuming the hybrid XGBoost+LSTM architecture should be
reused: that model's LSTM branch exists specifically to read a genuine month-by-month payment
trajectory (Home Credit's installments_payments.csv). LendingClub's public schema has no such
trajectory. A hand-built proxy sequence tested earlier had real content in only 12/300 rows --
fabricating one and feeding it through an LSTM would inject fake signal, not real. This script
trains a PLAIN XGBoost classifier on rich real tabular features instead. That is the honest,
appropriate choice for this data, not a shortcut.

Data leakage guard: this file hard-asserts that none of LendingClub's post-origination fields
(payment history, recoveries, LendingClub's own risk grade) ever reach the feature matrix -- see
LC_LEAKAGE_BLOCKLIST and the assertion in engineer_features().

Run:  uv run python src/train_new_market.py data/raw/lendingclub/loan.csv --market lendingclub
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from build_model_registry import build as build_registry  # noqa: E402

COST_FN = 5.0   # cost of approving a defaulter
COST_FP = 1.0   # cost of declining a good customer
SEED = 42

# --- LendingClub-specific mapping (a future market gets its own small block like this one;
# this is intentionally NOT a generic feature-engineering DSL -- see plan critique: over-
# generalizing before a second real market exists just recreates the "scattered mess" problem
# one abstraction layer up) ---

LC_RESOLVED_STATUSES = {"Fully Paid": 0, "Charged Off": 1}

LC_NUMERIC_PASSTHROUGH = [
    "loan_amnt", "annual_inc", "dti", "open_acc", "revol_bal", "revol_util",
    "total_acc", "delinq_2yrs", "inq_last_6mths", "pub_rec",
]
LC_CATEGORICAL = ["home_ownership", "verification_status", "purpose"]

# Fields that are consequences of the loan's outcome (post-origination), or LendingClub's own
# proprietary risk assessment (grade/sub_grade/int_rate -- using these just teaches the model to
# copy an existing score instead of being independently predictive). Must NEVER be features.
LC_LEAKAGE_BLOCKLIST = [
    "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d",
    "last_credit_pull_d", "out_prncp", "out_prncp_inv", "funded_amnt", "funded_amnt_inv",
    "policy_code", "grade", "sub_grade", "int_rate", "installment", "url", "desc", "title",
    "zip_code", "id", "member_id", "loan_status", "pymnt_plan", "initial_list_status",
]

US_CENSUS_REGION = {
    **dict.fromkeys(["CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"], "Northeast"),
    **dict.fromkeys(["IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"], "Midwest"),
    **dict.fromkeys(["DE", "FL", "GA", "MD", "NC", "SC", "VA", "DC", "WV", "AL", "KY", "MS",
                     "TN", "AR", "LA", "OK", "TX"], "South"),
    **dict.fromkeys(["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "AK", "CA", "HI", "OR", "WA"], "West"),
}

EMP_LENGTH_MAP = {
    "< 1 year": 0.5, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
    "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9, "10+ years": 10,
}

MIN_FAIRNESS_GROUP_SIZE = 30  # below this, a group's fairness numbers are too noisy to report


def load_and_filter(csv_path: Path) -> pd.DataFrame:
    """Load raw CSV, keep only loans with a RESOLVED outcome. Everything else (Current, Late,
    In Grace Period, ...) is censored/unresolved -- silently treating "not Charged Off" as good
    would mislabel unresolved loans as goods, a real labeling bug, not just noise."""
    df = pd.read_csv(csv_path, low_memory=False)
    n_raw = len(df)
    unresolved = sorted(set(df["loan_status"].dropna().unique()) - set(LC_RESOLVED_STATUSES))
    df = df[df["loan_status"].isin(LC_RESOLVED_STATUSES)].copy()
    df["default"] = df["loan_status"].map(LC_RESOLVED_STATUSES).astype(int)
    assert df["loan_status"].isin(LC_RESOLVED_STATUSES).all()
    print(f"loaded {n_raw} raw rows; kept {len(df)} with a resolved outcome "
          f"(dropped unresolved statuses: {unresolved})")
    print(f"actual default rate: {df['default'].mean():.4f}  "
          f"({df['default'].sum()} charged off / {(df['default'] == 0).sum()} fully paid)")
    return df


def _fix_two_digit_year_dates(df: pd.DataFrame) -> pd.DataFrame:
    """"Sep-68"-style two-digit years parse to 2068, not 1968, for any year before ~the current
    century boundary. Verified on this file: naive parsing puts real rows' earliest_cr_line AFTER
    issue_d, which is impossible -- a credit line can't postdate the loan that used it as
    underwriting history. Fix: wherever that happens, the parsed year is off by a century."""
    df["_issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%y")
    df["_earliest_cr_line"] = pd.to_datetime(df["earliest_cr_line"], format="%b-%y")
    wrapped = df["_earliest_cr_line"] > df["_issue_d"]
    n_wrapped = int(wrapped.sum())
    df.loc[wrapped, "_earliest_cr_line"] -= pd.DateOffset(years=100)
    assert (df["_earliest_cr_line"] <= df["_issue_d"]).all(), \
        "credit history must predate loan issuance -- century fix did not fully resolve wraparound"
    print(f"fixed {n_wrapped} two-digit-year wraparounds in earliest_cr_line "
          f"(out of {len(df)} rows, {n_wrapped / len(df):.2%})")
    return df


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], str]:
    """Returns (dataframe with feature columns + default + region, feature_cols, fairness_col).
    region is carried for the fairness audit only -- it is never in feature_cols."""
    df = _fix_two_digit_year_dates(df.copy())
    df["credit_history_years"] = (df["_issue_d"] - df["_earliest_cr_line"]).dt.days / 365.25

    # revol_util is stored as a percent-formatted string ("83.70%"), not numeric -- strip the
    # sign and coerce; true missing stays NaN (not coerced to 0, same reasoning as employment_years).
    df["revol_util"] = df["revol_util"].astype(str).str.rstrip("%").replace("nan", np.nan).astype(float)

    # Employment length: ordinal buckets -> numeric years. True missing ("n/a") stays NaN, NOT
    # coerced to 0 -- 0 would misleadingly mean "no job history," matching this project's existing
    # ext_source_* precedent of trusting XGBoost's native missing-value handling.
    df["employment_years"] = df["emp_length"].map(EMP_LENGTH_MAP)

    # Geographic fairness-audit dimension ONLY -- never a scoring feature. Not a true protected
    # attribute (same caveat already used for Home Credit's REGION_RATING_CLIENT elsewhere in
    # this project): a rough proxy, illustrative, not a substitute for a real demographic audit.
    df["region"] = df["addr_state"].map(US_CENSUS_REGION)

    numeric_cols = LC_NUMERIC_PASSTHROUGH + ["employment_years", "credit_history_years"]
    cat_dummies = pd.get_dummies(df[LC_CATEGORICAL], prefix=LC_CATEGORICAL, dtype="float32")
    feature_cols = numeric_cols + list(cat_dummies.columns)

    leaked = set(feature_cols) & set(LC_LEAKAGE_BLOCKLIST)
    assert not leaked, f"leakage guard failed -- blocklisted columns reached the feature set: {leaked}"

    features = pd.concat(
        [df[numeric_cols].reset_index(drop=True), cat_dummies.reset_index(drop=True)], axis=1)
    out = pd.concat(
        [features, df[["default", "region"]].reset_index(drop=True)], axis=1)
    print(f"engineered {len(feature_cols)} features "
          f"({len(numeric_cols)} numeric + {len(cat_dummies.columns)} one-hot categorical)")
    return out, feature_cols, "region"


def _split_indices(y: np.ndarray, seed: int = SEED):
    """Train 55% / val 15% / cal 15% / test 15%, stratified -- matches this project's existing
    4-way split convention (see src/train_home_credit_models.py)."""
    idx = np.arange(len(y))
    i_tr, i_tmp = train_test_split(idx, test_size=0.45, stratify=y, random_state=seed)
    i_val, i_tmp2 = train_test_split(i_tmp, test_size=0.667, stratify=y[i_tmp], random_state=seed)
    i_cal, i_test = train_test_split(i_tmp2, test_size=0.5, stratify=y[i_tmp2], random_state=seed)
    return i_tr, i_val, i_cal, i_test


def _fit_xgb(X: np.ndarray, y: np.ndarray, i_tr: np.ndarray, i_val: np.ndarray, seed: int = SEED):
    spw = (y[i_tr] == 0).sum() / (y[i_tr] == 1).sum()
    xgb = XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, n_jobs=-1, random_state=seed,
        early_stopping_rounds=30, eval_metric="auc",
    )
    xgb.fit(X[i_tr], y[i_tr], eval_set=[(X[i_val], y[i_val])], verbose=False)
    return xgb


def _calibrate(xgb, X, y, i_cal):
    raw_cal = xgb.predict_proba(X[i_cal])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw_cal, y[i_cal])
    grid_x = np.linspace(0.0, 1.0, 201)
    grid_y = iso.predict(grid_x)
    return grid_x, grid_y


def _cost_optimal_threshold(cal_cal: np.ndarray, y_cal: np.ndarray) -> dict:
    """Threshold search on the CAL split only -- never test. Matches this project's established
    cost-based policy (COST_FN/COST_FP = 5:1), not the recall-target policy used on models_real/
    models_real_rich (that was a post-hoc manual config edit, not a tested code path -- not worth
    replicating for a single new model; see the plan critique)."""
    ts = np.linspace(0.005, 0.6, 1191)
    costs = [COST_FN * ((cal_cal < t) & (y_cal == 1)).sum()
             + COST_FP * ((cal_cal >= t) & (y_cal == 0)).sum() for t in ts]
    decline_t = float(ts[int(np.argmin(costs))])
    approve_t = round(max(0.01, decline_t * 0.5), 4)
    return {"approve": approve_t, "decline": round(decline_t, 4)}


def evaluate(y_true: np.ndarray, calibrated_pd: np.ndarray, decline_t: float) -> dict:
    """Confusion matrix + full metrics, all computed live from the given arrays -- nothing here
    is a literal/hardcoded number."""
    pred = (calibrated_pd >= decline_t).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    n = len(y_true)
    return {
        "confusion_matrix": {"true_positive": tp, "false_positive": fp,
                             "true_negative": tn, "false_negative": fn},
        "n_test": n,
        "test_default_rate": round(float(y_true.mean()), 4),
        "test_accuracy": round((tp + tn) / n, 4),
        "test_precision": round(tp / (tp + fp), 4) if (tp + fp) else 0.0,
        "test_recall": round(tp / (tp + fn), 4) if (tp + fn) else 0.0,
        "test_auc": round(float(roc_auc_score(y_true, calibrated_pd)), 4),
        "test_brier": round(float(brier_score_loss(y_true, calibrated_pd)), 4),
        "test_mean_predicted_pd": round(float(calibrated_pd.mean()), 4),
    }


def fairness_audit(region: np.ndarray, y_true: np.ndarray, approved: np.ndarray) -> dict:
    """Demographic-parity + equalized-odds gap by region, same methodology used elsewhere in this
    project for Home Credit's gender/region audit. Groups below MIN_FAIRNESS_GROUP_SIZE are
    flagged, not silently included as if their numbers were stable."""
    groups = sorted(set(region.tolist()))
    rows, thin_groups = {}, []
    for g in groups:
        m = region == g
        n = int(m.sum())
        if n < MIN_FAIRNESS_GROUP_SIZE:
            thin_groups.append(g)
        good_m = m & (y_true == 0)
        approval_rate = float(approved[m].mean()) if n else None
        tpr = float(approved[good_m].mean()) if good_m.sum() else None
        rows[g] = {"n": n, "approval_rate": round(approval_rate, 4) if approval_rate is not None else None,
                   "tpr_good_approved": round(tpr, 4) if tpr is not None else None}
    rates = [r["approval_rate"] for r in rows.values() if r["approval_rate"] is not None]
    tprs = [r["tpr_good_approved"] for r in rows.values() if r["tpr_good_approved"] is not None]
    return {
        "by_group": rows,
        "demographic_parity_gap": round(max(rates) - min(rates), 4) if len(rates) > 1 else 0.0,
        "equalized_odds_gap": round(max(tprs) - min(tprs), 4) if len(tprs) > 1 else 0.0,
        "thin_groups_below_min_size": thin_groups,
        "min_group_size_threshold": MIN_FAIRNESS_GROUP_SIZE,
    }


def cross_validate(df: pd.DataFrame, feature_cols: list[str], n_folds: int = 5, seed: int = SEED) -> dict:
    """5-fold stratified CV, full retrain per fold -- proves the single-split numbers are stable,
    not a lucky split. member_id was verified unique across all rows (no repeated borrowers), so
    plain StratifiedKFold has no group-leakage risk here."""
    y = df["default"].values.astype("float32")
    X = df[feature_cols].values.astype("float32")

    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_metrics = []
    print(f"\n{'fold':<6}{'n_test':<9}{'test_default%':<16}{'AUC':<10}{'Brier':<10}"
          f"{'Recall':<10}{'Precision':<10}")
    for fold_i, (train_full_idx, test_idx) in enumerate(kfold.split(X, y), start=1):
        y_train_full = y[train_full_idx]
        i_tr_rel, i_tmp_rel = train_test_split(
            np.arange(len(train_full_idx)), test_size=0.30, stratify=y_train_full, random_state=seed)
        i_val_rel, i_cal_rel = train_test_split(
            i_tmp_rel, test_size=0.5, stratify=y_train_full[i_tmp_rel], random_state=seed)
        i_tr = train_full_idx[i_tr_rel]
        i_val = train_full_idx[i_val_rel]
        i_cal = train_full_idx[i_cal_rel]

        xgb = _fit_xgb(X, y, i_tr, i_val, seed=seed)
        grid_x, grid_y = _calibrate(xgb, X, y, i_cal)
        cal_cal = np.interp(xgb.predict_proba(X[i_cal])[:, 1], grid_x, grid_y)
        thresholds = _cost_optimal_threshold(cal_cal, y[i_cal])

        cal_test = np.interp(xgb.predict_proba(X[test_idx])[:, 1], grid_x, grid_y)
        y_test = y[test_idx]
        m = evaluate(y_test, cal_test, thresholds["decline"])
        print(f"{fold_i:<6}{m['n_test']:<9}{m['test_default_rate']:<16.4%}{m['test_auc']:<10.4f}"
              f"{m['test_brier']:<10.4f}{m['test_recall']:<10.4f}{m['test_precision']:<10.4f}")
        fold_metrics.append(m)

    aucs = [m["test_auc"] for m in fold_metrics]
    recalls = [m["test_recall"] for m in fold_metrics]
    summary = {
        "n_folds": n_folds,
        "auc_mean": round(float(np.mean(aucs)), 4), "auc_std": round(float(np.std(aucs)), 4),
        "auc_min": round(float(min(aucs)), 4), "auc_max": round(float(max(aucs)), 4),
        "recall_mean": round(float(np.mean(recalls)), 4), "recall_std": round(float(np.std(recalls)), 4),
        "per_fold": fold_metrics,
    }
    print(f"\nAUC across {n_folds} folds: mean={summary['auc_mean']:.4f} std={summary['auc_std']:.4f} "
          f"range=[{summary['auc_min']:.4f}, {summary['auc_max']:.4f}]")
    return summary


def train_market_model(csv_path: Path, market: str) -> dict:
    df = load_and_filter(csv_path)
    df, feature_cols, fairness_col = engineer_features(df)

    y = df["default"].values.astype("float32")
    X = df[feature_cols].values.astype("float32")
    region = df[fairness_col].values

    i_tr, i_val, i_cal, i_test = _split_indices(y)
    print(f"splits: train {len(i_tr)}  val {len(i_val)}  cal {len(i_cal)}  test {len(i_test)}")

    xgb = _fit_xgb(X, y, i_tr, i_val)
    print(f"XGBoost stopped at {xgb.best_iteration + 1} boosting rounds "
          f"(early stopping on validation AUC)")

    grid_x, grid_y = _calibrate(xgb, X, y, i_cal)
    cal_cal = np.interp(xgb.predict_proba(X[i_cal])[:, 1], grid_x, grid_y)
    thresholds = _cost_optimal_threshold(cal_cal, y[i_cal])
    print(f"cost-optimal thresholds ({COST_FN:.0f}:{COST_FP:.0f}): "
          f"approve < {thresholds['approve']}  decline >= {thresholds['decline']}")

    raw_test = xgb.predict_proba(X[i_test])[:, 1]
    cal_test = np.interp(raw_test, grid_x, grid_y)
    y_test = y[i_test]
    metrics = evaluate(y_test, cal_test, thresholds["decline"])
    cm = metrics["confusion_matrix"]
    print(f"\ntest confusion matrix: TP={cm['true_positive']} FP={cm['false_positive']} "
          f"TN={cm['true_negative']} FN={cm['false_negative']}")
    print(f"test AUC={metrics['test_auc']:.4f}  recall={metrics['test_recall']:.4f}  "
          f"precision={metrics['test_precision']:.4f}  accuracy={metrics['test_accuracy']:.4f}")

    approved_test = cal_test < thresholds["decline"]
    fairness = fairness_audit(region[i_test], y_test, approved_test)
    print(f"fairness ({fairness_col}): demographic_parity_gap={fairness['demographic_parity_gap']:.4f}  "
          f"equalized_odds_gap={fairness['equalized_odds_gap']:.4f}"
          + (f"  (thin groups: {fairness['thin_groups_below_min_size']})"
             if fairness["thin_groups_below_min_size"] else ""))

    cv_summary = cross_validate(df, feature_cols)

    model_dir = ROOT / f"models_{market}"
    model_dir.mkdir(exist_ok=True)
    joblib.dump(xgb, model_dir / "xgb.joblib")

    config = {
        "data_source": f"REAL LendingClub historical loans (2007-2011, resolved outcomes only) "
                       f"-- market={market}",
        "architecture": "plain XGBoost, no sequence/LSTM branch (see module docstring for why)",
        "n_total": len(df),
        "static_cols": feature_cols,
        "calibration": {"grid_x": grid_x.tolist(), "grid_y": grid_y.round(6).tolist()},
        "thresholds": thresholds,
        "cost_ratio": COST_FN / COST_FP,
        "metrics": metrics,
        "fairness_audit": {fairness_col: fairness},
        "cross_validation": cv_summary,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (model_dir / "model_config.json").write_text(json.dumps(config, indent=2))
    print(f"\nsaved -> {model_dir}/xgb.joblib, model_config.json")

    registry = build_registry()
    (ROOT / "model_registry.json").write_text(json.dumps(registry, indent=2) + "\n")
    print(f"registered in model_registry.json: {registry['models'][f'models_{market}']['fingerprint']}")

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="raw lending data CSV")
    parser.add_argument("--market", default="lendingclub", help="model directory suffix: models_<market>")
    args = parser.parse_args()
    train_market_model(args.csv_path, args.market)
