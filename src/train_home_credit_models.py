"""Consolidated training pipeline for all three Home Credit-family models: build data -> train ->
calibrate -> cross-validate -> fairness audit -> registry, in one file, selected with --variant.

Replaces 5 previously-scattered files (train_synth_model.py, train_real_data_model.py,
train_real_rich_model.py, cross_validate_real.py, fairness_mitigation_real_rich.py) that shared
almost identical structure -- same 4-way split, same XGBoost-OOF-then-LSTM-fusion approach, same
isotonic calibration, same threshold search -- with only the feature set and a few hyperparameters
differing per variant. This file is a faithful consolidation, not a rewrite: every hyperparameter,
split ratio, and threshold-search range below was copied from the script it replaces and verified
to reproduce identical metrics (see the verification note in docs/model_creation_summary.md).

Deliberately NOT folded in here (genuinely distinct or genuinely shared, not duplicated logic):
  - src/synth_data.py, src/build_real_data.py, src/build_real_data_rich.py -- each variant's own
    data builder; still imported from here, not copy-pasted, since synth_data.py is also imported
    directly by tests/test_scoring.py and tests/test_drift_monitor.py.
  - src/hybrid_model.py -- the shared XGBoost+LSTM fusion architecture class.

Run:
  uv run python src/train_home_credit_models.py --variant synthetic
  uv run python src/train_home_credit_models.py --variant real
  uv run python src/train_home_credit_models.py --variant real_rich
  uv run python src/train_home_credit_models.py --variant real --skip-cv   # faster iteration
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier  # noqa: I001  (xgboost before torch -- OpenMP conflict)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from build_model_registry import build as build_registry
from build_real_data import SEQ_COLS as REAL_SEQ_COLS
from build_real_data import STATIC_COLS as REAL_STATIC_COLS
from build_real_data import build as build_real
from build_real_data_rich import RICH_COLS, build_rich
from hybrid_model import Hybrid
from synth_data import SEQ_COLS as SYNTH_SEQ_COLS
from synth_data import STATIC_COLS as SYNTH_STATIC_COLS
from synth_data import generate as build_synthetic

warnings.filterwarnings("ignore")
HIDDEN = 32
COST_FN = 5.0   # cost of approving a defaulter
COST_FP = 1.0   # cost of declining a good customer (5:1, user-chosen)
SEED = 42

# --- per-variant configuration: every value here was read directly out of the script it
# replaces, not re-derived -- see the module docstring for why this matters ---

VARIANTS = {
    "synthetic": {
        "model_dir": "models",
        "data_source": "SYNTHETIC (src/synth_data.py)",
        "build_fn": lambda: build_synthetic(150_000, seed=SEED),
        "static_cols": SYNTH_STATIC_COLS, "seq_cols": SYNTH_SEQ_COLS,
        "xgb_params": dict(n_estimators=350, max_depth=4),
        "approve_floor": 0.05,
        "threshold_ts": np.linspace(0.02, 0.98, 481),
        "threshold_policy": "cost_based",
    },
    "real": {
        "model_dir": "models_real",
        "data_source": "REAL Home Credit (application_train + bureau + installments_payments)",
        "build_fn": build_real,
        "static_cols": REAL_STATIC_COLS, "seq_cols": REAL_SEQ_COLS,
        "xgb_params": dict(n_estimators=350, max_depth=4),
        "approve_floor": 0.02,
        "threshold_ts": np.linspace(0.01, 0.60, 591),
        "threshold_policy": "recall_target", "target_recall": 0.65,
    },
    "real_rich": {
        "model_dir": "models_real_rich",
        "data_source": "REAL Home Credit, ENRICHED (all 7 tables: application, bureau, "
                       "bureau_balance, previous_application, POS_CASH_balance, "
                       "credit_card_balance, installments_payments)",
        "build_fn": build_rich,
        "static_cols": REAL_STATIC_COLS + RICH_COLS, "seq_cols": REAL_SEQ_COLS,
        "xgb_params": dict(n_estimators=400, max_depth=5),
        "approve_floor": 0.01,
        "threshold_ts": np.linspace(0.005, 0.60, 1191),
        "threshold_policy": "recall_target", "target_recall": 0.65,
    },
}


def _split_indices(y: np.ndarray, seed: int = SEED):
    """Train 55% / val 15% / cal 15% / test 15%, stratified -- identical ratios across all
    three variants in the original scripts."""
    idx = np.arange(len(y))
    i_tr, i_tmp = train_test_split(idx, test_size=0.45, stratify=y, random_state=seed)
    i_val, i_tmp2 = train_test_split(i_tmp, test_size=0.667, stratify=y[i_tmp], random_state=seed)
    i_cal, i_test = train_test_split(i_tmp2, test_size=0.5, stratify=y[i_tmp2], random_state=seed)
    return i_tr, i_val, i_cal, i_test


def _xgb_static_branch(Xs: np.ndarray, y: np.ndarray, i_tr, i_val, i_cal, i_test, xgb_params: dict):
    spw = (y[i_tr] == 0).sum() / (y[i_tr] == 1).sum()
    params = dict(learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                 scale_pos_weight=spw, n_jobs=-1, random_state=SEED, **xgb_params)
    oof_tr = cross_val_predict(XGBClassifier(**params), Xs[i_tr], y[i_tr], cv=5,
                               method="predict_proba")[:, 1]
    xgb = XGBClassifier(**params).fit(Xs[i_tr], y[i_tr])
    xgb_score = {"tr": oof_tr, "val": xgb.predict_proba(Xs[i_val])[:, 1],
                 "cal": xgb.predict_proba(Xs[i_cal])[:, 1], "test": xgb.predict_proba(Xs[i_test])[:, 1]}
    return xgb, xgb_score, spw


def _train_hybrid(seq, xgb_score, y, i_tr, i_val, i_cal, i_test, spw: float, epochs: int = 30):
    """LSTM + fusion MLP (PyTorch, training only), early-stopped on best validation AUC.
    epochs=30 for a full training run, epochs=20 inside cross-validation (5x more retrains --
    this was the original scripts' own speed/rigor tradeoff, preserved here, not a new one)."""
    def tens(a):
        return torch.tensor(a)
    seq_t = {k: tens(seq[i]) for k, i in [("tr", i_tr), ("val", i_val), ("cal", i_cal), ("test", i_test)]}
    sc_t = {k: tens(xgb_score[k]).float().unsqueeze(1) for k in xgb_score}
    ytr_t = tens(y[i_tr]).unsqueeze(1)

    model = Hybrid(HIDDEN)
    crit = nn.BCEWithLogitsLoss(pos_weight=tens([spw]).float())
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    loader = DataLoader(TensorDataset(seq_t["tr"], sc_t["tr"], ytr_t), batch_size=512, shuffle=True)
    best, best_state = 0.0, None
    for _epoch in range(epochs):
        model.train()
        for xb, sb, yb in loader:
            opt.zero_grad(); crit(model(xb, sb), yb).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            va = roc_auc_score(y[i_val], torch.sigmoid(model(seq_t["val"], sc_t["val"])).numpy().ravel())
        if va > best:
            best, best_state = va, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()

    def raw_pd(split):
        with torch.no_grad():
            return torch.sigmoid(model(seq_t[split], sc_t[split])).numpy().ravel()
    return model, raw_pd


def _calibrate(raw_cal: np.ndarray, y_cal: np.ndarray):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw_cal, y_cal)
    grid_x = np.linspace(0.0, 1.0, 201)
    grid_y = iso.predict(grid_x)
    return grid_x, grid_y


def _cost_based_threshold(pc: np.ndarray, yc: np.ndarray, ts: np.ndarray, approve_floor: float) -> dict:
    costs = [COST_FN * ((pc < t) & (yc == 1)).sum() + COST_FP * ((pc >= t) & (yc == 0)).sum() for t in ts]
    t_decline = float(ts[int(np.argmin(costs))])
    t_approve = round(max(approve_floor, t_decline * 0.5), 3)
    return {"approve": t_approve, "decline": round(t_decline, 3)}


def _recall_target_threshold(pc: np.ndarray, yc: np.ndarray, target_recall: float,
                             approve_floor: float) -> dict:
    """Pick the decline threshold whose caught-defaulter rate on the cal split is closest to
    target_recall. This was previously a one-off manual script; now real, tested code, since
    it's the policy actually deployed for models_real/models_real_rich (catch 60-70% of
    defaulters, per explicit business choice)."""
    ts = np.linspace(0.005, 0.5, 991)
    good_default_mask = yc == 1
    recalls = np.array([(pc[good_default_mask] >= t).mean() for t in ts])
    t_decline = float(ts[int(np.argmin(np.abs(recalls - target_recall)))])
    t_approve = round(max(approve_floor, t_decline * 0.5), 3)
    return {"approve": t_approve, "decline": round(t_decline, 3)}


def _fairness_audit(df, y_test, rec_test, i_test) -> dict:
    """Demographic parity + equalized odds by gender/region, held OUT of scoring -- identical
    methodology across all three variants."""
    gender_test = df["gender"].values[i_test]
    region_test = df["region"].values[i_test]
    approved = rec_test != "decline"
    fairness = {}
    for attr_name, attr_vals in [("gender", gender_test), ("region", region_test)]:
        groups = sorted(set(attr_vals.tolist()))
        rows = {}
        for g in groups:
            m = attr_vals == g
            good_m = m & (y_test == 0)
            bad_m = m & (y_test == 1)
            tpr = float(approved[good_m].mean()) if good_m.sum() else None
            fpr = float(approved[bad_m].mean()) if bad_m.sum() else None
            rows[str(g)] = {
                "n": int(m.sum()), "approval_rate": round(float(approved[m].mean()), 4),
                "tpr_good_approved": round(tpr, 4) if tpr is not None else None,
                "fpr_bad_approved": round(fpr, 4) if fpr is not None else None,
            }
        rates = [r["approval_rate"] for r in rows.values()]
        tprs = [r["tpr_good_approved"] for r in rows.values() if r["tpr_good_approved"] is not None]
        fairness[attr_name] = {
            "by_group": rows,
            "demographic_parity_gap": round(max(rates) - min(rates), 4),
            "equalized_odds_gap": round(max(tprs) - min(tprs), 4) if len(tprs) > 1 else 0.0,
        }
    return fairness


def cross_validate(variant: str, n_folds: int = 5) -> dict:
    """5-fold stratified CV, full retrain per fold (20 epochs/fold vs 30 for the main run --
    the original scripts' own tradeoff for 5x the retrains). Proves the single-split numbers
    are stable, not a lucky split."""
    cfg = VARIANTS[variant]
    df = cfg["build_fn"]()
    y = df["default"].values.astype("float32")
    Xs = df[cfg["static_cols"]].values.astype("float32")
    seq_raw = df[cfg["seq_cols"]].values.astype("float32")
    seq_mean, seq_std = float(seq_raw.mean()), float(seq_raw.std())
    seq = ((seq_raw - seq_mean) / seq_std)[:, :, None]

    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_metrics = []
    print(f"\n{'fold':<6}{'n_test':<9}{'test_default%':<16}{'AUC':<10}{'Brier':<10}"
          f"{'Recall':<10}{'Precision':<10}")
    for fold_i, (train_full_idx, test_idx) in enumerate(kfold.split(Xs, y), start=1):
        y_train_full = y[train_full_idx]
        i_tr_rel, i_tmp_rel = train_test_split(
            np.arange(len(train_full_idx)), test_size=0.30, stratify=y_train_full, random_state=SEED)
        i_val_rel, i_cal_rel = train_test_split(
            i_tmp_rel, test_size=0.5, stratify=y_train_full[i_tmp_rel], random_state=SEED)
        i_tr = train_full_idx[i_tr_rel]
        i_val = train_full_idx[i_val_rel]
        i_cal = train_full_idx[i_cal_rel]

        xgb, xgb_score, spw = _xgb_static_branch(Xs, y, i_tr, i_val, i_cal, test_idx, cfg["xgb_params"])
        model, raw_pd = _train_hybrid(seq, xgb_score, y, i_tr, i_val, i_cal, test_idx, spw, epochs=20)
        grid_x, grid_y = _calibrate(raw_pd("cal"), y[i_cal])
        pc = np.interp(raw_pd("cal"), grid_x, grid_y)
        if cfg["threshold_policy"] == "recall_target":
            thresholds = _recall_target_threshold(pc, y[i_cal], cfg["target_recall"], cfg["approve_floor"])
        else:
            thresholds = _cost_based_threshold(pc, y[i_cal], cfg["threshold_ts"], cfg["approve_floor"])

        cal_test = np.interp(raw_pd("test"), grid_x, grid_y)
        y_test = y[test_idx]
        pred = (cal_test >= thresholds["decline"]).astype(int)
        auc = roc_auc_score(y_test, cal_test)
        brier = brier_score_loss(y_test, cal_test)
        rec = recall_score(y_test, pred, zero_division=0)
        prec = precision_score(y_test, pred, zero_division=0)
        print(f"{fold_i:<6}{len(test_idx):<9}{y_test.mean():<16.4%}{auc:<10.4f}{brier:<10.4f}"
              f"{rec:<10.4f}{prec:<10.4f}")
        fold_metrics.append({"auc": auc, "brier": brier, "recall": rec, "precision": prec})

    aucs = [m["auc"] for m in fold_metrics]
    recalls = [m["recall"] for m in fold_metrics]
    summary = {
        "n_folds": n_folds,
        "auc_mean": round(float(np.mean(aucs)), 4), "auc_std": round(float(np.std(aucs)), 4),
        "auc_min": round(float(min(aucs)), 4), "auc_max": round(float(max(aucs)), 4),
        "recall_mean": round(float(np.mean(recalls)), 4), "recall_std": round(float(np.std(recalls)), 4),
    }
    print(f"\nAUC across {n_folds} folds: mean={summary['auc_mean']:.4f} std={summary['auc_std']:.4f} "
          f"range=[{summary['auc_min']:.4f}, {summary['auc_max']:.4f}]")
    return summary


def train_variant(variant: str, run_cv: bool = True) -> dict:
    cfg = VARIANTS[variant]
    torch.manual_seed(SEED); np.random.seed(SEED)
    print(f"=== training '{variant}' -> {cfg['model_dir']}/ ===")
    print(f"data source: {cfg['data_source']}")

    df = cfg["build_fn"]()
    y = df["default"].values.astype("float32")
    Xs = df[cfg["static_cols"]].values.astype("float32")
    seq_raw = df[cfg["seq_cols"]].values.astype("float32")
    seq_mean, seq_std = float(seq_raw.mean()), float(seq_raw.std())
    seq = ((seq_raw - seq_mean) / seq_std)[:, :, None]
    print(f"n={len(df)}, default rate={y.mean():.4f}, static features={len(cfg['static_cols'])}")

    i_tr, i_val, i_cal, i_test = _split_indices(y)
    print(f"splits: train {len(i_tr)}  val {len(i_val)}  cal {len(i_cal)}  test {len(i_test)}")

    xgb, xgb_score, spw = _xgb_static_branch(Xs, y, i_tr, i_val, i_cal, i_test, cfg["xgb_params"])
    print(f"XGBoost static AUC (test): {roc_auc_score(y[i_test], xgb_score['test']):.4f}")

    model, raw_pd = _train_hybrid(seq, xgb_score, y, i_tr, i_val, i_cal, i_test, spw, epochs=30)
    print(f"fused (uncalibrated) AUC (test): {roc_auc_score(y[i_test], raw_pd('test')):.4f}")

    grid_x, grid_y = _calibrate(raw_pd("cal"), y[i_cal])
    def calibrate(p): return np.interp(p, grid_x, grid_y)
    cal_test = calibrate(raw_pd("test"))
    brier_cal = brier_score_loss(y[i_test], cal_test)
    final_auc = roc_auc_score(y[i_test], cal_test)
    print(f"calibrated AUC (test): {final_auc:.4f}  |  Brier: {brier_cal:.4f}")
    print(f"mean predicted PD: {cal_test.mean():.4f}  vs actual rate: {y[i_test].mean():.4f}")

    pc, yc = calibrate(raw_pd("cal")), y[i_cal]
    if cfg["threshold_policy"] == "recall_target":
        thresholds = _recall_target_threshold(pc, yc, cfg["target_recall"], cfg["approve_floor"])
        policy_note = (f"Recall-target policy: catch ~{cfg['target_recall']:.0%} of defaulters, "
                       f"per explicit business choice. See reports/real_data_model_report.md.")
    else:
        thresholds = _cost_based_threshold(pc, yc, cfg["threshold_ts"], cfg["approve_floor"])
        policy_note = f"Cost-optimal threshold, {COST_FN:.0f}:{COST_FP:.0f} cost ratio."
    print(f"thresholds ({cfg['threshold_policy']}): approve < {thresholds['approve']}  "
          f"decline >= {thresholds['decline']}")

    rec_test = np.where(cal_test >= thresholds["decline"], "decline",
                        np.where(cal_test >= thresholds["approve"], "review", "approve"))
    pred = (cal_test >= thresholds["decline"]).astype(int)
    rec = recall_score(y[i_test], pred, zero_division=0)
    prec = precision_score(y[i_test], pred, zero_division=0)
    print(f"test recall: {rec:.4f}  precision: {prec:.4f}")

    fairness = _fairness_audit(df, y[i_test], rec_test, i_test)
    for attr, f in fairness.items():
        print(f"fairness ({attr}): demographic_parity_gap={f['demographic_parity_gap']:.4f}  "
              f"equalized_odds_gap={f['equalized_odds_gap']:.4f}")

    cv_summary = cross_validate(variant) if run_cv else None

    ranges = {}
    for c in cfg["static_cols"]:
        col = df[c].values
        ranges[c] = {"min": float(np.percentile(col, 1)), "max": float(np.percentile(col, 99)),
                     "median": float(np.median(col))}

    model_dir = ROOT / cfg["model_dir"]
    model_dir.mkdir(exist_ok=True)
    joblib.dump(xgb, model_dir / "hybrid_xgb.joblib")
    sd = model.state_dict()
    np.savez(model_dir / "hybrid_fusion.npz",
             W_ih=sd["lstm.weight_ih_l0"].numpy(), W_hh=sd["lstm.weight_hh_l0"].numpy(),
             b_ih=sd["lstm.bias_ih_l0"].numpy(), b_hh=sd["lstm.bias_hh_l0"].numpy(),
             h0w=sd["head.0.weight"].numpy(), h0b=sd["head.0.bias"].numpy(),
             h2w=sd["head.2.weight"].numpy(), h2b=sd["head.2.bias"].numpy())

    config = {
        "data_source": cfg["data_source"], "n_total": len(df),
        "static_cols": cfg["static_cols"], "seq_len": len(cfg["seq_cols"]),
        "seq_mean": seq_mean, "seq_std": seq_std, "lstm_hidden": HIDDEN,
        "calibration": {"grid_x": grid_x.tolist(), "grid_y": grid_y.round(6).tolist()},
        "thresholds": thresholds, "cost_ratio": COST_FN / COST_FP,
        "threshold_policy": {
            "method": cfg["threshold_policy"], "note": policy_note,
            **({"target_recall": cfg["target_recall"]} if cfg["threshold_policy"] == "recall_target" else {}),
        },
        "feature_ranges": ranges,
        "metrics": {
            "test_auc": round(float(final_auc), 4), "test_brier": round(float(brier_cal), 4),
            "test_accuracy": round(float((pred == y[i_test]).mean()), 4),
            "test_precision": round(float(prec), 4), "test_recall": round(float(rec), 4),
            "test_default_rate": round(float(y[i_test].mean()), 4),
            "test_mean_predicted_pd": round(float(cal_test.mean()), 4),
        },
        "fairness_audit": fairness,
        "cross_validation": cv_summary,
    }
    (model_dir / "hybrid_config.json").write_text(json.dumps(config, indent=2))

    # verify NumPy reimplementation matches PyTorch (torch-free serving depends on this)
    d = np.load(model_dir / "hybrid_fusion.npz")
    def np_forward(s, score, H=HIDDEN):
        h = np.zeros(H); c = np.zeros(H)
        for v in s:
            g = d["W_ih"] @ np.array([v]) + d["b_ih"] + d["W_hh"] @ h + d["b_hh"]
            i_, f_, gg, o = g[:H], g[H:2*H], g[2*H:3*H], g[3*H:4*H]
            i_, f_, gg, o = 1/(1+np.exp(-i_)), 1/(1+np.exp(-f_)), np.tanh(gg), 1/(1+np.exp(-o))
            c = f_*c + i_*gg; h = o*np.tanh(c)
        z = np.concatenate([h, [score]]); z1 = np.maximum(0, d["h0w"] @ z + d["h0b"])
        return float(1/(1+np.exp(-(d["h2w"] @ z1 + d["h2b"])[0])))

    raw_test_arr = raw_pd("test")
    md = 0.0
    for j in range(min(300, len(i_test))):
        md = max(md, abs(raw_test_arr[j] - np_forward(seq[i_test][j].ravel(), xgb_score["test"][j])))
    print(f"max |torch - numpy| over 300 test cases = {md:.2e}  {'OK' if md < 1e-5 else 'MISMATCH'}")
    print(f"saved -> {model_dir}/hybrid_xgb.joblib, hybrid_fusion.npz, hybrid_config.json")

    registry = build_registry()
    (ROOT / "model_registry.json").write_text(json.dumps(registry, indent=2) + "\n")
    print(f"registered: {registry['models'][cfg['model_dir']]['fingerprint']}")

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=[*VARIANTS.keys(), "all"], default="all")
    parser.add_argument("--skip-cv", action="store_true", help="skip the 5-fold CV pass (faster iteration)")
    args = parser.parse_args()
    targets = list(VARIANTS.keys()) if args.variant == "all" else [args.variant]
    for v in targets:
        train_variant(v, run_cv=not args.skip_cv)
