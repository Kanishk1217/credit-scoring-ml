"""Test whether group-aware (per-gender, per-region) decline thresholds can close the
equalized-odds gap that opened up in the enriched real-data model (EXT_SOURCE features),
without retraining -- reuses the already-trained models_real_rich/ artifacts.

Mitigation approach: instead of one global decline threshold, pick a threshold per group that
equalizes TPR-on-good-borrowers (equalized odds) at a matched overall approval rate, then report
the real before/after cost in accuracy/precision/recall so the tradeoff is visible, not asserted.

Run:  uv run python src/fairness_mitigation_real_rich.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from build_real_data import SEQ_COLS, STATIC_COLS
from build_real_data_rich import RICH_COLS, build_rich

MODELS_DIR = ROOT / "models_real_rich"
HIDDEN = 32
ALL_STATIC = STATIC_COLS + RICH_COLS

print("Rebuilding enriched real dataset (same split as training, random_state=42)...")
df = build_rich()
y = df["default"].values.astype("float32")
Xs = df[ALL_STATIC].values.astype("float32")
seq_raw = df[SEQ_COLS].values.astype("float32")

config = json.loads((MODELS_DIR / "hybrid_config.json").read_text())
seq_mean, seq_std = config["seq_mean"], config["seq_std"]
seq = ((seq_raw - seq_mean) / seq_std)

idx = np.arange(len(y))
i_tr, i_tmp = train_test_split(idx, test_size=0.45, stratify=y, random_state=42)
i_val, i_tmp2 = train_test_split(i_tmp, test_size=0.667, stratify=y[i_tmp], random_state=42)
i_cal, i_test = train_test_split(i_tmp2, test_size=0.5, stratify=y[i_tmp2], random_state=42)

xgb = joblib.load(MODELS_DIR / "hybrid_xgb.joblib")
xgb_score_test = xgb.predict_proba(Xs[i_test])[:, 1]
xgb_score_cal = xgb.predict_proba(Xs[i_cal])[:, 1]

d = np.load(MODELS_DIR / "hybrid_fusion.npz")
def np_forward(s, score, H=HIDDEN):
    h = np.zeros(H); c = np.zeros(H)
    for v in s:
        g = d["W_ih"] @ np.array([v]) + d["b_ih"] + d["W_hh"] @ h + d["b_hh"]
        i_, f, gg, o = g[:H], g[H:2*H], g[2*H:3*H], g[3*H:4*H]
        i_, f, gg, o = 1/(1+np.exp(-i_)), 1/(1+np.exp(-f)), np.tanh(gg), 1/(1+np.exp(-o))
        c = f*c + i_*gg; h = o*np.tanh(c)
    z = np.concatenate([h, [score]]); z1 = np.maximum(0, d["h0w"] @ z + d["h0b"])
    return float(1/(1+np.exp(-(d["h2w"] @ z1 + d["h2b"])[0])))

grid_x = np.array(config["calibration"]["grid_x"])
grid_y = np.array(config["calibration"]["grid_y"])
def calibrate(p): return np.interp(p, grid_x, grid_y)

print("Scoring test + cal sets through the saved NumPy-reimplemented fusion head...")
raw_test = np.array([np_forward(seq[i].ravel(), s)
                     for i, s in zip(i_test, xgb_score_test, strict=True)])
raw_cal = np.array([np_forward(seq[i].ravel(), s)
                    for i, s in zip(i_cal, xgb_score_cal, strict=True)])
cal_test = calibrate(raw_test)
cal_cal = calibrate(raw_cal)
y_test = y[i_test]
y_cal = y[i_cal]

t_decline_global = config["thresholds"]["decline"]
gender_test = df["gender"].values[i_test]
gender_cal = df["gender"].values[i_cal]
region_test = df["region"].values[i_test]
region_cal = df["region"].values[i_cal]


def audit(pred, attr_vals, y_arr):
    groups = sorted(set(attr_vals.tolist()))
    rows = {}
    for g in groups:
        m = attr_vals == g
        good_m = m & (y_arr == 0)
        rows[g] = {
            "n": int(m.sum()),
            "approval_rate": float((~pred[m].astype(bool)).mean()),
            "tpr_good_approved": float((~pred[good_m].astype(bool)).mean()) if good_m.sum() else None,
        }
    rates = [r["approval_rate"] for r in rows.values()]
    tprs = [r["tpr_good_approved"] for r in rows.values() if r["tpr_good_approved"] is not None]
    dp_gap = round(max(rates) - min(rates), 4)
    eo_gap = round(max(tprs) - min(tprs), 4) if len(tprs) > 1 else 0.0
    return rows, dp_gap, eo_gap


print(f"\n{'='*70}\nBASELINE: one global threshold ({t_decline_global}) for everyone\n{'='*70}")
pred_global = (cal_test >= t_decline_global).astype(int)
rec_global = recall_score(y_test, pred_global, zero_division=0)
prec_global = precision_score(y_test, pred_global, zero_division=0)
print(f"overall recall={rec_global:.4f}  precision={prec_global:.4f}")
for attr_name, test_vals in [("gender", gender_test), ("region", region_test)]:
    rows, dp, eo = audit(pred_global, test_vals, y_test)
    print(f"  {attr_name}: demographic_parity_gap={dp:.4f}  equalized_odds_gap={eo:.4f}")


def find_group_thresholds_for_target_tpr(target_tpr, attr_vals_cal, groups):
    """For each group, find the decline threshold on the CAL split whose approved-good-borrower
    rate (TPR) is closest to target_tpr, searching the same grid used for the global threshold."""
    ts = np.linspace(0.005, 0.60, 1191)
    out = {}
    for g in groups:
        m = attr_vals_cal == g
        good_m = m & (y_cal == 0)
        if good_m.sum() == 0:
            out[g] = t_decline_global
            continue
        tprs = np.array([(cal_cal[good_m] < t).mean() for t in ts])
        out[g] = float(ts[np.argmin(np.abs(tprs - target_tpr))])
    return out


print(f"\n{'='*70}\nMITIGATION: per-group thresholds equalizing TPR to the population-wide TPR\n{'='*70}")
GROUP_ATTRS = [("gender", gender_test, gender_cal), ("region", region_test, region_cal)]
for attr_name, test_vals, cal_vals in GROUP_ATTRS:
    groups = sorted(set(test_vals.tolist()))
    good_all_cal = y_cal == 0
    target_tpr = float((cal_cal[good_all_cal] < t_decline_global).mean())
    group_thresh = find_group_thresholds_for_target_tpr(target_tpr, cal_vals, groups)
    print(f"\n{attr_name} -- target TPR={target_tpr:.4f}, per-group thresholds: "
          f"{ {g: round(t,3) for g,t in group_thresh.items()} }")

    pred_mit = np.zeros(len(y_test), dtype=int)
    for g in groups:
        m = test_vals == g
        pred_mit[m] = (cal_test[m] >= group_thresh[g]).astype(int)

    rec_mit = recall_score(y_test, pred_mit, zero_division=0)
    prec_mit = precision_score(y_test, pred_mit, zero_division=0)
    rows, dp, eo = audit(pred_mit, test_vals, y_test)
    print(f"  after mitigation: overall recall={rec_mit:.4f} (was {rec_global:.4f})  "
          f"precision={prec_mit:.4f} (was {prec_global:.4f})")
    print(f"  {attr_name}: demographic_parity_gap={dp:.4f} (was baseline)  "
          f"equalized_odds_gap={eo:.4f} (was baseline)")
    for g, r in rows.items():
        print(f"    group {g}: n={r['n']:>6}  approval_rate={r['approval_rate']:.3f}  "
              f"TPR={r['tpr_good_approved']}")
