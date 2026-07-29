"""Train the hybrid model on REAL Home Credit data (not synthetic).

Identical pipeline to train_synth_model.py (same architecture, same calibration methodology,
same cost-based thresholds, same fairness audit), swapping only the data source -- proving the
"data-source-agnostic pipeline" promise made from day one of this project. Every other line of
the modeling logic is unchanged.

Saves to models_real/ (NOT models/) so the currently-deployed synthetic model is untouched until
a decision is made to promote this one.

Run:  uv run python src/train_real_data_model.py
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import cross_val_predict, train_test_split
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier  # xgboost before torch (OpenMP conflict)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from build_real_data import SEQ_COLS, STATIC_COLS, build
from hybrid_model import Hybrid

warnings.filterwarnings("ignore")
torch.manual_seed(42); np.random.seed(42)
COST_FN = 5.0
COST_FP = 1.0
HIDDEN = 32

print("Building real dataset from Home Credit (application + bureau + installments)...")
df = build()
y = df["default"].values.astype("float32")
Xs = df[STATIC_COLS].values.astype("float32")
seq_raw = df[SEQ_COLS].values.astype("float32")
seq_mean, seq_std = float(seq_raw.mean()), float(seq_raw.std())
seq = ((seq_raw - seq_mean) / seq_std)[:, :, None]
print(f"n={len(df)}, default rate={y.mean():.4f}")

idx = np.arange(len(y))
i_tr, i_tmp = train_test_split(idx, test_size=0.45, stratify=y, random_state=42)
i_val, i_tmp2 = train_test_split(i_tmp, test_size=0.667, stratify=y[i_tmp], random_state=42)
i_cal, i_test = train_test_split(i_tmp2, test_size=0.5, stratify=y[i_tmp2], random_state=42)
print(f"splits: train {len(i_tr)}  val {len(i_val)}  cal {len(i_cal)}  test {len(i_test)}")

spw = (y[i_tr] == 0).sum() / (y[i_tr] == 1).sum()
xgb_params = dict(n_estimators=350, learning_rate=0.05, max_depth=4, subsample=0.8,
                  colsample_bytree=0.8, scale_pos_weight=spw, n_jobs=-1, random_state=42)
oof_tr = cross_val_predict(XGBClassifier(**xgb_params), Xs[i_tr], y[i_tr], cv=5,
                           method="predict_proba")[:, 1]
xgb = XGBClassifier(**xgb_params).fit(Xs[i_tr], y[i_tr])
xgb_score = {"tr": oof_tr, "val": xgb.predict_proba(Xs[i_val])[:, 1],
             "cal": xgb.predict_proba(Xs[i_cal])[:, 1], "test": xgb.predict_proba(Xs[i_test])[:, 1]}
print(f"XGBoost static AUC (test): {roc_auc_score(y[i_test], xgb_score['test']):.4f}")

def tens(a): return torch.tensor(a)
seq_t = {k: tens(seq[i]) for k, i in [("tr", i_tr), ("val", i_val), ("cal", i_cal), ("test", i_test)]}
sc_t = {k: tens(xgb_score[k]).float().unsqueeze(1) for k in xgb_score}
ytr_t = tens(y[i_tr]).unsqueeze(1)

model = Hybrid(HIDDEN)
crit = nn.BCEWithLogitsLoss(pos_weight=tens([spw]).float())
opt = torch.optim.Adam(model.parameters(), lr=5e-3)
loader = DataLoader(TensorDataset(seq_t["tr"], sc_t["tr"], ytr_t), batch_size=512, shuffle=True)
best, best_state = 0.0, None
for _epoch in range(30):
    model.train()
    for xb, sb, yb in loader:
        opt.zero_grad(); crit(model(xb, sb), yb).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        va = roc_auc_score(y[i_val], torch.sigmoid(model(seq_t["val"], sc_t["val"])).numpy().ravel())
    if va > best:
        best, best_state = va, {k: v.clone() for k, v in model.state_dict().items()}
model.load_state_dict(best_state); model.eval()

def raw_pd(split):
    with torch.no_grad():
        return torch.sigmoid(model(seq_t[split], sc_t[split])).numpy().ravel()
print(f"fused (uncalibrated) AUC (test): {roc_auc_score(y[i_test], raw_pd('test')):.4f}")

iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
iso.fit(raw_pd("cal"), y[i_cal])
grid_x = np.linspace(0.0, 1.0, 201)
grid_y = iso.predict(grid_x)
def calibrate(p): return np.interp(p, grid_x, grid_y)

cal_test = calibrate(raw_pd("test"))
brier_raw = brier_score_loss(y[i_test], raw_pd("test"))
brier_cal = brier_score_loss(y[i_test], cal_test)
print(f"calibrated AUC (test): {roc_auc_score(y[i_test], cal_test):.4f}  "
      f"| Brier raw {brier_raw:.4f} -> cal {brier_cal:.4f}")
print(f"mean predicted PD: {cal_test.mean():.4f}  vs actual rate: {y[i_test].mean():.4f}")

print("\nreliability (test set, by decile of calibrated score):")
order = np.argsort(cal_test); n = len(order)
for k in range(10):
    ii = order[k*n//10:(k+1)*n//10]
    print(f"  decile {k+1:2d}: predicted {cal_test[ii].mean():.3f}   actual {y[i_test][ii].mean():.3f}")

pc = calibrate(raw_pd("cal")); yc = y[i_cal]
ts = np.linspace(0.01, 0.60, 591)
costs = [COST_FN * ((pc < t) & (yc == 1)).sum() + COST_FP * ((pc >= t) & (yc == 0)).sum() for t in ts]
t_decline = float(ts[int(np.argmin(costs))])
t_approve = round(max(0.02, t_decline * 0.5), 3)
thresholds = {"approve": t_approve, "decline": round(t_decline, 3)}
print(f"\ncost-optimal thresholds (5:1): approve < {t_approve}  |  decline >= {round(t_decline, 3)}")

rec_test = np.where(cal_test >= t_decline, "decline",
                    np.where(cal_test >= t_approve, "review", "approve"))
mix = {r: round(float((rec_test == r).mean()), 3) for r in ["approve", "review", "decline"]}
print("decision mix on test (REAL data model):", mix)

print("\n=== FAIRNESS AUDIT on REAL gender/region (held out of scoring) ===")
gender_test = df["gender"].values[i_test]
region_test = df["region"].values[i_test]
approved = (rec_test != "decline")
fairness = {}
for attr_name, attr_vals in [("gender", gender_test), ("region", region_test)]:
    groups = sorted(set(attr_vals.tolist()))
    rows = {}
    for g in groups:
        m = attr_vals == g
        approval_rate = float(approved[m].mean())
        good_m = m & (y[i_test] == 0); bad_m = m & (y[i_test] == 1)
        tpr = float(approved[good_m].mean()) if good_m.sum() else None
        fpr = float(approved[bad_m].mean()) if bad_m.sum() else None
        rows[str(g)] = {"n": int(m.sum()), "approval_rate": round(approval_rate, 4),
                        "tpr_good_approved": round(tpr, 4) if tpr is not None else None,
                        "fpr_bad_approved": round(fpr, 4) if fpr is not None else None}
    rates = [r["approval_rate"] for r in rows.values()]
    dp_gap = round(max(rates) - min(rates), 4)
    tprs = [r["tpr_good_approved"] for r in rows.values() if r["tpr_good_approved"] is not None]
    eo_gap = round(max(tprs) - min(tprs), 4) if len(tprs) > 1 else 0.0
    fairness[attr_name] = {"by_group": rows, "demographic_parity_gap": dp_gap, "equalized_odds_gap": eo_gap}
    print(f"  {attr_name}: demographic-parity gap = {dp_gap:.4f}, equalized-odds gap = {eo_gap:.4f}")
    for g, r in rows.items():
        print(f"    group {g}: n={r['n']:>6}  approval_rate={r['approval_rate']:.3f}  "
              f"TPR={r['tpr_good_approved']}  FPR={r['fpr_bad_approved']}")

ranges = {}
for c in STATIC_COLS:
    col = df[c].values
    ranges[c] = {"min": float(np.percentile(col, 1)), "max": float(np.percentile(col, 99)),
                 "median": float(np.median(col))}

models_dir = ROOT / "models_real"; models_dir.mkdir(exist_ok=True)
joblib.dump(xgb, models_dir / "hybrid_xgb.joblib")
sd = model.state_dict()
np.savez(models_dir / "hybrid_fusion.npz",
         W_ih=sd["lstm.weight_ih_l0"].numpy(), W_hh=sd["lstm.weight_hh_l0"].numpy(),
         b_ih=sd["lstm.bias_ih_l0"].numpy(), b_hh=sd["lstm.bias_hh_l0"].numpy(),
         h0w=sd["head.0.weight"].numpy(), h0b=sd["head.0.bias"].numpy(),
         h2w=sd["head.2.weight"].numpy(), h2b=sd["head.2.bias"].numpy())
config = {
    "data_source": "REAL Home Credit (application_train + bureau + installments_payments)",
    "n_total": len(df), "static_cols": STATIC_COLS, "seq_len": len(SEQ_COLS),
    "seq_mean": seq_mean, "seq_std": seq_std, "lstm_hidden": HIDDEN,
    "calibration": {"grid_x": grid_x.tolist(), "grid_y": grid_y.round(6).tolist()},
    "thresholds": thresholds, "cost_ratio": COST_FN / COST_FP,
    "feature_ranges": ranges,
    "metrics": {
        "test_auc": round(float(roc_auc_score(y[i_test], cal_test)), 4),
        "test_brier": round(float(brier_score_loss(y[i_test], cal_test)), 4),
        "test_default_rate": round(float(y[i_test].mean()), 4),
        "test_mean_predicted_pd": round(float(cal_test.mean()), 4),
    },
    "fairness_audit": fairness,
}
(models_dir / "hybrid_config.json").write_text(json.dumps(config, indent=2))

d = np.load(models_dir / "hybrid_fusion.npz")
def np_forward(s, score, H=HIDDEN):
    h = np.zeros(H); c = np.zeros(H)
    for v in s:
        g = d["W_ih"] @ np.array([v]) + d["b_ih"] + d["W_hh"] @ h + d["b_hh"]
        i, f, gg, o = g[:H], g[H:2*H], g[2*H:3*H], g[3*H:4*H]
        i, f, gg, o = 1/(1+np.exp(-i)), 1/(1+np.exp(-f)), np.tanh(gg), 1/(1+np.exp(-o))
        c = f*c + i*gg; h = o*np.tanh(c)
    z = np.concatenate([h, [score]]); z1 = np.maximum(0, d["h0w"] @ z + d["h0b"])
    return float(1/(1+np.exp(-(d["h2w"] @ z1 + d["h2b"])[0])))

raw_test_arr = raw_pd("test")
md = 0.0
for j in range(min(300, len(i_test))):
    md = max(md, abs(raw_test_arr[j] - np_forward(seq[i_test][j].ravel(), xgb_score["test"][j])))
print(f"\nmax |torch - numpy| over 300 test cases = {md:.2e}  {'OK' if md < 1e-5 else 'MISMATCH'}")
print(f"saved -> {models_dir}/hybrid_xgb.joblib, hybrid_fusion.npz, hybrid_config.json")
