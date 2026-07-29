"""5-fold stratified cross-validation of the ENRICHED real-data hybrid pipeline, matching the
rigor already applied to the synthetic model (src/cross_validate.py). Verifies the AUC/recall/
fairness numbers from a single split are stable, not a lucky (or unlucky) split.

Run:  uv run python src/cross_validate_real.py
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier  # noqa: I001  (xgboost must import before torch -- OpenMP conflict)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from build_real_data import SEQ_COLS, STATIC_COLS
from build_real_data_rich import RICH_COLS, build_rich
from hybrid_model import Hybrid

warnings.filterwarnings("ignore")
torch.manual_seed(42); np.random.seed(42)
HIDDEN = 32
N_FOLDS = 5
ALL_STATIC = STATIC_COLS + RICH_COLS

print("Building enriched real dataset (this only needs to happen once)...")
df = build_rich()
y_all = df["default"].values.astype("float32")
Xs_all = df[ALL_STATIC].values.astype("float32")
seq_raw = df[SEQ_COLS].values.astype("float32")
seq_mean, seq_std = float(seq_raw.mean()), float(seq_raw.std())
seq_all = ((seq_raw - seq_mean) / seq_std)[:, :, None]
print(f"n={len(df)}, default rate={y_all.mean():.4f}, static features={len(ALL_STATIC)}\n")

kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
print(f"{'fold':<6}{'n_test':<9}{'test_default%':<16}{'AUC':<10}{'Brier':<10}{'Recall':<10}{'Precision':<10}")

fold_metrics = []
for fold_i, (train_full_idx, test_idx) in enumerate(kfold.split(Xs_all, y_all), start=1):
    y_train_full = y_all[train_full_idx]
    i_tr, i_tmp = train_test_split(train_full_idx, test_size=0.30, stratify=y_train_full, random_state=42)
    i_val, i_cal = train_test_split(i_tmp, test_size=0.5, stratify=y_all[i_tmp], random_state=42)

    spw = (y_all[i_tr] == 0).sum() / (y_all[i_tr] == 1).sum()
    xgb_params = dict(n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8,
                      colsample_bytree=0.8, scale_pos_weight=spw, n_jobs=-1, random_state=42)
    oof_tr = cross_val_predict(XGBClassifier(**xgb_params), Xs_all[i_tr], y_all[i_tr], cv=5,
                               method="predict_proba")[:, 1]
    xgb = XGBClassifier(**xgb_params).fit(Xs_all[i_tr], y_all[i_tr])
    sc_val = xgb.predict_proba(Xs_all[i_val])[:, 1]
    sc_cal = xgb.predict_proba(Xs_all[i_cal])[:, 1]
    sc_test = xgb.predict_proba(Xs_all[test_idx])[:, 1]

    def tens(a): return torch.tensor(a)
    seq_tr_t, seq_val_t = tens(seq_all[i_tr]), tens(seq_all[i_val])
    seq_cal_t, seq_test_t = tens(seq_all[i_cal]), tens(seq_all[test_idx])
    sc_tr_t, sc_val_t = tens(oof_tr).float().unsqueeze(1), tens(sc_val).float().unsqueeze(1)
    sc_cal_t, sc_test_t = tens(sc_cal).float().unsqueeze(1), tens(sc_test).float().unsqueeze(1)
    ytr_t = tens(y_all[i_tr]).unsqueeze(1)

    model = Hybrid(HIDDEN)
    crit = nn.BCEWithLogitsLoss(pos_weight=tens([spw]).float())
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    loader = DataLoader(TensorDataset(seq_tr_t, sc_tr_t, ytr_t), batch_size=512, shuffle=True)
    best, best_state = 0.0, None
    for _epoch in range(20):
        model.train()
        for xb, sb, yb in loader:
            opt.zero_grad(); crit(model(xb, sb), yb).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            va = roc_auc_score(y_all[i_val], torch.sigmoid(model(seq_val_t, sc_val_t)).numpy().ravel())
        if va > best:
            best, best_state = va, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()

    with torch.no_grad():
        raw_cal = torch.sigmoid(model(seq_cal_t, sc_cal_t)).numpy().ravel()
        raw_test = torch.sigmoid(model(seq_test_t, sc_test_t)).numpy().ravel()

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw_cal, y_all[i_cal])
    grid_x = np.linspace(0.0, 1.0, 201)
    grid_y = iso.predict(grid_x)
    cal_test = np.interp(raw_test, grid_x, grid_y)
    y_test = y_all[test_idx]

    # cost-optimal threshold on this fold's cal set
    pc = np.interp(raw_cal, grid_x, grid_y); yc = y_all[i_cal]
    ts = np.linspace(0.01, 0.6, 591)
    costs = [5.0 * ((pc < t) & (yc == 1)).sum() + 1.0 * ((pc >= t) & (yc == 0)).sum() for t in ts]
    t_decline = float(ts[int(np.argmin(costs))])
    pred = (cal_test >= t_decline).astype(int)

    auc = roc_auc_score(y_test, cal_test)
    brier = brier_score_loss(y_test, cal_test)
    rec = recall_score(y_test, pred, zero_division=0)
    prec = precision_score(y_test, pred, zero_division=0)

    print(f"{fold_i:<6}{len(test_idx):<9}{y_test.mean():<16.4%}{auc:<10.4f}{brier:<10.4f}{rec:<10.4f}{prec:<10.4f}")
    fold_metrics.append({"auc": auc, "brier": brier, "recall": rec, "precision": prec})

aucs = [m["auc"] for m in fold_metrics]
recalls = [m["recall"] for m in fold_metrics]
print(f"\n{'='*60}")
print(f"AUC across {N_FOLDS} folds:    mean={np.mean(aucs):.4f}  std={np.std(aucs):.4f}  "
     f"min={min(aucs):.4f}  max={max(aucs):.4f}")
print(f"Recall across {N_FOLDS} folds: mean={np.mean(recalls):.4f}  std={np.std(recalls):.4f}")
print(f"{'='*60}")
