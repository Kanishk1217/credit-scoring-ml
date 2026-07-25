"""Hybrid model, XGBoost branch (run this FIRST, in its own process).

Produces the static-feature XGBoost scores and the payment sequences, saved to
data/processed/hybrid_feats.npz, which notebook 09 (torch) then loads to train the fusion.

Two processes are required because PyTorch + XGBoost segfault when imported in the same
process on this machine (OpenMP conflict).

Usage:  uv run python src/make_hybrid_features.py
"""
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict, train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data/raw/taiwan_credit/UCI_Credit_Card.csv")
target = "default.payment.next.month"
seq_cols = ["PAY_6", "PAY_5", "PAY_4", "PAY_3", "PAY_2", "PAY_0"]           # temporal branch
static_cols = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",         # static branch
               "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
               "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]

y = df[target].values
Xs = df[static_cols].values.astype("float32")
seq = df[seq_cols].values.astype("float32")
seq = (seq - seq.mean()) / seq.std()
seq = seq[:, :, None]                                                       # (N, 6, 1)

idx = np.arange(len(y))
itr, ite = train_test_split(idx, test_size=.2, stratify=y, random_state=42)
itr, iva = train_test_split(itr, test_size=.25, stratify=y[itr], random_state=42)

spw = (y[itr] == 0).sum() / (y[itr] == 1).sum()
params = dict(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=.8,
              colsample_bytree=.8, scale_pos_weight=spw, n_jobs=-1, random_state=42,
              eval_metric="auc")

# out-of-fold scores for TRAIN (no leakage into the fusion), direct predict for val/test
oof_tr = cross_val_predict(XGBClassifier(**params), Xs[itr], y[itr], cv=5,
                           method="predict_proba")[:, 1]
m = XGBClassifier(**params).fit(Xs[itr], y[itr])
s_va = m.predict_proba(Xs[iva])[:, 1]
s_te = m.predict_proba(Xs[ite])[:, 1]
xgb_auc = roc_auc_score(y[ite], s_te)
print(f"XGBoost (static only) test AUC = {xgb_auc:.4f}")

out = ROOT / "data/processed/hybrid_feats.npz"
np.savez(out,
         seq_tr=seq[itr], seq_va=seq[iva], seq_te=seq[ite],
         s_tr=oof_tr.astype("float32"), s_va=s_va.astype("float32"), s_te=s_te.astype("float32"),
         y_tr=y[itr].astype("float32"), y_va=y[iva].astype("float32"), y_te=y[ite].astype("float32"),
         xgb_auc=np.float32(xgb_auc))
print("saved", out)

# ---- DEPLOYMENT ARTIFACTS: the model becomes a FILE ----
models = ROOT / "models"; models.mkdir(exist_ok=True)
joblib.dump(m, models / "hybrid_xgb.joblib")                       # the trained XGBoost, as a file
config = {                                                        # everything needed to preprocess raw input
    "static_cols": static_cols,
    "seq_cols": seq_cols,
    "seq_mean": float(df[seq_cols].values.astype("float32").mean()),
    "seq_std":  float(df[seq_cols].values.astype("float32").std()),
    "lstm_hidden": 32,
}
(models / "hybrid_config.json").write_text(json.dumps(config, indent=2))
print("saved", models / "hybrid_xgb.joblib")
print("saved", models / "hybrid_config.json")
