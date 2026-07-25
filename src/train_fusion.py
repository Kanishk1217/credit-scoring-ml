"""Hybrid model, fusion-network branch (run AFTER make_hybrid_features.py, torch process).

Loads data/processed/hybrid_feats.npz, trains the LSTM+fusion head, and saves it to
models/hybrid_fusion.pt. Separate process from the XGBoost branch (OpenMP conflict).

Usage:  uv run python src/train_fusion.py
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
from hybrid_model import Hybrid

torch.manual_seed(42); np.random.seed(42)
ROOT = Path(__file__).resolve().parents[1]
d = np.load(ROOT / "data/processed/hybrid_feats.npz")

seq_tr, seq_va, seq_te = [torch.tensor(d[k]) for k in ("seq_tr", "seq_va", "seq_te")]
s_tr, s_va, s_te = [torch.tensor(d[k]).unsqueeze(1) for k in ("s_tr", "s_va", "s_te")]
y_tr, y_va, y_te = [d[k] for k in ("y_tr", "y_va", "y_te")]
ytr_t = torch.tensor(y_tr).unsqueeze(1)

model = Hybrid(hidden=32)
crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(y_tr == 0).sum() / (y_tr == 1).sum()]))
opt = torch.optim.Adam(model.parameters(), lr=5e-3)
loader = DataLoader(TensorDataset(seq_tr, s_tr, ytr_t), batch_size=256, shuffle=True)

best, best_state = 0, None
for epoch in range(30):
    model.train()
    for xb, sb, yb in loader:
        opt.zero_grad(); crit(model(xb, sb), yb).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        va = roc_auc_score(y_va, torch.sigmoid(model(seq_va, s_va)).numpy().ravel())
    if va > best:
        best, best_state = va, {k: v.clone() for k, v in model.state_dict().items()}

model.load_state_dict(best_state); model.eval()
with torch.no_grad():
    test_auc = roc_auc_score(y_te, torch.sigmoid(model(seq_te, s_te)).numpy().ravel())
print(f"hybrid test AUC = {test_auc:.4f}")

out = ROOT / "models" / "hybrid_fusion.pt"
torch.save(model.state_dict(), out)     # the network becomes a FILE
print("saved", out)
