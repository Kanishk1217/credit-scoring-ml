"""Torch-free forward pass of the LSTM + fusion head.

Runs the exact trained network in pure NumPy from weights exported to hybrid_fusion.npz, so the
serving image needs no PyTorch (which keeps it small enough for a free 512MB host). Verified to
match the PyTorch output to ~1e-7.
"""
from pathlib import Path

import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class NumpyHybrid:
    def __init__(self, npz_path: str | Path, hidden: int = 32) -> None:
        d = np.load(npz_path)
        self.W_ih, self.W_hh = d["W_ih"], d["W_hh"]
        self.b_ih, self.b_hh = d["b_ih"], d["b_hh"]
        self.h0w, self.h0b = d["h0w"], d["h0b"]
        self.h2w, self.h2b = d["h2w"], d["h2b"]
        self.H = hidden

    def predict(self, seq: list[float], score: float) -> float:
        """seq = standardized 6-month payment sequence; score = XGBoost static score. Returns PD."""
        H = self.H
        h = np.zeros(H)
        c = np.zeros(H)
        for v in seq:                                        # LSTM cell, one month at a time
            x = np.array([v], dtype=float)
            g = self.W_ih @ x + self.b_ih + self.W_hh @ h + self.b_hh   # (4H,)
            i, f, gg, o = g[:H], g[H:2*H], g[2*H:3*H], g[3*H:4*H]
            i, f, gg, o = _sigmoid(i), _sigmoid(f), np.tanh(gg), _sigmoid(o)
            c = f * c + i * gg
            h = o * np.tanh(c)
        z = np.concatenate([h, [score]])                     # fuse LSTM memory + XGBoost score
        z1 = np.maximum(0.0, self.h0w @ z + self.h0b)        # Linear + ReLU
        logit = self.h2w @ z1 + self.h2b
        return float(_sigmoid(logit)[0])
