"""The hybrid fusion network architecture, shared by training and serving.

Keeping the model class in one place means the API loads the exact architecture that was trained.
"""
import torch
import torch.nn as nn


class Hybrid(nn.Module):
    """LSTM over the payment sequence, fused with the XGBoost static score, then an MLP head."""

    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden + 1, 16),   # +1 for the XGBoost score
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x_seq: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
        # x_seq: (batch, timesteps, 1);  score: (batch, 1)
        _, (hn, _) = self.lstm(x_seq)    # hn: (1, batch, hidden) — final memory
        emb = hn[-1]                     # (batch, hidden)
        return self.head(torch.cat([emb, score], dim=1))   # (batch, 1) logit
