"""Tied-weight bottleneck autoencoder from Anthropic's Toy Models of Superposition."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyMLP(nn.Module):
    def __init__(self, n_features: int, n_hidden: int):
        super().__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.W = nn.Parameter(torch.empty(n_hidden, n_features))
        self.b_h = nn.Parameter(torch.zeros(n_hidden))
        self.b_d = nn.Parameter(torch.zeros(n_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(x @ self.W.T + self.b_h)
        return h @ self.W + self.b_d
