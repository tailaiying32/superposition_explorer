"""Synthetic sparse feature data."""
from __future__ import annotations

import torch


def generate_batch(
    n_features: int,
    sparsity: float,
    batch_size: int,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample a batch of sparse feature vectors.

    Each feature is independently active with probability (1 - sparsity).
    Active features take values in [0, 1); inactive features are exactly 0.
    """
    assert 0.0 <= sparsity <= 1.0, f"sparsity must be in [0, 1], got {sparsity}"
    assert n_features >= 1 and batch_size >= 1

    shape = (batch_size, n_features)
    p_active = 1.0 - sparsity
    mask = torch.bernoulli(
        torch.full(shape, p_active, device=device), generator=generator
    )
    values = torch.rand(shape, device=device, generator=generator)
    return mask * values
