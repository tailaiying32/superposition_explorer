"""Train ToyMLP on synthetic sparse data."""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from data import generate_batch
from model import ToyMLP

VAL_BATCH_SIZE = 1024
LOG_EVERY = 50


def train(
    n_features: int,
    n_hidden: int,
    sparsity: float,
    n_steps: int = 2000,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | str = "cpu",
    seed: int = 42,
) -> tuple[ToyMLP, list[tuple[int, float]], float]:
    """Train a ToyMLP and return (model, val_loss_history, elapsed_seconds).

    val_loss_history is a list of (step, loss) pairs evaluated on a fixed
    validation batch, so curves are directly comparable across runs.
    """
    device = torch.device(device)
    torch.manual_seed(seed)

    # Fixed validation batch — same across all runs with the same seed/sparsity/n_features.
    val_gen = torch.Generator(device=device).manual_seed(seed)
    x_val = generate_batch(n_features, sparsity, VAL_BATCH_SIZE, device, generator=val_gen)

    train_gen = torch.Generator(device=device).manual_seed(seed + 1)

    model = ToyMLP(n_features, n_hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: list[tuple[int, float]] = []
    start = time.perf_counter()

    for step in range(n_steps):
        x = generate_batch(n_features, sparsity, batch_size, device, generator=train_gen)
        x_hat = model(x)
        loss = F.mse_loss(x_hat, x)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % LOG_EVERY == 0 or step == n_steps - 1:
            with torch.no_grad():
                val_loss = F.mse_loss(model(x_val), x_val).item()
            history.append((step, val_loss))

    elapsed = time.perf_counter() - start
    return model, history, elapsed
