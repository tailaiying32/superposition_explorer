"""CLI smoke test: exercises model, data, and train end-to-end."""
from __future__ import annotations

import torch

from data import generate_batch
from model import ToyMLP
from train import train


def test_data():
    x = generate_batch(n_features=10, sparsity=0.8, batch_size=64)
    assert x.shape == (64, 10), x.shape
    assert (x >= 0).all() and (x <= 1).all()
    # All zeros when sparsity = 1
    x_zero = generate_batch(n_features=10, sparsity=1.0, batch_size=32)
    assert (x_zero == 0).all()
    # Active fraction roughly matches (1 - sparsity)
    x_big = generate_batch(n_features=20, sparsity=0.7, batch_size=4096)
    active_frac = (x_big > 0).float().mean().item()
    assert 0.25 < active_frac < 0.35, f"expected ~0.30, got {active_frac:.3f}"
    print(f"  data OK   (active_frac={active_frac:.3f})")


def test_model():
    model = ToyMLP(n_features=20, n_hidden=5)
    x = torch.randn(8, 20)
    y = model(x)
    assert y.shape == x.shape
    assert model.W.shape == (5, 20)
    print("  model OK")


def test_train():
    model, history, elapsed = train(
        n_features=20, n_hidden=5, sparsity=0.9, n_steps=1000, seed=0
    )
    assert isinstance(model, ToyMLP)
    assert len(history) > 0
    first_loss = history[0][1]
    last_loss = history[-1][1]
    assert last_loss < first_loss, f"loss did not decrease: {first_loss} -> {last_loss}"
    print(
        f"  train OK  (steps={len(history)}, "
        f"loss {first_loss:.4f} -> {last_loss:.4f}, {elapsed:.2f}s)"
    )


def test_train_with_decay():
    _, history, _ = train(
        n_features=20, n_hidden=5, sparsity=0.9, n_steps=500,
        weight_decay=0.01, seed=0,
    )
    assert history[-1][1] < history[0][1]
    print("  train+wd OK")


def test_val_batch_reproducibility():
    """Same seed/config produces identical val-loss curves -> fair comparisons."""
    _, h1, _ = train(20, 5, 0.9, n_steps=300, seed=7)
    _, h2, _ = train(20, 5, 0.9, n_steps=300, seed=7)
    for (s1, l1), (s2, l2) in zip(h1, h2):
        assert s1 == s2 and abs(l1 - l2) < 1e-8, (s1, l1, s2, l2)
    print("  reproducibility OK")


if __name__ == "__main__":
    print("Running smoke tests...")
    test_data()
    test_model()
    test_train()
    test_train_with_decay()
    test_val_batch_reproducibility()
    print("All smoke tests passed.")
