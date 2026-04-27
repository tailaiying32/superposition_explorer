"""Lightweight assertions across data, model, train, and plots.

Run: `python tests.py`
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # no GUI; safe for headless runs

import matplotlib.pyplot as plt
import numpy as np
import torch

from data import generate_batch
from model import ToyMLP
from plots import (
    participation_ratio,
    plot_gram_matrix,
    plot_loss_curve,
    plot_polysemanticity,
    plot_regime_heatmap,
    plot_sweep_heatmap,
    plot_weight_map,
)
from sweep import (
    METRIC_KEYS,
    SWEEP_COLUMNS,
    compute_metrics,
    is_interference_regime,
    is_polysemantic_heuristic,
    run_sweep,
)
from train import train


# ---------------- data ---------------- #

def test_data_shape_and_range():
    x = generate_batch(n_features=10, sparsity=0.8, batch_size=64)
    assert x.shape == (64, 10)
    assert x.dtype == torch.float32
    assert (x >= 0).all() and (x <= 1).all()


def test_data_sparsity_extremes():
    x_zero = generate_batch(n_features=10, sparsity=1.0, batch_size=128)
    assert (x_zero == 0).all(), "sparsity=1 must produce all zeros"
    x_dense = generate_batch(n_features=10, sparsity=0.0, batch_size=4096)
    active_frac = (x_dense > 0).float().mean().item()
    assert active_frac > 0.99, f"sparsity=0 should be almost fully active, got {active_frac}"


def test_data_active_fraction_matches_sparsity():
    x = generate_batch(n_features=20, sparsity=0.7, batch_size=8192)
    active_frac = (x > 0).float().mean().item()
    assert 0.27 < active_frac < 0.33, f"expected ~0.30, got {active_frac:.3f}"


# ---------------- model ---------------- #

def test_model_shapes():
    model = ToyMLP(n_features=20, n_hidden=5)
    assert model.W.shape == (5, 20)
    assert model.b_h.shape == (5,)
    assert model.b_d.shape == (20,)
    x = torch.randn(8, 20)
    y = model(x)
    assert y.shape == (8, 20)


def test_model_forward_with_zero_input():
    """Zero input + zero biases -> zero output (sanity-check linearity path)."""
    model = ToyMLP(n_features=8, n_hidden=3)
    x = torch.zeros(4, 8)
    y = model(x)
    # Biases initialized to zero, so output should be exactly zero.
    assert torch.allclose(y, torch.zeros_like(y))


# ---------------- train ---------------- #

def test_train_runs_and_decreases_loss():
    model, history, elapsed = train(
        n_features=20, n_hidden=5, sparsity=0.9, n_steps=1000, seed=0
    )
    assert isinstance(model, ToyMLP)
    assert len(history) > 0
    assert all(isinstance(s, int) and isinstance(l, float) for s, l in history)
    assert history[-1][1] < history[0][1], "validation loss did not decrease"
    assert elapsed > 0


def test_train_reproducibility():
    """Same seed/config -> identical val curves."""
    _, h1, _ = train(20, 5, 0.9, n_steps=300, seed=7)
    _, h2, _ = train(20, 5, 0.9, n_steps=300, seed=7)
    assert len(h1) == len(h2)
    for (s1, l1), (s2, l2) in zip(h1, h2):
        assert s1 == s2 and abs(l1 - l2) < 1e-8


def test_train_weight_decay_accepted():
    _, history, _ = train(
        n_features=10, n_hidden=4, sparsity=0.8,
        n_steps=300, weight_decay=0.05, seed=0,
    )
    assert history[-1][1] < history[0][1]


# ---------------- plots ---------------- #

def test_participation_ratio_extremes():
    # One-hot rows -> PR = 1 (monosemantic)
    W_mono = np.eye(5, 10)
    assert np.allclose(participation_ratio(W_mono), 1.0)
    # Uniform row of length k -> PR = k
    k = 8
    W_uniform = np.ones((1, k))
    assert np.isclose(participation_ratio(W_uniform)[0], float(k))


def test_plots_return_figures():
    model, history, _ = train(
        n_features=12, n_hidden=4, sparsity=0.85, n_steps=200, seed=0
    )
    figs = [
        plot_gram_matrix(model),
        plot_weight_map(model),
        plot_loss_curve(history),
        plot_polysemanticity(model),
    ]
    for f in figs:
        assert isinstance(f, plt.Figure)
        plt.close(f)


# ---------------- sweep ---------------- #

def test_compute_metrics_keys_and_types():
    model, history, elapsed = train(
        n_features=10, n_hidden=3, sparsity=0.8, n_steps=200, seed=0
    )
    m = compute_metrics(model, history, elapsed)
    assert set(m.keys()) == set(METRIC_KEYS), m.keys()
    for k, v in m.items():
        assert isinstance(v, float), f"{k} -> {type(v)}"
        assert np.isfinite(v), f"{k} -> {v}"
    assert m["max_pr"] >= m["mean_pr"]
    assert 0.0 <= m["frac_polysemantic"] <= 1.0


def test_compute_metrics_monosemantic_eye():
    """Hand-build a model with W = I; expect PR=1, no off-diagonal interference."""
    model = ToyMLP(n_features=6, n_hidden=6)
    with torch.no_grad():
        model.W.copy_(torch.eye(6))
        model.b_h.zero_()
        model.b_d.zero_()
    history = [(0, 1.0), (10, 0.5)]
    m = compute_metrics(model, history, elapsed=0.01)
    assert abs(m["mean_pr"] - 1.0) < 1e-6, m["mean_pr"]
    assert abs(m["max_pr"] - 1.0) < 1e-6
    assert m["interference"] < 1e-8, m["interference"]
    assert m["frac_polysemantic"] == 0.0


def test_classifier_logic():
    """Both classifiers fire on their own signal, ignore the other."""
    base = {
        "mean_pr": 1.0, "max_pr": 1.0, "frac_polysemantic": 0.0,
        "interference": 0.0, "final_val_loss": 0.0, "elapsed_s": 0.0,
    }

    # polysemantic flag toggles on PR
    high_pr = {**base, "mean_pr": 5.0}
    assert is_polysemantic_heuristic(high_pr)
    assert not is_interference_regime(high_pr)

    # ...or on frac
    high_frac = {**base, "frac_polysemantic": 0.6}
    assert is_polysemantic_heuristic(high_frac)
    assert not is_interference_regime(high_frac)

    # interference flag is independent
    high_int = {**base, "interference": 0.2}
    assert not is_polysemantic_heuristic(high_int)
    assert is_interference_regime(high_int)

    # both off
    assert not is_polysemantic_heuristic(base)
    assert not is_interference_regime(base)

    # both on
    both = {**base, "mean_pr": 5.0, "interference": 0.2}
    assert is_polysemantic_heuristic(both)
    assert is_interference_regime(both)


def test_run_sweep_shape():
    df = run_sweep(
        n_features=12,
        widths=(2, 4),
        sparsities=(0.5, 0.9),
        seeds=(0, 1),
        n_steps=100,
    )
    assert len(df) == 2 * 2 * 2
    assert tuple(df.columns) == SWEEP_COLUMNS, df.columns
    assert df["polysemantic_heuristic"].dtype == bool
    assert df["interference_regime"].dtype == bool
    # numeric metrics finite
    for col in METRIC_KEYS:
        assert df[col].apply(np.isfinite).all(), col


def test_run_sweep_progress_callback():
    seen = []
    df = run_sweep(
        n_features=8, widths=(2,), sparsities=(0.5, 0.9), seeds=(0,),
        n_steps=50, progress_callback=lambda d, t: seen.append((d, t)),
    )
    assert len(df) == 2
    assert seen == [(1, 2), (2, 2)], seen


def test_sweep_plots_return_figures():
    df = run_sweep(
        n_features=10, widths=(2, 4), sparsities=(0.5, 0.9), seeds=(0, 1),
        n_steps=80,
    )
    figs = [
        plot_sweep_heatmap(df, "mean_pr"),
        plot_sweep_heatmap(df, "interference"),
        plot_regime_heatmap(df, "polysemantic_heuristic"),
        plot_regime_heatmap(df, "interference_regime"),
    ]
    for f in figs:
        assert isinstance(f, plt.Figure)
        plt.close(f)


# ---------------- runner ---------------- #

def main():
    tests = [
        test_data_shape_and_range,
        test_data_sparsity_extremes,
        test_data_active_fraction_matches_sparsity,
        test_model_shapes,
        test_model_forward_with_zero_input,
        test_train_runs_and_decreases_loss,
        test_train_reproducibility,
        test_train_weight_decay_accepted,
        test_participation_ratio_extremes,
        test_plots_return_figures,
        test_compute_metrics_keys_and_types,
        test_compute_metrics_monosemantic_eye,
        test_classifier_logic,
        test_run_sweep_shape,
        test_run_sweep_progress_callback,
        test_sweep_plots_return_figures,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            failures += 1
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print()
    if failures:
        print(f"{failures} test(s) failed")
        raise SystemExit(1)
    print(f"All {len(tests)} tests passed.")


if __name__ == "__main__":
    main()
