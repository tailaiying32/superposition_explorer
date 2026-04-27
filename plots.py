"""Diagnostic plots for visualizing superposition in a trained ToyMLP."""
from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import torch

from model import ToyMLP

if TYPE_CHECKING:
    import pandas as pd

_SEQUENTIAL_CMAPS = {
    "mean_pr": "viridis",
    "max_pr": "viridis",
    "frac_polysemantic": "viridis",
    "interference": "viridis",
    "final_val_loss": "magma",
    "elapsed_s": "magma",
}


def _W_numpy(model: ToyMLP) -> np.ndarray:
    return model.W.detach().cpu().numpy()


def plot_gram_matrix(model: ToyMLP) -> plt.Figure:
    """Heatmap of WᵀW: feature interference matrix.

    Diagonal = squared norm of each feature direction in hidden space.
    Off-diagonal = inner product (interference) between feature directions.
    """
    W = _W_numpy(model)  # (n_hidden, n_features)
    gram = W.T @ W       # (n_features, n_features)

    n = gram.shape[0]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    vmax = float(np.abs(gram).max()) or 1.0
    im = ax.imshow(gram, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    ax.set_title("Feature Interference (WᵀW)")
    ax.set_xlabel("Feature j")
    ax.set_ylabel("Feature i")
    if n <= 24:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_weight_map(model: ToyMLP) -> plt.Figure:
    """Heatmap of W: rows = hidden neurons, cols = input features."""
    W = _W_numpy(model)
    m, n = W.shape

    fig, ax = plt.subplots(figsize=(6, max(2.5, 0.4 * m + 1.2)))
    vmax = float(np.abs(W).max()) or 1.0
    im = ax.imshow(W, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title("Neuron–Feature Weights (W)")
    ax.set_xlabel("Input feature")
    ax.set_ylabel("Hidden neuron")
    ax.set_yticks(range(m))
    if n <= 24:
        ax.set_xticks(range(n))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_loss_curve(history: list[tuple[int, float]]) -> plt.Figure:
    """Validation loss (fixed batch) vs training step, log-y."""
    steps = [s for s, _ in history]
    losses = [l for _, l in history]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(steps, losses, color="C0", linewidth=1.8)
    ax.set_yscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel("MSE loss (log scale)")
    ax.set_title("Validation Loss (fixed batch)")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig


def participation_ratio(W: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Per-neuron participation ratio: PR_j = (Σ |W_ji|)² / Σ W_ji².

    Soft count of how many features each neuron meaningfully attends to.
    PR ≈ 1 -> monosemantic; PR > 1 -> polysemantic.
    """
    abs_sum = np.abs(W).sum(axis=1)
    sq_sum = (W ** 2).sum(axis=1)
    return (abs_sum ** 2) / (sq_sum + eps)


def plot_polysemanticity(model: ToyMLP) -> plt.Figure:
    """Bar chart of participation ratio per hidden neuron."""
    W = _W_numpy(model)
    pr = participation_ratio(W)
    m = len(pr)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(range(m), pr, color="C2", edgecolor="black", linewidth=0.5)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label="monosemantic (PR=1)")
    ax.set_xlabel("Hidden neuron")
    ax.set_ylabel("Participation ratio")
    ax.set_title("Effective Feature Count per Neuron")
    ax.set_xticks(range(m))
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def _pivot_grid(df: "pd.DataFrame", value_col: str, agg: str = "mean"):
    """Pivot a sweep DataFrame to a (sparsity × n_hidden) grid via the named agg."""
    return df.pivot_table(
        index="sparsity", columns="n_hidden", values=value_col, aggfunc=agg
    ).sort_index()


def _annotate_cells(ax, grid: np.ndarray, fmt: str = "{:.2f}", color="black"):
    rows, cols = grid.shape
    for i in range(rows):
        for j in range(cols):
            v = grid[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, fmt.format(v), ha="center", va="center",
                    fontsize=8, color=color)


def plot_sweep_heatmap(
    df: "pd.DataFrame",
    metric: str,
    agg: str = "mean",
    title: str | None = None,
) -> plt.Figure:
    """Heatmap of `metric` across (sparsity, n_hidden), aggregated across seeds."""
    pivot = _pivot_grid(df, metric, agg=agg)
    grid = pivot.values
    sparsities = pivot.index.to_numpy()
    widths = pivot.columns.to_numpy()
    cmap = _SEQUENTIAL_CMAPS.get(metric, "viridis")

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(grid, cmap=cmap, aspect="auto", origin="lower")
    ax.set_xticks(range(len(widths)))
    ax.set_xticklabels(widths)
    ax.set_yticks(range(len(sparsities)))
    ax.set_yticklabels([f"{s:.2f}" for s in sparsities])
    ax.set_xlabel("Hidden width (m)")
    ax.set_ylabel("Sparsity")
    ax.set_title(title or f"{metric} ({agg} across seeds)")
    _annotate_cells(ax, grid, fmt="{:.2g}", color="white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_regime_heatmap(
    df: "pd.DataFrame",
    flag: str = "polysemantic_heuristic",
) -> plt.Figure:
    """Heatmap of fraction-of-seeds with `flag == True` across the grid."""
    pivot = _pivot_grid(df, flag, agg="mean")
    grid = pivot.values
    sparsities = pivot.index.to_numpy()
    widths = pivot.columns.to_numpy()

    pretty = {
        "polysemantic_heuristic": "Polysemantic-neuron regime (fraction of seeds)",
        "interference_regime": "Interference regime (fraction of seeds)",
    }.get(flag, f"{flag} (fraction of seeds)")

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(grid, cmap="RdBu_r", aspect="auto", origin="lower",
                   vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(widths)))
    ax.set_xticklabels(widths)
    ax.set_yticks(range(len(sparsities)))
    ax.set_yticklabels([f"{s:.2f}" for s in sparsities])
    ax.set_xlabel("Hidden width (m)")
    ax.set_ylabel("Sparsity")
    ax.set_title(pretty)
    _annotate_cells(ax, grid, fmt="{:.2f}", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig
