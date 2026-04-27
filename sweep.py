"""Hyperparameter sweep + heuristic regime detection for ToyMLP."""
from __future__ import annotations

import itertools
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from model import ToyMLP
from plots import participation_ratio
from train import train

DEFAULT_PR_THRESHOLD = 2.0
DEFAULT_FRAC_THRESHOLD = 0.3
DEFAULT_INTERFERENCE_THRESHOLD = 0.05

METRIC_KEYS = (
    "mean_pr",
    "max_pr",
    "frac_polysemantic",
    "interference",
    "final_val_loss",
    "elapsed_s",
)

SWEEP_COLUMNS = (
    "n_features",
    "n_hidden",
    "sparsity",
    "seed",
    *METRIC_KEYS,
    "polysemantic_heuristic",
    "interference_regime",
)


def _off_diag_abs_mean(W: np.ndarray) -> float:
    """Mean |off-diagonal entry| of WᵀW (feature interference score)."""
    gram = W.T @ W
    n = gram.shape[0]
    if n < 2:
        return 0.0
    mask = ~np.eye(n, dtype=bool)
    return float(np.abs(gram[mask]).mean())


def compute_metrics(
    model: ToyMLP,
    history: list[tuple[int, float]],
    elapsed: float,
    pr_threshold: float = DEFAULT_PR_THRESHOLD,
) -> dict[str, float]:
    """Scalar summary metrics for one trained run."""
    W = model.W.detach().cpu().numpy()
    pr = participation_ratio(W)
    return {
        "mean_pr": float(pr.mean()),
        "max_pr": float(pr.max()),
        "frac_polysemantic": float((pr > pr_threshold).mean()),
        "interference": _off_diag_abs_mean(W),
        "final_val_loss": float(history[-1][1]),
        "elapsed_s": float(elapsed),
    }


def is_polysemantic_heuristic(
    metrics: dict[str, float],
    pr_threshold: float = DEFAULT_PR_THRESHOLD,
    frac_threshold: float = DEFAULT_FRAC_THRESHOLD,
) -> bool:
    """Neuron-level: are hidden neurons attending to many features?"""
    return (metrics["mean_pr"] > pr_threshold) or (
        metrics["frac_polysemantic"] > frac_threshold
    )


def is_interference_regime(
    metrics: dict[str, float],
    interference_threshold: float = DEFAULT_INTERFERENCE_THRESHOLD,
) -> bool:
    """Feature-level: are feature directions overlapping in hidden space?"""
    return metrics["interference"] > interference_threshold


def run_sweep(
    n_features: int = 20,
    widths: Sequence[int] = (2, 3, 5, 8, 10),
    sparsities: Sequence[float] = (0.3, 0.6, 0.8, 0.9, 0.95),
    seeds: Sequence[int] = (0, 1, 2),
    n_steps: int = 2000,
    weight_decay: float = 0.0,
    pr_threshold: float = DEFAULT_PR_THRESHOLD,
    frac_threshold: float = DEFAULT_FRAC_THRESHOLD,
    interference_threshold: float = DEFAULT_INTERFERENCE_THRESHOLD,
    progress_callback: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Train ToyMLP across the (width × sparsity × seed) grid; return DataFrame.

    One row per configuration. Reuses train.train (unchanged).
    """
    grid = list(itertools.product(widths, sparsities, seeds))
    total = len(grid)
    rows = []

    for done, (m, s, seed) in enumerate(grid, start=1):
        model, history, elapsed = train(
            n_features=n_features,
            n_hidden=int(m),
            sparsity=float(s),
            n_steps=n_steps,
            weight_decay=weight_decay,
            seed=int(seed),
        )
        metrics = compute_metrics(model, history, elapsed, pr_threshold=pr_threshold)
        rows.append(
            {
                "n_features": n_features,
                "n_hidden": int(m),
                "sparsity": float(s),
                "seed": int(seed),
                **metrics,
                "polysemantic_heuristic": is_polysemantic_heuristic(
                    metrics, pr_threshold, frac_threshold
                ),
                "interference_regime": is_interference_regime(
                    metrics, interference_threshold
                ),
            }
        )
        if progress_callback is not None:
            progress_callback(done, total)

    df = pd.DataFrame(rows, columns=list(SWEEP_COLUMNS))
    df["polysemantic_heuristic"] = df["polysemantic_heuristic"].astype(bool)
    df["interference_regime"] = df["interference_regime"].astype(bool)
    return df
