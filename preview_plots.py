"""Render all four plots to PNG for visual inspection."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

from plots import (
    plot_gram_matrix,
    plot_loss_curve,
    plot_polysemanticity,
    plot_weight_map,
)
from train import train


def main():
    out = Path(__file__).parent / "preview"
    out.mkdir(exist_ok=True)

    model, history, elapsed = train(
        n_features=20, n_hidden=5, sparsity=0.9, n_steps=2000, seed=0
    )
    print(f"trained in {elapsed:.2f}s, final val loss = {history[-1][1]:.5f}")

    figs = {
        "gram_matrix.png": plot_gram_matrix(model),
        "weight_map.png": plot_weight_map(model),
        "loss_curve.png": plot_loss_curve(history),
        "polysemanticity.png": plot_polysemanticity(model),
    }
    for name, fig in figs.items():
        path = out / name
        fig.savefig(path, dpi=120, bbox_inches="tight")
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
