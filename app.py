"""Streamlit UI for the superposition explorer.

Run: `streamlit run app.py` (inside the `superposition` mamba env).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import streamlit as st

from plots import (
    plot_gram_matrix,
    plot_loss_curve,
    plot_polysemanticity,
    plot_regime_heatmap,
    plot_sweep_heatmap,
    plot_weight_map,
)
from sweep import (
    DEFAULT_FRAC_THRESHOLD,
    DEFAULT_INTERFERENCE_THRESHOLD,
    DEFAULT_PR_THRESHOLD,
    run_sweep,
)
from train import train

st.set_page_config(page_title="Superposition Explorer", layout="wide")

st.title("Superposition Explorer")
st.caption(
    "Tiny tied-weight autoencoder trained on sparse synthetic features "
    "(Anthropic, *Toy Models of Superposition*). "
    "Inspect a single config, or sweep across a grid to find regime boundaries."
)


def _parse_csv_numbers(text: str, cast):
    out = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(cast(tok))
    return out


tab_single, tab_sweep = st.tabs(["Single run", "Sweep"])

# =============================================================== #
# Single-run tab                                                  #
# =============================================================== #

with tab_single:
    with st.sidebar:
        st.header("Single run")
        n_features = st.slider("Input features (n)", 2, 24, 20)
        n_hidden = st.slider("Hidden neurons (m)", 1, 10, 5)
        sparsity = st.slider(
            "Sparsity", 0.0, 0.98, 0.90, step=0.01,
            help="Probability a feature is zero in a sample",
        )
        weight_decay = st.slider(
            "Weight decay", 0.0, 0.1, 0.0, step=0.001, format="%.3f"
        )
        n_steps = st.slider("Training steps", 500, 5000, 2000, step=100)
        seed = st.number_input("Random seed", min_value=0, value=0, step=1)
        train_clicked = st.button(
            "Train", type="primary", use_container_width=True
        )
        if n_hidden >= n_features:
            st.info(
                f"With m={n_hidden} ≥ n={n_features}, there is no compression — "
                "expect monosemantic neurons (PR ≈ 1)."
            )

    if train_clicked:
        with st.spinner(f"Training for {n_steps} steps..."):
            model, history, elapsed = train(
                n_features=n_features,
                n_hidden=n_hidden,
                sparsity=sparsity,
                n_steps=n_steps,
                weight_decay=weight_decay,
                seed=int(seed),
            )
        st.session_state["result"] = {
            "model": model,
            "history": history,
            "elapsed": elapsed,
            "config": {
                "n_features": n_features,
                "n_hidden": n_hidden,
                "sparsity": sparsity,
                "weight_decay": weight_decay,
                "n_steps": n_steps,
                "seed": int(seed),
            },
        }

    result = st.session_state.get("result")
    if result is None:
        st.info("Set hyperparameters in the sidebar and press **Train** to begin.")
    else:
        model = result["model"]
        history = result["history"]
        elapsed = result["elapsed"]
        cfg = result["config"]

        m1, m2, m3 = st.columns(3)
        m1.metric("Training time", f"{elapsed:.2f} s")
        m2.metric("Final val loss", f"{history[-1][1]:.5f}")
        m3.metric("Compression", f"{cfg['n_features']} → {cfg['n_hidden']}")

        with st.expander("Run config", expanded=False):
            st.json(cfg)

        row1_left, row1_right = st.columns(2)
        with row1_left:
            st.pyplot(plot_gram_matrix(model), clear_figure=True)
        with row1_right:
            st.pyplot(plot_weight_map(model), clear_figure=True)

        row2_left, row2_right = st.columns(2)
        with row2_left:
            st.pyplot(plot_loss_curve(history), clear_figure=True)
        with row2_right:
            st.pyplot(plot_polysemanticity(model), clear_figure=True)

# =============================================================== #
# Sweep tab                                                       #
# =============================================================== #

with tab_sweep:
    st.subheader("Grid sweep across (width × sparsity)")
    st.caption(
        "Trains a small grid of configs (multiple seeds per cell) and flags "
        "two regimes independently: **polysemantic-neuron** (mean/frac PR high) "
        "and **interference** (mean |off-diag WᵀW| high)."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        sweep_n_features = st.number_input(
            "n_features", min_value=4, max_value=24, value=20, step=1
        )
        widths_str = st.text_input(
            "Hidden widths (comma-separated)", value="2,3,5,8,10"
        )
        sparsities_str = st.text_input(
            "Sparsities (comma-separated)", value="0.3,0.6,0.8,0.9,0.95"
        )
    with col_b:
        n_seeds = st.number_input(
            "Seeds per cell", min_value=1, max_value=5, value=2, step=1
        )
        sweep_n_steps = st.number_input(
            "Training steps per run", min_value=200, max_value=5000,
            value=1500, step=100,
        )
        sweep_weight_decay = st.number_input(
            "Weight decay", min_value=0.0, max_value=0.1,
            value=0.0, step=0.001, format="%.3f",
        )

    with st.expander("Heuristic thresholds", expanded=False):
        pr_threshold = st.number_input(
            "PR threshold (neuron polysemantic if PR >)",
            value=float(DEFAULT_PR_THRESHOLD), step=0.1,
        )
        frac_threshold = st.number_input(
            "Fraction threshold (regime flag if frac_polysemantic >)",
            value=float(DEFAULT_FRAC_THRESHOLD), step=0.05, format="%.2f",
        )
        interference_threshold = st.number_input(
            "Interference threshold (regime flag if mean |off-diag WᵀW| >)",
            value=float(DEFAULT_INTERFERENCE_THRESHOLD),
            step=0.005, format="%.3f",
        )

    try:
        widths = _parse_csv_numbers(widths_str, int)
        sparsities = _parse_csv_numbers(sparsities_str, float)
    except ValueError as e:
        st.error(f"Could not parse inputs: {e}")
        st.stop()

    total = len(widths) * len(sparsities) * int(n_seeds)
    st.caption(
        f"Grid: {len(widths)} widths × {len(sparsities)} sparsities × "
        f"{int(n_seeds)} seeds = **{total} runs** "
        f"(≈{total * (sweep_n_steps / 2000) * 2:.0f}s on CPU)."
    )

    sweep_clicked = st.button(
        "Run sweep", type="primary", use_container_width=True
    )

    if sweep_clicked:
        progress = st.progress(0.0, text="Starting sweep...")

        def _cb(done: int, total: int):
            progress.progress(done / total, text=f"Run {done}/{total}")

        df = run_sweep(
            n_features=int(sweep_n_features),
            widths=tuple(widths),
            sparsities=tuple(sparsities),
            seeds=tuple(range(int(n_seeds))),
            n_steps=int(sweep_n_steps),
            weight_decay=float(sweep_weight_decay),
            pr_threshold=float(pr_threshold),
            frac_threshold=float(frac_threshold),
            interference_threshold=float(interference_threshold),
            progress_callback=_cb,
        )
        progress.empty()
        st.session_state["sweep_df"] = df

    df = st.session_state.get("sweep_df")
    if df is None:
        st.info("Configure the grid above and press **Run sweep**.")
    else:
        st.success(f"Sweep complete: {len(df)} runs.")

        h1, h2 = st.columns(2)
        with h1:
            st.pyplot(plot_sweep_heatmap(df, "mean_pr"), clear_figure=True)
        with h2:
            st.pyplot(plot_sweep_heatmap(df, "interference"), clear_figure=True)

        h3, h4 = st.columns(2)
        with h3:
            st.pyplot(
                plot_regime_heatmap(df, "polysemantic_heuristic"),
                clear_figure=True,
            )
        with h4:
            st.pyplot(
                plot_regime_heatmap(df, "interference_regime"),
                clear_figure=True,
            )

        st.subheader("Raw results")
        st.dataframe(df, use_container_width=True)
