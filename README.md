# Superposition Explorer

An interactive tool for visualizing **superposition** in tiny MLPs, based on Anthropic's *Toy Models of Superposition*.

## 1. Project Overview

A small tied-weight bottleneck autoencoder is trained on synthetic sparse feature vectors. A Streamlit UI lets you:

- **Single run**: pick (n_features, n_hidden, sparsity, weight_decay, steps) and inspect four diagnostic plots.
- **Sweep**: run a grid over (width × sparsity × seeds), compute summary metrics into a `pandas.DataFrame`, and view phase-diagram-style heatmaps of mean participation ratio, interference, and two regime flags.

## 2. Theory (one paragraph)

When a network has fewer hidden dimensions than features (`m < n`) and inputs are **sparse** (most features inactive at any time), the model can compress information by storing multiple features along non-orthogonal directions in hidden space. Each neuron then represents several features — it is **polysemantic** — and the off-diagonal entries of `WᵀW` quantify the resulting **interference**. As feature sparsity rises and width shrinks, superposition becomes the optimal strategy.

Reference: Elhage et al., *Toy Models of Superposition* — https://transformer-circuits.pub/2022/toy_model/index.html

## 3. How to Run

Environment uses [mamba](https://mamba.readthedocs.io/):

```bash
mamba create -n superposition -c conda-forge -c pytorch \
    python=3.11 pytorch cpuonly numpy matplotlib pandas streamlit -y

mamba run -n superposition python tests.py        # 16 tests
mamba run -n superposition streamlit run app.py   # launch UI
```

## 4. Interpreting Single-Run Plots

| Plot | What to look at |
|---|---|
| **Feature Interference (WᵀW)** | Diagonal = how strongly each feature is represented. Off-diagonal = pairs of features sharing hidden directions (red/blue = positive/negative interference). A clean diagonal ⇒ no superposition. Off-diagonal blocks ⇒ features bound together. |
| **Neuron–Feature Weights (W)** | One row per hidden neuron. A row with one strong column = monosemantic. A row with several strong columns of mixed sign = polysemantic. |
| **Validation Loss** | MSE on a fixed validation batch (so curves are comparable across runs). Look for plateau ⇒ converged. |
| **Effective Feature Count (PR)** | Per-neuron participation ratio `(Σ\|w\|)² / Σw²`. PR ≈ 1 ⇒ monosemantic; PR ≫ 1 ⇒ polysemantic. Red dashed line marks PR=1. |

## 5. Interpreting Sweep Plots

The sweep trains across a (hidden width × sparsity) grid with multiple seeds per cell, then renders four heatmaps:

| Heatmap | Reading |
|---|---|
| **mean_pr** | Average participation ratio across neurons. Rises toward the **high-sparsity / narrow-width** corner — the canonical superposition regime. |
| **interference** | Mean abs off-diagonal of `WᵀW`. High values mean feature directions overlap. Note: this score grows mechanically with width, so compare along sparsity rather than across widths. |
| **Polysemantic-neuron regime** | Fraction of seeds with `mean_pr > pr_threshold` OR `frac_polysemantic > frac_threshold`. Neuron-level flag. |
| **Interference regime** | Fraction of seeds with `interference > interference_threshold`. Feature-level flag. |

A run is "fully in superposition" when **both** flags fire. The two are tracked separately because diffuse weights without real interference often indicate under-training rather than superposition.

The default thresholds (PR=2.0, frac=0.3, interference=0.05) are starting points — tune them in the Sweep tab to find a phase boundary for your grid.

## 6. Limitations & Future Work

- **No importance weighting**: every feature contributes equally to the MSE. The paper uses geometric importance to surface the prioritization story.
- **Plain MSE only**: no ReLU on the output, no per-feature loss balancing.
- **Unnormalized interference score**: `|off-diag WᵀW|` scales with width and weight magnitude, making cross-width comparisons noisy. Normalizing by feature norms or using cosine interference would help.
- **No stochasticity over data importance / correlations**: features are i.i.d. Bernoulli-uniform, so correlated-feature regimes from the paper aren't reproducible.
- **CPU-only by default**; small enough that GPU isn't needed, but the code accepts a `device` arg.

**Possible extensions**: importance-weighted loss, dimensionality-per-feature plot from §3 of the paper, correlated feature distributions, larger grids with caching, and interactive thresholding (drag a slider, regime heatmap recolors live without re-training).
