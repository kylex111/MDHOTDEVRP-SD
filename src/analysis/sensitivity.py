"""Plot RA parameter sensitivity without reading historical tables at import time."""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_ra_sensitivity(path, value_column, *, tick_step=2, label=None):
    """Read the supplied parameter summary and return the figure and axes."""
    df = pd.read_excel(path)
    grid = df.pivot(index="ra_safe_param", columns="ra_risk_param", values=value_column)
    grid = grid.sort_index(axis=0).sort_index(axis=1)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(
        np.ma.masked_invalid(grid.values.astype(float)),
        origin="lower",
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
    )
    x_values, y_values = (grid.columns.to_numpy(), grid.index.to_numpy())
    ax.set_xticks(np.arange(0, len(x_values), tick_step))
    ax.set_xticklabels(
        [f"{v:.2f}" for v in x_values[::tick_step]], rotation=45, ha="right"
    )
    ax.set_yticks(np.arange(0, len(y_values), tick_step))
    ax.set_yticklabels([f"{v:.2f}" for v in y_values[::tick_step]])
    ax.set_xlabel("RA$_{risk}$")
    ax.set_ylabel("RA$_{safe}$")
    fig.colorbar(im, ax=ax).set_label(label or value_column)
    fig.tight_layout()
    return (fig, ax)
