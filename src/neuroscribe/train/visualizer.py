"""Visualization utilities."""

from __future__ import annotations

import os
from typing import Optional

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

_COLOR_RAW = "#9aa0a6"
_COLOR_TRUE = "#111111"
_COLOR_PRED = "#1a73e8"
_BG = "#f7f7f7"


def _style_axes(ax):
    ax.set_facecolor(_BG)
    ax.grid(True, axis="y", color="#d0d0d0", linewidth=0.6, alpha=0.5)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#b0b0b0")
    ax.spines["bottom"].set_color("#b0b0b0")
    ax.tick_params(axis="both", labelsize=8, colors="#303030")


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str,
    title: Optional[str] = None,
    y_raw: Optional[np.ndarray] = None,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    t = np.arange(y_true.shape[0])

    n_dim = y_true.shape[1]
    fig, axes = plt.subplots(n_dim, 1, figsize=(10, 2.4 * n_dim), sharex=True)
    if n_dim == 1:
        axes = [axes]
    fig.set_facecolor("white")

    for d in range(n_dim):
        if y_raw is not None:
            axes[d].plot(
                t,
                y_raw[:, d],
                label=None,
                linewidth=1.0,
                alpha=0.5,
                color=_COLOR_RAW,
            )
        axes[d].plot(
            t,
            y_true[:, d],
            label="target" if d == 0 else None,
            linewidth=1.6,
            color=_COLOR_TRUE,
        )
        axes[d].plot(
            t,
            y_pred[:, d],
            label="pred" if d == 0 else None,
            linewidth=1.4,
            alpha=0.9,
            color=_COLOR_PRED,
        )
        axes[d].set_ylabel(f"dim {d}", fontsize=8, color="#303030")
        axes[d].margins(x=0.0)
        if t.size:
            axes[d].set_xlim(t[0], t[-1])
        _style_axes(axes[d])

    axes[-1].set_xlabel("time steps", fontsize=9, color="#303030")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(labels),
            frameon=False,
            bbox_to_anchor=(0.5, 0.98),
            fontsize=9,
        )
    top = 0.88 if handles else 0.96
    fig.tight_layout(rect=(0.0, 0.0, 1.0, top))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_mean_window(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str,
    title: Optional[str] = None,
    y_raw: Optional[np.ndarray] = None,
) -> None:
    plot_predictions(y_true, y_pred, out_path, title=title, y_raw=y_raw)
