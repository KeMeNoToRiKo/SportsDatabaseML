"""Static report charts (matplotlib) for Group 24. Used by train.py."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage

# Validated palette (light surface): blue, orange, aqua, violet pass all-pairs CVD checks
# for scatter plots. Marker shapes are a second identity cue.
CLUSTER_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
CLUSTER_MARKERS = ["o", "s", "^", "D"]
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "legend.frameon": False,
    "legend.labelcolor": INK_2,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
})


def _style_line(ax, x, y, color, label):
    ax.plot(x, y, color=color, linewidth=2, solid_capstyle="round", label=label,
            marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=2)


def plot_k_selection(k_table: pd.DataFrame, chosen_k: int, path) -> None:
    """Elbow and silhouette side by side (different scales, so two panels - never a dual axis)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))

    _style_line(ax1, k_table["k"], k_table["kmeans_inertia"], CLUSTER_COLORS[0], "K-Means")
    ax1.set_title("Elbow: K-Means within-cluster error")
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Inertia (lower = tighter)")

    _style_line(ax2, k_table["k"], k_table["kmeans_silhouette"], CLUSTER_COLORS[0], "K-Means")
    _style_line(ax2, k_table["k"], k_table["hierarchical_silhouette"], CLUSTER_COLORS[1],
                "Hierarchical (average)")
    ax2.set_title("Silhouette score by k")
    ax2.set_xlabel("Number of clusters (k)")
    ax2.set_ylabel("Silhouette (higher = better separated)")
    ax2.legend(loc="lower left")

    best = k_table.set_index("k").loc[chosen_k]
    for ax, y in [(ax1, best["kmeans_inertia"]), (ax2, best["kmeans_silhouette"])]:
        ax.axvline(chosen_k, color=AXIS, linewidth=1, zorder=0)
        ax.annotate(f"chosen k = {chosen_k}", (chosen_k, y), xytext=(8, 8),
                    textcoords="offset points", color=INK, fontsize=9)
    for ax in (ax1, ax2):
        ax.set_xticks(k_table["k"])
        ax.margins(y=0.14)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _jitter(n, seed):
    return np.random.default_rng(seed).uniform(-0.09, 0.09, n)


def plot_cluster_scatter(df: pd.DataFrame, panels: dict[str, tuple[np.ndarray, list[str]]], path) -> None:
    """One panel per algorithm. x = perimeter skill, y = interior skill (both on the 1-10 scale)."""
    perimeter = df[["shooting", "passing"]].mean(axis=1).to_numpy() + _jitter(len(df), 1)
    interior = df[["defense", "rebounding"]].mean(axis=1).to_numpy() + _jitter(len(df), 2)

    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.6), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (title, (labels, names)) in zip(axes, panels.items()):
        for c, name in enumerate(names):
            mask = labels == c
            ax.scatter(perimeter[mask], interior[mask], s=58 if CLUSTER_MARKERS[c % 4] != "^" else 74, c=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
                       marker=CLUSTER_MARKERS[c % len(CLUSTER_MARKERS)], edgecolors=SURFACE, linewidths=1.5,
                       label=f"C{c + 1} {name} ({mask.sum()})", zorder=3)
        ax.set_title(title)
        ax.set_xlabel("Perimeter skill (avg of shooting, passing)")
        ax.set_xlim(1.5, 10)
        ax.set_ylim(1.5, 10)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, fontsize=8.5)
    axes[0].set_ylabel("Interior skill (avg of defense, rebounding)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_dendrogram(X: np.ndarray, ids: list[str], k: int, method: str, path) -> None:
    Z = linkage(X, method=method)
    cut = (Z[-k, 2] + Z[-k + 1, 2]) / 2       # height between the k-th and (k-1)-th merge
    fig, ax = plt.subplots(figsize=(11, 4.2))
    dendrogram(Z, labels=ids, ax=ax, color_threshold=0, above_threshold_color=INK_2, leaf_font_size=6.5)
    ax.axhline(cut, color=CLUSTER_COLORS[1], linewidth=1.2)
    ax.annotate(f"cut here -> {k} clusters", (0, cut), xytext=(4, 4), textcoords="offset points",
                color=INK, fontsize=9)
    ax.set_title(f"Hierarchical clustering dendrogram ({method} linkage, standardized inputs)")
    ax.set_ylabel("Merge distance")
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", colors=INK_2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_team_balance(balance: pd.DataFrame, n_teams: int, path) -> None:
    """Small multiples: gap between strongest and weakest team for each method, one panel per measure."""
    measures = [("skill_gap", "Skill gap (rating points)"),
                ("fitness_gap", "Fitness gap (rating points)"),
                ("availability_gap", "Availability gap (% points)")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    y = np.arange(len(balance))
    for ax, (col, title) in zip(axes, measures):
        values = balance[col].to_numpy()
        ax.barh(y, values, height=0.36, color=CLUSTER_COLORS[0], zorder=3)
        for yi, v in zip(y, values):
            ax.annotate(f"{v:.2f}", (v, yi), xytext=(4, 0), textcoords="offset points",
                        va="center", color=INK, fontsize=9)
        ax.set_yticks(y, balance.index if ax is axes[0] else [""] * len(y))
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, values.max() * 1.3 if values.max() > 0 else 1)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", colors=INK_2, length=0)
    fig.suptitle(f"Strongest minus weakest team average ({n_teams} teams, lower = more balanced)",
                 x=0.01, ha="left", color=INK, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
