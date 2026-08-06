"""Figures for the write-up.

Deliberately plain: one idea per panel, no dual axes, no chartjunk. The point
of each figure is to let a reader check the claim, not to be impressed.
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import OUTPUT_DIR

log = logging.getLogger(__name__)

INK = "#1b1b1b"
VELO_C = "#c2452d"
SPIN_C = "#2d6fc2"
GRID = "#d8d8d8"


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, labelsize=9)


def plot_incremental_skill(ladder_fwd: pd.DataFrame, ladder_rev: pd.DataFrame, filename: str = "fig_ladder.png"):
    """Incremental cross-validated R² per block, under both orderings.

    Showing both orders is the honest presentation: the difference between the
    two bars for a block is the variance it shares with the other block.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, frame, title in (
        (axes[0], ladder_fwd, "Velocity entered first"),
        (axes[1], ladder_rev, "Spin entered first"),
    ):
        blocks = frame[frame["model"] != "controls"]
        colors = [VELO_C if "velo" in m else SPIN_C if "spin" in m else "#7a7a7a" for m in blocks["model"]]
        ax.bar(blocks["model"], blocks["incremental_r2"], color=colors, width=0.6)
        ax.axhline(0, color=INK, linewidth=0.8)
        ax.set_title(title, fontsize=11, color=INK)
        _style(ax)
    axes[0].set_ylabel("incremental CV $R^2$", fontsize=10, color=INK)
    fig.suptitle("Unique predictive value added by each feature block", fontsize=13, color=INK)
    fig.tight_layout()
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", path.name)
    return path


def plot_season_effects(effects: pd.DataFrame, filename: str = "fig_season_effects.png"):
    """Per-season effects in physical units and in standard-deviation units.

    The two panels are the whole argument about H2: a per-mph effect that holds
    flat while the per-SD effect falls means the league changed, not the physics.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(effects["season"], effects["velo_coef_per_mph"], "o-", color=VELO_C, label="velocity (per mph)")
    axes[0].set_title("Physical effect: runs per extra unit", fontsize=11, color=INK)
    axes[0].set_ylabel("runs saved per pitch", fontsize=10, color=INK)

    ax2 = axes[0].twinx()
    ax2.plot(
        effects["season"], effects["spin_coef_per_100rpm"], "s--", color=SPIN_C, label="spin (per 100 rpm)"
    )
    ax2.spines[["top"]].set_visible(False)
    ax2.tick_params(labelsize=9)

    axes[1].plot(effects["season"], effects["velo_coef_per_sd"], "o-", color=VELO_C, label="velocity")
    axes[1].plot(effects["season"], effects["spin_coef_per_sd"], "s--", color=SPIN_C, label="spin (resid.)")
    axes[1].set_title("Practical effect: runs per standard deviation", fontsize=11, color=INK)
    axes[1].set_ylabel("runs saved per pitch", fontsize=10, color=INK)
    axes[1].legend(frameon=False, fontsize=9)

    for ax in axes:
        ax.axhline(0, color=INK, linewidth=0.8)
        _style(ax)
        ax.set_xlabel("season", fontsize=10, color=INK)

    fig.suptitle("Is velocity worth less than it used to be?", fontsize=13, color=INK)
    fig.tight_layout()
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", path.name)
    return path


def plot_binned_response(
    df: pd.DataFrame,
    *,
    target: str = "run_value_pitcher",
    filename: str = "fig_binned_response.png",
    n_bins: int = 12,
):
    """Raw binned means: run value against velocity and against residual spin.

    No model, no controls -- just the data. If a relationship is not visible
    here at all, any model that finds one deserves suspicion.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

    for ax, col, color, label in (
        (axes[0], "release_speed", VELO_C, "release speed (mph)"),
        (axes[1], "spin_resid", SPIN_C, "spin residual (rpm, velocity removed)"),
    ):
        frame = df[[col, target]].dropna()
        # Quantile bins so every point carries the same sample size, which
        # keeps the error bars comparable across the range.
        bins = pd.qcut(frame[col], n_bins, duplicates="drop")
        grouped = frame.groupby(bins, observed=True)[target]
        centers = frame.groupby(bins, observed=True)[col].mean()
        means, counts = grouped.mean(), grouped.size()
        se = grouped.std() / np.sqrt(counts)

        ax.errorbar(centers, means, yerr=1.96 * se, fmt="o-", color=color, capsize=3, linewidth=1.4)
        ax.axhline(frame[target].mean(), color=INK, linewidth=0.8, linestyle=":")
        ax.set_xlabel(label, fontsize=10, color=INK)
        _style(ax)

    axes[0].set_ylabel(f"mean {target}", fontsize=10, color=INK)
    fig.suptitle("Raw relationship before any modelling (95% CI)", fontsize=13, color=INK)
    fig.tight_layout()
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", path.name)
    return path
