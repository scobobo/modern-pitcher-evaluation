"""Render the paper's four figures as PNGs for the Word edition.

The web paper draws these as inline SVG, which Word cannot embed. These are
redrawn in matplotlib at print resolution, using the same palette and the same
numbers as the HTML version so the two editions cannot drift.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import OUTPUT_DIR

FIG_DIR = OUTPUT_DIR / "paper" / "figures"

# Same three validated categorical slots the web edition uses.
C_SHAPE = "#1baf7a"
C_COMP = "#2a78d6"
C_SPIN = "#eb6834"
C_NEUTRAL = "#98a5b2"
INK = "#11161c"
INK2 = "#4a5563"
RULE = "#dce1e7"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.edgecolor": RULE,
        "axes.labelcolor": INK2,
        "text.color": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def _clean(ax, *, keep_left: bool = False) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if not keep_left:
        ax.spines["left"].set_visible(False)
    ax.set_axisbelow(True)


def fig_trajectory(path):
    """Two trajectories, same release and same destination, different geometry."""
    fig, ax = plt.subplots(figsize=(6.6, 2.9))

    t = np.linspace(0, 1, 200)

    def bezier(p0, p1, p2):
        return (
            (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t**2 * p2[0],
            (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t**2 * p2[1],
        )

    start, end = (0.0, 6.0), (55.0, 2.6)
    x_low, y_low = bezier(start, (28.0, 7.6), end)      # low IVB: arcs up, drops steeply
    x_high, y_high = bezier(start, (32.0, 5.4), end)    # high IVB: carries, arrives flat

    ax.plot(x_low, y_low, color=C_NEUTRAL, lw=2.4, label="Low IVB — arrives steep (≈ −6.5°)")
    ax.plot(x_high, y_high, color=C_SHAPE, lw=2.4, label="High IVB — arrives flat (≈ −4.2°)")

    ax.add_patch(plt.Rectangle((53.8, 1.6), 2.6, 2.0, fill=False, ec=RULE, ls=(0, (3, 3)), lw=1))
    ax.text(55.1, 1.2, "strike zone", ha="center", fontsize=7.5, color=INK2)

    ax.plot([start[0]], [start[1]], "o", color=INK, ms=6)
    ax.text(0.4, 6.45, "release", fontsize=7.5, color=INK2)
    ax.plot([end[0]], [end[1]], "o", color=INK, ms=5)

    # Arrival tangents, extended back from the plate along each final direction
    # so the difference in arrival angle is the visible point of the figure.
    for (xs, ys), col, dy_label, text in (
        ((x_low, y_low), C_NEUTRAL, 0.62, "steep"),
        ((x_high, y_high), C_SHAPE, -0.30, "flat"),
    ):
        dx, dy = xs[-1] - xs[-6], ys[-1] - ys[-6]
        norm = np.hypot(dx, dy)
        bx, by = end[0] - dx / norm * 17, end[1] - dy / norm * 17
        ax.plot([end[0], bx], [end[1], by], color=col, ls=(0, (4, 3)), lw=1.1)
        ax.text(bx - 1.2, by + dy_label, text, fontsize=7.5, color=col, ha="right", va="center")

    ax.set_xlim(-3, 62)
    ax.set_ylim(0.6, 8.2)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines[:].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0.02, -0.04))
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_ladder(path):
    """Incremental CV R2 by block, two panels with independent scales."""
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5))

    for ax, vals, title, xmax, fmt in (
        (axes[0], [0.00024, 0.00004, 0.00108], "Predicting run value", 0.0012, "{:.5f}"),
        (axes[1], [0.00580, 0.00169, 0.01952], "Predicting whiffs on swings", 0.020, "{:.4f}"),
    ):
        labels = ["Velocity", "Spin", "Shape"]
        colors = [C_COMP, C_SPIN, C_SHAPE]
        y = np.arange(3)[::-1]
        ax.barh(y, vals, height=0.55, color=colors)
        for yi, v in zip(y, vals):
            ax.text(v + xmax * 0.025, yi, fmt.format(v), va="center", fontsize=8, color=INK, fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.set_xlim(0, xmax * 1.32)
        ax.set_title(title, fontsize=9, color=INK2, pad=8)
        ax.set_xlabel("incremental CV $R^2$", fontsize=8)
        ax.grid(axis="x", color=RULE, lw=0.7)
        _clean(ax)

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_reliability(path):
    """Year-over-year correlation, shape metrics versus outcome metrics."""
    metrics = [
        ("Release height", 0.956, "shape"),
        ("Extension", 0.949, "shape"),
        ("VAA (adjusted)", 0.940, "shape"),
        ("Velocity", 0.925, "shape"),
        ("Residual spin", 0.915, "shape"),
        ("Horizontal break", 0.886, "shape"),
        ("Induced vert. break", 0.836, "shape"),
        ("Whiff rate", 0.628, "outcome"),
        ("Run value / pitch", 0.216, "outcome"),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    y = np.arange(len(metrics))[::-1]
    vals = [m[1] for m in metrics]
    colors = [C_SHAPE if m[2] == "shape" else C_NEUTRAL for m in metrics]

    ax.barh(y, vals, height=0.6, color=colors)
    for yi, v in zip(y, vals):
        ax.text(v + 0.012, yi, f"{v:.3f}", va="center", fontsize=8, color=INK, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([m[0] for m in metrics], fontsize=8.5)
    ax.set_xlim(0, 1.08)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("correlation between consecutive seasons (1,889 pitcher-season pairs)", fontsize=8)
    ax.grid(axis="x", color=RULE, lw=0.7)
    _clean(ax)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=C_SHAPE),
        plt.Rectangle((0, 0), 1, 1, color=C_NEUTRAL),
    ]
    ax.legend(handles, ["Shape & delivery", "Outcome statistics"], frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_crossover(path):
    """Next-season whiff prediction: shape versus past results, by sample size."""
    n = np.array([30, 60, 125, 250, 500])
    shape = np.array([0.1882, 0.1927, 0.1915, 0.2062, 0.1884])
    results = np.array([0.0941, 0.1597, 0.2502, 0.3209, 0.3621])

    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    # The lines cross between the 60- and 125-pitch samples; linear interpolation
    # in log space puts it at ~80. Drawn where it actually happens, not rounded.
    ax.axvline(80, color=RULE, ls=(0, (3, 4)), lw=1.2)
    ax.text(84, 0.372, "crossover ≈ 80 pitches", fontsize=8, color=INK2)

    ax.plot(n, shape, "o-", color=C_SHAPE, lw=2.2, ms=6, label="Shape only", mec="white", mew=1.4)
    ax.plot(n, results, "s-", color=C_COMP, lw=2.2, ms=6, label="His own past results", mec="white", mew=1.4)

    ax.set_xscale("log")
    ax.set_xticks(n)
    ax.set_xticklabels([str(v) for v in n], fontsize=8.5)
    ax.minorticks_off()
    ax.set_ylim(0, 0.41)
    ax.set_xlabel("four-seamers observed (log scale) — each point built from exactly this many pitches", fontsize=8)
    ax.set_ylabel("out-of-sample $R^2$", fontsize=8)
    ax.grid(axis="y", color=RULE, lw=0.7)
    _clean(ax, keep_left=True)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_all() -> dict[str, "object"]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "trajectory": FIG_DIR / "fig1_trajectory.png",
        "ladder": FIG_DIR / "fig2_ladder.png",
        "reliability": FIG_DIR / "fig3_reliability.png",
        "crossover": FIG_DIR / "fig4_crossover.png",
    }
    fig_trajectory(paths["trajectory"])
    fig_ladder(paths["ladder"])
    fig_reliability(paths["reliability"])
    fig_crossover(paths["crossover"])
    return paths


if __name__ == "__main__":
    for name, p in build_all().items():
        print(f"{name}: {p.name}")
