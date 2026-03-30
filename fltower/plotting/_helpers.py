"""Shared helpers for plotting modules."""

import matplotlib.pyplot as plt
import numpy as np

# ── Label mapping ─────────────────────────────────────────────────────

LABEL_MAP = {
    "BL1-H": "GFP",
    "YL2-H": "RFP",
    "SSC-A": "SSC-A",
    "SSC-H": "SSC-H",
    # Add more mappings here as needed
}


def get_label(param):
    return LABEL_MAP.get(param, param)


# ── Log-axis formatting ──────────────────────────────────────────────


def log_tick_formatter(x, pos):
    """Format a log-scale tick as $10^{n}$."""
    return f"$10^{{{int(np.log10(x))}}}$"


def apply_log_axis_formatting(ax, axis="x"):
    """Apply log-scale tick formatting, major and minor locators.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    axis : ``"x"`` or ``"y"``
    """
    ax_obj = ax.xaxis if axis == "x" else ax.yaxis
    ax_obj.set_major_formatter(plt.FuncFormatter(log_tick_formatter))
    ax_obj.set_major_locator(plt.LogLocator(numticks=6))
    ax_obj.set_minor_locator(plt.LogLocator(subs="all", numticks=10))


# ── 96-well plate grid helpers ────────────────────────────────────────

NUM_ROWS = 8
NUM_COLS = 12


def create_plate_grid():
    """Create a (NUM_ROWS+1) × (NUM_COLS+1) subplot grid."""
    return plt.subplots(
        NUM_ROWS + 1,
        NUM_COLS + 1,
        figsize=(42, 28),
        gridspec_kw={
            "height_ratios": [0.5] + [1] * NUM_ROWS,
            "width_ratios": [0.5] + [1] * NUM_COLS,
        },
    )


def add_plate_labels(ax):
    """Add column (1–12) and row (A–H) labels to a plate grid."""
    for col in range(1, NUM_COLS + 1):
        ax[0, col].text(
            0.5,
            0.2,
            str(col),
            ha="center",
            va="center",
            fontweight="bold",
            fontsize=18,
        )
        ax[0, col].axis("off")
    for row in range(1, NUM_ROWS + 1):
        ax[row, 0].text(
            0.5,
            0.5,
            chr(64 + row),
            ha="center",
            va="center",
            fontweight="bold",
            fontsize=18,
        )
        ax[row, 0].axis("off")
    ax[0, 0].axis("off")


def adjust_scatter_positions(ax):
    """Nudge scatter subplot rows upward and enforce square aspect."""
    for row in range(1, NUM_ROWS + 1):
        for col in range(1, NUM_COLS + 1):
            pos = ax[row, col].get_position()
            pos.y0 += 0.02 * (row - 1)
            pos.y1 += 0.02 * (row - 1)
            ax[row, col].set_position(pos)
            ax[row, col].set_aspect("equal", adjustable="box")
