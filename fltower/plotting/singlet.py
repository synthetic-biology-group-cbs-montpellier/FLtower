"""Singlet gate hexbin plot (SSC-A vs SSC-H)."""

import logging

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from fltower.core.gating.singlet import SINGLET_RATIO_LOWER, SINGLET_RATIO_UPPER
from fltower.plotting._helpers import get_label

logger = logging.getLogger("fltower")


def plot_singlet_gate(
    data,
    ssc_a="SSC-A",
    ssc_h="SSC-H",
    ax=None,
    file_name=None,
    singlet_lower=SINGLET_RATIO_LOWER,
    singlet_upper=SINGLET_RATIO_UPPER,
):
    """
    Plot SSC-A vs SSC-H hexbin plot with the singlet gate for original data on a given axis.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    # Filter out non-positive values
    mask = (data[ssc_a] > 0) & (data[ssc_h] > 0)
    data_filtered = data[mask]

    if len(data_filtered) == 0:
        logger.warning(f"No valid data found for {file_name}")
        ax.text(0.5, 0.5, "No Data", ha="center", va="center")
        ax.axis("off")
        return 0

    # Calculate the ratio of SSC-H to SSC-A
    ssc_ratio = data_filtered[ssc_h] / data_filtered[ssc_a]

    # Create a boolean mask for singlets (same thresholds as remove_doublets)
    singlet_mask = (ssc_ratio >= singlet_lower) & (ssc_ratio <= singlet_upper)

    # Plot hexbin
    hb = ax.hexbin(
        data_filtered[ssc_a],
        data_filtered[ssc_h],
        gridsize=50,
        cmap="viridis",
        bins="log",
        xscale="log",
        yscale="log",
    )

    ax.set_xlabel(get_label(ssc_a))
    ax.set_ylabel(get_label(ssc_h))
    ax.set_title(file_name if file_name else "Singlet Gate", fontsize=8)

    # Plot the singlet gate
    x = np.logspace(
        np.log10(data_filtered[ssc_a].min()), np.log10(data_filtered[ssc_a].max()), 100
    )
    ax.plot(x, singlet_lower * x, "r--", linewidth=0.5)
    ax.plot(x, singlet_upper * x, "r--", linewidth=0.5)

    ax.set_xscale("log")
    ax.set_yscale("log")

    # Remove tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Ensure square aspect ratio
    ax.set_aspect("equal", adjustable="box")

    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(hb, cax=cax)

    # Adjust colorbar height to match the plot area
    plt.draw()  # This is necessary to update the plot layout
    ax_bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
    cax.set_position(
        [cax.get_position().x0, ax_bbox.y0, cax.get_position().width, ax_bbox.height]
    )

    # Calculate and return the percentage of singlets
    singlet_percentage = (singlet_mask.sum() / len(data)) * 100
    ax.text(
        0.05,
        0.95,
        f"Singlets: {singlet_percentage:.1f}%",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        verticalalignment="top",
    )

    return singlet_percentage
