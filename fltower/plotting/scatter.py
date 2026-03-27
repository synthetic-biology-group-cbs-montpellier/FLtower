"""Scatter plot with manual quadrant gates."""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from fltower.core.cleaning import clean_data
from fltower.core.gating.quadrant import compute_quadrant_stats
from fltower.plotting._helpers import get_label

logger = logging.getLogger("fltower")


def plot_scatter_with_manual_gates(
    data,
    x_param,
    y_param,
    file_name,
    ax,
    scatter_type="scatter",
    cmap="viridis",
    x_scale="linear",
    y_scale="linear",
    xlim=None,
    ylim=None,
    gridsize=100,
    quadrant_gates=None,
):
    logger.debug(f"Plotting {scatter_type} scatter with manual gates for {file_name}")
    logger.debug(f"Quadrant gates: {quadrant_gates}")

    # Check if both parameters exist in the data
    if x_param not in data.columns or y_param not in data.columns:
        logger.error(
            f"One or both parameters ({x_param}, {y_param}) not found in data for {file_name}"
        )
        return None

    # Clean the data
    cleaned_data = clean_data(data, [x_param, y_param])

    # Downsample if there are too many points
    if len(cleaned_data) > 1000000:
        cleaned_data = cleaned_data.sample(n=1000000, random_state=42)

    # Handle non-positive values for log scale
    if x_scale == "log":
        n_clipped_x = (cleaned_data[x_param] <= 0).sum()
        if n_clipped_x > 0:
            logger.warning(
                "%d events with %s <= 0 clipped to 1 for log scale in %s",
                n_clipped_x,
                x_param,
                file_name,
            )
        cleaned_data[x_param] = cleaned_data[x_param].clip(lower=1)
    if y_scale == "log":
        n_clipped_y = (cleaned_data[y_param] <= 0).sum()
        if n_clipped_y > 0:
            logger.warning(
                "%d events with %s <= 0 clipped to 1 for log scale in %s",
                n_clipped_y,
                y_param,
                file_name,
            )
        cleaned_data[y_param] = cleaned_data[y_param].clip(lower=1)

    if scatter_type == "density":
        # Plot hexbin
        hb = ax.hexbin(
            cleaned_data[x_param],
            cleaned_data[y_param],
            gridsize=gridsize,
            cmap=cmap,
            xscale=x_scale,
            yscale=y_scale,
            bins="log",
            mincnt=1,
            rasterized=True,
        )

        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(hb, cax=cax, label="Count")
    else:
        # Plot scatter
        ax.scatter(
            cleaned_data[x_param], cleaned_data[y_param], c="blue", s=0.1, alpha=0.5
        )

    ax.set_xlabel(get_label(x_param))
    ax.set_ylabel(get_label(y_param))
    ax.set_title(os.path.basename(file_name), fontsize=12, fontweight="bold")

    # Set scales and keep log ticks
    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)

    # Custom formatter function
    def log_tick_formatter(x, pos):
        return f"$10^{{{int(np.log10(x))}}}$"

    if x_scale == "log":
        ax.xaxis.set_major_formatter(plt.FuncFormatter(log_tick_formatter))
        ax.xaxis.set_major_locator(plt.LogLocator(numticks=6))
        ax.xaxis.set_minor_locator(plt.LogLocator(subs="all", numticks=10))
    if y_scale == "log":
        ax.yaxis.set_major_formatter(plt.FuncFormatter(log_tick_formatter))
        ax.yaxis.set_major_locator(plt.LogLocator(numticks=6))
        ax.yaxis.set_minor_locator(plt.LogLocator(subs="all", numticks=10))

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    # Ensure square aspect ratio
    ax.set_aspect("equal", adjustable="box")

    # Compute quadrant statistics (pure computation)
    gate_stats, x_mid, y_mid = compute_quadrant_stats(
        cleaned_data, x_param, y_param, quadrant_gates
    )

    # Define positions for labels
    label_positions = {
        "Q1": (0.95, 0.95),
        "Q2": (0.05, 0.95),
        "Q3": (0.05, 0.05),
        "Q4": (0.95, 0.05),
    }

    # Add labels to corners
    for quad_name, position in label_positions.items():
        percentage = gate_stats[f"{quad_name}_Percentage"]
        ax.text(
            position[0],
            position[1],
            f"{quad_name}\n{percentage:.1f}%",
            horizontalalignment=(
                "right" if "Q1" in quad_name or "Q4" in quad_name else "left"
            ),
            verticalalignment=(
                "top" if "Q1" in quad_name or "Q2" in quad_name else "bottom"
            ),
            transform=ax.transAxes,
            fontsize=6,
            fontweight="bold",
            color="red",
        )

    # Add quadrant lines
    ax.axvline(x_mid, color="red", linestyle="--", linewidth=1)
    ax.axhline(y_mid, color="red", linestyle="--", linewidth=1)

    logger.debug(
        "Finished plotting %s scatter with manual gates for %s", scatter_type, file_name
    )
    return gate_stats
