"""Scatter plot with manual quadrant gates."""

import logging
import os

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from fltower.core.cleaning import clean_data
from fltower.core.gating.quadrant import compute_quadrant_stats
from fltower.plotting._helpers import apply_log_axis_formatting, get_label

logger = logging.getLogger("fltower")


def _clip_for_log_scale(data, param, scale, file_name):
    """Clip non-positive values to 1 when log scale is requested."""
    if scale != "log":
        return data
    n_clipped = (data[param] <= 0).sum()
    if n_clipped > 0:
        logger.warning(
            "%d events with %s <= 0 clipped to 1 for log scale in %s",
            n_clipped,
            param,
            file_name,
        )
    data[param] = data[param].clip(lower=1)
    return data


def _annotate_quadrants(ax, gate_stats):
    """Add quadrant percentage labels in the four corners of *ax*."""
    label_positions = {
        "Q1": (0.95, 0.95),
        "Q2": (0.05, 0.95),
        "Q3": (0.05, 0.05),
        "Q4": (0.95, 0.05),
    }
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
    cleaned_data = _clip_for_log_scale(cleaned_data, x_param, x_scale, file_name)
    cleaned_data = _clip_for_log_scale(cleaned_data, y_param, y_scale, file_name)

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

    if x_scale == "log":
        apply_log_axis_formatting(ax, "x")
    if y_scale == "log":
        apply_log_axis_formatting(ax, "y")

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

    # Add labels to corners
    _annotate_quadrants(ax, gate_stats)

    # Add quadrant lines
    ax.axvline(x_mid, color="red", linestyle="--", linewidth=1)
    ax.axhline(y_mid, color="red", linestyle="--", linewidth=1)

    logger.debug(
        "Finished plotting %s scatter with manual gates for %s", scatter_type, file_name
    )
    return gate_stats
