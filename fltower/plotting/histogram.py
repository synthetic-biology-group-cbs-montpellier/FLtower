"""Histogram plotting for single-channel distributions."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import gmean

from fltower.core.cleaning import clean_data
from fltower.core.gating.interval import compute_interval_stats
from fltower.plotting._helpers import get_label


def plot_histogram(
    singlets,
    x_param,
    file_name,
    ax,
    x_scale="linear",
    kde=False,
    color="blue",
    xlim=None,
    gates=None,
):
    """
    Plot a histogram of the singlets data for a given parameter and calculate statistics.
    """
    cleaned_data = clean_data(singlets, [x_param])
    if x_scale == "log" and cleaned_data[x_param].min() <= 0:
        cleaned_data = cleaned_data[cleaned_data[x_param] > 0]

    sns.histplot(
        cleaned_data[x_param],
        bins=100,
        kde=kde,
        ax=ax,
        log_scale=(x_scale == "log"),
        color=color,
    )

    if xlim:
        ax.set_xlim(xlim)

    # Keep log ticks if x_scale is 'log'
    if x_scale == "log":
        ax.set_xscale("log")

        # Custom formatter function
        def log_tick_formatter(x, pos):
            return f"$10^{{{int(np.log10(x))}}}$"

        ax.xaxis.set_major_formatter(plt.FuncFormatter(log_tick_formatter))
        ax.xaxis.set_major_locator(plt.LogLocator(numticks=6))
        ax.xaxis.set_minor_locator(plt.LogLocator(subs="all", numticks=10))

    # Calculate global statistics
    global_gm = gmean(cleaned_data[x_param])
    global_median = cleaned_data[x_param].median()
    num_events = len(cleaned_data)

    stats = {
        "Global_GM": global_gm,
        "Global_Median": global_median,
        "Num_Events": num_events,
    }

    if gates:
        interval_stats = compute_interval_stats(cleaned_data, x_param, gates)
        stats.update(interval_stats)

        gate_colors = plt.cm.rainbow(np.linspace(0, 1, len(gates)))
        y_max = ax.get_ylim()[1]
        for i, ((gate_min, gate_max), gate_color) in enumerate(zip(gates, gate_colors)):
            ax.axvline(gate_min, color=gate_color, linestyle="--")
            ax.axvline(gate_max, color=gate_color, linestyle="--")

            percentage = stats[f"Gate_{i + 1}_Percentage"]

            # Add gate label with percentage
            gate_center = (gate_min + gate_max) / 2
            y_pos = y_max * (0.95 - i * 0.1)  # Adjust vertical position for each gate
            ax.text(
                gate_center,
                y_pos,
                f"Gate {i+1}: {percentage:.2f}%",
                color=gate_color,
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10,
                bbox=dict(facecolor="white", edgecolor=gate_color, alpha=0.7, pad=2),
            )

    ax.set_xlabel(get_label(x_param))
    ax.set_ylabel("Count")
    ax.set_title(
        f"{file_name} - {get_label(x_param)}", fontsize=10, fontweight="bold", pad=20
    )

    return stats
