"""Bar-chart visualisation of triplicate statistics."""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

logger = logging.getLogger("fltower")


def plot_triplicate_stats(
    triplicate_stats, metric_column, output_dir, title=None, custom_names=None
):
    if len(triplicate_stats) == 0:
        logger.warning(
            "No complete triplicates found for %s. Skipping plot.", metric_column
        )
        return

    # Set the style
    plt.style.use("seaborn-v0_8-whitegrid")

    # Define a custom color palette with 4 colors from coolwarm colormap
    custom_palette = ["#3a4cc0", "#90afdb", "#f1a38b", "#b40326"]

    fig, ax = plt.subplots(figsize=(16, 8))  # Increased figure size

    groups = triplicate_stats["Group"]
    means = triplicate_stats[f"{metric_column}_Mean"]
    stds = triplicate_stats[f"{metric_column}_Std"]

    # Calculate positions for grouped bars
    group_size = 4
    bar_width = 0.6
    group_width = group_size * bar_width + 1.2  # Extra space between groups
    positions = [
        i * group_width + j * bar_width
        for i in range(len(groups) // group_size + 1)
        for j in range(group_size)
    ][: len(groups)]

    # Create a color list that repeats the custom palette
    colors = [custom_palette[i % len(custom_palette)] for i in range(len(groups))]

    # Plot bars
    bars = ax.bar(
        positions,
        means,
        width=bar_width,
        align="center",
        alpha=0.8,
        color=colors,
        edgecolor="black",
        linewidth=1.5,
    )

    # Plot error bars (only positive)
    ax.errorbar(
        positions,
        means,
        yerr=[np.zeros_like(stds), stds],
        fmt="none",
        color="black",
        capsize=5,
        capthick=1.5,
        elinewidth=1.5,
    )

    # Customize the plot
    ax.set_xlabel("Sample Group", fontsize=12, fontweight="bold")
    ax.set_ylabel(metric_column, fontsize=12, fontweight="bold")
    ax.set_title(
        title or f"{metric_column} by Sample Group",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Set x-ticks and labels
    if custom_names and len(custom_names) == len(groups):
        ax.set_xticks(positions)
        ax.set_xticklabels(custom_names, rotation=45, ha="right", fontsize=10)
    else:
        ax.set_xticks(positions)
        ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=10)

    # Customize grid
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    # Calculate the maximum height (including error bars)
    max_height = max(mean + std for mean, std in zip(means, stds))
    label_height_1 = (
        max_height * 1.05
    )  # Place first line of labels 5% above the highest point
    label_height_2 = (
        max_height * 1.10
    )  # Place second line of labels 10% above the highest point

    # Add value labels above each bar, alternating between two lines
    for i, (bar, mean) in enumerate(zip(bars, means)):
        if i % 2 == 0:
            height = label_height_1
            va = "bottom"
        else:
            height = label_height_2
            va = "bottom"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{mean:.2f}",
            ha="center",
            va=va,
            fontsize=8,
            fontweight="bold",
        )

    # Adjust y-axis limit to accommodate labels
    ax.set_ylim(0, label_height_2 * 1.1)  # Add 10% padding above the highest labels

    # Add row labels and horizontal lines
    for i in range(0, len(groups), group_size):
        group_start = positions[i]
        group_end = positions[min(i + group_size - 1, len(positions) - 1)]
        group_center = (group_start + group_end) / 2

        # Add horizontal line
        ax.axhline(
            y=-0.05 * max_height,
            xmin=(group_start - bar_width / 2) / ax.get_xlim()[1],
            xmax=(group_end + bar_width / 2) / ax.get_xlim()[1],
            color="black",
            linewidth=1,
        )

        # Add row label
        row_letter = chr(65 + i // group_size)  # 65 is ASCII for 'A'
        ax.text(
            group_center,
            -0.1 * max_height,
            f"Row {row_letter}",
            ha="center",
            va="top",
            fontweight="bold",
        )

    # Adjust bottom margin to make room for row labels
    plt.subplots_adjust(bottom=0.2)

    # Customize spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_visible(False)  # Hide bottom spine

    # Add a legend for color groups
    legend_elements = [
        Patch(facecolor=color, edgecolor="black", label=f"Group {i+1}")
        for i, color in enumerate(custom_palette)
    ]
    ax.legend(
        handles=legend_elements,
        title="Color Groups",
        loc="upper left",
        bbox_to_anchor=(1, 1),
        fontsize=8,
    )

    plt.tight_layout()
    plot_filename = f"{title.replace(' ', '_').lower()}_triplicate_stats.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved triplicate plot for {metric_column} to: {plot_path}")
