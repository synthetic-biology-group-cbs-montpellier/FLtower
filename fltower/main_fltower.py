#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main file of FLtower software.
"""

import logging
import os
import re
import sys
import time
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

from fltower.__version__ import __version__
from fltower.core.gating.singlet import (
    SINGLET_RATIO_LOWER,
    SINGLET_RATIO_UPPER,
    remove_doublets,
)
from fltower.core.statistics import calculate_triplicate_stats
from fltower.data_manager import load_parameters, save_parameters
from fltower.io.fcs_reader import read_fcs
from fltower.plotting._helpers import get_label
from fltower.plotting.histogram import plot_histogram
from fltower.plotting.plate_view import plot_96well_grid
from fltower.plotting.scatter import plot_scatter_with_manual_gates
from fltower.run_args import parse_run_args

# Suppress specific FutureWarnings from seaborn related to pandas deprecation
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

# Suppress Intel MKL warnings
os.environ["MKL_DISABLE_FAST_MM"] = "1"
warnings.filterwarnings("ignore", message=".*Intel MKL.*")

# Suppress RuntimeWarnings from numerical libraries (log of zero, etc.)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")

# Suppress matplotlib colorbar cross-figure warning (known issue with multi-subplot grids)
warnings.filterwarnings("ignore", message=".*Adding colorbar to a different Figure.*")

logger = logging.getLogger("fltower")


def setup_logging(verbose=False, quiet=False, log_file=None):
    """Configure logging for the FLtower application.

    Parameters
    ----------
    verbose : bool
        If True, set log level to DEBUG.
    quiet : bool
        If True, set log level to WARNING.
    log_file : str, optional
        Path to a log file. If provided, a FileHandler is added.
    """
    console_level = (
        logging.DEBUG if verbose else (logging.WARNING if quiet else logging.INFO)
    )
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )

    # Clear any existing handlers (avoid accumulation when called multiple times)
    logger.handlers.clear()

    # Logger gate always at DEBUG so file handler receives everything
    logger.setLevel(logging.DEBUG)

    # Console handler filters per user preference (--verbose / --quiet)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (always at DEBUG level for full traceability)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def extract_well_key(filename):
    base_name = os.path.splitext(os.path.basename(filename))[0]
    match = re.search(r"([A-H])([0-9]{1,2})(?!.*[A-H][0-9]{1,2})", base_name)
    if match:
        letter_part = match.group(1)
        number_part = int(match.group(2))
        return match.group(0), (letter_part, number_part)
    else:
        return base_name, (base_name, 0)


def create_output_structure(results_directory):
    """
    Create necessary subdirectories within the results folder for storing different outputs.
    """
    plots_dir = os.path.join(results_directory, "plots")
    well_plots_dir = os.path.join(results_directory, "96well_plots")
    stats_dir = os.path.join(results_directory, "statistics")
    triplicate_stats_dir = os.path.join(results_directory, "triplicate_statistics")
    triplicate_plots_dir = os.path.join(results_directory, "triplicate_plots")

    for directory in [
        plots_dir,
        well_plots_dir,
        stats_dir,
        triplicate_stats_dir,
        triplicate_plots_dir,
    ]:
        os.makedirs(directory, exist_ok=True)

    return (
        plots_dir,
        well_plots_dir,
        stats_dir,
        triplicate_stats_dir,
        triplicate_plots_dir,
    )


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

    # Define a custom color palette with 4 colors from inferno colormap
    # custom_palette = ['#000004', '#851255', '#f6715b', '#fcfdbf']
    # Define a custom color palette with 4 colors from viridis colormap
    # custom_palette = ['#440154', '#40478b', '#29788e', '#fde725']
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


def process_fcs_files(directory, plots_config, results_directory):
    start_time = time.time()

    # Extract singlet gate thresholds from config (always present after validation)
    singlet_cfg = plots_config.get("singlet_gate", {})
    singlet_lower = singlet_cfg.get("lower", SINGLET_RATIO_LOWER)
    singlet_upper = singlet_cfg.get("upper", SINGLET_RATIO_UPPER)
    logger.info(
        f"Singlet gate thresholds: lower={singlet_lower}, upper={singlet_upper}"
    )

    # Separate plot configs from non-plot keys (e.g. singlet_gate)
    plot_configs = {
        k: v for k, v in plots_config.items() if isinstance(v, dict) and "type" in v
    }

    # Create output structure
    (
        plots_dir,
        well_plots_dir,
        stats_dir,
        triplicate_stats_dir,
        triplicate_plots_dir,
    ) = create_output_structure(results_directory)

    scatter_dfs = {}
    histogram_dfs = {}
    singlet_stats = []

    files = [
        os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".fcs")
    ]
    if not files:
        logger.warning("No FCS files found in the directory.")
        return pd.DataFrame(), 0  # Return empty DataFrame and 0 runtime

    logger.info(f"Found {len(files)} FCS files in {directory}")

    # Sort files using extract_well_key
    files.sort(key=lambda f: extract_well_key(os.path.basename(f))[1])

    # Define the number of rows and columns for the plots
    num_rows = 8
    num_cols = 12

    # Create figure for singlet gate plots
    fig_singlets, axes_singlets = plt.subplots(
        num_rows + 1,
        num_cols + 1,
        figsize=(42, 28),  # Adjusted figure size to 12:8 ratio
        gridspec_kw={
            "height_ratios": [0.5] + [1] * num_rows,
            "width_ratios": [0.5] + [1] * num_cols,
        },
    )
    fig_singlets.suptitle("Singlet Gates", fontsize=24, fontweight="bold", y=0.98)

    # Adjust the spacing between subplots
    fig_singlets.subplots_adjust(hspace=0.1, wspace=0.4)  # Adjusted hspace and wspace

    # Create figures for each plot configuration
    figs = {}
    axes = {}
    for config in plot_configs.values():
        plot_key = f"{config['type']}_{config['x_param']}_{config.get('y_param', '')}"
        fig, ax = plt.subplots(
            num_rows + 1,
            num_cols + 1,
            figsize=(42, 28),  # Adjusted figure size to 12:8 ratio
            gridspec_kw={
                "height_ratios": [0.5] + [1] * num_rows,
                "width_ratios": [0.5] + [1] * num_cols,
            },
        )
        fig.suptitle(
            f"{config['type'].capitalize()} Plot: {config['x_param']} vs {config.get('y_param', '')}",
            fontsize=24,
            fontweight="bold",
            y=0.95,
        )  # Adjusted y to reduce space

        # Adjust the spacing between subplots
        if config["type"] == "scatter":
            fig.subplots_adjust(
                hspace=0.6, wspace=1
            )  # Reduced hspace for scatter plots
        else:
            fig.subplots_adjust(hspace=0.6, wspace=0.5)  # Reduced hspace for histograms

        figs[plot_key] = fig
        axes[plot_key] = ax

    # Add column and row labels for all plots
    for plot_key, fig in figs.items():
        ax = axes[plot_key]
        for col in range(1, num_cols + 1):
            ax[0, col].text(
                0.5,
                0.2,
                str(col),
                ha="center",
                va="center",
                fontweight="bold",
                fontsize=18,
            )  # Increased font size and boldness
            ax[0, col].axis("off")

        for row in range(1, num_rows + 1):
            ax[row, 0].text(
                0.5,
                0.5,
                chr(64 + row),
                ha="center",
                va="center",
                fontweight="bold",
                fontsize=18,
            )  # Increased font size and boldness
            ax[row, 0].axis("off")

        # Remove the top-left empty cell
        ax[0, 0].axis("off")

        # For scatter plots, adjust the position of subplots to bring rows closer
        if "scatter" in plot_key:
            for row in range(
                1, num_rows + 1
            ):  # Start from the first row of actual plots
                for col in range(1, num_cols + 1):
                    pos = ax[row, col].get_position()
                    pos.y0 = pos.y0 + 0.02 * (
                        row - 1
                    )  # Move up by 2% of figure height for each row
                    pos.y1 = pos.y1 + 0.02 * (row - 1)
                    ax[row, col].set_position(pos)
                    ax[row, col].set_aspect(
                        "equal", adjustable="box"
                    )  # Make each subplot square

    # Ensure tight layout to minimize blank space
    for fig in figs.values():
        fig.tight_layout(
            rect=[0, 0, 1, 0.95]
        )  # Adjust rect to leave less space for suptitle

    # Initialize a set to keep track of wells with data
    wells_with_data = set()

    # Calculate total number of iterations
    total_iterations = len(files) * (len(plot_configs) + 1)  # +1 for singlet plots

    # Create a progress bar (disabled in quiet mode)
    console_level = logging.INFO
    for h in logger.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(
            h, logging.FileHandler
        ):
            console_level = h.level
            break
    with tqdm(
        total=total_iterations,
        desc="Processing files",
        unit="plot",
        file=sys.stdout,
        disable=console_level >= logging.WARNING,
    ) as pbar:
        for file in files:
            well_key, (row_letter, col_number) = extract_well_key(file)
            row = (
                ord(row_letter) - 64
            )  # Convert letter to row number (A=1, B=2, ..., H=8)
            col = col_number

            if row > num_rows or col > num_cols:
                logger.warning(
                    "Skipping file %s with well key %s as it is out of the %dx%d grid.",
                    file,
                    well_key,
                    num_rows,
                    num_cols,
                )
                continue

            wells_with_data.add((row, col))

            try:
                data, _ = read_fcs(file)
                if data is None:
                    pbar.update(1)
                    continue

                # Remove doublets
                (
                    singlets,
                    singlet_percentage,
                    total_events,
                    singlet_events,
                ) = remove_doublets(
                    data,
                    singlet_lower=singlet_lower,
                    singlet_upper=singlet_upper,
                )
                logger.info(
                    "File: %s, Singlet percentage: %.2f%%, Total events: %d, Singlet events: %d",
                    file,
                    singlet_percentage,
                    total_events,
                    singlet_events,
                )
                singlet_stats.append(
                    {
                        "Well": well_key,
                        "Singlet_Percentage": singlet_percentage,
                        "Total_Events": total_events,
                        "Singlet_Events": singlet_events,
                    }
                )

                # Plot singlet gate
                ax_singlet = axes_singlets[row, col]
                plot_singlet_gate(
                    data,
                    ax=ax_singlet,
                    file_name=well_key,
                    singlet_lower=singlet_lower,
                    singlet_upper=singlet_upper,
                )

                for config in plot_configs.values():
                    plot_key = f"{config['type']}_{config['x_param']}_{config.get('y_param', '')}"
                    if config["type"] == "scatter":
                        # Process scatter plot
                        ax = axes[plot_key][row, col]
                        gate_stats = plot_scatter_with_manual_gates(
                            singlets,
                            config["x_param"],
                            config["y_param"],
                            well_key,
                            ax,
                            scatter_type=config.get("scatter_type", "scatter"),
                            cmap=config.get("cmap", "viridis"),
                            x_scale=config.get("x_scale", "linear"),
                            y_scale=config.get("y_scale", "linear"),
                            xlim=config.get("xlim"),
                            ylim=config.get("ylim"),
                            gridsize=config.get("gridsize", 100),
                            quadrant_gates=config.get("quadrant_gates"),
                        )

                        if gate_stats is not None:
                            gate_stats["Well"] = well_key
                            new_df = pd.DataFrame([gate_stats])
                            if plot_key not in scatter_dfs:
                                scatter_dfs[plot_key] = []
                            scatter_dfs[plot_key].append(new_df)

                    elif config["type"] == "histogram":
                        # Process histogram plot
                        ax = axes[plot_key][row, col]
                        stats = plot_histogram(
                            singlets,
                            config["x_param"],
                            well_key,
                            ax,
                            x_scale=config.get("x_scale", "linear"),
                            kde=config.get("kde", False),
                            color=config.get("color", "blue"),
                            xlim=config.get("xlim"),
                            gates=config.get("gates"),
                        )

                        if stats is not None:
                            stats["Well"] = well_key
                            new_df = pd.DataFrame([stats])
                            if plot_key not in histogram_dfs:
                                histogram_dfs[plot_key] = []
                            histogram_dfs[plot_key].append(new_df)

                pbar.update(1)  # Update progress bar

            except Exception as e:
                logger.error(f"Error processing file {file}: {e}")
                continue

        # Fill empty wells with "No Data"
        for row in range(1, num_rows + 1):
            for col in range(1, num_cols + 1):
                if (row, col) not in wells_with_data:
                    # Fill singlet gate plot
                    ax_singlet = axes_singlets[row, col]
                    ax_singlet.text(0.5, 0.5, "No Data", ha="center", va="center")
                    ax_singlet.axis("off")

                    # Fill other plots
                    for ax in axes.values():
                        ax[row, col].text(0.5, 0.5, "No Data", ha="center", va="center")
                        ax[row, col].axis("off")

        # Concatenate DataFrames and save statistics to CSV files
        for plot_key, df_list in scatter_dfs.items():
            if df_list:
                df = pd.concat(df_list, ignore_index=True)
                scatter_csv_path = os.path.join(
                    stats_dir,
                    f"{plot_key}_statistics_{os.path.basename(results_directory)}.csv",
                )
                df.to_csv(scatter_csv_path, index=False)
                logger.info(f"Saved {plot_key} statistics to: {scatter_csv_path}")
                scatter_dfs[plot_key] = df  # Replace list with concatenated DataFrame

        for plot_key, df_list in histogram_dfs.items():
            if df_list:
                df = pd.concat(df_list, ignore_index=True)
                histogram_csv_path = os.path.join(
                    stats_dir,
                    f"{plot_key}_statistics_{os.path.basename(results_directory)}.csv",
                )
                df.to_csv(histogram_csv_path, index=False)
                logger.info(f"Saved {plot_key} statistics to: {histogram_csv_path}")
                histogram_dfs[plot_key] = df  # Replace list with concatenated DataFrame

        # Save singlet statistics
        singlet_df = pd.DataFrame(singlet_stats)
        singlet_csv_path = os.path.join(
            stats_dir, f"singlet_statistics_{os.path.basename(results_directory)}.csv"
        )
        singlet_df.to_csv(singlet_csv_path, index=False)
        logger.info(f"Saved singlet statistics to: {singlet_csv_path}")

        # Process triplicate plots based on configuration
        for config in plot_configs.values():
            plot_key = (
                f"{config['type']}_{config['x_param']}_{config.get('y_param', '')}"
            )
            if "triplicate_plots" in config:
                df_to_use = (
                    scatter_dfs[plot_key]
                    if config["type"] == "scatter"
                    else histogram_dfs[plot_key]
                )
                for plot_spec in config["triplicate_plots"]:
                    metric = plot_spec["metric"]
                    title = plot_spec["title"]
                    if metric in df_to_use.columns:
                        triplicate_stats = calculate_triplicate_stats(df_to_use, metric)

                        if not triplicate_stats.empty:
                            # Save triplicate statistics
                            triplicate_stats_filename = (
                                f"{plot_key}_{metric}_triplicate_statistics.csv"
                            )
                            triplicate_stats_path = os.path.join(
                                triplicate_stats_dir, triplicate_stats_filename
                            )
                            triplicate_stats.to_csv(triplicate_stats_path, index=False)
                            logger.info(
                                "Saved triplicate statistics for %s to: %s",
                                metric,
                                triplicate_stats_path,
                            )

                            # Plot triplicate statistics
                            plot_triplicate_stats(
                                triplicate_stats,
                                metric,
                                triplicate_plots_dir,
                                title=f"{plot_key}_{title}",
                            )
                        else:
                            logger.warning(
                                "No complete triplicates found for %s in %s. Skipping plot and statistics.",
                                metric,
                                plot_key,
                            )
                    else:
                        logger.warning(
                            "Metric %s not found in dataframe for %s. Skipping triplicate plot.",
                            metric,
                            plot_key,
                        )

        # Save updated dataframes
        for plot_key, df in scatter_dfs.items():
            csv_filename = f"{plot_key}_statistics_with_triplicates.csv"
            csv_path = os.path.join(stats_dir, csv_filename)
            df.to_csv(csv_path, index=False)
            logger.info(
                "Saved %s statistics with triplicates to: %s", plot_key, csv_path
            )

        for plot_key, df in histogram_dfs.items():
            csv_filename = f"{plot_key}_statistics_with_triplicates.csv"
            csv_path = os.path.join(stats_dir, csv_filename)
            df.to_csv(csv_path, index=False)
            logger.info(
                "Saved %s statistics with triplicates to: %s", plot_key, csv_path
            )

        # Save all main plots
        for plot_key, fig in figs.items():
            plot_filename = f"{plot_key}_plot_{os.path.basename(results_directory)}.png"
            plot_path = os.path.join(plots_dir, plot_filename)
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved {plot_key} plot: {plot_path}")

        # Save singlet gate plot
        singlet_plot_filename = (
            f"singlet_gates_plot_{os.path.basename(results_directory)}.png"
        )
        singlet_plot_path = os.path.join(plots_dir, singlet_plot_filename)
        fig_singlets.savefig(singlet_plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig_singlets)
        logger.info(f"Saved singlet gates plot: {singlet_plot_path}")

        # Generate and save 96-well plots
        for config in plot_configs.values():
            plot_key = (
                f"{config['type']}_{config['x_param']}_{config.get('y_param', '')}"
            )
            if "96well_plots" in config:
                df = (
                    scatter_dfs[plot_key]
                    if config["type"] == "scatter"
                    else histogram_dfs[plot_key]
                )
                for plot_spec in config["96well_plots"]:
                    metric = plot_spec["metric"]
                    title = plot_spec["title"]
                    if metric in df.columns:
                        parameter_name = config["x_param"].split("-")[
                            0
                        ]  # Extract parameter name (e.g., 'BL1' from 'BL1-H')
                        plot_96well_grid(
                            df,
                            metric,
                            title,
                            well_plots_dir,
                            parameter_name=parameter_name,
                        )
                    else:
                        logger.warning(
                            "Metric %s not found in data for %s. Skipping 96-well plot.",
                            metric,
                            plot_key,
                        )

        return scatter_dfs, histogram_dfs, singlet_df, time.time() - start_time


def compile_summary_report(results_directory, plots_config):
    # Separate plot configs from non-plot keys (e.g. singlet_gate)
    plot_configs = {
        k: v for k, v in plots_config.items() if isinstance(v, dict) and "type" in v
    }
    pdf_path = os.path.join(results_directory, "summary_report.pdf")
    plots_dir = os.path.join(results_directory, "plots")
    well_plots_dir = os.path.join(results_directory, "96well_plots")
    triplicate_plots_dir = os.path.join(results_directory, "triplicate_plots")

    with PdfPages(pdf_path) as pdf:
        # Add a title page
        plt.figure(figsize=(11.69, 8.27))
        directory_name = os.path.basename(results_directory)
        current_datetime = datetime.now()
        formatted_date = current_datetime.strftime("%d %B %Y")
        formatted_time = current_datetime.strftime("%H:%M")

        plt.text(
            0.5,
            0.7,
            "Flow Cytometry Analysis Summary Report",
            ha="center",
            va="center",
            fontsize=24,
            fontweight="bold",
        )
        plt.text(
            0.5,
            0.6,
            f"FLtower version: {__version__}",
            ha="center",
            va="center",
            fontsize=18,
        )
        plt.text(
            0.5,
            0.5,
            f"Directory: {directory_name}",
            ha="center",
            va="center",
            fontsize=18,
        )
        plt.text(
            0.5, 0.4, f"Date: {formatted_date}", ha="center", va="center", fontsize=18
        )
        plt.text(
            0.5, 0.3, f"Time: {formatted_time}", ha="center", va="center", fontsize=18
        )
        plt.axis("off")
        pdf.savefig()
        plt.close()

        # Compile main plots (histogram, scatter, singlets)
        for config in plot_configs.values():
            plot_key = (
                f"{config['type']}_{config['x_param']}_{config.get('y_param', '')}"
            )
            plot_filename = f"{plot_key}_plot_{os.path.basename(results_directory)}.png"
            plot_path = os.path.join(plots_dir, plot_filename)
            logger.debug(f"Searching for {plot_key} plot at: {plot_path}")

            if os.path.exists(plot_path):
                logger.debug(f"Found {plot_key} plot")
                img = plt.imread(plot_path)
                fig, ax = plt.subplots(figsize=(11.69, 8.27))
                ax.imshow(img)
                ax.axis("off")
                plt.title(f"{plot_key.capitalize()} Plot", fontsize=16)
                plt.tight_layout()
                pdf.savefig(fig, orientation="landscape", dpi=150)
                plt.close(fig)
            else:
                logger.warning(f"{plot_key} plot not found at {plot_path}")

        # Add each 96-well plot to the PDF
        for config in plot_configs.values():
            if "96well_plots" in config:
                parameter_name = config["x_param"].split("-")[
                    0
                ]  # Extract parameter name
                for plot_spec in config["96well_plots"]:
                    metric = plot_spec["metric"]
                    title = plot_spec["title"]
                    plot_filename = f"{parameter_name}_{metric.replace(' ', '_').lower()}_96well_grid.png"
                    plot_path = os.path.join(well_plots_dir, plot_filename)
                    logger.debug(f"Searching for 96-well plot at: {plot_path}")

                    if os.path.exists(plot_path):
                        logger.debug(f"Adding 96-well plot for {metric} to PDF")
                        img = plt.imread(plot_path)
                        fig, ax = plt.subplots(figsize=(11.69, 8.27))
                        ax.imshow(img)
                        ax.axis("off")
                        # plt.title(f"{title}\n({config['type']} plot)", fontsize=16)
                        plt.tight_layout()
                        pdf.savefig(fig, orientation="landscape")
                        plt.close(fig)
                    else:
                        logger.warning(
                            f"96-well plot for {metric} not found at {plot_path}"
                        )

        # Add triplicate plots to the PDF
        for config in plot_configs.values():
            if "triplicate_plots" in config:
                plot_key = (
                    f"{config['type']}_{config['x_param']}_{config.get('y_param', '')}"
                )
                for plot_spec in config["triplicate_plots"]:
                    metric = plot_spec["metric"]
                    title = plot_spec["title"]
                    plot_filename = f"{plot_key}_{title.replace(' ', '_').lower()}_triplicate_stats.png"
                    plot_path = os.path.join(triplicate_plots_dir, plot_filename)
                    logger.debug(f"Searching for triplicate plot at: {plot_path}")

                    if os.path.exists(plot_path):
                        logger.debug(f"Adding triplicate plot for {metric} to PDF")
                        img = plt.imread(plot_path)
                        fig, ax = plt.subplots(figsize=(11.69, 8.27))
                        ax.imshow(img)
                        ax.axis("off")
                        plt.title(f"{title}\n(Triplicate plot)", fontsize=16)
                        plt.tight_layout()
                        pdf.savefig(fig, orientation="landscape")
                        plt.close(fig)
                    else:
                        logger.warning(
                            f"Triplicate plot for {metric} not found at {plot_path}"
                        )

    logger.info(f"Summary report saved to: {pdf_path}")


def main(command_line_arguments=None):
    """Main function of FLtower

    Parameters
    ----------
    command_line_arguments : List[str], optional
        Used to give inputs for the runtime when you call this function like a module.
        For example, to test the FLtower run from tests folder.
        By default None.
    """
    start_time = time.time()
    run_args = parse_run_args(command_line_arguments)

    # Setup logging (before any log call)
    setup_logging(
        verbose=getattr(run_args, "verbose", False),
        quiet=getattr(run_args, "quiet", False),
    )

    logger.info(f"FLtower version: {__version__}")

    try:
        input_folder = run_args.input
        output_folder = run_args.output
        plots_config = load_parameters(input_folder, run_args.parameters)

        # Create results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_directory = os.path.join(output_folder, f"results_{timestamp}")
        os.makedirs(results_directory, exist_ok=True)
        logger.info(f"Created results directory: {results_directory}")

        # Add file handler now that results directory exists
        log_file = os.path.join(results_directory, "fltower.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
            )
        )
        logger.addHandler(file_handler)
        logger.debug(f"Log file: {log_file}")

        used_params = save_parameters(
            plots_config, results_directory, "used_parameters.json"
        )
        logger.info(f"Save used parameters: {used_params}")

        scatter_dfs, histogram_dfs, singlet_df, runtime = process_fcs_files(
            input_folder, plots_config, results_directory
        )

        logger.info(f"All results saved in: {results_directory}")
        logger.debug("Scatter Plot Statistics:")
        for plot_key, df in scatter_dfs.items():
            logger.debug(f"{plot_key} Statistics:\n{df}")
        logger.debug("Histogram Plot Statistics:")
        for plot_key, df in histogram_dfs.items():
            logger.debug(f"{plot_key} Statistics:\n{df}")
        logger.debug(f"Singlet Statistics:\n{singlet_df}")

        # Compile summary report
        compile_summary_report(results_directory, plots_config)

    except FileNotFoundError as e:
        logger.error(f"{e}")
        logger.error(
            "Please check if the specified directory exists and you have the necessary permissions."
        )
    except ValueError as e:
        logger.error(f"{e}")
        logger.error("Please ensure that the base directory contains valid FCS files.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        end_time = time.time()
        total_runtime = end_time - start_time
        logger.info(f"Total script runtime: {total_runtime:.2f} seconds")


if __name__ == "__main__":
    main()
