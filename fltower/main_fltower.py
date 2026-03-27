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
import pandas as pd
from tqdm import tqdm

from fltower.__version__ import __version__
from fltower.core.gating.singlet import (
    SINGLET_RATIO_LOWER,
    SINGLET_RATIO_UPPER,
    remove_doublets,
)
from fltower.core.statistics import calculate_triplicate_stats
from fltower.data_manager import load_parameters, save_parameters
from fltower.io.export import (
    save_singlet_stats_csv,
    save_statistics_csv,
    save_stats_with_triplicates_csv,
    save_triplicate_stats_csv,
)
from fltower.io.fcs_reader import read_fcs
from fltower.plotting.histogram import plot_histogram
from fltower.plotting.plate_view import plot_96well_grid
from fltower.plotting.report import compile_summary_report
from fltower.plotting.scatter import plot_scatter_with_manual_gates
from fltower.plotting.singlet import plot_singlet_gate
from fltower.plotting.triplicate import plot_triplicate_stats
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
        results_name = os.path.basename(results_directory)
        for plot_key, df_list in scatter_dfs.items():
            df = save_statistics_csv(df_list, plot_key, stats_dir, results_name)
            if df is not None:
                scatter_dfs[plot_key] = df

        for plot_key, df_list in histogram_dfs.items():
            df = save_statistics_csv(df_list, plot_key, stats_dir, results_name)
            if df is not None:
                histogram_dfs[plot_key] = df

        # Save singlet statistics
        save_singlet_stats_csv(singlet_stats, stats_dir, results_name)

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
                            save_triplicate_stats_csv(
                                triplicate_stats,
                                plot_key,
                                metric,
                                triplicate_stats_dir,
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
            save_stats_with_triplicates_csv(df, plot_key, stats_dir)

        for plot_key, df in histogram_dfs.items():
            save_stats_with_triplicates_csv(df, plot_key, stats_dir)

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

        return scatter_dfs, histogram_dfs, singlet_stats, time.time() - start_time


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
        logger.debug(f"Singlet Statistics:\n{pd.DataFrame(singlet_df)}")

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
