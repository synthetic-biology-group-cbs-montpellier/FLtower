"""Pipeline orchestrator — composable steps for FCS processing.

Replaces the monolithic ``process_fcs_files()`` with 10 named steps
that share state through a :class:`PipelineContext` dataclass.
Each step is independently testable.

Typical usage::

    result = run_pipeline(directory, plots_config, results_directory)
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from fltower.core.gating.auto.resolve import (
    resolve_histogram_gates,
    resolve_quadrant_gates,
)
from fltower.core.gating.hierarchy import apply_gating_hierarchy, get_gating_summary
from fltower.core.statistics import calculate_triplicate_stats
from fltower.io.export import (
    save_singlet_stats_csv,
    save_statistics_csv,
    save_stats_with_triplicates_csv,
    save_triplicate_stats_csv,
)
from fltower.io.fcs_reader import read_fcs
from fltower.plotting.histogram import plot_histogram
from fltower.plotting.plate_view import plot_96well_grid
from fltower.plotting.scatter import plot_scatter_with_manual_gates
from fltower.plotting.singlet import plot_singlet_gate
from fltower.plotting.triplicate import plot_triplicate_stats

logger = logging.getLogger("fltower")

NUM_ROWS = 8
NUM_COLS = 12


# ── Utility functions ────────────────────────────────────────────────


def extract_well_key(filename):
    """Extract well identifier (e.g. ``'A1'``) from an FCS filename."""
    base_name = os.path.splitext(os.path.basename(filename))[0]
    match = re.search(r"([A-H])([0-9]{1,2})(?!.*[A-H][0-9]{1,2})", base_name)
    if match:
        letter_part = match.group(1)
        number_part = int(match.group(2))
        return match.group(0), (letter_part, number_part)
    else:
        return base_name, (base_name, 0)


def create_output_structure(results_directory):
    """Create subdirectories for the different output types.

    Returns a dict with keys ``plots``, ``well_plots``, ``stats``,
    ``triplicate_stats``, ``triplicate_plots``.
    """
    dirs = {
        "plots": os.path.join(results_directory, "plots"),
        "well_plots": os.path.join(results_directory, "96well_plots"),
        "stats": os.path.join(results_directory, "statistics"),
        "triplicate_stats": os.path.join(results_directory, "triplicate_statistics"),
        "triplicate_plots": os.path.join(results_directory, "triplicate_plots"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def make_plot_key(config):
    """Build the canonical plot key from a plot config dict."""
    return f"{config['type']}_{config['x_param']}_{config.get('y_param', '')}"


# ── Pipeline context ─────────────────────────────────────────────────


@dataclass
class PipelineContext:
    """Mutable state bag passed through every pipeline step."""

    # Inputs
    directory: str
    plots_config: dict
    results_directory: str

    # Step 1 – discover
    files: list = field(default_factory=list)

    # Step 2 – prepare config
    plot_configs: dict = field(default_factory=dict)
    singlet_lower: float = 0.7
    singlet_upper: float = 2.0

    # Step 3 – output dirs
    output_dirs: dict = field(default_factory=dict)

    # Step 4 – figures (plot_key → fig / axes)
    figs: dict = field(default_factory=dict)
    axes: dict = field(default_factory=dict)
    fig_singlets: object = None
    axes_singlets: object = None

    # Step 5 – per-well accumulators
    scatter_dfs: dict = field(default_factory=dict)
    histogram_dfs: dict = field(default_factory=dict)
    singlet_stats: list = field(default_factory=list)
    gating_roots: list = field(default_factory=list)
    wells_with_data: set = field(default_factory=set)


# ── Step 1: Discover FCS files ───────────────────────────────────────


def step_discover_files(ctx: PipelineContext) -> None:
    """Find and sort FCS files in the input directory."""
    ctx.files = sorted(
        (
            os.path.join(ctx.directory, f)
            for f in os.listdir(ctx.directory)
            if f.endswith(".fcs")
        ),
        key=lambda f: extract_well_key(os.path.basename(f))[1],
    )
    if ctx.files:
        logger.info("Found %d FCS files in %s", len(ctx.files), ctx.directory)
    else:
        logger.warning("No FCS files found in the directory.")


# ── Step 2: Prepare configuration ────────────────────────────────────


def step_prepare_config(ctx: PipelineContext) -> None:
    """Extract plot configs and singlet-gate thresholds from raw config."""
    singlet_cfg = ctx.plots_config.get("singlet_gate", {})
    ctx.singlet_lower = singlet_cfg.get("lower", 0.7)
    ctx.singlet_upper = singlet_cfg.get("upper", 2.0)
    ctx.plot_configs = {
        k: v for k, v in ctx.plots_config.items() if isinstance(v, dict) and "type" in v
    }


# ── Step 3: Create output directories ────────────────────────────────


def step_create_output(ctx: PipelineContext) -> None:
    """Create the output directory tree."""
    ctx.output_dirs = create_output_structure(ctx.results_directory)


# ── Step 4: Create matplotlib figures ─────────────────────────────────


def _create_plate_grid():
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


def _add_plate_labels(ax):
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


def _adjust_scatter_positions(ax):
    """Nudge scatter subplot rows upward and enforce square aspect."""
    for row in range(1, NUM_ROWS + 1):
        for col in range(1, NUM_COLS + 1):
            pos = ax[row, col].get_position()
            pos.y0 += 0.02 * (row - 1)
            pos.y1 += 0.02 * (row - 1)
            ax[row, col].set_position(pos)
            ax[row, col].set_aspect("equal", adjustable="box")


def step_create_figures(ctx: PipelineContext) -> None:
    """Create all plate figure grids (singlet + one per plot config)."""
    # Singlet figure
    fig_s, ax_s = _create_plate_grid()
    fig_s.suptitle("Singlet Gates", fontsize=24, fontweight="bold", y=0.98)
    fig_s.subplots_adjust(hspace=0.1, wspace=0.4)
    ctx.fig_singlets = fig_s
    ctx.axes_singlets = ax_s

    # Per-config figures
    for config in ctx.plot_configs.values():
        plot_key = make_plot_key(config)
        title = (
            f"{config['type'].capitalize()} Plot: "
            f"{config['x_param']} vs {config.get('y_param', '')}"
        )
        fig, ax = _create_plate_grid()
        fig.suptitle(title, fontsize=24, fontweight="bold", y=0.95)

        if config["type"] == "scatter":
            fig.subplots_adjust(hspace=0.6, wspace=1)
        else:
            fig.subplots_adjust(hspace=0.6, wspace=0.5)

        ctx.figs[plot_key] = fig
        ctx.axes[plot_key] = ax

    # Labels, scatter adjustments, tight layout
    for plot_key in ctx.figs:
        _add_plate_labels(ctx.axes[plot_key])
        if "scatter" in plot_key:
            _adjust_scatter_positions(ctx.axes[plot_key])

    for fig in ctx.figs.values():
        fig.tight_layout(rect=[0, 0, 1, 0.95])


# ── Step 5: Process all wells ─────────────────────────────────────────


def _process_single_well(file, ctx):
    """Read one FCS file, apply gating, plot, and collect statistics.

    Returns ``True`` on success, ``False`` if the file was skipped.
    """
    well_key, (row_letter, col_number) = extract_well_key(file)
    row = ord(row_letter) - 64
    col = col_number

    if row > NUM_ROWS or col > NUM_COLS:
        logger.warning(
            "Skipping file %s with well key %s as it is out of the %dx%d grid.",
            file,
            well_key,
            NUM_ROWS,
            NUM_COLS,
        )
        return False

    ctx.wells_with_data.add((row, col))

    data, _ = read_fcs(file)
    if data is None:
        return False

    # Gating hierarchy
    gating_root = apply_gating_hierarchy(data, ctx.plots_config)
    ctx.gating_roots.append((well_key, gating_root))
    singlet_node = gating_root.children[0]
    singlets = singlet_node.data
    logger.info(
        "File: %s, Singlet percentage: %.2f%%, Total events: %d, Singlet events: %d",
        file,
        singlet_node.percentage,
        singlet_node.parent_events,
        singlet_node.gated_events,
    )
    ctx.singlet_stats.append({"Well": well_key, **singlet_node.stats})

    # Singlet gate plot
    plot_singlet_gate(
        data,
        ax=ctx.axes_singlets[row, col],
        file_name=well_key,
        singlet_lower=ctx.singlet_lower,
        singlet_upper=ctx.singlet_upper,
    )

    # Per-config plots
    for config in ctx.plot_configs.values():
        plot_key = make_plot_key(config)
        if config["type"] == "scatter":
            resolved_qgates = resolve_quadrant_gates(singlets, config)
            ax = ctx.axes[plot_key][row, col]
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
                quadrant_gates=resolved_qgates,
            )
            if gate_stats is not None:
                gate_stats["Well"] = well_key
                ctx.scatter_dfs.setdefault(plot_key, []).append(
                    pd.DataFrame([gate_stats])
                )

        elif config["type"] == "histogram":
            resolved_hgates = resolve_histogram_gates(singlets, config)
            ax = ctx.axes[plot_key][row, col]
            stats = plot_histogram(
                singlets,
                config["x_param"],
                well_key,
                ax,
                x_scale=config.get("x_scale", "linear"),
                kde=config.get("kde", False),
                color=config.get("color", "blue"),
                xlim=config.get("xlim"),
                gates=resolved_hgates,
            )
            if stats is not None:
                stats["Well"] = well_key
                ctx.histogram_dfs.setdefault(plot_key, []).append(pd.DataFrame([stats]))

    return True


def step_process_wells(ctx: PipelineContext) -> None:
    """Iterate over all FCS files, processing each well."""
    console_level = logging.INFO
    for h in logger.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(
            h, logging.FileHandler
        ):
            console_level = h.level
            break

    total = len(ctx.files) * (len(ctx.plot_configs) + 1)
    with tqdm(
        total=total,
        desc="Processing files",
        unit="plot",
        file=sys.stdout,
        disable=console_level >= logging.WARNING,
    ) as pbar:
        for file in ctx.files:
            try:
                _process_single_well(file, ctx)
            except Exception as e:
                logger.error(f"Error processing file {file}: {e}")
            pbar.update(1)


# ── Step 6: Fill empty wells ──────────────────────────────────────────


def step_fill_empty_wells(ctx: PipelineContext) -> None:
    """Mark wells without data as 'No Data'."""
    for row in range(1, NUM_ROWS + 1):
        for col in range(1, NUM_COLS + 1):
            if (row, col) not in ctx.wells_with_data:
                ax_s = ctx.axes_singlets[row, col]
                ax_s.text(0.5, 0.5, "No Data", ha="center", va="center")
                ax_s.axis("off")

                for axes_arr in ctx.axes.values():
                    axes_arr[row, col].text(
                        0.5, 0.5, "No Data", ha="center", va="center"
                    )
                    axes_arr[row, col].axis("off")


# ── Step 7: Save statistics ───────────────────────────────────────────


def step_save_statistics(ctx: PipelineContext) -> None:
    """Concatenate per-well statistics DataFrames and write CSVs."""
    stats_dir = ctx.output_dirs["stats"]
    results_name = os.path.basename(ctx.results_directory)

    for plot_key, df_list in ctx.scatter_dfs.items():
        df = save_statistics_csv(df_list, plot_key, stats_dir, results_name)
        if df is not None:
            ctx.scatter_dfs[plot_key] = df

    for plot_key, df_list in ctx.histogram_dfs.items():
        df = save_statistics_csv(df_list, plot_key, stats_dir, results_name)
        if df is not None:
            ctx.histogram_dfs[plot_key] = df

    save_singlet_stats_csv(ctx.singlet_stats, stats_dir, results_name)


# ── Step 8: Compute triplicates ───────────────────────────────────────


def step_compute_triplicates(ctx: PipelineContext) -> None:
    """Calculate triplicate statistics and generate triplicate plots."""
    triplicate_stats_dir = ctx.output_dirs["triplicate_stats"]
    triplicate_plots_dir = ctx.output_dirs["triplicate_plots"]

    for config in ctx.plot_configs.values():
        plot_key = make_plot_key(config)
        if "triplicate_plots" not in config:
            continue

        df = (
            ctx.scatter_dfs[plot_key]
            if config["type"] == "scatter"
            else ctx.histogram_dfs[plot_key]
        )

        for plot_spec in config["triplicate_plots"]:
            metric = plot_spec["metric"]
            title = plot_spec["title"]

            if metric not in df.columns:
                logger.warning(
                    "Metric %s not found in dataframe for %s. Skipping triplicate plot.",
                    metric,
                    plot_key,
                )
                continue

            trip_stats = calculate_triplicate_stats(df, metric)
            if trip_stats.empty:
                logger.warning(
                    "No complete triplicates found for %s in %s. Skipping plot and statistics.",
                    metric,
                    plot_key,
                )
                continue

            save_triplicate_stats_csv(
                trip_stats,
                plot_key,
                metric,
                triplicate_stats_dir,
            )
            plot_triplicate_stats(
                trip_stats,
                metric,
                triplicate_plots_dir,
                title=f"{plot_key}_{title}",
            )

    # Save statistics with triplicate columns
    stats_dir = ctx.output_dirs["stats"]
    for plot_key, df in ctx.scatter_dfs.items():
        save_stats_with_triplicates_csv(df, plot_key, stats_dir)
    for plot_key, df in ctx.histogram_dfs.items():
        save_stats_with_triplicates_csv(df, plot_key, stats_dir)


# ── Step 9: Save figures ──────────────────────────────────────────────


def step_save_figures(ctx: PipelineContext) -> None:
    """Save all plate figure PNGs, 96-well heatmaps, and close figures."""
    plots_dir = ctx.output_dirs["plots"]
    well_plots_dir = ctx.output_dirs["well_plots"]
    results_name = os.path.basename(ctx.results_directory)

    # Main plots
    for plot_key, fig in ctx.figs.items():
        plot_path = os.path.join(plots_dir, f"{plot_key}_plot_{results_name}.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {plot_key} plot: {plot_path}")

    # Singlet gate plot
    singlet_path = os.path.join(plots_dir, f"singlet_gates_plot_{results_name}.png")
    ctx.fig_singlets.savefig(singlet_path, dpi=150, bbox_inches="tight")
    plt.close(ctx.fig_singlets)
    logger.info(f"Saved singlet gates plot: {singlet_path}")

    # 96-well heatmap plots
    for config in ctx.plot_configs.values():
        plot_key = make_plot_key(config)
        if "96well_plots" not in config:
            continue

        df = (
            ctx.scatter_dfs[plot_key]
            if config["type"] == "scatter"
            else ctx.histogram_dfs[plot_key]
        )
        for plot_spec in config["96well_plots"]:
            metric = plot_spec["metric"]
            title = plot_spec["title"]
            if metric not in df.columns:
                logger.warning(
                    "Metric %s not found in data for %s. Skipping 96-well plot.",
                    metric,
                    plot_key,
                )
                continue
            parameter_name = config["x_param"].split("-")[0]
            plot_96well_grid(
                df,
                metric,
                title,
                well_plots_dir,
                parameter_name=parameter_name,
            )


# ── Step 10: Log gating summary ──────────────────────────────────────


def step_log_gating_summary(ctx: PipelineContext) -> None:
    """Log gating hierarchy summary across all wells."""
    if not ctx.gating_roots:
        return

    rows = []
    for well_key, root in ctx.gating_roots:
        for entry in get_gating_summary(root):
            entry["Well"] = well_key
            rows.append(entry)

    df = pd.DataFrame(rows)
    agg = (
        df.groupby("Gate")
        .agg(
            Wells=("Well", "count"),
            Total_Parent=("Parent_Events", "sum"),
            Total_Gated=("Gated_Events", "sum"),
            Mean_Pct=("Percentage", "mean"),
        )
        .reset_index()
    )
    logger.info("Gating hierarchy summary:\n%s", agg.to_string(index=False))


# ── Orchestrator ──────────────────────────────────────────────────────


def run_pipeline(directory, plots_config, results_directory):
    """Execute the full FCS processing pipeline.

    Returns ``(scatter_dfs, histogram_dfs, singlet_stats, runtime)``.
    """
    start_time = time.time()

    ctx = PipelineContext(
        directory=directory,
        plots_config=plots_config,
        results_directory=results_directory,
    )

    step_discover_files(ctx)  # 1
    if not ctx.files:
        return {}, {}, [], time.time() - start_time

    step_prepare_config(ctx)  # 2
    step_create_output(ctx)  # 3
    step_create_figures(ctx)  # 4
    step_process_wells(ctx)  # 5
    step_fill_empty_wells(ctx)  # 6
    step_save_statistics(ctx)  # 7
    step_compute_triplicates(ctx)  # 8
    step_save_figures(ctx)  # 9
    step_log_gating_summary(ctx)  # 10

    return (
        ctx.scatter_dfs,
        ctx.histogram_dfs,
        ctx.singlet_stats,
        time.time() - start_time,
    )
