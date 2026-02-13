#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FLtower - HISTOGRAM ONLY main file.

Keeps:
- FCS reading
- Singlet (doublet removal) logic
- Histogram plots + histogram statistics CSV
- Singlet statistics CSV

Removes:
- Scatter plots
- Singlet gate plots
- 96-well grid plots (original)
- Triplicate plots
- PDF summary report
"""

import io
import os
import re
import sys
import time
import traceback
import warnings
from datetime import datetime

import fcsparser
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Wedge
from scipy.stats import gmean

from fltower.__version__ import __version__
from fltower.data_manager import load_parameters, save_parameters
from fltower.run_args import parse_run_args

# Suppress specific FutureWarnings from seaborn related to pandas deprecation
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

# Suppress Intel MKL warnings
os.environ["MKL_DISABLE_FAST_MM"] = "1"
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*Intel MKL.*")


def plot_shift_plate_circles_tri(
    stats_df,
    metrics,
    max_abs_log,
    max_abs_pct,
    output_dir=".",
    title="Combined shifts (Median / GM / GFP+)",
    cmap_name="coolwarm_r",
):
    """
    96-well circular plate, each well split into 3 sectors:

    - Sector 1: Shift_log10_Median
    - Sector 2: Shift_log10_GM
    - Sector 3: Delta_GFPposPct_Gate2

    A single shared colormap is used, centered at 0.
    Left colorbar axis = Δlog10 shifts
    Right colorbar axis = Δ% GFP+ cells
    """

    os.makedirs(output_dir, exist_ok=True)

    rows = ["A", "B", "C", "D", "E", "F", "G", "H"]
    cols = range(1, 13)

    df = stats_df.copy()
    if "Well" not in df.columns:
        print("Error: Well column missing.")
        return

    df["Row"] = df["Well"].str[0]
    df["Column"] = df["Well"].str[1:].astype(int)

    cmap = getattr(plt.cm, cmap_name)

    fig, ax = plt.subplots(figsize=(16, 9))

    # 3 sectors (120° each)
    angles = [(0, 120), (120, 240), (240, 360)]

    # --- Plot wells ---
    for row_idx, row in enumerate(rows):
        for col in cols:

            x = col - 1
            y = 7 - row_idx

            subset = df[(df["Row"] == row) & (df["Column"] == col)]

            # Missing well
            if subset.empty:
                ax.add_artist(plt.Circle((x, y), 0.4, fill=False, color="gray"))
                ax.text(x, y, "NA", ha="center", va="center", fontsize=7, color="gray")
                continue

            # Draw 3 wedges
            for metric, (a1, a2) in zip(metrics, angles):

                value = subset.iloc[0].get(metric, np.nan)

                if not np.isfinite(value):
                    face = "lightgray"

                else:
                    # --- Normalisation centrée sur 0 ---
                    if metric in ["Shift_log10_Median", "Shift_log10_GM"]:
                        t = value / max_abs_log

                    elif metric == "Delta_GFPposPct_Gate2":
                        t = value / max_abs_pct

                    else:
                        t = 0

                    # Clamp between [-1, +1]
                    t = np.clip(t, -1, 1)

                    # Convert [-1,+1] → [0,1] for colormap
                    face = cmap((t + 1) / 2)

                wedge = Wedge(
                    (x, y),
                    r=0.4,
                    theta1=a1,
                    theta2=a2,
                    facecolor=face,
                    edgecolor="black",
                    linewidth=0.4,
                )
                ax.add_patch(wedge)

    # --- Layout ---
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-0.5, 7.5)
    ax.axis("off")

    # Row labels
    for i, rlab in enumerate(rows):
        ax.text(
            -0.8,
            7 - i,
            rlab,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

    # Column labels
    for i in cols:
        ax.text(
            i - 1,
            7.6,
            str(i),
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

    plt.title(title, fontsize=16, pad=20)

    # ==========================================================
    # SINGLE COLORBAR WITH DOUBLE SCALE
    # ==========================================================

    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(
        sm,
        ax=ax,
        orientation="vertical",
        fraction=0.03,
        aspect=60,
        pad=0.08,
        shrink=0.8,
    )

    ticks = [-1, -0.5, 0, 0.5, 1]
    cbar.set_ticks(ticks)

    # LEFT side → Δlog10
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.yaxis.set_label_position("left")
    left_labels = [f"{t*max_abs_log:.2f}" for t in ticks]
    cbar.ax.set_yticklabels(left_labels)
    cbar.ax.set_ylabel("Δlog10 (Median / GM)", fontsize=11)

    # RIGHT side → Δ% GFP+
    cbar_ax2 = cbar.ax.twinx()
    cbar_ax2.set_ylim(cbar.ax.get_ylim())
    cbar_ax2.set_yticks(ticks)
    right_labels = [f"{t*max_abs_pct:.0f}" for t in ticks]
    cbar_ax2.set_yticklabels(right_labels)
    cbar_ax2.set_ylabel("Δ% GFP+ (Gate 2)", fontsize=11)

    # ==========================================================

    # ==========================================================
    # MINI LEGEND CIRCLE ABOVE COLORBAR
    # ==========================================================

    legend_ax = cbar.ax.inset_axes([0.0, 1.05, 1.0, 0.25])
    legend_ax.set_aspect("equal")
    legend_ax.axis("off")

    cx, cy = 0.5, 0.5
    r = 0.45

    legend_angles = [(0, 120), (120, 240), (240, 360)]

    for a1, a2 in legend_angles:
        wedge = Wedge(
            (cx, cy),
            r,
            theta1=a1,
            theta2=a2,
            facecolor="lightgray",
            edgecolor="black",
            linewidth=0.8,
        )
        legend_ax.add_patch(wedge)

    legend_ax.text(0.5, 1.15, "Median", ha="center", fontsize=9)
    legend_ax.text(-0.2, 0.15, "GM", ha="center", fontsize=9)
    legend_ax.text(1.2, 0.15, "GFP+%", ha="center", fontsize=9)

    # ==========================================================

    # Save
    plt.tight_layout()

    outpath = os.path.join(output_dir, "combined_shift_circular_plate.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved combined circular plate: {outpath}")


def plot_shift_plate_circles(
    stats_df,
    metric="Shift_log10_Median",
    output_dir=".",
    title="Δlog10(Median GFP)",
    vmin=None,
    vmax=None,
):
    """
    Circular 96-well plate plot inspired from original plot_96well_grid().

    stats_df must contain:
      - Well column (A1..H12)
      - metric column (Shift_log10_Median)

    Negative values = GFP loss
    Positive values = GFP gain
    """

    print(f"Plotting circular 96-well plate for metric: {metric}")

    os.makedirs(output_dir, exist_ok=True)

    # Plate layout
    rows = ["A", "B", "C", "D", "E", "F", "G", "H"]
    cols = range(1, 13)

    # Ensure Well exists
    if "Well" not in stats_df.columns:
        print("Error: Well column missing.")
        return

    # Extract row/col
    df = stats_df.copy()
    df["Row"] = df["Well"].str[0]
    df["Column"] = df["Well"].str[1:].astype(int)

    # Valid values
    valid_values = df[metric][np.isfinite(df[metric])]
    if len(valid_values) == 0:
        print(f"No valid values found for {metric}")
        return

    # Auto scale if not provided
    if vmin is None or vmax is None:
        max_abs = np.max(np.abs(valid_values))
        vmin = -max_abs
        vmax = max_abs

    # Figure
    fig, ax = plt.subplots(figsize=(16, 9))

    # Build lookup dictionary
    data_dict = {
        (r, c): df[(df["Row"] == r) & (df["Column"] == c)][metric].values[0]
        for r in rows
        for c in cols
        if not df[(df["Row"] == r) & (df["Column"] == c)].empty
    }

    # Plot wells as circles
    for row_idx, row in enumerate(rows):
        for col in cols:

            value = data_dict.get((row, col), np.nan)

            if np.isfinite(value):
                # Normalize color around center=0
                color_val = (value - vmin) / (vmax - vmin)

                circle = plt.Circle(
                    (col - 1, 7 - row_idx),
                    0.4,
                    fill=True,
                    color=plt.cm.coolwarm_r(color_val),
                )
                ax.add_artist(circle)

                # Text color depending brightness
                bg = plt.cm.coolwarm(color_val)
                text_color = "white" if bg[0] < 0.4 else "black"

                ax.text(
                    col - 1,
                    7 - row_idx,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color=text_color,
                )

            else:
                # Missing well
                circle = plt.Circle(
                    (col - 1, 7 - row_idx),
                    0.4,
                    fill=False,
                    color="gray",
                )
                ax.add_artist(circle)

                ax.text(
                    col - 1,
                    7 - row_idx,
                    "NA",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="gray",
                )

    # Formatting
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-0.5, 7.5)
    ax.axis("off")

    # Row labels
    for i, row in enumerate(rows):
        ax.text(
            -0.8,
            7 - i,
            row,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

    # Column labels
    for i in cols:
        ax.text(
            i - 1,
            7.6,
            str(i),
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

    # Title
    plt.title(title, fontsize=16, pad=20)

    # Colorbar centered on 0
    sm = plt.cm.ScalarMappable(
        cmap="coolwarm_r",
        norm=plt.Normalize(vmin=vmin, vmax=vmax),
    )
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", aspect=30, pad=0.08)
    cbar.set_label(metric, fontsize=12)

    # Save
    outpath = os.path.join(output_dir, f"{metric}_circular_plate.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved circular plate shift plot: {outpath}")


def compare_experiments_kde_grid(
    exp1_dir,
    exp2_dir,
    results_directory,
    plots_config,
    x_param="BL1-H",
    exp1_label="EXP1",
    exp2_label="EXP2",
):
    """
    Build 96-well grid, overlay KDE lines for exp1 and exp2 per well,
    and save stats (exp1, exp2, deltas).
    """

    plots_dir, stats_dir = create_output_structure(results_directory)

    cfg = None
    for v in plots_config.values():
        if v.get("type") == "histogram" and v.get("x_param") == x_param:
            cfg = v
            break
    if cfg is None:
        raise ValueError(f"No histogram config found for {x_param} in parameters.")

    x_scale = cfg.get("x_scale", "linear")
    xlim = cfg.get("xlim")
    gates = cfg.get("gates")

    def list_fcs(d):
        files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".fcs")]
        files.sort(key=lambda f: extract_well_key(os.path.basename(f))[1])
        return files

    exp1_files = list_fcs(exp1_dir)
    exp2_files = list_fcs(exp2_dir)

    def index_by_well(files):
        m = {}
        for f in files:
            well, _ = extract_well_key(f)
            m[well] = f
        return m

    m1 = index_by_well(exp1_files)
    m2 = index_by_well(exp2_files)

    common_wells = sorted(
        set(m1.keys()) & set(m2.keys()), key=lambda w: extract_well_key(w)[1]
    )
    if not common_wells:
        raise ValueError("No common wells found between EXP1 and EXP2.")

    num_rows, num_cols = 8, 12
    fig, ax = plt.subplots(
        num_rows + 1,
        num_cols + 1,
        figsize=(42, 28),
        gridspec_kw={
            "height_ratios": [0.5] + [1] * num_rows,
            "width_ratios": [0.5] + [1] * num_cols,
        },
    )
    fig.suptitle(
        f"KDE comparison {x_param}: {exp1_label} vs {exp2_label}",
        fontsize=24,
        fontweight="bold",
        y=0.95,
    )
    fig.subplots_adjust(hspace=0.6, wspace=0.5)

    for c in range(1, num_cols + 1):
        ax[0, c].text(
            0.5, 0.2, str(c), ha="center", va="center", fontweight="bold", fontsize=18
        )
        ax[0, c].axis("off")
    for r in range(1, num_rows + 1):
        ax[r, 0].text(
            0.5,
            0.5,
            chr(64 + r),
            ha="center",
            va="center",
            fontweight="bold",
            fontsize=18,
        )
        ax[r, 0].axis("off")
    ax[0, 0].axis("off")

    # Stats
    rows = []
    singlet_rows = []

    wells_with_data = set()

    for well in common_wells:
        row_letter = well[0]
        col_number = int(well[1:])
        r = ord(row_letter) - 64
        c = col_number
        if not (1 <= r <= num_rows and 1 <= c <= num_cols):
            continue

        wells_with_data.add((r, c))
        cell = ax[r, c]

        # ---------- EXP1 ----------
        data1, _ = read_fcs(m1[well])
        if data1 is None:
            continue
        sing1, pct1, tot1, n1 = remove_doublets(data1)
        s1 = plot_kde_line(
            sing1,
            x_param,
            well,
            cell,
            x_scale=x_scale,
            color="seagreen",
            xlim=xlim,
            gates=gates,
            label=exp1_label,
        )

        # ---------- EXP2 ----------
        data2, _ = read_fcs(m2[well])
        if data2 is None:
            continue
        sing2, pct2, tot2, n2 = remove_doublets(data2)
        s2 = plot_kde_line(
            sing2,
            x_param,
            well,
            cell,
            x_scale=x_scale,
            color="coral",
            xlim=xlim,
            gates=gates,
            label=exp2_label,
        )

        singlet_rows.append(
            {
                "Experiment": exp1_label,
                "Well": well,
                "Singlet_Percentage": pct1,
                "Total_Events": tot1,
                "Singlet_Events": n1,
            }
        )
        singlet_rows.append(
            {
                "Experiment": exp2_label,
                "Well": well,
                "Singlet_Percentage": pct2,
                "Total_Events": tot2,
                "Singlet_Events": n2,
            }
        )

        # stats BL1-H + delta
        if s1 and s2:
            row = {
                "Well": well,
                f"{exp1_label}_Global_GM": s1.get("Global_GM", np.nan),
                f"{exp1_label}_Global_Median": s1.get("Global_Median", np.nan),
                f"{exp1_label}_Num_Events": s1.get("Num_Events", np.nan),
                f"{exp2_label}_Global_GM": s2.get("Global_GM", np.nan),
                f"{exp2_label}_Global_Median": s2.get("Global_Median", np.nan),
                f"{exp2_label}_Num_Events": s2.get("Num_Events", np.nan),
            }

            for k, v in s1.items():
                if k.startswith("Gate_"):
                    row[f"{exp1_label}_{k}"] = v
            for k, v in s2.items():
                if k.startswith("Gate_"):
                    row[f"{exp2_label}_{k}"] = v

            # --- Extract medians and GM safely ---
            med1 = row[f"{exp1_label}_Global_Median"]
            med2 = row[f"{exp2_label}_Global_Median"]

            gm1 = row[f"{exp1_label}_Global_GM"]
            gm2 = row[f"{exp2_label}_Global_GM"]

            # --- Raw deltas ---
            row["Delta_Median"] = med2 - med1
            row["Delta_GM"] = gm2 - gm1

            # --- Log shifts ---
            row["Shift_log10_Median"] = (
                np.log10(med2) - np.log10(med1) if (med1 > 0 and med2 > 0) else np.nan
            )

            row["Shift_log10_GM"] = (
                np.log10(gm2) - np.log10(gm1) if (gm1 > 0 and gm2 > 0) else np.nan
            )

            # --- Fold change ---
            row["FoldChange_Median_24h_over_6h"] = med2 / med1 if med1 > 0 else np.nan

            row["FoldChange_GM_24h_over_6h"] = gm2 / gm1 if gm1 > 0 else np.nan

            if (
                f"{exp1_label}_Gate_2_Percentage" in row
                and f"{exp2_label}_Gate_2_Percentage" in row
            ):
                p1 = row[f"{exp1_label}_Gate_2_Percentage"]
                p2 = row[f"{exp2_label}_Gate_2_Percentage"]
                row["Delta_GFPposPct_Gate2"] = p2 - p1

            rows.append(row)

        if s1 and s2 and s1["Global_Median"] > 0 and s2["Global_Median"] > 0:
            shift = np.log10(s2["Global_Median"]) - np.log10(s1["Global_Median"])
            cell.text(
                0.05,
                0.95,
                f"Δlog10 med: {shift:.2f}",
                transform=cell.transAxes,
                ha="left",
                va="top",
                fontsize=7,
                fontweight="bold",
            )

        cell.legend(loc="upper right", fontsize=6, frameon=False)

    for r in range(1, num_rows + 1):
        for c in range(1, num_cols + 1):
            if (r, c) not in wells_with_data:
                ax[r, c].text(0.5, 0.5, "No Data", ha="center", va="center")
                ax[r, c].axis("off")

    out_png = os.path.join(plots_dir, f"{x_param}_KDE_{exp1_label}_vs_{exp2_label}.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved KDE grid: {out_png}")

    stats_df = pd.DataFrame(rows)
    stats_csv = os.path.join(stats_dir, f"{x_param}_kde_comparison_stats.csv")
    stats_df.to_csv(stats_csv, index=False)
    print(f"Saved comparison stats: {stats_csv}")

    max_abs = np.nanmax(
        np.abs(stats_df[["Shift_log10_Median", "Shift_log10_GM"]].values)
    )
    # Plot circular plate shift (same style as original 96well plots)
    plot_shift_plate_circles(
        stats_df,
        metric="Shift_log10_Median",
        output_dir=plots_dir,
        title="GFP Loss Heatmap (Δlog10 Median BL1-H)",
        vmin=-max_abs,
        vmax=max_abs,
    )

    # Plot circular plate shift (GEOMETRIC MEAN)
    # # Median fluorescence represents the typical cell,
    # # whereas geometric mean fluorescence (GM) provides a log-space average
    # # and is more sensitive to bright subpopulations.
    plot_shift_plate_circles(
        stats_df,
        metric="Shift_log10_GM",
        output_dir=plots_dir,
        title="GFP Loss Heatmap (Δlog10 Geometric Mean BL1-H)",
        vmin=-max_abs,
        vmax=max_abs,
    )

    # Plot circular plate for GFP+ percentage difference (Gate 2)
    # # Delta_GFPposPct_Gate2 represents the change in
    # # the fraction of GFP-positive cells (Gate 2) between 24h and 6h.
    max_abs = np.nanmax(np.abs(stats_df["Delta_GFPposPct_Gate2"]))
    plot_shift_plate_circles(
        stats_df,
        metric="Delta_GFPposPct_Gate2",
        output_dir=plots_dir,
        title="Δ GFP+ Cells (% in Gate 2)",
        vmin=-max_abs,
        vmax=max_abs,
    )

    max_abs_log = np.nanmax(
        np.abs(stats_df[["Shift_log10_Median", "Shift_log10_GM"]].values)
    )

    max_abs_pct = np.nanmax(np.abs(stats_df["Delta_GFPposPct_Gate2"].values))

    plot_shift_plate_circles_tri(
        stats_df,
        metrics=[
            "Shift_log10_Median",
            "Shift_log10_GM",
            "Delta_GFPposPct_Gate2",
        ],
        max_abs_log=max_abs_log,
        max_abs_pct=max_abs_pct,
        output_dir=plots_dir,
        title="Combined shifts: Median / GM / GFP+%",
    )

    print("Summary shift (BL1-H):")
    print(stats_df["Shift_log10_Median"].describe())
    print(
        "Fraction of wells with decrease (shift < 0):",
        np.mean(stats_df["Shift_log10_Median"] < 0),
    )

    singlet_df = pd.DataFrame(singlet_rows)
    singlet_csv = os.path.join(stats_dir, "singlet_stats_by_experiment.csv")
    singlet_df.to_csv(singlet_csv, index=False)
    print(f"Saved singlet stats: {singlet_csv}")

    return stats_df, singlet_df


def load_experiment_data(exp_dir, x_param="BL1-H"):

    files = [
        os.path.join(exp_dir, f) for f in os.listdir(exp_dir) if f.endswith(".fcs")
    ]

    well_data = {}

    for file in files:
        well_key, _ = extract_well_key(file)

        data, _ = read_fcs(file)
        if data is None:
            continue

        singlets, _, _, _ = remove_doublets(data)

        if x_param not in singlets.columns:
            continue

        values = singlets[x_param]
        values = values[values > 0]  # log safe

        well_data[well_key] = values

    return well_data


def read_fcs(file_path):
    try:
        meta, data = fcsparser.parse(file_path, reformat_meta=True)
        return data, list(data.columns)
    except Exception as e:
        print(f"Failed to read {file_path}: {e}")
        print(f"Error type: {type(e).__name__}")
        return None, []


# Mapping labels for channels
LABEL_MAP = {
    "BL1-H": "GFP",
    "YL2-H": "RFP",
    "SSC-A": "SSC-A",
    "SSC-H": "SSC-H",
    # Add more mappings here as needed
}


def extract_well_key(filename_or_path):
    """
    Extracts the last plate-like pattern A1..H12 found in the filename.
    """
    base_name = os.path.splitext(os.path.basename(filename_or_path))[0]
    match = re.search(r"([A-H])([0-9]{1,2})(?!.*[A-H][0-9]{1,2})", base_name)
    if match:
        letter_part = match.group(1)
        number_part = int(match.group(2))
        return match.group(0), (letter_part, number_part)
    return base_name, (base_name, 0)


def clean_data(data, columns, remove_zeros=False):
    """Remove rows with NaN or infinite values in specified columns."""
    initial_rows = len(data)
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=columns)
    if remove_zeros:
        for col in columns:
            data = data[data[col] > 0]
    removed_rows = initial_rows - len(data)
    if removed_rows > 0:
        print(f"Removed {removed_rows} rows with NaN or infinite values.")
    return data


def create_output_structure(results_directory):
    """
    Create necessary subdirectories within the results folder for storing outputs.
    Hist-only: plots + statistics.
    """
    plots_dir = os.path.join(results_directory, "plots")
    stats_dir = os.path.join(results_directory, "statistics")

    for directory in [plots_dir, stats_dir]:
        os.makedirs(directory, exist_ok=True)

    return plots_dir, stats_dir


def remove_doublets(data, ssc_a="SSC-A", ssc_h="SSC-H"):
    """
    Remove doublets based on SSC-A vs SSC-H ratio using vectorized operations.
    Returns:
      singlets_df, singlet_percentage, total_events, singlet_events
    """
    # Filter out non-positive values
    mask = (data[ssc_a] > 0) & (data[ssc_h] > 0)
    data_filtered = data[mask]

    if len(data_filtered) == 0:
        print("Warning: No positive values found for doublet removal")
        return data, 0.0, len(data), 0

    ssc_ratio = data_filtered[ssc_h] / data_filtered[ssc_a]
    singlet_mask = (ssc_ratio >= 0.7) & (ssc_ratio <= 2.0)  # Adjust if needed

    singlets = data_filtered[singlet_mask]
    total_events = len(data)
    singlet_events = len(singlets)
    singlet_percentage = (singlet_events / total_events) * 100 if total_events else 0.0

    return singlets, singlet_percentage, total_events, singlet_events


def plot_kde_line(
    singlets,
    x_param,
    file_name,
    ax,
    x_scale="linear",
    color="blue",
    xlim=None,
    gates=None,
    label=None,
):
    """
    KDE line only (no histogram bars) + statistics (Global_GM, Global_Median, Num_Events, gates...).
    """
    if x_param not in singlets.columns:
        ax.text(0.5, 0.5, f"Missing\n{x_param}", ha="center", va="center")
        ax.axis("off")
        return None

    cleaned = clean_data(singlets, [x_param])

    # Log-safe
    if x_scale == "log":
        cleaned = cleaned[cleaned[x_param] > 0]

    if cleaned.empty:
        ax.text(0.5, 0.5, "No Data", ha="center", va="center")
        ax.axis("off")
        return None

    # ---- LINE ONLY: KDE ----
    sns.kdeplot(
        cleaned[x_param],
        ax=ax,
        color=color,
        linewidth=2,
        log_scale=(x_scale == "log"),
        label=label,
    )

    if xlim:
        ax.set_xlim(xlim)

    if x_scale == "log":
        ax.set_xscale("log")

    # ---- Stats ----
    vals = cleaned[x_param].to_numpy()
    num_events = len(vals)
    global_median = float(np.median(vals))

    if np.all(vals > 0):
        global_gm = float(gmean(vals))
    else:
        pos = vals[vals > 0]
        global_gm = float(gmean(pos)) if len(pos) else 0.0

    stats = {
        "Global_GM": global_gm,
        "Global_Median": global_median,
        "Num_Events": int(num_events),
    }

    # ---- Gates (optional) ----
    if gates:
        gate_colors = plt.cm.rainbow(np.linspace(0, 1, len(gates)))
        for i, ((gmin, gmax), gcol) in enumerate(zip(gates, gate_colors)):
            ax.axvline(gmin, color=gcol, linestyle="--", linewidth=1)
            ax.axvline(gmax, color=gcol, linestyle="--", linewidth=1)

            gate_df = cleaned[(cleaned[x_param] >= gmin) & (cleaned[x_param] <= gmax)]
            pct = (len(gate_df) / num_events) * 100 if num_events else 0.0

            gv = gate_df[x_param].to_numpy()
            if len(gv) and np.all(gv > 0):
                ggm = float(gmean(gv))
            else:
                gpos = gv[gv > 0]
                ggm = float(gmean(gpos)) if len(gpos) else 0.0

            stats[f"Gate_{i+1}_Percentage"] = float(pct)
            stats[f"Gate_{i+1}_GM"] = float(ggm)

    ax.set_title(file_name, fontsize=8, fontweight="bold")
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    return stats


def main(command_line_arguments=None):
    start_time = time.time()
    print(f"FLtower version: {__version__} (KDE COMPARE)")

    run_args = parse_run_args(command_line_arguments)

    try:
        plots_config = load_parameters(run_args.input, run_args.parameters)

        EXP1_DIR = run_args.exp1_dir
        EXP2_DIR = run_args.exp2_dir

        if not EXP1_DIR or not EXP2_DIR:
            raise ValueError("Both --exp1-dir and --exp2-dir must be provided.")

        # Output
        output_folder = run_args.output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_directory = os.path.join(output_folder, f"compare_{timestamp}")
        os.makedirs(results_directory, exist_ok=True)
        print(f"Created results directory: {results_directory}")

        save_parameters(plots_config, results_directory, "used_parameters.json")

        compare_experiments_kde_grid(
            exp1_dir=EXP1_DIR,
            exp2_dir=EXP2_DIR,
            results_directory=results_directory,
            plots_config=plots_config,
            x_param="BL1-H",
            exp1_label=run_args.exp1_label,
            exp2_label=run_args.exp2_label,
        )

    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        traceback.print_exc()
    finally:
        total_runtime = time.time() - start_time
        print(f"\nTotal script runtime: {total_runtime:.2f} seconds")


if __name__ == "__main__":
    main()
