"""96-well plate grid visualisation."""

import logging
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger("fltower")


def plot_96well_grid(
    data, metric, plot_title, well_plots_dir, parameter_name=None, vmin=None, vmax=None
):
    logger.debug(f"Starting plot_96well_grid for metric: {metric}")
    fig, ax = plt.subplots(figsize=(16, 9))

    # Define the full 96-well plate layout
    rows = ["A", "B", "C", "D", "E", "F", "G", "H"]
    columns = range(1, 13)

    # Ensure 'Well' column exists in the data
    if "Well" not in data.columns:
        logger.error(
            f"'Well' column not found in data for metric {metric}. Skipping plot."
        )
        plt.close(fig)
        return

    # Extract row and column from Well
    data["Row"] = data["Well"].str[0]
    data["Column"] = data["Well"].str[1:].astype(int)

    # Determine color scale range
    valid_values = data[metric][np.isfinite(data[metric])]
    if len(valid_values) == 0:
        logger.error(f"No valid data for metric {metric}. Skipping plot.")
        plt.close(fig)
        return

    if vmin is None and vmax is None:
        vmin = 0 if "Percentage" in metric else valid_values.min()
        vmax = 100 if "Percentage" in metric else valid_values.max()

    logger.debug(f"Color scale range: {vmin} to {vmax}")

    # Ensure vmin and vmax are not equal
    if vmin == vmax:
        vmax = vmin + 1

    # Create a dictionary for quick data lookup
    data_dict = {
        (row, col): data[(data["Row"] == row) & (data["Column"] == col)][metric].values[
            0
        ]
        for row in rows
        for col in columns
        if not data[(data["Row"] == row) & (data["Column"] == col)].empty
    }

    # Plot each well
    for row_idx, row in enumerate(rows):
        for col in columns:
            value = data_dict.get((row, col), np.nan)
            if np.isfinite(value):
                color_val = (value - vmin) / (vmax - vmin)
                circle = plt.Circle(
                    (col - 1, 7 - row_idx),
                    0.4,
                    fill=True,
                    color=plt.cm.viridis(color_val),
                )
                ax.add_artist(circle)

                # Determine text color based on background brightness
                bg_color = plt.cm.viridis(color_val)
                text_color = (
                    "white" if mcolors.rgb_to_hsv(bg_color[:3])[2] < 0.6 else "black"
                )

                ax.text(
                    col - 1,
                    7 - row_idx,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                    fontweight="bold",
                )
            else:
                # Draw an empty circle for missing data
                circle = plt.Circle(
                    (col - 1, 7 - row_idx), 0.4, fill=False, color="gray"
                )
                ax.add_artist(circle)
                ax.text(
                    col - 1,
                    7 - row_idx,
                    "No Data",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    # Set the limits and remove axes
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-0.5, 7.5)
    ax.axis("off")

    # Add row labels
    for i, row in enumerate(rows):
        ax.text(
            -0.8, 7 - i, row, ha="center", va="center", fontsize=12, fontweight="bold"
        )

    # Add column labels closer to the samples
    for i in columns:
        ax.text(
            i - 1, 7.6, str(i), ha="center", va="center", fontsize=12, fontweight="bold"
        )

    # Add title at the top
    plt.title(plot_title, fontsize=16, pad=20)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", aspect=30, pad=0.08)
    cbar.set_label(metric, fontsize=12)

    # Save the plot
    if parameter_name:
        plot_filename = (
            f"{parameter_name}_{metric.replace(' ', '_').lower()}_96well_grid.png"
        )
    else:
        plot_filename = f"{metric.replace(' ', '_').lower()}_96well_grid.png"
    plot_path = os.path.join(well_plots_dir, plot_filename)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved 96-well grid plot for {metric} to: {plot_path}")
