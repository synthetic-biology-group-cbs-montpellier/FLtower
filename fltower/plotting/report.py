"""PDF summary report generation."""

import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from fltower.__version__ import __version__

logger = logging.getLogger("fltower")


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
