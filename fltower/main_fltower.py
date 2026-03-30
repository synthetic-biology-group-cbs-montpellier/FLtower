#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main file of FLtower software.
"""

import logging
import os
import sys
import time
import warnings
from datetime import datetime

from fltower.__version__ import __version__
from fltower.core.pipeline import (  # noqa: F401 – re-exported for backward compat
    create_output_structure,
    extract_well_key,
    run_pipeline,
)
from fltower.data_manager import load_parameters, save_parameters
from fltower.plotting.report import compile_summary_report
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
    setup_logging(verbose=run_args.verbose, quiet=run_args.quiet)

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

        # Re-init logging with a file handler now that the output dir exists
        log_file = os.path.join(results_directory, "fltower.log")
        setup_logging(verbose=run_args.verbose, quiet=run_args.quiet, log_file=log_file)
        logger.debug(f"Log file: {log_file}")

        used_params = save_parameters(
            plots_config, results_directory, "used_parameters.json"
        )
        logger.info(f"Save used parameters: {used_params}")

        scatter_dfs, histogram_dfs, singlet_df, runtime = run_pipeline(
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
