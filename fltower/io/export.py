"""Centralised CSV export helpers."""

import logging
import os

import pandas as pd

logger = logging.getLogger("fltower")


def save_statistics_csv(df_list, plot_key, stats_dir, results_name):
    """Concatenate a list of per-well DataFrames and save to CSV.

    Returns the concatenated DataFrame (or *None* if *df_list* is empty).
    """
    if not df_list:
        return None
    df = pd.concat(df_list, ignore_index=True)
    csv_path = os.path.join(stats_dir, f"{plot_key}_statistics_{results_name}.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved {plot_key} statistics to: {csv_path}")
    return df


def save_singlet_stats_csv(singlet_stats, stats_dir, results_name):
    """Save singlet gating statistics to CSV."""
    df = pd.DataFrame(singlet_stats)
    csv_path = os.path.join(stats_dir, f"singlet_statistics_{results_name}.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved singlet statistics to: {csv_path}")


def save_triplicate_stats_csv(triplicate_df, plot_key, metric, triplicate_stats_dir):
    """Save triplicate statistics to CSV."""
    filename = f"{plot_key}_{metric}_triplicate_statistics.csv"
    csv_path = os.path.join(triplicate_stats_dir, filename)
    triplicate_df.to_csv(csv_path, index=False)
    logger.info("Saved triplicate statistics for %s to: %s", metric, csv_path)


def save_stats_with_triplicates_csv(df, plot_key, stats_dir):
    """Save final statistics (with triplicate columns) to CSV."""
    csv_path = os.path.join(stats_dir, f"{plot_key}_statistics_with_triplicates.csv")
    df.to_csv(csv_path, index=False)
    logger.info("Saved %s statistics with triplicates to: %s", plot_key, csv_path)
