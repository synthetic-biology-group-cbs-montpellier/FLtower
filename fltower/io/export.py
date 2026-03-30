"""Centralised CSV export helpers."""

import logging
import os

import pandas as pd

logger = logging.getLogger("fltower")


def _save_csv(df, csv_path, label):
    """Write a DataFrame to CSV and log the path."""
    df.to_csv(csv_path, index=False)
    logger.info("Saved %s to: %s", label, csv_path)


def save_statistics_csv(df_list, plot_key, stats_dir, results_name):
    """Concatenate a list of per-well DataFrames and save to CSV.

    Returns the concatenated DataFrame (or *None* if *df_list* is empty).
    """
    if not df_list:
        return None
    df = pd.concat(df_list, ignore_index=True)
    csv_path = os.path.join(stats_dir, f"{plot_key}_statistics_{results_name}.csv")
    _save_csv(df, csv_path, f"{plot_key} statistics")
    return df


def save_singlet_stats_csv(singlet_stats, stats_dir, results_name):
    """Save singlet gating statistics to CSV."""
    df = pd.DataFrame(singlet_stats)
    csv_path = os.path.join(stats_dir, f"singlet_statistics_{results_name}.csv")
    _save_csv(df, csv_path, "singlet statistics")


def save_triplicate_stats_csv(triplicate_df, plot_key, metric, triplicate_stats_dir):
    """Save triplicate statistics to CSV."""
    filename = f"{plot_key}_{metric}_triplicate_statistics.csv"
    csv_path = os.path.join(triplicate_stats_dir, filename)
    _save_csv(triplicate_df, csv_path, f"triplicate statistics for {metric}")


def save_stats_with_triplicates_csv(df, plot_key, stats_dir):
    """Save final statistics (with triplicate columns) to CSV."""
    csv_path = os.path.join(stats_dir, f"{plot_key}_statistics_with_triplicates.csv")
    _save_csv(df, csv_path, f"{plot_key} statistics with triplicates")
