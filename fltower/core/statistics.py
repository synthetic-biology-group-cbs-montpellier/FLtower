"""Statistical functions for flow cytometry data analysis."""

import logging

import numpy as np
from scipy.stats import gmean

logger = logging.getLogger("fltower")


def compute_statistics(data):
    """
    Calculate mean, geometric mean, and median for the data.
    """
    if not data.empty:
        mean = data.mean()
        geometric_mean = gmean(
            data[data > 0]
        )  # geometric mean only for positive values
        median = data.median()
    else:
        mean = geometric_mean = median = np.nan
    return mean, geometric_mean, median


def calculate_gate_statistics(data, gate_min, gate_max):
    """
    Calculate the percentage and geometric mean of cells within a specified gate.
    """
    total_cells = len(data)
    if total_cells == 0:
        return 0.0, 0
    cells_in_gate = data[(data >= gate_min) & (data <= gate_max)]
    percentage = (len(cells_in_gate) / total_cells) * 100
    geo_mean = gmean(cells_in_gate) if len(cells_in_gate) > 0 else 0
    return percentage, geo_mean


def calculate_triplicate_stats(df, metric_column, well_column="Well"):
    """
    Calculate mean and standard deviation for triplicates, handling partial plates.
    """
    df["Group"] = df[well_column].apply(lambda x: f"{x[0]}{(int(x[1:]) - 1) // 3 + 1}")
    grouped = df.groupby("Group")

    triplicate_stats = grouped.agg(
        {
            metric_column: [
                "mean",
                "std",
                "count",
            ]  # Add count to check for incomplete triplicates
        }
    ).reset_index()

    triplicate_stats.columns = [
        "Group",
        f"{metric_column}_Mean",
        f"{metric_column}_Std",
        f"{metric_column}_Count",
    ]

    # Filter out incomplete triplicates
    triplicate_stats = triplicate_stats[triplicate_stats[f"{metric_column}_Count"] == 3]

    return triplicate_stats
