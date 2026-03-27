"""Data cleaning utilities for flow cytometry DataFrames."""

import logging

import numpy as np

logger = logging.getLogger("fltower")


def clean_data(data, columns, remove_zeros=False):
    """Remove rows with NaN or infinite values in specified columns."""
    initial_rows = len(data)
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=columns)
    if remove_zeros:
        for col in columns:
            data = data[data[col] > 0]
    removed_rows = initial_rows - len(data)
    if removed_rows > 0:
        logger.debug(f"Removed {removed_rows} rows with NaN or infinite values.")
    return data
