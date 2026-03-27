"""Quadrant gating — split 2D data into four quadrants and compute statistics."""

import logging

import numpy as np
from scipy.stats import gmean

logger = logging.getLogger("fltower")


def compute_quadrant_stats(data, x_param, y_param, quadrant_gates=None):
    """Compute quadrant percentages, geometric means and medians.

    Parameters
    ----------
    data : DataFrame
        Cleaned flow-cytometry data (no NaN/inf expected).
    x_param, y_param : str
        Column names for the two axes.
    quadrant_gates : dict or None
        ``{"x": float, "y": float}`` thresholds.  Falls back to medians.

    Returns
    -------
    dict
        Keys like ``Q1_Percentage``, ``Q2_BL1-H_GM``, ``Global_BL1-H_Median``, …
    """
    if quadrant_gates and "x" in quadrant_gates and "y" in quadrant_gates:
        x_mid = quadrant_gates["x"]
        y_mid = quadrant_gates["y"]
        logger.debug(f"Using provided quadrant gates: x={x_mid}, y={y_mid}")
    else:
        x_mid = np.median(data[x_param])
        y_mid = np.median(data[y_param])
        logger.debug(f"Using median values for quadrant gates: x={x_mid}, y={y_mid}")

    quadrants = {
        "Q1": (data[x_param] >= x_mid) & (data[y_param] >= y_mid),
        "Q2": (data[x_param] < x_mid) & (data[y_param] >= y_mid),
        "Q3": (data[x_param] < x_mid) & (data[y_param] < y_mid),
        "Q4": (data[x_param] >= x_mid) & (data[y_param] < y_mid),
    }

    total_cells = len(data)
    gate_stats = {}

    for quad_name, quad_mask in quadrants.items():
        cells_in_quad = data[quad_mask]
        percentage = (len(cells_in_quad) / total_cells) * 100
        gate_stats[f"{quad_name}_Percentage"] = percentage

        for param in [x_param, y_param]:
            if not any(dim in param for dim in ["FSC", "SSC"]):
                gate_stats[f"{quad_name}_{param}_GM"] = (
                    gmean(cells_in_quad[param]) if len(cells_in_quad) > 0 else 0
                )
                gate_stats[f"{quad_name}_{param}_Median"] = (
                    cells_in_quad[param].median() if len(cells_in_quad) > 0 else 0
                )

    for param in [x_param, y_param]:
        if not any(dim in param for dim in ["FSC", "SSC"]):
            gate_stats[f"Global_{param}_GM"] = gmean(data[param])
            gate_stats[f"Global_{param}_Median"] = data[param].median()

    return gate_stats, x_mid, y_mid
