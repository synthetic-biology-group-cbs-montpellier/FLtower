"""Singlet gating — remove doublets based on SSC-H/SSC-A ratio."""

import logging

logger = logging.getLogger("fltower")

SINGLET_RATIO_LOWER = 0.7
SINGLET_RATIO_UPPER = 2.0


def remove_doublets(
    data,
    ssc_a="SSC-A",
    ssc_h="SSC-H",
    singlet_lower=SINGLET_RATIO_LOWER,
    singlet_upper=SINGLET_RATIO_UPPER,
):
    """
    Remove doublets based on SSC-A vs SSC-H plot using vectorized operations.
    Returns the filtered data, the percentage of singlets, total events, and number of singlets.
    """
    # Filter out non-positive values
    mask = (data[ssc_a] > 0) & (data[ssc_h] > 0)
    data_filtered = data[mask]

    if len(data_filtered) == 0:
        logger.warning("No positive values found for doublet removal")
        return data, 0, len(data), 0

    ssc_ratio = data_filtered[ssc_h] / data_filtered[ssc_a]
    singlet_mask = (ssc_ratio >= singlet_lower) & (ssc_ratio <= singlet_upper)
    singlets = data_filtered[singlet_mask]
    total_events = len(data)
    singlet_events = len(singlets)
    singlet_percentage = (singlet_events / total_events) * 100

    return singlets, singlet_percentage, total_events, singlet_events
