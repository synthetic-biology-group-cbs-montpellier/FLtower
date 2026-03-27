"""Interval gating — filter data within min/max bounds and compute statistics."""

from scipy.stats import gmean


def compute_interval_stats(data, column, gates):
    """Compute percentage and geometric mean for each interval gate.

    Parameters
    ----------
    data : DataFrame
        Cleaned flow-cytometry data.
    column : str
        Column name to gate on.
    gates : list of (float, float)
        Each element is ``(gate_min, gate_max)``.

    Returns
    -------
    dict
        Keys like ``Gate_1_Percentage``, ``Gate_1_GM``, …
    """
    num_events = len(data)
    stats = {}
    for i, (gate_min, gate_max) in enumerate(gates):
        gate_data = data[(data[column] >= gate_min) & (data[column] <= gate_max)]
        percentage = (len(gate_data) / num_events) * 100 if num_events > 0 else 0.0
        gate_gm = gmean(gate_data[column]) if len(gate_data) > 0 else 0
        stats[f"Gate_{i + 1}_Percentage"] = percentage
        stats[f"Gate_{i + 1}_GM"] = gate_gm
    return stats
