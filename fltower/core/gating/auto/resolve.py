"""Resolve ``"auto"`` gate specifications to concrete numeric values.

This module bridges the config layer (which may contain ``"auto"``)
and the plotting / statistics layers (which expect numeric gates).
"""

import logging

from fltower.core.cleaning import clean_data
from fltower.core.gating.auto.otsu import otsu_threshold, otsu_threshold_log

logger = logging.getLogger("fltower")


def resolve_quadrant_gates(singlets, config):
    """Return a ``{"x": float, "y": float}`` dict or *None*.

    If ``config["quadrant_gates"]`` is:

    * a dict with ``x`` and ``y`` → returned as-is (manual).
    * the string ``"auto"`` → Otsu threshold computed per axis in
      the appropriate scale (log or linear).
    * absent or ``None`` → ``None`` (no gates).
    """
    raw = config.get("quadrant_gates")
    if raw is None:
        return None
    if isinstance(raw, dict) and "x" in raw and "y" in raw:
        return raw  # manual gates

    if raw == "auto":
        x_param = config["x_param"]
        y_param = config["y_param"]
        x_scale = config.get("x_scale", "linear")
        y_scale = config.get("y_scale", "linear")

        cleaned = clean_data(singlets, [x_param, y_param])

        x_thresh = _auto_threshold(cleaned[x_param], x_scale)
        y_thresh = _auto_threshold(cleaned[y_param], y_scale)

        logger.info(
            "Auto quadrant gates (Otsu): %s=%.2f, %s=%.2f",
            x_param,
            x_thresh,
            y_param,
            y_thresh,
        )
        return {"x": x_thresh, "y": y_thresh}

    return raw  # fall through for unexpected values


def resolve_histogram_gates(singlets, config):
    """Return a list of ``[min, max]`` pairs or *None*.

    If ``config["gates"]`` is:

    * a list of ``[min, max]`` → returned as-is (manual).
    * the string ``"auto"`` → a single Otsu threshold splits the
      data range into two intervals: ``[data_min, threshold]`` and
      ``[threshold, data_max]``.
    * absent or ``None`` → ``None`` (no gates).
    """
    raw = config.get("gates")
    if raw is None:
        return None
    if isinstance(raw, list):
        return raw  # manual gates

    if raw == "auto":
        x_param = config["x_param"]
        x_scale = config.get("x_scale", "linear")

        cleaned = clean_data(singlets, [x_param])
        col = cleaned[x_param]

        threshold = _auto_threshold(col, x_scale)
        data_min = float(col.min())
        data_max = float(col.max())

        logger.info(
            "Auto histogram gate (Otsu) for %s: threshold=%.2f, range=[%.2f, %.2f]",
            x_param,
            threshold,
            data_min,
            data_max,
        )
        return [[data_min, threshold], [threshold, data_max]]

    return raw  # fall through


def _auto_threshold(series, scale):
    """Compute Otsu threshold in the appropriate scale."""
    if scale == "log":
        return otsu_threshold_log(series)
    return otsu_threshold(series)
