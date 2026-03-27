"""Otsu thresholding for automatic 1-D gating.

Otsu's method finds the threshold that minimises the weighted
intra-class variance of a bimodal distribution — exactly the
separation between a negative and a positive fluorescence population.

The implementation works on the raw event values (not binned pixels),
using a histogram with *n_bins* bins as the discrete approximation.
"""

import logging

import numpy as np

logger = logging.getLogger("fltower")


def otsu_threshold(values, n_bins=256):
    """Compute the Otsu threshold on a 1-D array of values.

    Parameters
    ----------
    values : array-like
        1-D fluorescence intensities (e.g. a single channel from a
        DataFrame column).  NaN and non-finite values are silently
        dropped.
    n_bins : int, optional
        Number of histogram bins used for the computation (default 256).

    Returns
    -------
    float
        The optimal threshold separating the two populations.

    Raises
    ------
    ValueError
        If *values* contains fewer than 2 finite data points.
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]

    if len(arr) < 2:
        raise ValueError(
            f"otsu_threshold requires at least 2 finite values, got {len(arr)}"
        )

    # Build the histogram
    counts, bin_edges = np.histogram(arr, bins=n_bins)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Normalise to probabilities
    total = counts.sum()
    probs = counts / total

    # Cumulative sums
    cum_prob = np.cumsum(probs)  # omega(t)
    cum_mean = np.cumsum(probs * bin_centres)  # mu(t) * omega(t)
    global_mean = cum_mean[-1]

    # Between-class variance for every possible threshold
    # sigma_b^2 = [mu_T * omega(t) - mu(t)]^2 / [omega(t) * (1 - omega(t))]
    denom = cum_prob * (1 - cum_prob)
    # Avoid division by zero at boundaries
    valid = denom > 0
    sigma_b_sq = np.zeros_like(probs)
    sigma_b_sq[valid] = (global_mean * cum_prob[valid] - cum_mean[valid]) ** 2 / denom[
        valid
    ]

    best_idx = int(np.argmax(sigma_b_sq))
    threshold = float(bin_centres[best_idx])

    logger.debug(
        "Otsu threshold: %.4f (n_bins=%d, n_values=%d)", threshold, n_bins, len(arr)
    )
    return threshold


def otsu_threshold_log(values, n_bins=256):
    """Otsu threshold computed in log-space, returned in linear scale.

    Useful for fluorescence data displayed on a log axis. The method
    applies ``log10`` to strictly-positive values, runs Otsu in that
    space, then converts back with ``10**threshold``.

    Parameters
    ----------
    values : array-like
        1-D fluorescence intensities.  Values <= 0 are excluded.
    n_bins : int, optional
        Number of histogram bins (default 256).

    Returns
    -------
    float
        Threshold in the original linear scale.

    Raises
    ------
    ValueError
        If fewer than 2 positive finite values remain.
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr) & (arr > 0)]

    if len(arr) < 2:
        raise ValueError(
            f"otsu_threshold_log requires at least 2 positive finite values, got {len(arr)}"
        )

    log_threshold = otsu_threshold(np.log10(arr), n_bins=n_bins)
    threshold = 10.0**log_threshold

    logger.debug(
        "Otsu log-space threshold: %.4f (linear: %.4f)", log_threshold, threshold
    )
    return threshold
