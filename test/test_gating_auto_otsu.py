"""Tests for fltower.core.gating.auto.otsu."""

import numpy as np
import pytest

from fltower.core.gating.auto.otsu import otsu_threshold, otsu_threshold_log

# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def bimodal_linear():
    """Two well-separated Gaussian populations in linear space."""
    rng = np.random.default_rng(42)
    low = rng.normal(100, 20, size=500)
    high = rng.normal(500, 30, size=500)
    return np.concatenate([low, high])


@pytest.fixture
def bimodal_log():
    """Two populations separated on a log scale (typical fluorescence)."""
    rng = np.random.default_rng(42)
    neg = rng.lognormal(mean=4, sigma=0.5, size=600)  # ~55
    pos = rng.lognormal(mean=8, sigma=0.5, size=400)  # ~2981
    return np.concatenate([neg, pos])


# ── otsu_threshold ───────────────────────────────────────────────────


class TestOtsuThreshold:
    def test_bimodal_finds_valley(self, bimodal_linear):
        t = otsu_threshold(bimodal_linear)
        # Threshold should land between the two peaks (~100, ~500)
        assert 150 < t < 400

    def test_symmetric_bimodal(self):
        rng = np.random.default_rng(0)
        data = np.concatenate([rng.normal(0, 1, 1000), rng.normal(10, 1, 1000)])
        t = otsu_threshold(data)
        assert 3 < t < 7

    def test_unequal_populations(self):
        rng = np.random.default_rng(1)
        data = np.concatenate([rng.normal(50, 5, 900), rng.normal(200, 10, 100)])
        t = otsu_threshold(data)
        assert 60 < t < 180

    def test_returns_float(self, bimodal_linear):
        assert isinstance(otsu_threshold(bimodal_linear), float)

    def test_custom_bins(self, bimodal_linear):
        t64 = otsu_threshold(bimodal_linear, n_bins=64)
        t512 = otsu_threshold(bimodal_linear, n_bins=512)
        # Both should find roughly the same valley
        assert abs(t64 - t512) < 50

    def test_nan_values_ignored(self, bimodal_linear):
        data_with_nan = np.append(bimodal_linear, [np.nan, np.nan, np.inf])
        t_clean = otsu_threshold(bimodal_linear)
        t_dirty = otsu_threshold(data_with_nan)
        assert abs(t_clean - t_dirty) < 1e-6

    def test_error_on_empty(self):
        with pytest.raises(ValueError, match="at least 2"):
            otsu_threshold([])

    def test_error_on_single_value(self):
        with pytest.raises(ValueError, match="at least 2"):
            otsu_threshold([42.0])

    def test_error_on_all_nan(self):
        with pytest.raises(ValueError, match="at least 2"):
            otsu_threshold([np.nan, np.nan])

    def test_uniform_data(self):
        """On uniform data the threshold should still be a valid float."""
        rng = np.random.default_rng(99)
        data = rng.uniform(0, 100, 1000)
        t = otsu_threshold(data)
        assert 0 <= t <= 100


# ── otsu_threshold_log ───────────────────────────────────────────────


class TestOtsuThresholdLog:
    def test_bimodal_log_finds_valley(self, bimodal_log):
        t = otsu_threshold_log(bimodal_log)
        # Should fall between the two log-normal peaks (~55 and ~2981)
        assert 100 < t < 2000

    def test_returns_linear_scale(self, bimodal_log):
        t = otsu_threshold_log(bimodal_log)
        # The threshold should be in the original scale, not log10
        assert t > 10  # not a log10 value

    def test_negative_values_excluded(self):
        rng = np.random.default_rng(42)
        pos = np.concatenate([rng.lognormal(3, 0.5, 500), rng.lognormal(7, 0.5, 500)])
        data = np.append(pos, [-5, -10, 0])
        t = otsu_threshold_log(data)
        assert t > 0

    def test_error_on_all_nonpositive(self):
        with pytest.raises(ValueError, match="at least 2"):
            otsu_threshold_log([-1, 0, -5])

    def test_consistent_with_manual_log(self, bimodal_log):
        """The log-space Otsu should give roughly 10^(otsu on log10(data))."""
        pos = bimodal_log[bimodal_log > 0]
        manual_t = 10 ** otsu_threshold(np.log10(pos))
        auto_t = otsu_threshold_log(bimodal_log)
        assert abs(manual_t - auto_t) < 1e-6
