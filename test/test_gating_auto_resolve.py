"""Tests for fltower.core.gating.auto.resolve."""

import numpy as np
import pandas as pd
import pytest

from fltower.core.gating.auto.resolve import (
    resolve_histogram_gates,
    resolve_quadrant_gates,
)


@pytest.fixture
def singlets():
    """Bimodal DataFrame with two fluorescence channels."""
    rng = np.random.default_rng(42)
    # BL1-H: bimodal in log-space (neg ~50, pos ~3000)
    bl1_neg = rng.lognormal(mean=4, sigma=0.5, size=600)
    bl1_pos = rng.lognormal(mean=8, sigma=0.5, size=400)
    bl1 = np.concatenate([bl1_neg, bl1_pos])

    # YL2-H: bimodal in log-space (neg ~30, pos ~2000)
    yl2_neg = rng.lognormal(mean=3.5, sigma=0.5, size=700)
    yl2_pos = rng.lognormal(mean=7.5, sigma=0.5, size=300)
    yl2 = np.concatenate([yl2_neg, yl2_pos])

    return pd.DataFrame({"BL1-H": bl1, "YL2-H": yl2})


# ── resolve_quadrant_gates ───────────────────────────────────────────


class TestResolveQuadrantGates:
    def test_none_returns_none(self, singlets):
        config = {"x_param": "BL1-H", "y_param": "YL2-H"}
        assert resolve_quadrant_gates(singlets, config) is None

    def test_manual_passthrough(self, singlets):
        config = {
            "x_param": "BL1-H",
            "y_param": "YL2-H",
            "quadrant_gates": {"x": 2600, "y": 2000},
        }
        result = resolve_quadrant_gates(singlets, config)
        assert result == {"x": 2600, "y": 2000}

    def test_auto_returns_dict(self, singlets):
        config = {
            "x_param": "BL1-H",
            "y_param": "YL2-H",
            "x_scale": "log",
            "y_scale": "log",
            "quadrant_gates": "auto",
        }
        result = resolve_quadrant_gates(singlets, config)
        assert isinstance(result, dict)
        assert "x" in result and "y" in result

    def test_auto_thresholds_between_populations(self, singlets):
        config = {
            "x_param": "BL1-H",
            "y_param": "YL2-H",
            "x_scale": "log",
            "y_scale": "log",
            "quadrant_gates": "auto",
        }
        result = resolve_quadrant_gates(singlets, config)
        # BL1-H: neg ~55, pos ~2981 → threshold should be between
        assert 80 < result["x"] < 2500
        # YL2-H: neg ~33, pos ~1808 → threshold should be between
        assert 50 < result["y"] < 1500

    def test_auto_linear_scale(self, singlets):
        config = {
            "x_param": "BL1-H",
            "y_param": "YL2-H",
            "x_scale": "linear",
            "y_scale": "linear",
            "quadrant_gates": "auto",
        }
        result = resolve_quadrant_gates(singlets, config)
        assert isinstance(result["x"], float)
        assert isinstance(result["y"], float)


# ── resolve_histogram_gates ──────────────────────────────────────────


class TestResolveHistogramGates:
    def test_none_returns_none(self, singlets):
        config = {"x_param": "BL1-H"}
        assert resolve_histogram_gates(singlets, config) is None

    def test_manual_passthrough(self, singlets):
        config = {"x_param": "BL1-H", "gates": [[10, 800], [800, 1e5]]}
        result = resolve_histogram_gates(singlets, config)
        assert result == [[10, 800], [800, 1e5]]

    def test_auto_returns_two_intervals(self, singlets):
        config = {"x_param": "BL1-H", "x_scale": "log", "gates": "auto"}
        result = resolve_histogram_gates(singlets, config)
        assert isinstance(result, list)
        assert len(result) == 2
        assert len(result[0]) == 2
        assert len(result[1]) == 2

    def test_auto_intervals_cover_range(self, singlets):
        config = {"x_param": "BL1-H", "x_scale": "log", "gates": "auto"}
        result = resolve_histogram_gates(singlets, config)
        # First interval starts at data min
        assert result[0][0] == pytest.approx(singlets["BL1-H"].min())
        # Second interval ends at data max
        assert result[1][1] == pytest.approx(singlets["BL1-H"].max())
        # Intervals meet at threshold
        assert result[0][1] == result[1][0]

    def test_auto_threshold_between_populations(self, singlets):
        config = {"x_param": "BL1-H", "x_scale": "log", "gates": "auto"}
        result = resolve_histogram_gates(singlets, config)
        threshold = result[0][1]
        # BL1-H neg ~55, pos ~2981
        assert 80 < threshold < 2500
