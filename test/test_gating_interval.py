"""Unit tests for fltower.core.gating.interval (compute_interval_stats)."""

import pandas as pd
import pytest

from fltower.core.gating.interval import compute_interval_stats


class TestComputeIntervalStats:
    def test_single_gate_all_in(self):
        df = pd.DataFrame({"x": [10.0, 50.0, 100.0]})
        stats = compute_interval_stats(df, "x", [(0, 200)])
        assert stats["Gate_1_Percentage"] == pytest.approx(100.0)
        assert stats["Gate_1_GM"] > 0

    def test_single_gate_none_in(self):
        df = pd.DataFrame({"x": [10.0, 50.0, 100.0]})
        stats = compute_interval_stats(df, "x", [(200, 500)])
        assert stats["Gate_1_Percentage"] == pytest.approx(0.0)
        assert stats["Gate_1_GM"] == 0

    def test_single_gate_partial(self):
        df = pd.DataFrame({"x": [10.0, 50.0, 100.0, 200.0]})
        stats = compute_interval_stats(df, "x", [(50, 100)])
        assert stats["Gate_1_Percentage"] == pytest.approx(50.0)

    def test_multiple_gates(self):
        df = pd.DataFrame({"x": [10.0, 50.0, 100.0, 200.0]})
        stats = compute_interval_stats(df, "x", [(0, 50), (100, 200)])
        assert "Gate_1_Percentage" in stats
        assert "Gate_2_Percentage" in stats
        assert stats["Gate_1_Percentage"] == pytest.approx(50.0)
        assert stats["Gate_2_Percentage"] == pytest.approx(50.0)

    def test_empty_data(self):
        df = pd.DataFrame({"x": pd.Series([], dtype=float)})
        stats = compute_interval_stats(df, "x", [(0, 100)])
        assert stats["Gate_1_Percentage"] == pytest.approx(0.0)
        assert stats["Gate_1_GM"] == 0

    def test_boundary_inclusive(self):
        df = pd.DataFrame({"x": [10.0, 50.0, 100.0]})
        stats = compute_interval_stats(df, "x", [(10, 100)])
        assert stats["Gate_1_Percentage"] == pytest.approx(100.0)
