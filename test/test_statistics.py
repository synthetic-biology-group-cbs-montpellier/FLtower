"""Unit tests for fltower.core.statistics."""

import numpy as np
import pandas as pd
import pytest

from fltower.core.statistics import (
    calculate_gate_statistics,
    calculate_triplicate_stats,
    compute_statistics,
)


# ---------------------------------------------------------------------------
# compute_statistics
# ---------------------------------------------------------------------------
class TestComputeStatistics:
    def test_normal_data(self):
        s = pd.Series([10.0, 100.0, 1000.0])
        mean, gmean, median = compute_statistics(s)
        assert mean == pytest.approx(370.0)
        assert median == pytest.approx(100.0)
        assert gmean == pytest.approx(100.0)  # geometric mean of 10, 100, 1000

    def test_empty_series(self):
        s = pd.Series([], dtype=float)
        mean, gmean, median = compute_statistics(s)
        assert np.isnan(mean)
        assert np.isnan(gmean)
        assert np.isnan(median)

    def test_single_value(self):
        s = pd.Series([42.0])
        mean, gmean, median = compute_statistics(s)
        assert mean == pytest.approx(42.0)
        assert median == pytest.approx(42.0)
        assert gmean == pytest.approx(42.0)

    def test_with_nan(self):
        s = pd.Series([10.0, np.nan, 100.0])
        mean, gmean, median = compute_statistics(s)
        assert mean == pytest.approx(55.0)  # nanmean of [10, 100]
        assert median == pytest.approx(55.0)

    def test_all_negative(self):
        # gmean only on positive values → no positive → should handle gracefully
        s = pd.Series([-1.0, -2.0, -3.0])
        mean, gmean, median = compute_statistics(s)
        assert mean == pytest.approx(-2.0)
        assert median == pytest.approx(-2.0)


# ---------------------------------------------------------------------------
# calculate_gate_statistics
# ---------------------------------------------------------------------------
class TestCalculateGateStatistics:
    def test_all_in_gate(self):
        s = pd.Series([10.0, 50.0, 100.0])
        pct, gm = calculate_gate_statistics(s, 0, 200)
        assert pct == pytest.approx(100.0)
        assert gm > 0

    def test_none_in_gate(self):
        s = pd.Series([10.0, 50.0, 100.0])
        pct, gm = calculate_gate_statistics(s, 200, 500)
        assert pct == pytest.approx(0.0)
        assert gm == 0

    def test_partial_gate(self):
        s = pd.Series([10.0, 50.0, 100.0, 200.0])
        pct, gm = calculate_gate_statistics(s, 50, 100)
        assert pct == pytest.approx(50.0)  # 2 out of 4

    def test_boundary_inclusive(self):
        s = pd.Series([10.0, 50.0, 100.0])
        pct, gm = calculate_gate_statistics(s, 10, 100)
        assert pct == pytest.approx(100.0)

    def test_empty_data(self):
        s = pd.Series([], dtype=float)
        pct, gm = calculate_gate_statistics(s, 0, 100)
        assert pct == pytest.approx(0.0)
        assert gm == 0

    def test_inverted_gate(self):
        # gate_min > gate_max → no cells match
        s = pd.Series([10.0, 50.0])
        pct, gm = calculate_gate_statistics(s, 100, 0)
        assert pct == pytest.approx(0.0)
        assert gm == 0


# ---------------------------------------------------------------------------
# calculate_triplicate_stats
# ---------------------------------------------------------------------------
class TestCalculateTriplicateStats:
    def test_complete_triplicates(self):
        df = pd.DataFrame(
            {
                "Well": ["A1", "A2", "A3", "A4", "A5", "A6"],
                "GFP_Median": [100, 110, 105, 200, 210, 205],
            }
        )
        result = calculate_triplicate_stats(df, "GFP_Median")
        assert len(result) == 2  # A1 group and A2 group
        assert "GFP_Median_Mean" in result.columns
        assert "GFP_Median_Std" in result.columns

    def test_incomplete_triplicates_excluded(self):
        df = pd.DataFrame(
            {
                "Well": ["A1", "A2"],  # only 2, not 3
                "GFP_Median": [100, 110],
            }
        )
        result = calculate_triplicate_stats(df, "GFP_Median")
        assert len(result) == 0  # incomplete triplicates filtered out

    def test_empty_dataframe(self):
        df = pd.DataFrame(
            {"Well": pd.Series([], dtype=str), "Value": pd.Series([], dtype=float)}
        )
        result = calculate_triplicate_stats(df, "Value")
        assert len(result) == 0

    def test_grouping_logic(self):
        # A1,A2,A3 → group A1 ; A4,A5,A6 → group A2
        df = pd.DataFrame(
            {
                "Well": ["A1", "A2", "A3", "A4", "A5", "A6"],
                "Val": [10, 20, 30, 40, 50, 60],
            }
        )
        result = calculate_triplicate_stats(df, "Val")
        groups = result["Group"].tolist()
        assert "A1" in groups
        assert "A2" in groups
        # A1 group mean = 20.0, A2 group mean = 50.0
        a1_row = result[result["Group"] == "A1"]
        assert a1_row["Val_Mean"].values[0] == pytest.approx(20.0)

    def test_multiple_rows(self):
        # B1,B2,B3 → group B1
        df = pd.DataFrame(
            {
                "Well": ["B1", "B2", "B3"],
                "Metric": [5, 10, 15],
            }
        )
        result = calculate_triplicate_stats(df, "Metric")
        assert len(result) == 1
        assert result["Metric_Mean"].values[0] == pytest.approx(10.0)
        assert result["Metric_Std"].values[0] == pytest.approx(5.0)
