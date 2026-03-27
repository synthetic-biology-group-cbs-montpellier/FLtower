"""Unit tests for individual functions in main_fltower.py (ticket 1.4)."""

import numpy as np
import pandas as pd
import pytest

from fltower.main_fltower import (
    calculate_gate_statistics,
    calculate_triplicate_stats,
    compute_statistics,
    extract_well_key,
    remove_doublets,
)


# ---------------------------------------------------------------------------
# extract_well_key
# ---------------------------------------------------------------------------
class TestExtractWellKey:
    def test_standard_filename(self):
        key, (letter, num) = extract_well_key("Experiment - P1 - Caff_A1.fcs")
        assert key == "A1"
        assert letter == "A"
        assert num == 1

    def test_two_digit_well(self):
        key, (letter, num) = extract_well_key("Sample_H12.fcs")
        assert key == "H12"
        assert letter == "H"
        assert num == 12

    def test_multiple_well_patterns_takes_last(self):
        # B2 appears first, but A3 is the last well pattern
        key, _ = extract_well_key("B2_something_A3.fcs")
        assert key == "A3"

    def test_no_well_pattern(self):
        key, (name, num) = extract_well_key("random_file.fcs")
        assert num == 0  # fallback: no match

    def test_full_path(self):
        key, _ = extract_well_key("/data/experiment/Sample_C5.fcs")
        assert key == "C5"

    def test_no_extension(self):
        key, _ = extract_well_key("Sample_D7")
        assert key == "D7"


# ---------------------------------------------------------------------------
# remove_doublets
# ---------------------------------------------------------------------------
class TestRemoveDoublets:
    def _make_data(self, ssc_a, ssc_h):
        return pd.DataFrame({"SSC-A": ssc_a, "SSC-H": ssc_h})

    def test_all_singlets(self):
        # ratio = 1.0, within [0.7, 2.0]
        data = self._make_data([100, 200, 300], [100, 200, 300])
        singlets, pct, total, n_singlets = remove_doublets(data)
        assert n_singlets == 3
        assert total == 3
        assert pct == pytest.approx(100.0)

    def test_all_doublets(self):
        # ratio = 0.1, outside [0.7, 2.0]
        data = self._make_data([1000, 2000], [100, 200])
        singlets, pct, total, n_singlets = remove_doublets(data)
        assert n_singlets == 0
        assert total == 2

    def test_mixed(self):
        # row 0: ratio=1.0 (singlet), row 1: ratio=0.1 (doublet)
        data = self._make_data([100, 1000], [100, 100])
        singlets, pct, total, n_singlets = remove_doublets(data)
        assert n_singlets == 1
        assert total == 2
        assert pct == pytest.approx(50.0)

    def test_empty_dataframe(self):
        data = self._make_data([], [])
        singlets, pct, total, n_singlets = remove_doublets(data)
        assert total == 0
        assert n_singlets == 0

    def test_all_non_positive(self):
        data = self._make_data([0, -1], [0, -2])
        singlets, pct, total, n_singlets = remove_doublets(data)
        assert n_singlets == 0

    def test_custom_column_names(self):
        data = pd.DataFrame({"myA": [100, 200], "myH": [100, 200]})
        singlets, pct, total, n_singlets = remove_doublets(
            data, ssc_a="myA", ssc_h="myH"
        )
        assert n_singlets == 2

    def test_boundary_ratios(self):
        # ratio exactly 0.7 and exactly 2.0 should be included
        data = self._make_data([100, 100], [70, 200])
        singlets, pct, total, n_singlets = remove_doublets(data)
        assert n_singlets == 2

    def test_custom_thresholds(self):
        # ratio = 1.0 for all rows; narrow gate [0.9, 1.1] should still include them
        data = self._make_data([100, 200], [100, 200])
        singlets, pct, total, n_singlets = remove_doublets(
            data, singlet_lower=0.9, singlet_upper=1.1
        )
        assert n_singlets == 2

    def test_custom_thresholds_excludes(self):
        # ratio = 1.0; gate [1.5, 2.5] should exclude all
        data = self._make_data([100, 200], [100, 200])
        singlets, pct, total, n_singlets = remove_doublets(
            data, singlet_lower=1.5, singlet_upper=2.5
        )
        assert n_singlets == 0


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
