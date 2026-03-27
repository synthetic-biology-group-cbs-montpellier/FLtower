"""Unit tests for fltower.core.gating.singlet (remove_doublets)."""

import pandas as pd
import pytest

from fltower.core.gating.singlet import remove_doublets


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
