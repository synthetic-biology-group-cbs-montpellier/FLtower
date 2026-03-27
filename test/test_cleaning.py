"""Unit tests for fltower.core.cleaning (clean_data)."""

import numpy as np
import pandas as pd

from fltower.core.cleaning import clean_data


class TestCleanData:
    def test_removes_nan(self):
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0], "y": [4.0, 5.0, 6.0]})
        result = clean_data(df, ["x"])
        assert len(result) == 2
        assert result["x"].tolist() == [1.0, 3.0]

    def test_removes_inf(self):
        df = pd.DataFrame({"x": [1.0, np.inf, -np.inf], "y": [1.0, 2.0, 3.0]})
        result = clean_data(df, ["x"])
        assert len(result) == 1

    def test_remove_zeros(self):
        df = pd.DataFrame({"x": [0, -1, 5, 10], "y": [1, 2, 3, 4]})
        result = clean_data(df, ["x"], remove_zeros=True)
        assert len(result) == 2
        assert result["x"].tolist() == [5, 10]

    def test_empty_dataframe(self):
        df = pd.DataFrame({"x": pd.Series([], dtype=float)})
        result = clean_data(df, ["x"])
        assert len(result) == 0

    def test_all_nan(self):
        df = pd.DataFrame({"x": [np.nan, np.nan]})
        result = clean_data(df, ["x"])
        assert len(result) == 0

    def test_no_rows_removed(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        result = clean_data(df, ["x"])
        assert len(result) == 3

    def test_multiple_columns(self):
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0], "y": [np.nan, 5.0, 6.0]})
        result = clean_data(df, ["x", "y"])
        assert len(result) == 1  # only row 2 has both valid
