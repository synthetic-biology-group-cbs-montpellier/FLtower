"""Unit tests for fltower.core.gating.quadrant (compute_quadrant_stats)."""

import numpy as np
import pandas as pd
import pytest

from fltower.core.gating.quadrant import compute_quadrant_stats


class TestComputeQuadrantStats:
    def _make_data(self, n=100):
        """Create a simple 2D dataset spread across four quadrants."""
        rng = np.random.RandomState(42)
        return pd.DataFrame(
            {
                "BL1-H": rng.uniform(0, 200, n),
                "YL2-H": rng.uniform(0, 200, n),
            }
        )

    def test_returns_all_quadrant_percentages(self):
        df = self._make_data()
        stats, x_mid, y_mid = compute_quadrant_stats(df, "BL1-H", "YL2-H")
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            assert f"{q}_Percentage" in stats
        total = sum(stats[f"{q}_Percentage"] for q in ["Q1", "Q2", "Q3", "Q4"])
        assert total == pytest.approx(100.0)

    def test_custom_gates(self):
        df = pd.DataFrame({"BL1-H": [10, 20, 30, 40], "YL2-H": [10, 20, 30, 40]})
        stats, x_mid, y_mid = compute_quadrant_stats(
            df, "BL1-H", "YL2-H", quadrant_gates={"x": 25, "y": 25}
        )
        assert x_mid == 25
        assert y_mid == 25
        assert stats["Q1_Percentage"] == pytest.approx(50.0)  # 30,40 >= 25
        assert stats["Q3_Percentage"] == pytest.approx(50.0)  # 10,20 < 25

    def test_falls_back_to_median(self):
        df = pd.DataFrame({"BL1-H": [10, 20, 30, 40], "YL2-H": [10, 20, 30, 40]})
        stats, x_mid, y_mid = compute_quadrant_stats(df, "BL1-H", "YL2-H")
        assert x_mid == pytest.approx(np.median([10, 20, 30, 40]))
        assert y_mid == pytest.approx(np.median([10, 20, 30, 40]))

    def test_global_gm_and_median(self):
        df = pd.DataFrame({"BL1-H": [10, 100], "YL2-H": [20, 200]})
        stats, _, _ = compute_quadrant_stats(df, "BL1-H", "YL2-H")
        assert "Global_BL1-H_GM" in stats
        assert "Global_BL1-H_Median" in stats
        assert "Global_YL2-H_GM" in stats

    def test_fsc_ssc_channels_excluded_from_gm(self):
        df = pd.DataFrame({"FSC-A": [10, 20], "SSC-A": [30, 40]})
        stats, _, _ = compute_quadrant_stats(df, "FSC-A", "SSC-A")
        # FSC/SSC channels should NOT have GM/Median keys
        assert "Q1_FSC-A_GM" not in stats
        assert "Global_FSC-A_GM" not in stats
