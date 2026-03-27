"""Tests for fltower.core.gating.hierarchy."""

import numpy as np
import pandas as pd
import pytest

from fltower.core.gating.hierarchy import (
    GatingResult,
    apply_gating_hierarchy,
    get_gating_summary,
)


@pytest.fixture
def raw_data():
    """DataFrame that mimics a small FCS dataset with SSC-A/SSC-H columns."""
    rng = np.random.default_rng(42)
    n = 200
    ssc_a = rng.uniform(100, 10000, n)
    # Most events have SSC-H/SSC-A ratio in [0.7, 2.0] (singlets).
    # A few outliers will be removed as doublets.
    ssc_h = ssc_a * rng.uniform(0.5, 2.5, n)
    return pd.DataFrame(
        {
            "SSC-A": ssc_a,
            "SSC-H": ssc_h,
            "BL1-H": rng.uniform(1, 1000, n),
            "YL2-H": rng.uniform(1, 1000, n),
        }
    )


# ── GatingResult dataclass ──────────────────────────────────────────


class TestGatingResult:
    def test_leaf_no_children(self, raw_data):
        node = GatingResult(
            name="root",
            data=raw_data,
            parent_events=len(raw_data),
            gated_events=len(raw_data),
            percentage=100.0,
        )
        assert node.leaf is node

    def test_leaf_with_children(self, raw_data):
        child = GatingResult(
            name="singlet",
            data=raw_data.iloc[:100],
            parent_events=len(raw_data),
            gated_events=100,
            percentage=50.0,
        )
        root = GatingResult(
            name="root",
            data=raw_data,
            parent_events=len(raw_data),
            gated_events=len(raw_data),
            percentage=100.0,
            children=[child],
        )
        assert root.leaf is child

    def test_leaf_deep_chain(self, raw_data):
        grandchild = GatingResult(
            name="debris",
            data=raw_data.iloc[:50],
            parent_events=100,
            gated_events=50,
            percentage=50.0,
        )
        child = GatingResult(
            name="singlet",
            data=raw_data.iloc[:100],
            parent_events=200,
            gated_events=100,
            percentage=50.0,
            children=[grandchild],
        )
        root = GatingResult(
            name="root",
            data=raw_data,
            parent_events=200,
            gated_events=200,
            percentage=100.0,
            children=[child],
        )
        assert root.leaf is grandchild

    def test_default_stats_and_children(self, raw_data):
        node = GatingResult(
            name="x",
            data=raw_data,
            parent_events=1,
            gated_events=1,
            percentage=100.0,
        )
        assert node.stats == {}
        assert node.children == []


# ── apply_gating_hierarchy ───────────────────────────────────────────


class TestApplyGatingHierarchy:
    def test_returns_root_with_singlet_child(self, raw_data):
        root = apply_gating_hierarchy(raw_data, {})
        assert root.name == "root"
        assert root.parent_events == len(raw_data)
        assert len(root.children) == 1
        assert root.children[0].name == "singlet"

    def test_singlet_child_has_fewer_or_equal_events(self, raw_data):
        root = apply_gating_hierarchy(raw_data, {})
        singlet = root.children[0]
        assert singlet.gated_events <= root.parent_events

    def test_singlet_stats_keys(self, raw_data):
        root = apply_gating_hierarchy(raw_data, {})
        stats = root.children[0].stats
        assert "Singlet_Percentage" in stats
        assert "Total_Events" in stats
        assert "Singlet_Events" in stats

    def test_leaf_is_singlet(self, raw_data):
        root = apply_gating_hierarchy(raw_data, {})
        assert root.leaf.name == "singlet"

    def test_custom_singlet_thresholds(self, raw_data):
        config_narrow = {"singlet_gate": {"lower": 0.9, "upper": 1.1}}
        config_wide = {"singlet_gate": {"lower": 0.5, "upper": 2.5}}
        root_narrow = apply_gating_hierarchy(raw_data, config_narrow)
        root_wide = apply_gating_hierarchy(raw_data, config_wide)
        assert (
            root_narrow.children[0].gated_events <= root_wide.children[0].gated_events
        )

    def test_singlet_data_is_dataframe(self, raw_data):
        root = apply_gating_hierarchy(raw_data, {})
        assert isinstance(root.leaf.data, pd.DataFrame)

    def test_percentage_consistent(self, raw_data):
        root = apply_gating_hierarchy(raw_data, {})
        singlet = root.children[0]
        expected_pct = (singlet.gated_events / singlet.parent_events) * 100
        assert abs(singlet.percentage - expected_pct) < 0.01


# ── get_gating_summary ──────────────────────────────────────────────


class TestGetGatingSummary:
    def test_summary_length(self, raw_data):
        root = apply_gating_hierarchy(raw_data, {})
        summary = get_gating_summary(root)
        assert len(summary) == 2  # root + singlet

    def test_summary_fields(self, raw_data):
        root = apply_gating_hierarchy(raw_data, {})
        summary = get_gating_summary(root)
        for entry in summary:
            assert "Gate" in entry
            assert "Depth" in entry
            assert "Parent_Events" in entry
            assert "Gated_Events" in entry
            assert "Percentage" in entry

    def test_summary_depths(self, raw_data):
        root = apply_gating_hierarchy(raw_data, {})
        summary = get_gating_summary(root)
        assert summary[0]["Depth"] == 0
        assert summary[1]["Depth"] == 1

    def test_summary_names(self, raw_data):
        root = apply_gating_hierarchy(raw_data, {})
        summary = get_gating_summary(root)
        assert summary[0]["Gate"] == "root"
        assert summary[1]["Gate"] == "singlet"

    def test_summary_deep_tree(self, raw_data):
        grandchild = GatingResult(
            name="analysis",
            data=raw_data.iloc[:10],
            parent_events=50,
            gated_events=10,
            percentage=20.0,
        )
        child = GatingResult(
            name="singlet",
            data=raw_data.iloc[:50],
            parent_events=200,
            gated_events=50,
            percentage=25.0,
            children=[grandchild],
        )
        root = GatingResult(
            name="root",
            data=raw_data,
            parent_events=200,
            gated_events=200,
            percentage=100.0,
            children=[child],
        )
        summary = get_gating_summary(root)
        assert len(summary) == 3
        assert [e["Depth"] for e in summary] == [0, 1, 2]
