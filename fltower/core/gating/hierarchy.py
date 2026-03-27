"""Hierarchical gating — orchestrate gates in cascade with traceability."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from fltower.core.gating.singlet import (
    SINGLET_RATIO_LOWER,
    SINGLET_RATIO_UPPER,
    remove_doublets,
)

logger = logging.getLogger("fltower")


@dataclass
class GatingResult:
    """A node in the gating hierarchy tree.

    Each node holds the filtered data produced by one gating step,
    together with event counts for traceability.  Children represent
    subsequent gates applied on this node's data.
    """

    name: str
    data: pd.DataFrame
    parent_events: int
    gated_events: int
    percentage: float
    stats: dict = field(default_factory=dict)
    children: list[GatingResult] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def leaf(self) -> GatingResult:
        """Return the deepest single-child descendant (the final gate)."""
        node = self
        while node.children:
            node = node.children[0]
        return node


def _apply_singlet_gate(
    data: pd.DataFrame,
    singlet_lower: float,
    singlet_upper: float,
) -> GatingResult:
    """Apply singlet gating and wrap the outcome in a *GatingResult*."""
    singlets, pct, total, n_singlets = remove_doublets(
        data,
        singlet_lower=singlet_lower,
        singlet_upper=singlet_upper,
    )
    return GatingResult(
        name="singlet",
        data=singlets,
        parent_events=total,
        gated_events=n_singlets,
        percentage=pct,
        stats={
            "Singlet_Percentage": pct,
            "Total_Events": total,
            "Singlet_Events": n_singlets,
        },
    )


def apply_gating_hierarchy(
    data: pd.DataFrame,
    plots_config: dict,
) -> GatingResult:
    """Apply the full gating hierarchy to raw FCS data.

    Current cascade::

        root (raw) → singlet gate

    Returns the **root** :class:`GatingResult`.  Call ``root.leaf`` to
    obtain the terminal (most-filtered) data ready for analysis.
    """
    singlet_cfg = plots_config.get("singlet_gate", {})
    singlet_lower = singlet_cfg.get("lower", SINGLET_RATIO_LOWER)
    singlet_upper = singlet_cfg.get("upper", SINGLET_RATIO_UPPER)

    root = GatingResult(
        name="root",
        data=data,
        parent_events=len(data),
        gated_events=len(data),
        percentage=100.0,
    )

    singlet_result = _apply_singlet_gate(data, singlet_lower, singlet_upper)
    root.children.append(singlet_result)

    logger.debug(
        "Gating hierarchy: %d raw → %d singlets (%.1f%%)",
        root.parent_events,
        singlet_result.gated_events,
        singlet_result.percentage,
    )

    return root


def get_gating_summary(root: GatingResult) -> list[dict]:
    """Flatten the gating tree into a list of dicts for reporting.

    Each dict contains *Gate*, *Depth*, *Parent_Events*, *Gated_Events*
    and *Percentage*.
    """
    summary: list[dict] = []
    _collect_summary(root, summary, depth=0)
    return summary


def _collect_summary(
    node: GatingResult,
    summary: list[dict],
    depth: int,
) -> None:
    summary.append(
        {
            "Gate": node.name,
            "Depth": depth,
            "Parent_Events": node.parent_events,
            "Gated_Events": node.gated_events,
            "Percentage": node.percentage,
        }
    )
    for child in node.children:
        _collect_summary(child, summary, depth + 1)
