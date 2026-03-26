"""Pydantic models for validating parameters.json configuration."""

from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


class WellPlotSpec(BaseModel):
    """Specification for a 96-well or triplicate plot."""

    metric: str
    title: str


class QuadrantGates(BaseModel):
    """Quadrant gate positions for scatter plots."""

    x: float
    y: float


class ScatterPlotConfig(BaseModel):
    """Configuration for a scatter plot."""

    type: Literal["scatter"]
    x_param: str
    y_param: str
    x_scale: Literal["linear", "log"] = "linear"
    y_scale: Literal["linear", "log"] = "linear"
    xlim: Optional[list[float]] = None
    ylim: Optional[list[float]] = None
    cmap: str = "viridis"
    gridsize: int = Field(default=100, gt=0)
    scatter_type: Literal["scatter", "density"] = "scatter"
    quadrant_gates: Optional[QuadrantGates] = None
    well_plots: list[WellPlotSpec] = Field(default_factory=list, alias="96well_plots")
    triplicate_plots: list[WellPlotSpec] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class HistogramPlotConfig(BaseModel):
    """Configuration for a histogram plot."""

    type: Literal["histogram"]
    x_param: str
    x_scale: Literal["linear", "log"] = "linear"
    xlim: Optional[list[float]] = None
    color: str = "blue"
    kde: bool = False
    gates: Optional[list[list[float]]] = None
    well_plots: list[WellPlotSpec] = Field(default_factory=list, alias="96well_plots")
    triplicate_plots: list[WellPlotSpec] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


PlotConfig = Union[ScatterPlotConfig, HistogramPlotConfig]


class ParametersConfig(BaseModel):
    """Root model: a dict of named plot configurations.

    Accepts any key matching ``plots_config_*`` pattern.
    Unknown keys are rejected.
    """

    configs: dict[str, PlotConfig]

    @model_validator(mode="before")
    @classmethod
    def _wrap_raw_dict(cls, data):
        """Accept the raw JSON dict (keys are plot config names) and wrap it."""
        if isinstance(data, dict) and "configs" not in data:
            return {"configs": data}
        return data


def validate_parameters(raw: dict) -> dict:
    """Validate a raw parameters dict against the schema.

    Parameters
    ----------
    raw : dict
        The raw dict loaded from JSON.

    Returns
    -------
    dict
        The original *raw* dict (unchanged), if validation passes.

    Raises
    ------
    ValueError
        If validation fails, with a human-readable error message.
    """
    try:
        ParametersConfig.model_validate(raw)
    except Exception as e:
        raise ValueError(f"Invalid parameters.json:\n{e}") from e
    return raw
