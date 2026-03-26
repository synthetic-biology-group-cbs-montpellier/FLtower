"""Tests for config_schema validation."""

import pytest

from fltower.config_schema import validate_parameters


@pytest.fixture
def valid_scatter_config():
    return {
        "plots_config_1": {
            "type": "scatter",
            "x_param": "BL1-H",
            "y_param": "YL2-H",
            "x_scale": "log",
            "y_scale": "log",
            "xlim": [1, 200000],
            "ylim": [1, 200000],
            "cmap": "inferno",
            "gridsize": 100,
            "scatter_type": "density",
            "quadrant_gates": {"x": 2600, "y": 2000},
            "96well_plots": [
                {"metric": "Q3_Percentage", "title": "OFF cells"},
            ],
            "triplicate_plots": [
                {"metric": "Q3_Percentage", "title": "OFF cells"},
            ],
        }
    }


@pytest.fixture
def valid_histogram_config():
    return {
        "plots_config_2": {
            "type": "histogram",
            "x_param": "BL1-H",
            "x_scale": "log",
            "xlim": [1, 200000],
            "color": "seagreen",
            "kde": True,
            "gates": [[10, 800], [800, 100000]],
            "96well_plots": [
                {"metric": "Global_Median", "title": "GFP Median"},
            ],
            "triplicate_plots": [
                {"metric": "Global_Median", "title": "GFP-Median"},
            ],
        }
    }


class TestValidConfigs:
    def test_valid_scatter(self, valid_scatter_config):
        result = validate_parameters(valid_scatter_config)
        assert result["plots_config_1"] == valid_scatter_config["plots_config_1"]
        assert "singlet_gate" in result

    def test_valid_histogram(self, valid_histogram_config):
        result = validate_parameters(valid_histogram_config)
        assert result["plots_config_2"] == valid_histogram_config["plots_config_2"]
        assert "singlet_gate" in result

    def test_valid_mixed(self, valid_scatter_config, valid_histogram_config):
        combined = {**valid_scatter_config, **valid_histogram_config}
        result = validate_parameters(combined)
        assert result["plots_config_1"] == valid_scatter_config["plots_config_1"]
        assert result["plots_config_2"] == valid_histogram_config["plots_config_2"]
        assert "singlet_gate" in result

    def test_scatter_minimal(self):
        """Scatter with only required fields."""
        config = {
            "my_plot": {
                "type": "scatter",
                "x_param": "FSC-A",
                "y_param": "SSC-A",
            }
        }
        result = validate_parameters(config)
        assert result["my_plot"] == config["my_plot"]
        assert "singlet_gate" in result

    def test_histogram_minimal(self):
        """Histogram with only required fields."""
        config = {
            "my_plot": {
                "type": "histogram",
                "x_param": "BL1-H",
            }
        }
        result = validate_parameters(config)
        assert result["my_plot"] == config["my_plot"]
        assert "singlet_gate" in result

    def test_empty_config(self):
        """Empty dict is valid (no plots configured)."""
        result = validate_parameters({})
        assert "singlet_gate" in result


class TestInvalidConfigs:
    def test_unknown_type(self):
        config = {"p1": {"type": "boxplot", "x_param": "BL1-H"}}
        with pytest.raises(ValueError, match="Invalid parameters.json"):
            validate_parameters(config)

    def test_scatter_missing_y_param(self):
        config = {"p1": {"type": "scatter", "x_param": "BL1-H"}}
        with pytest.raises(ValueError, match="Invalid parameters.json"):
            validate_parameters(config)

    def test_histogram_missing_x_param(self):
        config = {"p1": {"type": "histogram"}}
        with pytest.raises(ValueError, match="Invalid parameters.json"):
            validate_parameters(config)

    def test_invalid_scale(self):
        config = {
            "p1": {
                "type": "histogram",
                "x_param": "BL1-H",
                "x_scale": "exponential",
            }
        }
        with pytest.raises(ValueError, match="Invalid parameters.json"):
            validate_parameters(config)

    def test_invalid_gridsize_zero(self):
        config = {
            "p1": {
                "type": "scatter",
                "x_param": "BL1-H",
                "y_param": "SSC-A",
                "gridsize": 0,
            }
        }
        with pytest.raises(ValueError, match="Invalid parameters.json"):
            validate_parameters(config)

    def test_invalid_scatter_type(self):
        config = {
            "p1": {
                "type": "scatter",
                "x_param": "BL1-H",
                "y_param": "SSC-A",
                "scatter_type": "heatmap",
            }
        }
        with pytest.raises(ValueError, match="Invalid parameters.json"):
            validate_parameters(config)

    def test_invalid_well_plot_spec(self):
        config = {
            "p1": {
                "type": "histogram",
                "x_param": "BL1-H",
                "96well_plots": [{"metric": "Median"}],  # missing title
            }
        }
        with pytest.raises(ValueError, match="Invalid parameters.json"):
            validate_parameters(config)

    def test_quadrant_gates_missing_y(self):
        config = {
            "p1": {
                "type": "scatter",
                "x_param": "BL1-H",
                "y_param": "SSC-A",
                "quadrant_gates": {"x": 100},
            }
        }
        with pytest.raises(ValueError, match="Invalid parameters.json"):
            validate_parameters(config)


class TestSingletGateConfig:
    def test_default_injected_when_absent(self):
        config = {"p1": {"type": "histogram", "x_param": "BL1-H"}}
        result = validate_parameters(config)
        assert "singlet_gate" in result
        assert result["singlet_gate"]["lower"] == pytest.approx(0.7)
        assert result["singlet_gate"]["upper"] == pytest.approx(2.0)

    def test_custom_thresholds(self):
        config = {
            "singlet_gate": {"lower": 0.5, "upper": 3.0},
            "p1": {"type": "histogram", "x_param": "BL1-H"},
        }
        result = validate_parameters(config)
        assert result["singlet_gate"]["lower"] == pytest.approx(0.5)
        assert result["singlet_gate"]["upper"] == pytest.approx(3.0)

    def test_partial_override_lower(self):
        config = {
            "singlet_gate": {"lower": 0.5, "upper": 2.0},
            "p1": {"type": "histogram", "x_param": "BL1-H"},
        }
        result = validate_parameters(config)
        assert result["singlet_gate"]["lower"] == pytest.approx(0.5)
        assert result["singlet_gate"]["upper"] == pytest.approx(2.0)

    def test_invalid_lower_ge_upper(self):
        config = {
            "singlet_gate": {"lower": 3.0, "upper": 1.0},
            "p1": {"type": "histogram", "x_param": "BL1-H"},
        }
        with pytest.raises(ValueError, match="Invalid parameters.json"):
            validate_parameters(config)

    def test_invalid_negative_lower(self):
        config = {
            "singlet_gate": {"lower": -0.5, "upper": 2.0},
            "p1": {"type": "histogram", "x_param": "BL1-H"},
        }
        with pytest.raises(ValueError, match="Invalid parameters.json"):
            validate_parameters(config)

    def test_invalid_equal(self):
        config = {
            "singlet_gate": {"lower": 1.0, "upper": 1.0},
            "p1": {"type": "histogram", "x_param": "BL1-H"},
        }
        with pytest.raises(ValueError, match="Invalid parameters.json"):
            validate_parameters(config)
