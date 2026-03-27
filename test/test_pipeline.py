"""Tests for fltower.core.pipeline — step functions and utilities."""

import os

from fltower.core.pipeline import (
    NUM_COLS,
    NUM_ROWS,
    PipelineContext,
    create_output_structure,
    extract_well_key,
    make_plot_key,
    step_create_output,
    step_discover_files,
    step_prepare_config,
)

# ── extract_well_key ─────────────────────────────────────────────────


class TestExtractWellKey:
    """Ensure extract_well_key still works after its move to pipeline."""

    def test_standard(self):
        key, (letter, num) = extract_well_key("Experiment_A1.fcs")
        assert key == "A1"
        assert letter == "A"
        assert num == 1

    def test_no_match(self):
        key, (name, num) = extract_well_key("random_file.fcs")
        assert key == "random_file"
        assert num == 0


# ── make_plot_key ────────────────────────────────────────────────────


class TestMakePlotKey:
    def test_scatter(self):
        assert (
            make_plot_key({"type": "scatter", "x_param": "BL1-H", "y_param": "YL2-H"})
            == "scatter_BL1-H_YL2-H"
        )

    def test_histogram(self):
        assert (
            make_plot_key({"type": "histogram", "x_param": "BL1-H"})
            == "histogram_BL1-H_"
        )


# ── PipelineContext ──────────────────────────────────────────────────


class TestPipelineContext:
    def test_defaults(self):
        ctx = PipelineContext(
            directory="/tmp/in",
            plots_config={},
            results_directory="/tmp/out",
        )
        assert ctx.files == []
        assert ctx.plot_configs == {}
        assert ctx.singlet_lower == 0.7
        assert ctx.singlet_upper == 2.0
        assert ctx.scatter_dfs == {}
        assert ctx.histogram_dfs == {}
        assert ctx.singlet_stats == []
        assert ctx.gating_roots == []
        assert ctx.wells_with_data == set()


# ── step_discover_files ──────────────────────────────────────────────


class TestStepDiscoverFiles:
    def test_finds_and_sorts(self, tmp_path):
        # Create fake FCS files out of alphabetical order
        for name in ["Sample_B2.fcs", "Sample_A1.fcs", "Sample_A3.fcs"]:
            (tmp_path / name).write_text("")

        ctx = PipelineContext(
            directory=str(tmp_path), plots_config={}, results_directory=""
        )
        step_discover_files(ctx)

        basenames = [os.path.basename(f) for f in ctx.files]
        assert basenames == ["Sample_A1.fcs", "Sample_A3.fcs", "Sample_B2.fcs"]

    def test_empty_directory(self, tmp_path):
        ctx = PipelineContext(
            directory=str(tmp_path), plots_config={}, results_directory=""
        )
        step_discover_files(ctx)
        assert ctx.files == []

    def test_ignores_non_fcs(self, tmp_path):
        (tmp_path / "data.csv").write_text("")
        (tmp_path / "Sample_A1.fcs").write_text("")

        ctx = PipelineContext(
            directory=str(tmp_path), plots_config={}, results_directory=""
        )
        step_discover_files(ctx)
        assert len(ctx.files) == 1


# ── step_prepare_config ──────────────────────────────────────────────


class TestStepPrepareConfig:
    def test_extracts_plot_configs(self):
        config = {
            "singlet_gate": {"lower": 0.5, "upper": 1.5},
            "plot_scatter": {"type": "scatter", "x_param": "BL1-H", "y_param": "YL2-H"},
            "plot_histo": {"type": "histogram", "x_param": "BL1-H"},
        }
        ctx = PipelineContext(directory="", plots_config=config, results_directory="")
        step_prepare_config(ctx)

        assert ctx.singlet_lower == 0.5
        assert ctx.singlet_upper == 1.5
        assert set(ctx.plot_configs.keys()) == {"plot_scatter", "plot_histo"}

    def test_defaults_when_no_singlet_gate(self):
        ctx = PipelineContext(directory="", plots_config={}, results_directory="")
        step_prepare_config(ctx)
        assert ctx.singlet_lower == 0.7
        assert ctx.singlet_upper == 2.0
        assert ctx.plot_configs == {}

    def test_ignores_non_plot_keys(self):
        config = {
            "singlet_gate": {"lower": 0.7, "upper": 2.0},
            "some_string": "not a dict",
            "a_dict_without_type": {"x_param": "BL1-H"},
        }
        ctx = PipelineContext(directory="", plots_config=config, results_directory="")
        step_prepare_config(ctx)
        assert ctx.plot_configs == {}


# ── step_create_output ───────────────────────────────────────────────


class TestStepCreateOutput:
    def test_creates_directories(self, tmp_path):
        results_dir = str(tmp_path / "results")
        ctx = PipelineContext(
            directory="", plots_config={}, results_directory=results_dir
        )
        step_create_output(ctx)

        assert os.path.isdir(os.path.join(results_dir, "plots"))
        assert os.path.isdir(os.path.join(results_dir, "96well_plots"))
        assert os.path.isdir(os.path.join(results_dir, "statistics"))
        assert os.path.isdir(os.path.join(results_dir, "triplicate_statistics"))
        assert os.path.isdir(os.path.join(results_dir, "triplicate_plots"))

        assert "plots" in ctx.output_dirs
        assert "stats" in ctx.output_dirs


# ── create_output_structure ──────────────────────────────────────────


class TestCreateOutputStructure:
    def test_returns_dict(self, tmp_path):
        result = create_output_structure(str(tmp_path / "res"))
        assert isinstance(result, dict)
        assert set(result.keys()) == {
            "plots",
            "well_plots",
            "stats",
            "triplicate_stats",
            "triplicate_plots",
        }


# ── Constants ────────────────────────────────────────────────────────


def test_plate_dimensions():
    assert NUM_ROWS == 8
    assert NUM_COLS == 12
