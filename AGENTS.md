# Project Guidelines

## Overview

FLtower is a Python CLI for 96-well plate flow cytometry analysis. It reads `.fcs` files, applies gating hierarchies, computes statistics, and produces plate-view plots and CSV exports.

Entry point: `fltower=fltower.main_fltower:main`

## Architecture

```
fltower/
  main_fltower.py        # CLI entry point, logging setup, thin wrapper
  core/
    pipeline.py           # 10-step orchestrator (PipelineContext dataclass)
    cleaning.py           # Data cleaning (NaN/inf removal)
    statistics.py         # Descriptive stats, triplicate aggregation
    gating/
      hierarchy.py        # GatingResult tree, apply_gating_hierarchy
      singlet.py          # Doublet removal (SSC-H/SSC-A ratio)
      quadrant.py         # Quadrant gating for scatter plots
      interval.py         # Interval gating for histograms
      auto/
        otsu.py           # Otsu thresholding (linear + log-space)
        resolve.py        # Resolve "auto" config → numeric gates
  io/
    fcs_reader.py         # FCS file parsing (fcsparser wrapper)
    export.py             # CSV export helpers
  plotting/
    scatter.py, histogram.py, singlet.py, plate_view.py,
    triplicate.py, report.py, _helpers.py
  config_schema.py        # Pydantic v2 models for parameters.json validation
  data_manager.py         # Load/save parameters.json
  run_args.py             # argparse CLI arguments
test/
  reference_input/        # FCS files + parameters.json for integration test
  reference_output/       # Expected CSVs (byte-for-byte comparison)
```

## Code Style

- Python 3.11, formatted with **black** (line-length 88) and **isort** (profile: black)
- Linted with **flake8** (E501/W503/E203 ignored)
- Type hints encouraged but not enforced everywhere
- Logging via `logging.getLogger("fltower")` — never `print()`

## Build and Test

```bash
# Environment
conda activate fltower
pip install -e .

# Run tests (includes coverage check ≥ 50%)
pytest test/ -v

# Run a single test file
pytest test/test_pipeline.py -v

# Lint
flake8 fltower/
```

## Conventions

- **Pipeline pattern**: Processing logic lives in `core/pipeline.py` as composable `step_*` functions sharing a `PipelineContext`. Do not add logic directly to `main_fltower.py`.
- **Gating hierarchy**: New gates go through `GatingResult` nodes in `core/gating/hierarchy.py`.
- **Config validation**: All config changes must update `config_schema.py` (Pydantic v2 models) and pass existing schema tests.
- **Backward compat**: `extract_well_key` and `create_output_structure` are re-exported from `main_fltower.py` — do not remove these imports.
- **Integration test**: `test_main_fltower.py::test_normal_pipeline` compares outputs byte-for-byte against `test/reference_output/`. Any change to output format requires updating reference files.
- **No matplotlib at import time**: `matplotlib.pyplot` is only imported in modules that actually plot (pipeline.py, plotting/*), not in main_fltower.py.
