"""Page 1 — Upload FCS files and parameters.json."""

import json
import os
import tempfile

import streamlit as st

from fltower.config_schema import validate_parameters
from fltower.core.gating.auto.resolve import (
    resolve_histogram_gates,
    resolve_quadrant_gates,
)
from fltower.core.gating.hierarchy import apply_gating_hierarchy
from fltower.core.pipeline import extract_well_key
from fltower.io.fcs_reader import read_fcs

st.set_page_config(page_title="FLtower — Upload", page_icon="📤", layout="wide")
st.title("📤  Upload FCS files & configuration")

# ── 1. Upload parameters.json ────────────────────────────────────────

st.header("1. Parameters file")

params_file = st.file_uploader(
    "Upload ``parameters.json``", type=["json"], key="params_upload"
)

use_default = st.checkbox("Use built-in default parameters", value=params_file is None)

if params_file is not None:
    raw_params = json.load(params_file)
elif use_default:
    default_path = os.path.join(
        os.path.dirname(__file__), "..", "fltower", "resource", "parameters.json"
    )
    if os.path.exists(default_path):
        with open(default_path) as f:
            raw_params = json.load(f)
    else:
        st.warning("Default parameters.json not found. Please upload one.")
        st.stop()
else:
    st.info("Please upload a ``parameters.json`` file or use the defaults.")
    st.stop()

# Validate
try:
    validate_parameters(raw_params)
    st.success("Configuration validated ✓")
except Exception as exc:
    st.error(f"Invalid parameters.json: {exc}")
    st.stop()

with st.expander("View parameters"):
    st.json(raw_params)

st.session_state["parameters"] = raw_params

# ── 2. Upload FCS files ──────────────────────────────────────────────

st.header("2. FCS files")

uploaded_fcs = st.file_uploader(
    "Upload ``.fcs`` files (up to 96)",
    type=["fcs"],
    accept_multiple_files=True,
    key="fcs_upload",
)

if not uploaded_fcs:
    st.info("Upload one or more .fcs files to continue.")
    st.stop()

st.write(f"**{len(uploaded_fcs)} files** selected.")

# ── 3. Parse & gate ──────────────────────────────────────────────────


@st.cache_data(show_spinner="Parsing FCS files…")
def _parse_all_fcs(file_contents, param_json):
    """Parse each FCS file, apply gating hierarchy, compute per-well stats.

    Returns
    -------
    well_data : dict
        ``{well_key: {"raw": DataFrame, "singlets": DataFrame,
        "channels": list, "filename": str, "stats": dict}}``
    plot_configs : dict
        Plot configuration dicts keyed by ``plots_config_N``.
    """
    # Rebuild params (to resolve auto-gates etc.)
    params = param_json
    # singlet_lower = params.get("singlet_gate", {}).get("lower", 0.7)
    # singlet_upper = params.get("singlet_gate", {}).get("upper", 2.0)

    plot_configs = {
        k: v for k, v in params.items() if isinstance(v, dict) and "type" in v
    }

    well_data = {}

    for name, content in file_contents:
        # Write to temp file for fcsparser
        with tempfile.NamedTemporaryFile(suffix=".fcs", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        data, channels = read_fcs(tmp_path)
        os.unlink(tmp_path)

        if data is None:
            continue

        well_key, _ = extract_well_key(name)

        gating_root = apply_gating_hierarchy(data, params)
        singlet_node = gating_root.children[0]
        singlets = singlet_node.data

        # Compute stats per plot config
        well_stats = {
            "Well": well_key,
            "Singlet_Percentage": singlet_node.percentage,
            "Total_Events": singlet_node.parent_events,
            "Singlet_Events": singlet_node.gated_events,
        }

        for cfg_key, config in plot_configs.items():
            if config["type"] == "scatter":
                gate_stats = _compute_scatter_stats(singlets, config)
                if gate_stats:
                    well_stats.update(
                        {f"{cfg_key}__{k}": v for k, v in gate_stats.items()}
                    )

            elif config["type"] == "histogram":
                hist_stats = _compute_histogram_stats(singlets, config)
                if hist_stats:
                    well_stats.update(
                        {f"{cfg_key}__{k}": v for k, v in hist_stats.items()}
                    )

        well_data[well_key] = {
            "raw": data,
            "singlets": singlets,
            "channels": channels,
            "filename": name,
            "stats": well_stats,
        }

    return well_data, plot_configs


def _compute_scatter_stats(singlets, config):
    """Compute quadrant stats without plotting."""
    from fltower.core.gating.quadrant import compute_quadrant_stats

    x_param = config["x_param"]
    y_param = config["y_param"]
    quadrant_gates = resolve_quadrant_gates(singlets, config)
    try:
        stats, _, _ = compute_quadrant_stats(singlets, x_param, y_param, quadrant_gates)
        return stats
    except Exception:
        return None


def _compute_histogram_stats(singlets, config):
    """Compute interval stats without plotting."""
    from fltower.core.gating.interval import compute_interval_stats

    x_param = config["x_param"]
    gates = resolve_histogram_gates(singlets, config)
    try:
        return compute_interval_stats(singlets, x_param, gates)
    except Exception:
        return None


# ── Run parsing ───────────────────────────────────────────────────────

if st.button("🔬  Run Analysis", type="primary", use_container_width=True):
    # Prepare serialisable inputs for cache
    file_contents = [(f.name, f.read()) for f in uploaded_fcs]
    param_json = st.session_state["parameters"]

    well_data, plot_configs = _parse_all_fcs(file_contents, param_json)

    if not well_data:
        st.error("No FCS files could be parsed.")
        st.stop()

    st.session_state["well_data"] = well_data
    st.session_state["plot_configs"] = plot_configs
    st.session_state["selected_well"] = None

    st.success(
        f"**{len(well_data)} wells** parsed successfully! "
        "Navigate to **🔬 Plate Overview**."
    )

# Show status if already loaded
if "well_data" in st.session_state:
    n = len(st.session_state["well_data"])
    st.sidebar.success(f"{n} wells loaded")
