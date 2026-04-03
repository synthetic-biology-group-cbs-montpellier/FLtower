"""Streamlit prototype — interactive histogram gating for FCS files.

Upload one or two FCS files and adjust a shared gate threshold interactively.
Two contiguous gates are derived from a single threshold slider:
  Gate 1 = [data min, threshold]   Gate 2 = [threshold, data max]

Run with::

    streamlit run streamlit_app.py
"""

import json
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st
from scipy.stats import gmean

from fltower.core.cleaning import clean_data
from fltower.core.gating.interval import compute_interval_stats
from fltower.core.gating.singlet import remove_doublets
from fltower.io.fcs_reader import read_fcs

# ── Page configuration ────────────────────────────────────────────────

st.set_page_config(page_title="FLtower — Histogram Gating", layout="wide")
st.title("FLtower — Interactive Histogram Gating")


# ── Helpers ───────────────────────────────────────────────────────────


def _load_fcs(uploaded):
    """Write an uploaded file to a temp path and parse it."""
    with tempfile.NamedTemporaryFile(suffix=".fcs", delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name
    return read_fcs(tmp_path)


def _apply_singlet(data, channels, lower, upper):
    """Optionally apply singlet gating."""
    has_ssc = "SSC-A" in channels and "SSC-H" in channels
    if not has_ssc:
        return data, None
    singlets, pct, total, n_singlets = remove_doublets(
        data, singlet_lower=lower, singlet_upper=upper
    )
    return singlets, {"pct": pct, "total": total, "n": n_singlets}


def _make_histogram(
    ax,
    values,
    channel,
    title,
    x_scale,
    show_kde,
    color,
    gates,
    gate_colors,
    gate_stats=None,
):
    """Draw a histogram with gate regions and inline statistics on *ax*."""
    sns.histplot(
        values,
        bins=100,
        kde=show_kde,
        ax=ax,
        log_scale=(x_scale == "log"),
        color=color,
    )
    y_max = ax.get_ylim()[1]
    for i, ((g_min, g_max), g_color) in enumerate(zip(gates, gate_colors)):
        ax.axvline(g_min, color=g_color, linestyle="--", linewidth=1.5)
        ax.axvline(g_max, color=g_color, linestyle="--", linewidth=1.5)
        ax.axvspan(g_min, g_max, alpha=0.08, color=g_color)
        center = (g_min + g_max) / 2 if x_scale == "linear" else np.sqrt(g_min * g_max)

        # Build label with stats if available
        label = f"Gate {i + 1}"
        if gate_stats:
            pct = gate_stats.get(f"Gate_{i + 1}_Percentage", 0)
            gm = gate_stats.get(f"Gate_{i + 1}_GM", 0)
            label += f"\n{pct:.1f}%  ·  GM {gm:,.0f}"

        ax.text(
            center,
            y_max * (0.92 - i * 0.12),
            label,
            color=g_color,
            ha="center",
            fontweight="bold",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor=g_color, alpha=0.85, pad=3),
        )

    # Global stats in top-left corner
    global_median = float(values.median())
    positive = values[values > 0]
    global_gm = float(gmean(positive)) if len(positive) > 0 else 0.0
    ax.text(
        0.02,
        0.98,
        f"n = {len(values):,}\nMedian = {global_median:,.1f}\nGM = {global_gm:,.1f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="0.3",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2),
    )

    ax.set_xlabel(channel)
    ax.set_ylabel("Count")
    ax.set_title(title, fontsize=13, fontweight="bold")


def _show_stats(container, values, channel, gates, cleaned):
    """Display global + per-gate statistics in a Streamlit container."""
    with container:
        global_median = float(values.median())
        positive = values[values > 0]
        global_gm = float(gmean(positive)) if len(positive) > 0 else 0.0

        st.metric("Events", f"{len(values):,}")
        st.metric("Median", f"{global_median:,.1f}")
        st.metric("Geometric Mean", f"{global_gm:,.1f}")

        if gates:
            gate_stats = compute_interval_stats(cleaned, channel, gates)
            for i, (g_min, g_max) in enumerate(gates):
                pct = gate_stats[f"Gate_{i + 1}_Percentage"]
                gm = gate_stats[f"Gate_{i + 1}_GM"]
                st.markdown(f"**Gate {i + 1}** ({g_min:,.0f} – {g_max:,.0f})")
                st.metric(f"G{i+1} %", f"{pct:.2f}%", label_visibility="collapsed")
                st.metric(f"G{i+1} GM", f"{gm:,.1f}", label_visibility="collapsed")


# ── File upload (1 or 2 files) ────────────────────────────────────────

uploaded_files = st.file_uploader(
    "Upload one or two FCS files",
    type=["fcs"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload one or two FCS files to get started.")
    st.stop()

if len(uploaded_files) > 2:
    st.warning("Only the first two files will be used.")
    uploaded_files = uploaded_files[:2]

# Parse files
datasets = []  # list of (name, data, channels)
for uf in uploaded_files:
    data, channels = _load_fcs(uf)
    if data is None:
        st.error(f"Failed to read **{uf.name}**. Check the file format.")
        st.stop()
    datasets.append((uf.name, data, channels))
    st.success(f"**{uf.name}** — {len(data):,} events, {len(channels)} channels")

# ── Sidebar: Singlet Gate ─────────────────────────────────────────────

st.sidebar.header("Singlet Gate")

# Check SSC availability across all files
all_channels = set(datasets[0][2])
for _, _, ch in datasets[1:]:
    all_channels &= set(ch)

has_ssc = "SSC-A" in all_channels and "SSC-H" in all_channels
apply_singlet = st.sidebar.checkbox(
    "Apply singlet gate", value=has_ssc, disabled=not has_ssc
)

singlet_lower = 0.7
singlet_upper = 2.0
if apply_singlet and has_ssc:
    singlet_lower = st.sidebar.slider("SSC-H/SSC-A lower bound", 0.1, 2.0, 0.7, 0.05)
    singlet_upper = st.sidebar.slider("SSC-H/SSC-A upper bound", 1.0, 5.0, 2.0, 0.1)

# Apply gating to each dataset
plot_datasets = []  # (name, plot_data, channels)
for name, data, channels in datasets:
    if apply_singlet and has_ssc:
        gated, info = _apply_singlet(data, channels, singlet_lower, singlet_upper)
        if info:
            st.sidebar.text(f"{name}: {info['pct']:.1f}% singlets")
        plot_datasets.append((name, gated, channels))
    else:
        plot_datasets.append((name, data, channels))

# ── Sidebar: Histogram Settings ───────────────────────────────────────

st.sidebar.header("Histogram Settings")

common_channels = sorted(all_channels)
default_ch = next(
    (c for c in common_channels if "BL1" in c or "FL1" in c),
    common_channels[0] if common_channels else None,
)

if not common_channels:
    st.error("The uploaded files share no common channels.")
    st.stop()

channel = st.sidebar.selectbox(
    "Channel",
    common_channels,
    index=common_channels.index(default_ch) if default_ch else 0,
)

x_scale = st.sidebar.radio("X scale", ["linear", "log"], index=1)
show_kde = st.sidebar.checkbox("Show KDE", value=True)
color = st.sidebar.color_picker("Histogram color", "#2e8b57")

# ── Clean data and compute global range across all files ──────────────

cleaned_list = []
for name, pdata, _ in plot_datasets:
    cl = clean_data(pdata, [channel], remove_zeros=(x_scale == "log"))
    cleaned_list.append((name, cl, cl[channel]))

all_values = np.concatenate([v.values for _, _, v in cleaned_list])
if len(all_values) == 0:
    st.error(f"No valid data in channel **{channel}** after cleaning.")
    st.stop()

val_min = float(all_values.min())
val_max = float(all_values.max())

# ── Sidebar: Gate Threshold ───────────────────────────────────────────

st.sidebar.header("Gate Threshold")
st.sidebar.caption("A single threshold defines two contiguous gates:")
st.sidebar.caption("Gate 1 = [min → threshold]  ·  Gate 2 = [threshold → max]")

if x_scale == "log" and val_min > 0:
    log_min = float(np.log10(max(val_min, 1)))
    log_max = float(np.log10(val_max))
    log_default = (log_min + log_max) / 2
    log_threshold = st.sidebar.slider(
        "Threshold (log₁₀)",
        min_value=log_min,
        max_value=log_max,
        value=log_default,
        step=0.05,
    )
    threshold = 10**log_threshold
else:
    default_thr = (val_min + val_max) / 2
    threshold = st.sidebar.slider(
        "Threshold",
        min_value=val_min,
        max_value=val_max,
        value=default_thr,
    )

st.sidebar.markdown(f"**Threshold** = {threshold:,.1f}")

gates = [(val_min, threshold), (threshold, val_max)]
gate_colors = ["#3a4cc0", "#b40326"]  # blue / red

# ── Plots ─────────────────────────────────────────────────────────────

n_files = len(cleaned_list)
fig, axes = plt.subplots(1, n_files, figsize=(8 * n_files, 5), squeeze=False)

all_gate_stats = []
for idx, (name, cleaned, values) in enumerate(cleaned_list):
    gs = compute_interval_stats(cleaned, channel, gates) if gates else {}
    all_gate_stats.append(gs)
    ax = axes[0, idx]
    _make_histogram(
        ax,
        values,
        channel,
        name,
        x_scale,
        show_kde,
        color,
        gates,
        gate_colors,
        gate_stats=gs,
    )

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ── Statistics side by side ───────────────────────────────────────────

cols = st.columns(n_files)
for idx, (name, cleaned, values) in enumerate(cleaned_list):
    with cols[idx]:
        st.subheader(name)
        _show_stats(cols[idx], values, channel, gates, cleaned)

# ── Export parameters.json snippet ────────────────────────────────────

st.divider()
st.subheader("Export Configuration")
st.markdown("Copy this into your `parameters.json` to reuse these gates in the CLI:")

config_snippet = {
    "type": "histogram",
    "x_param": channel,
    "x_scale": x_scale,
    "color": color,
    "kde": show_kde,
    "gates": [[round(g[0], 2), round(g[1], 2)] for g in gates],
}
st.code(json.dumps({"plots_config_X": config_snippet}, indent=4), language="json")
