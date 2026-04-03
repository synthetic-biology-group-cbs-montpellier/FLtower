"""Page 3 — Detailed per-well plots (scatter, histogram, singlet gate)."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st
from scipy.stats import gmean

from fltower.core.cleaning import clean_data
from fltower.core.gating.auto.resolve import (
    resolve_histogram_gates,
    resolve_quadrant_gates,
)
from fltower.core.gating.interval import compute_interval_stats
from fltower.core.gating.quadrant import compute_quadrant_stats

st.set_page_config(page_title="FLtower — Well Detail", page_icon="📊", layout="wide")
st.title("📊  Well Detail")

# ── Guard ─────────────────────────────────────────────────────────────

if "well_data" not in st.session_state:
    st.warning("No data loaded. Go to **📤 Upload** first.")
    st.stop()

well_data = st.session_state["well_data"]
plot_configs = st.session_state.get("plot_configs", {})
params = st.session_state.get("parameters", {})

# ── Well selector ─────────────────────────────────────────────────────

well_keys = sorted(
    well_data.keys(), key=lambda w: (w[0], int(w[1:]) if w[1:].isdigit() else 0)
)

preselected = st.session_state.get("selected_well")
default_idx = well_keys.index(preselected) if preselected in well_keys else 0

selected = st.selectbox("Select well", well_keys, index=default_idx)
st.session_state["selected_well"] = selected

wd = well_data[selected]
raw = wd["raw"]
singlets = wd["singlets"]
channels = wd["channels"]

st.markdown(
    f"**{wd['filename']}** — "
    f"{len(raw):,} total events → {len(singlets):,} singlets "
    f"({wd['stats'].get('Singlet_Percentage', 0):.1f}%)"
)

# ── 1. Singlet gate plot ─────────────────────────────────────────────

st.header("Singlet Gate")

if "SSC-A" in channels and "SSC-H" in channels:
    singlet_lower = params.get("singlet_gate", {}).get("lower", 0.7)
    singlet_upper = params.get("singlet_gate", {}).get("upper", 2.0)

    fig_sg, ax_sg = plt.subplots(figsize=(6, 4))
    ssc_a = raw["SSC-A"]
    ssc_h = raw["SSC-H"]

    ax_sg.scatter(ssc_a, ssc_h, s=1, alpha=0.2, c="steelblue", rasterized=True)

    # Draw singlet bounds
    xlim = ax_sg.get_xlim()
    x_line = np.linspace(max(0, xlim[0]), xlim[1], 200)
    ax_sg.plot(
        x_line,
        x_line * singlet_lower,
        "r--",
        linewidth=1,
        label=f"lower={singlet_lower}",
    )
    ax_sg.plot(
        x_line,
        x_line * singlet_upper,
        "r--",
        linewidth=1,
        label=f"upper={singlet_upper}",
    )
    ax_sg.fill_between(
        x_line,
        x_line * singlet_lower,
        x_line * singlet_upper,
        alpha=0.08,
        color="green",
    )

    ax_sg.set_xlabel("SSC-A")
    ax_sg.set_ylabel("SSC-H")
    ax_sg.set_title(f"Singlet gate — {selected}")
    ax_sg.legend(fontsize=8)
    fig_sg.tight_layout()
    st.pyplot(fig_sg)
    plt.close(fig_sg)
else:
    st.info("SSC-A / SSC-H channels not available — singlet gate not shown.")

# ── 2. Plot configs (scatter + histogram) ────────────────────────────

for cfg_key, config in plot_configs.items():
    plot_type = config["type"]
    x_param = config["x_param"]

    if plot_type == "scatter":
        y_param = config["y_param"]
        st.header(f"Scatter: {x_param} vs {y_param}")

        x_scale = config.get("x_scale", "linear")
        y_scale = config.get("y_scale", "linear")
        xlim = config.get("xlim")
        ylim = config.get("ylim")
        quadrant_gates = resolve_quadrant_gates(singlets, config)

        fig_sc, ax_sc = plt.subplots(figsize=(7, 6))

        # Clean data
        cleaned = clean_data(singlets, [x_param, y_param])
        x_vals = cleaned[x_param]
        y_vals = cleaned[y_param]

        scatter_type = config.get("scatter_type", "scatter")
        if scatter_type == "density" and len(cleaned) > 0:
            plot_x = np.clip(x_vals, 1, None) if x_scale == "log" else x_vals
            plot_y = np.clip(y_vals, 1, None) if y_scale == "log" else y_vals
            hb = ax_sc.hexbin(
                plot_x,
                plot_y,
                gridsize=config.get("gridsize", 100),
                cmap=config.get("cmap", "viridis"),
                mincnt=1,
                xscale=x_scale,
                yscale=y_scale,
            )
            fig_sc.colorbar(hb, ax=ax_sc, label="Count")
        else:
            ax_sc.scatter(
                x_vals, y_vals, s=1, alpha=0.3, c="steelblue", rasterized=True
            )
            if x_scale == "log":
                ax_sc.set_xscale("log")
            if y_scale == "log":
                ax_sc.set_yscale("log")

        # Quadrant lines
        if quadrant_gates:
            x_mid = quadrant_gates.get("x")
            y_mid = quadrant_gates.get("y")
            if x_mid is not None:
                ax_sc.axvline(x_mid, color="red", linewidth=1, linestyle="--")
            if y_mid is not None:
                ax_sc.axhline(y_mid, color="red", linewidth=1, linestyle="--")

        if xlim:
            ax_sc.set_xlim(xlim)
        if ylim:
            ax_sc.set_ylim(ylim)

        ax_sc.set_xlabel(x_param)
        ax_sc.set_ylabel(y_param)
        ax_sc.set_title(f"{selected} — {x_param} vs {y_param}")
        fig_sc.tight_layout()
        st.pyplot(fig_sc)
        plt.close(fig_sc)

        # Show quadrant stats
        if quadrant_gates:
            try:
                qstats, _, _ = compute_quadrant_stats(
                    singlets, x_param, y_param, quadrant_gates
                )
                cols = st.columns(4)
                for i, q in enumerate(["Q1", "Q2", "Q3", "Q4"]):
                    with cols[i]:
                        pct = qstats.get(f"{q}_Percentage", 0)
                        st.metric(q, f"{pct:.1f}%")
            except Exception:
                pass

    elif plot_type == "histogram":
        st.header(f"Histogram: {x_param}")

        x_scale = config.get("x_scale", "linear")
        gates = resolve_histogram_gates(singlets, config)
        color = config.get("color", "seagreen")
        kde = config.get("kde", False)
        xlim_cfg = config.get("xlim")

        cleaned = clean_data(singlets, [x_param], remove_zeros=(x_scale == "log"))
        values = cleaned[x_param]

        if len(values) == 0:
            st.warning(f"No valid data for {x_param}.")
            continue

        fig_h, ax_h = plt.subplots(figsize=(8, 4))

        sns.histplot(
            values,
            bins=100,
            kde=kde,
            ax=ax_h,
            log_scale=(x_scale == "log"),
            color=color,
        )

        # Draw gates
        gate_colors = ["#3a4cc0", "#b40326", "#228b22", "#ff8c00"]
        if gates:
            y_top = ax_h.get_ylim()[1]
            for i, (g_min, g_max) in enumerate(gates):
                gc = gate_colors[i % len(gate_colors)]
                ax_h.axvline(g_min, color=gc, linestyle="--", linewidth=1.2)
                ax_h.axvline(g_max, color=gc, linestyle="--", linewidth=1.2)
                ax_h.axvspan(g_min, g_max, alpha=0.06, color=gc)

                center = (
                    np.sqrt(g_min * g_max)
                    if x_scale == "log" and g_min > 0
                    else (g_min + g_max) / 2
                )
                ax_h.text(
                    center,
                    y_top * (0.92 - i * 0.12),
                    f"Gate {i + 1}",
                    color=gc,
                    ha="center",
                    fontweight="bold",
                    fontsize=9,
                    bbox=dict(facecolor="white", edgecolor=gc, alpha=0.8, pad=2),
                )

        # Global stats annotation
        median_val = float(values.median())
        positive = values[values > 0]
        gm_val = float(gmean(positive)) if len(positive) > 0 else 0
        ax_h.text(
            0.02,
            0.98,
            f"n={len(values):,}  Median={median_val:,.1f}  GM={gm_val:,.1f}",
            transform=ax_h.transAxes,
            va="top",
            fontsize=8,
            color="0.3",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2),
        )

        if xlim_cfg:
            ax_h.set_xlim(xlim_cfg)

        ax_h.set_xlabel(x_param)
        ax_h.set_title(f"{selected} — {x_param} histogram")
        fig_h.tight_layout()
        st.pyplot(fig_h)
        plt.close(fig_h)

        # Show interval stats
        if gates:
            try:
                istats = compute_interval_stats(cleaned, x_param, gates)
                cols = st.columns(len(gates))
                for i in range(len(gates)):
                    with cols[i]:
                        pct = istats.get(f"Gate_{i+1}_Percentage", 0)
                        gm = istats.get(f"Gate_{i+1}_GM", 0)
                        st.metric(f"Gate {i+1}", f"{pct:.1f}%")
                        st.caption(f"GM = {gm:,.1f}")
            except Exception:
                pass

# ── 3. Raw data preview ──────────────────────────────────────────────

with st.expander("View raw data"):
    st.dataframe(singlets.head(500), use_container_width=True, height=300)

# ── 4. Well stats summary ────────────────────────────────────────────

with st.expander("All statistics for this well"):
    stats = wd["stats"]
    # Format nicely
    formatted = {k: f"{v:.4f}" if isinstance(v, float) else v for k, v in stats.items()}
    st.json(formatted)
