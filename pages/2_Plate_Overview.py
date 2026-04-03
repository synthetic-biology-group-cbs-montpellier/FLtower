"""Page 2 — Interactive 96-well plate heatmap (Plotly)."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="FLtower — Plate Overview", page_icon="🔬", layout="wide")
st.title("🔬  Plate Overview")

# ── Guard: need data ──────────────────────────────────────────────────

if "well_data" not in st.session_state:
    st.warning("No data loaded. Go to **📤 Upload** first.")
    st.stop()

well_data = st.session_state["well_data"]
plot_configs = st.session_state.get("plot_configs", {})

ROWS = list("ABCDEFGH")
COLS = list(range(1, 13))


# ── Build stats DataFrame from well_data ──────────────────────────────


def _build_plate_df(well_data):
    """Flatten per-well stats into a single DataFrame with Row/Col columns."""
    records = []
    for wk, wd in well_data.items():
        rec = dict(wd["stats"])
        # Parse row/col from well key
        if len(wk) >= 2 and wk[0].isalpha():
            rec["_row"] = wk[0]
            rec["_col"] = int(wk[1:])
        records.append(rec)
    return pd.DataFrame(records)


plate_df = _build_plate_df(well_data)

# ── Metric selector ──────────────────────────────────────────────────

# Find all numeric columns as available metrics
base_metrics = ["Singlet_Percentage", "Total_Events", "Singlet_Events"]
extra_metrics = [
    c
    for c in plate_df.columns
    if c not in base_metrics
    and c not in ("Well", "_row", "_col")
    and pd.api.types.is_numeric_dtype(plate_df[c])
]

all_metrics = base_metrics + sorted(extra_metrics)


# Give human-readable labels
def _metric_label(m):
    if "__" in m:
        cfg_key, stat_name = m.split("__", 1)
        cfg = plot_configs.get(cfg_key, {})
        plot_type = cfg.get("type", "")
        params = cfg.get("x_param", "")
        if cfg.get("y_param"):
            params += f" vs {cfg['y_param']}"
        return f"{plot_type.capitalize()} {params} — {stat_name}"
    return m


metric = st.selectbox(
    "Metric to display",
    all_metrics,
    format_func=_metric_label,
)

# ── Build 8×12 matrix for heatmap ────────────────────────────────────


def _build_plate_matrix(plate_df, metric):
    """Return an 8×12 numpy array + text array for the 96-well plate."""
    z = np.full((8, 12), np.nan)
    text = [[" " for _ in range(12)] for _ in range(8)]
    hover = [[" " for _ in range(12)] for _ in range(8)]

    for _, row in plate_df.iterrows():
        if "_row" not in row or "_col" not in row:
            continue
        r = ROWS.index(row["_row"]) if row["_row"] in ROWS else -1
        c = int(row["_col"]) - 1
        if 0 <= r < 8 and 0 <= c < 12:
            val = row.get(metric, np.nan)
            z[r, c] = val
            well_key = f"{ROWS[r]}{c + 1}"
            if np.isfinite(val):
                text[r][c] = f"{val:.2f}"
                hover[r][c] = f"{well_key}<br>{_metric_label(metric)}: {val:.2f}"
            else:
                text[r][c] = "—"
                hover[r][c] = f"{well_key}<br>No data"

    return z, text, hover


z, text, hover = _build_plate_matrix(plate_df, metric)

# ── Color scale ───────────────────────────────────────────────────────

is_percentage = "Percentage" in metric
vmin = 0 if is_percentage else float(np.nanmin(z)) if np.any(np.isfinite(z)) else 0
vmax = 100 if is_percentage else float(np.nanmax(z)) if np.any(np.isfinite(z)) else 1

col_opts1, col_opts2 = st.columns(2)
with col_opts1:
    cmap = st.selectbox(
        "Color map", ["Viridis", "Inferno", "Plasma", "Cividis", "RdYlGn"]
    )
with col_opts2:
    scale_range = st.slider(
        "Color range", float(vmin), float(vmax), (float(vmin), float(vmax))
    )

# ── Plotly heatmap ────────────────────────────────────────────────────

fig = go.Figure(
    data=go.Heatmap(
        z=z,
        x=[str(c) for c in COLS],
        y=ROWS,
        text=text,
        texttemplate="%{text}",
        hovertext=hover,
        hoverinfo="text",
        colorscale=cmap,
        zmin=scale_range[0],
        zmax=scale_range[1],
        colorbar=dict(title=_metric_label(metric)),
        xgap=3,
        ygap=3,
    )
)

fig.update_layout(
    height=500,
    xaxis=dict(
        title="Column",
        side="top",
        dtick=1,
        tickmode="array",
        tickvals=list(range(12)),
        ticktext=[str(c) for c in COLS],
    ),
    yaxis=dict(
        title="Row",
        autorange="reversed",
        dtick=1,
        tickmode="array",
        tickvals=list(range(8)),
        ticktext=ROWS,
    ),
    title=dict(text=_metric_label(metric), font=dict(size=18)),
    margin=dict(t=80, b=40),
)

# Clickable heatmap
event = st.plotly_chart(
    fig, use_container_width=True, on_select="rerun", key="plate_heatmap"
)

# ── Handle click → select well ───────────────────────────────────────

if event and event.selection and event.selection.points:
    point = event.selection.points[0]
    row_idx = point.get("y", point.get("point_index", [0, 0]))
    col_idx = point.get("x", 0)

    # Plotly returns axis values
    try:
        r = row_idx if isinstance(row_idx, int) else ROWS.index(str(row_idx))
        c = col_idx if isinstance(col_idx, int) else COLS.index(int(col_idx))
    except (ValueError, IndexError):
        r, c = 0, 0

    well_key = f"{ROWS[r]}{COLS[c]}"

    if well_key in well_data:
        st.session_state["selected_well"] = well_key
        st.success(
            f"Selected well **{well_key}** — go to **📊 Well Detail** to explore."
        )
    else:
        st.info(f"Well **{well_key}** has no data.")

# ── Quick summary table ──────────────────────────────────────────────

with st.expander("View raw statistics table"):
    display_cols = ["Well", metric] if metric in plate_df.columns else ["Well"]
    show_df = plate_df[
        [c for c in plate_df.columns if not c.startswith("_")]
    ].sort_values("Well")
    st.dataframe(show_df, use_container_width=True, height=400)
