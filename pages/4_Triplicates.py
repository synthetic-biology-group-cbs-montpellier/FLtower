"""Page 4 — Triplicate statistics and bar charts."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="FLtower — Triplicates", page_icon="📈", layout="wide")
st.title("📈  Triplicate Statistics")

# ── Guard ─────────────────────────────────────────────────────────────

if "well_data" not in st.session_state:
    st.warning("No data loaded. Go to **📤 Upload** first.")
    st.stop()

well_data = st.session_state["well_data"]
plot_configs = st.session_state.get("plot_configs", {})


# ── Build triplicate groups ──────────────────────────────────────────


def _assign_triplicate_group(well_key):
    """Group wells in consecutive triplets: A1-A3 → 'A1', A4-A6 → 'A4', etc."""
    if len(well_key) < 2 or not well_key[0].isalpha():
        return well_key
    row = well_key[0]
    try:
        col = int(well_key[1:])
    except ValueError:
        return well_key
    group_col = ((col - 1) // 3) * 3 + 1
    return f"{row}{group_col}"


def _build_stats_df(well_data, plot_configs):
    """Build a flat DataFrame with all per-well stats."""
    records = []
    for wk, wd in well_data.items():
        rec = dict(wd["stats"])
        rec["Triplicate_Group"] = _assign_triplicate_group(wk)
        records.append(rec)
    return pd.DataFrame(records)


stats_df = _build_stats_df(well_data, plot_configs)

if len(stats_df) == 0:
    st.info("No statistics available.")
    st.stop()

# ── Metric selector ──────────────────────────────────────────────────

numeric_cols = [
    c
    for c in stats_df.columns
    if c not in ("Well", "Triplicate_Group", "_row", "_col")
    and pd.api.types.is_numeric_dtype(stats_df[c])
]


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
    "Metric to aggregate",
    numeric_cols,
    format_func=_metric_label,
)

# ── Triplicate aggregation ────────────────────────────────────────────

grouped = (
    stats_df.groupby("Triplicate_Group")[metric]
    .agg(["mean", "std", "count"])
    .reset_index()
    .rename(columns={"mean": "Mean", "std": "Std", "count": "N"})
    .sort_values(
        "Triplicate_Group",
        key=lambda s: s.apply(lambda w: (w[0], int(w[1:]) if w[1:].isdigit() else 0)),
    )
)

# ── Bar chart with error bars ─────────────────────────────────────────

st.subheader(f"Triplicate means — {_metric_label(metric)}")

fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=grouped["Triplicate_Group"],
        y=grouped["Mean"],
        error_y=dict(type="data", array=grouped["Std"].fillna(0).tolist()),
        marker_color="#2e8b57",
        text=grouped["Mean"].apply(lambda v: f"{v:.2f}"),
        textposition="outside",
    )
)

is_pct = "Percentage" in metric
fig.update_layout(
    xaxis_title="Triplicate Group",
    yaxis_title=_metric_label(metric),
    yaxis=dict(range=[0, 105] if is_pct else None),
    height=450,
    margin=dict(t=40, b=40),
)

st.plotly_chart(fig, use_container_width=True)

# ── Detailed table ────────────────────────────────────────────────────

st.subheader("Triplicate summary table")

# Show individual values alongside aggregation
detail_df = stats_df[["Well", "Triplicate_Group", metric]].sort_values(
    "Well",
    key=lambda s: s.apply(
        lambda w: (w[0], int(w[1:]) if len(w) > 1 and w[1:].isdigit() else 0)
    ),
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Individual wells**")
    st.dataframe(detail_df, use_container_width=True, height=400)

with col2:
    st.markdown("**Aggregated triplicates**")
    st.dataframe(grouped, use_container_width=True, height=400)

# ── Multi-metric comparison ───────────────────────────────────────────

with st.expander("Compare multiple metrics"):
    selected_metrics = st.multiselect(
        "Select metrics to compare",
        numeric_cols,
        default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols,
        format_func=_metric_label,
    )

    if selected_metrics:
        multi_grouped = (
            stats_df.groupby("Triplicate_Group")[selected_metrics].mean().reset_index()
        )

        # Melt for grouped bar chart
        melted = multi_grouped.melt(
            id_vars="Triplicate_Group",
            value_vars=selected_metrics,
            var_name="Metric",
            value_name="Value",
        )
        melted["Metric_Label"] = melted["Metric"].map(_metric_label)

        fig_multi = px.bar(
            melted,
            x="Triplicate_Group",
            y="Value",
            color="Metric_Label",
            barmode="group",
            labels={"Value": "Mean", "Triplicate_Group": "Triplicate"},
            height=500,
        )
        st.plotly_chart(fig_multi, use_container_width=True)
