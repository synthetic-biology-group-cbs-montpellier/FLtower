"""Page 5 — Export statistics and parameters."""

import io
import json

import pandas as pd
import streamlit as st

st.set_page_config(page_title="FLtower — Export", page_icon="💾", layout="wide")
st.title("💾  Export")

# ── Guard ─────────────────────────────────────────────────────────────

if "well_data" not in st.session_state:
    st.warning("No data loaded. Go to **📤 Upload** first.")
    st.stop()

well_data = st.session_state["well_data"]
plot_configs = st.session_state.get("plot_configs", {})
params = st.session_state.get("parameters", {})


# ── 1. Parameters.json ────────────────────────────────────────────────

st.header("1. Parameters")

st.download_button(
    "📥  Download parameters.json",
    data=json.dumps(params, indent=4),
    file_name="parameters.json",
    mime="application/json",
)

# ── 2. Per-well statistics CSV ────────────────────────────────────────

st.header("2. Per-well statistics")


def _build_full_stats_df(well_data):
    records = []
    for wk, wd in well_data.items():
        rec = dict(wd["stats"])
        records.append(rec)
    df = pd.DataFrame(records)
    if "Well" in df.columns:
        df = df.sort_values(
            "Well",
            key=lambda s: s.apply(
                lambda w: (w[0], int(w[1:]) if len(w) > 1 and w[1:].isdigit() else 0)
            ),
        )
    return df


full_df = _build_full_stats_df(well_data)

st.dataframe(full_df, use_container_width=True, height=300)

csv_buffer = io.StringIO()
full_df.to_csv(csv_buffer, index=False)

st.download_button(
    "📥  Download all statistics (CSV)",
    data=csv_buffer.getvalue(),
    file_name="fltower_statistics.csv",
    mime="text/csv",
)

# ── 3. Per-plot-config CSVs ───────────────────────────────────────────

st.header("3. Per-plot statistics")

for cfg_key, config in plot_configs.items():
    plot_type = config["type"]
    x_param = config["x_param"]
    y_param = config.get("y_param", "")
    label = f"{plot_type.capitalize()} — {x_param}"
    if y_param:
        label += f" vs {y_param}"

    prefix = f"{cfg_key}__"
    plot_cols = ["Well"] + [c for c in full_df.columns if c.startswith(prefix)]

    if len(plot_cols) <= 1:
        continue

    plot_df = full_df[plot_cols].copy()
    # Clean column names for export
    plot_df.columns = [
        c.replace(prefix, "") if c != "Well" else c for c in plot_df.columns
    ]

    with st.expander(f"📊 {label}"):
        st.dataframe(plot_df, use_container_width=True)

        buf = io.StringIO()
        plot_df.to_csv(buf, index=False)
        st.download_button(
            f"📥  Download {label} CSV",
            data=buf.getvalue(),
            file_name=f"{cfg_key}_statistics.csv",
            mime="text/csv",
            key=f"dl_{cfg_key}",
        )


# ── 4. Triplicate statistics CSV ─────────────────────────────────────

st.header("4. Triplicate statistics")


def _assign_triplicate_group(well_key):
    if len(well_key) < 2 or not well_key[0].isalpha():
        return well_key
    row = well_key[0]
    try:
        col = int(well_key[1:])
    except ValueError:
        return well_key
    group_col = ((col - 1) // 3) * 3 + 1
    return f"{row}{group_col}"


full_df["Triplicate_Group"] = full_df["Well"].apply(_assign_triplicate_group)

numeric_cols = full_df.select_dtypes(include="number").columns.tolist()
if numeric_cols:
    trip_df = (
        full_df.groupby("Triplicate_Group")[numeric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    # Flatten multi-level columns
    trip_df.columns = [f"{a}_{b}" if b else a for a, b in trip_df.columns]

    st.dataframe(trip_df, use_container_width=True, height=300)

    buf = io.StringIO()
    trip_df.to_csv(buf, index=False)
    st.download_button(
        "📥  Download triplicate statistics (CSV)",
        data=buf.getvalue(),
        file_name="fltower_triplicate_statistics.csv",
        mime="text/csv",
    )
else:
    st.info("No numeric columns available for triplicate aggregation.")
