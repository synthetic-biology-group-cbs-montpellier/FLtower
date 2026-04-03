"""FLtower — Multi-page Streamlit app for 96-well plate flow cytometry.

Run with::

    streamlit run streamlit_app.py

Architecture
------------
- This file is the **Home** page.
- Additional pages live in ``pages/`` and are auto-discovered by Streamlit.
- Shared state (parsed FCS data, parameters) is stored in ``st.session_state``.
"""

import streamlit as st

st.set_page_config(
    page_title="FLtower",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔬 FLtower — Flow Cytometry Analysis")

st.markdown("""
Welcome to the **FLtower** interactive analysis app.

### Workflow

1. **📤 Upload** — Load your FCS files and ``parameters.json`` configuration.
2. **🔬 Plate Overview** — Explore a 96-well heatmap of any metric.
   Click a well to drill down.
3. **📊 Well Detail** — View scatter plots, histograms, and gating
   for a selected well.
4. **📈 Triplicates** — Aggregate statistics over triplicates with bar charts.
5. **💾 Export** — Download CSVs and parameters.

Use the **sidebar** to navigate between pages.
""")

# Show current session status
if "well_data" in st.session_state:
    n = len(st.session_state["well_data"])
    st.success(f"**{n} wells** loaded and ready for analysis.")
    st.info("Navigate to **🔬 Plate Overview** in the sidebar to start exploring.")
else:
    st.info("Start by going to **📤 Upload** in the sidebar.")
