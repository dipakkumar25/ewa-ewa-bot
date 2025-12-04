# src/ewa_heatmap_app.py

import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path

DATA_FILE = Path("data/ewa_html_traffic_lights_summary13_A1C.csv")

COLOR_MAP = {
    "GREEN": "#00B050",
    "YELLOW": "#FFC000",
    "RED": "#FF0000",
}
STATUS_ORDER = {"GREEN": 0, "YELLOW": 1, "RED": 2}

PRIMARY_KPI_ORDER = [
    "Service summary",
    "Service Data Quality and Service Readiness",
    "Software Configuration for A1C",
    "Hardware Capacity",
    "Performance Overview A1C",
    "SAP System Operating A1C",
    "Security",
    "Software Change and Transport Management of A1C",
    "Financial Data Quality",
    "Upgrade Planning",
    "SAP HANA Database A1H",
    "SAP Netwear gateway",
    "UI Technologies checks",
]

st.set_page_config(page_title="EWA KPI Heatmap (13 KPIs)", layout="wide")
st.title("📊 SAP EarlyWatch – 13 KPI Heatmap (A1C)")

if not DATA_FILE.exists():
    st.error(f"Summary CSV not found: {DATA_FILE}\n\nRun: python -m src.ewa_html_processor")
    st.stop()

df = pd.read_csv(DATA_FILE)
df["report_date"] = pd.to_datetime(df["report_date"])
df["primary_kpi"] = pd.Categorical(df["primary_kpi"], PRIMARY_KPI_ORDER, ordered=True)

tab1, tab2 = st.tabs(["📈 Heatmap Overview", "📋 Daily KPI View"])

with tab1:
    pivot = df.pivot(index="primary_kpi", columns="report_date", values="status_name")
    numeric = pivot.replace(STATUS_ORDER)

    fig = px.imshow(
        numeric,
        color_continuous_scale=[
            [0.0, COLOR_MAP["GREEN"]],
            [0.5, COLOR_MAP["YELLOW"]],
            [1.0, COLOR_MAP["RED"]],
        ],
        aspect="auto",
    )
    fig.update_layout(
        xaxis_title="Report Date",
        yaxis_title="KPI",
        coloraxis_showscale=False,
        height=600,
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("🟢 = OK  🟡 = Warning  🔴 = Critical")

with tab2:
    st.subheader("Select a report date")
    selected_date = st.selectbox(
        "Report Date",
        sorted(df["report_date"].unique(), reverse=True),
        index=0,
    )
    df_day = df[df["report_date"] == selected_date].sort_values("primary_kpi")
    df_day_view = df_day[["primary_kpi", "status_symbol", "status_name", "source_file"]]

    st.dataframe(df_day_view, hide_index=True, use_container_width=True)

    csv_export = df_day_view.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download CSV for selected date",
        data=csv_export,
        file_name=f"EWA_A1C_{selected_date.date()}.csv",
        mime="text/csv",
    )
