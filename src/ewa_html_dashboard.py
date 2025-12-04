# src/ewa_html_dashboard.py

import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path

DATA_FILE = Path("data/ewa_html_traffic_lights_summary13_A1C.csv")

COLOR_MAP = {"GREEN": "#00B050", "YELLOW": "#FFC000", "RED": "#FF0000"}
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

st.set_page_config(page_title="EWA 13-KPI Executive Dashboard", layout="wide")
st.title("📊 SAP EarlyWatch – 13 KPI Executive Dashboard (A1C)")

if not DATA_FILE.exists():
    st.error(f"Summary CSV not found: {DATA_FILE}\n\nRun: python -m src.ewa_html_processor")
    st.stop()

df = pd.read_csv(DATA_FILE)
df["report_date"] = pd.to_datetime(df["report_date"])
df["primary_kpi"] = pd.Categorical(df["primary_kpi"], PRIMARY_KPI_ORDER, ordered=True)
df = df.sort_values(["report_date", "primary_kpi"])

status_rank = {"GREEN": 3, "YELLOW": 2, "RED": 1}


def add_deviation(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    df["prev_status"] = df.groupby("primary_kpi")["status_name"].shift(1)

    def classify(row):
        cur = row["status_name"]
        prev = row["prev_status"]
        if pd.isna(prev):
            return "🆕 New"
        if status_rank[cur] > status_rank[prev]:
            return "➕ Improvement"
        if status_rank[cur] < status_rank[prev]:
            return "➖ Deterioration"
        return "🔄 No Change"

    df["deviation"] = df.apply(classify, axis=1)
    return df


df = add_deviation(df)
dates = sorted(df["report_date"].unique())

tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Heatmap + Deviation", "📋 Weekly KPI Detail", "📉 WoW Changes", "🔍 Compare Two Weeks"]
)

# TAB 1 – Heatmap with deviation tooltip
with tab1:
    st.subheader("Heatmap with deviation in hover text")

    pivot_status = df.pivot(index="primary_kpi", columns="report_date", values="status_name")
    pivot_num = pivot_status.replace(STATUS_ORDER)
    pivot_dev = df.pivot(index="primary_kpi", columns="report_date", values="deviation")

    fig = px.imshow(
        pivot_num,
        color_continuous_scale=[
            [0.0, COLOR_MAP["GREEN"]],
            [0.5, COLOR_MAP["YELLOW"]],
            [1.0, COLOR_MAP["RED"]],
        ],
        aspect="auto",
    )

    # Attach hover text
    fig.update_traces(
        text=pivot_dev.values,
        customdata=pivot_status.values,
        hovertemplate="KPI=%{y}<br>Date=%{x}<br>Status=%{customdata}<br>Deviation=%{text}<extra></extra>",
    )

    fig.update_layout(
        xaxis_title="Report Date",
        yaxis_title="KPI",
        coloraxis_showscale=False,
        height=650,
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption("🟢 OK | 🟡 Warning | 🔴 Critical | Hover to see deviation vs previous week.")

# TAB 2 – Weekly detail
with tab2:
    st.subheader("KPI detail for a specific report date")
    picked = st.selectbox("Pick report date", dates, index=len(dates) - 1)
    df_day = df[df["report_date"] == picked].sort_values("primary_kpi")
    st.dataframe(
        df_day[["primary_kpi", "status_symbol", "status_name", "prev_status", "deviation"]],
        use_container_width=True,
    )

# TAB 3 – Only changed KPIs week-over-week
with tab3:
    st.subheader("Week-over-week KPI changes")
    picked2 = st.selectbox("Select report date (current week)", dates, index=len(dates) - 1)
    ddf = df[df["report_date"] == picked2]
    ddf = ddf[ddf["deviation"].isin(["➕ Improvement", "➖ Deterioration"])]
    if ddf.empty:
        st.info("No KPI changes vs previous week for this date. ✔")
    else:
        st.dataframe(
            ddf[["primary_kpi", "prev_status", "status_name", "deviation"]],
            use_container_width=True,
        )

# TAB 4 – Compare any two weeks
with tab4:
    st.subheader("Compare two arbitrary report weeks")

    c1, c2 = st.columns(2)
    w1 = c1.selectbox("Week 1 (older)", dates, index=0)
    w2 = c2.selectbox("Week 2 (newer)", dates, index=len(dates) - 1)

    df1 = df[df["report_date"] == w1][["primary_kpi", "status_name", "status_symbol"]]
    df2 = df[df["report_date"] == w2][["primary_kpi", "status_name", "status_symbol"]]

    merged = df1.merge(df2, on="primary_kpi", suffixes=("_w1", "_w2"))

    def change(row):
        s1 = row["status_name_w1"]
        s2 = row["status_name_w2"]
        if status_rank[s2] > status_rank[s1]:
            return "➕ Improvement"
        if status_rank[s2] < status_rank[s1]:
            return "➖ Deterioration"
        return "🔄 No Change"

    merged["Change"] = merged.apply(change, axis=1)

    st.dataframe(
        merged[
            [
                "primary_kpi",
                "status_symbol_w1",
                "status_name_w1",
                "status_symbol_w2",
                "status_name_w2",
                "Change",
            ]
        ],
        use_container_width=True,
    )
