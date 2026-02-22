# src/ewa_html_dashboard.py
"""
SAP EWA 13-KPI Executive Dashboard with ML: trends, forecasting, and attention scoring.
"""

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from .ewa_kpi_ml import compute_trend, forecast_severity, attention_score

# Paths
DATA_DIR = Path("data")
MASTER_FILE = DATA_DIR / "ewa_kpi_clean_summary_all.csv"
LEGACY_FILE = DATA_DIR / "ewa_html_traffic_lights_summary13_A1C.csv"

COLOR_MAP = {"GREEN": "#00B050", "YELLOW": "#FFC000", "RED": "#FF0000"}
STATUS_ORDER = {"GREEN": 0, "YELLOW": 1, "RED": 2}
REV_SEVERITY = {1: "GREEN", 2: "YELLOW", 3: "RED"}
status_rank = {"GREEN": 3, "YELLOW": 2, "RED": 1}

st.set_page_config(page_title="EWA 13-KPI Executive Dashboard", layout="wide")


def load_data():
    """Load master CSV if available, else fall back to legacy A1C file. Normalize to primary_kpi."""
    if MASTER_FILE.exists():
        df = pd.read_csv(MASTER_FILE)
        df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
        df = df[df["report_date"].notna()]
        if "clean_section" in df.columns and "primary_kpi" not in df.columns:
            df["primary_kpi"] = df["clean_section"]
        return df, list(df["system"].unique()) if "system" in df.columns else []
    if LEGACY_FILE.exists():
        df = pd.read_csv(LEGACY_FILE)
        df["report_date"] = pd.to_datetime(df["report_date"])
        df["system"] = "A1C"
        return df, ["A1C"]
    return None, []


def add_deviation(df_in: pd.DataFrame, kpi_col: str = "primary_kpi") -> pd.DataFrame:
    df = df_in.copy()
    group = [kpi_col]
    if "system" in df.columns:
        group = ["system", kpi_col]
    df["prev_status"] = df.groupby(group)["status_name"].shift(1)

    def classify(row):
        cur = row["status_name"]
        prev = row["prev_status"]
        if pd.isna(prev):
            return "🆕 New"
        if status_rank.get(cur, 0) > status_rank.get(prev, 0):
            return "➕ Improvement"
        if status_rank.get(cur, 0) < status_rank.get(prev, 0):
            return "➖ Deterioration"
        return "🔄 No Change"

    df["deviation"] = df.apply(classify, axis=1)
    return df


# Load and select system
data, systems = load_data()
if data is None:
    st.error("No summary CSV found. Run: python -m src.ewa_html_processor and/or python -m src.ewa_kpi_master_builder")
    st.stop()

selected_system = None
if systems:
    selected_system = st.sidebar.selectbox("System", systems, index=len(systems) - 1)
    df_full = data[data["system"] == selected_system].copy() if "system" in data.columns else data.copy()
else:
    df_full = data.copy()

df_full = add_deviation(df_full)
dates = sorted(df_full["report_date"].unique())
kpi_order = sorted(df_full["primary_kpi"].unique().tolist(), key=lambda x: (x or ""))
df_full["primary_kpi"] = pd.Categorical(df_full["primary_kpi"], categories=kpi_order, ordered=True)
df_full = df_full.sort_values(["report_date", "primary_kpi"])

st.title(f"📊 SAP EarlyWatch – 13 KPI Executive Dashboard{f' ({selected_system})' if selected_system else ''}")

tabs = [
    "📈 Heatmap + Deviation",
    "📋 Weekly KPI Detail",
    "📉 WoW Changes",
    "🔍 Compare Two Weeks",
    "📉 Trends",
    "🔮 Forecast",
    "⚠️ KPIs Needing Attention",
]
tab1, tab2, tab3, tab4, tab_trend, tab_forecast, tab_attention = st.tabs(tabs)

# TAB 1 – Heatmap
with tab1:
    st.subheader("Heatmap with deviation in hover text")
    pivot_status = df_full.pivot(index="primary_kpi", columns="report_date", values="status_name")
    pivot_num = pivot_status.replace(STATUS_ORDER)
    pivot_dev = df_full.pivot(index="primary_kpi", columns="report_date", values="deviation")

    fig = px.imshow(
        pivot_num,
        color_continuous_scale=[
            [0.0, COLOR_MAP["GREEN"]],
            [0.5, COLOR_MAP["YELLOW"]],
            [1.0, COLOR_MAP["RED"]],
        ],
        aspect="auto",
    )
    fig.update_traces(
        text=pivot_dev.values,
        customdata=pivot_status.values,
        hovertemplate="KPI=%{y}<br>Date=%{x}<br>Status=%{customdata}<br>Deviation=%{text}<extra></extra>",
    )
    fig.update_layout(xaxis_title="Report Date", yaxis_title="KPI", coloraxis_showscale=False, height=650)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("🟢 OK | 🟡 Warning | 🔴 Critical | Hover to see deviation vs previous week.")

# TAB 2 – Weekly detail
with tab2:
    st.subheader("KPI detail for a specific report date")
    picked = st.selectbox("Pick report date", dates, index=len(dates) - 1, key="tab2_date")
    df_day = df_full[df_full["report_date"] == picked].sort_values("primary_kpi")
    cols = ["primary_kpi", "status_symbol", "status_name", "prev_status", "deviation"]
    st.dataframe(df_day[[c for c in cols if c in df_day.columns]], use_container_width=True)

# TAB 3 – WoW changes
with tab3:
    st.subheader("Week-over-week KPI changes")
    picked2 = st.selectbox("Select report date (current week)", dates, index=len(dates) - 1, key="tab3_date")
    ddf = df_full[df_full["report_date"] == picked2]
    ddf = ddf[ddf["deviation"].isin(["➕ Improvement", "➖ Deterioration"])]
    if ddf.empty:
        st.info("No KPI changes vs previous week for this date. ✔")
    else:
        st.dataframe(ddf[["primary_kpi", "prev_status", "status_name", "deviation"]], use_container_width=True)

# TAB 4 – Compare two weeks
with tab4:
    st.subheader("Compare two arbitrary report weeks")
    c1, c2 = st.columns(2)
    w1 = c1.selectbox("Week 1 (older)", dates, index=0, key="w1")
    w2 = c2.selectbox("Week 2 (newer)", dates, index=len(dates) - 1, key="w2")
    df1 = df_full[df_full["report_date"] == w1][["primary_kpi", "status_name", "status_symbol"]]
    df2 = df_full[df_full["report_date"] == w2][["primary_kpi", "status_name", "status_symbol"]]
    merged = df1.merge(df2, on="primary_kpi", suffixes=("_w1", "_w2"))

    def change(row):
        s1, s2 = row["status_name_w1"], row["status_name_w2"]
        if status_rank.get(s2, 0) > status_rank.get(s1, 0):
            return "➕ Improvement"
        if status_rank.get(s2, 0) < status_rank.get(s1, 0):
            return "➖ Deterioration"
        return "🔄 No Change"

    merged["Change"] = merged.apply(change, axis=1)
    st.dataframe(
        merged[["primary_kpi", "status_symbol_w1", "status_name_w1", "status_symbol_w2", "status_name_w2", "Change"]],
        use_container_width=True,
    )

# TAB – Trends (ML)
with tab_trend:
    st.subheader("KPI severity trends (ML)")
    st.caption("Linear trend on severity over time: improving = getting better, deteriorating = getting worse.")
    try:
        trend_df = compute_trend(data, system=selected_system, kpi_col="primary_kpi")
        if trend_df.empty:
            st.info("Not enough history to compute trends.")
        else:
            display_cols = [c for c in ["system", "kpi", "slope", "trend_direction", "n_weeks"] if c in trend_df.columns]
            trend_df = trend_df.sort_values("slope", ascending=False)
            st.dataframe(trend_df[display_cols], use_container_width=True)
            # Bar chart: trend direction counts
            dir_counts = trend_df["trend_direction"].value_counts()
            fig_t = px.bar(x=dir_counts.index, y=dir_counts.values, labels={"x": "Trend", "y": "Count"}, title="Trend direction summary")
            st.plotly_chart(fig_t, use_container_width=True)
    except Exception as e:
        st.warning(f"Trend computation failed: {e}")

# TAB – Forecast (ML)
with tab_forecast:
    st.subheader("Severity forecast (next few weeks)")
    periods = st.slider("Weeks to forecast", 2, 8, 4, key="forecast_periods")
    try:
        forecast_df = forecast_severity(
            data, system=selected_system, kpi_col="primary_kpi", periods_ahead=periods
        )
        if forecast_df.empty:
            st.info("Not enough data to forecast.")
        else:
            display_f = forecast_df.copy()
            display_f["severity_label"] = display_f["severity_pred"].map(lambda x: REV_SEVERITY.get(round(x), "?"))
            st.dataframe(display_f, use_container_width=True)
            # Line chart: forecast per KPI
            fig_f = px.line(
                forecast_df, x="report_date", y="severity_pred", color="kpi",
                title="Predicted severity (1=Green, 2=Yellow, 3=Red)", labels={"severity_pred": "Severity (predicted)"}
            )
            fig_f.update_yaxis(dtick=1)
            st.plotly_chart(fig_f, use_container_width=True)
    except Exception as e:
        st.warning(f"Forecast failed: {e}")

# TAB – KPIs Needing Attention (ML)
with tab_attention:
    st.subheader("KPIs needing special attention")
    st.caption("Ranked by attention score: current severity, deteriorating trend, % of weeks in RED, and volatility.")
    try:
        att_df = attention_score(data, system=selected_system, kpi_col="primary_kpi")
        if att_df.empty:
            st.info("No KPI attention scores available.")
        else:
            st.dataframe(att_df, use_container_width=True)
            top = att_df.head(10)
            fig_a = go.Figure(go.Bar(x=top["attention_score"], y=top["kpi"], orientation="h"))
            fig_a.update_layout(title="Top 10 KPIs by attention score", xaxis_title="Attention score", height=400)
            st.plotly_chart(fig_a, use_container_width=True)
    except Exception as e:
        st.warning(f"Attention score failed: {e}")
