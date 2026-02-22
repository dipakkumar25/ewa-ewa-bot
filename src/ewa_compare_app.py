"""
Intelligent SAP EWA – KPI Risk & Comparison Dashboard (ML Enhanced)
------------------------------------------------------------------
Adds:
• Logistic Regression (Risk Prediction)
• Linear Regression (Trend Forecasting)
• Time-Series Early Warning Signals
• KPI Trends (per-KPI slope: improving / deteriorating / stable)
• Severity Forecast (ExponentialSmoothing time series)
• Attention Score (rank KPIs needing special attention)
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

try:
    from .ewa_kpi_ml import compute_trend, forecast_severity, attention_score
except ImportError:
    from ewa_kpi_ml import compute_trend, forecast_severity, attention_score

# =========================
# CONFIG
# =========================
DATA_FILE = Path("data/ewa_kpi_clean_summary_all.csv")

STATUS_MAP = {"GREEN": 0, "YELLOW": 1, "RED": 2}
REVERSE_STATUS = {0: "GREEN", 1: "YELLOW", 2: "RED"}

st.set_page_config(page_title="Intelligent SAP EWA Dashboard", layout="wide")
st.title("📊 Intelligent SAP EWA – KPI Risk & Comparison Dashboard")

# =========================
# LOAD DATA
# =========================
if not DATA_FILE.exists():
    st.error(f"File not found: {DATA_FILE}")
    st.stop()

df = pd.read_csv(DATA_FILE)
df["report_date"] = pd.to_datetime(df["report_date"], dayfirst=True, errors="coerce")
df = df[df["report_date"].notna()]

# Normalize columns
df["final_status"] = df.get("final_status", df.get("status_name"))
df["severity"] = df["final_status"].map(STATUS_MAP)
df = df.dropna(subset=["severity"])

# =========================
# SID FILTER
# =========================
systems = sorted(df["system"].unique())
sid = st.sidebar.selectbox("Select System SID", systems)
df = df[df["system"] == sid].copy()

dates = sorted(df["report_date"].unique())

# =========================
# DEDUPLICATION (WORST WINS)
# =========================
df = (
    df.sort_values("severity", ascending=False)
      .groupby(["system", "report_date", "clean_section"], as_index=False)
      .first()
)

# =========================
# ML FEATURE PREP
# =========================
df["is_risk"] = (df["final_status"] == "RED").astype(int)
df["date_ordinal"] = df["report_date"].map(pd.Timestamp.toordinal)

# ML module expects severity 1=GREEN, 2=YELLOW, 3=RED
df_ml = df.copy()
df_ml["severity"] = df_ml["final_status"].str.upper().map({"GREEN": 1, "YELLOW": 2, "RED": 3})
df_ml["status_name"] = df_ml["final_status"]

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔥 Heatmap",
    "📅 Weekly KPI View",
    "📉 Top Risks",
    "🤖 Risk Prediction (ML)",
    "📈 Trend Forecast (ML)",
    "⚠️ Early Warning"
])

# =====================================================
# TAB 1 – HEATMAP
# =====================================================
with tab1:
    pivot = df.pivot(index="clean_section", columns="report_date", values="severity")
    fig = px.imshow(
        pivot,
        color_continuous_scale=[[0, "#00B050"], [0.5, "#FFC000"], [1, "#FF0000"]],
        aspect="auto"
    )
    fig.update_layout(height=700, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# TAB 2 – WEEKLY VIEW
# =====================================================
with tab2:
    sel_date = st.selectbox("Select report date", dates, index=len(dates)-1)
    view = df[df["report_date"] == sel_date].sort_values("severity", ascending=False)
    st.dataframe(view[["clean_section", "final_status"]], use_container_width=True)

# =====================================================
# TAB 3 – TOP RISKS (attention score + rule-based)
# =====================================================
with tab3:
    st.subheader("⚠️ KPIs Needing Special Attention (ML)")
    st.caption("Ranked by attention score: current severity, trend, % weeks in RED, volatility.")
    try:
        att = attention_score(df_ml, system=sid, kpi_col="clean_section")
        if not att.empty:
            st.dataframe(att, use_container_width=True)
            top = att.head(10)
            fig_a = go.Figure(go.Bar(x=top["attention_score"], y=top["kpi"], orientation="h"))
            fig_a.update_layout(title="Top 10 KPIs by Attention Score", xaxis_title="Attention score", height=400)
            st.plotly_chart(fig_a, use_container_width=True)
        else:
            st.info("No attention scores available.")
    except Exception as e:
        st.warning(f"Attention score failed: {e}")

    st.subheader("📋 Rule-based Top Risks (selected week)")
    risk_date = st.selectbox("Risk week", dates, index=len(dates)-1, key="risk_date")
    risks = df[df["report_date"] == risk_date]
    risks["risk_score"] = risks["severity"] * 30
    top10 = risks.sort_values("risk_score", ascending=False).head(10)
    st.dataframe(top10[["clean_section", "final_status", "risk_score"]], use_container_width=True)

# =====================================================
# TAB 4 – LOGISTIC REGRESSION (RISK)
# =====================================================
with tab4:
    st.subheader("🤖 Risk Prediction (Logistic Regression)")

    X = df[["severity"]]
    y = df["is_risk"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_scaled, y)

    preds = model.predict(X_scaled)
    acc = accuracy_score(y, preds)

    st.metric("Model Accuracy", f"{acc:.2f}")
    st.text("Classification Report")
    st.text(classification_report(y, preds))

# =====================================================
# TAB 5 – TREND + FORECAST (ML)
# =====================================================
with tab5:
    st.subheader("📈 KPI Trends & Severity Forecast (ML)")

    # Per-KPI trend table
    st.markdown("**Trend by KPI** (improving: slope below -0.03 | deteriorating: slope above 0.03)")
    try:
        trend_df = compute_trend(df_ml, system=sid, kpi_col="clean_section")
        if not trend_df.empty:
            display_cols = [c for c in ["kpi", "slope", "trend_direction", "n_weeks"] if c in trend_df.columns]
            st.dataframe(trend_df.sort_values("slope", ascending=False)[display_cols], use_container_width=True)
            dir_counts = trend_df["trend_direction"].value_counts()
            fig_t = px.bar(x=dir_counts.index, y=dir_counts.values, labels={"x": "Trend", "y": "Count"}, title="Trend Direction Summary")
            st.plotly_chart(fig_t, use_container_width=True)
        else:
            st.info("Not enough history for trends.")
    except Exception as e:
        st.warning(f"Trend failed: {e}")

    # Severity forecast
    st.markdown("**Forecast: Next N Weeks**")
    periods = st.slider("Weeks to forecast", 2, 8, 4, key="forecast_periods")
    try:
        fcast = forecast_severity(df_ml, system=sid, kpi_col="clean_section", periods_ahead=periods)
        if not fcast.empty:
            REV = {1: "GREEN", 2: "YELLOW", 3: "RED"}
            fcast_disp = fcast.copy()
            fcast_disp["severity_label"] = fcast_disp["severity_pred"].map(lambda x: REV.get(round(x), "?"))
            st.dataframe(fcast_disp, use_container_width=True)
            fig_f = px.line(fcast, x="report_date", y="severity_pred", color="kpi",
                            title="Predicted Severity (1=Green, 2=Yellow, 3=Red)")
            fig_f.update_yaxis(dtick=1)
            st.plotly_chart(fig_f, use_container_width=True)
        else:
            st.info("Not enough data for forecast.")
    except Exception as e:
        st.warning(f"Forecast failed: {e}")

    # Single-KPI linear regression (legacy)
    st.markdown("**Single KPI: Linear Regression Fit**")
    kpi = st.selectbox("Select KPI", sorted(df["clean_section"].unique()), key="trend_kpi")
    kdf = df[df["clean_section"] == kpi].sort_values("report_date")
    if len(kdf) >= 3:
        X = kdf[["date_ordinal"]]
        y = kdf["severity"]
        lr = LinearRegression()
        lr.fit(X, y)
        kdf = kdf.copy()
        kdf["predicted_severity"] = lr.predict(X)
        fig = px.line(kdf, x="report_date", y=["severity", "predicted_severity"])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data points for this KPI.")

# =====================================================
# TAB 6 – EARLY WARNING (TIME SERIES)
# =====================================================
with tab6:
    st.subheader("⚠️ Early Warning Signals (Rolling Trend)")
    st.caption("KPIs with sudden severity increase vs rolling mean. See Tab 3 for ML attention ranking.")

    ew = df.copy()
    ew = ew.sort_values(["clean_section", "report_date"])

    ew["rolling_mean"] = ew.groupby("clean_section")["severity"].transform(
        lambda x: x.rolling(3, min_periods=2).mean()
    )

    ew["trend"] = ew.groupby("clean_section")["rolling_mean"].diff()

    warnings = ew[(ew["trend"] > 0.5) & (ew["severity"] >= 1)]

    if warnings.empty:
        st.info("No early warning signals detected.")
    else:
        st.dataframe(
            warnings[["clean_section", "report_date", "final_status", "trend"]].sort_values("trend", ascending=False),
            use_container_width=True
        )

st.caption("🧠 Intelligent SAP EWA | ML-Enhanced Risk & Trend Analysis")
