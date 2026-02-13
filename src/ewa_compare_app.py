"""
Intelligent SAP EWA – KPI Risk & Comparison Dashboard (ML Enhanced)
------------------------------------------------------------------
Adds:
• Logistic Regression (Risk Prediction)
• Linear Regression (Trend Forecasting)
• Time-Series Early Warning Signals

Preserves:
• Existing rule-based KPI logic
• Existing dashboard structure
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from pathlib import Path

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

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
# TAB 3 – TOP RISKS
# =====================================================
with tab3:
    risk_date = st.selectbox("Risk week", dates, index=len(dates)-1)
    risks = df[df["report_date"] == risk_date]
    risks["risk_score"] = risks["severity"] * 30
    top10 = risks.sort_values("risk_score", ascending=False).head(10)
    st.dataframe(top10[["clean_section", "final_status", "risk_score"]])

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
# TAB 5 – LINEAR REGRESSION (TREND)
# =====================================================
with tab5:
    st.subheader("📈 KPI Trend Forecast")

    kpi = st.selectbox("Select KPI", sorted(df["clean_section"].unique()))
    kdf = df[df["clean_section"] == kpi].sort_values("report_date")

    X = kdf[["date_ordinal"]]
    y = kdf["severity"]

    if len(kdf) >= 3:
        lr = LinearRegression()
        lr.fit(X, y)
        kdf["predicted_severity"] = lr.predict(X)

        fig = px.line(kdf, x="report_date", y=["severity", "predicted_severity"])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data points for trend prediction.")

# =====================================================
# TAB 6 – EARLY WARNING (TIME SERIES)
# =====================================================
with tab6:
    st.subheader("⚠️ Early Warning Signals")

    ew = df.copy()
    ew = ew.sort_values(["clean_section", "report_date"])

    ew["rolling_mean"] = ew.groupby("clean_section")["severity"].transform(
        lambda x: x.rolling(3, min_periods=2).mean()
    )

    ew["trend"] = ew.groupby("clean_section")["rolling_mean"].diff()

    warnings = ew[(ew["trend"] > 0.5) & (ew["severity"] >= 1)]

    st.dataframe(
        warnings[[
            "clean_section",
            "report_date",
            "final_status",
            "trend"
        ]].sort_values("trend", ascending=False),
        use_container_width=True
    )

st.caption("🧠 Intelligent SAP EWA | ML-Enhanced Risk & Trend Analysis")
