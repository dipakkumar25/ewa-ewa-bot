# src/ewa_compare_app.py
"""
Intelligent SAP EWA KPI Comparison Dashboard
---------------------------------------------
• Multi-SID support
• Robust report_date parsing
• Deduplicates KPIs per date (worst severity wins)
• KPI heatmap
• Weekly KPI view
• Top 10 weekly risks
• Two-week comparison
"""

import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path

# ---------------- CONFIG ----------------
DATA_FILE = Path("data/ewa_kpi_clean_summary_all.csv")

STATUS_ORDER = {"GREEN": 0, "YELLOW": 1, "RED": 2}
STATUS_RANK = {"GREEN": 1, "YELLOW": 2, "RED": 3}

st.set_page_config(page_title="Intelligent SAP EWA Dashboard", layout="wide")
st.title("📊 Intelligent SAP EWA – KPI Risk & Comparison Dashboard")

# ---------------- LOAD DATA ----------------
if not DATA_FILE.exists():
    st.error(f"❌ File not found: {DATA_FILE}")
    st.stop()

df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip().str.lower()

if "final_status" not in df.columns and "status_name" in df.columns:
    df["final_status"] = df["status_name"]

if "clean_section" not in df.columns and "section" in df.columns:
    df["clean_section"] = df["section"]

df["report_date"] = pd.to_datetime(df["report_date"], dayfirst=True, errors="coerce")
df = df[df["report_date"].notna()]

df["final_status"] = df["final_status"].fillna("GREEN")
df["severity"] = df["final_status"].map(STATUS_RANK).fillna(1)

# -------- Deduplicate (worst wins) --------
df = (
    df.sort_values("severity", ascending=False)
      .groupby(["system", "report_date", "clean_section"], as_index=False)
      .first()
)

# ---------------- SID FILTER ----------------
systems = sorted(df["system"].dropna().unique())
sid = st.sidebar.selectbox("Select System SID", systems)
df = df[df["system"] == sid]

dates = sorted(df["report_date"].unique())
if not dates:
    st.warning("No valid report dates found.")
    st.stop()

# ---------------- DEVIATION LOGIC ----------------
df = df.sort_values(["clean_section", "report_date"])
df["prev_status"] = df.groupby("clean_section")["final_status"].shift(1)

def deviation(old, new):
    if pd.isna(old): return "🆕 New"
    if STATUS_RANK[new] > STATUS_RANK[old]: return "➖ Deterioration"
    if STATUS_RANK[new] < STATUS_RANK[old]: return "➕ Improvement"
    return "🔄 No Change"

df["deviation"] = df.apply(lambda r: deviation(r["prev_status"], r["final_status"]), axis=1)

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔥 KPI Heatmap",
    "📅 Weekly KPI View",
    "📉 Top Weekly Risks",
    "🔍 Compare Two Weeks"
])

# ================= TAB 1 =================
with tab1:
    pivot = df.pivot_table(index="clean_section", columns="report_date",
                           values="final_status", aggfunc="first")
    pivot_num = pivot.replace(STATUS_ORDER)

    fig = px.imshow(
        pivot_num,
        color_continuous_scale=[[0, "#00B050"], [0.5, "#FFC000"], [1, "#FF0000"]],
        aspect="auto"
    )
    fig.update_layout(height=750, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

# ================= TAB 2 =================
with tab2:
    st.subheader("KPI Status by Date")
    selected_date = st.selectbox("Select report date", dates, index=len(dates)-1)

    df_day = (
        df[df["report_date"] == selected_date]
        .sort_values("severity", ascending=False)
        .reset_index(drop=True)
    )

    df_day.index = df_day.index + 1

    st.dataframe(df_day[[
        "clean_section", "final_status", "prev_status", "deviation"
    ]], use_container_width=True)

# ================= TAB 3 =================
with tab3:
    st.subheader("🚨 Top 10 Weekly Risks")
    risk_date = st.selectbox("Select week", dates, index=len(dates)-1)

    df_risk = df[df["report_date"] == risk_date].copy()
    df_risk["risk_score"] = df_risk["severity"] * 30

    top10 = (
        df_risk.sort_values("risk_score", ascending=False)
               .head(10)
               .reset_index(drop=True)
    )

    top10.index = top10.index + 1

    st.dataframe(top10[[
        "clean_section", "final_status", "prev_status", "deviation", "risk_score"
    ]], use_container_width=True)

# ================= TAB 4 =================
with tab4:
    st.subheader("Compare Any Two EWA Reports")

    c1, c2 = st.columns(2)
    week1 = c1.selectbox("Older report", dates, index=0)
    week2 = c2.selectbox("Newer report", dates, index=len(dates)-1)

    d1 = df[df["report_date"] == week1][["clean_section", "final_status"]]
    d2 = df[df["report_date"] == week2][["clean_section", "final_status"]]

    merged = d1.merge(d2, on="clean_section", how="outer", suffixes=("_old", "_new"))
    merged.fillna("GREEN", inplace=True)

    merged["Change"] = merged.apply(
        lambda r: deviation(r["final_status_old"], r["final_status_new"]), axis=1
    )

    merged["Severity Shift"] = merged["final_status_old"] + " → " + merged["final_status_new"]

    st.dataframe(merged, use_container_width=True)

    st.download_button(
        "⬇️ Download Comparison CSV",
        data=merged.to_csv(index=False).encode("utf-8"),
        file_name=f"EWA_Comparison_{sid}.csv",
        mime="text/csv"
    )

st.caption("🧠 Intelligent EWA | Risk-based SAP KPI Dashboard")
