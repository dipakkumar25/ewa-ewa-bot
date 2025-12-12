import streamlit as st
import pandas as pd
import plotly.express as px
import re
from pathlib import Path

# ---------------------------------------------
# FILE PATHS
# ---------------------------------------------
CLEAN_SUMMARY = Path("data/ewa_kpi_clean_summary.csv")
DETAIL_FILE   = Path("data/ewa_html_traffic_lights_A1C.csv")

# ---------------------------------------------
# CONSTANTS
# ---------------------------------------------
STATUS_RANK = {"GREEN": 3, "YELLOW": 2, "RED": 1}
COLOR_MAP = {"GREEN": "#00B050", "YELLOW": "#FFC000", "RED": "#FF0000"}

st.set_page_config(page_title="EWA KPI Comparison Dashboard", layout="wide")
st.title("📊 SAP EWA – KPI Comparison Dashboard (A1C)")

# ---------------------------------------------
# LOAD DATA
# ---------------------------------------------
if not CLEAN_SUMMARY.exists():
    st.error("❌ Missing file: ewa_kpi_clean_summary.csv. Run ewa_kpi_cleaner.py first.")
    st.stop()

if not DETAIL_FILE.exists():
    st.error("❌ Missing file: ewa_html_traffic_lights_A1C.csv. Run ewa_html_processor.py first.")
    st.stop()

df = pd.read_csv(CLEAN_SUMMARY)
df["report_date"] = pd.to_datetime(df["report_date"])

df_detail = pd.read_csv(DETAIL_FILE)
df_detail["report_date"] = pd.to_datetime(df_detail["report_date"])
df_detail["severity"] = df_detail["status_name"].map({"GREEN": 1, "YELLOW": 2, "RED": 3})

# ---------------------------------------------
# KPI Column Auto-detection
# ---------------------------------------------
KPI_COL_OPTIONS = ["clean_section", "primary_kpi", "section"]
KPI_COL = None
for col in KPI_COL_OPTIONS:
    if col in df.columns:
        KPI_COL = col
        break

if KPI_COL is None:
    st.error("Could not find KPI column (clean_section / primary_kpi / section).")
    st.stop()

# ---------------------------------------------
# DATE SELECTION
# ---------------------------------------------
dates = sorted(df["report_date"].unique())
col1, col2 = st.columns(2)
date1 = col1.selectbox("📅 Select FIRST (Older) Date", dates, index=0)
date2 = col2.selectbox("📅 Select SECOND (Newer) Date", dates, index=len(dates) - 1)

df1 = df[df["report_date"] == date1]
df2 = df[df["report_date"] == date2]

merged = df1[[KPI_COL, "final_status"]].merge(
    df2[[KPI_COL, "final_status"]],
    on=KPI_COL,
    suffixes=("_old", "_new")
)

# ---------------------------------------------
# STATUS CHANGE CLASSIFICATION
# ---------------------------------------------
def classify(old, new):
    if pd.isna(old) or pd.isna(new):
        return "No Data"
    if STATUS_RANK[new] > STATUS_RANK[old]:
        return "➕ Improvement"
    if STATUS_RANK[new] < STATUS_RANK[old]:
        return "➖ Deterioration"
    return "🔄 No Change"

merged["Change"] = merged.apply(
    lambda r: classify(r["final_status_old"], r["final_status_new"]),
    axis=1
)

# ---------------------------------------------
# SMART ROOT-CAUSE SUMMARIZER
# ---------------------------------------------
def summarize_root_cause(kpi_name, date):
    """
    Summarizes the most severe causes from the detailed file for a KPI & date.
    Auto-detects which column maps to KPI.
    """

    possible_cols = ["primary_kpi", "section", "clean_section", "kpi"]
    kpi_col = None
    for c in possible_cols:
        if c in df_detail.columns:
            kpi_col = c
            break

    if kpi_col is None:
        return "Root cause unavailable: KPI column missing."

    sub = df_detail[
        (df_detail[kpi_col].str.lower() == str(kpi_name).lower()) &
        (df_detail["report_date"] == date)
    ].sort_values("severity", ascending=False)

    if sub.empty:
        return "No underlying alerts found for this KPI."

    findings = []
    for _, r in sub.iterrows():
        text = r.get("kpi_text", "")
        if isinstance(text, str) and len(text) > 4:
            clean = re.sub(r"^\d+(\.\d+)*\s*", "", text)
            findings.append(clean)

    if not findings:
        return "Issues detected, but no technical description available."

    return "; ".join(findings[:3])


def get_summary(row):
    kpi = row[KPI_COL]

    if row["Change"] == "➖ Deterioration":
        return summarize_root_cause(kpi, date2)

    if row["Change"] == "➕ Improvement":
        return "KPI improved — previous issues are no longer present."

    return "No change in underlying symptoms."

merged["Root Cause"] = merged.apply(get_summary, axis=1)

# ---------------------------------------------
# HEATMAP VISUALIZATION
# ---------------------------------------------
st.subheader("🔥 Heatmap of KPI Status (Two Dates)")

heatmap_df = merged[[KPI_COL, "final_status_old", "final_status_new"]].copy()
heatmap_df = heatmap_df.melt(id_vars=KPI_COL, var_name="Date", value_name="Status")

heatmap_df["Date"] = heatmap_df["Date"].map({
    "final_status_old": date1.strftime("%Y-%m-%d"),
    "final_status_new": date2.strftime("%Y-%m-%d")
})

heatmap_df["Status_num"] = heatmap_df["Status"].map({"GREEN": 1, "YELLOW": 2, "RED": 3})

fig = px.imshow(
    heatmap_df.pivot(index=KPI_COL, columns="Date", values="Status_num"),
    color_continuous_scale=[COLOR_MAP["GREEN"], COLOR_MAP["YELLOW"], COLOR_MAP["RED"]],
    aspect="auto",
)

fig.update_layout(height=600, coloraxis_showscale=False)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------
# COMPARISON TABLE
# ---------------------------------------------
st.subheader("📋 Detailed KPI Comparison")
st.dataframe(
    merged[[KPI_COL, "final_status_old", "final_status_new", "Change", "Root Cause"]],
    use_container_width=True
)

# ---------------------------------------------
# EXPORT CSV
# ---------------------------------------------
st.download_button(
    label="⬇️ Download Comparison CSV",
    data=merged.to_csv(index=False).encode("utf-8"),
    file_name=f"EWA_Comparison_{date1.date()}_vs_{date2.date()}.csv",
    mime="text/csv"
)
