# src/ewa_compare_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import re

# ----------------------------
# Config / file paths
# ----------------------------
BASE = Path(__file__).resolve().parent.parent
FALLBACK_CLEAN = BASE / "data" / "ewa_kpi_clean_summary.csv"
COMPARE_CSV = BASE / "data" / "ewa_compare_dashboard.csv"  # user-provided name
DETAIL_CSV = BASE / "data" / "ewa_html_traffic_lights_A1C.csv"  # original detail file

# choose which clean summary to read
if COMPARE_CSV.exists():
    CLEAN_CSV = COMPARE_CSV
elif FALLBACK_CLEAN.exists():
    CLEAN_CSV = FALLBACK_CLEAN
else:
    st.error("No clean KPI summary CSV found. Run the cleaner first.")
    st.stop()

# ----------------------------
# Utilities
# ----------------------------
def clean_text(t: str) -> str:
    if pd.isna(t): return ""
    t = re.sub(r"^\s*\d+(\.\d+)*\s*", "", str(t))    # remove leading numbers
    t = re.sub(r"\s+", " ", t).strip()
    t = t.replace("_", " ")
    # basic capitalization fixes
    t = t.title()
    for bad, good in [("Sap", "SAP"), ("Abap", "ABAP"), ("Hana", "HANA"), ("Netwear", "Netweaver")]:
        t = t.replace(bad, good)
    return t

STATUS_ORDER = {"GREEN": 1, "YELLOW": 2, "RED": 3}
SYMBOL = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}

def classify_change(s_old, s_new):
    if pd.isna(s_old):
        return "🆕 New"
    if s_old == s_new:
        return "🔄 No change"
    if STATUS_ORDER[s_new] > STATUS_ORDER[s_old]:
        return "➖ Deterioration"  # severity number larger -> worse
    if STATUS_ORDER[s_new] < STATUS_ORDER[s_old]:
        return "➕ Improvement"
    return "🔄 No change"

def root_cause_for_kpi(detail_df, kpi, date):
    """Return short root cause message for KPI at given date (prefer highest severity)."""
    if detail_df is None:
        return "No details"
    # try both 'clean_section' and 'section' fields
    df = detail_df.copy()
    # normalize names used in detail
    if "clean_section" not in df.columns:
        df["clean_section"] = df.get("section", "").astype(str).apply(clean_text)
    # filter by date and KPI (allow substring match)
    subset = df[(pd.to_datetime(df["report_date"]).dt.date == pd.to_datetime(date).date()) &
                (df["clean_section"].str.lower() == str(kpi).lower())]
    if subset.empty:
        # try fuzzy substring match in kpi_text
        subset = df[(pd.to_datetime(df["report_date"]).dt.date == pd.to_datetime(date).date()) &
                    (df["kpi_text"].str.contains(str(kpi), case=False, na=False))]
    if subset.empty:
        return "No alerts"
    # sort by severity if available (map from status_name)
    if "status_name" in subset.columns:
        subset["sev"] = subset["status_name"].map(STATUS_ORDER).fillna(1)
        subset = subset.sort_values("sev", ascending=False)
    # return concatenation of top unique kpi_texts (limit length)
    top = subset["kpi_text"].dropna().unique()[:5]
    return " | ".join(top)

# ----------------------------
# Load data
# ----------------------------
df_clean = pd.read_csv(CLEAN_CSV)
# canonical columns expected: report_date, clean_section, final_status (or status_name)
# normalize column names
if "final_status" in df_clean.columns:
    df_clean["status_name"] = df_clean["final_status"]
if "clean_section" not in df_clean.columns and "section" in df_clean.columns:
    df_clean["clean_section"] = df_clean["section"].apply(clean_text)
else:
    df_clean["clean_section"] = df_clean["clean_section"].astype(str).apply(clean_text)

df_clean["report_date"] = pd.to_datetime(df_clean["report_date"])
df_clean = df_clean.dropna(subset=["clean_section"])  # drop any blank KPI rows

# load detail CSV for root cause (optional)
detail_df = None
if DETAIL_CSV.exists():
    detail_df = pd.read_csv(DETAIL_CSV)
    # ensure columns
    if "kpi_text" not in detail_df.columns and "kpi" in detail_df.columns:
        detail_df = detail_df.rename(columns={"kpi": "kpi_text"})
    # add clean_section if missing
    if "clean_section" not in detail_df.columns:
        detail_df["clean_section"] = detail_df.get("section", detail_df.get("kpi_text", "")).astype(str).apply(clean_text)
    # standardize report_date
    detail_df["report_date"] = pd.to_datetime(detail_df["report_date"], errors="coerce")

# ----------------------------
# UI
# ----------------------------
st.set_page_config(layout="wide", page_title="EWA KPI Compare")
st.title("📊 EWA KPI Compare — Compare two dates (All KPIs)")

dates = sorted(df_clean["report_date"].dt.normalize().unique())
if len(dates) < 2:
    st.warning("Need at least two report dates to compare. Run processor + cleaner first.")
    st.stop()

col1, col2 = st.columns([1, 2])
with col1:
    st.info(f"Loaded clean summary: {CLEAN_CSV.name} — {len(df_clean)} rows")
    date_old = st.selectbox("Date (older)", dates, index=max(0, len(dates)-2))
    date_new = st.selectbox("Date (newer)", dates, index=len(dates)-1)

# ensure order
if pd.to_datetime(date_old) >= pd.to_datetime(date_new):
    st.error("Please pick an older date for the left selector and a newer date for the right selector.")
    st.stop()

# filter per selected dates
df_old = df_clean[df_clean["report_date"].dt.normalize() == pd.to_datetime(date_old).normalize()]
df_new = df_clean[df_clean["report_date"].dt.normalize() == pd.to_datetime(date_new).normalize()]

# base KPI list (union)
kpis = sorted(set(df_old["clean_section"].unique()) | set(df_new["clean_section"].unique()))

# build comparison table
rows = []
for k in kpis:
    s_old = df_old[df_old["clean_section"].str.lower() == k.lower()]["status_name"]
    s_new = df_new[df_new["clean_section"].str.lower() == k.lower()]["status_name"]
    st_old = s_old.iloc[0] if not s_old.empty else None
    st_new = s_new.iloc[0] if not s_new.empty else None
    symbol_old = SYMBOL.get(st_old, " ")
    symbol_new = SYMBOL.get(st_new, " ")
    change = classify_change(st_old, st_new if st_new is not None else st_old)
    rc = root_cause_for_kpi(detail_df, k, date_new) if detail_df is not None else "No details file"
    rows.append({
        "primary_kpi": k,
        "status_old": st_old if st_old else "MISSING",
        "symbol_old": symbol_old,
        "status_new": st_new if st_new else "MISSING",
        "symbol_new": symbol_new,
        "change": change,
        "root_cause": rc
    })

df_compare = pd.DataFrame(rows)

# ----------------------------
# Heatmap (KPIs x 2 dates)
# ----------------------------
pivot = pd.DataFrame({
    "primary_kpi": kpis
}).set_index("primary_kpi")
# map statuses to numeric for plotting (1 green -> 3 red to get green->low numeric)
def status_to_val(s):
    if pd.isna(s) or s in [None, "MISSING"]: return 0
    return STATUS_ORDER.get(s.upper(), 0)

left_vals = [status_to_val(df_old[df_old["clean_section"].str.lower() == k.lower()]["status_name"].iloc[0]) 
             if not df_old[df_old["clean_section"].str.lower() == k.lower()]["status_name"].empty else 0
             for k in kpis]
right_vals = [status_to_val(df_new[df_new["clean_section"].str.lower() == k.lower()]["status_name"].iloc[0]) 
              if not df_new[df_new["clean_section"].str.lower() == k.lower()]["status_name"].empty else 0
              for k in kpis]

heat_df = pd.DataFrame({
    "old": left_vals,
    "new": right_vals
}, index=kpis)

# color scale mapping (0 -> grey for missing, 1 green, 2 yellow, 3 red)
color_map = [
    [0.0, "lightgrey"],
    [0.00001, "#00B050"],  # green
    [0.333, "#00B050"],
    [0.3331, "#FFC000"],  # yellow
    [0.666, "#FFC000"],
    [0.6661, "#FF0000"],  # red
    [1.0, "#FF0000"]
]

fig = px.imshow(
    heat_df.values,
    x=[pd.to_datetime(date_old).date(), pd.to_datetime(date_new).date()],
    y=heat_df.index,
    color_continuous_scale=["lightgrey", "#00B050", "#FFC000", "#FF0000"],
    aspect="auto",
    origin="lower"
)
# override hover text
custom_text = [[
    f"KPI: {k}<br>Date: {d}<br>Status: {('MISSING' if val==0 else list(STATUS_ORDER.keys())[list(STATUS_ORDER.values()).index(val)])}"
    for d, val in zip([date_old, date_new], [heat_df.loc[k, "old"], heat_df.loc[k, "new"]])
] for k in heat_df.index]
fig.update_traces(hoverinfo="text", text=custom_text)
fig.update_coloraxes(showscale=False)
fig.update_layout(height=700, margin=dict(l=180, r=20, t=60, b=40))
st.subheader("Heatmap — older vs newer")
st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Comparison table & download
# ----------------------------
st.subheader("Detailed comparison (per KPI)")
st.dataframe(df_compare[["primary_kpi", "symbol_old", "status_old", "symbol_new", "status_new", "change", "root_cause"]], use_container_width=True)

csv_bytes = df_compare.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download comparison CSV", data=csv_bytes, file_name=f"ewa_compare_{pd.to_datetime(date_old).date()}_vs_{pd.to_datetime(date_new).date()}.csv", mime="text/csv")
