import os
import re
import pandas as pd
import streamlit as st
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()  # <-- THIS loads .env into environment




# =========================
# CONFIG
# =========================
CLEAN_FILE = Path("data/ewa_kpi_clean_summary.csv")
DETAIL_FILE = Path("data/ewa_html_traffic_lights_A1C.csv")

STATUS_RANK = {"GREEN": 3, "YELLOW": 2, "RED": 1}

# KPI column name used in clean CSV
KPI_COL = "clean_section"

# =========================
# OPTIONAL LLM SETUP
# =========================
USE_LLM = True
OPENAI_AVAILABLE = True

# ----------------------------
# OpenAI Client Initialization
# ----------------------------
client = None

try:
    from openai import OpenAI
    import os

    api_key = os.getenv("OPENAI_API_KEY")
    print("OPENAI_API_KEY =", api_key)
    if api_key:
        client = OpenAI(api_key=api_key)
        OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False



def llm_action_advice(kpi, prev, curr, change, root_cause):
    if not OPENAI_AVAILABLE or client is None:
        return "LLM not available (API key not configured)."

    prompt = f"""
KPI: {kpi}
Previous Status: {prev}
Current Status: {curr}
Change: {change}
Root Cause: {root_cause}

Suggest corrective actions in 2–3 concise bullet points.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an SAP BASIS and SAP Operations expert. "
                        "Base your answer ONLY on the provided information. "
                        "Do not assume missing data. Be precise and actionable."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"LLM execution error: {str(e)}"



# =========================
# ROOT CAUSE SUMMARIZER
# =========================
def summarize_root_cause(kpi, date, df_detail):
    """Extract worst technical finding for KPI/date."""
    possible_cols = ["clean_section", "section", "primary_kpi"]

    kpi_col = next((c for c in possible_cols if c in df_detail.columns), None)
    if not kpi_col:
        return "Root cause unavailable."

    subset = df_detail[
        (df_detail[kpi_col].str.lower() == kpi.lower())
        & (df_detail["report_date"] == date)
    ].copy()

    if subset.empty:
        return "No technical alerts found."

    subset["severity"] = subset["status_name"].map(STATUS_RANK)
    subset = subset.sort_values("severity")

    findings = []
    for t in subset["kpi_text"].dropna():
        t = re.sub(r"^\d+(\.\d+)*\s*", "", t.strip())
        if len(t) > 5:
            findings.append(t)

    return "; ".join(findings[:3]) if findings else "Issues detected."


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="EWA KPI Comparison with AI Actions", layout="wide")
st.title("📊 SAP EWA KPI Comparison & AI Action Advisor")

if not CLEAN_FILE.exists():
    st.error(f"❌ Clean KPI file not found: {CLEAN_FILE}")
    st.stop()

df = pd.read_csv(CLEAN_FILE)
df["report_date"] = pd.to_datetime(df["report_date"])

df_detail = pd.read_csv(DETAIL_FILE)
df_detail["report_date"] = pd.to_datetime(df_detail["report_date"])

dates = sorted(df["report_date"].unique())

# =========================
# DATE SELECTION
# =========================
c1, c2 = st.columns(2)
date1 = c1.selectbox("Older report date", dates, index=0)
date2 = c2.selectbox("Newer report date", dates, index=len(dates) - 1)

df_old = df[df["report_date"] == date1].set_index(KPI_COL)
df_new = df[df["report_date"] == date2].set_index(KPI_COL)

merged = df_old.join(
    df_new,
    lsuffix="_old",
    rsuffix="_new",
    how="outer"
)

# Fill missing KPIs as GREEN
merged["final_status_old"] = merged["final_status_old"].fillna("GREEN")
merged["final_status_new"] = merged["final_status_new"].fillna("GREEN")

# =========================
# CHANGE CLASSIFICATION
# =========================
def classify(old, new):
    if STATUS_RANK[new] > STATUS_RANK[old]:
        return "➕ Improvement"
    if STATUS_RANK[new] < STATUS_RANK[old]:
        return "➖ Deterioration"
    return "🔄 No Change"


merged["Change"] = merged.apply(
    lambda r: classify(r["final_status_old"], r["final_status_new"]),
    axis=1
)

# =========================
# ROOT CAUSE
# =========================
merged["Root Cause"] = merged.apply(
    lambda r: summarize_root_cause(r.name, date2, df_detail),
    axis=1
)

# =========================
# OPTIONAL LLM TOGGLE
# =========================
USE_LLM = st.checkbox("🤖 Generate AI Action Recommendations", value=False)

if USE_LLM and OPENAI_AVAILABLE:
    merged["AI Action"] = merged.apply(
        lambda r: llm_action_advice(
            r.name,
            r["final_status_old"],
            r["final_status_new"],
            r["Change"],
            r["Root Cause"],
        )
        if r["Change"] == "➖ Deterioration"
        else "No action required.",
        axis=1,
    )
elif USE_LLM:
    merged["AI Action"] = "LLM not enabled."

# =========================
# DISPLAY
# =========================
display_cols = [
    "final_status_old",
    "final_status_new",
    "Change",
    "Root Cause",
]

if "AI Action" in merged.columns:
    display_cols.append("AI Action")

st.subheader("📌 KPI Comparison Result")
st.dataframe(merged[display_cols], use_container_width=True)

# =========================
# DOWNLOAD
# =========================
st.download_button(
    "⬇️ Download Comparison Report",
    data=merged.reset_index().to_csv(index=False).encode("utf-8"),
    file_name=f"EWA_Comparison_{date2.date()}_vs_{date1.date()}.csv",
    mime="text/csv",
)
