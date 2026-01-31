"""
ewa_kpi_cleaner.py

Multi-SID KPI cleaner for SAP EWA project.

Features:
- Reads all ewa_html_traffic_lights_<SID>.csv files
- Merges all systems into one dataset
- Cleans section names (removes numbering)
- Normalizes KPI text
- Applies worst-severity logic per KPI per day
- Outputs a single consolidated clean file

Output:
data/ewa_kpi_clean_summary_all.csv
"""

import re
from pathlib import Path
import pandas as pd

# -------------------
# CONFIG
# -------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_FILE = DATA_DIR / "ewa_kpi_clean_summary_all.csv"

SEVERITY_RANK = {"GREEN": 1, "YELLOW": 2, "RED": 3}
REVERSE_RANK = {1: "GREEN", 2: "YELLOW", 3: "RED"}

# -------------------
# CLEANING FUNCTIONS
# -------------------

def clean_section(text: str) -> str:
    """Remove numbering like '10.1.4', '1 ', '2.3 ' from section headers"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"^\d+(\.\d+)*\s*", "", text)
    return text.strip()

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()

# -------------------
# LOAD ALL SIDS
# -------------------

def load_all_sid_files() -> pd.DataFrame:
    files = list(DATA_DIR.glob("ewa_html_traffic_lights_*.csv"))
    if not files:
        raise FileNotFoundError("No ewa_html_traffic_lights_<SID>.csv files found")

    print(f"📂 Found {len(files)} SID file(s)")

    all_df = []
    for f in files:
        print(f"✔ Loading {f.name}")
        df = pd.read_csv(f)
        df["system"] = df["system"].astype(str)
        df["report_date"] = pd.to_datetime(df["report_date"])
        all_df.append(df)

    return pd.concat(all_df, ignore_index=True)

# -------------------
# CORE CLEANING LOGIC
# -------------------

def clean_and_aggregate(df: pd.DataFrame) -> pd.DataFrame:

    print("🧹 Cleaning section names & KPI text...")

    df["clean_section"] = df["section"].apply(clean_section)
    df["clean_kpi"] = df["kpi_text"].apply(normalize_text)

    df = df[[
        "system",
        "report_date",
        "clean_section",
        "clean_kpi",
        "status_name",
        "source_file"
    ]]

    print("⚖ Applying worst-severity logic per KPI per day...")

    df["severity"] = df["status_name"].map(SEVERITY_RANK)

    worst = (
        df.sort_values("severity", ascending=False)
          .groupby(["system", "report_date", "clean_section", "clean_kpi"], as_index=False)
          .first()
    )

    worst["final_status"] = worst["severity"].map(REVERSE_RANK)

    final_df = worst.drop(columns=["severity", "status_name"])\
                    .rename(columns={"final_status": "status_name"})

    return final_df.sort_values(["system", "report_date", "clean_section"])

# -------------------
# MAIN
# -------------------

def main():
    print("\n🚀 Starting multi-SID KPI cleaning process...\n")

    df_raw = load_all_sid_files()
    df_clean = clean_and_aggregate(df_raw)

    df_clean.to_csv(OUTPUT_FILE, index=False)

    print("\n✅ CLEAN KPI MASTER FILE GENERATED")
    print(f"📄 {OUTPUT_FILE}")
    print("\nSample:")
    print(df_clean.head(12))

if __name__ == "__main__":
    main()

