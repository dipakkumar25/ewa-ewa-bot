"""
ewa_kpi_cleaner.py

Cleans and consolidates KPI signals extracted by ewa_html_processor.py.
Produces a stable summary file for dashboards, trend analysis, and WoW comparison.

Input:
    data/ewa_html_traffic_lights_A1C.csv

Output:
    data/ewa_kpi_clean_summary.csv
"""

import pandas as pd
import re
from pathlib import Path

# Input & output CSV locations
BASE_DIR = Path(__file__).resolve().parent.parent
DETAIL_FILE = BASE_DIR / "data" / "ewa_html_traffic_lights_A1C.csv"
OUTPUT_FILE = BASE_DIR / "data" / "ewa_kpi_clean_summary.csv"

# ----------------------------------------
# 1. Load Detail Data
# ----------------------------------------
def load_detail_file():
    if not DETAIL_FILE.exists():
        raise FileNotFoundError(f"ERROR: Input CSV not found → {DETAIL_FILE}")

    df = pd.read_csv(DETAIL_FILE)
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")

    print(f"Loaded {len(df)} rows from {DETAIL_FILE}")
    return df


# ----------------------------------------
# 2. Remove numeric prefixes from the section text
# ----------------------------------------
def clean_section_name(text):
    if pd.isna(text):
        return ""
    # Remove patterns like "1", "1.1", "10.3.5"
    cleaned = re.sub(r"^\s*\d+(\.\d+)*\s*", "", str(text)).strip()
    return cleaned


# ----------------------------------------
# 3. Normalize KPI names (remove extra spaces, set proper capitalization)
# ----------------------------------------
def normalize_section(text):
    if pd.isna(text):
        return ""
    
    t = text.replace("_", " ").strip()
    t = re.sub(r"\s+", " ", t)
    t = t.title()

    # Fix common patterns
    fixes = {
        "Sap": "SAP",
        "Abap": "ABAP",
        "Hana": "HANA",
        "Netwear": "Netweaver",
    }
    for bad, good in fixes.items():
        t = t.replace(bad, good)

    return t


# ----------------------------------------
# 4. Severity Mapping
# ----------------------------------------
SEV_MAP = {"GREEN": 1, "YELLOW": 2, "RED": 3}
REV_MAP = {1: "GREEN", 2: "YELLOW", 3: "RED"}
SYM = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}


# ----------------------------------------
# 5. Build Final KPI Summary (worst severity per date)
# ----------------------------------------
def build_summary(df):
    df["severity"] = df["status_name"].map(SEV_MAP)

    # Consolidation:
    # If multiple rows exist for same KPI in same date → take worst severity
    summary = (
        df.groupby(["report_date", "clean_section"], as_index=False)
          .agg({"severity": "max"})
    )

    summary["final_status"] = summary["severity"].map(REV_MAP)
    summary["status_symbol"] = summary["final_status"].map(SYM)

    return summary.sort_values(["report_date", "clean_section"])


# ----------------------------------------
# MAIN PIPELINE
# ----------------------------------------
def main():
    print("\n=== KPI CLEANER STARTED ===")

    df = load_detail_file()

    # Step A: Clean numeric prefixes
    df["clean_section"] = df["section"].apply(clean_section_name)

    # Step B: Normalize names for consistency
    df["clean_section"] = df["clean_section"].apply(normalize_section)

    # 🚨 IMPORTANT FIX: Remove blank/empty KPI groups
    df = df[df["clean_section"].str.strip() != ""]


    # Build summary
    df_summary = build_summary(df)

    # Save cleaned summary
    df_summary.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✔ Clean KPI summary saved → {OUTPUT_FILE}")
    print(f"Total KPI groups: {df_summary['clean_section'].nunique()}")
    print(df_summary.head(15))

    print("\n=== KPI CLEANER COMPLETED ===")


if __name__ == "__main__":
    main()
