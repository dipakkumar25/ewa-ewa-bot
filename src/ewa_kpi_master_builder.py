# src/ewa_kpi_master_builder.py
"""
Master KPI Summary Builder (Multi-SID)

• Reads all system-level summary13 CSVs
• Normalizes schema
• Enforces worst severity per SID/date/KPI
• Builds one enterprise clean file:
    data/ewa_kpi_clean_summary_all.csv
"""

import re
from pathlib import Path
from typing import List

import pandas as pd

try:
    from .config import BASE_DIR
    DATA_DIR = Path(BASE_DIR) / "data"
except ImportError:
    DATA_DIR = Path(__file__).resolve().parent.parent / "data"

INPUT_FILES: List[Path] = sorted(DATA_DIR.glob("ewa_html_traffic_lights_summary13_*.csv"))
OUTPUT_FILE = DATA_DIR / "ewa_kpi_clean_summary_all.csv"
SEVERITY_RANK = {"GREEN": 1, "YELLOW": 2, "RED": 3}


def _infer_system_from_filename(filename: str) -> str:
    """Infer system ID from summary13 filename, e.g. summary13_A1C.csv -> A1C."""
    m = re.search(r"summary13_([A-Z0-9]+)\.csv$", filename, re.I)
    return m.group(1).upper() if m else "UNKNOWN"


# ===============================
# LOAD + VALIDATE
# ===============================
if not INPUT_FILES:
    raise FileNotFoundError("❌ No summary13 files found in data/")

frames = []
for file in INPUT_FILES:
    print(f"📂 Reading {file.name}")
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.lower()

    if "primary_kpi" in df.columns and "KPI name" not in df.columns:
        df.rename(columns={"primary_kpi": "KPI name"}, inplace=True)
    if "clean_section" in df.columns and "KPI name" not in df.columns:
        df.rename(columns={"clean_section": "KPI name"}, inplace=True)
    if "kpi name" in df.columns and "KPI name" not in df.columns:
        df.rename(columns={"kpi name": "KPI name"}, inplace=True)
    if "system" not in df.columns:
        df["system"] = _infer_system_from_filename(file.name)

    required = {"report_date", "KPI name", "status_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{file.name} missing required columns {missing}")

    keep_cols = [c for c in ["system", "report_date", "KPI name", "status_name", "status_symbol", "source_file"] if c in df.columns]
    df = df[keep_cols]
    frames.append(df)

if not frames:
    raise RuntimeError("❌ No valid KPI summary files could be loaded.")

df_all = pd.concat(frames, ignore_index=True)

# ===============================
# CLEANING
# ===============================
df_all["report_date"] = pd.to_datetime(df_all["report_date"], errors="coerce", dayfirst=True)
df_all = df_all[df_all["report_date"].notna()]

df_all["status_name"] = df_all["status_name"].astype(str).str.upper().str.strip()
df_all["severity"] = df_all["status_name"].map(SEVERITY_RANK)

df_all["KPI name"] = df_all["KPI name"].astype(str).str.strip()

# ===============================
# ENFORCE WORST KPI PER DATE
# ===============================
df_final = (
    df_all.sort_values("severity", ascending=False)
          .groupby(["system", "report_date", "KPI name"], as_index=False)
          .first()
)

# ===============================
# FINAL TOUCHES
# ===============================
df_final = df_final.sort_values(["system","report_date","KPI name"]).reset_index(drop=True)
df_final["row_id"] = df_final.index + 1

# ===============================
# EXPORT
# ===============================
df_final.to_csv(OUTPUT_FILE, index=False)

print("\n✅ MASTER KPI SUMMARY CREATED")
print("➡", OUTPUT_FILE)
print("Rows:", len(df_final))
print(df_final.head(20))
