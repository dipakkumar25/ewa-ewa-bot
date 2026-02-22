# src/ewa_html_processor.py
"""
Multi-SID SAP EWA HTML KPI extractor (Corrected)

Fixes:
✔ Generic KPI names (no A1C/P1C/Q1C leakage)
✔ Clean, reusable KPI name column
✔ Proper row numbering (starts from 1)
✔ Multi-SID safe
"""

import re
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict

import pandas as pd
from bs4 import BeautifulSoup

from .config import HTML_DIR

DATA_DIR = Path(HTML_DIR).parent

# ======================================================
# 🔹 GENERIC KPI MASTER (NO SYSTEM ID)
# ======================================================
PRIMARY_KPI_ORDER = [
    "Service summary",
    "Service Data Quality and Service Readiness",
    "Software Configuration",
    "Hardware Capacity",
    "Performance Overview",
    "SAP System Operating",
    "Security",
    "Software Change and Transport Management",
    "Financial Data Quality",
    "Upgrade Planning",
    "SAP HANA Database",
    "SAP Netweaver Gateway",
    "UI Technologies checks",
]

PRIMARY_KPI_KEYWORDS: Dict[str, List[str]] = {
    "Service summary": ["service summary"],
    "Service Data Quality and Service Readiness": ["data quality", "service readiness"],
    "Software Configuration": ["software configuration", "configuration"],
    "Hardware Capacity": ["hardware capacity", "cpu", "memory", "disk"],
    "Performance Overview": ["performance overview", "response time"],
    "SAP System Operating": ["system operating", "background job"],
    "Security": ["security", "authorization", "ssl", "vulnerability"],
    "Software Change and Transport Management": ["transport", "change management"],
    "Financial Data Quality": ["financial"],
    "Upgrade Planning": ["upgrade", "maintenance"],
    "SAP HANA Database": ["hana", "database"],
    "SAP Netweaver Gateway": ["gateway", "netweaver"],
    "UI Technologies checks": ["ui", "fiori", "web dynpro"],
}

SEVERITY_ORDER = {"GREEN": 1, "YELLOW": 2, "RED": 3}
REV_SEVERITY = {1: "GREEN", 2: "YELLOW", 3: "RED"}
SYM = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}

# ======================================================
# UTILITIES
# ======================================================
def extract_sid_and_date(filename: str) -> Tuple[str, Optional[datetime.date]]:
    sid_match = re.search(r"EWA_([^~]+)~", filename)
    date_match = re.search(r"(\d{8})", filename)

    sid = sid_match.group(1).upper() if sid_match else "UNKNOWN"
    date = datetime.strptime(date_match.group(1), "%Y%m%d").date() if date_match else None
    return sid, date


def detect_status_from_img(img):
    txt = f"{img.get('alt','')} {img.get('src','')}".lower()
    if "red" in txt or "critical" in txt:
        return "RED", SYM["RED"]
    if "yellow" in txt or "warning" in txt:
        return "YELLOW", SYM["YELLOW"]
    if "green" in txt or "ok" in txt:
        return "GREEN", SYM["GREEN"]
    return None


def detect_status_from_style(style):
    if not style:
        return None
    s = style.lower()
    if "255,0,0" in s or "red" in s:
        return "RED", SYM["RED"]
    if "255,255,0" in s or "yellow" in s:
        return "YELLOW", SYM["YELLOW"]
    if "0,128,0" in s or "green" in s:
        return "GREEN", SYM["GREEN"]
    return None


def map_to_primary_kpi(text: str) -> Optional[str]:
    t = text.lower()
    for kpi, kws in PRIMARY_KPI_KEYWORDS.items():
        if any(kw in t for kw in kws):
            return kpi
    return None


# ======================================================
# HTML PARSER
# ======================================================
def parse_single_html(path: Path) -> List[dict]:
    sid, report_date = extract_sid_and_date(path.name)
    soup = BeautifulSoup(path.read_text(errors="ignore"), "html.parser")

    rows = []

    for tr in soup.find_all("tr"):
        tds = tr.find_all("td")
        if not tds:
            continue

        detected = None
        img = tr.find("img")
        if img:
            detected = detect_status_from_img(img)

        if not detected:
            for td in tds:
                detected = detect_status_from_style(td.get("style"))
                if detected:
                    break

        if not detected:
            continue

        label = max(
            [td.get_text(strip=True) for td in tds if td.get_text(strip=True)],
            key=len,
            default=""
        )

        rows.append({
            "system": sid,
            "report_date": report_date,
            "kpi_text": label,
            "status_name": detected[0],
            "status_symbol": detected[1],
            "source_file": path.name,
        })

    return rows


# ======================================================
# PIPELINE
# ======================================================
def process_all_systems():
    files = list(Path(HTML_DIR).glob("*.htm")) + list(Path(HTML_DIR).glob("*.html"))
    if not files:
        raise FileNotFoundError("No EWA HTML files found.")

    sid_groups = {}
    for f in files:
        sid, _ = extract_sid_and_date(f.name)
        sid_groups.setdefault(sid, []).append(f)

    for sid, sid_files in sid_groups.items():
        all_rows = []
        for f in sid_files:
            all_rows.extend(parse_single_html(f))

        if not all_rows:
            continue

        df = pd.DataFrame(all_rows)

        df["KPI name"] = df["kpi_text"].apply(map_to_primary_kpi)
        df = df[df["KPI name"].notna()].copy()

        df["severity"] = df["status_name"].map(SEVERITY_ORDER)

        # Worst severity per KPI/date
        summary = (
            df.sort_values("severity", ascending=False)
              .groupby(["system", "report_date", "KPI name"], as_index=False)
              .first()
        )

        summary["row_id"] = range(1, len(summary) + 1)

        out_file = DATA_DIR / f"ewa_html_traffic_lights_summary13_{sid}.csv"
        summary.to_csv(out_file, index=False)

        print(f"✔ Summary created: {out_file.name}")


def main():
    print("\n=== Intelligent EWA Multi-SID Processor (Clean KPI Names) ===")
    process_all_systems()
    print("============================================================")


if __name__ == "__main__":
    main()
