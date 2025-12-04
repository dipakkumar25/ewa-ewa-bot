# src/ewa_html_processor.py
"""
Extract traffic-light KPIs from SAP EWA HTML and aggregate into 13 top exec KPIs.

Outputs:
- data/ewa_html_traffic_lights_A1C.csv           (detailed rows)
- data/ewa_html_traffic_lights_summary13_A1C.csv (13-KPI summary per week)
"""

import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from bs4 import BeautifulSoup
import pandas as pd

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
HTML_DIR = DATA_DIR / "html"
DETAIL_CSV = DATA_DIR / "ewa_html_traffic_lights_A1C.csv"
SUMMARY13_CSV = DATA_DIR / "ewa_html_traffic_lights_summary13_A1C.csv"

HTML_DIR.mkdir(parents=True, exist_ok=True)

STATUS_ORDER = {"GREEN": 0, "YELLOW": 1, "RED": 2}

PRIMARY_KPI_ORDER = [
    "Service summary",
    "Service Data Quality and Service Readiness",
    "Software Configuration for A1C",
    "Hardware Capacity",
    "Performance Overview A1C",
    "SAP System Operating A1C",
    "Security",
    "Software Change and Transport Management of A1C",
    "Financial Data Quality",
    "Upgrade Planning",
    "SAP HANA Database A1H",
    "SAP Netwear gateway",
    "UI Technologies checks",
]

PRIMARY_KPI_KEYWORDS: Dict[str, List[str]] = {
    "Service summary": ["service summary", "service summey"],
    "Service Data Quality and Service Readiness": ["data quality", "service readiness"],
    "Software Configuration for A1C": ["software configuration", "configuration for a1c"],
    "Hardware Capacity": ["hardware capacity", "cpu capacity", "disk capacity"],
    "Performance Overview A1C": ["performance overview a1c", "performance overview", "dialog response", "cpu load"],
    "SAP System Operating A1C": ["sap system operating a1c", "system operating a1c", "system operation"],
    "Security": ["security", "authorization", "password", "tls", "ssl"],
    "Software Change and Transport Management of A1C": ["transport management", "software change", "change management"],
    "Financial Data Quality": ["financial data quality"],
    "Upgrade Planning": ["upgrade planning", "maintenance strategy"],
    "SAP HANA Database A1H": ["sap hana database a1h", "hana stability", "hana resource consumption"],
    "SAP Netwear gateway": ["sap netwear gateway", "sap netweaver gateway", "gateway"],
    "UI Technologies checks": ["ui technologies checks", "fiori", "web dynpro"],
}


def parse_system_and_date_from_filename(filename: str) -> Tuple[str, datetime]:
    base = os.path.basename(filename)
    system = "A1C"
    m_s = re.search(r"EWA_([^~]+)~", base)
    if m_s:
        system = m_s.group(1)
    m_d = re.search(r"(\d{8})", base)
    if not m_d:
        raise ValueError(f"Cannot find date in filename: {base}")
    report_date = datetime.strptime(m_d.group(1), "%Y%m%d").date()
    return system, report_date


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def map_alt_to_status(alt: str):
    alt = (alt or "").lower()
    if "red" in alt:
        return "RED", "🔴"
    if "yellow" in alt:
        return "YELLOW", "🟡"
    if "green" in alt:
        return "GREEN", "🟢"
    if "no rating" in alt:
        return None
    return None


def find_section_heading(table) -> str:
    for tag in table.find_all_previous(["h1", "h2", "h3", "h4"], limit=6):
        return normalize_text(tag.get_text())
    return ""


def extract_from_html(path: Path) -> List[dict]:
    system, report_date = parse_system_and_date_from_filename(path.name)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        html = f.read()
    soup = BeautifulSoup(html, "lxml")

    rows = []
    for table in soup.find_all("table", class_="sa-table"):
        section = find_section_heading(table)
        for img in table.find_all("img"):
            mapped = map_alt_to_status(img.get("alt", ""))
            if not mapped:
                continue
            status_name, status_symbol = mapped

            kpi_text = "(no label)"
            td = img.find_parent("td")
            if td:
                # prefer right sibling as label
                sib = td.find_next_sibling("td")
                if sib:
                    t = normalize_text(sib.get_text())
                    if t:
                        kpi_text = t

            rows.append(
                {
                    "system": system,
                    "report_date": report_date,
                    "section": section,
                    "kpi_text": kpi_text,
                    "status_name": status_name,
                    "status_symbol": status_symbol,
                    "source_file": path.name,
                }
            )
    return rows


def map_to_primary(section: str, text: str) -> Optional[str]:
    combo = f"{section} {text}".lower()
    for name, kws in PRIMARY_KPI_KEYWORDS.items():
        for kw in kws:
            if kw.lower() in combo:
                return name
    return None


def build_summary(df_detail: pd.DataFrame) -> pd.DataFrame:
    df = df_detail.copy()
    df["primary_kpi"] = df.apply(
        lambda r: map_to_primary(str(r["section"]), str(r["kpi_text"])), axis=1
    )
    df = df[df["primary_kpi"].notna()]
    if df.empty:
        return df

    df["severity"] = df["status_name"].map(STATUS_ORDER)
    worst = (
        df.sort_values("severity", ascending=False)
        .groupby(["system", "report_date", "primary_kpi"])
        .first()
        .reset_index()
    )
    return worst[
        ["system", "report_date", "primary_kpi", "status_name", "status_symbol", "source_file"]
    ]


def pad_missing_kpis(df_summary: pd.DataFrame) -> pd.DataFrame:
    df = df_summary.copy()
    df["report_date"] = pd.to_datetime(df["report_date"])

    full = (
        df.set_index(["report_date", "primary_kpi"])
        .reindex(
            pd.MultiIndex.from_product(
                [sorted(df["report_date"].unique()), PRIMARY_KPI_ORDER],
                names=["report_date", "primary_kpi"],
            )
        )
        .reset_index()
    )

    full["system"] = full["system"].fillna("A1C")
    full["status_name"] = full["status_name"].fillna("GREEN")
    full["status_symbol"] = full["status_symbol"].fillna("🟢")
    full["source_file"] = full["source_file"].fillna("No alert")

    return full.sort_values(["report_date", "primary_kpi"]).reset_index(drop=True)


def main():
    html_files = sorted(HTML_DIR.glob("*.htm")) + sorted(HTML_DIR.glob("*.html"))
    if not html_files:
        print(f"No HTML files found in {HTML_DIR}")
        sys.exit(1)

    all_rows: List[dict] = []
    for path in html_files:
        print(f"Parsing: {path.name}")
        all_rows.extend(extract_from_html(path))

    df_detail = pd.DataFrame(all_rows)
    df_detail.to_csv(DETAIL_CSV, index=False)
    print(f"✔ Detailed KPI table saved → {DETAIL_CSV}")

    df_summary = build_summary(df_detail)
    if df_summary.empty:
        print("⚠ No primary KPI mappings found.")
        sys.exit(1)

    df_summary = pad_missing_kpis(df_summary)
    df_summary.to_csv(SUMMARY13_CSV, index=False)
    print(f"✔ 13-KPI summary table saved → {SUMMARY13_CSV}")
    print(df_summary.head(26))


if __name__ == "__main__":
    main()
