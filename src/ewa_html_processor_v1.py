# src/ewa_html_processor.py
"""
Robust EWA HTML KPI extractor:
- Detects traffic-light status from icons (img alt/src/base64) and inline styles
- Reads icons under headings and inside tables
- Maps rows/sections into 13 executive KPIs
- Writes DETAIL_CSV and SUMMARY_CSV (paths from src.config)
"""

import re
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict

from bs4 import BeautifulSoup
import pandas as pd

# relative config import
from .config import HTML_DIR, DETAIL_CSV, SUMMARY_CSV

# ---------------------- KPI DEFINITIONS ----------------------
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
    "Service summary": ["service summary", "service overview"],
    "Service Data Quality and Service Readiness": ["data quality", "service readiness"],
    "Software Configuration for A1C": ["software configuration"],
    "Hardware Capacity": ["hardware", "cpu", "disk", "memory", "capacity"],
    "Performance Overview A1C": ["performance overview", "response time", "dialog"],
    "SAP System Operating A1C": ["system operating", "background jobs"],
    "Security": ["security", "authorization", "tls", "ssl", "vulnerab"],
    "Software Change and Transport Management of A1C": ["transport", "change management", "stms"],
    "Financial Data Quality": ["financial"],
    "Upgrade Planning": ["upgrade", "maintenance"],
    "SAP HANA Database A1H": ["hana", "hdb", "index server"],
    "SAP Netwear gateway": ["gateway", "netweaver"],
    "UI Technologies checks": ["ui", "fiori", "web dynpro"],
}

SEVERITY_ORDER = {"GREEN": 1, "YELLOW": 2, "RED": 3}
REV_SEVERITY = {1: "GREEN", 2: "YELLOW", 3: "RED"}
SYM = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}


# ---------------------- HELPER FUNCTIONS ----------------------
def normalize(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def extract_date_from_filename(fname: str) -> Optional[datetime.date]:
    m = re.search(r"(\d{8})", fname)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%d").date()


def detect_status_from_img(img) -> Optional[Tuple[str, str]]:
    """Detect RED / YELLOW / GREEN even if alt/src text is not perfect."""
    if img is None:
        return None

    alt = (img.get("alt") or "").lower()
    src = (img.get("src") or "").lower()
    text = f"{alt} {src}"

    # RED
    if any(k in text for k in ("red", "critical", "high", "severe")):
        return "RED", SYM["RED"]

    # YELLOW
    if any(k in text for k in ("yellow", "warn", "warning", "medium")):
        return "YELLOW", SYM["YELLOW"]

    # GREEN — expanded keywords
    if any(k in text for k in ("green", "ok", "good", "healthy", "normal", "low", "success")):
        return "GREEN", SYM["GREEN"]

    # file-pattern fallbacks
    if any(x in src for x in ("green.png", ".green.", "_green", "green.gif", "traffic_light_green")):
        return "GREEN", SYM["GREEN"]
    if any(x in src for x in ("yellow.png", "_yellow", "yellow.gif")):
        return "YELLOW", SYM["YELLOW"]
    if any(x in src for x in ("red.png", "_red", "red.gif")):
        return "RED", SYM["RED"]

    return None


def detect_status_from_style(style: Optional[str]) -> Optional[Tuple[str, str]]:
    if not style:
        return None
    s = style.lower()

    # rgb() with spaces
    m = re.search(r"rgb\(\s*([\d\s,]+)\s*\)", s)
    if m:
        rgb = m.group(1).replace(" ", "")
        if rgb == "255,0,0":
            return "RED", SYM["RED"]
        if rgb == "255,255,0":
            return "YELLOW", SYM["YELLOW"]
        if rgb in ("0,128,0", "0,176,80"):
            return "GREEN", SYM["GREEN"]

    # hex colors
    m2 = re.search(r"#([0-9a-f]{6})", s)
    if m2:
        hexv = m2.group(1)
        if hexv in ("ff0000",):
            return "RED", SYM["RED"]
        if hexv in ("ffff00",):
            return "YELLOW", SYM["YELLOW"]
        if hexv in ("00b050", "008000", "00ff00"):
            return "GREEN", SYM["GREEN"]

    # color words
    if "background-color" in s:
        if "red" in s:
            return "RED", SYM["RED"]
        if "yellow" in s:
            return "YELLOW", SYM["YELLOW"]
        if "green" in s:
            return "GREEN", SYM["GREEN"]

    return None


def find_nearest_heading(elem) -> str:
    for tag in elem.find_all_previous(["h1", "h2", "h3", "h4"], limit=6):
        t = normalize(tag.get_text())
        if t:
            return t
    return ""


def map_to_primary_kpi(section: str, text: str) -> Optional[str]:
    combo = f"{section} {text}".lower()
    for kpi, keys in PRIMARY_KPI_KEYWORDS.items():
        if any(k in combo for k in keys):
            return kpi
    return None


# ---------------------- PARSE HTML ----------------------
def parse_single_html(path: Path) -> List[dict]:
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    system = "A1C"
    report_date = extract_date_from_filename(path.name)
    rows: List[dict] = []

    # 1) Heading-based icons
    for h in soup.find_all(["h1", "h2", "h3"]):
        txt = normalize(h.get_text())
        for prim in PRIMARY_KPI_ORDER:
            if prim.lower() in txt:
                img = h.find_next("img")
                status = detect_status_from_img(img)
                if status:
                    rows.append({
                        "system": system,
                        "report_date": report_date,
                        "section": prim,
                        "kpi_text": prim,
                        "status_name": status[0],
                        "status_symbol": status[1],
                        "source_file": path.name
                    })

    # 2) Table row scanning — FIXED VERSION WITH GREEN SUPPORT
    for tbl in soup.find_all("table"):
        section = find_nearest_heading(tbl)
        for tr in tbl.find_all("tr"):
            tds = tr.find_all(["td", "th"])
            if not tds:
                continue

            # Scan EVERY TD for img/style (fixes missing green)
            detected = None
            for td in tds:
                for img in td.find_all("img"):
                    detected = detect_status_from_img(img)
                    if detected:
                        break
                if detected:
                    break

                style_status = detect_status_from_style(td.get("style"))
                if style_status:
                    detected = style_status
                    break

            if not detected:
                continue

            # improved label detection
            texts = [
                normalize(td.get_text(" ", strip=True))
                for td in tds
            ]
            label_candidates = [t for t in texts if len(t) > 2 and not re.fullmatch(r"\d+", t)]
            label = max(label_candidates, key=len) if label_candidates else texts[0]

            rows.append({
                "system": system,
                "report_date": report_date,
                "section": section,
                "kpi_text": label,
                "status_name": detected[0],
                "status_symbol": detected[1],
                "source_file": path.name,
            })

    return rows


# ---------------------- BUILD DATAFRAMES ----------------------
def build_detail_and_summary():
    folder = Path(HTML_DIR)
    files = sorted(folder.glob("*.htm")) + sorted(folder.glob("*.html"))
    if not files:
        raise FileNotFoundError(f"No HTML files found in {HTML_DIR}")

    all_rows = []
    for f in files:
        print(f"Processing: {f.name}")
        parsed = parse_single_html(f)
        print(f"  → {len(parsed)} entries")
        all_rows.extend(parsed)

    df_detail = pd.DataFrame(all_rows)
    df_detail["report_date"] = pd.to_datetime(df_detail["report_date"])
    df_detail.to_csv(DETAIL_CSV, index=False)

    # Map detail rows into 13 KPIs
    df_detail["primary_kpi"] = df_detail.apply(
        lambda r: map_to_primary_kpi(str(r["section"]), str(r["kpi_text"])),
        axis=1
    )
    df_sum = df_detail.dropna(subset=["primary_kpi"]).copy()

    # severity
    df_sum["severity"] = df_sum["status_name"].map(SEVERITY_ORDER)
    grouped = df_sum.groupby(
        ["system", "report_date", "primary_kpi"], as_index=False
    ).agg({"severity": "max"})

    grouped["status_name"] = grouped["severity"].map(REV_SEVERITY)
    grouped["status_symbol"] = grouped["status_name"].map(SYM)

    # pad missing KPIs
    all_dates = sorted(grouped["report_date"].unique())
    idx = pd.MultiIndex.from_product(
        [["A1C"], all_dates, PRIMARY_KPI_ORDER],
        names=["system", "report_date", "primary_kpi"]
    )

    summary = grouped.set_index(
        ["system", "report_date", "primary_kpi"]
    ).reindex(idx).reset_index()

    summary["status_name"].fillna("GREEN", inplace=True)
    summary["status_symbol"].fillna(SYM["GREEN"], inplace=True)

    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"✔ Summary saved to {SUMMARY_CSV}")

    return df_detail, summary


def main():
    df_detail, df_summary = build_detail_and_summary()
    print(df_summary.head(13))


if __name__ == "__main__":
    main()
