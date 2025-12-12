# src/ewa_html_processor.py
"""
Robust EWA HTML KPI extractor (updated):
- Detects traffic-light status from img alt/src/base64 and inline/background styles/classes
- Handles heading-level icons and table rows
- Maps rows/sections into 13 executive KPIs
- Writes DETAIL_CSV and SUMMARY_CSV (paths read from src.config)
"""
from pathlib import Path
from datetime import datetime
import re
from typing import List, Dict, Optional, Tuple
from bs4 import BeautifulSoup
import pandas as pd

# relative config import: ensure src/config.py defines HTML_DIR, DETAIL_CSV, SUMMARY_CSV
from .config import HTML_DIR, DETAIL_CSV, SUMMARY_CSV

# --- Primary KPI order / keyword mapping (13 KPIs) ---
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
    "Service summary": ["service summary", "service summey", "service overview"],
    "Service Data Quality and Service Readiness": ["data quality", "service readiness", "data integrity"],
    "Software Configuration for A1C": ["software configuration", "configuration for a1c", "parameters"],
    "Hardware Capacity": ["hardware capacity", "cpu capacity", "disk capacity", "hardware"],
    "Performance Overview A1C": ["performance overview", "dialog response", "throughput", "performance"],
    "SAP System Operating A1C": ["system operating", "system operation", "system operating a1c"],
    "Security": ["security", "authorization", "password", "critical authorizations"],
    "Software Change and Transport Management of A1C": ["transport management", "software change", "transports", "stms"],
    "Financial Data Quality": ["financial data quality", "financial"],
    "Upgrade Planning": ["upgrade planning", "maintenance strategy", "upgrade"],
    "SAP HANA Database A1H": ["sap hana database", "hana", "hana database", "index server"],
    "SAP Netwear gateway": ["sap netwear gateway", "sap netweaver gateway", "netweaver", "gateway"],
    "UI Technologies checks": ["ui technologies", "fiori", "web dynpro", "ui technologies checks"],
}

# severity mapping
SEV_ORDER = {"GREEN": 1, "YELLOW": 2, "RED": 3}
REV_SEV = {1: "GREEN", 2: "YELLOW", 3: "RED"}
SYM = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}

# ---------- helpers ----------
def _norm(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def _extract_date_from_filename(fname: str) -> Optional[datetime.date]:
    m = re.search(r"(\d{8})", fname)
    return datetime.strptime(m.group(1), "%Y%m%d").date() if m else None

def detect_status_from_img(img) -> Optional[Tuple[str,str]]:
    """Detect status from <img> (alt, title, src). Handles base64 if alt present."""
    if img is None:
        return None
    alt = (img.get("alt") or "").lower()
    title = (img.get("title") or "").lower()
    src = (img.get("src") or "").lower()
    text = f"{alt} {title} {src}"
    # alt often contains 'Yellow rating' in your HTML
    if "red" in text or "critical" in text:
        return "RED", SYM["RED"]
    if "yellow" in text or "warning" in text:
        return "YELLOW", SYM["YELLOW"]
    if "green" in text or "ok" in text or "good" in text:
        return "GREEN", SYM["GREEN"]
    return None

def detect_status_from_style(style: Optional[str], classes: List[str]=None) -> Optional[Tuple[str,str]]:
    """Detect status from inline style or CSS class names."""
    if not style and not classes:
        return None
    s = (style or "").lower()
    # rgb(...) patterns
    m = re.search(r"rgb\(([\d\s,]+)\)", s)
    if m:
        rgb = m.group(1).replace(" ", "")
        if "255,0,0" in rgb or "255,103,88" in rgb:  # some reports use different red shades
            return "RED", SYM["RED"]
        if "255,255,0" in rgb or "255,248,67" in rgb:
            return "YELLOW", SYM["YELLOW"]
        if "0,128,0" in rgb or "148,216,143" in rgb or "0,176,80" in rgb:
            return "GREEN", SYM["GREEN"]
    # look for color words
    if "background-color" in s:
        if "red" in s:
            return "RED", SYM["RED"]
        if "yellow" in s or "gold" in s:
            return "YELLOW", SYM["YELLOW"]
        if "green" in s:
            return "GREEN", SYM["GREEN"]
    # classes: some EWA CSS uses .sa-table-cell-custom1/2/3
    classes = classes or []
    cls_txt = " ".join(classes).lower()
    if "sa-table-cell-custom1" in cls_txt or "green" in cls_txt:
        return "GREEN", SYM["GREEN"]
    if "sa-table-cell-custom3" in cls_txt or "yellow" in cls_txt:
        return "YELLOW", SYM["YELLOW"]
    if "sa-table-cell-custom2" in cls_txt or "red" in cls_txt:
        return "RED", SYM["RED"]
    return None

def find_nearest_heading(elem) -> str:
    """Return nearest heading text (h1..h4) before elem"""
    for h in elem.find_all_previous(["h1","h2","h3","h4"], limit=6):
        txt = _norm(h.get_text())
        if txt:
            return txt
    # fallback to caption or strong text
    cap = elem.find_previous("caption")
    if cap:
        return _norm(cap.get_text())
    return ""

def map_to_primary(section: str, text: str) -> Optional[str]:
    combo = f"{section} {text}".lower()
    for kpi, kws in PRIMARY_KPI_KEYWORDS.items():
        for kw in kws:
            if kw in combo:
                return kpi
    return None

# ---------- parse a single HTML ----------
def parse_single_html(path: Path) -> List[dict]:
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    report_date = _extract_date_from_filename(path.name)
    system = "A1C"

    rows: List[dict] = []

    # (A) Heading-level icons: look for headings (h1..h3) that mention KPI and an adjacent img
    for heading in soup.find_all(["h1","h2","h3"]):
        htxt = _norm(heading.get_text())
        if not htxt:
            continue
        # check if heading contains a KPI keyword
        for prim in PRIMARY_KPI_ORDER:
            if prim.lower() in htxt.lower():
                # find nearest image in heading block (next or child)
                img = heading.find("img") or heading.find_next("img")
                detected = detect_status_from_img(img) or detect_status_from_style(img.get("style") if img else None, [*(img.get("class") or [])] if img else [])
                if detected:
                    rows.append({
                        "system": system,
                        "report_date": report_date,
                        "section": prim,
                        "kpi_text": prim,
                        "status_name": detected[0],
                        "status_symbol": detected[1],
                        "source_file": path.name
                    })
                break

    # (B) Table scanning: many KPIs are inside tables (td text + img or colored td)
    for tbl in soup.find_all("table"):
        section = find_nearest_heading(tbl)
        for tr in tbl.find_all("tr"):
            tds = tr.find_all(["td","th"])
            if not tds:
                continue

            # 1) try to detect img status in the row (first priority)
            img = tr.find("img")
            detected = detect_status_from_img(img) if img else None

            # 2) fallback: check inline styles / classes on each td for color
            if not detected:
                for td in tds:
                    st = detect_status_from_style(td.get("style", ""), td.get("class", []))
                    if st:
                        detected = st
                        break

            if not detected:
                continue

            # extract label: prefer the longest textual TD (>2 chars) excluding small numeric cells
            texts = [ _norm(td.get_text(" ", strip=True)) for td in tds ]
            candidates = [(i,t) for i,t in enumerate(texts) if t and len(t) > 2 and not re.fullmatch(r"\d+", t)]
            label = ""
            if candidates:
                # pick longest candidate
                label = max(candidates, key=lambda x: len(x[1]))[1]
            else:
                # fallback to first non-empty
                label = next((t for t in texts if t), "")

            rows.append({
                "system": system,
                "report_date": report_date,
                "section": section,
                "kpi_text": label or "(no label)",
                "status_name": detected[0],
                "status_symbol": detected[1],
                "source_file": path.name
            })

    # (C) If key primary KPIs still missing, search text for KPI names and pick nearest image
    present = {r["section"] for r in rows}
    for prim in PRIMARY_KPI_ORDER:
        if prim in present:
            continue
        found_text_node = soup.find(text=re.compile(re.escape(prim), re.IGNORECASE))
        if found_text_node:
            parent = found_text_node.parent
            img = parent.find_next("img") or parent.find_previous("img")
            detected = detect_status_from_img(img) if img else None
            if not detected:
                # maybe the status is indicated via the parent td style/class
                td = parent.find_parent("td") if parent else None
                if td:
                    detected = detect_status_from_style(td.get("style",""), td.get("class", []))
            if detected:
                rows.append({
                    "system": system,
                    "report_date": report_date,
                    "section": prim,
                    "kpi_text": prim,
                    "status_name": detected[0],
                    "status_symbol": detected[1],
                    "source_file": path.name
                })

    return rows

# ---------- build details & summary ----------
def build_detail_and_summary():
    html_dir = Path(HTML_DIR)
    files = sorted([p for p in html_dir.glob("*.htm")] + [p for p in html_dir.glob("*.html")])
    if not files:
        raise FileNotFoundError(f"No HTML files found in {HTML_DIR}")

    all_rows = []
    for p in files:
        print(f"Processing: {p.name}")
        parsed = parse_single_html(p)
        print(f"  -> {len(parsed)} entries")
        all_rows.extend(parsed)

    if not all_rows:
        raise RuntimeError("No KPI entries extracted — check HTML format.")

    df_detail = pd.DataFrame(all_rows)
    df_detail["report_date"] = pd.to_datetime(df_detail["report_date"])
    df_detail.to_csv(DETAIL_CSV, index=False)
    print(f"✔ Detail saved -> {DETAIL_CSV} ({len(df_detail)} rows)")

    # map to primary KPI using section+kpi_text
    df_detail["primary_kpi"] = df_detail.apply(lambda r: map_to_primary(str(r["section"] or ""), str(r["kpi_text"] or "")), axis=1)

    # keep only mapped rows for summary
    df_mapped = df_detail[df_detail["primary_kpi"].notna()].copy()
    if df_mapped.empty:
        raise RuntimeError("No rows mapped to primary KPI — check keyword mapping.")

    # compute worst status per KPI per date
    df_mapped["severity"] = df_mapped["status_name"].map(SEV_ORDER)
    grouped = df_mapped.sort_values("severity", ascending=False).groupby(["system","report_date","primary_kpi"], as_index=False).agg({
        "severity":"max",
        "status_name":"first",
        "status_symbol":"first",
        "source_file":"first"
    })
    # map back severity->name and symbol
    grouped["status_name"] = grouped["severity"].map(lambda s: REV_SEV.get(s, "GREEN"))
    grouped["status_symbol"] = grouped["status_name"].map(SYM)

    summary = grouped[["system","report_date","primary_kpi","status_name","status_symbol","source_file"]]

    # pad missing KPIs for each date as GREEN
    all_dates = sorted(summary["report_date"].unique())
    idx = pd.MultiIndex.from_product([["A1C"], all_dates, PRIMARY_KPI_ORDER], names=["system","report_date","primary_kpi"])
    pad = summary.set_index(["system","report_date","primary_kpi"]).reindex(idx).reset_index()
    pad["status_name"] = pad["status_name"].fillna("GREEN")
    pad["status_symbol"] = pad["status_symbol"].fillna(SYM["GREEN"])
    pad["source_file"] = pad["source_file"].fillna("No alert")
    pad.to_csv(SUMMARY_CSV, index=False)
    print(f"✔ Summary saved -> {SUMMARY_CSV} ({len(pad)} rows)")

    return df_detail, pad

# ---------------- main ----------------
def main():
    df_detail, df_summary = build_detail_and_summary()
    # show first block for user feedback
    print(df_summary.head(13))

if __name__ == "__main__":
    main()
