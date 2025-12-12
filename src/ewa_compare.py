# src/ewa_compare.py

import pandas as pd
from pathlib import Path

# CSV file paths
SUMMARY = Path("data/ewa_html_traffic_lights_summary13_A1C.csv")
DETAIL = Path("data/ewa_html_traffic_lights_A1C.csv")

STATUS_RANK = {"GREEN": 1, "YELLOW": 2, "RED": 3}
CHANGE_LABEL = {
    1: "➕ Improvement",
    0: "🔄 No Change",
    -1: "➖ Deterioration"
}

def compare_summary(date1, date2):
    print(f"\n=== EXECUTIVE 13 KPI COMPARISON ({date2} vs {date1}) ===\n")

    df = pd.read_csv(SUMMARY)
    df["report_date"] = pd.to_datetime(df["report_date"]).dt.date

    df1 = df[df["report_date"] == date1][["primary_kpi", "status_name", "source_file"]]
    df2 = df[df["report_date"] == date2][["primary_kpi", "status_name", "source_file"]]

    merged = df1.merge(df2, on="primary_kpi", suffixes=("_old", "_new"))

    def classify(old, new):
        diff = STATUS_RANK[new] - STATUS_RANK[old]
        if diff > 0: return "➖ Deterioration"
        if diff < 0: return "➕ Improvement"
        return "🔄 No Change"

    merged["Change"] = merged.apply(
        lambda r: classify(r["status_name_old"], r["status_name_new"]),
        axis=1
    )

    print(merged.to_string(index=False))
    return merged


def compare_detail(date1, date2):
    print(f"\n=== DETAILED KPI ROW COMPARISON ({date2} vs {date1}) ===\n")

    df = pd.read_csv(DETAIL)
    df["report_date"] = pd.to_datetime(df["report_date"]).dt.date

    d1 = df[df["report_date"] == date1]
    d2 = df[df["report_date"] == date2]

    # merge detailed KPIs by kpi_text
    merged = d1.merge(d2, on="kpi_text", suffixes=("_old", "_new"))

    def classify(old, new):
        diff = STATUS_RANK[new] - STATUS_RANK[old]
        if diff > 0: return "➖ Worse"
        if diff < 0: return "➕ Better"
        return "🔄 Same"

    merged["Detail_Change"] = merged.apply(
        lambda r: classify(r["status_name_old"], r["status_name_new"]),
        axis=1
    )

    print(merged[[
        "kpi_text", "status_name_old", "status_name_new", "Detail_Change"
    ]].to_string(index=False))

    return merged


def main():
    # Choose dates to compare
    date_old = pd.to_datetime("2025-11-18").date()
    date_new = pd.to_datetime("2025-11-25").date()

    print("\nComparing:", date_old, "→", date_new)

    summary_cmp = compare_summary(date_old, date_new)
    detail_cmp = compare_detail(date_old, date_new)

    # Save comparison reports
    summary_cmp.to_csv("data/EWA_Compare_Summary.csv", index=False)
    detail_cmp.to_csv("data/EWA_Compare_Detail.csv", index=False)

    print("\n✔ Comparison CSV saved:")
    print("  - data/EWA_Compare_Summary.csv")
    print("  - data/EWA_Compare_Detail.csv")

if __name__ == "__main__":
    main()
