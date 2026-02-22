"""
╔══════════════════════════════════════════════════════════════════════╗
║     SAP EWA Intelligence Hub – Enhanced Dashboard                   ║
║     AI: Trend · Forecast · Anomaly Detection · Early Warning        ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SAP EWA Intelligence Hub",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# GLOBAL CSS  –  Dark luxe terminal aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #E2E8F0;
}
.stApp {
    background: #080C14;
}
section[data-testid="stSidebar"] {
    background: #0D1320;
    border-right: 1px solid #1E293B;
}
section[data-testid="stSidebar"] * { color: #CBD5E1; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem; max-width: 1600px; }

/* ── Custom header ── */
.ewa-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1.2rem 1.8rem;
    background: linear-gradient(135deg, #0F1923 0%, #111827 60%, #0A1628 100%);
    border: 1px solid #1E3A5F;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.ewa-header::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 80% 50%, rgba(56,189,248,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.ewa-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.55rem;
    font-weight: 700;
    color: #F0F9FF;
    letter-spacing: -0.5px;
    line-height: 1.1;
}
.ewa-sub {
    font-size: 0.8rem;
    color: #64748B;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 2px;
}
.ewa-badge {
    background: rgba(56,189,248,0.12);
    border: 1px solid rgba(56,189,248,0.3);
    color: #38BDF8;
    padding: 4px 12px;
    border-radius: 20px;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.05em;
    white-space: nowrap;
}
.live-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #22C55E;
    box-shadow: 0 0 8px #22C55E;
    animation: pulse 2s infinite;
    display: inline-block;
    margin-right: 6px;
}
@keyframes pulse {
    0%,100% { opacity: 1; }
    50%      { opacity: 0.4; }
}

/* ── Metric cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 14px;
    margin-bottom: 1.5rem;
    width: 100%;
}
.metric-card {
    background: #0F1923;
    border: 1px solid #1E293B;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #334155; }
.metric-card::after {
    content: '';
    position: absolute; bottom: 0; left: 0; right: 0;
    height: 2px;
    border-radius: 0 0 10px 10px;
}
.metric-card.green::after  { background: linear-gradient(90deg,#22C55E,#16A34A); }
.metric-card.yellow::after { background: linear-gradient(90deg,#EAB308,#CA8A04); }
.metric-card.red::after    { background: linear-gradient(90deg,#EF4444,#B91C1C); }
.metric-card.blue::after   { background: linear-gradient(90deg,#38BDF8,#0284C7); }
.metric-label {
    font-size: 0.7rem;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-family: 'Space Mono', monospace;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #F0F9FF;
    line-height: 1;
    font-family: 'Space Mono', monospace;
}
.metric-delta {
    font-size: 0.72rem;
    margin-top: 4px;
    color: #64748B;
}
.metric-delta.up   { color: #EF4444; }
.metric-delta.down { color: #22C55E; }

/* ── Section headers ── */
.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #38BDF8;
    border-left: 3px solid #38BDF8;
    padding-left: 10px;
    margin: 1.5rem 0 0.8rem;
}

/* ── Status pills ── */
.pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    font-family: 'Space Mono', monospace;
}
.pill-green  { background: rgba(34,197,94,0.12);  color: #22C55E;  border:1px solid rgba(34,197,94,0.25); }
.pill-yellow { background: rgba(234,179,8,0.12);  color: #EAB308;  border:1px solid rgba(234,179,8,0.25); }
.pill-red    { background: rgba(239,68,68,0.12);  color: #EF4444;  border:1px solid rgba(239,68,68,0.25); }

/* ── Warning banner ── */
.warn-banner {
    background: linear-gradient(90deg, rgba(239,68,68,0.08), rgba(239,68,68,0.04));
    border: 1px solid rgba(239,68,68,0.25);
    border-left: 4px solid #EF4444;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.82rem;
    color: #FCA5A5;
}
.info-banner {
    background: linear-gradient(90deg, rgba(56,189,248,0.08), rgba(56,189,248,0.03));
    border: 1px solid rgba(56,189,248,0.2);
    border-left: 4px solid #38BDF8;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.82rem;
    color: #BAE6FD;
}
.ok-banner {
    background: linear-gradient(90deg, rgba(34,197,94,0.08), rgba(34,197,94,0.03));
    border: 1px solid rgba(34,197,94,0.2);
    border-left: 4px solid #22C55E;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.82rem;
    color: #86EFAC;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: transparent;
    border-bottom: 1px solid #1E293B;
    padding-bottom: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border: 1px solid #1E293B;
    border-bottom: none;
    border-radius: 8px 8px 0 0;
    color: #64748B;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.04em;
    padding: 8px 16px;
    transition: all 0.15s;
}
.stTabs [aria-selected="true"] {
    background: #0F1923 !important;
    color: #38BDF8 !important;
    border-color: #1E3A5F !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #CBD5E1; }

/* ── Dataframe ── */
.dataframe-container {
    border: 1px solid #1E293B;
    border-radius: 8px;
    overflow: hidden;
}
[data-testid="stDataFrame"] { border: 1px solid #1E293B; border-radius: 8px; }
.stDataFrame th {
    background: #0F1923 !important;
    color: #64748B !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.68rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Selectbox / sliders ── */
[data-testid="stSelectbox"] label, [data-testid="stSlider"] label { color: #94A3B8; }
[data-baseweb="select"] { background: #0F1923 !important; border-color: #1E293B !important; }

/* ── Sidebar widgets ── */
.css-1d391kg { background: #0D1320; }
[data-testid="stSidebar"] .stSelectbox label { color: #94A3B8; font-size: 0.8rem; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #080C14; }
::-webkit-scrollbar-thumb { background: #1E293B; border-radius: 3px; }

/* ── Anomaly table highlight ── */
.anomaly-row { background: rgba(239,68,68,0.06); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
DATA_DIR = Path("data")
MASTER_FILE = DATA_DIR / "ewa_kpi_clean_summary_all.csv"
LEGACY_FILE = DATA_DIR / "ewa_html_traffic_lights_summary13_A1C.csv"
DETAIL_FILE  = DATA_DIR / "ewa_html_traffic_lights_A1C.csv"

STATUS_MAP     = {"GREEN": 1, "YELLOW": 2, "RED": 3}
REV_STATUS     = {1: "GREEN", 2: "YELLOW", 3: "RED"}
STATUS_RANK_HI = {"GREEN": 3, "YELLOW": 2, "RED": 1}  # higher = better
COLOR_HEX = {"GREEN": "#22C55E", "YELLOW": "#EAB308", "RED": "#EF4444"}
PILL_HTML = {
    "GREEN":  '<span class="pill pill-green">🟢 GREEN</span>',
    "YELLOW": '<span class="pill pill-yellow">🟡 YELLOW</span>',
    "RED":    '<span class="pill pill-red">🔴 RED</span>',
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#CBD5E1", size=12),
    xaxis=dict(gridcolor="#1E293B", zeroline=False, color="#64748B"),
    yaxis=dict(gridcolor="#1E293B", zeroline=False, color="#64748B"),
    margin=dict(l=10, r=10, t=40, b=10),
)

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data() -> tuple[pd.DataFrame, str]:
    if MASTER_FILE.exists():
        df = pd.read_csv(MASTER_FILE)
        for alias in ("KPI name", "clean_section", "primary_kpi"):
            if alias in df.columns and "KPI name" not in df.columns:
                df.rename(columns={alias: "KPI name"}, inplace=True)
        if "status_name" not in df.columns and "final_status" in df.columns:
            df["status_name"] = df["final_status"]
        return df, "master"
    if LEGACY_FILE.exists():
        df = pd.read_csv(LEGACY_FILE)
        if "primary_kpi" in df.columns:
            df.rename(columns={"primary_kpi": "KPI name"}, inplace=True)
        return df, "legacy"
    return pd.DataFrame(), "none"


def prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce", dayfirst=True)
    df = df[df["report_date"].notna()]
    if "KPI name" not in df.columns:
        for col in ("clean_section", "primary_kpi", "section"):
            if col in df.columns:
                df.rename(columns={col: "KPI name"}, inplace=True)
                break
    df["status_name"] = df["status_name"].astype(str).str.upper().str.strip()
    df["severity"] = df["status_name"].map(STATUS_MAP)
    df = df.dropna(subset=["severity"])
    # deduplicate – worst wins
    df = (df.sort_values("severity", ascending=False)
            .groupby(["system", "report_date", "KPI name"], as_index=False)
            .first())
    return df.sort_values(["system", "report_date", "KPI name"]).reset_index(drop=True)


@st.cache_data(ttl=300)
def load_detail() -> pd.DataFrame:
    if DETAIL_FILE.exists():
        d = pd.read_csv(DETAIL_FILE)
        d["report_date"] = pd.to_datetime(d["report_date"], errors="coerce")
        return d
    return pd.DataFrame()


# ─────────────────────────────────────────────
# ML / ANALYTICS FUNCTIONS
# ─────────────────────────────────────────────
def compute_trend(df: pd.DataFrame, kpi_col: str = "KPI name") -> pd.DataFrame:
    """Linear regression slope per (system, KPI) over time."""
    out = []
    group_cols = [c for c in ["system", kpi_col] if c in df.columns]
    for key, grp in df.groupby(group_cols, dropna=False):
        grp = grp.sort_values("report_date")
        y = grp["severity"].values.astype(float)
        if len(y) < 3:
            continue
        t = np.arange(len(y)).reshape(-1, 1)
        reg = LinearRegression().fit(t, y)
        slope = float(reg.coef_[0])
        r2    = float(reg.score(t, y))
        if slope > 0.08:
            direction = "🔴 Deteriorating"
        elif slope < -0.08:
            direction = "🟢 Improving"
        else:
            direction = "🔵 Stable"
        if isinstance(key, tuple):
            sys_, kpi = (key[0], key[-1]) if len(key) >= 2 else (None, key[0])
        else:
            sys_, kpi = None, key
        out.append({
            "system": sys_, "kpi": kpi,
            "slope": round(slope, 4), "r2": round(r2, 3),
            "trend": direction, "n_weeks": len(y),
            "current_severity": int(y[-1]),
        })
    return pd.DataFrame(out)


def linear_forecast(y: np.ndarray, periods: int):
    t = np.arange(len(y)).reshape(-1, 1)
    reg = LinearRegression().fit(t, y)
    t_f = np.arange(len(y), len(y) + periods).reshape(-1, 1)
    pred = reg.predict(t_f)
    residuals = y - reg.predict(t).flatten()
    std = residuals.std() if len(residuals) > 1 else 0.2
    return np.clip(pred, 1, 3), std


def forecast_severity(df: pd.DataFrame, kpi_col: str = "KPI name",
                      system: str = None, periods: int = 4) -> pd.DataFrame:
    if system:
        df = df[df["system"] == system]
    all_fc = []
    group_cols = [c for c in ["system", kpi_col] if c in df.columns]
    for key, grp in df.groupby(group_cols, dropna=False):
        grp = grp.sort_values("report_date").drop_duplicates("report_date")
        y = grp["severity"].values.astype(float)
        if len(y) < 3:
            continue
        sys_   = key[0] if isinstance(key, tuple) and len(key) >= 2 else None
        kpi_nm = key[-1] if isinstance(key, tuple) else key
        dates  = pd.to_datetime(grp["report_date"])
        freq   = "7D"  # weekly
        future_dates = pd.date_range(dates.iloc[-1], periods=periods + 1, freq=freq)[1:]
        pred, std = linear_forecast(y, periods)
        for i, d in enumerate(future_dates):
            all_fc.append({
                "system": sys_, "kpi": kpi_nm,
                "report_date": d,
                "severity_pred": float(np.clip(pred[i], 1, 3)),
                "upper": float(np.clip(pred[i] + 1.28 * std, 1, 3)),
                "lower": float(np.clip(pred[i] - 1.28 * std, 1, 3)),
            })
    return pd.DataFrame(all_fc)


def detect_anomalies(df: pd.DataFrame, kpi_col: str = "KPI name",
                     system: str = None, window: int = 4, z_thresh: float = 1.8) -> pd.DataFrame:
    """Z-score anomaly + sudden spike detection."""
    if system:
        df = df[df["system"] == system]
    anomalies = []
    group_cols = [c for c in ["system", kpi_col] if c in df.columns]
    for key, grp in df.groupby(group_cols, dropna=False):
        grp = grp.sort_values("report_date").reset_index(drop=True)
        y = grp["severity"].values.astype(float)
        if len(y) < window + 1:
            continue
        sys_   = key[0] if isinstance(key, tuple) and len(key) >= 2 else None
        kpi_nm = key[-1] if isinstance(key, tuple) else key
        for i in range(window, len(y)):
            window_y  = y[i - window:i]
            mu, sigma = window_y.mean(), window_y.std()
            if sigma < 0.01:
                sigma = 0.01
            z = (y[i] - mu) / sigma
            delta = y[i] - y[i - 1]
            anom_type = None
            if z > z_thresh and y[i] >= 2:
                anom_type = "Statistical Spike"
            elif delta >= 2:
                anom_type = "Sudden Jump (GREEN→RED)"
            elif delta >= 1 and y[i] == 3:
                anom_type = "Step to RED"
            if anom_type:
                anomalies.append({
                    "system": sys_, "kpi": kpi_nm,
                    "report_date": grp["report_date"].iloc[i],
                    "severity": int(y[i]),
                    "z_score": round(z, 2),
                    "delta": int(delta),
                    "anomaly_type": anom_type,
                    "status": REV_STATUS.get(int(y[i]), "?"),
                })
    return pd.DataFrame(anomalies)


def attention_score(df: pd.DataFrame, kpi_col: str = "KPI name",
                    system: str = None) -> pd.DataFrame:
    """Composite attention ranking."""
    trends = compute_trend(df, kpi_col=kpi_col)
    if system:
        df = df[df["system"] == system]
    rows = []
    group_cols = [c for c in ["system", kpi_col] if c in df.columns]
    trend_dir_map = {"🔴 Deteriorating": 1.0, "🔵 Stable": 0.4, "🟢 Improving": 0.0}

    for key, grp in df.groupby(group_cols, dropna=False):
        grp = grp.sort_values("report_date")
        y   = grp["severity"].values.astype(float)
        sys_   = key[0] if isinstance(key, tuple) and len(key) >= 2 else None
        kpi_nm = key[-1] if isinstance(key, tuple) else key

        current  = float(y[-1])
        red_pct  = float((y == 3).mean())
        vol      = float(np.diff(y).std()) if len(y) > 2 else 0.0
        consec   = 0
        for v in reversed(y):
            if v >= 2:
                consec += 1
            else:
                break

        tr_row = trends[trends["kpi"] == kpi_nm]
        if not tr_row.empty and "system" in tr_row.columns and sys_:
            tr_row = tr_row[tr_row["system"] == sys_]
        tdir  = tr_row["trend"].iloc[0] if not tr_row.empty else "🔵 Stable"
        slope = tr_row["slope"].iloc[0]  if not tr_row.empty else 0.0

        score = (
            0.30 * (current - 1) / 2.0
            + 0.25 * trend_dir_map.get(tdir, 0.4)
            + 0.25 * red_pct
            + 0.10 * min(vol / 1.5, 1.0)
            + 0.10 * min(consec / 4, 1.0)
        )
        rows.append({
            "system": sys_, "KPI": kpi_nm,
            "Attention Score": round(score, 3),
            "Current": REV_STATUS.get(int(current), "?"),
            "Trend": tdir, "Red %": f"{red_pct*100:.0f}%",
            "Volatility": round(vol, 2),
            "Consec. Weeks": consec,
        })
    return pd.DataFrame(rows).sort_values("Attention Score", ascending=False).reset_index(drop=True)


def kpi_correlation(df: pd.DataFrame, kpi_col: str = "KPI name") -> pd.DataFrame:
    """Pivot to KPI columns and compute Pearson correlation matrix."""
    pivot = df.pivot_table(index="report_date", columns=kpi_col,
                            values="severity", aggfunc="max")
    pivot.columns = [c[:22] for c in pivot.columns]  # truncate for display
    return pivot.corr().round(2)


def wow_delta(df: pd.DataFrame, kpi_col: str = "KPI name") -> pd.DataFrame:
    """Week-over-week change for every KPI in the latest two dates."""
    dates = sorted(df["report_date"].unique())
    if len(dates) < 2:
        return pd.DataFrame()
    d1, d2 = dates[-2], dates[-1]
    m1 = df[df["report_date"] == d1].set_index(kpi_col)["status_name"]
    m2 = df[df["report_date"] == d2].set_index(kpi_col)["status_name"]
    merged = m1.to_frame("prev").join(m2.to_frame("curr"), how="outer")
    def classify(r):
        old, new = r["prev"], r["curr"]
        if pd.isna(old) or pd.isna(new):
            return "🆕 New"
        if STATUS_MAP.get(new, 0) > STATUS_MAP.get(old, 0):
            return "📈 Deteriorated"
        if STATUS_MAP.get(new, 0) < STATUS_MAP.get(old, 0):
            return "📉 Improved"
        return "➡️ No Change"
    merged["change"] = merged.apply(classify, axis=1)
    merged["date"] = d2
    return merged.reset_index().rename(columns={"index": kpi_col})


def root_cause_summary(detail_df: pd.DataFrame, kpi_name: str,
                        date, n: int = 4) -> str:
    if detail_df is None or detail_df.empty:
        return "Detail data not available."
    try:
        kpi_cols = [c for c in detail_df.columns
                    if c in ("primary_kpi", "section", "clean_section", "kpi")]
        if not kpi_cols:
            return "No KPI column in detail file."
        kc = kpi_cols[0]
        # Safely coerce column to string, dropping NaN rows
        col_str = detail_df[kc].fillna("").astype(str).str.strip().str.lower()
        mask_kpi  = col_str == str(kpi_name).strip().lower()
        mask_date = detail_df["report_date"] == date
        sub = detail_df[mask_kpi & mask_date]
        if sub.empty:
            return "No detail rows found."
        texts = []
        for _, row in sub.iterrows():
            txt = str(row.get("kpi_text", "") or "")
            txt = re.sub(r"^\d+(\.\d+)*\s*", "", txt).strip()
            if len(txt) > 5:
                texts.append(txt)
        return " | ".join(texts[:n]) if texts else "No description available."
    except Exception as e:
        return f"Root cause unavailable ({e})"


# ─────────────────────────────────────────────
# PLOTLY HELPER
# ─────────────────────────────────────────────
def apply_dark(fig, title: str = "", height: int = 420) -> go.Figure:
    fig.update_layout(**PLOTLY_LAYOUT, title=dict(text=title, font=dict(size=13, color="#CBD5E1")), height=height)
    return fig


def status_color_scale():
    return [[0.0, COLOR_HEX["GREEN"]], [0.5, COLOR_HEX["YELLOW"]], [1.0, COLOR_HEX["RED"]]]


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
raw_df, source = load_data()
if raw_df.empty:
    st.error("❌ No SAP EWA data files found. Please run ewa_html_processor and ewa_kpi_master_builder first.")
    st.stop()

df_all = prep(raw_df)
detail_df = load_detail()

# ─────────────────────────────────────────────
# SIDEBAR  (ML sliders only – optional panel)
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="font-family:Space Mono,monospace;font-size:0.65rem;color:#38BDF8;'
        'text-transform:uppercase;letter-spacing:0.15em;margin-bottom:16px">'
        '⚙️ ML Settings</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div style="font-size:0.7rem;color:#64748B;margin-bottom:6px">Early Warning</div>', unsafe_allow_html=True)
    ew_window  = st.slider("Rolling window (weeks)", 2, 6, 3)
    ew_z       = st.slider("Z-score threshold", 1.0, 3.0, 1.8, 0.1)
    st.markdown('<div style="font-size:0.7rem;color:#64748B;margin-top:12px;margin-bottom:6px">Forecast</div>', unsafe_allow_html=True)
    fc_periods = st.slider("Forecast weeks", 2, 8, 4)


# ─────────────────────────────────────────────
# MAIN CONTROL BAR  (always visible)
# ─────────────────────────────────────────────
kpi_col = "KPI name"

# Header row
st.markdown("""
<div class="ewa-header">
  <div style="font-size:2rem">🛰️</div>
  <div style="flex:1">
    <div class="ewa-title">SAP EWA Intelligence Hub</div>
    <div class="ewa-sub">Early Watch Alert · Multi-SID · AI-Powered Risk Analysis</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Control bar: SID + date range + compare picker — always in main area
ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4 = st.columns([1, 1, 2, 1])

with ctrl_c1:
    systems = sorted(df_all["system"].unique())
    selected_sid = st.selectbox("🖥️ System SID", systems, index=len(systems) - 1)

df = df_all[df_all["system"] == selected_sid].copy()
dates = sorted(df["report_date"].unique())
kpis = sorted(df[kpi_col].unique())
_date_str_list = [str(d.date()) for d in dates]

with ctrl_c2:
    # Latest report date selector
    picked_latest = st.selectbox(
        "📅 Active Report Date",
        options=_date_str_list,
        index=len(_date_str_list) - 1,
    )
    # Override dates[-1] with user pick for weekly view
    active_date = pd.Timestamp(picked_latest)

with ctrl_c3:
    _default_cmp = _date_str_list[-2:] if len(_date_str_list) >= 2 else _date_str_list
    compare_dates = st.multiselect(
        "🔍 Compare Dates (pick 2 for Tab 4)",
        options=_date_str_list,
        default=_default_cmp,
    )

with ctrl_c4:
    if dates:
        date_range_str = f"{dates[0].strftime('%d %b %Y')} → {dates[-1].strftime('%d %b %Y')}"
        st.markdown(
            f'''<div style="background:#0A1628;border:1px solid #1E3A5F;border-radius:8px;
                         padding:10px 12px;margin-top:4px;font-family:Space Mono,monospace">
              <div style="font-size:0.6rem;color:#64748B;text-transform:uppercase;letter-spacing:0.08em">Date Range</div>
              <div style="font-size:0.72rem;color:#38BDF8;margin-top:3px">{date_range_str}</div>
              <div style="font-size:0.65rem;color:#475569;margin-top:2px">{len(dates)} reports · {len(kpis)} KPIs</div>
            </div>''',
            unsafe_allow_html=True,
        )
    active_date = active_date if "active_date" in dir() else dates[-1]

st.markdown("<div style='margin-bottom:1rem'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# KPI SUMMARY METRICS
# ─────────────────────────────────────────────
# Use active_date selected by user in control bar
_active = active_date if "active_date" in dir() else dates[-1]
latest_df = df[df["report_date"] == _active]
# Previous = the date just before active_date
_prev_dates = [d for d in dates if d < _active]
prev_df = df[df["report_date"] == _prev_dates[-1]] if _prev_dates else pd.DataFrame()

n_red    = int((latest_df["severity"] == 3).sum())
n_yellow = int((latest_df["severity"] == 2).sum())
n_green  = int((latest_df["severity"] == 1).sum())
n_total  = max(len(latest_df), 1)  # guard division by zero

prev_red   = int((prev_df["severity"] == 3).sum()) if not prev_df.empty else 0
red_delta  = n_red - prev_red
_arrow     = "▲" if red_delta > 0 else ("▼" if red_delta < 0 else "→")
_cls       = "up" if red_delta > 0 else ("down" if red_delta < 0 else "")
_delta_str = f"{_arrow} {abs(red_delta)} vs last week"
_health    = f"{n_green / n_total * 100:.0f}% health rate"

trend_data = compute_trend(df, kpi_col=kpi_col)
n_deteri   = int((trend_data["trend"] == "🔴 Deteriorating").sum()) if not trend_data.empty else 0

st.markdown(
    f'''<div class="metric-grid">
  <div class="metric-card red">
    <div class="metric-label">🔴 Critical</div>
    <div class="metric-value">{n_red}</div>
    <div class="metric-delta {_cls}">{_delta_str}</div>
  </div>
  <div class="metric-card yellow">
    <div class="metric-label">🟡 Warning</div>
    <div class="metric-value">{n_yellow}</div>
    <div class="metric-delta">of {n_total} KPIs monitored</div>
  </div>
  <div class="metric-card green">
    <div class="metric-label">🟢 Healthy</div>
    <div class="metric-value">{n_green}</div>
    <div class="metric-delta">{_health}</div>
  </div>
  <div class="metric-card blue">
    <div class="metric-label">📈 Deteriorating Trends</div>
    <div class="metric-value">{n_deteri}</div>
    <div class="metric-delta">KPIs worsening over time</div>
  </div>
</div>''',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tabs = st.tabs([
    "🔥  Heatmap",
    "📅  Weekly View",
    "⚡  WoW Changes",
    "🔍  Compare Weeks",
    "📈  Trends & Forecast",
    "🚨  Anomaly Detection",
    "⚠️  Early Warning",
    "🏆  Top KPI Attention",
    "🔗  KPI Correlation",
])
tab_heat, tab_weekly, tab_wow, tab_cmp, tab_trend, tab_anomaly, tab_ew, tab_top, tab_corr = tabs

# ════════════════════════════════════════════
# TAB 1 – HEATMAP
# ════════════════════════════════════════════
with tab_heat:
    st.markdown('<div class="section-title">KPI Status Heatmap Over Time</div>', unsafe_allow_html=True)

    col_heat, col_dist = st.columns([3, 1])
    with col_heat:
        pivot = df.pivot_table(index=kpi_col, columns="report_date",
                                values="severity", aggfunc="max")
        pivot_status = df.pivot_table(index=kpi_col, columns="report_date",
                                       values="status_name", aggfunc="first")

        fig = px.imshow(
            pivot,
            color_continuous_scale=status_color_scale(),
            aspect="auto",
            zmin=1, zmax=3,
        )
        fig.update_traces(
            customdata=pivot_status.values,
            hovertemplate="<b>%{y}</b><br>Date: %{x|%d %b %Y}<br>Status: <b>%{customdata}</b><extra></extra>",
        )
        apply_dark(fig, "Traffic Light Matrix – All KPIs", height=520)
        fig.update_layout(coloraxis_showscale=False, xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col_dist:
        st.markdown('<div class="section-title">Latest Distribution</div>', unsafe_allow_html=True)
        dist_data = latest_df["status_name"].value_counts().reset_index()
        dist_data.columns = ["Status", "Count"]
        colors = [COLOR_HEX.get(s, "#64748B") for s in dist_data["Status"]]
        fig_d = go.Figure(go.Pie(
            labels=dist_data["Status"], values=dist_data["Count"],
            marker=dict(colors=colors, line=dict(color="#080C14", width=2)),
            hole=0.62,
            textfont=dict(family="Space Mono", size=10, color="#E2E8F0"),
        ))
        apply_dark(fig_d, f"Week of {_active.strftime('%d %b')}", height=240)
        fig_d.update_layout(showlegend=True,
            legend=dict(orientation="v", font=dict(size=10), x=0.0),
            margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_d, use_container_width=True)

        # Trend bar
        st.markdown('<div class="section-title" style="margin-top:1rem">Trend Summary</div>', unsafe_allow_html=True)
        tc = trend_data["trend"].value_counts()
        fig_tc = go.Figure(go.Bar(
            x=tc.values, y=tc.index, orientation="h",
            marker_color=["#EF4444" if "Deterio" in i else "#22C55E" if "Impr" in i else "#38BDF8" for i in tc.index],
            text=tc.values, textposition="auto",
        ))
        apply_dark(fig_tc, "", height=180)
        fig_tc.update_layout(showlegend=False, yaxis=dict(tickfont=dict(size=9)))
        st.plotly_chart(fig_tc, use_container_width=True)

# ════════════════════════════════════════════
# TAB 2 – WEEKLY VIEW
# ════════════════════════════════════════════
with tab_weekly:
    st.markdown('<div class="section-title">KPI Status for Selected Week</div>', unsafe_allow_html=True)
    # Default to the active date chosen in the control bar
    _tab2_idx = _date_str_list.index(str(_active.date())) if str(_active.date()) in _date_str_list else len(dates)-1
    picked = dates[st.selectbox("Report date", range(len(dates)),
                                 format_func=lambda i: _date_str_list[i],
                                 index=_tab2_idx, key="tab2")]
    day_df = df[df["report_date"] == picked].sort_values("severity", ascending=False).reset_index(drop=True)

    # Color-coded table rows
    for _, row in day_df.iterrows():
        s = row["status_name"]
        pill = PILL_HTML.get(s, s)
        rc  = root_cause_summary(detail_df, row[kpi_col], picked)
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;padding:8px 12px;
                    background:#0F1923;border:1px solid #1E293B;border-radius:8px;margin-bottom:6px">
          <div style="flex:2;font-size:0.85rem;color:#E2E8F0">{row[kpi_col]}</div>
          <div style="flex:0 0 130px">{pill}</div>
          <div style="flex:3;font-size:0.72rem;color:#64748B;overflow:hidden;
                      text-overflow:ellipsis;white-space:nowrap" title="{rc}">{rc[:80]}{'…' if len(rc)>80 else ''}</div>
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════
# TAB 3 – WOW CHANGES
# ════════════════════════════════════════════
with tab_wow:
    st.markdown('<div class="section-title">Week-over-Week KPI Changes</div>', unsafe_allow_html=True)
    wow = wow_delta(df[df["report_date"].isin([_prev_dates[-1] if _prev_dates else dates[-2], _active])], kpi_col=kpi_col) if _prev_dates else wow_delta(df, kpi_col=kpi_col)
    if wow.empty:
        st.markdown('<div class="ok-banner">✅ Not enough data for WoW comparison.</div>', unsafe_allow_html=True)
    else:
        det = wow[wow["change"].isin(["📈 Deteriorated", "📉 Improved"])]
        nochange = wow[wow["change"] == "➡️ No Change"]
        if det.empty:
            st.markdown('<div class="ok-banner">✅ No KPI changes detected this week. System is stable.</div>', unsafe_allow_html=True)
        else:
            for _, r in det.iterrows():
                icon = "🔴" if r["change"] == "📈 Deteriorated" else "🟢"
                banner = "warn-banner" if r["change"] == "📈 Deteriorated" else "ok-banner"
                st.markdown(f'<div class="{banner}">{icon} <b>{r[kpi_col]}</b> — {r["prev"]} → <b>{r["curr"]}</b> ({r["change"]})</div>',
                            unsafe_allow_html=True)
        st.markdown(f'<div class="info-banner">ℹ️ {len(nochange)} KPIs unchanged this week.</div>', unsafe_allow_html=True)

        # Mini bar chart for severity distribution change
        c1, c2 = st.columns(2)
        for label, col_, date_ in [("Previous Week", c1, _prev_dates[-1] if _prev_dates else dates[-2]), ("This Week", c2, _active)]:
            with col_:
                d_ = df[df["report_date"] == date_]["status_name"].value_counts()
                fig_ = go.Figure(go.Bar(
                    x=d_.index, y=d_.values,
                    marker_color=[COLOR_HEX.get(s,"#64748B") for s in d_.index],
                    text=d_.values, textposition="auto",
                ))
                apply_dark(fig_, label, height=200)
                st.plotly_chart(fig_, use_container_width=True)

# ════════════════════════════════════════════
# TAB 4 – COMPARE TWO WEEKS
# ════════════════════════════════════════════
with tab_cmp:
    st.markdown('<div class="section-title">Side-by-Side Week Comparison</div>', unsafe_allow_html=True)
    if len(compare_dates) < 2:
        st.markdown('<div class="info-banner">ℹ️ Select exactly 2 dates in the sidebar to compare.</div>', unsafe_allow_html=True)
    else:
        d1 = pd.Timestamp(compare_dates[0])
        d2 = pd.Timestamp(compare_dates[1])
        # Normalize to match df report_date dtype
        d1 = df["report_date"].iloc[0].__class__(d1)
        d2 = df["report_date"].iloc[0].__class__(d2)
        df1 = df[df["report_date"] == d1].set_index(kpi_col)[["status_name"]]
        df2 = df[df["report_date"] == d2].set_index(kpi_col)[["status_name"]]
        merged = df1.join(df2, lsuffix="_d1", rsuffix="_d2", how="outer").reset_index()

        def cmp_classify(row):
            s1 = str(row.get("status_name_d1", ""))
            s2 = str(row.get("status_name_d2", ""))
            if STATUS_MAP.get(s2, 0) > STATUS_MAP.get(s1, 0): return "📈 Deteriorated"
            if STATUS_MAP.get(s2, 0) < STATUS_MAP.get(s1, 0): return "📉 Improved"
            return "➡️ No Change"

        merged["Change"] = merged.apply(cmp_classify, axis=1)
        merged["Root Cause"] = merged.apply(
            lambda r: root_cause_summary(detail_df, r[kpi_col], d2)
            if r["Change"] == "📈 Deteriorated" else "—", axis=1)

        for _, r in merged.iterrows():
            s1, s2 = str(r.get("status_name_d1","?")), str(r.get("status_name_d2","?"))
            p1 = PILL_HTML.get(s1, s1); p2 = PILL_HTML.get(s2, s2)
            bg = "#1F0A0A" if r["Change"] == "📈 Deteriorated" else "#0A1F0A" if r["Change"] == "📉 Improved" else "#0F1923"
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:12px;padding:8px 12px;
                        background:{bg};border:1px solid #1E293B;border-radius:8px;margin-bottom:5px">
              <div style="flex:2;font-size:0.82rem;color:#E2E8F0">{r[kpi_col]}</div>
              <div style="flex:0 0 120px">{p1}</div>
              <div style="color:#334155;font-size:1rem">→</div>
              <div style="flex:0 0 120px">{p2}</div>
              <div style="flex:1;font-size:0.75rem;color:#64748B">{r['Change']}</div>
            </div>
            """, unsafe_allow_html=True)

        csv = merged.to_csv(index=False).encode()
        st.download_button("⬇️ Export Comparison CSV", csv,
                           f"EWA_Compare_{d1.date()}_vs_{d2.date()}.csv", "text/csv")

# ════════════════════════════════════════════
# TAB 5 – TRENDS & FORECAST
# ════════════════════════════════════════════
with tab_trend:
    st.markdown('<div class="section-title">Severity Trend Analysis (Linear Regression per KPI)</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2])
    with c1:
        if trend_data.empty:
            st.info("Need ≥3 reports for trend.")
        else:
            disp = trend_data[["kpi","slope","r2","trend","n_weeks","current_severity"]].copy()
            disp.columns = ["KPI","Slope","R²","Trend","Weeks","Current"]
            disp["Current"] = disp["Current"].map({1:"🟢",2:"🟡",3:"🔴"})
            st.dataframe(disp.style.background_gradient(subset=["Slope"], cmap="RdYlGn_r"),
                         use_container_width=True, height=400)

    with c2:
        # Single KPI trend detail
        sel_kpi = st.selectbox("KPI trend detail", kpis, key="trend_kpi")
        kdf = df[df[kpi_col] == sel_kpi].sort_values("report_date")

        fig = go.Figure()
        # Actual
        fig.add_trace(go.Scatter(
            x=kdf["report_date"], y=kdf["severity"],
            mode="lines+markers",
            name="Actual",
            line=dict(color="#38BDF8", width=2),
            marker=dict(size=8, color=[COLOR_HEX.get(s,"#64748B") for s in kdf["status_name"]],
                        line=dict(width=1.5, color="#0F1923")),
        ))
        # Trend line
        if len(kdf) >= 3:
            t = np.arange(len(kdf)).reshape(-1, 1)
            reg = LinearRegression().fit(t, kdf["severity"].values)
            trend_y = reg.predict(t)
            fig.add_trace(go.Scatter(
                x=kdf["report_date"], y=trend_y,
                mode="lines", name="Trend",
                line=dict(color="#F59E0B", width=1.5, dash="dot"),
            ))

        # Forecast
        fc_df = forecast_severity(df, kpi_col=kpi_col, system=selected_sid, periods=fc_periods)
        fc_kpi = fc_df[fc_df["kpi"] == sel_kpi]
        if not fc_kpi.empty:
            fig.add_trace(go.Scatter(
                x=list(fc_kpi["report_date"]) + list(fc_kpi["report_date"])[::-1],
                y=list(fc_kpi["upper"]) + list(fc_kpi["lower"])[::-1],
                fill="toself", fillcolor="rgba(239,68,68,0.08)",
                line=dict(color="rgba(0,0,0,0)"), name="80% CI",
            ))
            fig.add_trace(go.Scatter(
                x=fc_kpi["report_date"], y=fc_kpi["severity_pred"],
                mode="lines+markers", name="Forecast",
                line=dict(color="#EF4444", width=2, dash="dash"),
                marker=dict(size=7),
            ))

        fig.update_layout(
            yaxis=dict(tickvals=[1,2,3], ticktext=["GREEN","YELLOW","RED"],
                       gridcolor="#1E293B", color="#64748B", range=[0.5, 3.5]),
            **{k:v for k,v in PLOTLY_LAYOUT.items() if k not in ("yaxis",)},
            height=380, title=dict(text=f"Trend & Forecast: {sel_kpi[:40]}", font=dict(size=12, color="#CBD5E1")),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Full forecast table
    st.markdown('<div class="section-title">Forecast: All KPIs Next Weeks</div>', unsafe_allow_html=True)
    fc_all = forecast_severity(df, kpi_col=kpi_col, system=selected_sid, periods=fc_periods)
    if not fc_all.empty:
        fc_all["Predicted Status"] = fc_all["severity_pred"].map(lambda x: REV_STATUS.get(round(x),"?"))
        st.dataframe(fc_all[["kpi","report_date","Predicted Status","severity_pred","lower","upper"]],
                     use_container_width=True, height=300)
    else:
        st.info("Not enough data for forecast.")

# ════════════════════════════════════════════
# TAB 6 – ANOMALY DETECTION
# ════════════════════════════════════════════
with tab_anomaly:
    st.markdown('<div class="section-title">Anomaly Detection — Statistical & Pattern-Based</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="info-banner">ℹ️ Rolling window: {ew_window} weeks · Z-threshold: {ew_z:.1f} · Detects sudden spikes, GREEN→RED jumps, statistical outliers.</div>',
                unsafe_allow_html=True)

    anom = detect_anomalies(df, kpi_col=kpi_col, system=selected_sid,
                             window=ew_window, z_thresh=ew_z)
    if anom.empty:
        st.markdown('<div class="ok-banner">✅ No anomalies detected with current thresholds.</div>', unsafe_allow_html=True)
    else:
        # Timeline chart
        fig_a = go.Figure()
        for kpi_nm in anom["kpi"].unique():
            sub = anom[anom["kpi"] == kpi_nm]
            kdf_ = df[df[kpi_col] == kpi_nm].sort_values("report_date")
            fig_a.add_trace(go.Scatter(
                x=kdf_["report_date"], y=kdf_["severity"],
                mode="lines", name=kpi_nm[:25],
                line=dict(width=1.5, color="#334155"), showlegend=False,
            ))
            fig_a.add_trace(go.Scatter(
                x=sub["report_date"], y=sub["severity"],
                mode="markers", name=kpi_nm[:25],
                marker=dict(size=12, color="#EF4444", symbol="x",
                            line=dict(width=2, color="#FCA5A5")),
                text=sub["anomaly_type"],
                hovertemplate="<b>%{text}</b><br>%{x|%d %b}<br>Z=%{customdata:.2f}<extra></extra>",
                customdata=sub["z_score"],
            ))
        apply_dark(fig_a, "Anomaly Timeline (✕ = detected anomaly)", height=380)
        fig_a.update_layout(
            yaxis=dict(tickvals=[1,2,3], ticktext=["GREEN","YELLOW","RED"],
                       gridcolor="#1E293B", color="#64748B"),
            showlegend=False,
        )
        st.plotly_chart(fig_a, use_container_width=True)

        # Detail table
        for _, r in anom.iterrows():
            st.markdown(f'<div class="warn-banner">🚨 <b>{r["kpi"]}</b> · {r["report_date"].strftime("%d %b %Y")} · {r["anomaly_type"]} · Status: {r["status"]} · Z={r["z_score"]}</div>',
                        unsafe_allow_html=True)

# ════════════════════════════════════════════
# TAB 7 – EARLY WARNING
# ════════════════════════════════════════════
with tab_ew:
    st.markdown('<div class="section-title">Early Warning System — Rolling Trend Signals</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="info-banner">ℹ️ Flags KPIs where rolling severity is increasing. Window = {ew_window} weeks.</div>',
                unsafe_allow_html=True)

    ew_df = df.sort_values([kpi_col, "report_date"]).copy()
    ew_df["rolling_mean"] = ew_df.groupby(kpi_col)["severity"].transform(
        lambda x: x.rolling(ew_window, min_periods=2).mean()
    )
    ew_df["rolling_trend"] = ew_df.groupby(kpi_col)["rolling_mean"].diff()

    # Warn on rising trend + severity ≥2
    warns = ew_df[(ew_df["rolling_trend"] > 0.3) & (ew_df["severity"] >= 2)].copy()
    warns = warns.sort_values("rolling_trend", ascending=False)

    if warns.empty:
        st.markdown('<div class="ok-banner">✅ No early warning signals. All rolling trends are stable or improving.</div>',
                    unsafe_allow_html=True)
    else:
        for _, r in warns.iterrows():
            severity_label = REV_STATUS.get(int(r["severity"]), "?")
            pill = PILL_HTML.get(severity_label, severity_label)
            st.markdown(f"""
            <div class="warn-banner">
              ⚠️ <b>{r[kpi_col]}</b> — {r['report_date'].strftime('%d %b %Y')} — {pill}
              — Rolling avg rising by <b>{r['rolling_trend']:.2f}</b> severity units
            </div>
            """, unsafe_allow_html=True)

    # Heatmap of rolling trend
    st.markdown('<div class="section-title">Rolling Severity Trend Heatmap</div>', unsafe_allow_html=True)
    trend_pivot = ew_df.pivot_table(index=kpi_col, columns="report_date",
                                     values="rolling_trend", aggfunc="mean")
    if not trend_pivot.empty:
        fig_ew = px.imshow(
            trend_pivot.fillna(0),
            color_continuous_scale=[[0, "#22C55E"], [0.5, "#334155"], [1, "#EF4444"]],
            aspect="auto",
        )
        apply_dark(fig_ew, "Rolling Trend (Red = worsening, Green = improving)", height=400)
        fig_ew.update_layout(coloraxis_showscale=True,
                              coloraxis_colorbar=dict(len=0.7, thickness=10, title="Δ Trend"))
        st.plotly_chart(fig_ew, use_container_width=True)

# ════════════════════════════════════════════
# TAB 8 – ATTENTION SCORE
# ════════════════════════════════════════════
with tab_top:
    st.markdown('<div class="section-title">Top KPIs Needing Attention — Composite ML Score</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-banner">ℹ️ Attention Score = weighted composite of: current severity (30%) · trend direction (25%) · % weeks RED (25%) · volatility (10%) · consecutive weeks at risk (10%)</div>""",
                unsafe_allow_html=True)

    att = attention_score(df, kpi_col=kpi_col, system=selected_sid)
    if att.empty:
        st.info("Need more data.")
    else:
        # Top N bar chart
        top_n = att.head(12)
        colors = [COLOR_HEX.get(c, "#64748B") for c in top_n["Current"]]
        fig_att = go.Figure(go.Bar(
            x=top_n["Attention Score"],
            y=[k[:30] for k in top_n["KPI"]],
            orientation="h",
            marker=dict(
                color=top_n["Attention Score"],
                colorscale=[[0, "#22C55E"], [0.5, "#EAB308"], [1, "#EF4444"]],
                cmin=0, cmax=1,
                showscale=True,
                colorbar=dict(len=0.7, thickness=10, title="Score"),
                line=dict(width=0),
            ),
            text=[f"{s:.2f}" for s in top_n["Attention Score"]],
            textposition="auto",
        ))
        apply_dark(fig_att, "Top 12 KPIs by Attention Score", height=420)
        fig_att.update_layout(xaxis_title="Attention Score (0=OK, 1=Critical)")
        st.plotly_chart(fig_att, use_container_width=True)

        st.markdown('<div class="section-title">Full Attention Ranking</div>', unsafe_allow_html=True)
        st.dataframe(att.drop(columns=["system"], errors="ignore"), use_container_width=True, height=350)

# ════════════════════════════════════════════
# TAB 9 – KPI CORRELATION
# ════════════════════════════════════════════
with tab_corr:
    st.markdown('<div class="section-title">KPI Correlation Matrix — Co-Movement Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-banner">ℹ️ High correlation (🔴) means KPIs often degrade together — suggests a shared root cause (e.g., hardware bottleneck affecting performance + DB).</div>',
                unsafe_allow_html=True)

    corr_df = kpi_correlation(df, kpi_col=kpi_col)
    if corr_df.empty or len(corr_df) < 3:
        st.info("Need more KPI diversity for correlation analysis.")
    else:
        fig_corr = px.imshow(
            corr_df,
            color_continuous_scale=[[0,"#1E3A5F"],[0.5,"#334155"],[1,"#EF4444"]],
            zmin=-1, zmax=1, aspect="auto",
            text_auto=".2f",
        )
        apply_dark(fig_corr, "Pearson Correlation: KPI Severity Co-Movement", height=550)
        fig_corr.update_layout(
            coloraxis_colorbar=dict(len=0.7, thickness=12, title="r"),
        )
        fig_corr.update_traces(textfont=dict(size=8))
        st.plotly_chart(fig_corr, use_container_width=True)

        # Highlight high correlations
        st.markdown('<div class="section-title">Strongly Correlated KPI Pairs (|r| ≥ 0.7)</div>', unsafe_allow_html=True)
        pairs = []
        for i in range(len(corr_df.columns)):
            for j in range(i+1, len(corr_df.columns)):
                r = corr_df.iloc[i, j]
                if abs(r) >= 0.70:
                    pairs.append({"KPI A": corr_df.columns[i], "KPI B": corr_df.columns[j], "r": round(r,3)})
        if pairs:
            pairs_df = pd.DataFrame(pairs).sort_values("r", key=abs, ascending=False)
            for _, p in pairs_df.iterrows():
                banner = "warn-banner" if p["r"] > 0 else "info-banner"
                label  = "co-degrade" if p["r"] > 0 else "inverse"
                st.markdown(f'<div class="{banner}">📊 <b>{p["KPI A"]}</b> ↔ <b>{p["KPI B"]}</b> · r = {p["r"]} ({label})</div>',
                            unsafe_allow_html=True)
        else:
            st.markdown('<div class="ok-banner">✅ No strongly correlated KPI pairs found (thres |r| ≥ 0.70).</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem;padding:1rem 1.5rem;border-top:1px solid #1E293B;
            display:flex;justify-content:space-between;align-items:center">
  <div style="font-family:Space Mono,monospace;font-size:0.65rem;color:#334155">
    🛰️ SAP EWA Intelligence Hub · ML: Linear Regression · Anomaly Detection · Attention Scoring · Early Warning
  </div>
  <div style="font-family:Space Mono,monospace;font-size:0.65rem;color:#334155">
    Powered by Streamlit · Plotly · scikit-learn
  </div>
</div>
""", unsafe_allow_html=True)