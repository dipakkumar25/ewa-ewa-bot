# src/ewa_kpi_ml.py
"""
EWA KPI ML Analytics

• Trend: severity trend per (system, KPI) – improving / deteriorating / stable
• Time series forecast: next N weeks severity prediction
• Attention score: rank KPIs that need special attention (red history, bad trend, volatility)
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Optional: use statsmodels for better time series forecast
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

from sklearn.linear_model import LinearRegression

# Severity: 1=GREEN, 2=YELLOW, 3=RED
SEVERITY_RANK = {"GREEN": 1, "YELLOW": 2, "RED": 3}
TREND_THRESHOLD = 0.03  # slope per week above this = deteriorating


def _ensure_severity(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure numeric severity column exists."""
    if "severity" not in df.columns and "status_name" in df.columns:
        df = df.copy()
        df["severity"] = df["status_name"].astype(str).str.upper().str.strip().map(SEVERITY_RANK)
    return df


def _kpi_column(df: pd.DataFrame) -> str:
    """Return the KPI column name (clean_section or primary_kpi)."""
    if "clean_section" in df.columns:
        return "clean_section"
    if "primary_kpi" in df.columns:
        return "primary_kpi"
    raise ValueError("DataFrame must have 'clean_section' or 'primary_kpi'")


def compute_trend(
    df: pd.DataFrame,
    system: Optional[str] = None,
    kpi_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute severity trend per (system, KPI): slope and direction.

    Returns DataFrame with columns: system, kpi, slope, trend_direction, n_weeks.
    trend_direction: 'improving' | 'deteriorating' | 'stable'
    """
    df = _ensure_severity(df)
    kpi_col = kpi_col or _kpi_column(df)
    if system is not None and "system" in df.columns:
        df = df[df["system"] == system].copy()
    else:
        df = df.copy()

    out = []
    group_cols = [c for c in ["system", kpi_col] if c in df.columns]
    for key, grp in df.groupby(group_cols, dropna=False):
        grp = grp.sort_values("report_date")
        t = np.arange(len(grp)).reshape(-1, 1)
        y = grp["severity"].values.astype(float)
        if len(y) < 2:
            continue
        reg = LinearRegression().fit(t, y)
        slope = float(reg.coef_[0])
        if slope > TREND_THRESHOLD:
            direction = "deteriorating"
        elif slope < -TREND_THRESHOLD:
            direction = "improving"
        else:
            direction = "stable"
        if isinstance(key, tuple):
            sys_, kpi = (key[0], key[1]) if len(key) == 2 else (None, key[0])
        else:
            sys_, kpi = None, key
        row = {"slope": slope, "trend_direction": direction, "n_weeks": len(y), "kpi": kpi}
        if sys_ is not None:
            row["system"] = sys_
        out.append(row)

    return pd.DataFrame(out)


def forecast_severity(
    df: pd.DataFrame,
    system: Optional[str] = None,
    kpi: Optional[str] = None,
    kpi_col: Optional[str] = None,
    periods_ahead: int = 4,
) -> pd.DataFrame:
    """
    Forecast severity for the next `periods_ahead` report dates per (system, KPI).

    Uses ExponentialSmoothing if statsmodels is available, else linear extrapolation.
    Returns DataFrame: report_date (future), severity_pred, kpi, system (if present).
    """
    df = _ensure_severity(df)
    kpi_col = kpi_col or _kpi_column(df)
    if system is not None and "system" in df.columns:
        df = df[df["system"] == system]
    if kpi is not None:
        df = df[df[kpi_col] == kpi]

    group_cols = [c for c in ["system", kpi_col] if c in df.columns]
    all_forecasts = []

    for key, grp in df.groupby(group_cols, dropna=False):
        grp = grp.sort_values("report_date").drop_duplicates("report_date")
        dates = pd.to_datetime(grp["report_date"])
        y = grp["severity"].values.astype(float)
        if len(y) < 2:
            continue

        sys_ = key[0] if isinstance(key, tuple) and len(key) == 2 else None
        kpi_name = key[1] if isinstance(key, tuple) and len(key) == 2 else key

        # Next dates (weekly assumption)
        last_date = dates.iloc[-1]
        freq = pd.infer_freq(dates) or "W-MON"
        next_dates = pd.date_range(start=last_date, periods=periods_ahead + 1, freq=freq)[1:]

        if HAS_STATSMODELS and len(y) >= 3:
            try:
                model = ExponentialSmoothing(y, trend="add")
                fit = model.fit(optimized=True)
                pred = fit.forecast(periods_ahead)
                pred = np.clip(pred, 1, 3)
            except Exception:
                pred = _linear_forecast(y, periods_ahead)
        else:
            pred = _linear_forecast(y, periods_ahead)

        for i, d in enumerate(next_dates):
            row = {"report_date": d, "severity_pred": float(np.clip(pred[i], 1, 3)), "kpi": kpi_name}
            if sys_ is not None:
                row["system"] = sys_
            all_forecasts.append(row)

    return pd.DataFrame(all_forecasts)


def _linear_forecast(y: np.ndarray, periods: int) -> np.ndarray:
    """Simple linear extrapolation for severity."""
    t = np.arange(len(y)).reshape(-1, 1)
    reg = LinearRegression().fit(t, y)
    t_future = np.arange(len(y), len(y) + periods).reshape(-1, 1)
    pred = reg.predict(t_future)
    return np.clip(pred, 1, 3)


def attention_score(
    df: pd.DataFrame,
    system: Optional[str] = None,
    kpi_col: Optional[str] = None,
    weight_current: float = 0.35,
    weight_trend: float = 0.25,
    weight_red_pct: float = 0.25,
    weight_volatility: float = 0.15,
) -> pd.DataFrame:
    """
    Rank KPIs by attention score (higher = needs more attention).

    Factors: current severity, deteriorating trend, % of weeks in RED, volatility.
    Returns DataFrame: system, kpi, attention_score, current_severity, trend_direction, red_pct, volatility.
    """
    df = _ensure_severity(df)
    kpi_col = kpi_col or _kpi_column(df)
    if system is not None and "system" in df.columns:
        df = df[df["system"] == system].copy()
    else:
        df = df.copy()

    trends = compute_trend(df, system=system, kpi_col=kpi_col)

    group_cols = [c for c in ["system", kpi_col] if c in df.columns]
    rows = []
    for key, grp in df.groupby(group_cols, dropna=False):
        grp = grp.sort_values("report_date")
        sys_ = key[0] if isinstance(key, tuple) and len(key) == 2 else None
        kpi_name = key[1] if isinstance(key, tuple) and len(key) == 2 else key

        current = float(grp["severity"].iloc[-1])
        red_pct = (grp["severity"] == 3).mean()
        volatility = float(grp["severity"].diff().abs().mean()) if len(grp) > 1 else 0.0

        if "system" in trends.columns:
            tr = trends[(trends["kpi"] == kpi_name) & (trends["system"] == sys_)]
        else:
            tr = trends[trends["kpi"] == kpi_name]
        trend_dir = tr["trend_direction"].iloc[0] if not tr.empty else "stable"
        slope = tr["slope"].iloc[0] if not tr.empty else 0.0

        # Normalized components (0–1 scale where 1 = bad)
        comp_current = (current - 1) / 2.0
        comp_trend = 1.0 if trend_dir == "deteriorating" else (0.5 if trend_dir == "stable" else 0.0)
        comp_red = float(red_pct)
        comp_vol = min(volatility / 2.0, 1.0)

        score = (
            weight_current * comp_current
            + weight_trend * comp_trend
            + weight_red_pct * comp_red
            + weight_volatility * comp_vol
        )

        row = {
            "kpi": kpi_name,
            "attention_score": round(score, 4),
            "current_severity": current,
            "trend_direction": trend_dir,
            "red_pct": round(red_pct, 2),
            "volatility": round(volatility, 3),
        }
        if sys_ is not None:
            row["system"] = sys_
        rows.append(row)

    result = pd.DataFrame(rows).sort_values("attention_score", ascending=False).reset_index(drop=True)
    return result
