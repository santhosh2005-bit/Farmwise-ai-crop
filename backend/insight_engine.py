"""
insight_engine.py — Post-processing layer that turns raw tool results
into user-friendly insights and chart instructions.

This module sits between the LLM response and the API response: it
computes automatic statistical insights from the data that complement
the LLM's narrative answer.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def generate_insight(data: list[dict[str, Any]], y_column: str | None = None) -> dict[str, Any]:
    """Compute automatic statistical insights from tool result data.

    Parameters
    ----------
    data : list[dict]
        Row data in records orientation (output from a tool function).
    y_column : str | None
        The primary numeric column to analyse.  If *None*, attempts to
        auto-detect the most relevant numeric column.

    Returns
    -------
    dict[str, Any]
        Keys:
        - ``row_count`` (int): number of rows
        - ``highest`` (dict): row with the max value
        - ``lowest`` (dict): row with the min value
        - ``average`` (float): mean of the y column
        - ``median`` (float): median of the y column
        - ``trend`` (str): "increasing", "decreasing", or "stable"
        - ``growth_rate`` (float | None): % change from first to last value
        - ``outlier_count`` (int): rows outside 1.5 * IQR
    """
    if not data:
        return {"row_count": 0, "message": "No data to analyse."}

    df = pd.DataFrame(data)

    # Auto-detect the y column if not specified
    if y_column is None:
        y_column = _detect_numeric_column(df)

    if y_column is None or y_column not in df.columns:
        return {
            "row_count": len(df),
            "message": "No suitable numeric column found for insight generation.",
        }

    series = pd.to_numeric(df[y_column], errors="coerce").dropna()

    if series.empty:
        return {"row_count": len(df), "message": f"Column '{y_column}' has no numeric values."}

    # ── Core statistics ──────────────────────────────────────
    avg = float(series.mean())
    med = float(series.median())
    std = float(series.std()) if len(series) > 1 else 0.0

    # ── Highest / Lowest ─────────────────────────────────────
    max_idx = series.idxmax()
    min_idx = series.idxmin()
    highest = _safe_record(df, max_idx)
    lowest = _safe_record(df, min_idx)

    # ── Trend detection ──────────────────────────────────────
    trend = _detect_trend(df, y_column)

    # ── Growth rate (first → last) ───────────────────────────
    growth_rate = _compute_growth_rate(series)

    # ── Outlier detection (IQR method) ───────────────────────
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_count = int(((series < lower_bound) | (series > upper_bound)).sum())

    return {
        "row_count": len(df),
        "column_analysed": y_column,
        "highest": highest,
        "lowest": lowest,
        "average": round(avg, 2),
        "median": round(med, 2),
        "std_dev": round(std, 2),
        "trend": trend,
        "growth_rate_pct": growth_rate,
        "outlier_count": outlier_count,
    }


def format_data_for_display(
    data: list[dict[str, Any]], max_rows: int = 20
) -> list[dict[str, Any]]:
    """Truncate and sanitise data for frontend display.

    Parameters
    ----------
    data : list[dict]
        Raw records from a tool result.
    max_rows : int
        Maximum rows to send to the frontend (default 20).

    Returns
    -------
    list[dict]
        Cleaned, truncated data with floats rounded to 2 decimal places
        and numpy types converted to native Python.
    """
    truncated = data[:max_rows]

    cleaned: list[dict[str, Any]] = []
    for row in truncated:
        clean_row: dict[str, Any] = {}
        for key, value in row.items():
            clean_row[key] = _sanitise_value(value)
        cleaned.append(clean_row)

    return cleaned


# ─── Internal helpers ────────────────────────────────────────


def _detect_numeric_column(df: pd.DataFrame) -> str | None:
    """Auto-detect the most relevant numeric column for insights.

    Prefers yield > rainfall > pesticides > temp, then falls back to
    any numeric column.
    """
    priority = [
        "hg/ha_yield",
        "average_rain_fall_mm_per_year",
        "pesticides_tonnes",
        "avg_temp",
    ]
    for col in priority:
        if col in df.columns:
            return col

    # Fall back to first numeric column
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return numeric_cols[0] if numeric_cols else None


def _detect_trend(df: pd.DataFrame, y_column: str) -> str:
    """Detect whether the data trends up, down, or is stable.

    Uses the sign of a simple linear slope over the Year column
    if present, otherwise over row order.
    """
    if "Year" in df.columns and len(df) >= 3:
        sorted_df = df.sort_values("Year")
        y_values = pd.to_numeric(sorted_df[y_column], errors="coerce").dropna()
    else:
        y_values = pd.to_numeric(df[y_column], errors="coerce").dropna()

    if len(y_values) < 2:
        return "insufficient data"

    # Simple slope: compare first third average vs last third average
    third = max(1, len(y_values) // 3)
    first_avg = float(y_values.iloc[:third].mean())
    last_avg = float(y_values.iloc[-third:].mean())

    if first_avg == 0:
        return "stable"

    change_pct = ((last_avg - first_avg) / abs(first_avg)) * 100

    if change_pct > 5:
        return "increasing"
    elif change_pct < -5:
        return "decreasing"
    else:
        return "stable"


def _compute_growth_rate(series: pd.Series) -> float | None:
    """Compute % change from first to last non-null value."""
    values = series.dropna()
    if len(values) < 2:
        return None

    first_val = float(values.iloc[0])
    last_val = float(values.iloc[-1])

    if first_val == 0:
        return None

    return round(((last_val - first_val) / abs(first_val)) * 100, 2)


def _safe_record(df: pd.DataFrame, idx: int) -> dict[str, Any]:
    """Safely extract a row as a dict, converting numpy types."""
    try:
        row = df.loc[idx]
        return {k: _sanitise_value(v) for k, v in row.to_dict().items()}
    except (KeyError, IndexError):
        return {}


def _sanitise_value(value: Any) -> Any:
    """Convert numpy/pandas types to native Python for JSON serialisation."""
    if isinstance(value, (np.integer,)):
        return int(value)
    elif isinstance(value, (np.floating,)):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(float(value), 2)
    elif isinstance(value, np.bool_):
        return bool(value)
    elif isinstance(value, pd.Timestamp):
        return value.isoformat()
    elif isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, 2)
    return value
