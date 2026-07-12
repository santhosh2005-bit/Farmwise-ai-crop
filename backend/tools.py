"""
tools.py — Tool definitions for Groq function-calling.

Each public function here is a *tool* the LLM can invoke.  The module
also exposes the ``TOOL_SCHEMAS`` list (JSON-Schema descriptions that
get sent to the Groq API) and a dispatcher that routes tool-call names
to their implementations.

Design notes
------------
* Keep every tool **small and focused** — one DataFrame operation each.
* Return plain Python dicts/lists so they serialise cleanly to JSON.
* The LLM decides *which* tool to call; nothing is hard-coded.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from backend.dataset import get_dataframe


# ─── Tool function implementations ──────────────────────────


def filter_by_country(country: str | None = None, year: int | None = None, name: str | None = None) -> list[dict[str, Any]]:
    """Filter the dataset for a specific country, optionally for a specific year.

    Parameters
    ----------
    country : str | None
        Country name (case-insensitive match against the *Area* column).
    year : int | None
        Optional year filter.
    name : str | None
        Alternative parameter for country name.

    Returns
    -------
    list[dict]
        Filtered rows as records.
    """
    if not country and name:
        country = name
    if not country:
        return [{"error": "Country parameter is required."}]

    df: pd.DataFrame = get_dataframe()
    mask = df["Area"].str.lower() == country.strip().lower()
    if year is not None:
        mask &= df["Year"] == year
    result = df[mask].sort_values("Year")
    return result.to_dict(orient="records")


def filter_by_item(item: str) -> list[dict[str, Any]]:
    """Filter the dataset for a specific crop item.

    Parameters
    ----------
    item : str
        Crop name (case-insensitive match against the *Item* column).

    Returns
    -------
    list[dict]
        Filtered rows as records.
    """
    df: pd.DataFrame = get_dataframe()
    mask = df["Item"].str.lower() == item.strip().lower()
    result = df[mask].sort_values(["Area", "Year"])
    return result.to_dict(orient="records")


def get_yield_trend(country: str | None = None, item: str | None = None, name: str | None = None) -> list[dict[str, Any]]:
    """Return the year-over-year yield trend for a country (and optionally a crop).

    Parameters
    ----------
    country : str | None
        Country name.
    item : str | None
        Optional crop name to narrow the trend.
    name : str | None
        Alternative parameter for country name.

    Returns
    -------
    list[dict]
        Yearly yield data as records, sorted by year.
    """
    if not country and name:
        country = name
    if not country:
        return [{"error": "Country parameter is required."}]

    df: pd.DataFrame = get_dataframe()
    mask = df["Area"].str.lower() == country.strip().lower()

    if item and item.strip().lower() in ("null", "none", ""):
        item = None

    if item:
        mask &= df["Item"].str.lower() == item.strip().lower()

    result = df[mask].sort_values("Year")
    return result.to_dict(orient="records")


def compare_countries(countries: list[str], metric: str = "hg/ha_yield", item: str | None = None) -> list[dict[str, Any]]:
    """Compare data across multiple countries for a given metric.

    Parameters
    ----------
    countries : list[str]
        List of country names to compare.
    metric : str
        The metric to compare (e.g., 'hg/ha_yield', 'average_rain_fall_mm_per_year', 'avg_temp').
    item : str | None
        Optional crop to compare. Required if metric is 'hg/ha_yield'.

    Returns
    -------
    list[dict]
        Combined data for the requested countries, sorted by year.
    """
    df: pd.DataFrame = get_dataframe()
    countries_lower = [c.strip().lower() for c in countries]
    
    mask = df["Area"].str.lower().isin(countries_lower)
    
    if item and item.strip().lower() in ("null", "none", ""):
        item = None
    
    if item:
        mask &= df["Item"].str.lower() == item.strip().lower()
        
    result = df[mask].sort_values(["Area", "Year"])
    
    if metric != "hg/ha_yield":
        result = result.drop_duplicates(subset=["Area", "Year"])
        
    # Ensure the metric exists and isn't NaN
    if metric not in result.columns:
        return [{"error": f"Invalid metric. Available: {list(result.columns)}"}]
        
    result = result.dropna(subset=[metric])
    
    # Check if any requested countries are missing from the result
    found_countries = result["Area"].str.lower().unique()
    missing_countries = [c for c in countries if c.strip().lower() not in found_countries]
    
    response = {"data": result.to_dict(orient="records")}
    if missing_countries:
        response["warning"] = f"No data found for the following countries: {', '.join(missing_countries)}. They might not be in the dataset."
        
    return [response]


def get_top_countries(
    metric: str = "hg/ha_yield", item: str | None = None, year: int | None = None, top_n: int = 10
) -> list[dict[str, Any]]:
    """Return the top *N* countries for a given metric (e.g., rainfall, yield).

    Parameters
    ----------
    metric : str
        The column to rank by (e.g., 'hg/ha_yield', 'average_rain_fall_mm_per_year', 'avg_temp').
    item : str | None
        Optional crop name. Only needed if the metric is 'hg/ha_yield'.
    year : int | None
        Optional year filter; uses the latest available year when *None*.
    top_n : int
        Number of top countries to return (default 10).

    Returns
    -------
    list[dict]
        Top countries as records, sorted descending by the metric.
    """
    df: pd.DataFrame = get_dataframe()
    subset = df

    if item and item.strip().lower() in ("null", "none", ""):
        item = None

    if item:
        mask = df["Item"].str.lower() == item.strip().lower()
        subset = df[mask]

    if subset.empty:
        return []

    # Use specified year or fall back to latest
    target_year = year if year is not None else int(subset["Year"].max())
    subset = subset[subset["Year"] == target_year]

    # Ensure the metric exists and isn't NaN
    if metric not in subset.columns:
        return [{"error": f"Invalid metric. Available: {list(subset.columns)}"}]
    
    subset = subset.dropna(subset=[metric])
    
    # Sort descending by metric first so drop_duplicates keeps the highest value for each country
    subset = subset.sort_values(by=metric, ascending=False)
    
    # Drop duplicates by Area, Item, Year to clean up any duplicate rows (e.g. duplicate temp entries)
    subset = subset.drop_duplicates(subset=["Area", "Item", "Year"])
    
    # For ranking countries, make sure each country is represented at most once in the output
    subset = subset.drop_duplicates(subset=["Area", "Year"])

    result = subset.head(top_n)
    return result.to_dict(orient="records")


def get_correlation(col_x: str, col_y: str) -> dict[str, Any]:
    """Compute the Pearson correlation between two numeric columns.

    Parameters
    ----------
    col_x : str
        First column name.
    col_y : str
        Second column name.

    Returns
    -------
    dict
        Keys: ``col_x``, ``col_y``, ``correlation``, ``data`` (sample
        scatter-plot points).
    """
    df: pd.DataFrame = get_dataframe()

    # Validate columns exist and are numeric
    for col in (col_x, col_y):
        if col not in df.columns:
            return {"error": f"Column '{col}' not found. Available: {list(df.columns)}"}

    numeric_df = df[[col_x, col_y]].dropna()
    correlation: float = float(numeric_df[col_x].corr(numeric_df[col_y]))

    # Return a sample of points for scatter plot (limit to 500 for performance)
    sample = numeric_df.sample(n=min(500, len(numeric_df)), random_state=42)

    return {
        "col_x": col_x,
        "col_y": col_y,
        "correlation": round(correlation, 4),
        "data_points": len(numeric_df),
        "data": sample.to_dict(orient="records"),
    }


def get_summary_statistics(column: str) -> dict[str, Any]:
    """Return summary statistics (mean, median, std, min, max) for a column.

    Parameters
    ----------
    column : str
        Numeric column name.

    Returns
    -------
    dict
        Descriptive statistics including count, mean, median, std, min,
        max, q25, q75.
    """
    df: pd.DataFrame = get_dataframe()

    if column not in df.columns:
        return {"error": f"Column '{column}' not found. Available: {list(df.columns)}"}

    series = df[column].dropna()

    return {
        "summary": {
            "column": column,
            "count": int(series.count()),
            "mean": round(float(series.mean()), 2),
            "median": round(float(series.median()), 2),
            "std": round(float(series.std()), 2),
            "min": round(float(series.min()), 2),
            "max": round(float(series.max()), 2),
            "q25": round(float(series.quantile(0.25)), 2),
            "q75": round(float(series.quantile(0.75)), 2),
        },
        # Include Year and Area in the sample so the chart generator can do time-series if needed
        "data": df[["Year", "Area", column]].dropna(subset=[column]).sample(n=min(500, len(series))).to_dict(orient="records")
    }


def get_pesticide_usage(country: str | None = None) -> list[dict[str, Any]]:
    """Return pesticide-usage data, optionally filtered by country.

    Parameters
    ----------
    country : str | None
        Optional country filter.

    Returns
    -------
    list[dict]
        Pesticide usage rows as records, sorted by year.
    """
    df: pd.DataFrame = get_dataframe()

    if country:
        mask = df["Area"].str.lower() == country.strip().lower()
        result = df[mask][["Area", "Year", "pesticides_tonnes"]].drop_duplicates()
    else:
        # Aggregate by country+year (avoid duplicate rows from multiple crops)
        result = (
            df.groupby(["Area", "Year"])["pesticides_tonnes"]
            .first()
            .reset_index()
        )

    return result.sort_values(["Area", "Year"]).to_dict(orient="records")


def forecast_yield_trend(
    country: str, item: str, forecast_years: int = 5
) -> list[dict[str, Any]]:
    """Fit a linear regression trend line to historical yield data and project future yields.

    Parameters
    ----------
    country : str
        Country name.
    item : str
        Crop name.
    forecast_years : int
        Number of years into the future to forecast (default 5).

    Returns
    -------
    list[dict]
        Historical and forecasted yield records.
    """
    df: pd.DataFrame = get_dataframe()
    mask = (df["Area"].str.lower() == country.strip().lower()) & (
        df["Item"].str.lower() == item.strip().lower()
    )
    subset = df[mask].sort_values("Year")

    # Group by Year to avoid duplicates affecting regressions/plots
    grouped = subset.groupby("Year")[["hg/ha_yield"]].mean().reset_index()

    if len(grouped) < 2:
        return [
            {
                "error": "Insufficient data to build a trend line (need at least 2 historical years)."
            }
        ]

    x = grouped["Year"].values
    y = grouped["hg/ha_yield"].values

    slope, intercept = np.polyfit(x, y, 1)

    y_pred = slope * x + intercept
    y_mean = np.mean(y)
    ss_tot = np.sum((y - y_mean) ** 2)
    ss_res = np.sum((y - y_pred) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 1.0

    results: list[dict[str, Any]] = []
    # Historical records
    for _, row in grouped.iterrows():
        results.append(
            {
                "Area": country,
                "Item": item,
                "Year": int(row["Year"]),
                "hg/ha_yield": float(row["hg/ha_yield"]),
                "type": "Historical",
                "trend_line": float(slope * row["Year"] + intercept),
            }
        )

    # Forecast records
    last_year = int(x[-1])
    for i in range(1, forecast_years + 1):
        f_year = last_year + i
        f_yield = max(0.0, float(slope * f_year + intercept))
        results.append(
            {
                "Area": country,
                "Item": item,
                "Year": f_year,
                "hg/ha_yield": f_yield,
                "type": "Forecast",
                "trend_line": f_yield,
                "r_squared": round(float(r_squared), 4),
                "slope": round(float(slope), 2),
            }
        )

    return results


def get_yield_regression_factors(country: str, item: str) -> dict[str, Any]:
    """Analyze the impact of climate factors and pesticide usage on crop yield.

    Parameters
    ----------
    country : str
        Country name.
    item : str
        Crop name.

    Returns
    -------
    dict
        Correlation statistics and regression parameters.
    """
    df: pd.DataFrame = get_dataframe()
    mask = (df["Area"].str.lower() == country.strip().lower()) & (
        df["Item"].str.lower() == item.strip().lower()
    )
    subset = df[mask].dropna(subset=["hg/ha_yield"])

    # Group by Year taking the mean of all relevant numeric columns
    cols_to_group = [
        "hg/ha_yield",
        "average_rain_fall_mm_per_year",
        "pesticides_tonnes",
        "avg_temp",
    ]
    grouped = subset.groupby("Year")[cols_to_group].mean().reset_index()

    if len(grouped) < 5:
        return {
            "error": "Insufficient data points (need at least 5) to perform factor regression."
        }

    factors = {
        "Rainfall": "average_rain_fall_mm_per_year",
        "Pesticides": "pesticides_tonnes",
        "Temperature": "avg_temp",
    }

    results: dict[str, Any] = {}
    correlations: dict[str, float] = {}

    for factor_name, col in factors.items():
        if col in grouped.columns:
            valid_data = grouped[[col, "hg/ha_yield"]].dropna()
            if len(valid_data) >= 3:
                x_val = valid_data[col].values
                y_val = valid_data["hg/ha_yield"].values
                corr = float(valid_data[col].corr(valid_data["hg/ha_yield"]))

                try:
                    slope, intercept = np.polyfit(x_val, y_val, 1)
                except Exception:
                    slope, intercept = 0.0, 0.0

                correlations[factor_name] = corr
                results[factor_name] = {
                    "correlation": round(corr, 4),
                    "slope": round(float(slope), 4),
                    "intercept": round(float(intercept), 2),
                    "data_points": len(valid_data),
                }

    strongest_factor = None
    max_abs_corr = -1.0
    for factor_name, corr in correlations.items():
        if not np.isnan(corr) and abs(corr) > max_abs_corr:
            max_abs_corr = abs(corr)
            strongest_factor = factor_name

    desc = "No clear climatic driver found."
    if strongest_factor:
        corr_val = correlations[strongest_factor]
        desc = (
            f"Yield vs climatic factors analysis. Strongest driver is {strongest_factor} "
            f"with correlation={corr_val:.4f}."
        )

    return {
        "country": subset.iloc[0]["Area"],
        "item": subset.iloc[0]["Item"],
        "factors": results,
        "strongest_driver": strongest_factor,
        "description": desc,
    }


# ─── Tool registry ──────────────────────────────────────────
# Maps the function name (used by the LLM) → callable.
TOOL_REGISTRY: dict[str, callable] = {
    "filter_by_country": filter_by_country,
    "filter_by_item": filter_by_item,
    "get_yield_trend": get_yield_trend,
    "compare_countries": compare_countries,
    "get_top_countries": get_top_countries,
    "get_correlation": get_correlation,
    "get_summary_statistics": get_summary_statistics,
    "get_pesticide_usage": get_pesticide_usage,
    "forecast_yield_trend": forecast_yield_trend,
    "get_yield_regression_factors": get_yield_regression_factors,
}


# ─── JSON-Schema definitions sent to the Groq API ───────────
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "filter_by_country",
            "description": "Get all data (crops, rainfall, temperature, pesticides) for a country. Use this to analyze country-level metrics over time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "country": {"type": "string", "description": "Country name e.g. 'India'"},
                    "name": {"type": "string", "description": "Country name (alternative schema name)"},
                    "year": {"type": ["integer", "null"], "description": "Optional year e.g. 2013"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_by_item",
            "description": "Get data for a specific crop across all countries and years.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {"type": "string", "description": "Crop name e.g. 'Wheat', 'Rice, paddy', 'Maize'"},
                },
                "required": ["item"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_yield_trend",
            "description": "Year-over-year yield trend for a country, optionally for one crop.",
            "parameters": {
                "type": "object",
                "properties": {
                    "country": {"type": "string", "description": "Country name"},
                    "name": {"type": "string", "description": "Country name (alternative schema name)"},
                    "item": {"type": ["string", "null"], "description": "Optional crop name"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_countries",
            "description": "Compare data across multiple countries for a given metric (e.g., to calculate the difference or comparison in crop yield, rainfall, temperature, or pesticide usage between countries).",
            "parameters": {
                "type": "object",
                "properties": {
                    "countries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of countries to compare",
                    },
                    "metric": {"type": "string", "description": "Metric to compare (e.g. 'average_rain_fall_mm_per_year', 'avg_temp', 'hg/ha_yield')"},
                    "item": {"type": ["string", "null"], "description": "Optional crop name to compare. Required if metric is 'hg/ha_yield'"},
                },
                "required": ["countries", "metric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_countries",
            "description": "Get the top N countries ranked by a specific metric (e.g. highest rainfall, highest yield, highest temperature).",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string", "description": "Metric to rank by: 'average_rain_fall_mm_per_year', 'avg_temp', 'pesticides_tonnes', or 'hg/ha_yield'"},
                    "item": {"type": ["string", "null"], "description": "Crop name, ONLY required if metric is 'hg/ha_yield'"},
                    "year": {"type": ["integer", "null"], "description": "Optional year filter"},
                    "top_n": {"type": "integer", "description": "Number of countries (default 10)"},
                },
                "required": ["metric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_correlation",
            "description": "Pearson correlation between two numeric columns. Columns: hg/ha_yield, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp.",
            "parameters": {
                "type": "object",
                "properties": {
                    "col_x": {"type": "string", "description": "First column"},
                    "col_y": {"type": "string", "description": "Second column"},
                },
                "required": ["col_x", "col_y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_summary_statistics",
            "description": "Summary statistics (mean, median, std, min, max) for a numeric column. Columns: hg/ha_yield, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Year.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {"type": "string", "description": "Numeric column name"},
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_pesticide_usage",
            "description": "Pesticide usage (tonnes) over time, optionally filtered by country.",
            "parameters": {
                "type": "object",
                "properties": {
                    "country": {"type": ["string", "null"], "description": "Optional country name"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forecast_yield_trend",
            "description": "Linear regression forecast of future crop yields for a country+crop.",
            "parameters": {
                "type": "object",
                "properties": {
                    "country": {"type": "string", "description": "Country name"},
                    "item": {"type": "string", "description": "Crop name"},
                    "forecast_years": {"type": "integer", "description": "Years to forecast (default 5)"},
                },
                "required": ["country", "item"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_yield_regression_factors",
            "description": "Impact of rainfall, temperature, and pesticides on yield for a country+crop.",
            "parameters": {
                "type": "object",
                "properties": {
                    "country": {"type": "string", "description": "Country name"},
                    "item": {"type": "string", "description": "Crop name"},
                },
                "required": ["country", "item"],
            },
        },
    },
]



def get_tools_description() -> str:
    """Return a human-readable summary of all available tools.

    Used to inject tool context into the system prompt.
    """
    lines: list[str] = []
    for schema in TOOL_SCHEMAS:
        func = schema["function"]
        params = func["parameters"].get("properties", {})
        param_str = ", ".join(
            f"{k}: {v.get('type', 'any')}" for k, v in params.items()
        )
        lines.append(f"- {func['name']}({param_str}): {func['description']}")
    return "\n".join(lines)


def dispatch_tool_call(name: str, arguments: dict[str, Any]) -> Any:
    """Look up *name* in the registry and invoke it with *arguments*.

    Parameters
    ----------
    name : str
        Tool function name as returned by the LLM.
    arguments : dict[str, Any]
        Keyword arguments to forward.

    Returns
    -------
    Any
        The tool function's return value.

    Raises
    ------
    ValueError
        If *name* is not found in ``TOOL_REGISTRY``.
    """
    func = TOOL_REGISTRY.get(name)
    if func is None:
        raise ValueError(f"Unknown tool: {name}")
    return func(**arguments)
