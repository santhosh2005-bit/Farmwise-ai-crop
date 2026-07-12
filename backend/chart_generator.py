"""
chart_generator.py — Dynamic Plotly chart generation.

The LLM decides *what* chart to render and passes structured parameters.
This module translates those parameters into Plotly figures without any
hardcoded question-to-chart mapping.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio


# ─── Consistent professional theme ──────────────────────────
_CHART_TEMPLATE: str = "plotly_white"
_COLOR_SEQUENCE: list[str] = [
    "#10B981",  # Emerald Green (Agritech feel)
    "#3B82F6",  # Indigo/Blue
    "#06B6D4",  # Cyan/Teal
    "#8B5CF6",  # Purple
    "#F97316",  # Orange
    "#EC4899",  # Pink
    "#F59E0B",  # Amber
    "#14B8A6",  # Teal
    "#6366F1",  # Indigo
    "#EF4444",  # Coral Red
]

# ─── Supported chart types ───────────────────────────────────
CHART_TYPES: list[str] = [
    "bar",
    "line",
    "scatter",
    "pie",
    "histogram",
    "box",
    "heatmap",
    "area",
]

# ─── Chart builder map ───────────────────────────────────────
# Maps chart_type string → plotly express function.
_CHART_BUILDERS: dict[str, Any] = {
    "bar": px.bar,
    "line": px.line,
    "scatter": px.scatter,
    "pie": px.pie,
    "histogram": px.histogram,
    "box": px.box,
    "area": px.area,
}


def create_chart(
    chart_type: str,
    data: list[dict[str, Any]],
    x: str,
    y: str,
    title: str = "",
    color: str | None = None,
    labels: dict[str, str] | None = None,
) -> go.Figure:
    """Build a Plotly figure from LLM-specified parameters.

    Parameters
    ----------
    chart_type : str
        One of ``CHART_TYPES`` (e.g. ``"bar"``, ``"line"``).
    data : list[dict]
        Row data in records orientation.
    x : str
        Column name for the X axis.
    y : str
        Column name for the Y axis.
    title : str
        Chart title.
    color : str | None
        Optional column for colour grouping.
    labels : dict[str, str] | None
        Optional axis-label overrides.

    Returns
    -------
    plotly.graph_objects.Figure
        A fully-configured Plotly figure ready for rendering.

    Raises
    ------
    ValueError
        If *chart_type* is not supported.
    """
    chart_type = chart_type.strip().lower()

    if chart_type not in CHART_TYPES:
        raise ValueError(
            f"Unsupported chart type '{chart_type}'. "
            f"Choose from: {CHART_TYPES}"
        )

    df = pd.DataFrame(data)

    if df.empty:
        # Return an empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for this query.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="grey"),
        )
        fig.update_layout(title=title or "No Data")
        return fig

    # For bar charts, sort data by y-value descending to show a clear structure (highest to lowest)
    if chart_type == "bar" and y in df.columns:
        df = df.sort_values(by=y, ascending=False)

    # For line charts, sort by x ascending and aggregate y to prevent overlapping zig-zag lines
    if chart_type == "line" and x in df.columns and y in df.columns:
        df = df.sort_values(by=x, ascending=True)
        group_cols = [x]
        if color and color in df.columns:
            group_cols.append(color)
        if pd.api.types.is_numeric_dtype(df[y]):
            df = df.groupby(group_cols)[y].mean().reset_index()

    # Build common kwargs
    kwargs: dict[str, Any] = {
        "data_frame": df,
        "title": title,
        "template": _CHART_TEMPLATE,
        "color_discrete_sequence": _COLOR_SEQUENCE,
    }

    if labels:
        kwargs["labels"] = labels

    # ── Color handling ────────────────────────────────────────
    if color and color in df.columns:
        kwargs["color"] = color
        # If the color column is numeric (e.g. Year), use a vivid colorscale
        # instead of the faint default viridis
        if pd.api.types.is_numeric_dtype(df[color]):
            # Remove discrete sequence — will use continuous color scale instead
            kwargs.pop("color_discrete_sequence", None)
            kwargs["color_continuous_scale"] = [
                [0.0,  "#3B82F6"],   # bright blue
                [0.25, "#8B5CF6"],   # purple
                [0.5,  "#10B981"],   # emerald
                [0.75, "#F97316"],   # orange
                [1.0,  "#EF4444"],   # red
            ]

    # ── Special handling per chart type ───────────────────────

    if chart_type == "pie":
        # Pie charts use 'names' and 'values' instead of x/y
        kwargs["names"] = x
        kwargs["values"] = y
        # Pie charts are categorical so always use discrete sequence
        kwargs.pop("color_continuous_scale", None)
        if "color_discrete_sequence" not in kwargs:
            kwargs["color_discrete_sequence"] = _COLOR_SEQUENCE
        fig = px.pie(**kwargs)

    elif chart_type == "histogram":
        # Histograms only need the x column
        kwargs["x"] = x
        fig = px.histogram(**kwargs)

    elif chart_type == "heatmap":
        # Heatmap via density_heatmap
        kwargs["x"] = x
        kwargs["y"] = y
        fig = px.density_heatmap(**kwargs)

    else:
        # Standard x/y charts: bar, line, scatter, box, area
        kwargs["x"] = x
        kwargs["y"] = y

        builder = _CHART_BUILDERS[chart_type]
        fig = builder(**kwargs)

    # ── Polish layout ────────────────────────────────────────

    # Style modifications specific to chart types
    if chart_type == "line":
        fig.update_traces(line=dict(width=3.5, shape="linear"), marker=dict(size=6))
    elif chart_type == "bar":
        try:
            fig.update_traces(marker_cornerradius=8)
        except Exception:
            pass
    elif chart_type == "scatter":
        fig.update_traces(marker=dict(size=9, opacity=0.85, line=dict(width=1, color='white')))
    elif chart_type == "pie":
        fig.update_traces(
            hole=0.45,
            textinfo="percent+label",
            textfont=dict(size=13, color="#1E293B"),
            marker=dict(line=dict(color="#ffffff", width=2)),
        )

    # Axis text — always high-contrast dark so it's readable on any background
    _AXIS_TEXT_COLOR = "#1E293B"
    _AXIS_TITLE_COLOR = "#0F172A"
    _GRID_COLOR = "rgba(148, 163, 184, 0.4)"

    fig.update_xaxes(
        showgrid=False,
        linecolor=_GRID_COLOR,
        tickfont=dict(color=_AXIS_TEXT_COLOR, size=11),
        title_font=dict(color=_AXIS_TITLE_COLOR, size=12, family="Inter, system-ui, sans-serif"),
        zeroline=False,
    )

    # Rotate x-axis labels if there are many unique values or if the x column is categorical
    if x in df.columns:
        is_categorical = df[x].dtype == 'object' or isinstance(df[x].dtype, pd.CategoricalDtype)
        if is_categorical or df[x].nunique() > 10:
            fig.update_xaxes(tickangle=45)

    fig.update_yaxes(
        showgrid=True,
        gridcolor=_GRID_COLOR,
        gridwidth=1,
        linecolor=_GRID_COLOR,
        tickfont=dict(color=_AXIS_TEXT_COLOR, size=11),
        title_font=dict(color=_AXIS_TITLE_COLOR, size=12, family="Inter, system-ui, sans-serif"),
        zeroline=False,
    )

    # Determine if we should show the legend
    show_legend = True
    if color and color in df.columns:
        if color == x or df[color].nunique() > 10:
            show_legend = False

    fig.update_layout(
        font=dict(family="Inter, system-ui, sans-serif", size=13, color=_AXIS_TEXT_COLOR),
        title_font=dict(size=18, color="#0F172A", family="Inter, system-ui, sans-serif"),
        plot_bgcolor="rgba(255,255,255,0.92)",
        paper_bgcolor="rgba(255,255,255,0.0)",
        margin=dict(l=50, r=20, t=60, b=50),
        coloraxis_colorbar=dict(
            tickfont=dict(color=_AXIS_TEXT_COLOR),
            title_font=dict(color=_AXIS_TEXT_COLOR),
        ),
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.28,
            xanchor="center",
            x=0.5,
            font=dict(color=_AXIS_TEXT_COLOR, size=11),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(148,163,184,0.3)",
            borderwidth=1,
        ),
        hovermode="closest",
    )

    return fig



def figure_to_json(fig: go.Figure) -> str:
    """Serialise a Plotly figure to a JSON string for API transport.

    Parameters
    ----------
    fig : go.Figure
        The Plotly figure to serialise.

    Returns
    -------
    str
        JSON representation consumable by ``plotly.io.from_json``.
    """
    return pio.to_json(fig)
