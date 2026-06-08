"""Thème Plotly commun — couleurs, hauteurs et mise en page du dashboard."""

from __future__ import annotations

import plotly.graph_objects as go

PLOTLY_TEMPLATE = "plotly_white"

CHART_COLORS = {
    "primary": "#1565c0",
    "portfolio": "#1e40af",
    "current": "#7b1fa2",
    "target": "#ffd600",
    "positive": "#2e7d32",
    "negative": "#c62828",
    "neutral": "#64748b",
    "accent": "#2563eb",
    "random_portfolio": "#ef5350",
    "frontier": "#1565c0",
    "risk_free": "#212121",
}

CHART_HEIGHT = {
    "heatmap": 560,
    "scatter": 480,
    "time_series": 460,
    "frontier": 520,
    "radar": 420,
    "pedagogy": 400,
    "treemap": 440,
    "drawdown": 320,
    "stacked_area": 480,
    "technical_overview": 720,
    "technical_osc": 340,
    "gauge": 320,
    "histogram": 360,
}

CORRELATION_COLORSCALE = "RdBu_r"


def apply_chart_theme(
    fig: go.Figure,
    *,
    height: int | None = None,
    title: str | None = None,
    legend_horizontal: bool = True,
) -> go.Figure:
    """Applique le template, les marges et la typographie communs."""
    layout: dict = {
        "template": PLOTLY_TEMPLATE,
        "font": dict(
            family="Segoe UI, Roboto, Helvetica, Arial, sans-serif",
            size=12,
            color="#1e293b",
        ),
        "margin": dict(l=48, r=32, t=64 if title else 44, b=52),
        "paper_bgcolor": "rgba(255,255,255,0)",
        "plot_bgcolor": "rgba(255,255,255,0)",
    }
    if title:
        layout["title"] = dict(
            text=title,
            x=0,
            xanchor="left",
            font=dict(size=16, color="#0f172a"),
        )
    if height:
        layout["height"] = height
    if legend_horizontal:
        layout["legend"] = dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=1,
            xanchor="right",
            bgcolor="rgba(255,255,255,0.7)",
        )
    fig.update_layout(**layout)
    return fig
