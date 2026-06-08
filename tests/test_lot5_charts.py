"""Tests graphiques Lot 5 (dividendes, fondamentaux, suggestions)."""
import pandas as pd

from analytics import build_suggestions_tradeoff_scatter
from dividendes import build_dividend_yield_score_scatter, radar_score_chart
from fundamentals import build_fair_value_gauge_figure


def test_dividend_yield_score_scatter_uses_markers_only():
    df = pd.DataFrame(
        {
            "Ticker": ["AI.PA", "BNP.PA"],
            "Société": ["Air Liquide", "BNP"],
            "Rendement": [0.02, 0.05],
            "Score": [45, 72],
            "Marché": ["Paris", "Paris"],
        }
    )
    fig = build_dividend_yield_score_scatter(df)
    assert fig is not None
    assert fig.data[0].mode == "markers"
    assert fig.data[0].text is None


def test_dividend_radar_normalized_0_100():
    scores = {
        "stability": 30,
        "growth": 25,
        "payout": 20,
        "fcf_payout": 20,
        "yield": 5,
    }
    fig = radar_score_chart(scores, "Test")
    assert fig is not None
    assert fig.layout.polar.radialaxis.range == (0, 100)


def test_fair_value_gauge_figure():
    fig = build_fair_value_gauge_figure(90.0, 105.0, ticker_label="BNP.PA")
    assert fig is not None
    assert fig.layout.height >= 300


def test_suggestions_tradeoff_scatter():
    df = pd.DataFrame(
        {
            "Ticker": ["A", "B"],
            "Corr. moy. portefeuille": [0.2, 0.5],
            "Rendement candidat": [0.08, 0.04],
            "Score composite": [0.7, 0.4],
        }
    )
    fig = build_suggestions_tradeoff_scatter(df)
    assert fig is not None
    assert fig.data[0].mode == "markers"
