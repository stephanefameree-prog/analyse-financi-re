"""Histogrammes et filtres plage — univers dividendes."""
import pandas as pd

from dividendes import (
    build_dividend_universe_histogram,
    build_dividend_yield_coverage_scatter,
    filter_dividend_universe_by_ranges,
)


def test_filter_dividend_universe_by_ranges():
    df = pd.DataFrame(
        {
            "Ticker": ["A", "B", "C", "D"],
            "Rendement (%)": [0.02, 0.05, 0.08, 0.03],
            "Ratio couverture": [1.2, 2.0, 0.8, 1.5],
        }
    )
    out = filter_dividend_universe_by_ranges(
        df,
        yield_min=0.03,
        yield_max=0.07,
        coverage_min=1.0,
        coverage_max=1.8,
    )
    assert list(out["Ticker"]) == ["D"]


def test_filter_dividend_universe_by_frequency():
    df = pd.DataFrame(
        {
            "Ticker": ["A", "B", "C", "D"],
            "Fréquence": ["Trimestrielle", "Annuelle", "Trimestrielle", "Mensuelle"],
        }
    )
    out = filter_dividend_universe_by_ranges(
        df,
        frequencies=["Trimestrielle", "Mensuelle"],
    )
    assert list(out["Ticker"]) == ["A", "C", "D"]

    empty = filter_dividend_universe_by_ranges(df, frequencies=[])
    assert empty.empty


def test_dividend_frequency_options_sorted():
    df = pd.DataFrame(
        {
            "Fréquence": ["Annuelle", "Trimestrielle", "Mensuelle", "Annuelle"],
        }
    )
    from dividendes import _dividend_frequency_options

    assert _dividend_frequency_options(df) == [
        "Mensuelle",
        "Trimestrielle",
        "Annuelle",
    ]


def test_build_dividend_universe_histogram():
    values = pd.Series([0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])
    fig = build_dividend_universe_histogram(
        values,
        title="Test rendements",
        xaxis_title="Rendement",
        tickformat=".1%",
    )
    assert fig is not None
    assert len(fig.data) == 1
    assert fig.data[0].type == "histogram"


def test_build_dividend_yield_coverage_scatter():
    df = pd.DataFrame(
        {
            "Ticker": ["A", "B", "C"],
            "Société": ["Alpha", "Beta", "Gamma"],
            "Rendement (%)": [0.03, 0.05, 0.02],
            "Ratio couverture": [1.5, 2.2, 0.9],
            "Marché": ["Paris", "Paris", "NYSE"],
        }
    )
    fig = build_dividend_yield_coverage_scatter(df)
    assert fig is not None
    assert fig.data[0].mode == "markers"
