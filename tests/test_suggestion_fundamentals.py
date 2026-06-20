"""Tests enrichissement fondamental des suggestions d'actifs."""
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from analytics import (
    build_risk_metrics_boxplot,
    build_risk_returns_boxplot,
    enrich_suggestions_with_technical,
    filter_suggestions_by_statistics,
    filter_suggestions_by_technical,
    filter_suggestions_by_latent_returns,
    merge_suggestion_latent_returns,
    merge_technical_columns,
    SUGGESTION_COL_LATENT_1M,
    SUGGESTION_COL_LATENT_1W,
    SUGGESTION_COL_LATENT_1Y,
)
import fundamentals
from fundamentals import (
    filter_suggestions_by_quality,
    merge_fundamentals_columns,
)
import dividendes


def test_merge_fundamentals_columns():
    suggestions = pd.DataFrame(
        {
            "Ticker": ["AAA", "BBB"],
            "Score composite": [0.5, 0.3],
        }
    )
    rows = [
        {"Ticker": "AAA", "ROE": 0.18, "Score Piotroski": 8, "PER": 14.0},
        {"Ticker": "BBB", "ROE": 0.05, "Score Piotroski": 4, "PER": 30.0},
    ]
    merged = merge_fundamentals_columns(suggestions, rows)
    assert merged.loc[0, "ROE"] == 0.18
    assert merged.loc[1, "Score Piotroski"] == 4


def test_filter_suggestions_by_quality():
    df = pd.DataFrame(
        {
            "Ticker": ["A", "B", "C"],
            "Score Piotroski": [8, 5, 2],
            "ROE": [0.20, 0.10, 0.01],
            "PER": [12.0, 18.0, 35.0],
            "Dette / Capitaux": [0.5, 1.2, 2.0],
        }
    )
    out = filter_suggestions_by_quality(
        df,
        min_piotroski=7,
        min_roe=0.15,
        max_per=15.0,
        max_debt_equity=1.0,
    )
    assert list(out["Ticker"]) == ["A"]


def test_merge_technical_columns():
    suggestions = pd.DataFrame({"Ticker": ["AAA"], "Score composite": [0.5]})
    tech = pd.DataFrame(
        {
            "Ticker": ["AAA"],
            "RSI": [42.0],
            "Bollinger %B": [0.55],
        }
    )
    merged = merge_technical_columns(suggestions, tech)
    assert merged.loc[0, "RSI"] == 42.0
    assert merged.loc[0, "Bollinger %B"] == 0.55


def test_enrich_suggestions_with_technical():
    dates = pd.date_range("2024-01-01", periods=120, freq="B")
    prices = pd.DataFrame(
        {
            "AAA": 100 + pd.Series(range(120)).values * 0.1,
            "BBB": 50 - pd.Series(range(120)).values * 0.05,
        },
        index=dates,
    )
    suggestions = pd.DataFrame({"Ticker": ["AAA", "BBB"], "Score composite": [0.5, 0.3]})
    enriched = enrich_suggestions_with_technical(suggestions, prices)
    assert "RSI" in enriched.columns
    assert enriched["RSI"].notna().any()


def test_filter_suggestions_by_technical():
    df = pd.DataFrame(
        {
            "Ticker": ["A", "B", "C"],
            "RSI": [28.0, 55.0, 72.0],
            "Bollinger %B": [0.1, 0.5, 0.9],
        }
    )
    out = filter_suggestions_by_technical(
        df, max_rsi=65.0, min_rsi=30.0, max_bollinger_b=0.7, min_bollinger_b=0.15
    )
    assert list(out["Ticker"]) == ["B"]


def test_filter_suggestions_by_statistics():
    df = pd.DataFrame(
        {
            "Ticker": ["A", "B", "C"],
            "Rendement candidat": [0.12, 0.05, 0.08],
            "Volatilité candidat": [0.18, 0.30, 0.22],
            "Kurtosis candidat": [2.0, 6.0, 3.0],
            "Skewness candidat": [0.5, -0.2, 0.1],
            "Sharpe candidat": [1.2, 0.3, 0.8],
            "Corr. moy. portefeuille": [0.4, 0.8, 0.55],
            "Δ Rendement portef.": [0.02, -0.01, 0.005],
            "Δ Volatilité portef.": [0.01, -0.02, 0.0],
            "Δ Kurtosis portef.": [0.3, -0.5, 0.1],
            "Δ Skewness portef.": [0.2, -0.1, 0.05],
            "Δ Sharpe portef.": [0.15, -0.2, 0.05],
            "Δ Corr. interne": [0.05, -0.1, 0.02],
        }
    )
    out = filter_suggestions_by_statistics(
        df,
        min_candidate_return=0.10,
        max_candidate_vol=0.25,
        max_candidate_kurtosis=4.0,
        min_candidate_skewness=0.0,
        min_candidate_sharpe=0.5,
        max_corr_portfolio=0.6,
        min_delta_return=0.0,
        min_delta_vol=0.0,
        min_delta_kurtosis=0.0,
        min_delta_skewness=0.0,
        min_delta_sharpe=0.0,
        min_delta_corr_internal=0.0,
    )
    assert list(out["Ticker"]) == ["A"]


def test_merge_dividend_columns():
    suggestions = pd.DataFrame({"Ticker": ["AAA"], "Score composite": [0.5]})
    rows = [
        {
            "Ticker": "AAA",
            "Statut": "Actif",
            "Rendement": 0.04,
            "CAGR 5 ans": 0.05,
            "Années de croissance": 6,
            "Payout": 0.5,
            "FCF Payout": 0.55,
            "Ratio couverture": 2.0,
            "Score": 75,
        }
    ]
    merged = dividendes.merge_dividend_columns(suggestions, rows)
    assert merged.loc[0, "Rendement"] == 0.04
    assert merged.loc[0, "Ratio couverture"] == 2.0


def test_normalize_suggestion_dividend_row():
    row = dividendes._normalize_suggestion_dividend_row(
        {"Ticker": "X", "Payout": 0.5, "FCF Payout": 0.4}
    )
    assert row["Ratio couverture"] == 2.0
    assert row["Ratio couverture FCF"] == 2.5


def test_filter_suggestions_by_dividends():
    df = pd.DataFrame(
        {
            "Ticker": ["A", "B", "C"],
            "Statut": ["Actif", "Actif", "Suspendu"],
            "Rendement": [0.04, 0.015, 0.06],
            "Ratio couverture": [1.8, 1.0, 2.5],
            "CAGR 5 ans": [0.05, 0.02, 0.08],
            "Années de croissance": [8, 2, 10],
            "Payout": [0.55, 0.95, 0.40],
            "FCF Payout": [0.60, 0.70, 0.50],
            "Score": [80, 40, 90],
        }
    )
    out = dividendes.filter_suggestions_by_dividends(
        df,
        min_yield=0.02,
        min_coverage=1.5,
        min_cagr_5y=0.03,
        min_growth_years=5,
        max_payout=0.80,
        max_fcf_payout=0.75,
        active_only=True,
    )
    assert list(out["Ticker"]) == ["A"]


def test_fundamental_ticker_disk_cache_roundtrip(tmp_path, monkeypatch):
    cache_file = tmp_path / "fundamentals_cache.json"
    monkeypatch.setattr(fundamentals, "_fundamentals_cache_path", lambda: cache_file)
    row = {
        "Ticker": "TEST.PA",
        "ROE": 0.2,
        "_fetched_at": datetime.now().isoformat(timespec="seconds"),
    }
    fundamentals.save_fundamental_ticker_to_disk("TEST.PA", row)
    loaded = fundamentals.load_fundamental_ticker_from_disk("TEST.PA")
    assert loaded["ROE"] == 0.2


def test_fundamentals_cache_write_fallback_when_replace_blocked(tmp_path, monkeypatch):
    cache_file = tmp_path / "fundamentals_cache.json"
    monkeypatch.setattr(fundamentals, "_fundamentals_cache_path", lambda: cache_file)
    real_replace = Path.replace

    def blocked_replace(self, target):
        if self.suffix == ".tmp":
            raise PermissionError(5, "Accès refusé")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", blocked_replace)
    row = {
        "Ticker": "BLOCK.PA",
        "ROE": 0.1,
        "_fetched_at": datetime.now().isoformat(timespec="seconds"),
    }
    fundamentals.save_fundamental_ticker_to_disk("BLOCK.PA", row)
    assert cache_file.is_file()
    loaded = fundamentals.load_fundamental_ticker_from_disk("BLOCK.PA")
    assert loaded["ROE"] == 0.1


def test_dividend_ticker_disk_cache_roundtrip(tmp_path, monkeypatch):
    cache_file = tmp_path / "dividendes_cache.json"
    monkeypatch.setattr(dividendes, "_dividends_cache_path", lambda: cache_file)
    row = dividendes._normalize_suggestion_dividend_row(
        {"Ticker": "TEST.PA", "Rendement": 0.03, "Payout": 0.5}
    )
    dividendes.save_dividend_ticker_to_disk("TEST.PA", row)
    loaded = dividendes.load_dividend_ticker_from_disk("TEST.PA")
    assert loaded["Rendement"] == 0.03


def test_merge_suggestion_latent_returns():
    n = 260
    aaa = [100.0] * (n - 6) + [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    prices = pd.DataFrame({"AAA": aaa, "BBB": [200.0] * n})
    suggestions = pd.DataFrame(
        {
            "Ticker": ["AAA", "BBB"],
            "Score composite": [0.5, 0.3],
            "Rendement candidat": [0.12, 0.05],
        }
    )
    merged = merge_suggestion_latent_returns(suggestions, prices)
    assert merged.loc[0, SUGGESTION_COL_LATENT_1W] == pytest.approx(0.05)
    assert merged.loc[1, SUGGESTION_COL_LATENT_1W] == 0.0
    assert pd.notna(merged.loc[0, SUGGESTION_COL_LATENT_1Y])
    assert list(merged.columns).index(SUGGESTION_COL_LATENT_1W) == 3


def test_filter_suggestions_by_latent_returns():
    df = pd.DataFrame(
        {
            "Ticker": ["A", "B", "C", "D"],
            SUGGESTION_COL_LATENT_1W: [0.05, -0.02, 0.01, -0.01],
            SUGGESTION_COL_LATENT_1M: [0.10, -0.05, 0.02, 0.03],
            SUGGESTION_COL_LATENT_1Y: [-0.20, 0.15, -0.08, -0.30],
        }
    )
    rebound = filter_suggestions_by_latent_returns(
        df,
        min_latent_1w=0.0,
        max_latent_1y=0.0,
    )
    assert list(rebound["Ticker"]) == ["A", "C"]

    custom = filter_suggestions_by_latent_returns(
        df,
        min_latent_1m=-0.10,
        max_latent_1m=0.05,
    )
    assert set(custom["Ticker"]) == {"B", "C", "D"}
