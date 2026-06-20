"""Sync cache dividendes → dividendes_universe.json."""
import json
from pathlib import Path

import pandas as pd
import pytest

import dividendes as div


def test_cache_row_to_universe_entry_maps_active_ticker():
    row = {
        "Ticker": "AI.PA",
        "Statut": "Actif",
        "Prix": 165.0,
        "Dividende annuel (TTM)": 3.25,
        "Rendement": 0.0197,
        "Dernier versement": "2024-05-15",
        "CAGR 5 ans": 0.06,
        "Années de croissance": 8,
        "Payout": 0.55,
        "FCF Payout": 0.48,
    }
    listing = {
        "Société": "Air Liquide",
        "Secteur": "Basic Materials",
        "Marché": "Euronext Paris",
        "Devise": "EUR",
    }
    entry = div._cache_row_to_universe_entry(row, listing=listing, source="YahooQuery")
    assert entry["Ticker"] == "AI.PA"
    assert entry["Société"] == "Air Liquide"
    assert entry["Rendement (%)"] == 0.0197
    assert entry["Dividende TTM"] == 3.25
    assert entry["A des dividendes"] is True
    assert entry["Série croissance (ans)"] == 8
    assert entry["Ratio couverture"] == pytest.approx(1 / 0.55)


def test_sync_preserves_existing_universe_detail_fields(tmp_path, monkeypatch):
    universe_path = tmp_path / "dividendes_universe.json"
    universe_path.write_text(
        json.dumps(
            {
                "version": 1,
                "meta": {"updated_at": "2024-01-01T00:00:00"},
                "tickers": {
                    "AI.PA": {
                        "Ticker": "AI.PA",
                        "Société": "Air Liquide",
                        "Fréquence": "Annuelle",
                        "Ex-dividende à venir": "2025-05-10",
                        "A des dividendes": True,
                    }
                },
                "failed_tickers": {"AI.PA": "old error"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(div, "_dividends_universe_path", lambda: universe_path)

    row = {
        "Ticker": "AI.PA",
        "Statut": "Actif",
        "Prix": 170.0,
        "Dividende annuel (TTM)": 3.40,
        "Rendement": 0.02,
        "Dernier versement": "2025-05-16",
    }
    div.sync_dividend_universe_from_cache_row("AI.PA", row, source="yfinance")

    data = json.loads(universe_path.read_text(encoding="utf-8"))
    entry = data["tickers"]["AI.PA"]
    assert entry["Dividende TTM"] == 3.40
    assert entry["Fréquence"] == "Annuelle"
    assert entry["Ex-dividende à venir"] == "2025-05-10"
    assert "AI.PA" not in data["failed_tickers"]


def test_dedupe_universe_merges_case_variants():
    universe = {
        "tickers": {
            "AI.PA": {"Ticker": "AI.PA", "Rendement (%)": 0.02, "Fréquence": "Annuelle"},
            "ai.pa": {"Ticker": "ai.pa", "Rendement (%)": 0.03},
        },
        "failed_tickers": {"AI.PA": "stale error"},
    }
    div._dedupe_universe_ticker_maps(universe)
    assert len(universe["tickers"]) == 1
    key = next(iter(universe["tickers"]))
    assert key in ("AI.PA", "ai.pa")
    entry = universe["tickers"][key]
    assert entry["Fréquence"] == "Annuelle"
    assert entry["Rendement (%)"] == 0.03
    assert universe["failed_tickers"] == {}


def test_universe_rows_from_store_uses_json_keys():
    rows = div._universe_rows_from_store(
        {"BNP.PA": {"Société": "BNP", "Rendement (%)": 0.05}}
    )
    assert len(rows) == 1
    assert rows[0]["Ticker"] == "BNP.PA"
    assert rows[0]["Société"] == "BNP"


def test_is_ticker_in_universe_case_insensitive():
    universe = {"tickers": {"MC.PA": {}}, "failed_tickers": {}}
    assert div._is_ticker_in_universe("mc.pa", universe)
    assert not div._is_ticker_in_universe("SAN.PA", universe)


def test_build_dividend_universe_listing_df_filters_dividends():
    universe = {
        "tickers": {
            "A.PA": {"Société": "Alpha", "A des dividendes": True},
            "B.PA": {"Société": "Beta", "A des dividendes": False},
        }
    }
    all_df = div.build_dividend_universe_listing_df(universe, only_with_dividends=False)
    assert set(all_df["Ticker"]) == {"A.PA", "B.PA"}
    div_df = div.build_dividend_universe_listing_df(universe, only_with_dividends=True)
    assert list(div_df["Ticker"]) == ["A.PA"]


def test_dividend_universe_company_names():
    df = pd.DataFrame({"Ticker": ["A.PA", "B.PA"], "Société": ["Alpha SA", "Beta SA"]})
    names = div.dividend_universe_company_names(df)
    assert names["A.PA"] == "Alpha SA"
