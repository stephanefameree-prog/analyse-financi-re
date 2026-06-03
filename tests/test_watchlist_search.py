"""Recherche Yahoo/FMP pour ajout watchlist par nom de société."""
from data_loader import parse_fmp_search_results, parse_yahoo_search_quotes, search_market_symbols
import watchlists as wl


def test_parse_yahoo_search_quotes_filters_equities():
    quotes = [
        {"symbol": "AAPL", "shortname": "Apple Inc.", "quoteType": "EQUITY", "exchange": "NMS"},
        {"symbol": "AAPL.NEWS", "quoteType": "NEWS"},
        {"symbol": "MC.PA", "longname": "LVMH", "quoteType": "EQUITY", "exchange": "PAR"},
    ]
    out = parse_yahoo_search_quotes(quotes)
    assert [x["symbol"] for x in out] == ["AAPL", "MC.PA"]


def test_looks_like_ticker():
    assert wl._looks_like_ticker("MC.PA")
    assert wl._looks_like_ticker("^FCHI")
    assert not wl._looks_like_ticker("apple")
    assert not wl._looks_like_ticker("AAPL")
    assert not wl._looks_like_ticker("LVMH")


def test_pick_watchlist_ticker_from_search():
    hits = [{"symbol": "AAPL", "name": "Apple Inc.", "exchange": "NMS", "type": "EQUITY"}]
    label = wl._format_search_hit(hits[0])
    assert wl.pick_watchlist_ticker("apple", hits, label) == "AAPL"


def test_pick_watchlist_ticker_direct_symbol():
    assert wl.pick_watchlist_ticker("MC.PA", [], None) == "MC.PA"


def test_parse_fmp_search_results():
    items = [
        {
            "symbol": "AAPL",
            "name": "Apple Inc.",
            "exchangeShortName": "NASDAQ",
        },
        {"symbol": "", "name": "Invalid"},
    ]
    out = parse_fmp_search_results(items)
    assert [x["symbol"] for x in out] == ["AAPL"]
    assert out[0]["name"] == "Apple Inc."
    assert out[0]["exchange"] == "NASDAQ"


def test_search_market_symbols_falls_back_to_fmp(monkeypatch):
    monkeypatch.setattr(
        "data_loader.search_yahoo_symbols",
        lambda query, max_results=8: [],
    )
    monkeypatch.setattr(
        "data_loader.search_fmp_symbols",
        lambda query, max_results=8: [
            {"symbol": "AAPL", "name": "Apple Inc.", "exchange": "NASDAQ", "type": "EQUITY"}
        ],
    )
    search_market_symbols.clear()
    hits = search_market_symbols("apple")
    assert hits[0]["symbol"] == "AAPL"
