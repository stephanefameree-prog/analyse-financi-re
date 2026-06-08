"""Tests cache OHLCV par ticker (data_loader)."""
import pandas as pd
import pytest

from data_loader import (
    _assemble_bundle_from_ticker_caches,
    _extract_ticker_payload,
    _merge_series,
    _merge_ticker_payload,
    _save_ticker_cache,
    _load_ticker_cache,
    clear_market_cache,
)


@pytest.fixture(autouse=True)
def _clean_cache():
    clear_market_cache()
    yield
    clear_market_cache()


def test_merge_series_keeps_latest_on_duplicate_index():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    old = pd.Series([1.0, 2.0, 3.0], index=idx)
    new = pd.Series([3.5], index=[idx[-1]])
    merged = _merge_series(old, new)
    assert float(merged.iloc[-1]) == 3.5
    assert len(merged) == 3


def test_ticker_cache_roundtrip_and_assemble():
    start = "2020-01-01"
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    bundle = {
        "prices": pd.DataFrame({"AAA": pd.Series(range(5), index=idx, dtype=float)}),
        "volumes": pd.DataFrame({"AAA": pd.Series([100] * 5, index=idx, dtype=float)}),
        "highs": pd.DataFrame(),
        "lows": pd.DataFrame(),
        "opens": pd.DataFrame(),
        "ohlc_closes": pd.DataFrame(),
    }
    payload = _extract_ticker_payload(bundle, "AAA")
    _save_ticker_cache("AAA", start, payload)

    loaded = _load_ticker_cache("AAA", start)
    assert loaded is not None
    assert float(loaded["prices"].iloc[-1]) == 4.0

    assembled = _assemble_bundle_from_ticker_caches(["AAA"], start)
    assert "AAA" in assembled["prices"].columns
    assert len(assembled["prices"]) == 5


def test_clear_market_cache_with_signature_deletes_ticker_files():
    start = "2020-01-01"
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    bundle = {
        "prices": pd.DataFrame({"AAA": pd.Series([1.0, 2.0, 3.0], index=idx)}),
        "volumes": pd.DataFrame(),
        "highs": pd.DataFrame(),
        "lows": pd.DataFrame(),
        "opens": pd.DataFrame(),
        "ohlc_closes": pd.DataFrame(),
    }
    payload = _extract_ticker_payload(bundle, "AAA")
    _save_ticker_cache("AAA", start, payload)
    assert _load_ticker_cache("AAA", start) is not None

    from data_loader import _tickers_signature

    sig = _tickers_signature(["AAA"], start)
    clear_market_cache(sig, tickers=["AAA"], start=start)
    assert _load_ticker_cache("AAA", start) is None


def test_merge_ticker_payload_incremental():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    base = {
        "prices": pd.Series([1.0, 2.0, 3.0], index=idx),
        "volumes": pd.Series(dtype=float),
        "highs": pd.Series(dtype=float),
        "lows": pd.Series(dtype=float),
        "opens": pd.Series(dtype=float),
        "ohlc_closes": pd.Series(dtype=float),
    }
    recent = {
        "prices": pd.Series([3.5], index=[idx[-1]]),
        "volumes": pd.Series(dtype=float),
        "highs": pd.Series(dtype=float),
        "lows": pd.Series(dtype=float),
        "opens": pd.Series(dtype=float),
        "ohlc_closes": pd.Series(dtype=float),
    }
    merged = _merge_ticker_payload(base, recent)
    assert float(merged["prices"].iloc[-1]) == 3.5
