"""Tests indicateur SuperTrend."""
import numpy as np
import pandas as pd

from analytics import (
    build_technical_overview_figure,
    compute_supertrend,
    compute_technical_indicators,
    interpret_supertrend_comment,
    supertrend_signal,
)


def _sample_ohlc(n=120, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(rng.normal(0.2, 1.0, n))
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    return idx, close, high, low


def test_compute_supertrend_returns_line_and_direction():
    idx, close, high, low = _sample_ohlc()
    st_line, st_dir = compute_supertrend(
        pd.Series(high, index=idx),
        pd.Series(low, index=idx),
        pd.Series(close, index=idx),
        period=10,
        multiplier=3.0,
    )
    assert len(st_line) == len(st_dir)
    assert not st_line.empty
    assert set(st_dir.dropna().unique()).issubset({1.0, -1.0})


def test_supertrend_signal_labels():
    assert supertrend_signal(1) == "Haussier"
    assert supertrend_signal(-1) == "Baissier"
    assert supertrend_signal(None) == "N/A"


def test_compute_technical_indicators_includes_supertrend():
    idx, close, high, low = _sample_ohlc(n=80)
    prices = pd.DataFrame({"AAA": close}, index=idx)
    highs = pd.DataFrame({"AAA": high}, index=idx)
    lows = pd.DataFrame({"AAA": low}, index=idx)
    tech = compute_technical_indicators(prices, highs=highs, lows=lows)
    assert not tech.empty
    assert "SuperTrend" in tech.columns
    assert "Signal SuperTrend" in tech.columns
    assert tech.iloc[0]["Signal SuperTrend"] in ("Haussier", "Baissier", "Neutre", "N/A")


def test_build_technical_overview_with_supertrend():
    idx, close, high, low = _sample_ohlc()
    s = pd.Series(close, index=idx)
    st_line, st_dir = compute_supertrend(
        pd.Series(high, index=idx),
        pd.Series(low, index=idx),
        s,
    )
    fig = build_technical_overview_figure(
        s,
        detail_label="Test",
        supertrend_series=st_line,
        supertrend_direction=st_dir,
    )
    assert fig is not None
    names = [t.name for t in fig.data]
    assert any("SuperTrend" in n for n in names)


def test_interpret_supertrend_comment():
    idx, close, high, low = _sample_ohlc()
    s = pd.Series(close, index=idx)
    st_line, st_dir = compute_supertrend(
        pd.Series(high, index=idx),
        pd.Series(low, index=idx),
        s,
    )
    text = interpret_supertrend_comment(s, st_line, st_dir)
    assert "SuperTrend" in text or "haussi" in text.lower() or "baissi" in text.lower()
