"""Tests graphiques analyse technique (Lot 4)."""
import numpy as np
import pandas as pd

from analytics import (
    build_technical_macd_figure,
    build_technical_overview_figure,
    build_technical_stochastic_figure,
    compute_rsi,
    compute_support_resistance,
)


def _sample_ohlc(n=80):
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.default_rng(0).normal(0, 1, n))
    ohlc = pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1,
            "low": close - 1,
            "close": close,
        },
        index=idx,
    )
    return close, ohlc


def test_build_technical_overview_figure_with_sr_and_rsi():
    close, ohlc = _sample_ohlc()
    s = pd.Series(close, index=ohlc.index)
    rsi = compute_rsi(s, 14)
    supports, resistances = compute_support_resistance(s, window=5, n_levels=2)
    vol = pd.Series(np.random.default_rng(1).integers(1000, 5000, len(s)), index=s.index)
    fig = build_technical_overview_figure(
        s,
        detail_label="Test",
        ohlc=ohlc,
        volume_series=vol,
        rsi_series=rsi,
        rsi_period=14,
        supports=supports,
        resistances=resistances,
    )
    assert fig is not None
    assert len(fig.data) >= 4
    assert fig.layout.height >= 700


def test_build_technical_stochastic_figure():
    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    k = pd.Series(np.linspace(40, 60, 30), index=idx)
    d = k.rolling(3).mean()
    fig = build_technical_stochastic_figure(k, d, detail_label="Test")
    assert fig is not None


def test_build_technical_macd_figure():
    idx = pd.date_range("2024-01-01", periods=40, freq="B")
    macd = pd.Series(np.sin(np.linspace(0, 4, 40)), index=idx)
    signal = macd.rolling(3).mean()
    hist = macd - signal
    fig = build_technical_macd_figure(macd, signal, hist, detail_label="Test")
    assert fig is not None
