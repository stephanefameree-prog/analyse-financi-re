"""Canal de régression linéaire / log."""
import numpy as np
import pandas as pd

from analytics import (
    build_regression_channel_figure,
    compute_linear_regression_channel,
    compute_log_regression_channel,
)


def test_linear_regression_channel_bands():
    y = np.linspace(100, 130, 30) + np.random.default_rng(0).normal(0, 2, 30)
    s = pd.Series(y, index=pd.date_range("2024-01-01", periods=30, freq="B"))
    ch = compute_linear_regression_channel(s)
    assert ch is not None
    assert np.allclose(ch["plus_1std"].values - ch["regression"].values, ch["std"])
    assert np.allclose(ch["plus_2std"].values - ch["regression"].values, 2 * ch["std"])


def test_log_regression_channel_positive():
    t = np.arange(40)
    y = 100 * np.exp(0.001 * t) * (1 + 0.02 * np.sin(t / 3))
    s = pd.Series(y, index=pd.date_range("2024-01-01", periods=40, freq="B"))
    ch = compute_log_regression_channel(s)
    assert ch is not None
    assert (ch["minus_2std"] > 0).all()
    assert (ch["plus_1std"] > ch["regression"]).all()
    assert (ch["minus_1std"] < ch["regression"]).all()


def test_build_regression_channel_figure_log():
    s = pd.Series(
        np.exp(np.linspace(4.6, 4.7, 25)),
        index=pd.date_range("2024-01-01", periods=25, freq="B"),
    )
    fig = build_regression_channel_figure(s, scale="log", series_name="Test")
    assert fig is not None
    assert fig.layout.yaxis.type == "log"
