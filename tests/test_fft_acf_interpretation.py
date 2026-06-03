"""Commentaire interprétatif du graphique d'autocorrélation FFT."""
import numpy as np
import pandas as pd

from analytics import compute_fft_periodicity, interpret_fft_acf_reading


def _synthetic_price(n=220, period=21, amp=0.06):
    t = np.arange(n)
    log_p = 0.0004 * t + amp * np.sin(2 * np.pi * t / period)
    return pd.Series(
        np.exp(log_p) * 100,
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
    )


def test_interpret_fft_acf_reading_returns_text():
    result = compute_fft_periodicity(_synthetic_price(), min_period_days=5, max_period_days=120)
    assert result is not None
    comment = interpret_fft_acf_reading(result)
    assert comment
    assert "Comment lire le graphique du bas" in comment
    assert "En pratique" in comment


def test_interpret_fft_acf_reading_none():
    assert interpret_fft_acf_reading(None) is None
