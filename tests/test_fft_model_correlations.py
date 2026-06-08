"""Corrélation cours vs modèle FFT (avec / sans dé-trend)."""
import numpy as np
import pandas as pd

from analytics import compute_fft_model_correlations, compute_fft_periodicity


def _synthetic_price(n=220, period=21, amp=0.06):
    t = np.arange(n)
    log_p = 0.0004 * t + amp * np.sin(2 * np.pi * t / period)
    return pd.Series(
        np.exp(log_p) * 100,
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
    )


def test_compute_fft_model_correlations_returns_both():
    result = compute_fft_periodicity(
        _synthetic_price(), min_period_days=5, max_period_days=120, top_n=3
    )
    assert result is not None
    indices = result["peaks"].head(3)["_freq_idx"].tolist()
    corrs = compute_fft_model_correlations(result, indices)
    assert corrs is not None
    assert "corr_with_trend_filter" in corrs
    assert "corr_without_trend_filter" in corrs
    assert corrs["corr_with_trend_filter"] > 0.5
    assert -1.0 <= corrs["corr_without_trend_filter"] <= 1.0


def test_compute_fft_model_correlations_none_inputs():
    assert compute_fft_model_correlations(None, [1]) is None
    result = compute_fft_periodicity(_synthetic_price(), min_period_days=5, max_period_days=120)
    assert compute_fft_model_correlations(result, []) is None
