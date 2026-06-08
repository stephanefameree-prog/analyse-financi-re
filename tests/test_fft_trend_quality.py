"""R² / RMSE tendance FFT — interprétation et unités."""
import numpy as np
import pandas as pd

from analytics import (
    FFT_PRICE_UNIT_EUR,
    FFT_PRICE_UNIT_INDEX,
    FFT_PRICE_UNIT_PORTFOLIO,
    _trend_fit_metrics,
    compute_fft_periodicity,
    format_fft_rmse_display,
    interpret_fft_trend_r2,
    summarize_fft_trend_quality,
)


def test_trend_fit_metrics_includes_rmse_pct():
    prices = np.array([100.0, 102.0, 105.0, 103.0, 108.0])
    trend = np.array([100.0, 101.5, 103.0, 104.5, 106.0])
    r2, rmse, mean_p, rmse_pct = _trend_fit_metrics(prices, trend)
    assert 0 <= r2 <= 1
    assert rmse > 0
    assert mean_p == 103.6
    assert abs(rmse_pct - rmse / mean_p * 100) < 1e-6


def test_interpret_fft_trend_r2_bands():
    assert interpret_fft_trend_r2(0.90)["label"] == "Excellente"
    assert interpret_fft_trend_r2(0.70)["label"] == "Bonne"
    assert interpret_fft_trend_r2(0.55)["label"] == "Modérée"
    assert interpret_fft_trend_r2(0.40)["label"] == "Faible"
    assert interpret_fft_trend_r2(0.13)["label"] == "Très faible"


def test_format_fft_rmse_display_units():
    abs_eur, pct = format_fft_rmse_display(5.5, 2.5, FFT_PRICE_UNIT_EUR)
    assert "€ / action" in abs_eur
    assert pct == "2.5 %"
    abs_idx, _ = format_fft_rmse_display(3.2, 3.2, FFT_PRICE_UNIT_INDEX)
    assert "pts" in abs_idx
    abs_pf, _ = format_fft_rmse_display(1797.85, 3.6, FFT_PRICE_UNIT_PORTFOLIO)
    assert "valorisation portefeuille" in abs_pf


def test_summarize_fft_trend_quality_from_fft_result():
    t = np.arange(120)
    log_p = 0.0003 * t + 0.05 * np.sin(2 * np.pi * t / 21)
    s = pd.Series(
        np.exp(log_p) * 100,
        index=pd.date_range("2020-01-01", periods=120, freq="B"),
    )
    result = compute_fft_periodicity(s, min_period_days=5, max_period_days=60)
    assert result is not None
    summary = summarize_fft_trend_quality(result, price_unit=FFT_PRICE_UNIT_EUR)
    assert "R²" in summary["caption_md"]
    assert "RMSE" in summary["caption_md"]
    assert "fiabilité" in summary["caption_md"].lower()
