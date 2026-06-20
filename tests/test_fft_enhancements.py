"""Tests modes FFT étendus (MM41, STL, harmonique)."""
import numpy as np
import pandas as pd

from analytics import (
    FFT_RECON_FFT,
    FFT_RECON_HARMONIC,
    FFT_TREND_LOG_LINEAR,
    FFT_TREND_MA41_CAUSAL,
    FFT_TREND_MA41_CENTER,
    FFT_TREND_STL,
    _prepare_fft_detrended_series,
    build_fft_cyclic_chart,
    build_fft_extended_series,
    compute_fft_model_correlations,
    compute_fft_holdout_validation,
    compute_fft_periodicity,
    reconstruct_harmonic_cycles,
)


def _synthetic_price(n=200):
    t = np.arange(n, dtype=float)
    log_p = 0.001 * t + 0.02 * np.sin(2 * np.pi * t / 21) + 0.01 * np.sin(2 * np.pi * t / 63)
    return pd.Series(np.exp(log_p), index=pd.date_range("2022-01-01", periods=n, freq="B"))


def test_ma41_center_prepare_has_smooth_prices():
    prep = _prepare_fft_detrended_series(_synthetic_price(), trend_mode=FFT_TREND_MA41_CENTER)
    assert prep is not None
    assert "smooth_prices" in prep
    assert len(prep["smooth_prices"]) == prep["n_obs"]


def test_ma41_causal_prepare():
    prep = _prepare_fft_detrended_series(_synthetic_price(), trend_mode=FFT_TREND_MA41_CAUSAL)
    assert prep is not None
    assert prep["trend_mode"] == FFT_TREND_MA41_CAUSAL


def test_stl_prepare():
    prep = _prepare_fft_detrended_series(_synthetic_price(180), trend_mode=FFT_TREND_STL)
    assert prep is not None
    assert "smooth_prices" in prep


def test_harmonic_reconstruction_nonzero():
    y = np.sin(2 * np.pi * np.arange(120) / 21)
    recon = reconstruct_harmonic_cycles(y, [21], 120)
    assert len(recon) == 120
    assert np.std(recon) > 0.01


def test_compute_fft_model_correlations_extended_keys():
    result = compute_fft_periodicity(
        _synthetic_price(),
        min_period_days=5,
        max_period_days=120,
        trend_mode=FFT_TREND_MA41_CAUSAL,
    )
    assert result is not None
    indices = result["peaks"]["_freq_idx"].tolist()[:3]
    corrs = compute_fft_model_correlations(result, indices, n_components=3)
    assert corrs is not None
    assert "corr_harmonic" in corrs
    assert "corr_vs_smooth" in corrs
    assert corrs["corr_vs_smooth"] is not None


def test_build_fft_extended_harmonic():
    result = compute_fft_periodicity(
        _synthetic_price(),
        min_period_days=5,
        max_period_days=120,
        trend_mode=FFT_TREND_LOG_LINEAR,
    )
    indices = result["peaks"]["_freq_idx"].tolist()[:3]
    ext = build_fft_extended_series(
        result,
        indices,
        recon_mode=FFT_RECON_HARMONIC,
        n_components=3,
    )
    assert ext is not None
    assert len(ext["price_model"]) > result["n_obs"]


def test_recency_fit_improves_recent_tracking():
    """La calibration récente rapproche le modèle du cours en fin de série."""
    n = 180
    t = np.arange(n, dtype=float)
    log_p = -0.001 * t + 0.03 * np.sin(2 * np.pi * t / 21)
    log_p[-15:] += np.linspace(0, 0.35, 15)
    s = pd.Series(np.exp(log_p), index=pd.date_range("2024-01-01", periods=n, freq="B"))
    result = compute_fft_periodicity(s, min_period_days=5, max_period_days=120)
    assert result is not None
    indices = result["peaks"]["_freq_idx"].tolist()[:4]

    def _tail_rmse(recency):
        ext = build_fft_extended_series(
            result,
            indices,
            recon_mode=FFT_RECON_FFT,
            recency_weighted=recency,
            recency_calibrate=recency,
        )
        prices = ext["prices_hist"]
        model = ext["price_model"][: ext["n_hist"]]
        tail = slice(-20, None)
        err = np.sqrt(np.mean((prices[tail] - model[tail]) ** 2))
        return err

    assert _tail_rmse(True) <= _tail_rmse(False) * 1.05


def test_build_fft_cyclic_chart_ma_mode():
    result = compute_fft_periodicity(
        _synthetic_price(),
        trend_mode=FFT_TREND_MA41_CENTER,
        min_period_days=5,
        max_period_days=120,
    )
    fig = build_fft_cyclic_chart(result, n_components=3, recon_mode=FFT_RECON_FFT)
    assert fig is not None
    trace_names = [t.name for t in fig.data]
    assert any("MM41" in (n or "") for n in trace_names)


def test_fft_holdout_validation_on_synthetic():
    s = _synthetic_price(280)
    holdout = compute_fft_holdout_validation(
        s,
        holdout_days=30,
        min_period_days=5,
        max_period_days=120,
        trend_mode=FFT_TREND_LOG_LINEAR,
        recon_mode=FFT_RECON_FFT,
        n_components=3,
    )
    assert holdout is not None
    assert holdout["shifted_window"] is False
    assert len(holdout["actual_prices"]) == 30
    assert len(holdout["model_prices"]) == 30
    assert len(holdout["model_in_sample"]) == holdout["train_n_obs"]
    assert holdout["corr"] is not None
    assert holdout["corr_in_sample"] is not None


def test_fft_holdout_shifted_window():
    s = _synthetic_price(280)
    holdout = compute_fft_holdout_validation(
        s,
        holdout_days=30,
        min_period_days=5,
        max_period_days=120,
        shifted_window=True,
        n_components=3,
    )
    assert holdout is not None
    assert holdout["shifted_window"] is True
    assert holdout["train_n_obs"] == len(s) - 60


def test_build_fft_cyclic_chart_with_holdout():
    s = _synthetic_price(280)
    result = compute_fft_periodicity(s, min_period_days=5, max_period_days=120)
    holdout = compute_fft_holdout_validation(
        s, holdout_days=30, min_period_days=5, max_period_days=120, n_components=3
    )
    fig = build_fft_cyclic_chart(result, holdout_validation=holdout)
    names = " ".join((t.name or "").lower() for t in fig.data)
    assert "holdout" in names
    assert "calibrage" in names
    assert "projection" in names