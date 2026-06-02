"""Cache Streamlit (st.cache_data) pour calculs et appels réseau du dashboard."""
from __future__ import annotations

import streamlit as st
import pandas as pd
import yfinance as yf

from analytics import (
    _align_weights_to_returns,
    build_markowitz_frontier_figure,
    compute_advanced_risk_metrics,
    compute_bollinger,
    compute_corr,
    compute_fft_periodicity,
    compute_fft_summary_for_prices,
    compute_macd,
    compute_markowitz,
    compute_returns,
    compute_rsi,
    compute_sma,
    compute_stochastic,
    compute_technical_indicators,
    suggest_portfolio_additions,
)
from data_loader import get_ohlc_for_ticker, load_prices_in_batches

COMPUTE_TTL = 3600
FX_TTL = 1800
MARKET_TTL = 3600


@st.cache_data(ttl=FX_TTL, show_spinner=False)
def cached_usd_to_eur_rate(default: float = 0.92) -> float:
    try:
        fx_data = yf.download("EURUSD=X", period="5d", progress=False, threads=False)
        if fx_data.empty:
            return default
        close = fx_data["Close"]
        if isinstance(close, pd.DataFrame):
            last_close = float(close.iloc[-1, 0])
        else:
            last_close = float(close.iloc[-1])
        return 1 / last_close if last_close else default
    except Exception:
        return default


@st.cache_data(ttl=MARKET_TTL, show_spinner=False)
def cached_benchmark_prices(bench_ticker: str, start_date_str: str) -> pd.DataFrame:
    prices = load_prices_in_batches(
        [bench_ticker], start=start_date_str, refresh="cache_only"
    )
    if prices.empty:
        prices = load_prices_in_batches([bench_ticker], start=start_date_str, refresh="full")
    return prices


@st.cache_data(ttl=MARKET_TTL, show_spinner=False)
def cached_candidate_market_data(
    candidate_sig: str,
    candidate_pool: tuple,
    start_date_str: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tickers = list(candidate_pool)
    prices = load_prices_in_batches(tickers, start=start_date_str, refresh="cache_only")
    if prices.empty:
        prices = load_prices_in_batches(tickers, start=start_date_str, refresh="full")
    return prices, compute_returns(prices)


@st.cache_data(ttl=COMPUTE_TTL, show_spinner=False)
def cached_technical_indicators(
    ohlcv_sig: str,
    rsi_period: int,
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    highs: pd.DataFrame,
    lows: pd.DataFrame,
) -> pd.DataFrame:
    return compute_technical_indicators(
        prices,
        rsi_period=rsi_period,
        volumes=volumes,
        highs=highs,
        lows=lows,
    )


@st.cache_data(ttl=COMPUTE_TTL, show_spinner=False)
def cached_technical_detail_bundle(
    ohlcv_sig: str,
    ticker: str,
    rsi_period: int,
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    highs: pd.DataFrame,
    lows: pd.DataFrame,
    opens: pd.DataFrame,
    ohlc_closes: pd.DataFrame,
) -> dict:
    s = prices[ticker].dropna()
    vol_s = (
        volumes[ticker].reindex(s.index)
        if not volumes.empty and ticker in volumes.columns
        else pd.Series(dtype=float)
    )
    ohlc = get_ohlc_for_ticker(opens, highs, lows, ohlc_closes, ticker, ref_index=s.index)
    return {
        "s": s,
        "rsi": compute_rsi(s, rsi_period),
        "macd": compute_macd(s),
        "sma50": compute_sma(s, 50),
        "sma200": compute_sma(s, 200),
        "bollinger": compute_bollinger(s),
        "stochastic": compute_stochastic(s),
        "vol_s": vol_s,
        "ohlc": ohlc,
        "has_volume": vol_s.dropna().size >= 5,
        "has_ohlc": ohlc is not None,
    }


@st.cache_data(ttl=COMPUTE_TTL, show_spinner=False)
def cached_advanced_risk_metrics(ohlcv_sig: str, returns: pd.DataFrame) -> pd.DataFrame:
    return compute_advanced_risk_metrics(returns)


@st.cache_data(ttl=COMPUTE_TTL, show_spinner=False)
def cached_corr_matrix(ohlcv_sig: str, returns: pd.DataFrame) -> pd.DataFrame:
    return compute_corr(returns)


@st.cache_data(ttl=COMPUTE_TTL, show_spinner=False)
def cached_markowitz_weights(ohlcv_sig: str, returns: pd.DataFrame) -> pd.Series:
    weights = compute_markowitz(returns)
    return _align_weights_to_returns(returns, weights)


@st.cache_data(ttl=COMPUTE_TTL, show_spinner=False)
def cached_markowitz_frontier_figure(
    ohlcv_sig: str,
    returns: pd.DataFrame,
    weights: pd.Series,
    risk_free_rate: float,
    n_random: int,
    has_current: bool,
    current_weights: pd.Series | None,
):
    return build_markowitz_frontier_figure(
        returns,
        weights,
        risk_free_rate=risk_free_rate,
        n_random=n_random,
        current_weights=current_weights if has_current else None,
    )


@st.cache_data(ttl=COMPUTE_TTL, show_spinner=False)
def cached_fft_periodicity(
    cache_sig: str,
    min_period: int,
    max_period: int,
    top_n: int,
    trend_mode: str,
    series: pd.Series,
):
    return compute_fft_periodicity(
        series,
        min_period_days=min_period,
        max_period_days=max_period,
        top_n=top_n,
        trend_mode=trend_mode,
    )


@st.cache_data(ttl=COMPUTE_TTL, show_spinner=False)
def cached_fft_summary(
    ohlcv_sig: str,
    min_period: int,
    max_period: int,
    top_n: int,
    trend_mode: str,
    prices: pd.DataFrame,
):
    return compute_fft_summary_for_prices(
        prices,
        min_period_days=min_period,
        max_period_days=max_period,
        top_n=top_n,
        trend_mode=trend_mode,
    )


@st.cache_data(ttl=COMPUTE_TTL, show_spinner=False)
def cached_suggest_portfolio_additions(
    portfolio_sig: str,
    candidate_sig: str,
    returns: pd.DataFrame,
    portfolio_tickers: tuple,
    candidate_returns: pd.DataFrame,
    candidate_weight: float,
    objective_weights_items: tuple,
):
    objective_weights = dict(objective_weights_items)
    return suggest_portfolio_additions(
        returns=returns,
        portfolio_tickers=list(portfolio_tickers),
        candidate_returns=candidate_returns,
        candidate_weight=candidate_weight,
        objective_weights=objective_weights,
    )


def clear_dashboard_compute_cache() -> None:
    cached_usd_to_eur_rate.clear()
    cached_benchmark_prices.clear()
    cached_candidate_market_data.clear()
    cached_technical_indicators.clear()
    cached_technical_detail_bundle.clear()
    cached_advanced_risk_metrics.clear()
    cached_corr_matrix.clear()
    cached_markowitz_weights.clear()
    cached_markowitz_frontier_figure.clear()
    cached_fft_periodicity.clear()
    cached_fft_summary.clear()
    cached_suggest_portfolio_additions.clear()
