"""Cache Streamlit (st.cache_data) pour calculs et appels réseau du dashboard."""
from __future__ import annotations

from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import yfinance as yf

from analytics import (
    _align_weights_to_returns,
    build_markowitz_frontier_figure,
    build_portfolio_price_for_fft,
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
    compute_supertrend,
    compute_technical_indicators,
    suggest_portfolio_additions,
)
from data_loader import get_ohlc_for_ticker, load_ohlcv_in_batches, load_prices_in_batches

_EUR_SUFFIXES = (".PA", ".DE", ".BR", ".AS", ".MI", ".MC", ".SW", ".L", ".IR", ".HE")
_USD_TICKERS = {"HMY", "XPL", "NUTX", "NYXH", "AAPL", "MSFT"}


def _is_usd_ticker(ticker: str) -> bool:
    tk = str(ticker).upper()
    if any(tk.endswith(suffix) for suffix in _EUR_SUFFIXES):
        return False
    if tk.endswith(".US"):
        return True
    if tk in _USD_TICKERS:
        return True
    if "." not in tk:
        return True
    return False

COMPUTE_TTL = 3600
FX_TTL = 1800
MARKET_TTL = 3600


@st.cache_data(ttl=FX_TTL, show_spinner=False)
def cached_usd_to_eur_rate(default: float = 0.92) -> float:
    """Taux spot USD→EUR via la série FX cachée (évite un 2e téléchargement Yahoo)."""
    short_start = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
    series = cached_usd_to_eur_series(short_start)
    if series is not None and not series.empty:
        val = float(series.iloc[-1])
        if val > 0:
            return val
    return default


def _parse_fx_close_series(fx_data: pd.DataFrame) -> pd.Series:
    """EURUSD=X → série USD→EUR (1 EURUSD)."""
    if fx_data is None or fx_data.empty:
        return pd.Series(dtype=float)
    close = fx_data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    rate = (1.0 / pd.to_numeric(close, errors="coerce")).dropna()
    rate = rate.sort_index()
    rate.index = pd.to_datetime(rate.index)
    if getattr(rate.index, "tz", None) is not None:
        rate.index = rate.index.tz_localize(None)
    rate.name = "USD_EUR"
    return rate


@st.cache_data(ttl=FX_TTL, show_spinner=False)
def cached_usd_to_eur_series(start_date_str: str) -> pd.Series:
    """Taux de change USD→EUR historique (1 / EURUSD), alignable sur les cours."""
    try:
        fx_data = yf.download(
            "EURUSD=X",
            start=start_date_str,
            progress=False,
            threads=False,
            auto_adjust=True,
        )
        return _parse_fx_close_series(fx_data)
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=MARKET_TTL, show_spinner=False)
def cached_benchmark_prices(bench_ticker: str, start_date_str: str) -> pd.DataFrame:
    prices = load_prices_in_batches(
        [bench_ticker], start=start_date_str, refresh="cache_only"
    )
    if prices.empty:
        prices = load_prices_in_batches([bench_ticker], start=start_date_str, refresh="full")
    return prices


@st.cache_data(ttl=COMPUTE_TTL, show_spinner=False)
def cached_compute_returns(
    ohlcv_sig: str,
    tickers_tuple: tuple,
    start_date_str: str,
) -> pd.DataFrame:
    """Rendements journaliers — clé légère (ohlcv_sig), rechargement depuis cache disque."""
    if not tickers_tuple:
        return pd.DataFrame()
    prices, _, _, _, _, _ = load_ohlcv_in_batches(
        list(tickers_tuple), start=start_date_str, refresh="cache_only"
    )
    if prices.empty:
        return pd.DataFrame()
    return compute_returns(prices)


@st.cache_data(ttl=COMPUTE_TTL, show_spinner=False)
def _cached_ohlcv_bundle(
    ohlcv_sig: str,
    tickers_tuple: tuple,
    start_date_str: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """OHLCV depuis cache disque — clé légère sans hash de DataFrames."""
    if not tickers_tuple:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty, empty
    return load_ohlcv_in_batches(
        list(tickers_tuple), start=start_date_str, refresh="cache_only"
    )


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
    tickers_tuple: tuple,
    start_date_str: str,
    supertrend_period: int = 10,
    supertrend_multiplier: float = 3.0,
) -> pd.DataFrame:
    prices, volumes, highs, lows, _, _ = _cached_ohlcv_bundle(
        ohlcv_sig, tickers_tuple, start_date_str
    )
    if prices.empty:
        return pd.DataFrame()
    return compute_technical_indicators(
        prices,
        rsi_period=rsi_period,
        volumes=volumes,
        highs=highs,
        lows=lows,
        supertrend_period=supertrend_period,
        supertrend_multiplier=supertrend_multiplier,
    )


@st.cache_data(ttl=COMPUTE_TTL, show_spinner=False)
def cached_technical_detail_bundle(
    ohlcv_sig: str,
    ticker: str,
    rsi_period: int,
    tickers_tuple: tuple,
    start_date_str: str,
    supertrend_period: int = 10,
    supertrend_multiplier: float = 3.0,
) -> dict:
    prices, volumes, highs, lows, opens, ohlc_closes = _cached_ohlcv_bundle(
        ohlcv_sig, tickers_tuple, start_date_str
    )
    if ticker not in prices.columns:
        return {}
    s = prices[ticker].dropna()
    vol_s = (
        volumes[ticker].reindex(s.index)
        if not volumes.empty and ticker in volumes.columns
        else pd.Series(dtype=float)
    )
    ohlc = get_ohlc_for_ticker(opens, highs, lows, ohlc_closes, ticker, ref_index=s.index)
    h_s = s
    l_s = s
    if not highs.empty and ticker in highs.columns:
        h_s = highs[ticker].reindex(s.index).fillna(s)
    if not lows.empty and ticker in lows.columns:
        l_s = lows[ticker].reindex(s.index).fillna(s)
    st_line, st_dir = compute_supertrend(
        h_s,
        l_s,
        s,
        period=supertrend_period,
        multiplier=supertrend_multiplier,
    )
    return {
        "s": s,
        "rsi": compute_rsi(s, rsi_period),
        "macd": compute_macd(s),
        "sma50": compute_sma(s, 50),
        "sma200": compute_sma(s, 200),
        "bollinger": compute_bollinger(s),
        "stochastic": compute_stochastic(s),
        "supertrend": st_line,
        "supertrend_direction": st_dir,
        "vol_s": vol_s,
        "ohlc": ohlc,
        "has_volume": vol_s.dropna().size >= 5,
        "has_ohlc": ohlc is not None,
    }


@st.cache_data(ttl=COMPUTE_TTL, show_spinner=False)
def cached_advanced_risk_metrics(
    ohlcv_sig: str,
    tickers_tuple: tuple,
    start_date_str: str,
) -> pd.DataFrame:
    returns = cached_compute_returns(ohlcv_sig, tickers_tuple, start_date_str)
    if returns.empty:
        return pd.DataFrame()
    return compute_advanced_risk_metrics(returns)


@st.cache_data(ttl=COMPUTE_TTL, show_spinner=False)
def cached_corr_matrix(
    ohlcv_sig: str,
    tickers_tuple: tuple,
    start_date_str: str,
) -> pd.DataFrame:
    returns = cached_compute_returns(ohlcv_sig, tickers_tuple, start_date_str)
    if returns.empty:
        return pd.DataFrame()
    return compute_corr(returns)


@st.cache_data(ttl=COMPUTE_TTL, show_spinner=False)
def cached_markowitz_weights(
    ohlcv_sig: str,
    tickers_tuple: tuple,
    start_date_str: str,
) -> pd.Series:
    returns = cached_compute_returns(ohlcv_sig, tickers_tuple, start_date_str)
    if returns.empty:
        return pd.Series(dtype=float)
    weights = compute_markowitz(returns)
    return _align_weights_to_returns(returns, weights)


@st.cache_data(ttl=COMPUTE_TTL, show_spinner=False)
def cached_markowitz_frontier_figure(
    ohlcv_sig: str,
    tickers_tuple: tuple,
    start_date_str: str,
    risk_free_rate: float,
    n_random: int,
    has_current: bool,
    current_weights_items: tuple | None,
):
    returns = cached_compute_returns(ohlcv_sig, tickers_tuple, start_date_str)
    if returns.empty:
        return None
    weights = cached_markowitz_weights(ohlcv_sig, tickers_tuple, start_date_str)
    current_weights = None
    if has_current and current_weights_items:
        current_weights = _align_weights_to_returns(
            returns, pd.Series(dict(current_weights_items))
        )
    return build_markowitz_frontier_figure(
        returns,
        weights,
        risk_free_rate=risk_free_rate,
        n_random=n_random,
        current_weights=current_weights,
    )


@st.cache_data(ttl=COMPUTE_TTL, show_spinner=False)
def cached_fft_periodicity_ticker(
    ohlcv_sig: str,
    tickers_tuple: tuple,
    start_date_str: str,
    ticker: str,
    min_period: int,
    max_period: int,
    top_n: int,
    trend_mode: str,
):
    prices, _, _, _, _, _ = _cached_ohlcv_bundle(ohlcv_sig, tickers_tuple, start_date_str)
    if prices.empty or ticker not in prices.columns:
        return None
    return compute_fft_periodicity(
        prices[ticker],
        min_period_days=min_period,
        max_period_days=max_period,
        top_n=top_n,
        trend_mode=trend_mode,
    )


@st.cache_data(ttl=COMPUTE_TTL, show_spinner=False)
def cached_fft_periodicity_portfolio(
    ohlcv_sig: str,
    tickers_tuple: tuple,
    start_date_str: str,
    quantities_items: tuple,
    use_usd_conversion: bool,
    min_period: int,
    max_period: int,
    top_n: int,
    trend_mode: str,
):
    prices, _, _, _, _, _ = _cached_ohlcv_bundle(ohlcv_sig, tickers_tuple, start_date_str)
    if prices.empty:
        return None
    portfolio_tickers = [t for t in tickers_tuple if t in prices.columns]
    quantities = dict(quantities_items) if quantities_items else None
    usd_to_eur = cached_usd_to_eur_rate() if use_usd_conversion else 1.0
    portfolio_series = build_portfolio_price_for_fft(
        prices,
        tickers=portfolio_tickers,
        quantities=quantities,
        usd_to_eur=usd_to_eur,
        is_usd_fn=_is_usd_ticker,
    )
    if portfolio_series is None or len(portfolio_series.dropna()) < 40:
        return None
    return compute_fft_periodicity(
        portfolio_series,
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
    tickers_tuple: tuple,
    start_date_str: str,
):
    prices, _, _, _, _, _ = _cached_ohlcv_bundle(ohlcv_sig, tickers_tuple, start_date_str)
    if prices.empty:
        return pd.DataFrame()
    return compute_fft_summary_for_prices(
        prices,
        min_period_days=min_period,
        max_period_days=max_period,
        top_n=top_n,
        trend_mode=trend_mode,
    )


@st.cache_data(ttl=COMPUTE_TTL, show_spinner=False)
def cached_candidate_technical_indicators(
    candidate_sig: str,
    tickers_tuple: tuple,
    start_date_str: str,
    rsi_period: int = 14,
) -> pd.DataFrame:
    """Indicateurs techniques pour les tickers candidats (clé légère)."""
    prices, volumes, highs, lows, _, _ = _cached_ohlcv_bundle(
        candidate_sig, tickers_tuple, start_date_str
    )
    tickers = [t for t in tickers_tuple if t in prices.columns]
    if not tickers:
        return pd.DataFrame()
    return compute_technical_indicators(
        prices[tickers],
        rsi_period=rsi_period,
        volumes=volumes,
        highs=highs,
        lows=lows,
    )


@st.cache_data(ttl=COMPUTE_TTL, show_spinner=False)
def cached_suggest_portfolio_additions(
    portfolio_sig: str,
    candidate_sig: str,
    portfolio_tickers: tuple,
    candidate_pool: tuple,
    start_date_str: str,
    candidate_weight: float,
    objective_weights_items: tuple,
):
    objective_weights = dict(objective_weights_items)
    port_tickers = tuple(sorted(portfolio_tickers))
    returns = cached_compute_returns(portfolio_sig, port_tickers, start_date_str)
    if returns.empty:
        return pd.DataFrame(), None, None
    _, candidate_returns = cached_candidate_market_data(
        candidate_sig, candidate_pool, start_date_str
    )
    if candidate_returns.empty:
        return pd.DataFrame(), None, None
    return suggest_portfolio_additions(
        returns=returns,
        portfolio_tickers=list(portfolio_tickers),
        candidate_returns=candidate_returns,
        candidate_weight=candidate_weight,
        objective_weights=objective_weights,
    )


def clear_dashboard_compute_cache() -> None:
    cached_usd_to_eur_rate.clear()
    cached_usd_to_eur_series.clear()
    cached_compute_returns.clear()
    _cached_ohlcv_bundle.clear()
    cached_benchmark_prices.clear()
    cached_candidate_market_data.clear()
    cached_technical_indicators.clear()
    cached_technical_detail_bundle.clear()
    cached_advanced_risk_metrics.clear()
    cached_corr_matrix.clear()
    cached_markowitz_weights.clear()
    cached_markowitz_frontier_figure.clear()
    cached_fft_periodicity_ticker.clear()
    cached_fft_periodicity_portfolio.clear()
    cached_fft_summary.clear()
    cached_candidate_technical_indicators.clear()
    cached_suggest_portfolio_additions.clear()
