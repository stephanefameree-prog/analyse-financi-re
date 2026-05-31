import hashlib
import pickle
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

MARKET_CACHE_DIR = "market_cache"
INCREMENTAL_LOOKBACK_DAYS = 10
FULL_CACHE_MAX_AGE_HOURS = 24
DEFAULT_BATCH_SIZE = 100


def filter_valid_tickers(tickers, start_date):
    try:
        start_str = pd.to_datetime(start_date).strftime("%Y-%m-%d")
        end_str = (pd.to_datetime(start_date) + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        ticker_test = tickers[0]
        df = yf.download(ticker_test, start=start_str, end=end_str, progress=False)
        return tickers
    except Exception:
        return tickers


def _market_cache_dir():
    path = Path(__file__).resolve().parent / MARKET_CACHE_DIR
    path.mkdir(exist_ok=True)
    return path


def _tickers_signature(tickers, start):
    normalized = "|".join(sorted({str(t).strip() for t in tickers if str(t).strip()}))
    start_str = pd.to_datetime(start).strftime("%Y-%m-%d")
    digest = hashlib.md5(f"{normalized}|{start_str}".encode("utf-8")).hexdigest()
    return digest[:16]


def _market_cache_path(signature):
    return _market_cache_dir() / f"ohlcv_{signature}.pkl"


def _load_market_cache(signature):
    path = _market_cache_path(signature)
    if not path.is_file():
        return None
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict) and "prices" in payload:
            return payload
    except Exception:
        pass
    return None


def _save_market_cache(signature, bundle):
    path = _market_cache_path(signature)
    bundle = dict(bundle)
    bundle["updated_at"] = datetime.now().isoformat(timespec="seconds")
    with open(path, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)


def clear_market_cache(signature=None):
    cache_dir = _market_cache_dir()
    if signature is None:
        for file in cache_dir.glob("ohlcv_*.pkl"):
            file.unlink(missing_ok=True)
        return
    path = _market_cache_path(signature)
    if path.is_file():
        path.unlink()


def _merge_market_frames(df_old, df_new):
    if df_old is None or df_old.empty:
        return df_new.copy() if df_new is not None else pd.DataFrame()
    if df_new is None or df_new.empty:
        return df_old.copy()
    cols = sorted(set(df_old.columns) | set(df_new.columns))
    out = pd.DataFrame()
    for col in cols:
        parts = []
        if col in df_old.columns:
            parts.append(df_old[col])
        if col in df_new.columns:
            parts.append(df_new[col])
        merged = pd.concat(parts).sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
        out[col] = merged
    return out.ffill().bfill()


def _merge_market_bundle(base, recent):
    if not base:
        return recent
    if not recent:
        return base
    return {
        "prices": _merge_market_frames(base.get("prices"), recent.get("prices")),
        "volumes": _merge_market_frames(base.get("volumes"), recent.get("volumes")),
        "highs": _merge_market_frames(base.get("highs"), recent.get("highs")),
        "lows": _merge_market_frames(base.get("lows"), recent.get("lows")),
        "opens": _merge_market_frames(base.get("opens"), recent.get("opens")),
        "ohlc_closes": _merge_market_frames(base.get("ohlc_closes"), recent.get("ohlc_closes")),
        "updated_at": recent.get("updated_at") or base.get("updated_at"),
    }


def _cache_is_fresh(cache_payload, max_age_hours=FULL_CACHE_MAX_AGE_HOURS):
    updated_at = cache_payload.get("updated_at")
    if not updated_at:
        return False
    try:
        ts = datetime.fromisoformat(updated_at)
        return (datetime.now() - ts).total_seconds() < max_age_hours * 3600
    except Exception:
        return False


def _bundle_to_tuple(bundle):
    if not bundle:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty, empty
    return (
        bundle.get("prices", pd.DataFrame()),
        bundle.get("volumes", pd.DataFrame()),
        bundle.get("highs", pd.DataFrame()),
        bundle.get("lows", pd.DataFrame()),
        bundle.get("opens", pd.DataFrame()),
        bundle.get("ohlc_closes", pd.DataFrame()),
    )


def _fetch_ohlcv_from_yahoo(tickers, start, end=None, batch_size=DEFAULT_BATCH_SIZE):
    """Télécharge OHLCV Yahoo pour une liste de tickers."""
    if not tickers:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty, empty

    price_data = []
    volume_data = []
    high_data = []
    low_data = []
    open_data = []
    ohlc_close_data = []
    failed = []
    start_str = pd.to_datetime(start).strftime("%Y-%m-%d")
    end_str = pd.to_datetime(end).strftime("%Y-%m-%d") if end is not None else None

    def _append_ticker(ticker, close_s, vol_s, high_s, low_s, open_s, ohlc_close_s):
        if close_s is None or close_s.dropna().empty:
            return
        price_data.append(pd.DataFrame({ticker: close_s}))
        if vol_s is not None and not vol_s.dropna().empty:
            volume_data.append(pd.DataFrame({ticker: vol_s}))
        if high_s is not None and not high_s.dropna().empty:
            high_data.append(pd.DataFrame({ticker: high_s}))
        if low_s is not None and not low_s.dropna().empty:
            low_data.append(pd.DataFrame({ticker: low_s}))
        if open_s is not None and not open_s.dropna().empty:
            open_data.append(pd.DataFrame({ticker: open_s}))
        if ohlc_close_s is not None and not ohlc_close_s.dropna().empty:
            ohlc_close_data.append(pd.DataFrame({ticker: ohlc_close_s}))

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        success = False
        attempts = 0
        data = pd.DataFrame()

        while not success and attempts < 3:
            try:
                kwargs = {
                    "start": start_str,
                    "group_by": "ticker",
                    "progress": False,
                    "threads": False,
                }
                if end_str:
                    kwargs["end"] = end_str
                data = yf.download(batch, **kwargs)
                if not data.empty:
                    success = True
                else:
                    attempts += 1
            except Exception:
                attempts += 1

        if data.empty:
            failed.extend(batch)
            continue

        for ticker in batch:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    if ticker not in data.columns.get_level_values(0):
                        continue
                    ticker_df = data[ticker]
                    close_s = (
                        ticker_df["Adj Close"]
                        if "Adj Close" in ticker_df.columns
                        else ticker_df.get("Close")
                    )
                    _append_ticker(
                        ticker,
                        close_s,
                        ticker_df.get("Volume"),
                        ticker_df.get("High"),
                        ticker_df.get("Low"),
                        ticker_df.get("Open"),
                        ticker_df.get("Close"),
                    )
                else:
                    close_col = "Adj Close" if "Adj Close" in data.columns else "Close"
                    if close_col not in data.columns:
                        continue
                    _append_ticker(
                        batch[0] if len(batch) == 1 else ticker,
                        data[close_col],
                        data.get("Volume"),
                        data.get("High"),
                        data.get("Low"),
                        data.get("Open"),
                        data.get("Close"),
                    )
            except Exception:
                continue

    if failed:
        st.warning(f"{len(failed)} tickers ignorés (Pas de données sur Yahoo) : {failed}")

    def _concat_frames(frames):
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, axis=1)
        out = out.loc[:, ~out.columns.duplicated()]
        return out.ffill().bfill()

    bundle = {
        "prices": _concat_frames(price_data),
        "volumes": _concat_frames(volume_data),
        "highs": _concat_frames(high_data),
        "lows": _concat_frames(low_data),
        "opens": _concat_frames(open_data),
        "ohlc_closes": _concat_frames(ohlc_close_data),
    }
    return _bundle_to_tuple(bundle)


def load_ohlcv_smart(tickers, start, batch_size=DEFAULT_BATCH_SIZE, refresh="auto"):
    """
    Charge OHLCV avec cache disque et mise à jour incrémentale.

    refresh:
      - full         : retélécharge tout l'historique
      - incremental  : met à jour les ~10 derniers jours seulement
      - cache_only   : lit le disque sans appeler Yahoo
      - auto         : cache disque frais sinon full, puis incremental si cache existe
    """
    tickers = [str(t).strip() for t in tickers if str(t).strip()]
    if not tickers:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty, empty

    signature = _tickers_signature(tickers, start)
    cached = _load_market_cache(signature)
    start_dt = pd.to_datetime(start)

    if refresh == "cache_only":
        return _bundle_to_tuple(cached)

    if refresh == "auto":
        if cached and _cache_is_fresh(cached):
            return _bundle_to_tuple(cached)
        refresh = "incremental" if cached else "full"

    if refresh == "incremental" and cached:
        inc_start = max(start_dt, pd.Timestamp.now() - pd.Timedelta(days=INCREMENTAL_LOOKBACK_DAYS))
        recent_tuple = _fetch_ohlcv_from_yahoo(
            tickers,
            inc_start,
            end=None,
            batch_size=batch_size,
        )
        recent = {
            "prices": recent_tuple[0],
            "volumes": recent_tuple[1],
            "highs": recent_tuple[2],
            "lows": recent_tuple[3],
            "opens": recent_tuple[4],
            "ohlc_closes": recent_tuple[5],
        }
        merged = _merge_market_bundle(cached, recent)
        _save_market_cache(signature, merged)
        return _bundle_to_tuple(merged)

    if refresh == "incremental" and not cached:
        refresh = "full"

    full_tuple = _fetch_ohlcv_from_yahoo(tickers, start_dt, end=None, batch_size=batch_size)
    bundle = {
        "prices": full_tuple[0],
        "volumes": full_tuple[1],
        "highs": full_tuple[2],
        "lows": full_tuple[3],
        "opens": full_tuple[4],
        "ohlc_closes": full_tuple[5],
    }
    if cached:
        bundle = _merge_market_bundle(cached, bundle)
    _save_market_cache(signature, bundle)
    return _bundle_to_tuple(bundle)


def load_ohlcv_in_batches(tickers, start, batch_size=DEFAULT_BATCH_SIZE, refresh="auto"):
    """Charge OHLCV (cache disque + mise à jour incrémentale)."""
    return load_ohlcv_smart(tickers, start, batch_size=batch_size, refresh=refresh)


def get_ohlc_for_ticker(opens, highs, lows, ohlc_closes, ticker, ref_index=None):
    """OHLC aligné pour chandeliers japonais (Open/High/Low/Close Yahoo)."""
    needed = (opens, highs, lows, ohlc_closes)
    if any(df.empty or ticker not in df.columns for df in needed):
        return None
    ohlc = pd.concat(
        [
            opens[ticker].rename("open"),
            highs[ticker].rename("high"),
            lows[ticker].rename("low"),
            ohlc_closes[ticker].rename("close"),
        ],
        axis=1,
    )
    if ref_index is not None:
        ohlc = ohlc.reindex(ref_index)
    ohlc = ohlc.dropna()
    if len(ohlc) < 2:
        return None
    return ohlc


CANDLE_UP = "#26a69a"
CANDLE_DOWN = "#ef5350"


def candlestick_trace(ohlc_df, name="Chandeliers"):
    """Trace Plotly chandelier japonais."""
    return go.Candlestick(
        x=ohlc_df.index,
        open=ohlc_df["open"],
        high=ohlc_df["high"],
        low=ohlc_df["low"],
        close=ohlc_df["close"],
        name=name,
        increasing_line_color=CANDLE_UP,
        decreasing_line_color=CANDLE_DOWN,
        increasing_fillcolor=CANDLE_UP,
        decreasing_fillcolor=CANDLE_DOWN,
    )


def load_prices_in_batches(tickers, start, batch_size=DEFAULT_BATCH_SIZE, refresh="auto"):
    prices, _, _, _, _, _ = load_ohlcv_smart(
        tickers, start, batch_size=batch_size, refresh=refresh
    )
    return prices


@st.cache_data(ttl=86400, show_spinner=False)
def get_ticker_names(tickers):
    """Récupère le nom court Yahoo par lots (sans appels lents ticker par ticker)."""
    tickers = tuple(sorted({str(t).strip() for t in tickers if str(t).strip()}))
    names = {t: t for t in tickers}
    if not tickers:
        return names

    try:
        from yahooquery import Ticker

        chunk_size = 100
        for i in range(0, len(tickers), chunk_size):
            chunk = list(tickers[i : i + chunk_size])
            payload = Ticker(chunk, timeout=25).price
            if isinstance(payload, dict):
                for tk in chunk:
                    block = payload.get(tk)
                    if isinstance(block, dict):
                        label = block.get("shortName") or block.get("longName")
                        if label:
                            names[tk] = str(label)
    except Exception:
        pass

    return names


def ticker_label(ticker, names, show_names=True):
    tk = str(ticker).strip()
    if not show_names:
        return tk
    name = names.get(tk, tk)
    if not name or name == tk:
        return tk
    return f"{tk} — {name}"


def add_company_names(df, names, ticker_col="Ticker", show_names=True):
    """Ajoute une colonne « Nom » après le ticker."""
    if not show_names or ticker_col not in df.columns:
        return df
    out = df.copy()
    out.insert(
        1,
        "Nom",
        out[ticker_col].map(lambda t: names.get(str(t).strip(), str(t).strip())),
    )
    return out


def label_index_with_names(df, names, show_names=True):
    """Renomme l'index (tickers) pour affichage avec nom d'entreprise."""
    if not show_names:
        return df
    out = df.copy()
    out.index = [ticker_label(t, names, True) for t in out.index]
    return out
