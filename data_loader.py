import hashlib
import os
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
TICKER_CACHE_SUBDIR = "tickers"
INCREMENTAL_LOOKBACK_DAYS = 10
FULL_CACHE_MAX_AGE_HOURS = 24
DEFAULT_BATCH_SIZE = 100
OHLCV_FRAME_KEYS = ("prices", "volumes", "highs", "lows", "opens", "ohlc_closes")


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
    tmp = path.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def _ticker_cache_dir():
    path = _market_cache_dir() / TICKER_CACHE_SUBDIR
    path.mkdir(exist_ok=True)
    return path


def _ticker_cache_path(ticker, start):
    start_str = pd.to_datetime(start).strftime("%Y-%m-%d")
    digest = hashlib.md5(f"{str(ticker).strip()}|{start_str}".encode("utf-8")).hexdigest()[:16]
    return _ticker_cache_dir() / f"{digest}.pkl"


def _load_ticker_cache(ticker, start):
    path = _ticker_cache_path(ticker, start)
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


def _save_ticker_cache(ticker, start, payload):
    path = _ticker_cache_path(ticker, start)
    payload = dict(payload)
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    tmp = path.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def _merge_series(old_s, new_s):
    if old_s is None or (isinstance(old_s, pd.Series) and old_s.dropna().empty):
        return new_s.copy() if new_s is not None else pd.Series(dtype=float)
    if new_s is None or (isinstance(new_s, pd.Series) and new_s.dropna().empty):
        return old_s.copy()
    merged = pd.concat([old_s, new_s]).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    return merged


def _extract_ticker_payload(bundle, ticker):
    payload = {}
    for key in OHLCV_FRAME_KEYS:
        df = bundle.get(key, pd.DataFrame())
        if isinstance(df, pd.DataFrame) and ticker in df.columns:
            payload[key] = df[ticker].copy()
        else:
            payload[key] = pd.Series(dtype=float, name=ticker)
    return payload


def _merge_ticker_payload(base, recent):
    if not base:
        return recent
    if not recent:
        return base
    out = {}
    for key in OHLCV_FRAME_KEYS:
        out[key] = _merge_series(base.get(key), recent.get(key))
    out["updated_at"] = recent.get("updated_at") or base.get("updated_at")
    return out


def _concat_frame_parts(frames):
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    return out.ffill().bfill()


def _assemble_bundle_from_ticker_caches(tickers, start):
    frame_parts = {key: [] for key in OHLCV_FRAME_KEYS}
    latest_update = None
    for ticker in tickers:
        tc = _load_ticker_cache(ticker, start)
        if not tc:
            continue
        updated_at = tc.get("updated_at")
        if updated_at and (latest_update is None or updated_at > latest_update):
            latest_update = updated_at
        for key in OHLCV_FRAME_KEYS:
            series = tc.get(key)
            if series is not None and not pd.Series(series).dropna().empty:
                frame_parts[key].append(pd.DataFrame({ticker: series}))
    bundle = {key: _concat_frame_parts(parts) for key, parts in frame_parts.items()}
    bundle["updated_at"] = latest_update or datetime.now().isoformat(timespec="seconds")
    return bundle


def _tickers_missing_from_bundle(bundle, tickers):
    prices = bundle.get("prices", pd.DataFrame())
    missing = []
    for ticker in tickers:
        if ticker not in prices.columns:
            missing.append(ticker)
            continue
        if prices[ticker].dropna().empty:
            missing.append(ticker)
    return missing


def _tuple_to_bundle(t_tuple):
    return {
        "prices": t_tuple[0],
        "volumes": t_tuple[1],
        "highs": t_tuple[2],
        "lows": t_tuple[3],
        "opens": t_tuple[4],
        "ohlc_closes": t_tuple[5],
    }


def _persist_ticker_caches_from_bundle(bundle, tickers, start):
    prices = bundle.get("prices", pd.DataFrame())
    for ticker in tickers:
        if ticker not in prices.columns:
            continue
        _save_ticker_cache(ticker, start, _extract_ticker_payload(bundle, ticker))


def _maybe_migrate_legacy_cache(tickers, start):
    """Découpe un cache legacy (signature portefeuille) en caches par ticker."""
    missing = [t for t in tickers if not _load_ticker_cache(t, start)]
    if not missing:
        return
    legacy = _load_market_cache(_tickers_signature(tickers, start))
    if not legacy:
        return
    for ticker in missing:
        if ticker in legacy.get("prices", pd.DataFrame()).columns:
            payload = _extract_ticker_payload(legacy, ticker)
            payload["updated_at"] = legacy.get("updated_at")
            _save_ticker_cache(ticker, start, payload)


def _apply_yahoo_updates(tickers, start, start_dt, batch_size, mode):
    """
    mode: 'full' | 'incremental'
    Met à jour les caches par ticker et retourne le bundle assemblé.
    """
    tickers = [str(t).strip() for t in tickers if str(t).strip()]
    if not tickers:
        return _assemble_bundle_from_ticker_caches([], start)

    caches = {t: _load_ticker_cache(t, start) for t in tickers}

    if mode == "incremental":
        to_full = [t for t in tickers if not caches.get(t)]
        to_inc = [t for t in tickers if caches.get(t)]
    else:
        to_full = list(tickers)
        to_inc = []

    if to_full:
        full_tuple = _fetch_ohlcv_from_yahoo(
            to_full, start_dt, end=None, batch_size=batch_size
        )
        full_bundle = _tuple_to_bundle(full_tuple)
        for ticker in to_full:
            if ticker in full_bundle.get("prices", pd.DataFrame()).columns:
                existing = caches.get(ticker)
                if existing and mode == "incremental":
                    merged = _merge_ticker_payload(
                        existing, _extract_ticker_payload(full_bundle, ticker)
                    )
                    _save_ticker_cache(ticker, start, merged)
                else:
                    _save_ticker_cache(
                        ticker, start, _extract_ticker_payload(full_bundle, ticker)
                    )

    if to_inc:
        inc_start = max(
            start_dt, pd.Timestamp.now() - pd.Timedelta(days=INCREMENTAL_LOOKBACK_DAYS)
        )
        recent_tuple = _fetch_ohlcv_from_yahoo(
            to_inc, inc_start, end=None, batch_size=batch_size
        )
        recent_bundle = _tuple_to_bundle(recent_tuple)
        for ticker in to_inc:
            if ticker not in recent_bundle.get("prices", pd.DataFrame()).columns:
                continue
            merged = _merge_ticker_payload(
                caches[ticker], _extract_ticker_payload(recent_bundle, ticker)
            )
            _save_ticker_cache(ticker, start, merged)

    return _assemble_bundle_from_ticker_caches(tickers, start)


def clear_market_cache(signature=None, tickers=None, start=None):
    """Vide le cache cours. Avec signature, supprime aussi les fichiers par ticker."""
    cache_dir = _market_cache_dir()
    if signature is None:
        for file in cache_dir.glob("ohlcv_*.pkl"):
            file.unlink(missing_ok=True)
        ticker_dir = cache_dir / TICKER_CACHE_SUBDIR
        if ticker_dir.is_dir():
            for file in ticker_dir.glob("*.pkl"):
                file.unlink(missing_ok=True)
        return
    path = _market_cache_path(signature)
    path.unlink(missing_ok=True)
    if tickers and start is not None:
        for ticker in tickers:
            tp = _ticker_cache_path(ticker, start)
            tp.unlink(missing_ok=True)


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
    Charge OHLCV avec cache disque par ticker et mise à jour incrémentale.

    refresh:
      - full         : retélécharge tout l'historique (par ticker)
      - incremental  : met à jour les ~10 derniers jours seulement
      - cache_only   : lit le disque sans appeler Yahoo
      - auto         : cache frais par ticker sinon fetch ciblé (nouveaux tickers seulement)
    """
    tickers = [str(t).strip() for t in tickers if str(t).strip()]
    if not tickers:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty, empty

    start_dt = pd.to_datetime(start)
    _maybe_migrate_legacy_cache(tickers, start)

    if refresh == "cache_only":
        bundle = _assemble_bundle_from_ticker_caches(tickers, start)
        if not _tickers_missing_from_bundle(bundle, tickers):
            return _bundle_to_tuple(bundle)
        legacy = _load_market_cache(_tickers_signature(tickers, start))
        if legacy:
            return _bundle_to_tuple(legacy)
        return _bundle_to_tuple(bundle)

    if refresh == "auto":
        caches = {t: _load_ticker_cache(t, start) for t in tickers}
        if all(caches[t] and _cache_is_fresh(caches[t]) for t in tickers):
            bundle = _assemble_bundle_from_ticker_caches(tickers, start)
            return _bundle_to_tuple(bundle)
        to_full = [t for t in tickers if not caches.get(t)]
        to_inc = [
            t for t in tickers if caches.get(t) and not _cache_is_fresh(caches[t])
        ]
        if to_full:
            _apply_yahoo_updates(to_full, start, start_dt, batch_size, mode="full")
        if to_inc:
            _apply_yahoo_updates(to_inc, start, start_dt, batch_size, mode="incremental")
        bundle = _assemble_bundle_from_ticker_caches(tickers, start)
        return _bundle_to_tuple(bundle)

    if refresh == "incremental":
        bundle = _apply_yahoo_updates(
            tickers, start, start_dt, batch_size, mode="incremental"
        )
        return _bundle_to_tuple(bundle)

    # full
    bundle = _apply_yahoo_updates(tickers, start, start_dt, batch_size, mode="full")
    signature = _tickers_signature(tickers, start)
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
def get_ticker_metadata(tickers, need_names: bool = True, need_sectors: bool = True):
    """Noms et/ou secteurs Yahoo en une passe (price + asset_profile par lot)."""
    tickers = tuple(sorted({str(t).strip() for t in tickers if str(t).strip()}))
    names: dict[str, str] = {}
    sectors: dict[str, str] = {}
    if not tickers:
        return names, sectors
    if not need_names and not need_sectors:
        return names, sectors

    if need_names:
        names = {t: t for t in tickers}

    try:
        from yahooquery import Ticker

        chunk_size = 100
        for i in range(0, len(tickers), chunk_size):
            chunk = list(tickers[i : i + chunk_size])
            yq = Ticker(chunk, timeout=25)
            if need_names:
                payload = yq.price
                if isinstance(payload, dict):
                    for tk in chunk:
                        block = payload.get(tk)
                        if isinstance(block, dict):
                            label = block.get("shortName") or block.get("longName")
                            if label:
                                names[tk] = str(label)
            if need_sectors:
                payload = yq.asset_profile
                if isinstance(payload, dict):
                    for tk in chunk:
                        block = payload.get(tk)
                        if isinstance(block, dict):
                            sector = block.get("sector")
                            if sector:
                                sectors[tk] = str(sector)
    except Exception:
        pass

    if need_names:
        missing = [t for t in tickers if names.get(t) == t]
        if missing:
            _fill_ticker_names_from_fmp(names, missing)

    return names, sectors


def get_ticker_names(tickers):
    """Récupère le nom court Yahoo par lots (sans appels lents ticker par ticker)."""
    return get_ticker_metadata(tickers, need_names=True, need_sectors=False)[0]


def get_ticker_sectors(tickers):
    """Secteur Yahoo Finance par lots (asset_profile)."""
    return get_ticker_metadata(tickers, need_names=False, need_sectors=True)[1]


def get_fmp_api_key() -> str:
    """Clé FMP via variable d'environnement FMP_API_KEY ou st.secrets."""
    key = os.environ.get("FMP_API_KEY", "").strip()
    if key:
        return key
    try:
        return str(st.secrets.get("FMP_API_KEY", "")).strip()
    except Exception:
        return ""


def _fill_ticker_names_from_fmp(names: dict[str, str], tickers: list[str]) -> None:
    api_key = get_fmp_api_key()
    if not api_key:
        return

    try:
        import requests
    except ImportError:
        return

    chunk_size = 20
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        try:
            response = requests.get(
                f"{FMP_PROFILE_URL}/{','.join(chunk)}",
                params={"apikey": api_key},
                timeout=12,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception:
            continue

        rows = payload if isinstance(payload, list) else [payload]
        for item in rows:
            if not isinstance(item, dict):
                continue
            symbol = str(item.get("symbol") or "").strip()
            label = item.get("companyName") or item.get("name")
            if symbol and label and symbol in names:
                names[symbol] = str(label).strip()


YAHOO_SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"
YAHOO_SEARCH_TYPES = frozenset({"EQUITY", "ETF"})
FMP_SEARCH_URL = "https://financialmodelingprep.com/api/v3/search"
FMP_PROFILE_URL = "https://financialmodelingprep.com/api/v3/profile"


def parse_yahoo_search_quotes(quotes, max_results=8):
    """Extrait symboles action/ETF d'une réponse Yahoo search."""
    results: list[dict[str, str]] = []
    if not quotes:
        return results
    for item in quotes:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol") or "").strip()
        if not symbol:
            continue
        quote_type = str(item.get("quoteType") or "").upper()
        if quote_type and quote_type not in YAHOO_SEARCH_TYPES:
            continue
        name = item.get("shortname") or item.get("longname") or symbol
        exchange = str(item.get("exchange") or "").strip()
        results.append(
            {
                "symbol": symbol,
                "name": str(name).strip(),
                "exchange": exchange,
                "type": quote_type or "EQUITY",
            }
        )
        if len(results) >= max_results:
            break
    return results


@st.cache_data(ttl=3600, show_spinner=False)
def search_yahoo_symbols(query: str, max_results: int = 8):
    """Recherche Yahoo Finance par nom de société ou ticker partiel."""
    query = str(query).strip()
    if len(query) < 2:
        return []

    try:
        import requests

        response = requests.get(
            YAHOO_SEARCH_URL,
            params={
                "q": query,
                "quotesCount": max_results,
                "newsCount": 0,
                "enableFuzzyQuery": True,
            },
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=12,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return []

    quotes = payload.get("quotes") if isinstance(payload, dict) else None
    return parse_yahoo_search_quotes(quotes, max_results=max_results)


def parse_fmp_search_results(items, max_results=8):
    """Extrait symboles action/ETF d'une réponse FMP search."""
    results: list[dict[str, str]] = []
    if not items:
        return results
    for item in items:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol") or "").strip()
        if not symbol:
            continue
        name = item.get("name") or symbol
        exchange = str(
            item.get("exchangeShortName") or item.get("stockExchange") or ""
        ).strip()
        results.append(
            {
                "symbol": symbol,
                "name": str(name).strip(),
                "exchange": exchange,
                "type": "EQUITY",
            }
        )
        if len(results) >= max_results:
            break
    return results


@st.cache_data(ttl=3600, show_spinner=False)
def search_fmp_symbols(query: str, max_results: int = 8):
    """Recherche FMP par nom de société ou ticker partiel (repli Yahoo)."""
    query = str(query).strip()
    if len(query) < 2:
        return []

    api_key = get_fmp_api_key()
    if not api_key:
        return []

    try:
        import requests

        response = requests.get(
            FMP_SEARCH_URL,
            params={"query": query, "limit": max_results, "apikey": api_key},
            timeout=12,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return []

    if not isinstance(payload, list):
        return []
    return parse_fmp_search_results(payload, max_results=max_results)


@st.cache_data(ttl=3600, show_spinner=False)
def search_market_symbols(query: str, max_results: int = 8):
    """Recherche par nom ou ticker : Yahoo Finance, puis repli FMP."""
    hits = search_yahoo_symbols(query, max_results)
    if hits:
        return hits
    return search_fmp_symbols(query, max_results)


def ticker_label(ticker, names, show_names=True):
    tk = str(ticker).strip()
    if not show_names:
        return tk
    name = names.get(tk, tk)
    if not name or name == tk:
        return tk
    return f"{tk} — {name}"


def add_company_names(
    df,
    names,
    ticker_col="Ticker",
    show_names=True,
    sectors=None,
):
    """Ajoute « Nom » et/ou « Secteur » après le ticker."""
    if ticker_col not in df.columns:
        return df
    if not show_names and sectors is None:
        return df

    out = df.copy()
    insert_at = 1
    if show_names:
        out.insert(
            1,
            "Nom",
            out[ticker_col].map(lambda t: names.get(str(t).strip(), str(t).strip())),
        )
        insert_at = 2
    if sectors is not None and "Secteur" not in out.columns:
        out.insert(
            insert_at,
            "Secteur",
            out[ticker_col].map(lambda t: sectors.get(str(t).strip()) or "—"),
        )
    return out


def label_index_with_names(df, names, show_names=True):
    """Renomme l'index (tickers) pour affichage avec nom d'entreprise."""
    if not show_names:
        return df
    out = df.copy()
    out.index = [ticker_label(t, names, True) for t in out.index]
    return out
