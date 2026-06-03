import re
import random
import time
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup
from yahooquery import Ticker

from display_units import (
    DIVIDEND_PORTFOLIO_CAPTION,
    DIVIDEND_PORTFOLIO_FORMAT_BASE,
    DIVIDEND_PORTFOLIO_LABELS,
    DIVIDEND_UNIVERSE_CAPTION,
    DIVIDEND_UNIVERSE_FORMAT,
    DIVIDEND_UNIVERSE_LABELS,
    expand_sentiment_columns,
    format_map_for_labeled_columns,
    internal_column_name,
    rename_columns_for_display,
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

_TICKER_SUFFIX_MARKET = {
    ".PA": "Euronext Paris",
    ".BR": "Euronext Bruxelles",
    ".AS": "Euronext Amsterdam",
    ".MI": "Borsa Italiana",
    ".MC": "Bolsas y Mercados Españoles",
    ".DE": "Xetra (Francfort)",
    ".L": "Londres (LSE)",
    ".SW": "SIX Swiss Exchange",
    ".HE": "Helsinki",
    ".IR": "Euronext Dublin",
    ".TO": "Toronto (TSX)",
    ".V": "TSX Venture",
    ".US": "NYSE / NASDAQ",
}

_TICKER_SUFFIX_CURRENCY = {
    ".PA": "EUR",
    ".BR": "EUR",
    ".AS": "EUR",
    ".MI": "EUR",
    ".MC": "EUR",
    ".DE": "EUR",
    ".L": "GBp",
    ".SW": "CHF",
    ".HE": "EUR",
    ".IR": "EUR",
    ".TO": "CAD",
    ".V": "CAD",
}

_EXCHANGE_LABELS = {
    "PAR": "Euronext Paris",
    "BRU": "Euronext Bruxelles",
    "AMS": "Euronext Amsterdam",
    "GER": "Xetra (Francfort)",
    "FRA": "Francfort",
    "LSE": "Londres (LSE)",
    "NYQ": "NYSE",
    "NMS": "NASDAQ",
    "NGM": "NASDAQ",
    "TOR": "Toronto (TSX)",
    "CNQ": "TSX Venture",
}


def _infer_market_from_ticker(ticker):
    tk = str(ticker).upper().strip()
    for suffix, label in _TICKER_SUFFIX_MARKET.items():
        if tk.endswith(suffix):
            return label
    if "." not in tk:
        return "NYSE / NASDAQ (US)"
    return None


def _infer_currency_from_ticker(ticker):
    tk = str(ticker).upper().strip()
    for suffix, cur in _TICKER_SUFFIX_CURRENCY.items():
        if tk.endswith(suffix):
            return cur
    if "." not in tk:
        return "USD"
    return None


def _normalize_market_label(raw_market, ticker):
    if raw_market is None or (isinstance(raw_market, float) and np.isnan(raw_market)):
        raw_market = None
    if raw_market:
        key = str(raw_market).strip().upper()
        if key in _EXCHANGE_LABELS:
            return _EXCHANGE_LABELS[key]
        if len(key) <= 5 and key.isalpha():
            mapped = _EXCHANGE_LABELS.get(key)
            if mapped:
                return mapped
        return str(raw_market).strip()
    inferred = _infer_market_from_ticker(ticker)
    return inferred or "—"


@st.cache_data(ttl=86400, show_spinner=False)
def get_ticker_listing_info(tickers):
    """Nom, marché et devise Yahoo (lots) avec repli sur le suffixe du ticker."""
    tickers = tuple(sorted({str(t).strip() for t in tickers if str(t).strip()}))
    out = {
        t: {
            "Société": t,
            "Marché": _infer_market_from_ticker(t) or "—",
            "Devise": _infer_currency_from_ticker(t) or "—",
            "Secteur": "—",
        }
        for t in tickers
    }
    if not tickers:
        return out

    try:
        chunk_size = 100
        for i in range(0, len(tickers), chunk_size):
            chunk = list(tickers[i : i + chunk_size])
            tq = Ticker(chunk, timeout=25)
            payload = tq.price
            profiles = tq.asset_profile
            if not isinstance(payload, dict):
                payload = {}
            if not isinstance(profiles, dict):
                profiles = {}
            for tk in chunk:
                block = payload.get(tk)
                if isinstance(block, dict):
                    name = block.get("shortName") or block.get("longName")
                    if name:
                        out[tk]["Société"] = str(name)
                    market = block.get("fullExchangeName") or block.get("exchange")
                    out[tk]["Marché"] = _normalize_market_label(market, tk)
                    currency = block.get("currency")
                    if currency:
                        out[tk]["Devise"] = str(currency)
                profile = profiles.get(tk)
                if isinstance(profile, dict):
                    sector = profile.get("sector")
                    if sector:
                        out[tk]["Secteur"] = str(sector)
    except Exception:
        pass

    return out


def _listing_fields_for_ticker(ticker):
    info_map = get_ticker_listing_info((str(ticker).strip(),))
    return info_map.get(str(ticker).strip(), {
        "Société": ticker,
        "Secteur": "—",
        "Marché": _infer_market_from_ticker(ticker) or "—",
        "Devise": _infer_currency_from_ticker(ticker) or "—",
    })


def _enrich_dividend_display_df(df):
    """Ajoute Société, Marché, Devise ; conserve Ticker pour filtres internes."""
    if df.empty or "Ticker" not in df.columns:
        return df
    out = df.copy()
    listing = get_ticker_listing_info(tuple(out["Ticker"].astype(str)))

    out["Société"] = out["Ticker"].map(lambda t: listing.get(str(t).strip(), {}).get("Société", t))
    out["Secteur"] = out["Ticker"].map(
        lambda t: listing.get(str(t).strip(), {}).get("Secteur", "—")
    )
    out["Marché"] = out["Ticker"].map(
        lambda t: listing.get(str(t).strip(), {}).get("Marché", _infer_market_from_ticker(t) or "—")
    )
    out["Devise"] = out["Ticker"].map(
        lambda t: listing.get(str(t).strip(), {}).get("Devise", _infer_currency_from_ticker(t) or "—")
    )

    for field in ("Société", "Secteur", "Marché", "Devise"):
        if field in df.columns:
            out[field] = df[field].combine_first(out[field])

    return out


def _dividend_table_column_order(df, leading=None):
    leading = leading or ["Société", "Secteur", "Marché", "Devise", "Statut"]
    cols = [c for c in leading if c in df.columns]
    cols += [c for c in df.columns if c not in cols and c != "Ticker"]
    return df[cols]


_STYLE_FAVORABLE = "background-color: #d4edda; color: #155724"
_STYLE_UNFAVORABLE = "background-color: #f8d7da; color: #721c24"

_DIVIDEND_SENTIMENT_COLUMNS = expand_sentiment_columns(
    {
        "Rendement (%)",
        "Rendement",
        "Rendement moy. 5 ans (%)",
        "Croissance 1 an (%)",
        "Croissance 3 ans (%)",
        "Croissance 5 ans (%)",
        "CAGR 5 ans",
        "Ratio couverture",
        "Taux versement (%)",
        "Payout",
        "FCF Payout",
        "Série versement (ans)",
        "Série croissance (ans)",
        "Années de croissance",
    },
    DIVIDEND_UNIVERSE_LABELS,
    DIVIDEND_PORTFOLIO_LABELS,
)


def _to_metric_float(val):
    if val is None:
        return None
    if isinstance(val, float) and np.isnan(val):
        return None
    if isinstance(val, str):
        s = val.strip().lower()
        if not s or s in {"-", "none", "nan"}:
            return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _dividend_metric_sentiment(column, val):
    """Vert = favorable, rouge = défavorable (seuils alignés sur la légende dividendes)."""
    column = internal_column_name(
        column, DIVIDEND_UNIVERSE_LABELS, DIVIDEND_PORTFOLIO_LABELS
    )
    v = _to_metric_float(val)
    if v is None:
        return ""

    if column in ("Rendement (%)", "Rendement", "Rendement moy. 5 ans (%)"):
        if v >= 0.02:
            return _STYLE_FAVORABLE
        if v < 0.01:
            return _STYLE_UNFAVORABLE

    elif column.startswith("Croissance") or column == "CAGR 5 ans":
        if v >= 0.03:
            return _STYLE_FAVORABLE
        if v < 0:
            return _STYLE_UNFAVORABLE

    elif column == "Ratio couverture":
        if v >= 1.5:
            return _STYLE_FAVORABLE
        if v < 1.0:
            return _STYLE_UNFAVORABLE

    elif column in ("Taux versement (%)", "Payout", "FCF Payout"):
        if v <= 0.60:
            return _STYLE_FAVORABLE
        if v > 0.80:
            return _STYLE_UNFAVORABLE

    elif column in ("Série versement (ans)", "Série croissance (ans)", "Années de croissance"):
        if v >= 5:
            return _STYLE_FAVORABLE
        if v <= 1:
            return _STYLE_UNFAVORABLE

    return ""


def _style_dividend_sentiment(styler):
    """Applique vert/rouge aux colonnes métriques dividendes."""

    def _color_column(series):
        if series.name not in _DIVIDEND_SENTIMENT_COLUMNS:
            return [""] * len(series)
        return [_dividend_metric_sentiment(series.name, v) for v in series]

    return styler.apply(_color_column, axis=0)


style_dividend_sentiment = _style_dividend_sentiment

SUGGESTION_DIVIDEND_COLUMNS = (
    "Statut",
    "Rendement",
    "CAGR 5 ans",
    "Années de croissance",
    "Payout",
    "FCF Payout",
    "Ratio couverture",
    "Ratio couverture FCF",
    "Score",
)


def _normalize_suggestion_dividend_row(row):
    if not row:
        return None
    out = dict(row)
    payout = out.get("Payout")
    fcf_payout = out.get("FCF Payout")
    if payout and payout > 0:
        out["Ratio couverture"] = 1.0 / payout
    if fcf_payout and fcf_payout > 0:
        out["Ratio couverture FCF"] = 1.0 / fcf_payout
    out.setdefault("_fetched_at", datetime.now().isoformat(timespec="seconds"))
    return out


def fetch_dividends_for_tickers(
    tickers,
    prices,
    cache=None,
    progress_cb=None,
    throttle=True,
    use_disk_cache=True,
):
    """Collecte les indicateurs dividendes (session + disque 24 h, puis Yahoo)."""
    cache = cache if cache is not None else {}
    rows = []
    stats = {"session": 0, "disk": 0, "fetched": 0, "missing": 0}
    tickers = [t for t in dict.fromkeys(tickers) if t]
    for i, t in enumerate(tickers):
        if progress_cb:
            progress_cb(t, i + 1, len(tickers), stats)
        cached = cache.get(t)
        if cached and _dividend_row_is_fresh(cached):
            rows.append(cached)
            stats["session"] += 1
            continue
        if use_disk_cache:
            disk_row = load_dividend_ticker_from_disk(t)
            if disk_row is not None:
                cache[t] = disk_row
                rows.append(disk_row)
                stats["disk"] += 1
                continue
        if throttle and stats["fetched"] > 0:
            time.sleep(random.uniform(0.15, 0.35))
        lp = None
        if prices is not None and t in prices.columns:
            s = prices[t].dropna()
            if len(s):
                lp = float(s.iloc[-1])
        if lp is None or lp <= 0:
            stats["missing"] += 1
            continue
        row, _ = fetch_ticker_dividend_bundle(t, lp)
        row = _normalize_suggestion_dividend_row(row)
        if row is not None:
            cache[t] = row
            if use_disk_cache:
                save_dividend_ticker_to_disk(t, row)
            rows.append(row)
            stats["fetched"] += 1
        else:
            stats["missing"] += 1
    return rows, cache, stats


def merge_dividend_columns(df, dividend_rows, columns=None):
    """Joint les indicateurs dividendes au tableau de suggestions (clé Ticker)."""
    if df is None or df.empty or not dividend_rows:
        return df
    columns = columns or SUGGESTION_DIVIDEND_COLUMNS
    div_df = pd.DataFrame(dividend_rows)
    if div_df.empty or "Ticker" not in div_df.columns:
        return df
    keep = ["Ticker"] + [c for c in columns if c in div_df.columns]
    div_df = div_df[keep].drop_duplicates(subset=["Ticker"], keep="last")
    return df.merge(div_df, on="Ticker", how="left")


def filter_suggestions_by_dividends(
    df,
    min_yield=None,
    max_yield=None,
    min_coverage=None,
    min_cagr_5y=None,
    min_growth_years=None,
    max_payout=None,
    max_fcf_payout=None,
    active_only=False,
    min_score=None,
):
    """Filtre les suggestions selon des seuils dividendes."""
    if df is None or df.empty:
        return df
    out = df.copy()
    if active_only and "Statut" in out.columns:
        out = out[out["Statut"] == "Actif"]
    if min_yield is not None and "Rendement" in out.columns:
        out = out[out["Rendement"].fillna(-1) >= min_yield]
    if max_yield is not None and "Rendement" in out.columns:
        out = out[out["Rendement"].isna() | (out["Rendement"] <= max_yield)]
    if min_coverage is not None and "Ratio couverture" in out.columns:
        out = out[out["Ratio couverture"].fillna(-1) >= min_coverage]
    if min_cagr_5y is not None and "CAGR 5 ans" in out.columns:
        out = out[out["CAGR 5 ans"].fillna(-1) >= min_cagr_5y]
    if min_growth_years is not None and "Années de croissance" in out.columns:
        out = out[out["Années de croissance"].fillna(-1) >= min_growth_years]
    if max_payout is not None and "Payout" in out.columns:
        out = out[out["Payout"].isna() | (out["Payout"] <= max_payout)]
    if max_fcf_payout is not None and "FCF Payout" in out.columns:
        out = out[out["FCF Payout"].isna() | (out["FCF Payout"] <= max_fcf_payout)]
    if min_score is not None and "Score" in out.columns:
        out = out[out["Score"].fillna(-1) >= min_score]
    return out


DIVIDENDS_DISK_CACHE_FILE = "dividendes_cache.json"
DIVIDENDS_UNIVERSE_FILE = "dividendes_universe.json"
DISK_CACHE_TTL_SECONDS = 86400


def _dividends_cache_path():
    return Path(__file__).resolve().parent / DIVIDENDS_DISK_CACHE_FILE


def _dividends_universe_path():
    return Path(__file__).resolve().parent / DIVIDENDS_UNIVERSE_FILE


def _empty_universe():
    return {
        "version": 1,
        "meta": {
            "updated_at": None,
            "total_target": 0,
            "processed": 0,
            "with_dividends": 0,
            "without_dividends": 0,
            "failed": 0,
        },
        "tickers": {},
        "failed_tickers": {},
    }


def load_dividend_universe():
    path = _dividends_universe_path()
    if not path.is_file():
        return _empty_universe()
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("tickers"), dict):
            data.setdefault("meta", {})
            data.setdefault("failed_tickers", {})
            return data
    except Exception:
        pass
    return _empty_universe()


def save_dividend_universe(data):
    data.setdefault("meta", {})["updated_at"] = datetime.now().isoformat(timespec="seconds")
    path = _dividends_universe_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def load_tickers_from_json(json_path="tickers.json"):
    path = Path(__file__).resolve().parent / json_path
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return sorted({str(t).strip() for values in payload.values() for t in values if str(t).strip()})


def _load_dividends_disk_store():
    path = _dividends_cache_path()
    if not path.is_file():
        return {"version": 1, "portfolios": {}}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("portfolios"), dict):
            data.setdefault("tickers", {})
            return data
    except Exception:
        pass
    return {"version": 1, "portfolios": {}, "tickers": {}}


def _save_dividends_disk_store(data):
    path = _dividends_cache_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def load_dividends_rows_from_disk(ticker_signature):
    """Charge les lignes dividendes depuis le disque (TTL 24 h)."""
    entry = _load_dividends_disk_store().get("portfolios", {}).get(ticker_signature)
    if not entry:
        return None, None
    updated_at = entry.get("updated_at")
    if updated_at:
        try:
            ts = datetime.fromisoformat(updated_at)
            if (datetime.now() - ts).total_seconds() > DISK_CACHE_TTL_SECONDS:
                return None, updated_at
        except Exception:
            pass
    rows = entry.get("rows")
    if not rows:
        return None, updated_at
    return rows, updated_at


def save_dividends_rows_to_disk(ticker_signature, rows):
    """Enregistre les résultats dividendes sur le disque."""
    store = _load_dividends_disk_store()
    store.setdefault("portfolios", {})[ticker_signature] = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "rows": rows,
    }
    _save_dividends_disk_store(store)


def _dividend_row_is_fresh(row, ttl_seconds=DISK_CACHE_TTL_SECONDS):
    ts = row.get("_fetched_at") if row else None
    if not ts:
        return False
    try:
        return (datetime.now() - datetime.fromisoformat(ts)).total_seconds() <= ttl_seconds
    except Exception:
        return False


def load_dividend_ticker_from_disk(ticker):
    """Charge les dividendes d'un ticker depuis le cache disque (TTL 24 h)."""
    entry = _load_dividends_disk_store().get("tickers", {}).get(ticker)
    if not entry:
        return None
    row = entry.get("row") if isinstance(entry, dict) else entry
    if row and _dividend_row_is_fresh(row):
        return row
    return None


def save_dividend_ticker_to_disk(ticker, row):
    """Enregistre les dividendes d'un ticker sur le disque."""
    if not row:
        return
    store = _load_dividends_disk_store()
    store.setdefault("tickers", {})[ticker] = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "row": row,
    }
    _save_dividends_disk_store(store)


def clear_dividends_disk_cache(ticker_signature=None):
    """Supprime le cache disque (une liste de tickers ou tout le fichier)."""
    if ticker_signature is None:
        path = _dividends_cache_path()
        if path.is_file():
            path.unlink()
        return
    store = _load_dividends_disk_store()
    portfolios = store.get("portfolios", {})
    if ticker_signature in portfolios:
        del portfolios[ticker_signature]
        _save_dividends_disk_store(store)


def _hydrate_session_from_disk(cache_key, ticker_signature):
    """Recharge session_state depuis le disque si la session est vide."""
    if st.session_state.get(cache_key):
        return False
    rows, updated_at = load_dividends_rows_from_disk(ticker_signature)
    if not rows:
        return False
    st.session_state[cache_key] = rows
    st.session_state[f"{cache_key}_disk_loaded"] = updated_at
    return True


def safe_get(url, max_retries=3, sleep_base=1.0):
    for attempt in range(max_retries):
        try:
            time.sleep(random.uniform(0.3, 0.8))
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                return r.text
            if r.status_code == 429:
                time.sleep(sleep_base * 4)
        except Exception:
            pass
        time.sleep(sleep_base * (2**attempt) + random.random() * 0.5)
    return None


def get_isin_from_yahoo(ticker):
    try:
        info = yf.Ticker(ticker).info or {}
        for key in ("isin", "ISIN", "isinCode"):
            if info.get(key):
                return info[key]
    except Exception:
        pass
    try:
        p = Ticker(ticker, timeout=15).price
        if isinstance(p, dict) and ticker in p:
            d = p[ticker]
            for key in ("isin", "ISIN", "isinCode"):
                if d.get(key):
                    return d[key]
    except Exception:
        pass
    return None


def get_dividends_yahoo(ticker):
    try:
        d = Ticker(ticker, timeout=20).dividends
        if isinstance(d, pd.DataFrame) and not d.empty:
            if isinstance(d.index, pd.MultiIndex):
                if ticker in d.index.get_level_values(0):
                    d = d.xs(ticker, level=0)
                else:
                    return None
            if "dividends" in d.columns:
                s = pd.to_numeric(d["dividends"], errors="coerce").dropna()
                s.index = pd.to_datetime(s.index)
                return s.sort_index() if not s.empty else None
    except Exception:
        pass
    return None


def _extract_dividends_from_history(hist, ticker_symbol):
    if hist is None or hist.empty:
        return None

    if isinstance(hist.columns, pd.MultiIndex):
        if ticker_symbol in hist.columns.get_level_values(1):
            div_col = ("Dividends", ticker_symbol)
            s = hist[div_col] if div_col in hist.columns else None
        elif ticker_symbol in hist.columns.get_level_values(0):
            div_col = (ticker_symbol, "Dividends")
            s = hist[div_col] if div_col in hist.columns else None
        else:
            level0 = hist.columns.get_level_values(0)
            if "Dividends" not in level0:
                return None
            div_frame = hist.xs("Dividends", axis=1, level=0)
            s = div_frame.iloc[:, 0] if not div_frame.empty else None
    else:
        s = hist["Dividends"] if "Dividends" in hist.columns else None

    if s is None:
        return None
    s = pd.to_numeric(s, errors="coerce").dropna()
    s = s[s > 0]
    if s.empty:
        return None
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s.sort_index()


def get_dividends_yfinance_native(ticker):
    try:
        s = yf.Ticker(ticker).dividends
        if s is not None and not s.empty:
            s = pd.to_numeric(s, errors="coerce").dropna()
            s = s[s > 0]
            if not s.empty:
                s.index = pd.to_datetime(s.index).tz_localize(None)
                return s.sort_index()
    except Exception:
        pass

    try:
        hist = yf.Ticker(ticker).history(period="max", auto_adjust=False)
        s = _extract_dividends_from_history(hist, ticker)
        if s is not None:
            return s
    except Exception:
        pass

    try:
        hist2 = yf.download(
            ticker,
            period="max",
            auto_adjust=False,
            actions=True,
            progress=False,
            threads=False,
        )
        return _extract_dividends_from_history(hist2, ticker)
    except Exception:
        pass
    return None


def normalize_boursorama_ticker(t):
    code = t.split(".")[0]
    if ".PA" in t:
        return f"1rP{code}"
    if ".BR" in t:
        return f"1rB{code}"
    return code


def get_dividends_boursorama(ticker):
    code = normalize_boursorama_ticker(ticker)
    url = f"https://www.boursorama.com/cours/dividendes/{code}/"
    html = safe_get(url)
    if html is None:
        return None

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", {"class": "c-table"}) or soup.find("table")
    if table is None:
        return None

    data = []
    for row in table.find_all("tr")[1:]:
        cells = row.find_all("td")
        if len(cells) < 2:
            continue
        date_txt = cells[0].get_text(strip=True)
        div_txt = cells[1].get_text(strip=True).replace(",", ".")
        if len(date_txt) == 4 and date_txt.isdigit():
            date_txt = f"{date_txt}-06-01"
        try:
            val = float(re.findall(r"[-+]?\d*\.?\d+", div_txt)[0])
        except Exception:
            continue
        data.append((date_txt, val))

    if not data:
        return None
    idx = pd.to_datetime([d[0] for d in data], errors="coerce")
    s = pd.Series([d[1] for d in data], index=idx).dropna().sort_index()
    return s if not s.empty else None


def get_dividends_lecho(isin):
    if not isin:
        return None
    html = safe_get(f"https://www.lecho.be/cours/{isin}")
    if html is None:
        return None
    text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    matches = re.findall(r"Dividende\s+([\d,\.]+)", text)
    vals = []
    for m in matches:
        try:
            vals.append(float(m.replace(",", ".")))
        except Exception:
            pass
    if not vals:
        return None
    idx = pd.to_datetime(["2020-01-01"] * len(vals)) + pd.to_timedelta(
        list(range(len(vals))), unit="Y"
    )
    return pd.Series(vals, index=idx).sort_index()


def fetch_dividend_series(ticker):
    for name, fn in (
        ("YahooQuery", get_dividends_yahoo),
        ("yfinance", get_dividends_yfinance_native),
        ("Boursorama", get_dividends_boursorama),
    ):
        s = fn(ticker)
        if s is not None and not s.empty:
            return s, name

    isin = get_isin_from_yahoo(ticker)
    if isin:
        s = get_dividends_lecho(isin)
        if s is not None and not s.empty:
            return s, "LEcho"
    return None, None


DIVIDEND_TTM_DAYS = 365


def _normalize_dividend_index(series):
    s = series.copy()
    if hasattr(s.index, "tz") and s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    return s.sort_index()


def _dividend_ttm(series, lookback_days=DIVIDEND_TTM_DAYS):
    """Somme des dividendes versés sur les N derniers jours (glissant depuis aujourd'hui)."""
    if series is None or series.empty:
        return 0.0
    s = _normalize_dividend_index(series)
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=lookback_days)
    return float(s[s.index >= cutoff].sum())


def _last_dividend_payment(series):
    """Date et montant du dernier versement positif."""
    if series is None or series.empty:
        return None, 0.0
    s = _normalize_dividend_index(series)
    positive = s[s > 0]
    if positive.empty:
        return None, 0.0
    last_date = positive.index.max()
    return last_date, float(positive.loc[last_date])


def _is_dividend_active(series, lookback_days=DIVIDEND_TTM_DAYS):
    """Verseur actif = au moins un paiement dans les 12 derniers mois."""
    return _dividend_ttm(series, lookback_days) > 0


def compute_dividend_metrics(s, last_price):
    if s is None or len(s) == 0 or last_price is None or last_price <= 0:
        return None
    s = _normalize_dividend_index(s)
    df = s.groupby(s.index.year).sum().sort_index()
    if df.empty:
        return None

    ttm = _dividend_ttm(s)
    last_payment_date, last_payment_amount = _last_dividend_payment(s)
    active = ttm > 0
    last_div = ttm if active else 0.0
    y = (ttm / last_price) if active else None

    cagr = None
    paid_years = df[df > 0]
    if len(paid_years) >= 5:
        start, end = paid_years.iloc[-5], paid_years.iloc[-1]
        if start > 0 and end > 0:
            cagr = (end / start) ** (1 / 4) - 1

    growth_years = 0
    if len(paid_years) >= 2:
        prev = paid_years.iloc[0]
        for val in paid_years.iloc[1:]:
            if val >= prev:
                growth_years += 1
            prev = val

    return {
        "last_year_div": float(last_div),
        "current_yield": float(y) if y is not None else None,
        "dividend_ttm": float(ttm),
        "dividend_active": active,
        "last_payment_date": last_payment_date,
        "last_payment_amount": float(last_payment_amount),
        "cagr_5y": float(cagr) if cagr is not None else None,
        "growth_years": int(growth_years),
        "years_available": int(len(paid_years)),
        "annual_series": df,
    }


def _infer_dividend_frequency(div_series):
    if div_series is None or div_series.empty:
        return None
    div_series = _normalize_dividend_index(div_series)
    recent = div_series[
        div_series.index >= (pd.Timestamp.now().normalize() - pd.Timedelta(days=DIVIDEND_TTM_DAYS))
    ]
    n = int((recent > 0).sum()) if not recent.empty else 0
    if n == 0:
        return "Aucun"
    if n == 1:
        return "Annuelle"
    if n == 2:
        return "Semestrielle"
    if n <= 4:
        return "Trimestrielle"
    if n >= 10:
        return "Mensuelle"
    return "Irregulière"


def _cagr_between(annual, years_back):
    if len(annual) < years_back + 1:
        return None
    start = float(annual.iloc[-1 - years_back])
    end = float(annual.iloc[-1])
    if start <= 0 or end <= 0:
        return None
    return (end / start) ** (1 / years_back) - 1


def _consecutive_payment_years(annual):
    streak = 0
    for val in reversed(annual.tolist()):
        if val and val > 0:
            streak += 1
        else:
            break
    return streak


def _consecutive_growth_years(annual):
    streak = 0
    vals = annual.tolist()
    for i in range(len(vals) - 1, 0, -1):
        if vals[i - 1] > 0 and vals[i] >= vals[i - 1]:
            streak += 1
        else:
            break
    return streak


def get_last_price_yahoo(ticker):
    try:
        info = yf.Ticker(ticker).info or {}
        for key in ("regularMarketPrice", "currentPrice", "previousClose"):
            val = info.get(key)
            if val is not None and float(val) > 0:
                return float(val)
    except Exception:
        pass
    try:
        hist = yf.download(ticker, period="10d", progress=False, threads=False)
        if hist is not None and not hist.empty:
            close = hist["Close"] if "Close" in hist.columns else hist.iloc[:, 0]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            last = float(close.dropna().iloc[-1])
            if last > 0:
                return last
    except Exception:
        pass
    return None


def get_upcoming_dividend_dates(ticker):
    ex_date = pay_date = None
    try:
        info = yf.Ticker(ticker).info or {}
        ex_raw = info.get("exDividendDate")
        pay_raw = info.get("dividendDate")
        if ex_raw is not None:
            ex_date = pd.to_datetime(ex_raw, unit="s", errors="coerce")
            if pd.isna(ex_date):
                ex_date = pd.to_datetime(ex_raw, errors="coerce")
        if pay_raw is not None:
            pay_date = pd.to_datetime(pay_raw, unit="s", errors="coerce")
            if pd.isna(pay_date):
                pay_date = pd.to_datetime(pay_raw, errors="coerce")
    except Exception:
        pass
    ex_str = ex_date.strftime("%Y-%m-%d") if ex_date is not None and not pd.isna(ex_date) else None
    pay_str = pay_date.strftime("%Y-%m-%d") if pay_date is not None and not pd.isna(pay_date) else None
    return ex_str, pay_str


def compute_avg_yield_5y(div_series, ticker):
    if div_series is None or div_series.empty:
        return None
    annual = div_series.groupby(div_series.index.year).sum().sort_index()
    if len(annual) < 2:
        return None
    try:
        hist = yf.download(ticker, period="6y", progress=False, threads=False)
        if hist is None or hist.empty:
            return None
        close = hist["Close"] if "Close" in hist.columns else hist.iloc[:, 0]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close.index = pd.to_datetime(close.index).tz_localize(None)
        yields = []
        for year in annual.index[-5:]:
            year_close = close[close.index.year == year]
            if year_close.empty or annual.loc[year] <= 0:
                continue
            price_eoy = float(year_close.iloc[-1])
            if price_eoy > 0:
                yields.append(float(annual.loc[year]) / price_eoy)
        if not yields:
            return None
        return float(np.mean(yields))
    except Exception:
        return None


def fetch_comprehensive_dividend_profile(ticker):
    """Profil dividendes complet (style Investing.com) pour un ticker."""
    ticker = str(ticker).strip()
    last_price = get_last_price_yahoo(ticker)
    series, source = fetch_dividend_series(ticker)
    ex_date, pay_date = get_upcoming_dividend_dates(ticker)
    payout, fcf_payout = get_payout_ratios_yahoo(ticker)
    listing = _listing_fields_for_ticker(ticker)
    base_listing = {
        "Société": listing["Société"],
        "Marché": listing["Marché"],
        "Devise": listing["Devise"],
    }

    if series is None or series.empty:
        return {
            "Ticker": ticker,
            **base_listing,
            "Prix": last_price,
            "Rendement (%)": None,
            "Fréquence": "Aucun",
            "Croissance 1 an (%)": None,
            "Croissance 3 ans (%)": None,
            "Croissance 5 ans (%)": None,
            "Ratio couverture": None,
            "Ratio couverture FCF": None,
            "Taux versement (%)": payout,
            "Série versement (ans)": 0,
            "Série croissance (ans)": 0,
            "Ex-dividende à venir": ex_date,
            "Paiement à venir": pay_date,
            "Rendement moy. 5 ans (%)": None,
            "Dividende / action": None,
            "Dividende TTM": None,
            "Dividende total historique": None,
            "Source": source,
            "A des dividendes": False,
        }

    metrics = compute_dividend_metrics(series, last_price)
    annual = metrics["annual_series"] if metrics else series.groupby(series.index.year).sum().sort_index()
    ttm = metrics["dividend_ttm"] if metrics else _dividend_ttm(series)
    active = metrics["dividend_active"] if metrics else ttm > 0
    total_hist = float(series.sum())
    dps = float(metrics["last_year_div"]) if metrics and active else None

    paid_annual = annual[annual > 0]
    growth_1y = None
    if len(paid_annual) >= 2 and paid_annual.iloc[-2] > 0:
        growth_1y = float(paid_annual.iloc[-1] / paid_annual.iloc[-2] - 1)

    coverage = (1.0 / payout) if payout and payout > 0 else None
    fcf_coverage = (1.0 / fcf_payout) if fcf_payout and fcf_payout > 0 else None

    last_pay = metrics.get("last_payment_date") if metrics else None
    last_pay_str = (
        pd.Timestamp(last_pay).strftime("%Y-%m-%d") if last_pay is not None else None
    )

    return {
        "Ticker": ticker,
        **base_listing,
        "Prix": last_price,
        "Statut": "Actif" if active else "Suspendu",
        "Rendement (%)": metrics["current_yield"] if metrics and active else None,
        "Fréquence": _infer_dividend_frequency(series) if active else "Aucun",
        "Croissance 1 an (%)": growth_1y if active else None,
        "Croissance 3 ans (%)": _cagr_between(paid_annual, 3) if active else None,
        "Croissance 5 ans (%)": metrics["cagr_5y"] if metrics and active else None,
        "Ratio couverture": coverage,
        "Ratio couverture FCF": fcf_coverage,
        "Taux versement (%)": payout,
        "Série versement (ans)": _consecutive_payment_years(paid_annual) if active else 0,
        "Série croissance (ans)": _consecutive_growth_years(paid_annual) if active else 0,
        "Ex-dividende à venir": ex_date if active else None,
        "Paiement à venir": pay_date if active else None,
        "Rendement moy. 5 ans (%)": compute_avg_yield_5y(series, ticker),
        "Dividende / action": dps,
        "Dividende TTM": ttm if active else 0.0,
        "Dernier versement": last_pay_str,
        "DPS dernier versement": metrics.get("last_payment_amount") if metrics else None,
        "Dividende total historique": total_hist,
        "Source": source,
        "A des dividendes": active,
    }


def process_dividend_universe_batch(
    tickers,
    universe=None,
    limit=None,
    sleep_seconds=0.25,
    save_every=20,
    progress_callback=None,
):
    """Traite un lot de tickers et met à jour dividendes_universe.json."""
    universe = universe or load_dividend_universe()
    universe.setdefault("tickers", {})
    universe.setdefault("failed_tickers", {})
    meta = universe.setdefault("meta", {})
    meta["total_target"] = len(tickers)

    pending = [t for t in tickers if t not in universe["tickers"] and t not in universe["failed_tickers"]]
    if limit is not None:
        pending = pending[:limit]

    processed_now = 0
    batch_total = len(pending)
    for i, ticker in enumerate(pending):
        if i > 0 and sleep_seconds:
            time.sleep(random.uniform(sleep_seconds * 0.6, sleep_seconds * 1.4))
        try:
            row = fetch_comprehensive_dividend_profile(ticker)
            universe["tickers"][ticker] = row
        except Exception as exc:
            universe["failed_tickers"][ticker] = str(exc)
        processed_now += 1
        if progress_callback and batch_total:
            progress_callback((i + 1) / batch_total, ticker, processed_now, batch_total)
        if processed_now % save_every == 0:
            meta["with_dividends"] = sum(
                1 for r in universe["tickers"].values() if r.get("A des dividendes")
            )
            meta["without_dividends"] = sum(
                1 for r in universe["tickers"].values() if not r.get("A des dividendes")
            )
            meta["failed"] = len(universe["failed_tickers"])
            meta["processed"] = len(universe["tickers"]) + len(universe["failed_tickers"])
            save_dividend_universe(universe)

    meta["with_dividends"] = sum(
        1 for r in universe["tickers"].values() if r.get("A des dividendes")
    )
    meta["without_dividends"] = sum(
        1 for r in universe["tickers"].values() if not r.get("A des dividendes")
    )
    meta["failed"] = len(universe["failed_tickers"])
    meta["processed"] = len(universe["tickers"]) + len(universe["failed_tickers"])
    save_dividend_universe(universe)
    return universe, processed_now


def get_payout_ratios_yahoo(ticker):
    payout = fcf = None
    try:
        t = Ticker(ticker, timeout=20)
        inc = t.income_statement(frequency="a")
        cf = t.cash_flow(frequency="a")
        if (
            isinstance(inc, pd.DataFrame)
            and not inc.empty
            and isinstance(cf, pd.DataFrame)
            and not cf.empty
        ):
            li = inc.sort_index().iloc[-1]
            lc = cf.sort_index().iloc[-1]
            div_cols = [c for c in lc.index if "dividend" in str(c).lower()]
            net_cols = [c for c in li.index if "net" in str(c).lower() and "income" in str(c).lower()]
            fcf_cols = [c for c in lc.index if "free" in str(c).lower() and "cash" in str(c).lower()]
            if div_cols and net_cols:
                dv, ni = float(lc[div_cols[0]]), float(li[net_cols[0]])
                if ni != 0:
                    payout = abs(dv) / abs(ni)
            if div_cols and fcf_cols:
                dv, fc = float(lc[div_cols[0]]), float(lc[fcf_cols[0]])
                if fc != 0:
                    fcf = abs(dv) / abs(fc)
    except Exception:
        pass
    return payout, fcf


def score_stability(g, y):
    if y <= 1:
        return 0
    r = g / (y - 1)
    if r >= 0.8:
        return 30
    if r >= 0.5:
        return 20
    if r > 0:
        return 10
    return 0


def score_growth(c):
    if c is None or c < 0:
        return 0
    if c < 0.03:
        return 10
    if c < 0.07:
        return 20
    return 25


def score_payout(p):
    if p is None:
        return 10
    if p > 0.8:
        return 0
    if p > 0.6:
        return 10
    if p > 0.4:
        return 15
    return 20


def score_fcf_payout(p):
    return score_payout(p)


def score_yield(y):
    if y is None or y < 0.01:
        return 0
    if y < 0.02:
        return 2
    if y < 0.04:
        return 4
    return 5


def compute_dividend_quality_score(m, p, fcf):
    return {
        "stability": score_stability(m["growth_years"], m["years_available"]),
        "growth": score_growth(m["cagr_5y"]),
        "payout": score_payout(p),
        "fcf_payout": score_fcf_payout(fcf),
        "yield": score_yield(m["current_yield"]),
    }


def radar_score_chart(scores, ticker):
    labels = ["Stabilité", "Croissance", "Payout", "FCF Payout", "Rendement"]
    values = [
        scores["stability"],
        scores["growth"],
        scores["payout"],
        scores["fcf_payout"],
        scores["yield"],
    ]
    values_closed = values + [values[0]]
    labels_closed = labels + [labels[0]]
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(r=values_closed, theta=labels_closed, fill="toself", marker_color="teal")
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 30])),
        title=f"Profil de Qualité du Dividende - {ticker}",
    )
    return fig


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_ticker_dividend_bundle(ticker, last_price):
    series, source = fetch_dividend_series(ticker)
    metrics = compute_dividend_metrics(series, last_price)
    if metrics is None:
        return None, source
    payout, fcf = get_payout_ratios_yahoo(ticker)
    scores = compute_dividend_quality_score(metrics, payout, fcf)
    active = metrics.get("dividend_active", False)
    last_pay = metrics.get("last_payment_date")
    row = {
        "Ticker": ticker,
        "Statut": "Actif" if active else "Suspendu",
        "Prix": last_price,
        "Dividende annuel (TTM)": metrics["last_year_div"] if active else None,
        "Rendement": metrics["current_yield"] if active else None,
        "Dernier versement": (
            pd.Timestamp(last_pay).strftime("%Y-%m-%d") if last_pay is not None else None
        ),
        "DPS dernier versement": metrics.get("last_payment_amount") if not active else None,
        "CAGR 5 ans": metrics["cagr_5y"] if active else None,
        "Années de croissance": metrics["growth_years"] if active else None,
        "Payout": payout,
        "FCF Payout": fcf,
        "Score": (
            scores["stability"] + scores["growth"] + scores["payout"]
            + scores["fcf_payout"] + scores["yield"]
            if active
            else 0
        ),
        "Score stabilité": scores["stability"] if active else 0,
        "Score croissance": scores["growth"] if active else 0,
        "Score payout": scores["payout"] if active else 0,
        "Score FCF": scores["fcf_payout"] if active else 0,
        "Score rendement": scores["yield"] if active else 0,
        "Source": source,
    }
    return row, source


def _analyze_ticker_list(tickers, last_prices, existing_rows, ticker_signature=None):
    existing_tickers = {row["Ticker"] for row in existing_rows}
    pending = [t for t in tickers if t not in existing_tickers]
    source_counts = {"YahooQuery": 0, "yfinance": 0, "Boursorama": 0, "LEcho": 0, "Echecs": 0}

    if not pending:
        return existing_rows, source_counts

    status = st.empty()
    progress = st.progress(0)
    status.info(f"Collecte en cours ({len(pending)} ticker(s))...")

    for i, t in enumerate(pending):
        status.text(f"Analyse : {t} ({i + 1}/{len(pending)})...")
        lp = float(last_prices[t]) if t in last_prices and not np.isnan(last_prices[t]) else None
        if lp is None or lp <= 0:
            source_counts["Echecs"] += 1
            progress.progress((i + 1) / len(pending))
            continue

        if i > 0:
            time.sleep(random.uniform(0.15, 0.35))

        row, source = fetch_ticker_dividend_bundle(t, lp)
        if row is not None:
            existing_rows.append(row)
            source_counts[source or "yfinance"] = source_counts.get(source or "yfinance", 0) + 1
            if ticker_signature:
                save_dividends_rows_to_disk(ticker_signature, existing_rows)
        else:
            source_counts["Echecs"] += 1
        progress.progress((i + 1) / len(pending))

    status.empty()
    progress.empty()
    return existing_rows, source_counts


def _render_dividend_legend():
    """Légende pédagogique des colonnes de la vue Dividendes & Qualité."""
    with st.expander("📖 Légende — que signifient ces indicateurs ?", expanded=False):
        st.markdown(
            """
Les données proviennent de **Yahoo Finance** (cours, dividendes, comptes), avec repli
**Boursorama** / **L'Echo** si nécessaire. Les montants sont **par action**.

---

### Versement actuel

| Terme | Signification |
|-------|----------------|
| **Statut** | **Actif** = au moins un versement sur les **12 derniers mois** (TTM). **Suspendu** = aucun versement récent (l'historique Yahoo peut encore afficher d'anciens dividendes). |
| **TTM** (*Trailing Twelve Months*) | Somme des dividendes **versés sur les 12 mois glissants** à partir d'aujourd'hui — base du rendement affiché. |
| **Dividende annuel (TTM)** | Total des dividendes par action sur les 12 derniers mois. |
| **Rendement** | Dividende TTM ÷ **prix actuel**. Ex. 5 % = 5 € de dividendes pour 100 € investis (sur la base des versements récents). |
| **Dernier versement** | Date du **dernier** paiement enregistré (utile pour repérer une suspension). |
| **DPS dernier versement** | *Dividend Per Share* — montant **par action** du dernier paiement (surtout affiché si le titre est suspendu). |

---

### Croissance et régularité

| Terme | Signification |
|-------|----------------|
| **CAGR 5 ans** | *Compound Annual Growth Rate* — taux de croissance **annualisé** du dividende sur 5 ans. Négatif = dividende en baisse sur la période. |
| **Années de croissance** | Nombre d'années consécutives où le dividende **annuel** a augmenté (ou stagné) par rapport à l'année précédente. |

---

### Couverture du dividende (fondamentaux)

| Terme | Signification |
|-------|----------------|
| **Payout** | *Taux de distribution* — part du **bénéfice net** versée aux actionnaires sous forme de dividendes. Ex. 60 % = 60 centimes de dividende pour 1 € de bénéfice. **> 80 %** = peu de marge de sécurité. |
| **FCF Payout** | Part du **Free Cash Flow** (*flux de trésorerie disponible*) distribuée en dividendes. Mesure si l'entreprise paie avec sa **trésorerie réelle** (pas seulement le résultat comptable). **> 80 %** = risque si le FCF baisse. |
| **FCF** (*Free Cash Flow*) | Trésorerie générée par l'activité **après** investissements — l'« argent disponible » pour dividendes, dette ou croissance. |

---

### Score de qualité (0 à 100)

Le **Score** global est la somme de 5 sous-scores (max ~100). Plus il est élevé, plus le profil dividende est considéré comme solide **sur la base des critères ci-dessous** (titres suspendus = score 0).

| Sous-score | Ce qu'il mesure | Logique simplifiée |
|------------|-----------------|-------------------|
| **Stabilité** (0–30) | Régularité des hausses sur l'historique | Bonus si le dividende augmente souvent d'année en année |
| **Croissance** (0–25) | Dynamique du CAGR 5 ans | Fort si croissance ≥ 7 %/an |
| **Payout** (0–20) | Prudence du taux de distribution | Meilleur entre 40 % et 60 % ; pénalisé si > 80 % |
| **FCF Payout** (0–20) | Même logique, basé sur le free cash flow | Idem payout, appliqué au FCF |
| **Rendement** (0–5) | Niveau du rendement TTM | Léger bonus entre 2 % et 4 % (rendement « raisonnable ») |

---

### Univers dividendes (expander tickers.json)

| Terme | Signification |
|-------|----------------|
| **Fréquence** | Rythme des versements : annuelle, trimestrielle, mensuelle… |
| **Ratio couverture** | Inverse du payout : bénéfices ÷ dividendes (> 1 = couverture suffisante). |
| **Série versement / croissance** | Années consécutives de paiement ou de **hausse** du dividende. |
| **Ex-dividende à venir** | Date limite pour détenir l'action et avoir droit au prochain dividende. |
| **Rendement moy. 5 ans** | Moyenne des rendements calculés année par année sur 5 ans. |
| **Dividende total historique** | Somme de tous les dividendes versés depuis le début de l'historique Yahoo. |
            """
        )
        st.caption(
            "Sources : Yahoo / yfinance / yahooquery. Les ratios payout et FCF dépendent de la "
            "qualité des états financiers Yahoo — peuvent être absents sur certaines valeurs."
        )


def _render_dividend_universe_section(portfolio_tickers=None):
    """Exploration / mise à jour de dividendes_universe.json (tickers.json)."""
    st.markdown("### 🌍 Univers dividendes (tickers.json)")
    universe = load_dividend_universe()
    meta = universe.get("meta", {})
    all_tickers = load_tickers_from_json()
    n_ok = len(universe.get("tickers", {}))
    n_fail = len(universe.get("failed_tickers", {}))
    n_done = n_ok + n_fail
    n_total = len(all_tickers)

    st.caption(f"Fichier local : `{_dividends_universe_path()}`")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tickers cible", n_total)
    c2.metric("Traités", n_done)
    c3.metric("Avec dividendes", meta.get("with_dividends", 0))
    c4.metric("Échecs", n_fail)

    if meta.get("updated_at"):
        st.caption(f"Dernière mise à jour : {meta['updated_at']}")

    batch_size = st.number_input(
        "Taille du lot (mise à jour)",
        min_value=5,
        max_value=500,
        value=25,
        step=5,
        help="Chaque clic traite les tickers pas encore présents dans le fichier univers.",
    )

    if n_total > 0:
        pct_global = min(1.0, n_done / n_total)
        st.progress(
            pct_global,
            text=(
                f"Avancement global : {n_done:,} / {n_total:,} tickers "
                f"({pct_global * 100:.1f} %)"
            ),
        )
        restant = max(0, n_total - n_done)
        if restant:
            lots_restants = (restant + int(batch_size) - 1) // int(batch_size)
            st.caption(
                f"Restant : {restant:,} ticker(s) "
                f"(≈ {lots_restants:,} clic(s) de {int(batch_size)} au lot actuel)."
            )

    if st.button("Mettre à jour l'univers dividendes (lot)", type="secondary"):
        progress_lot = st.progress(0.0, text="Préparation du lot…")

        def _on_batch_progress(pct, ticker, current, batch_len):
            progress_lot.progress(
                min(1.0, pct),
                text=f"Lot en cours : {current}/{batch_len} — {ticker}",
            )

        universe, processed = process_dividend_universe_batch(
            all_tickers,
            universe=universe,
            limit=int(batch_size),
            sleep_seconds=0.2,
            progress_callback=_on_batch_progress,
        )
        progress_lot.progress(1.0, text="Lot terminé.")
        st.success(f"{processed} ticker(s) traité(s) — fichier mis à jour.")
        st.rerun()

    rows = list(universe.get("tickers", {}).values())
    if not rows:
        st.info(
            "L'univers n'est pas encore construit. Cliquez sur **Mettre à jour l'univers dividendes** "
            "ou lancez `python build_dividend_universe.py` en ligne de commande "
            "(5000 tickers ≈ plusieurs heures, reprise automatique)."
        )
        return

    show_cols = [
        "Société",
        "Marché",
        "Devise",
        "Rendement (%)",
        "Fréquence",
        "Croissance 1 an (%)",
        "Croissance 3 ans (%)",
        "Croissance 5 ans (%)",
        "Ratio couverture",
        "Taux versement (%)",
        "Série versement (ans)",
        "Série croissance (ans)",
        "Ex-dividende à venir",
        "Paiement à venir",
        "Rendement moy. 5 ans (%)",
        "Dividende / action",
        "Dividende TTM",
        "Dividende total historique",
        "Source",
    ]

    full_df = _enrich_dividend_display_df(pd.DataFrame(rows))
    only_div = st.checkbox("Afficher seulement les titres avec dividendes", value=True)
    if only_div and "A des dividendes" in full_df.columns:
        full_df = full_df[full_df["A des dividendes"]]

    if portfolio_tickers:
        pf = st.checkbox("Limiter à mon portefeuille / watchlist actuelle", value=False)
        if pf:
            full_df = full_df[full_df["Ticker"].isin(portfolio_tickers)]

    search = st.text_input("Filtrer par société ou ticker")
    if search.strip():
        q = search.strip().casefold()
        mask = full_df["Ticker"].astype(str).str.contains(q, case=False, na=False)
        if "Société" in full_df.columns:
            mask = mask | full_df["Société"].astype(str).str.contains(q, case=False, na=False)
        full_df = full_df[mask]

    df = _dividend_table_column_order(full_df[[c for c in show_cols if c in full_df.columns]])
    df_sorted = df.sort_values("Rendement (%)", ascending=False, na_position="last")
    df_display = rename_columns_for_display(df_sorted, DIVIDEND_UNIVERSE_LABELS)
    universe_format = format_map_for_labeled_columns(
        df_display, DIVIDEND_UNIVERSE_LABELS, DIVIDEND_UNIVERSE_FORMAT
    )
    styled_universe = _style_dividend_sentiment(
        df_display.style.format(universe_format, na_rep="-")
    )
    st.caption(DIVIDEND_UNIVERSE_CAPTION)
    st.caption(
        "Couleurs : vert = favorable, rouge = défavorable "
        "(rendement ≥ 2 %, croissance ≥ 3 %, couverture ≥ 1,5, payout ≤ 60 %, séries ≥ 5 ans)."
    )
    st.dataframe(
        styled_universe,
        use_container_width=True,
        height=min(420, 80 + 35 * max(len(df_display), 1)),
    )

    csv_bytes = df_display.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Télécharger l'univers (CSV)",
        csv_bytes,
        "dividendes_universe.csv",
    )


@st.fragment
def render_dividendes_dashboard(prices, returns):
    st.subheader("Dividendes & Qualité")
    _render_dividend_legend()
    tickers = [str(t).strip() for t in prices.columns]

    with st.expander("Univers dividendes — tickers.json (5000 actions)", expanded=False):
        _render_dividend_universe_section(portfolio_tickers=tickers)

    st.markdown("---")
    last_prices = prices.iloc[-1]
    ticker_signature = "|".join(tickers)
    cache_key = f"data_divs_{abs(hash(ticker_signature))}"

    if cache_key not in st.session_state:
        st.session_state[cache_key] = []

    if _hydrate_session_from_disk(cache_key, ticker_signature):
        updated = st.session_state.get(f"{cache_key}_disk_loaded", "")
        st.success(
            f"📁 {len(st.session_state[cache_key])} résultat(s) rechargé(s) depuis le disque "
            f"(cache local, dernière analyse : {updated})."
        )
        st.caption(f"Fichier : `{_dividends_cache_path()}`")

    use_batches = len(tickers) > 20
    tickers_to_analyze = tickers

    if use_batches:
        max_default = min(50, len(tickers))
        max_to_analyze = st.slider(
            "Nombre max de tickers à analyser",
            min_value=10,
            max_value=len(tickers),
            value=min(max_default, len(tickers)),
            step=10,
        )
        batch_options = [b for b in [25, 50, 75, 100] if b <= max(len(tickers), 1)]
        if not batch_options:
            batch_options = [len(tickers)]
        default_batch_idx = 1 if len(batch_options) > 1 else 0
        batch_size = st.selectbox("Taille de lot", batch_options, index=default_batch_idx)
        limited = tickers[:max_to_analyze]
        total_batches = max(1, int(np.ceil(len(limited) / batch_size)))
        batch_key = f"{cache_key}_batch"
        if batch_key not in st.session_state:
            st.session_state[batch_key] = 0
        st.session_state[batch_key] = min(st.session_state[batch_key], total_batches - 1)
        batch_idx = st.session_state[batch_key]
        start_i = batch_idx * batch_size
        end_i = min(start_i + batch_size, len(limited))
        tickers_to_analyze = limited[start_i:end_i]
        st.caption(
            f"Lot {batch_idx + 1}/{total_batches} — tickers {start_i + 1} à {end_i} "
            f"sur {len(limited)}."
        )
        c1, c2, c3 = st.columns(3)
        analyze_clicked = c1.button("Analyser ce lot")
        next_clicked = c2.button("Lot suivant")
        reset_clicked = c3.button("Recommencer")
        if next_clicked and batch_idx < total_batches - 1:
            st.session_state[batch_key] = batch_idx + 1
            st.rerun()
    else:
        analyze_clicked = st.button("Analyser le portefeuille", type="primary")
        next_clicked = False
        reset_clicked = st.button("Vider le cache dividendes")
        st.caption(f"{len(tickers)} ticker(s) — analyse directe sans lot.")

    if reset_clicked:
        fetch_ticker_dividend_bundle.clear()
        clear_dividends_disk_cache(ticker_signature)
        try:
            from dashboard_cache import clear_dashboard_compute_cache

            clear_dashboard_compute_cache()
        except ImportError:
            pass
        if cache_key in st.session_state:
            del st.session_state[cache_key]
        disk_loaded_key = f"{cache_key}_disk_loaded"
        if disk_loaded_key in st.session_state:
            del st.session_state[disk_loaded_key]
        batch_key = f"{cache_key}_batch"
        if batch_key in st.session_state:
            del st.session_state[batch_key]
        st.rerun()

    if analyze_clicked and tickers_to_analyze:
        rows, source_counts = _analyze_ticker_list(
            tickers_to_analyze,
            last_prices,
            st.session_state[cache_key],
            ticker_signature=ticker_signature,
        )
        st.session_state[cache_key] = rows
        save_dividends_rows_to_disk(ticker_signature, rows)
        st.caption(f"💾 Résultats enregistrés localement : `{_dividends_cache_path()}`")
        st.info(
            "Lot terminé — "
            f"YahooQuery: {source_counts['YahooQuery']}, "
            f"yfinance: {source_counts['yfinance']}, "
            f"Boursorama: {source_counts['Boursorama']}, "
            f"L'Echo: {source_counts['LEcho']}, "
            f"échecs: {source_counts['Echecs']}"
        )

    rows = st.session_state[cache_key]
    if not rows:
        st.info(
            "Cliquez sur **Analyser le portefeuille** (ou **Analyser ce lot**) "
            "pour lancer la collecte des dividendes."
        )
        if st.button("Forcer une nouvelle tentative"):
            fetch_ticker_dividend_bundle.clear()
            clear_dividends_disk_cache(ticker_signature)
            if cache_key in st.session_state:
                del st.session_state[cache_key]
            disk_loaded_key = f"{cache_key}_disk_loaded"
            if disk_loaded_key in st.session_state:
                del st.session_state[disk_loaded_key]
            st.rerun()
        return

    df = pd.DataFrame(rows).sort_values("Score", ascending=False)
    df = _enrich_dividend_display_df(df)
    df_view = _dividend_table_column_order(df)
    df_display = rename_columns_for_display(df_view, DIVIDEND_PORTFOLIO_LABELS)

    money_cols = [
        "Prix (/ action)",
        "Dividende annuel TTM (/ action)",
        "Dividende annuel (/ action)",
        "DPS dernier versement (/ action)",
    ]
    format_map = format_map_for_labeled_columns(
        df_display, DIVIDEND_PORTFOLIO_LABELS, DIVIDEND_PORTFOLIO_FORMAT_BASE
    )
    for col in money_cols:
        if col in df_display.columns:
            format_map[col] = "{:.2f}"

    styled_portfolio = _style_dividend_sentiment(
        df_display.style.format(format_map, na_rep="-")
    )
    score_col = "Score qualité (pts / ~100)"
    styled_portfolio = styled_portfolio.background_gradient(
        cmap="RdYlGn",
        subset=[c for c in [score_col, "Score"] if c in df_display.columns],
    )
    st.caption(DIVIDEND_PORTFOLIO_CAPTION)
    st.caption(
        "Couleurs : vert = favorable, rouge = défavorable sur rendement, croissance, payout et couverture."
    )
    st.dataframe(styled_portfolio)

    suspended = (
        df_display[df_display.get("Statut", pd.Series(dtype=str)) == "Suspendu"]
        if "Statut" in df_display.columns
        else pd.DataFrame()
    )
    if not suspended.empty:
        st.caption(
            "ℹ️ Titres **Suspendu** : aucun versement sur les 12 derniers mois — "
            "le rendement affiché est vide (pas l'historique Yahoo)."
        )

    csv = df_display.to_csv(index=False).encode("utf-8")
    st.download_button("Télécharger le rapport (CSV)", csv, "dividendes.csv")

    st.write("---")
    fig = px.scatter(
        df,
        x="Rendement",
        y="Score",
        text="Société",
        size="Score",
        color="Score",
        color_continuous_scale="RdYlGn",
        title="Analyse Risque / Qualité : Rendement vs Score Global",
        hover_data={"Ticker": True, "Marché": True, "Devise": True},
    )
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)

    st.write("---")
    label_map = dict(zip(df["Ticker"], df["Société"]))
    tsel = st.selectbox(
        "Sélectionner un actif pour voir le détail :",
        df["Ticker"],
        format_func=lambda t: f"{label_map.get(t, t)} ({t})",
    )
    r = df[df["Ticker"] == tsel].iloc[0]
    sc = {
        "stability": r["Score stabilité"],
        "growth": r["Score croissance"],
        "payout": r["Score payout"],
        "fcf_payout": r["Score FCF"],
        "yield": r["Score rendement"],
    }
    st.plotly_chart(radar_score_chart(sc, label_map.get(tsel, tsel)), use_container_width=True)
