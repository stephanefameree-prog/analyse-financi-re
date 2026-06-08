"""Benchmark des sources de données du dashboard (latence réseau réelle)."""
from __future__ import annotations

import os
import statistics
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from yahooquery import Ticker

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loader import get_fmp_api_key  # noqa: E402
from fundamentals import fetch_fundamentals_for_ticker  # noqa: E402

QUERIES = ["apple", "LVMH", "AAPL", "MC.PA"]
TICKERS = [
    "AAPL",
    "MSFT",
    "MC.PA",
    "EDEN.PA",
    "TSLA",
    "NVDA",
    "OR.PA",
    "SAN.PA",
    "BNP.PA",
    "AIR.PA",
]
FUND_TICKERS = TICKERS[:3]
OHLCV_START = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")

YAHOO_SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"
FMP_SEARCH_URL = "https://financialmodelingprep.com/api/v3/search"
FMP_PROFILE_URL = "https://financialmodelingprep.com/api/v3/profile"

RESULTS: list[dict] = []


def bench(label: str, category: str, fn, runs: int = 3) -> dict:
    times_ms: list[float] = []
    hits = 0
    last_error = ""
    for _ in range(runs):
        t0 = time.perf_counter()
        try:
            result = fn()
            hits += 1 if result is not None and result is not False else 0
        except Exception as exc:
            last_error = str(exc)
            result = None
        times_ms.append((time.perf_counter() - t0) * 1000)
    times_ms.sort()
    row = {
        "label": label,
        "category": category,
        "median_ms": statistics.median(times_ms),
        "mean_ms": statistics.mean(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "hits": hits,
        "runs": runs,
    }
    if last_error and hits == 0:
        row["error"] = last_error
    RESULTS.append(row)
    return row


def fmt_row(row: dict) -> str:
    if row.get("error"):
        return f"  {row['label']}: ERREUR — {row['error']}"
    return (
        f"  {row['label']}: med={row['median_ms']:.0f} ms | "
        f"moy={row['mean_ms']:.0f} ms | succès={row['hits']}/{row['runs']}"
    )


# --- Yahoo HTTP search ---
def yahoo_search(query: str) -> int:
    r = requests.get(
        YAHOO_SEARCH_URL,
        params={"q": query, "quotesCount": 8, "newsCount": 0, "enableFuzzyQuery": True},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=12,
    )
    r.raise_for_status()
    quotes = (r.json() or {}).get("quotes") or []
    return len(quotes)


# --- FMP ---
def fmp_search(query: str, key: str) -> int:
    r = requests.get(
        FMP_SEARCH_URL,
        params={"query": query, "limit": 8, "apikey": key},
        timeout=12,
    )
    r.raise_for_status()
    payload = r.json()
    if isinstance(payload, dict) and payload.get("Error Message"):
        raise RuntimeError(payload["Error Message"])
    return len(payload) if isinstance(payload, list) else 0


def fmp_profile_batch(tickers: list[str], key: str) -> int:
    r = requests.get(
        f"{FMP_PROFILE_URL}/{','.join(tickers)}",
        params={"apikey": key},
        timeout=15,
    )
    r.raise_for_status()
    payload = r.json()
    if isinstance(payload, dict) and payload.get("Error Message"):
        raise RuntimeError(payload["Error Message"])
    rows = payload if isinstance(payload, list) else [payload]
    return sum(1 for x in rows if isinstance(x, dict) and x.get("companyName"))


# --- Yahooquery ---
def yq_price_batch(tickers: list[str]) -> int:
    payload = Ticker(tickers, timeout=25).price
    return sum(
        1
        for tk in tickers
        if isinstance(payload.get(tk), dict)
        and (payload[tk].get("shortName") or payload[tk].get("longName"))
    )


def yq_sectors_batch(tickers: list[str]) -> int:
    payload = Ticker(tickers, timeout=25).asset_profile
    return sum(
        1
        for tk in tickers
        if isinstance(payload.get(tk), dict) and payload[tk].get("sector")
    )


def yq_metadata_unified(tickers: list[str]) -> int:
    """Même logique que get_ticker_metadata (noms + secteurs, une passe)."""
    yq = Ticker(tickers, timeout=25)
    price = yq.price
    profile = yq.asset_profile
    n_names = sum(
        1
        for tk in tickers
        if isinstance(price.get(tk), dict)
        and (price[tk].get("shortName") or price[tk].get("longName"))
    )
    n_sectors = sum(
        1
        for tk in tickers
        if isinstance(profile.get(tk), dict) and profile[tk].get("sector")
    )
    return n_names + n_sectors


def yq_sequential_price(tickers: list[str]) -> int:
    n = 0
    for tk in tickers:
        payload = Ticker([tk], timeout=25).price
        if isinstance(payload.get(tk), dict) and (
            payload[tk].get("shortName") or payload[tk].get("longName")
        ):
            n += 1
    return n


# --- yfinance cours ---
def yf_download_batch(tickers: list[str]) -> int:
    df = yf.download(
        tickers,
        start=OHLCV_START,
        group_by="ticker",
        progress=False,
        threads=False,
    )
    return len(df.columns) if not df.empty else 0


def yf_download_sequential(tickers: list[str]) -> int:
    ok = 0
    for tk in tickers:
        df = yf.download(tk, start=OHLCV_START, progress=False, threads=False)
        if not df.empty:
            ok += 1
    return ok


# --- yfinance fondamentaux / dividendes ---
def yf_info_batch(tickers: list[str]) -> int:
    n = 0
    for tk in tickers:
        info = yf.Ticker(tk).info or {}
        if info.get("shortName") or info.get("longName"):
            n += 1
    return n


def yf_dividends_batch(tickers: list[str]) -> int:
    n = 0
    for tk in tickers:
        div = yf.Ticker(tk).dividends
        if div is not None and len(div) > 0:
            n += 1
    return n


def fundamentals_stack(tickers: list[str]) -> int:
    n = 0
    for tk in tickers:
        row = fetch_fundamentals_for_ticker(tk, 100.0, throttle=False)
        if row and row.get("PER") is not None:
            n += 1
    return n


def main() -> None:
    fmp_key = get_fmp_api_key()
    n = len(TICKERS)
    nf = len(FUND_TICKERS)

    print("=" * 72)
    print("BENCHMARK SOURCES DE DONNÉES — dashboard analyse financière")
    print("=" * 72)
    print(f"Date        : {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Tickers     : {n} ({', '.join(TICKERS[:4])}…)")
    print(f"OHLCV depuis: {OHLCV_START}")
    print(f"Clé FMP     : {'oui' if fmp_key else 'non (FMP_API_KEY)'}")
    print()

    print("--- 1. Recherche symbole (watchlist) ---")
    for q in QUERIES:
        row = bench(f"Yahoo search « {q} »", "recherche", lambda q=q: yahoo_search(q), runs=5)
        print(fmt_row(row))
        if fmp_key:
            row = bench(f"FMP search « {q} »", "recherche", lambda q=q: fmp_search(q, fmp_key), runs=5)
            print(fmt_row(row))

    print()
    print(f"--- 2. Métadonnées tickers (lot {n}) ---")
    for label, fn in [
        ("Yahoo yahooquery · price (noms)", lambda: yq_price_batch(TICKERS)),
        ("Yahoo yahooquery · asset_profile (secteurs)", lambda: yq_sectors_batch(TICKERS)),
        ("Yahoo yahooquery · noms+secteurs (1 passe)", lambda: yq_metadata_unified(TICKERS)),
        ("Yahoo yahooquery · noms séquentiels (×10)", lambda: yq_sequential_price(TICKERS)),
    ]:
        row = bench(label, "metadata", fn, runs=3)
        print(fmt_row(row))
    if fmp_key:
        row = bench(f"FMP profile (lot {n})", "metadata", lambda: fmp_profile_batch(TICKERS, fmp_key), runs=3)
        print(fmt_row(row))
    else:
        print("  FMP profile : non testé (pas de clé)")

    print()
    print(f"--- 3. Cours OHLCV ({n} tickers) ---")
    row_batch = bench(f"yfinance.download lot ({n})", "ohlcv", lambda: yf_download_batch(TICKERS), runs=2)
    print(fmt_row(row_batch))
    row_seq = bench(f"yfinance.download séquentiel ({n}×)", "ohlcv", lambda: yf_download_sequential(TICKERS), runs=1)
    print(fmt_row(row_seq))

    print()
    print(f"--- 4. Fondamentaux / dividendes ({nf} tickers échantillon) ---")
    for label, fn in [
        ("yfinance · .info séquentiel", lambda: yf_info_batch(FUND_TICKERS)),
        ("yfinance · .dividends séquentiel", lambda: yf_dividends_batch(FUND_TICKERS)),
        ("fundamentals.py · fetch complet", lambda: fundamentals_stack(FUND_TICKERS)),
    ]:
        row = bench(label, "fundamentaux", fn, runs=2)
        print(fmt_row(row))

    print()
    print("--- 5. Simulation 1er chargement watchlist (10 titres, estimé) ---")
    meta_med = next(r["median_ms"] for r in RESULTS if "noms+secteurs" in r["label"])
    ohlcv_med = row_batch["median_ms"]
    fund_med = next(r["median_ms"] for r in RESULTS if "fetch complet" in r["label"])
    fund_est_10 = fund_med * (n / nf)
    total_est = ohlcv_med + meta_med + fund_est_10
    print(f"  OHLCV batch (mesuré)           : {ohlcv_med:,.0f} ms")
    print(f"  Métadonnées noms+secteurs      : {meta_med:,.0f} ms")
    print(f"  Fondamentaux ×10 (extrapolé)   : {fund_est_10:,.0f} ms")
    print(f"  TOTAL estimé (sans cache)      : {total_est:,.0f} ms (~{total_est/1000:.1f} s)")
    print("  (Cache Streamlit 1–24 h réduit fortement les rechargements suivants.)")

    print()
    print("--- Classement par mediane (plus rapide -> plus lent) ---")
    ok_rows = [r for r in RESULTS if not r.get("error")]
    ok_rows.sort(key=lambda r: r["median_ms"])
    for i, row in enumerate(ok_rows[:15], 1):
        print(f"  {i:2d}. {row['median_ms']:7.0f} ms  [{row['category']:12s}]  {row['label']}")

    print()
    print("--- Recommandations ---")
    y_search = statistics.median(
        r["median_ms"] for r in RESULTS if r["category"] == "recherche" and r["label"].startswith("Yahoo")
    )
    print(f"• Recherche watchlist : Yahoo (~{y_search:.0f} ms) en priorité, FMP en repli uniquement.")
    print(f"• Noms + secteurs : une passe yahooquery (~{meta_med:.0f} ms), pas deux appels séparés au 1er écran.")
    if row_seq["median_ms"] > row_batch["median_ms"] * 2:
        print(
            f"• Cours : lot yfinance ({row_batch['median_ms']:.0f} ms) "
            f"≈ {row_seq['median_ms']/row_batch['median_ms']:.0f}× plus rapide que ticker par ticker."
        )
    print(f"• Goulot principal : fondamentaux (~{fund_est_10/1000:.1f} s pour 10 titres) — cache 24 h essentiel.")
    if fmp_key:
        fmp_prof = next((r for r in RESULTS if "FMP profile" in r["label"]), None)
        if fmp_prof and not fmp_prof.get("error"):
            y_names = next(r for r in RESULTS if "price (noms)" in r["label"])
            if fmp_prof["median_ms"] < y_names["median_ms"]:
                print("• FMP profile plus rapide que yahooquery price pour les noms (repli FMP pertinent).")
            else:
                print("• Yahoo yahooquery reste plus rapide que FMP profile pour les noms sur ce réseau.")


if __name__ == "__main__":
    main()
