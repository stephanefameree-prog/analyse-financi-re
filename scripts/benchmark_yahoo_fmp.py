"""Compare latence Yahoo vs FMP pour optimiser le chargement du dashboard."""
from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

import requests
from yahooquery import Ticker

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loader import get_fmp_api_key  # noqa: E402

QUERIES = ["apple", "LVMH", "edenred", "AAPL", "MC.PA"]
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
RUNS_SEARCH = 5
RUNS_BATCH = 3

YAHOO_SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"
FMP_SEARCH_URL = "https://financialmodelingprep.com/api/v3/search"
FMP_PROFILE_URL = "https://financialmodelingprep.com/api/v3/profile"


def bench(fn, runs: int) -> dict:
    times_ms: list[float] = []
    hits = 0
    last_error = ""
    for _ in range(runs):
        t0 = time.perf_counter()
        try:
            result = fn()
            hits += 1 if result else 0
        except Exception as exc:
            last_error = str(exc)
            result = None
        times_ms.append((time.perf_counter() - t0) * 1000)
    times_ms.sort()
    row = {
        "median_ms": statistics.median(times_ms),
        "mean_ms": statistics.mean(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "hits": hits,
        "runs": runs,
    }
    if last_error and hits == 0:
        row["error"] = last_error
    return row


def yahoo_search(query: str) -> int:
    response = requests.get(
        YAHOO_SEARCH_URL,
        params={
            "q": query,
            "quotesCount": 8,
            "newsCount": 0,
            "enableFuzzyQuery": True,
        },
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=12,
    )
    response.raise_for_status()
    payload = response.json()
    quotes = payload.get("quotes") if isinstance(payload, dict) else None
    return len(quotes or [])


def fmp_search(query: str, api_key: str) -> int:
    response = requests.get(
        FMP_SEARCH_URL,
        params={"query": query, "limit": 8, "apikey": api_key},
        timeout=12,
    )
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and payload.get("Error Message"):
        raise RuntimeError(str(payload["Error Message"]))
    return len(payload) if isinstance(payload, list) else 0


def yahoo_names_batch(tickers: list[str]) -> int:
    payload = Ticker(tickers, timeout=25).price
    count = 0
    if isinstance(payload, dict):
        for tk in tickers:
            block = payload.get(tk)
            if isinstance(block, dict) and (block.get("shortName") or block.get("longName")):
                count += 1
    return count


def fmp_names_batch(tickers: list[str], api_key: str) -> int:
    response = requests.get(
        f"{FMP_PROFILE_URL}/{','.join(tickers)}",
        params={"apikey": api_key},
        timeout=12,
    )
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and payload.get("Error Message"):
        raise RuntimeError(str(payload["Error Message"]))
    rows = payload if isinstance(payload, list) else [payload]
    return sum(
        1
        for item in rows
        if isinstance(item, dict) and (item.get("companyName") or item.get("name"))
    )


def yahoo_sectors_batch(tickers: list[str]) -> int:
    payload = Ticker(tickers, timeout=25).asset_profile
    return sum(
        1
        for tk in tickers
        if isinstance(payload.get(tk), dict) and payload[tk].get("sector")
    )


def fmt_row(label: str, row: dict) -> str:
    if row.get("error"):
        return f"  {label}: ERREUR — {row['error']}"
    return (
        f"  {label}: med={row['median_ms']:.0f} ms | "
        f"moy={row['mean_ms']:.0f} ms | "
        f"min={row['min_ms']:.0f} ms | "
        f"max={row['max_ms']:.0f} ms | "
        f"succès={row['hits']}/{row['runs']}"
    )


def main() -> None:
    fmp_key = get_fmp_api_key()
    key_source = "FMP_API_KEY définie" if fmp_key else "aucune clé FMP"

    print("=== Benchmark Yahoo vs FMP ===")
    print(f"Clé FMP : {key_source}")
    print(f"Requêtes testées : {QUERIES}")
    print(f"Tickers batch : {len(TICKERS)}")
    print()

    yahoo_rows: list[dict] = []
    fmp_rows: list[dict] = []

    print("--- Recherche par nom / ticker ---")
    for query in QUERIES:
        y_row = bench(lambda q=query: yahoo_search(q), RUNS_SEARCH)
        y_row["label"] = f'Yahoo "{query}"'
        yahoo_rows.append(y_row)
        print(fmt_row(y_row["label"], y_row))

        if fmp_key:
            f_row = bench(lambda q=query: fmp_search(q, fmp_key), RUNS_SEARCH)
            f_row["label"] = f'FMP "{query}"'
            fmp_rows.append(f_row)
            print(fmt_row(f_row["label"], f_row))
        print()

    print("--- Noms société (lot) ---")
    y_names = bench(lambda: yahoo_names_batch(TICKERS), RUNS_BATCH)
    y_names["label"] = f"Yahoo yahooquery ({len(TICKERS)} tickers)"
    print(fmt_row(y_names["label"], y_names))

    f_names: dict | None = None
    if fmp_key:
        f_names = bench(lambda: fmp_names_batch(TICKERS, fmp_key), RUNS_BATCH)
        f_names["label"] = f"FMP profile ({len(TICKERS)} tickers)"
        print(fmt_row(f_names["label"], f_names))
    else:
        print("  FMP profile : ignoré (pas de clé API)")

    print()
    print("--- Secteurs Yahoo (lot, référence dashboard) ---")
    y_sectors = bench(lambda: yahoo_sectors_batch(TICKERS), 2)
    y_sectors["label"] = f"Yahoo asset_profile ({len(TICKERS)} tickers)"
    print(fmt_row(y_sectors["label"], y_sectors))

    print()
    print("--- Synthèse ---")
    y_search_med = statistics.median(r["median_ms"] for r in yahoo_rows)
    print(f"Recherche Yahoo (médiane globale) : {y_search_med:.0f} ms")

    fmp_ok = [r for r in fmp_rows if not r.get("error")]
    if fmp_ok:
        f_search_med = statistics.median(r["median_ms"] for r in fmp_ok)
        ratio = f_search_med / y_search_med if y_search_med else 0
        faster = "Yahoo" if y_search_med < f_search_med else "FMP"
        print(f"Recherche FMP (médiane globale)   : {f_search_med:.0f} ms")
        print(f"Plus rapide en recherche          : {faster} ({ratio:.2f}x vs l'autre)")
    elif fmp_key:
        print("Recherche FMP                       : échec (clé invalide ou quota)")
    else:
        print("Recherche FMP                       : non mesurée (définir FMP_API_KEY)")

    if fmp_key and f_names and not f_names.get("error") and not y_names.get("error"):
        y_batch = y_names["median_ms"]
        f_batch = f_names["median_ms"]
        batch_faster = "Yahoo" if y_batch < f_batch else "FMP"
        print(f"Noms batch Yahoo                  : {y_batch:.0f} ms")
        print(f"Noms batch FMP                    : {f_batch:.0f} ms")
        print(f"Plus rapide en noms batch         : {batch_faster}")

    if not y_sectors.get("error"):
        print(f"Secteurs batch Yahoo              : {y_sectors['median_ms']:.0f} ms")

    print()
    print("Recommandations actuelles :")
    print("- Garder Yahoo en source primaire (recherche watchlist + noms batch).")
    print("- FMP uniquement en repli si Yahoo ne renvoie rien (évite double appel).")
    if fmp_key and fmp_rows and statistics.median(r["median_ms"] for r in fmp_rows) < y_search_med:
        print("- FMP semble plus rapide en recherche : envisager parallélisation, pas remplacement total.")
    if not fmp_key:
        print("- Définir FMP_API_KEY pour mesurer le repli FMP sur votre réseau.")


if __name__ == "__main__":
    main()
