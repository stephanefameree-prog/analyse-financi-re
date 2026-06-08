"""Importe les 10 tickers WarrenAI (Investing.com) dans dividendes_universe.json."""
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import dividendes as div

TICKERS = ["QFIN", "MRP", "PLTK", "DEC", "INSW", "GSL", "WEN", "WLKP", "ASC", "OMF"]

LISTINGS = {
    "QFIN": {"Société": "Qfin Holdings, Inc.", "Marché": "NASDAQ"},
    "MRP": {"Société": "Millrose Properties, Inc.", "Marché": "NYSE"},
    "PLTK": {"Société": "Playtika Holding Corp.", "Marché": "NASDAQ"},
    "DEC": {"Société": "Diversified Energy Company PLC", "Marché": "NYSE"},
    "INSW": {"Société": "International Seaways, Inc.", "Marché": "NYSE"},
    "GSL": {"Société": "Global Ship Lease, Inc.", "Marché": "NYSE"},
    "WEN": {"Société": "The Wendy's Company", "Marché": "NASDAQ"},
    "WLKP": {"Société": "Westlake Chemical Partners LP", "Marché": "NYSE"},
    "ASC": {"Société": "Ardmore Shipping Corporation", "Marché": "NYSE"},
    "OMF": {"Société": "OneMain Holdings, Inc.", "Marché": "NYSE"},
}

# Rendement ( décimal ) et couverture (×) — source WarrenAI / Investing.com, 2026-06-08
WARREN = {
    "QFIN": (0.459, 7.5),
    "MRP": (0.364, 8.9),
    "PLTK": (0.362, 3.8),
    "DEC": (0.146, 6.1),
    "INSW": (0.243, 19.0),
    "GSL": (0.489, 5.5),
    "WEN": (0.206, 3.0),
    "WLKP": (0.299, 5.2),
    "ASC": (0.330, 5.6),
    "OMF": (0.177, 10.3),
}


def _warren_fallback_entry(ticker):
    wy, wc = WARREN[ticker]
    info = LISTINGS[ticker]
    return {
        "Ticker": ticker,
        "Société": info["Société"],
        "Marché": info["Marché"],
        "Devise": "USD",
        "Rendement (%)": wy,
        "Ratio couverture": wc,
        "Taux versement (%)": (1.0 / wc) if wc else None,
        "A des dividendes": True,
        "Statut": "Actif",
        "Source": "warrenai/investing.com",
        "Fréquence": None,
    }


def main():
    universe = div.load_dividend_universe()
    tickers_map = universe.setdefault("tickers", {})
    failed = universe.setdefault("failed_tickers", {})

    for ticker in TICKERS:
        listing = LISTINGS[ticker]
        try:
            row = div._fetch_dividend_profile_with_timeout(
                ticker, lite=False, timeout_seconds=60
            )
            if not isinstance(row, dict):
                raise RuntimeError("profil vide")
            entry = div._cache_row_to_universe_entry(
                row,
                listing={
                    "Société": listing["Société"],
                    "Secteur": "—",
                    "Marché": listing["Marché"],
                    "Devise": "USD",
                },
                source="yfinance+warrenai",
            )
            if entry is None:
                raise RuntimeError("conversion échouée")
            wy, wc = WARREN[ticker]
            if entry.get("Rendement (%)") is None:
                entry["Rendement (%)"] = wy
            if entry.get("Ratio couverture") is None:
                entry["Ratio couverture"] = wc
            entry["A des dividendes"] = True
            entry["Statut"] = "Actif"
            entry["Ticker"] = ticker
            entry["Société"] = listing["Société"]
            entry["Marché"] = listing["Marché"]
            entry["Devise"] = "USD"
            existing = tickers_map.get(ticker, {})
            if existing:
                entry = div._merge_universe_entries(existing, entry)
            tickers_map[ticker] = entry
            div._drop_failed_ticker_variants(failed, ticker)
            print(
                f"OK {ticker}: yield={entry.get('Rendement (%)')} "
                f"coverage={entry.get('Ratio couverture')}"
            )
        except Exception as exc:
            entry = _warren_fallback_entry(ticker)
            existing = tickers_map.get(ticker, {})
            if existing:
                entry = div._merge_universe_entries(existing, entry)
            tickers_map[ticker] = entry
            div._drop_failed_ticker_variants(failed, ticker)
            print(f"FALLBACK {ticker}: {exc}")

    div._refresh_universe_meta(universe)
    universe["meta"]["updated_at"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    universe["meta"]["total_target"] = len(div.load_tickers_from_json())
    div.save_dividend_universe(universe)
    print(f"Enregistré — {len(TICKERS)} tickers WarrenAI dans dividendes_universe.json")


if __name__ == "__main__":
    main()
