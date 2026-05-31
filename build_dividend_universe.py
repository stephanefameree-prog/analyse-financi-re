#!/usr/bin/env python3
"""Construit dividendes_universe.json pour tous les tickers de tickers.json."""
import argparse
import sys
import time

from dividendes import (
    load_dividend_universe,
    load_tickers_from_json,
    process_dividend_universe_batch,
)


def main():
    parser = argparse.ArgumentParser(
        description="Collecte les indicateurs dividendes pour l'univers tickers.json"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Nombre max de tickers à traiter dans cette exécution",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.25,
        help="Pause moyenne entre tickers (secondes)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=20,
        help="Sauvegarde sur disque tous les N tickers",
    )
    args = parser.parse_args()

    tickers = load_tickers_from_json()
    if not tickers:
        print("Aucun ticker trouvé dans tickers.json", file=sys.stderr)
        sys.exit(1)

    universe = load_dividend_universe()
    already = len(universe.get("tickers", {})) + len(universe.get("failed_tickers", {}))
    remaining = len(tickers) - already

    print(f"Univers cible : {len(tickers)} tickers")
    print(f"Déjà traités : {already} | Restants : {remaining}")
    if args.limit:
        print(f"Lot demandé : {args.limit}")

    started = time.time()
    universe, processed = process_dividend_universe_batch(
        tickers,
        universe=universe,
        limit=args.limit,
        sleep_seconds=args.sleep,
        save_every=args.save_every,
    )
    elapsed = time.time() - started
    meta = universe.get("meta", {})
    print(
        f"Terminé : {processed} ticker(s) traités en {elapsed:.0f}s | "
        f"OK={len(universe.get('tickers', {}))} | "
        f"Échecs={len(universe.get('failed_tickers', {}))} | "
        f"Avec dividendes={meta.get('with_dividends', 0)}"
    )


if __name__ == "__main__":
    main()
