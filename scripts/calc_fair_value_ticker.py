"""Calcule la juste valeur d'un ticker (CLI). Usage: python scripts/calc_fair_value_ticker.py MRVL"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fundamentals import fetch_fundamentals_for_ticker  # noqa: E402
import fair_value as fv  # noqa: E402


def main(ticker: str) -> None:
    ticker = ticker.strip().upper()
    info = yf.Ticker(ticker).info or {}
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    name = info.get("shortName") or info.get("longName") or ticker
    print(f"Titre: {name} ({ticker})")
    print(f"Cours Yahoo: {price} USD")

    row = fetch_fundamentals_for_ticker(ticker, float(price) if price else None, throttle=False)
    if not row:
        print("ERREUR: fondamentaux indisponibles")
        sys.exit(1)

    models, meta = fv.load_dashboard_fair_value_models(ROOT, enrich_yahoo_price=False)
    unified = meta.get(fv.SECTOR_UNIFIED, {})
    mape = unified.get("cv_mape", "n/a")
    print(f"Modele unifie: {unified.get('status')} (MAPE CV: {mape})")

    enrich: dict = {}
    ma_df = fv.try_load_ma_vue_for_dashboard(ROOT, enrich_yahoo_price=False)
    if ma_df is not None:
        enrich[fv.SECTOR_MINIERES] = fv.build_ma_vue_yahoo_index(ma_df)
    tech_df = fv.try_load_tech_ma_vue_for_dashboard(ROOT, enrich_yahoo_price=False)
    if tech_df is not None:
        enrich[fv.SECTOR_TECH] = fv.build_ma_vue_yahoo_index(tech_df)

    pred = fv.predict_fair_value_for_fundamentals([row], models, enrich)
    if pred.empty:
        print("ERREUR: prediction vide (donnees insuffisantes pour le modele)")
        sys.exit(1)

    p = pred.iloc[0]
    jv = float(p["Juste valeur estimée"])
    upside = float(p["Upside juste valeur"])
    label = p["Libellé juste valeur"]
    damo_jv = p.get("Juste valeur Damodaran")
    damo_up = p.get("Upside Damodaran")

    print()
    print("=== JUSTE VALEUR (modele Ridge unifie) ===")
    print(f"Juste valeur estimee : {jv:.2f} USD")
    print(f"Cours actuel         : {float(price):.2f} USD")
    print(f"Ecart                : {jv - float(price):+.2f} USD")
    print(f"Upside               : {upside:+.1%}")
    print(f"Libelle              : {label}")
    if pd.notna(damo_jv):
        print()
        print("=== Damodaran (multiples sectoriels) ===")
        print(f"Juste valeur Damodaran : {float(damo_jv):.2f} USD")
        if pd.notna(damo_up):
            print(f"Upside Damodaran       : {float(damo_up):+.1%}")
        print(f"Profil                 : {p.get('Profil Damodaran', '—')}")

    print()
    print("=== Ratios cles (Yahoo) ===")
    keys = [
        "PER",
        "PER forward",
        "VE/EBITDA",
        "Rendement FCF",
        "Ratio PEG",
        "Objectif analystes",
        "Upside vs objectif",
        "ROIC",
        "Score Piotroski",
        "Secteur Yahoo",
    ]
    for k in keys:
        v = row.get(k)
        if v is None:
            continue
        if k in ("Upside vs objectif", "Rendement FCF") and isinstance(v, (int, float)):
            print(f"  {k}: {v:.2%}")
        elif isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    tk = sys.argv[1] if len(sys.argv) > 1 else "MRVL"
    main(tk)
