#!/usr/bin/env python3
"""Vérifie des listes screener et affiche les tickers à ajouter."""
import json
import sys
from pathlib import Path

import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
TICKERS_PATH = ROOT / "tickers.json"
UNIV_PATH = ROOT / "dividendes_universe.json"

CANDIDATES = [
    # list 1
    ("Europlasma", "ALESA.PA", 0.003),
    ("Neovacs", "ALNEV.PA", 0.0002),
    ("Capital B", "ALCBI.PA", 0.4151),
    ("Drone Volt", "ALDRV.PA", 0.428),
    ("AB Science", "AB.PA", 0.90),
    ("Trigano", "TRI.PA", 134.90),
    ("Europacorp", "ALECP.PA", 0.304),
    ("Wavestone", "WAVE.PA", 39.75),
    ("Sword Group", "SWP.PA", 30.00),
    ("Artmarket", "ALART.PA", 2.27),
    ("Poujoulat", "ALPOU.PA", 6.30),
    ("Odysse Tech", "ALODY.PA", 18.62),
    ("Odiot Holding", "ALODI.PA", 23.0),
    ("Locasystem", "ALLOI.PA", 10.10),
    ("Jacques Bogart", "JBOG.PA", 2.28),
    ("Groupe Crit", "CRI.PA", 53.60),
    # list 2
    ("Exel Industries", "EXEL.PA", 21.80, ["EXA.PA"]),
    ("Augros Cosm", "ALGRO.PA", 4.40),
    ("Hopening", "ALHOP.PA", 5.15),
    ("Plant Advanced", "ALPAU.PA", 5.40),
    ("Eduformaction", "ALEDU.PA", 0.252),
    ("Imprimerie Chirat", "ALCHI.PA", 3.60),
    ("Artea", "ALART.PA", 5.40),
    ("Barbara Bui", "BUI.PA", 3.80),
    ("Lleidanetworks", "ALLLN.PA", 1.02),
    ("Making Science", "ALMKS.PA", 7.45),
    ("Les Constructeurs", "ALCOO.PA", 2.34),
    ("Devernois", "ALDEV.PA", 10.20),
    ("Intexa", "ALINT.PA", 2.50),
    ("Eurasia Groupe", "ALEUA.PA", 1.22),
    ("Quantum Genomics", "ALQGC.PA", 0.072),
    ("Maison Clio Blue", "ALCLB.PA", 1.90),
    ("Condor Tech", "ALCDO.PA", 12.00),
    ("Reboost Blockchain", "ALRBB.PA", 0.075),
    # list 1 extras - try more tickers
    ("Sword Group", "ALSWP.PA", 30.00),
    ("Groupe Crit", "ALCRI.PA", 53.60),
    ("AB Science", "ALABS.PA", 0.90),
    ("Poujoulat", "POU.PA", 6.30),
    ("Exel Industries", "EXA.PA", 21.80),
    ("Barbara Bui", "ALBBUI.PA", 3.80),
    ("Consort NT", "ALCON.PA", 60.00),
]


def get_price(ticker):
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        p = info.get("regularMarketPrice") or info.get("currentPrice")
        if p:
            return float(p), info
        hist = tk.history(period="5d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1]), info
    except Exception:
        pass
    return None, {}


def price_match(rel, p, hint):
    if rel < 0.2:
        return True
    if hint < 1:
        return abs(p - hint) < max(0.15, hint * 0.5)
    return rel < 0.08


def main():
    with open(TICKERS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    all_json = set()
    for v in data.values():
        if isinstance(v, list):
            all_json.update(v)
    with open(UNIV_PATH, encoding="utf-8") as f:
        univ_keys = set(json.load(f).get("tickers", {}))

    seen = {}
    for entry in CANDIDATES:
        name = entry[0]
        primary = entry[1]
        hint = entry[2]
        alts = entry[3] if len(entry) > 3 else []
        opts = [primary] + [a for a in alts if a != primary]
        best = None
        for t in opts:
            p, info = get_price(t)
            if p is None:
                continue
            rel = abs(p - hint) / max(abs(hint), 0.001)
            if best is None or rel < best[0]:
                best = (rel, t, p, info)
        if best is None:
            continue
        rel, t, p, info = best
        if name in seen and seen[name][0] <= rel:
            continue
        div = info.get("dividendRate") or info.get("trailingAnnualDividendRate") or 0
        dy = info.get("dividendYield") or 0
        pays = float(div or 0) > 0 or (
            isinstance(dy, (int, float)) and float(dy) > 0
        )
        ok = price_match(rel, p, hint)
        seen[name] = (rel, t, p, pays, ok, t in all_json, t in univ_keys)

    to_json, to_univ = [], []
    print(f"{'NAME':22} {'TICKER':12} {'PRICE':>8} {'PAYS':5} {'JSON':5} {'UNIV':5}")
    for name, (rel, t, p, pays, ok, in_j, in_u) in sorted(seen.items()):
        if not ok:
            print(f"{name:22} {t:12} {p:8.4f} {'?':5} {str(in_j):5} {str(in_u):5}  NO MATCH")
            continue
        print(f"{name:22} {t:12} {p:8.4f} {str(pays):5} {str(in_j):5} {str(in_u):5}")
        if pays:
            if not in_j:
                to_json.append(t)
            if not in_u:
                to_univ.append(t)

    to_json = sorted(set(to_json))
    to_univ = sorted(set(to_univ))
    print("\nADD JSON:", to_json)
    print("ADD UNIV:", to_univ)
    return to_json, to_univ


if __name__ == "__main__":
    main()
