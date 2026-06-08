"""Screen tickers: rendement > 5 % et couverture > 2×."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import json

import dividendes as div

CANDIDATES = {
    # US
    "PRU": {"Société": "Prudential Financial, Inc.", "Marché": "NYSE", "Devise": "USD", "Pays": "US"},
    "MET": {"Société": "MetLife, Inc.", "Marché": "NYSE", "Devise": "USD", "Pays": "US"},
    "PFG": {"Société": "Principal Financial Group, Inc.", "Marché": "NASDAQ", "Devise": "USD", "Pays": "US"},
    "ALL": {"Société": "Allstate Corporation", "Marché": "NYSE", "Devise": "USD", "Pays": "US"},
    "VZ": {"Société": "Verizon Communications Inc.", "Marché": "NYSE", "Devise": "USD", "Pays": "US"},
    "MO": {"Société": "Altria Group, Inc.", "Marché": "NYSE", "Devise": "USD", "Pays": "US"},
    "PM": {"Société": "Philip Morris International Inc.", "Marché": "NYSE", "Devise": "USD", "Pays": "US"},
    "STX": {"Société": "Seagate Technology Holdings plc", "Marché": "NASDAQ", "Devise": "USD", "Pays": "US"},
    "HPQ": {"Société": "HP Inc.", "Marché": "NYSE", "Devise": "USD", "Pays": "US"},
    "ETD": {"Société": "Ethan Allen Interiors Inc.", "Marché": "NYSE", "Devise": "USD", "Pays": "US"},
    "VTRS": {"Société": "Viatris Inc.", "Marché": "NASDAQ", "Devise": "USD", "Pays": "US"},
    "IVZ": {"Société": "Invesco Ltd.", "Marché": "NYSE", "Devise": "USD", "Pays": "US"},
    # France
    "TTE.PA": {"Société": "TotalEnergies SE", "Marché": "Euronext Paris", "Devise": "EUR", "Pays": "FR"},
    "ORA.PA": {"Société": "Orange S.A.", "Marché": "Euronext Paris", "Devise": "EUR", "Pays": "FR"},
    "ACA.PA": {"Société": "Crédit Agricole S.A.", "Marché": "Euronext Paris", "Devise": "EUR", "Pays": "FR"},
    "VIV.PA": {"Société": "Vivendi SE", "Marché": "Euronext Paris", "Devise": "EUR", "Pays": "FR"},
    "RUI.PA": {"Société": "Rubis", "Marché": "Euronext Paris", "Devise": "EUR", "Pays": "FR"},
    "NEX.PA": {"Société": "Nexans S.A.", "Marché": "Euronext Paris", "Devise": "EUR", "Pays": "FR"},
    "VRLA.PA": {"Société": "Veralto Corporation", "Marché": "Euronext Paris", "Devise": "EUR", "Pays": "FR"},
    "BOUY.PA": {"Société": "Bouygues S.A.", "Marché": "Euronext Paris", "Devise": "EUR", "Pays": "FR"},
    # Germany
    "RWE.DE": {"Société": "RWE AG", "Marché": "XETRA", "Devise": "EUR", "Pays": "DE"},
    "ALV.DE": {"Société": "Allianz SE", "Marché": "XETRA", "Devise": "EUR", "Pays": "DE"},
    "MUV2.DE": {"Société": "Münchener Rückversicherungs-Gesellschaft AG", "Marché": "XETRA", "Devise": "EUR", "Pays": "DE"},
    "HNR1.DE": {"Société": "Hannover Rück SE", "Marché": "XETRA", "Devise": "EUR", "Pays": "DE"},
    "DTE.DE": {"Société": "Deutsche Telekom AG", "Marché": "XETRA", "Devise": "EUR", "Pays": "DE"},
    "BAS.DE": {"Société": "BASF SE", "Marché": "XETRA", "Devise": "EUR", "Pays": "DE"},
    # Belgium
    "KBC.BR": {"Société": "KBC Group NV", "Marché": "Euronext Brussels", "Devise": "EUR", "Pays": "BE"},
    "AGEAS.BR": {"Société": "Ageas", "Marché": "Euronext Brussels", "Devise": "EUR", "Pays": "BE"},
    "SOF.BR": {"Société": "Sofina SA", "Marché": "Euronext Brussels", "Devise": "EUR", "Pays": "BE"},
    "WDP.BR": {"Société": "Warehouses De Pauw SA", "Marché": "Euronext Brussels", "Devise": "EUR", "Pays": "BE"},
    "GBLB.BR": {"Société": "Groupe Bruxelles Lambert", "Marché": "Euronext Brussels", "Devise": "EUR", "Pays": "BE"},
    "PROX.BR": {"Société": "Proximus PLC", "Marché": "Euronext Brussels", "Devise": "EUR", "Pays": "BE"},
}


def _metrics(row, listing):
    entry = div._cache_row_to_universe_entry(
        row,
        listing={
            "Société": listing["Société"],
            "Secteur": "—",
            "Marché": listing["Marché"],
            "Devise": listing["Devise"],
        },
        source="yfinance",
    )
    if entry is None:
        return None
    y = entry.get("Rendement (%)")
    c = entry.get("Ratio couverture")
    if y is None:
        dps = entry.get("DPS dernier versement")
        px = entry.get("Prix")
        if dps and px and px > 0:
            y = (float(dps) * 4) / float(px)
            entry["Rendement (%)"] = y
    if c is None:
        payout = entry.get("Taux versement (%)")
        if payout and payout > 0:
            c = 1.0 / float(payout)
            entry["Ratio couverture"] = c
    return entry


def main():
    with open(ROOT / "tickers.json", encoding="utf-8") as f:
        all_tickers = set()
        for vals in json.load(f).values():
            all_tickers.update(vals)

    hits = []
    for ticker, listing in CANDIDATES.items():
        try:
            row = div._fetch_dividend_profile_with_timeout(ticker, lite=False, timeout_seconds=45)
            if not isinstance(row, dict):
                continue
            entry = _metrics(row, listing)
            if not entry:
                continue
            y = entry.get("Rendement (%)")
            c = entry.get("Ratio couverture")
            if y is None or c is None:
                continue
            if float(y) > 0.05 and float(c) > 2.0:
                hits.append(
                    {
                        "ticker": ticker,
                        "pays": listing["Pays"],
                        "yield": float(y),
                        "coverage": float(c),
                        "in_tickers": ticker in all_tickers,
                        "entry": entry,
                        "listing": listing,
                    }
                )
                print(
                    f"OK {ticker} ({listing['Pays']}) yield={y:.1%} cov={c:.1f}x "
                    f"in_json={ticker in all_tickers}"
                )
        except Exception as exc:
            print(f"SKIP {ticker}: {exc}")

    hits.sort(key=lambda x: (-x["yield"], -x["coverage"]))
    print("\n--- TOP ---")
    for h in hits[:20]:
        print(h["ticker"], h["pays"], f"{h['yield']:.1%}", f"{h['coverage']:.1f}x", h["in_tickers"])


if __name__ == "__main__":
    main()
