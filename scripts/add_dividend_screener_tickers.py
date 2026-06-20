#!/usr/bin/env python3
"""Ajoute les tickers screener dividendes (FR + US) à tickers.json."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TICKERS_PATH = ROOT / "tickers.json"

EU_DIVIDEND_SCREENER_FR = [
    "RUI.PA",
    "COFA.PA",
    "TFI.PA",
    "MERY.PA",
    "EGAB.PA",
    "MLCFM.PA",
    "PAT.PA",
    "ABCA.PA",
    "ALHIT.PA",
    "ALINN.PA",
]

USA_DIVIDEND_SCREENER = [
    "BTI",
    "VALE",
    "ET",
    "MET-PA",
    "MPLX",
    "CQP",
    "WES",
    "OHI",
    "PDI",
    "HESM",
    "KGS",
    "STWD",
    "MAIN",
    "AKO-B",
    "TRTN-PC",
    "TRTN-PA",
    "TRTN-PD",
    "TRTN-PB",
    "BNL",
    "DNP",
    "USAC",
    "KEN",
    "NEA",
    "AB",
    "HIW",
    "HTGC",
    "WU",
    "RVT",
    "CMRE-PB",
    "COTY",
    "GOF",
    "USA",
    "ETV",
    "IIPR-PA",
    "EFC-PC",
    "BST",
    "FLNG",
    "KRP",
    "VET",
    "TSLX",
    "UMH-PD",
    "PBT",
    "EOS",
    "CIM-PB",
    "XIFR",
    "ABR-PE",
    "ABR-PD",
    "EVV",
    "SBR",
    "GSBD",
    "CPAC",
    "BBDC",
    "PCN",
    "BCSF",
    "EOI",
    "THQ",
    "PFLT",
    "CODI-PA",
    "CODI-PC",
    "CODI-PB",
    "INN-PE",
    "NIE",
    "BIT",
    "CTO-PA",
    "BWMX",
    "PFN",
    "VTS",
    "RWT-PA",
    "EMD",
    "GRNT",
    "ETD",
    "DPG",
    "THW",
    "XRN",
    "IFN",
    "GHY",
    "ETB",
    "EAD",
    "SAR",
    "SCD",
    "RFI",
    "ACV",
    "BRBS",
    "HYI",
]


def main():
    with open(TICKERS_PATH, encoding="utf-8") as f:
        data = json.load(f)

    data["EU_DIVIDEND_SCREENER_FR"] = EU_DIVIDEND_SCREENER_FR
    data["USA_DIVIDEND_SCREENER"] = USA_DIVIDEND_SCREENER

    with open(TICKERS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")

    all_new = set(EU_DIVIDEND_SCREENER_FR) | set(USA_DIVIDEND_SCREENER)
    print(
        f"Ajouté {len(EU_DIVIDEND_SCREENER_FR)} FR + {len(USA_DIVIDEND_SCREENER)} US "
        f"= {len(all_new)} tickers uniques dans tickers.json"
    )


if __name__ == "__main__":
    main()
