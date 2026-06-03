#!/usr/bin/env python
"""
Met à jour sector_benchmarks.yaml depuis les fichiers Excel Damodaran (NYU Stern).

Usage :
  python update_sector_benchmarks.py
  python update_sector_benchmarks.py --dry-run
  python update_sector_benchmarks.py --bump-engine-version
  python update_sector_benchmarks.py --refresh-cache

Planification Windows (2× par an, ex. 15 janv. et 15 juil.) :
  schtasks /Create /TN "AnalyseFinanciere_Damodaran" /TR ^
    "\"C:\\Path\\To\\python.exe\" \"C:\\Path\\To\\update_sector_benchmarks.py\" --bump-engine-version" ^
    /SC MONTHLY /MO JAN,JUL /D 15 /ST 06:00
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import fair_value_benchmarks as fb


def main() -> int:
    parser = argparse.ArgumentParser(description="Actualise sector_benchmarks.yaml (Damodaran)")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=fb.DEFAULT_YAML_PATH,
        help="Fichier YAML de sortie",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche un résumé sans écrire le fichier",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Re-télécharge les .xls Damodaran (ignore le cache disque)",
    )
    parser.add_argument(
        "--bump-engine-version",
        action="store_true",
        help="Incrémente FAIR_VALUE_ENGINE_VERSION dans fundamentals.py (vide cache Streamlit)",
    )
    args = parser.parse_args()

    try:
        cfg = fb.fetch_and_merge(
            use_cache=not args.refresh_cache,
            existing_path=args.output if args.output.is_file() else fb.DEFAULT_YAML_PATH,
        )
    except Exception as exc:
        print(f"Échec mise à jour Damodaran : {exc}", file=sys.stderr)
        return 1

    meta = cfg.get("meta") or {}
    baseline = cfg.get("baseline_market") or {}
    print(f"Repères Damodaran — {meta.get('updated', '?')}")
    print(
        f"  Baseline US : PER {baseline.get('per')} · P/B {baseline.get('pb')} · P/S {baseline.get('ps')}"
    )
    fr = (cfg.get("countries") or {}).get("France") or {}
    print(f"  France (facteurs) : PER ×{fr.get('per')} · P/B ×{fr.get('pb')} · P/S ×{fr.get('ps')}")
    for profile in ("pharma", "finance", "énergie"):
        mult = ((cfg.get("profiles") or {}).get(profile) or {}).get("multiples") or {}
        print(f"  Profil {profile} (US) : PER {mult.get('per')} · P/B {mult.get('pb')} · P/S {mult.get('ps')}")

    if args.dry_run:
        print("Dry-run — fichier non modifié.")
        return 0

    out = fb.write_benchmark_yaml(cfg, args.output)
    print(f"Écrit : {out}")

    if args.bump_engine_version:
        new_ver = fb.bump_fair_value_engine_version()
        print(f"FAIR_VALUE_ENGINE_VERSION -> {new_ver} (relancez le dashboard / videz le cache Streamlit)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
