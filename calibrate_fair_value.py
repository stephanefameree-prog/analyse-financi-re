#!/usr/bin/env python
"""
Calibre et teste le modèle de juste valeur vs exports Investing.com.

Usage :
  python calibrate_fair_value.py
  python calibrate_fair_value.py --holdout "Warren Buffett - minière filtrée - 2026-06-02.xlsx"
  python calibrate_fair_value.py --ma-vue -o comparaison_ma_vue.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import fair_value as fv


def _default_train_paths(exclude: Path | None = None) -> list[Path]:
    paths = fv.glob_investing_exports(".", "*2026-06-02*.xlsx")
    if not paths:
        paths = fv.glob_investing_exports(".", "*.xlsx")
    holdout_names = {"warren", "défense", "defense", "xetra", "filtre analyse france"}
    out = []
    for p in paths:
        if exclude and p.resolve() == exclude.resolve():
            continue
        low = p.name.lower()
        if any(h in low for h in holdout_names):
            continue
        if "mon portefeuille" in low or low.startswith("paris"):
            out.append(p)
    return sorted(out)


def main():
    parser = argparse.ArgumentParser(description="Calibrage juste valeur vs Investing.com")
    parser.add_argument(
        "files",
        nargs="*",
        help="Fichiers .xlsx Investing pour entraînement",
    )
    parser.add_argument(
        "--method",
        choices=("pillars_ridge", "gradient_boosting", "ridge"),
        default="pillars_ridge",
        help="pillars_ridge = piliers Graham/multiples calibrés (défaut)",
    )
    parser.add_argument(
        "--holdout",
        help="Fichier holdout (ex. Warren Buffett) : entraîne sur les autres exports",
    )
    parser.add_argument(
        "--ma-vue",
        action="store_true",
        help="Calibre sur Minières - ma vue 1 + ma vue 2 (DCF, comparables…)",
    )
    parser.add_argument(
        "--no-yahoo-price",
        action="store_true",
        help="Ne pas compléter le cours via Yahoo pour ma vue",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="CSV de comparaison prédictions vs Investing",
    )
    args = parser.parse_args()

    if args.ma_vue:
        extra = [Path(p) for p in args.files] if args.files else None
        model, predictions, cv_metrics, submodels = fv.calibrate_from_ma_vue(
            directory=".",
            method=args.method,
            enrich_yahoo_price=not args.no_yahoo_price,
            extra_train_paths=extra,
        )
        valid = fv.filter_valid_ma_vue_rows(fv.load_ma_vue(enrich_yahoo_price=not args.no_yahoo_price))
        print(f"Ma vue : {len(valid)} titres calibrables (juste valeur + cours)")
        print()
        if not submodels.empty:
            print("Sous-modèles Investing vs Juste Valeur :")
            print(submodels.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
            print()
        print(fv.format_metrics_report(cv_metrics, title="Validation croisée ma vue"))
        if model.metrics_:
            print()
            print(fv.format_metrics_report(model.metrics_, title="Ajustement complet ma vue"))
        if args.method == "pillars_ridge":
            print()
            print("Poids des piliers (Ridge) :")
            print(fv.explain_pillar_weights(model).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        if args.output:
            predictions.to_csv(args.output, index=False, encoding="utf-8-sig")
            print(f"\nExport : {Path(args.output).resolve()}")
        return

    if args.holdout:
        holdout = Path(args.holdout)
        if not holdout.exists():
            holdout = next(Path(".").glob(f"*{args.holdout}*"))
        train_paths = [Path(p) for p in args.files] if args.files else _default_train_paths()
        train_paths = [p for p in train_paths if p.resolve() != holdout.resolve()]
        print("Holdout :", holdout.name)
        print("Entraînement :")
        for p in train_paths:
            print(f"  - {p.name}")
        model, predictions, metrics = fv.run_holdout_test(train_paths, holdout, method=args.method)
        print()
        print(fv.format_metrics_report(metrics, title=f"Holdout — {holdout.name}"))
        if args.method == "pillars_ridge":
            print()
            print("Poids des piliers (Ridge) :")
            print(fv.explain_pillar_weights(model).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        if args.output:
            predictions.to_csv(args.output, index=False, encoding="utf-8-sig")
            print(f"\nExport : {Path(args.output).resolve()}")
        return

    paths = [Path(p) for p in args.files] if args.files else _default_train_paths()
    if not paths:
        raise SystemExit("Aucun fichier .xlsx trouvé.")

    print(f"Fichiers ({len(paths)}) :")
    for p in paths:
        print(f"  - {p.name}")

    model, predictions, cv_metrics, n_valid, n_usable = fv.calibrate_from_exports(
        paths, method=args.method
    )
    print()
    print(f"Lignes avec juste valeur : {n_valid} | entraînement : {n_usable}")
    print(fv.format_metrics_report(cv_metrics))
    print()
    if model.metrics_:
        print(fv.format_metrics_report(model.metrics_, title="Ajustement complet (in-sample)"))
    if args.method == "pillars_ridge":
        print()
        print("Poids des piliers (Ridge) :")
        print(fv.explain_pillar_weights(model).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    worst = predictions.dropna(subset=["fair_value_investing", "abs_pct_error"]).nlargest(10, "abs_pct_error")
    if not worst.empty:
        print()
        print("Top 10 écarts relatifs :")
        cols = [
            "Nom",
            "price",
            "fair_value_investing",
            "fair_value_pred",
            "abs_pct_error",
            "label_investing",
            "label_pred",
        ]
        cols = [c for c in cols if c in worst.columns]
        print(worst[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if args.output:
        out_path = Path(args.output)
        predictions.to_csv(out_path, index=False, encoding="utf-8-sig")
        print()
        print(f"Comparaison exportée : {out_path.resolve()}")


if __name__ == "__main__":
    main()
