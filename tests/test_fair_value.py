"""Tests calibrage juste valeur vs exports Investing."""
from pathlib import Path

import pandas as pd
import pytest

import fair_value as fv

ROOT = Path(__file__).resolve().parents[1]
EXPORT_GLOB = "*2026-06-02*.xlsx"


def _export_paths():
    paths = sorted(ROOT.glob(EXPORT_GLOB))
    if not paths:
        pytest.skip(f"Aucun export Investing ({EXPORT_GLOB}) à la racine du projet.")
    return paths


def test_load_investing_exports():
    paths = _export_paths()
    df = fv.filter_valid_fair_value_rows(fv.load_investing_exports(paths))
    assert len(df) >= 20
    assert "fair_value" in df.columns
    assert "price" in df.columns
    assert df["fair_value"].notna().all()
    assert (df["fair_value"] > 0).all()


def test_label_thresholds_match_investing_bands():
    df = fv.filter_valid_fair_value_rows(fv.load_investing_exports(_export_paths()))
    th = fv.derive_label_thresholds(df)
    labels = fv.classify_upside_many(df["upside"].values, th)
    acc = (labels == df["label"].values).mean()
    assert acc >= 0.80


def test_cross_validate_mape():
    df = fv.rows_with_complete_features(
        fv.load_investing_exports(_export_paths()), method="pillars_ridge"
    )
    model = fv.FairValueModel(method="pillars_ridge")
    metrics = model.cross_validate(df)
    assert metrics.n >= 50
    assert metrics.mape < 0.25
    assert metrics.r2 > 0.85


def test_pillar_features_present():
    df = fv.filter_valid_fair_value_rows(fv.load_investing_exports(_export_paths()))
    X = fv.build_pillar_features(df)
    assert "pillar_graham_log" in X.columns
    assert "pillar_per_log" in X.columns


def test_warren_buffett_holdout():
    paths = _export_paths()
    warren = next((p for p in paths if "Warren" in p.name), None)
    if warren is None:
        pytest.skip("Export Warren Buffett absent.")
    train = [p for p in paths if "Warren" not in p.name and "ma vue" not in p.name.lower()]
    _, _, metrics = fv.run_holdout_test(train, warren, method="pillars_ridge")
    assert metrics.n >= 20
    assert metrics.mape < 0.25


def test_fit_predict_columns():
    df = fv.rows_with_complete_features(
        fv.load_investing_exports(_export_paths()), method="pillars_ridge"
    )
    model = fv.FairValueModel(method="pillars_ridge")
    model.fit(df)
    pred = model.predict_dataframe(df)
    for col in ("fair_value_pred", "upside_pred", "label_pred", "fair_value_investing"):
        assert col in pred.columns
    assert pred["fair_value_pred"].notna().sum() >= 50


def test_load_ma_vue():
    vue1, vue2 = fv.glob_ma_vue_exports(ROOT)
    if vue1 is None:
        pytest.skip("Exports ma vue absents.")
    df = fv.load_ma_vue(vue1, vue2, enrich_yahoo_price=False)
    assert len(df) >= 50
    assert "fair_value" in df.columns
    assert fv.COL_FV_COMPARABLE in df.columns or fv.COL_FV_DCF in df.columns


def test_ma_vue_investing_pillars():
    vue1, vue2 = fv.glob_ma_vue_exports(ROOT)
    if vue1 is None:
        pytest.skip("Exports ma vue absents.")
    df = fv.filter_valid_ma_vue_rows(fv.load_ma_vue(vue1, vue2, enrich_yahoo_price=False))
    if df.empty:
        pytest.skip("Aucune ligne ma vue valide sans Yahoo.")
    X = fv.build_pillar_features(df)
    assert "pillar_investing_comparable_log" in X.columns
    assert X["pillar_investing_comparable_log"].notna().any()


def test_calibrate_from_ma_vue():
    vue1, vue2 = fv.glob_ma_vue_exports(ROOT)
    if vue1 is None:
        pytest.skip("Exports ma vue absents.")
    _, _, metrics, submodels = fv.calibrate_from_ma_vue(
        vue1, vue2, directory=ROOT, enrich_yahoo_price=False
    )
    assert metrics.n >= 10
    assert metrics.mape < 0.20
    assert not submodels.empty


def test_build_features_from_fundamentals_mapping():
    row = {
        "Dernier cours": 10.0,
        "PER": 12.0,
        "Cours / val. comptable": 1.5,
        "Cours / ventes": 2.0,
        "Rendement dividende (%)": 0.03,
        "Dette / Capitaux": 0.4,
        "RSI (14)": 55.0,
        "Capitalisation de marché": 1_000_000_000,
    }
    mapped = fv.map_fundamentals_row(row)
    df = fv.normalize_investing_frame(
        fv.load_investing_exports(_export_paths()).head(1).assign(**{fv.COL_PRICE: 10.0})
    )
    single = df.head(1).copy()
    for k, v in mapped.items():
        single[k] = v
    features = fv.build_fair_value_features(single)
    assert features.notna().all(axis=1).iloc[0]


def test_predict_fundamentals_without_ma_vue_pillars():
    """Le modèle entraîné sur ma vue doit prédire sans piliers Investing absents."""
    vue1, vue2 = fv.glob_ma_vue_exports(ROOT)
    if vue1 is None:
        pytest.skip("Exports ma vue absents.")
    model, _, _, _ = fv.calibrate_from_ma_vue(
        vue1, vue2, directory=ROOT, enrich_yahoo_price=False
    )
    assert any(c.startswith("pillar_investing_") for c in model.feature_columns_)

    rows = [
        {
            "Ticker": "TEST.PA",
            "Dernier cours": 50.0,
            "PER": 14.0,
            "Cours / val. comptable": 1.2,
            "Cours / ventes": 1.8,
            "Rendement dividende (%)": 0.025,
            "Dette / Capitaux": 0.5,
            "ROE": 0.12,
            "Marge nette": 0.08,
        }
    ]
    pred = fv.predict_fair_value_for_fundamentals(
        rows, {fv.SECTOR_MINIERES: model}, {fv.SECTOR_MINIERES: {}}
    )
    assert len(pred) == 1
    assert pred["Juste valeur estimée"].notna().iloc[0]
    assert pred["Upside juste valeur"].notna().iloc[0]


def test_predict_lossmaking_miner_like_xpl():
    """Titres déficitaires (PER absent) : piliers analyste/ROE/P-B suffisent."""
    vue1, vue2 = fv.glob_ma_vue_exports(ROOT)
    if vue1 is None:
        pytest.skip("Exports ma vue absents.")
    model, _, _, _ = fv.calibrate_from_ma_vue(
        vue1, vue2, directory=ROOT, enrich_yahoo_price=False
    )
    row = {
        "Ticker": "XPL",
        "Dernier cours": 0.845,
        "Cours / val. comptable": 3.07,
        "ROE": -0.15,
        "ROIC": -0.18,
        "Marge nette": 0.0,
        "Upside vs objectif": 0.77,
        "Rendement actionnaire": 0.0,
    }
    pred = fv.predict_fair_value_for_fundamentals(
        [row], {fv.SECTOR_MINIERES: model}, {fv.SECTOR_MINIERES: {}}
    )
    assert len(pred) == 1
    assert pred["Juste valeur estimée"].notna().iloc[0]
    assert pred["Upside juste valeur"].notna().iloc[0]


def test_resolve_fair_value_sector_from_yahoo():
    assert fv.resolve_fair_value_sector(
        "ALNTG.PA", frozenset(), frozenset(), "Technology", "Communication Equipment"
    ) == fv.SECTOR_UNIFIED
    assert fv.resolve_fair_value_sector(
        "HMY", frozenset(), frozenset(), "Basic Materials", "Gold"
    ) == fv.SECTOR_UNIFIED
    assert fv.resolve_fair_value_sector(
        "XPL", frozenset(), frozenset(), "Basic Materials", "Other Industrial Metals & Mining"
    ) == fv.SECTOR_UNIFIED


def test_resolve_fair_value_profiles():
    assert (
        fv.resolve_fair_value_profile(
            "NUTX", frozenset(), frozenset(), "Healthcare", "Medical Care Facilities"
        )
        == fv.SECTOR_PHARMA
    )
    assert (
        fv.resolve_fair_value_profile(
            "EDEN.PA", frozenset(), frozenset(), "Financial Services", "Credit Services"
        )
        == fv.SECTOR_FINANCE
    )
    assert (
        fv.resolve_fair_value_profile(
            "ZAL.DE", frozenset(), frozenset(), "Consumer Cyclical", "Internet Retail"
        )
        == fv.SECTOR_CONSUMER
    )
    assert (
        fv.resolve_fair_value_profile(
            "ETL.PA", frozenset(), frozenset(), "Communication Services", "Telecom Services"
        )
        == fv.SECTOR_TECH
    )
    assert fv.PROFILE_BASE_MODEL[fv.SECTOR_PHARMA] == fv.SECTOR_UNIFIED


def test_sector_benchmarks_yaml_loads():
    cfg = fv.load_sector_benchmarks(force_reload=True)
    assert cfg.get("profiles")
    assert "France" in (cfg.get("countries") or {})
    base = fv.get_base_profile_multiples(cfg)
    assert base[fv.SECTOR_PHARMA]["per"] >= 15.0
    assert base[fv.SECTOR_TECH]["per"] == 28.0


def test_france_lowers_profile_multiples():
    us = fv.resolve_profile_multiples(fv.SECTOR_PHARMA, "PFE")
    fr = fv.resolve_profile_multiples(fv.SECTOR_PHARMA, "EDEN.PA")
    assert fr["per"] < us["per"]
    assert fr["pb"] < us["pb"]
    assert fv._country_for_ticker("EDEN.PA", fv.load_sector_benchmarks()) == "France"


def test_damodaran_fair_value_uses_sector_and_country():
    row = {
        "Ticker": "ZAL.DE",
        "Dernier cours": 22.93,
        "PER": 47.6,
        "Cours / val. comptable": 2.08,
        "Secteur Yahoo": "Consumer Cyclical",
        "Industrie Yahoo": "Internet Retail",
    }
    out = fv.compute_damodaran_fair_value(row)
    assert out["Profil Damodaran"] == fv.SECTOR_CONSUMER
    jv = out["Juste valeur Damodaran"]
    assert jv is not None and jv > 22.93
    assert out["Upside Damodaran"] == pytest.approx(jv / 22.93 - 1.0)
    fr_row = dict(row, Ticker="MC.PA", **{"Dernier cours": 475.0, "PER": 21.7, "Cours / val. comptable": 3.5})
    fr_out = fv.compute_damodaran_fair_value(fr_row)
    assert fr_out["Profil Damodaran"] == fv.SECTOR_CONSUMER
    us_out = fv.compute_damodaran_fair_value(
        {**fr_row, "Ticker": "NKE"},
    )
    assert us_out["Juste valeur Damodaran"] != fr_out["Juste valeur Damodaran"]


def test_predict_includes_damodaran_columns():
    v1, _ = fv.glob_ma_vue_exports(ROOT)
    if v1 is None:
        pytest.skip("Exports absents.")
    models, _ = fv.load_dashboard_fair_value_models(ROOT, enrich_yahoo_price=False)
    row = {
        "Ticker": "BNP.PA",
        "Dernier cours": 94.43,
        "PER": 8.8,
        "Cours / val. comptable": 0.8,
        "Secteur Yahoo": "Financial Services",
    }
    pred = fv.predict_fair_value_for_fundamentals([row], models, {})
    assert "Juste valeur Damodaran" in pred.columns
    assert pred["Profil Damodaran"].iloc[0] == fv.SECTOR_FINANCE


def test_all_profiles_use_unified_model():
    """Même modèle Ridge pour tech, finance, minières — seuls les multiples sectoriels varient."""
    vue2, eff, risk, vue3 = fv.glob_tech_exports(ROOT)
    v1, v2 = fv.glob_ma_vue_exports(ROOT)
    if vue2 is None or v1 is None:
        pytest.skip("Exports absents.")
    models, meta = fv.load_dashboard_fair_value_models(ROOT, enrich_yahoo_price=False)
    if fv.SECTOR_UNIFIED not in models:
        pytest.skip(f"Modèle unifié absent : {meta.get(fv.SECTOR_UNIFIED)}")
    row_tech = {
        "Ticker": "ALNTG.PA",
        "Dernier cours": 0.848,
        "PER": 17.0,
        "Cours / val. comptable": 1.15,
        "Cours / ventes": 0.89,
        "Dette / Capitaux": 7.3,
        "ROE": 0.06,
        "Upside vs objectif": 0.18,
        "Secteur Yahoo": "Technology",
        "Industrie Yahoo": "Communication Equipment",
    }
    row_finance = {
        **row_tech,
        "Ticker": "BNP.PA",
        "Dernier cours": 94.43,
        "PER": 8.91,
        "Cours / val. comptable": 0.80,
        "Cours / ventes": 2.12,
        "Secteur Yahoo": "Financial Services",
        "Industrie Yahoo": "Banks - Regional",
    }
    pred_tech = fv.predict_fair_value_for_fundamentals([row_tech], models, {})
    pred_fin = fv.predict_fair_value_for_fundamentals([row_finance], models, {})
    assert pred_tech["Modèle juste valeur"].iloc[0] == fv.SECTOR_UNIFIED
    assert pred_fin["Modèle juste valeur"].iloc[0] == fv.SECTOR_UNIFIED
    assert fv.resolve_unified_multiples("ALNTG.PA") == fv.resolve_unified_multiples("BNP.PA")


def test_xpl_comparable_from_ma_vue_upside():
    """Solitario : JV proche Investing via hausse comparable × cours Yahoo."""
    vue1, vue2 = fv.glob_ma_vue_exports(ROOT)
    if vue1 is None:
        pytest.skip("Exports ma vue absents.")
    model, _, _, _ = fv.calibrate_from_ma_vue(
        vue1, vue2, directory=ROOT, enrich_yahoo_price=False
    )
    ma = fv.try_load_ma_vue_for_dashboard(".")
    if ma is None:
        pytest.skip("Ma vue absente.")
    idx = fv.build_ma_vue_yahoo_index(ma)
    if "XPL" not in idx:
        pytest.skip("XPL absent de ma vue.")
    row = {
        "Ticker": "XPL",
        "Dernier cours": 0.845,
        "Cours / val. comptable": 3.07,
        "ROE": -0.15,
        "Upside vs objectif": 0.77,
    }
    pred = fv.predict_fair_value_for_fundamentals(
        [row], {fv.SECTOR_MINIERES: model}, {fv.SECTOR_MINIERES: idx}
    )
    jv = float(pred["Juste valeur estimée"].iloc[0])
    assert 0.85 <= jv <= 0.90


def test_predict_ignores_ma_vue_metric_enrichment():
    """Les métriques Investing ma vue ne doivent pas influencer la prédiction dashboard."""
    vue1, vue2 = fv.glob_ma_vue_exports(ROOT)
    if vue1 is None:
        pytest.skip("Exports ma vue absents.")
    model, _, _, _ = fv.calibrate_from_ma_vue(
        vue1, vue2, directory=ROOT, enrich_yahoo_price=False
    )
    row = {
        "Ticker": "TEST.PA",
        "Dernier cours": 50.0,
        "PER": 14.0,
        "Cours / val. comptable": 1.2,
        "Cours / ventes": 1.8,
        "Dette / Capitaux": 0.5,
        "ROE": 0.12,
        "Marge nette": 0.08,
    }
    fake = {
        "TEST.PA": {
            fv.COL_FV_DCF: 9999.0,
            fv.COL_FV_COMPARABLE: 8888.0,
            fv.COL_FAIR_VALUE: 7777.0,
        }
    }
    pred_plain = fv.predict_fair_value_for_fundamentals(
        [row], {fv.SECTOR_MINIERES: model}, {fv.SECTOR_MINIERES: {}}
    )
    pred_fake = fv.predict_fair_value_for_fundamentals(
        [row], {fv.SECTOR_MINIERES: model}, {fv.SECTOR_MINIERES: fake}
    )
    assert pred_plain["Juste valeur estimée"].iloc[0] == pred_fake["Juste valeur estimée"].iloc[0]


def test_load_tech_ma_vue():
    vue2, eff, risk, vue3 = fv.glob_tech_exports(ROOT)
    if vue2 is None:
        pytest.skip("Exports tech absents.")
    df = fv.load_tech_ma_vue(vue2, eff, risk, vue3, enrich_yahoo_price=False)
    assert len(df) >= 100
    assert df["fair_value"].notna().sum() >= 100
    assert df["price"].notna().sum() >= 100
    if vue3 is not None:
        assert "Règle des 40" in df.columns
        assert df["Règle des 40"].notna().sum() >= 50


def test_predict_fundamentals_without_dividend_yield():
    """Régression : prédiction sans colonne rendement dividende (Yahoo seul)."""
    paths = _export_paths()
    if not paths:
        pytest.skip("Exports absents.")
    model, _pred, _m, _sub = fv.calibrate_from_ma_vue(
        directory=ROOT, enrich_yahoo_price=False
    )
    row = {
        "Ticker": "TEST.PA",
        "Dernier cours": 10.0,
        "PER": 12.0,
        "Cours / val. comptable": 1.5,
        "Cours / ventes": 2.0,
        "Dette / Capitaux": 0.4,
        "ROE": 0.12,
        "Marge nette": 0.08,
    }
    pred = fv.predict_fair_value_for_fundamentals(
        [row], {fv.SECTOR_MINIERES: model}, {fv.SECTOR_MINIERES: {}}
    )
    assert len(pred) == 1
    assert pred["Juste valeur estimée"].notna().iloc[0]
    assert pred["Upside juste valeur"].notna().iloc[0]


def test_calibrate_from_tech_ma_vue():
    vue2, eff, risk, vue3 = fv.glob_tech_exports(ROOT)
    if vue2 is None:
        pytest.skip("Exports tech absents.")
    model, pred, metrics, submodels = fv.calibrate_from_tech_ma_vue(
        vue2, eff, risk, vue3, directory=ROOT, enrich_yahoo_price=False
    )
    assert model.sector == fv.SECTOR_TECH
    assert metrics.n >= 50
    assert metrics.mape < 0.35
    assert pred["fair_value_pred"].notna().sum() >= 50
    assert not submodels.empty


def test_normalize_preserves_tech_fair_value_when_mining_column_present():
    """Fusion minières + tech : ne pas écraser fair_value tech par NaN de « Juste Valeur »."""
    mining = pd.DataFrame(
        {
            fv.COL_TICKER: ["TSX:ABC"],
            fv.COL_FAIR_VALUE: [10.0],
            fv.COL_PRICE: [8.0],
        }
    )
    tech = pd.DataFrame(
        {
            fv.COL_TICKER: ["XTRA:SAP"],
            "fair_value": [200.0],
            "price": [180.0],
        }
    )
    combined = fv.normalize_investing_frame(pd.concat([mining, tech], ignore_index=True))
    assert combined.loc[0, "fair_value"] == 10.0
    assert combined.loc[1, "fair_value"] == 200.0


def test_investing_anchors_yaml_loads_bnp():
    anchors = fv.load_investing_anchors(ROOT / "investing_anchors.yaml")
    assert any(a.get("ticker") == "BNP.PA" for a in anchors)
    assert any(a.get("ticker") == "TTE.PA" for a in anchors)
    assert any(a.get("ticker") == "SAN.PA" for a in anchors)
    assert any(a.get("ticker") == "MC.PA" for a in anchors)
    assert any(a.get("ticker") == "AI.PA" for a in anchors)
    assert any(a.get("ticker") == "SOI.PA" for a in anchors)
    assert any(a.get("ticker") == "SU.PA" for a in anchors)
    frame = fv.investing_anchors_to_training_frame(anchors)
    assert len(frame) == 22
    bnp = frame[frame[fv.COL_NAME].astype(str).str.contains("BNP", na=False)]
    assert float(bnp["fair_value"].iloc[0]) == pytest.approx(95.18)


def test_bnp_anchor_pulls_prediction_toward_investing():
    """Repère BNP 95,18 € intégré au calibrage unifié."""
    v1, v2 = fv.glob_ma_vue_exports(ROOT)
    if v1 is None:
        pytest.skip("Exports ma vue absents.")
    models, _ = fv.load_dashboard_fair_value_models(ROOT, enrich_yahoo_price=False)
    if fv.SECTOR_UNIFIED not in models:
        pytest.skip("Modèle unifié absent.")
    row = {
        "Ticker": "BNP.PA",
        "Dernier cours": 94.43,
        "PER": 8.8,
        "Cours / val. comptable": 0.8,
        "Cours / ventes": 2.12,
        "Rendement dividende (%)": 10.5,
        "Dettes / capitaux": 85.5,
        "RSI 14j": 54.3,
        "ROE": 9.1,
        "ROIC": 1.6,
        "Upside vs objectif": 12.3,
        "Croissance BPA": 10.2,
        "Secteur Yahoo": "Financial Services",
        "Industrie Yahoo": "Banks - Regional",
    }
    pred = fv.predict_fair_value_for_fundamentals([row], models, {})
    jv = float(pred["Juste valeur estimée"].iloc[0])
    assert abs(jv - 95.18) < 6.0, jv


def test_tte_anchor_in_training_frame():
    anchors = fv.load_investing_anchors(ROOT / "investing_anchors.yaml")
    tte = next(a for a in anchors if a.get("ticker") == "TTE.PA")
    assert tte["fair_value"] == pytest.approx(93.98)
    assert tte["price"] == pytest.approx(76.65)


def test_san_anchor_in_training_frame():
    anchors = fv.load_investing_anchors(ROOT / "investing_anchors.yaml")
    san = next(a for a in anchors if a.get("ticker") == "SAN.PA")
    assert san["fair_value"] == pytest.approx(111.27)
    assert san["price"] == pytest.approx(73.42)
    assert san["label"] == "Sous-évaluée"
    assert san["comparable_fv"] == pytest.approx(95.62)
    assert san["dcf_fv"] == pytest.approx(124.55)
    assert san["ddm_fv"] == pytest.approx(128.45)


def test_mc_anchor_in_training_frame():
    anchors = fv.load_investing_anchors(ROOT / "investing_anchors.yaml")
    mc = next(a for a in anchors if a.get("ticker") == "MC.PA")
    assert mc["fair_value"] == pytest.approx(633.7)
    assert mc["price"] == pytest.approx(475.0)


def test_ai_anchor_in_training_frame():
    anchors = fv.load_investing_anchors(ROOT / "investing_anchors.yaml")
    ai = next(a for a in anchors if a.get("ticker") == "AI.PA")
    assert ai["fair_value"] == pytest.approx(170.38)
    assert ai["price"] == pytest.approx(176.54)
    assert ai["label"] == "Juste"


def test_soi_anchor_in_training_frame():
    anchors = fv.load_investing_anchors(ROOT / "investing_anchors.yaml")
    soi = next(a for a in anchors if a.get("ticker") == "SOI.PA")
    assert soi["fair_value"] == pytest.approx(94.08)
    assert soi["price"] == pytest.approx(153.05)
    assert soi["label"] == "Surévaluée"


def test_mc_anchor_submodel_enrichment():
    idx = fv.build_investing_anchors_index(ROOT / "investing_anchors.yaml")
    assert "MC.PA" in idx
    mc = idx["MC.PA"]
    assert fv.COL_FV_COMP_UP in mc
    assert fv.COL_FV_DCF_UP in mc
    assert fv.COL_FV_DDM_UP in mc
    assert mc[fv.COL_FV_COMP_UP] == pytest.approx(591.83 / 475.0 - 1.0, rel=1e-3)


def test_mc_prediction_near_investing_with_submodels():
    v1, _ = fv.glob_ma_vue_exports(ROOT)
    if v1 is None:
        pytest.skip("Exports absents.")
    models, _ = fv.load_dashboard_fair_value_models(ROOT, enrich_yahoo_price=False)
    if fv.SECTOR_UNIFIED not in models:
        pytest.skip("Modèle unifié absent.")
    row = {
        "Ticker": "MC.PA",
        "Dernier cours": 475.0,
        "PER": 21.7,
        "Cours / val. comptable": 3.5,
        "Rendement dividende (%)": 2.7,
        "Dettes / capitaux": 13.4,
        "RSI 14j": 48.2,
        "ROE": 16.1,
        "ROIC": 11.8,
        "Upside vs objectif": 26.3,
    }
    pred = fv.predict_fair_value_for_fundamentals([row], models, {})
    jv = float(pred["Juste valeur estimée"].iloc[0])
    assert abs(jv - 633.7) < 80.0, jv


def test_chtr_anchor_in_training_frame():
    anchors = fv.load_investing_anchors(ROOT / "investing_anchors.yaml")
    chtr = next(a for a in anchors if a.get("ticker") == "CHTR")
    assert chtr["fair_value"] == pytest.approx(246.81)
    assert chtr["price"] == pytest.approx(141.07)
    assert chtr["label"] == "Aubaine"
    assert chtr["comparable_fv"] == pytest.approx(236.96)
    assert chtr["dcf_fv"] == pytest.approx(261.58)


def test_eden_anchor_in_training_frame():
    anchors = fv.load_investing_anchors(ROOT / "investing_anchors.yaml")
    eden = next(a for a in anchors if a.get("ticker") == "EDEN.PA")
    assert eden["fair_value"] == pytest.approx(32.48)
    assert eden["price"] == pytest.approx(23.15)
    assert eden["label"] == "Sous-évaluée"
    assert eden["comparable_fv"] == pytest.approx(31.87)
    assert eden["dcf_fv"] == pytest.approx(44.57)


def test_mrvl_anchor_in_training_frame():
    anchors = fv.load_investing_anchors(ROOT / "investing_anchors.yaml")
    mrvl = next(a for a in anchors if a.get("ticker") == "MRVL")
    assert mrvl["fair_value"] == pytest.approx(158.76)
    assert mrvl["price"] == pytest.approx(305.64)
    assert mrvl["label"] == "Surévaluée"
    assert mrvl["comparable_fv"] == pytest.approx(161.53)
    assert mrvl["dcf_fv"] == pytest.approx(150.10)
    assert mrvl["ddm_fv"] == pytest.approx(194.13)


def test_mrvl_anchor_submodel_enrichment():
    idx = fv.build_investing_anchors_index(ROOT / "investing_anchors.yaml")
    assert "MRVL" in idx
    mrvl = idx["MRVL"]
    assert mrvl[fv.COL_FV_COMP_UP] == pytest.approx(161.53 / 305.64 - 1.0, rel=1e-3)
    assert mrvl[fv.COL_FV_DCF_UP] == pytest.approx(150.10 / 305.64 - 1.0, rel=1e-3)
    assert mrvl[fv.COL_FV_DDM_UP] == pytest.approx(194.13 / 305.64 - 1.0, rel=1e-3)


def test_nyxh_anchor_in_training_frame():
    anchors = fv.load_investing_anchors(ROOT / "investing_anchors.yaml")
    nyxh = next(a for a in anchors if a.get("ticker") == "NYXH")
    assert nyxh["fair_value"] == pytest.approx(3.83)
    assert nyxh["price"] == pytest.approx(2.97)
    assert nyxh["label"] == "Sous-évaluée"
    assert nyxh["comparable_fv"] == pytest.approx(3.83)


def test_nutx_anchor_in_training_frame():
    anchors = fv.load_investing_anchors(ROOT / "investing_anchors.yaml")
    nutx = next(a for a in anchors if a.get("ticker") == "NUTX")
    assert nutx["fair_value"] == pytest.approx(178.48)
    assert nutx["price"] == pytest.approx(125.03)
    assert nutx["label"] == "Sous-évaluée"
    assert nutx["comparable_fv"] == pytest.approx(151.33)
    assert nutx["dcf_fv"] == pytest.approx(232.77)


def test_zal_anchor_in_training_frame():
    anchors = fv.load_investing_anchors(ROOT / "investing_anchors.yaml")
    zal = next(a for a in anchors if a.get("ticker") == "ZAL.DE")
    assert zal["fair_value"] == pytest.approx(34.75)
    assert zal["price"] == pytest.approx(22.93)
    assert zal["label"] == "Sous-évaluée"
    assert zal["comparable_fv"] == pytest.approx(30.53)
    assert zal["dcf_fv"] == pytest.approx(40.01)


def test_zal_prediction_near_investing_with_submodels():
    v1, _ = fv.glob_ma_vue_exports(ROOT)
    if v1 is None:
        pytest.skip("Exports absents.")
    models, _ = fv.load_dashboard_fair_value_models(ROOT, enrich_yahoo_price=False)
    if fv.SECTOR_UNIFIED not in models:
        pytest.skip("Modèle unifié absent.")
    row = {
        "Ticker": "ZAL.DE",
        "Dernier cours": 22.93,
        "PER": 47.6,
        "Cours / val. comptable": 2.08,
        "ROE": 4.37,
        "ROIC": 4.79,
        "Upside vs objectif": 52.6,
    }
    pred = fv.predict_fair_value_for_fundamentals([row], models, {})
    jv = float(pred["Juste valeur estimée"].iloc[0])
    assert abs(jv - 34.75) < 3.0, jv


def test_etl_anchor_in_training_frame():
    anchors = fv.load_investing_anchors(ROOT / "investing_anchors.yaml")
    etl = next(a for a in anchors if a.get("ticker") == "ETL.PA")
    assert etl["fair_value"] == pytest.approx(2.84)
    assert etl["price"] == pytest.approx(3.50)
    assert etl["label"] == "Surévaluée"
    assert etl["comparable_fv"] == pytest.approx(3.32)
    assert etl["dcf_fv"] == pytest.approx(2.50)


def test_lr_anchor_in_training_frame():
    anchors = fv.load_investing_anchors(ROOT / "investing_anchors.yaml")
    lr = next(a for a in anchors if a.get("ticker") == "LR.PA")
    assert lr["fair_value"] == pytest.approx(135.91)
    assert lr["price"] == pytest.approx(149.0)
    assert lr["label"] == "Juste"
    assert lr["comparable_fv"] == pytest.approx(135.80)
    assert lr["dcf_fv"] == pytest.approx(136.49)
    assert lr["ddm_fv"] == pytest.approx(134.53)


def test_su_anchor_in_training_frame():
    anchors = fv.load_investing_anchors(ROOT / "investing_anchors.yaml")
    su = next(a for a in anchors if a.get("ticker") == "SU.PA")
    assert su["fair_value"] == pytest.approx(250.79)
    assert su["price"] == pytest.approx(287.15)
