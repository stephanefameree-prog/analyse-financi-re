"""
Calibrage d'une « juste valeur » à partir d'exports Investing.com.

Investing publie une juste valeur numérique (colonne « Juste Valeur ») et un libellé
(Aubaine, Sous-évaluée, Juste, Surévaluée) dérivé de l'écart au cours.

Ce module ne reproduit pas l'algorithme propriétaire d'Investing : il combine des
estimateurs inspirés de la documentation InvestingPro (Graham, multiples PER/P/B/P/S,
dividende, endettement, RSI) calibrés sur vos exports Premium pour minimiser l'écart
aux justes valeurs Investing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

INVESTING_HEADER_ROW = 7
INVESTING_SHEET = "sheet"

COL_NAME = "Nom"
COL_TICKER = "Full Ticker"
COL_PRICE = "Prix, actuel"
COL_FAIR_VALUE = "Juste Valeur"
COL_FAIR_LABEL = "Label de la juste valeur"
COL_RSI = "Relative Strength Index (14j)"
COL_PER = "PER"
COL_PB = "Ratio P/B"
COL_PS = "Prix / Ventes LTM"
COL_EV = "Valeur de l\u2019entreprise (EV)"
COL_DEBT_CAP = "Dette Totale/Capital Total"
COL_FLOAT = "Actions flottantes / En circulation"
COL_DIV_YIELD = "Rendement de dividende"
COL_MCAP = "Capitalisation boursi\u00e8re (ajust\u00e9)"

# Export « Minières - ma vue » (vues personnalisées InvestingPro)
COL_GRAHAM_VAL = "Formule Ben Graham Valeur"
COL_GRAHAM_UP = "Formule Ben Graham Upside"
COL_FV_COMPARABLE = "Juste valeur comparable"
COL_FV_DCF = "Juste valeur DCF"
COL_FV_DDM = "Juste valeur DDM"
COL_FV_EPV = "Juste valeur EPV"
COL_ROE_MA_VUE = "Rendement de Capitaux Propres Ordinaires"
COL_ROIC_MA_VUE = "Rendement des Capitaux Investis"
COL_PER_FWD_MA_VUE = "Ratio P/E (Fwd)"
COL_PEG_MA_VUE = "Ratio PEG"
COL_BETA_MA_VUE = "B\u00eata (5 ans)"
COL_EPS_GROWTH_MA_VUE = "Croissance du BPA de Base"

MA_VUE1_GLOB = "*ma vue - 2026-06-02*.xlsx"
MA_VUE2_GLOB = "*ma vue 2*2026-06-02*.xlsx"

TECH_EFFICIENCY_GLOB = "*tech*Efficiency*2026-06-02*.xlsx"
TECH_RISK_GLOB = "*tech*Risk*2026-06-02*.xlsx"
TECH_MA_VUE2_GLOB = "*tech*ma vue 2*2026-06-02*.xlsx"
TECH_MA_VUE3_GLOB = "*tech*ma vue 3*2026-06-02*.xlsx"

COL_FV_COMP_UP = "Hausse de la juste valeur comparable"
COL_FV_DCF_UP = "Hausse de la juste valeur DCF"
COL_FV_DDM_UP = "Hausse de la juste valeur DDM"

SECTOR_MINIERES = "minières"
SECTOR_TECH = "tech"
SECTOR_BIOTECH = "biotech"
SECTOR_PHARMA = "pharma"
SECTOR_ENERGY = "énergie"
SECTOR_DEFENSE = "défense"
SECTOR_FOOD = "alimentation"
SECTOR_FINANCE = "finance"
SECTOR_CONSUMER = "consommation"
SECTOR_INDUSTRIAL = "industrie"
# Modèle Ridge unique en prédiction dashboard (calibré minières + tech).
SECTOR_UNIFIED = "unifié"

# Routage multiples sectoriels si le titre n'est pas listé dans les exports ma vue.
YAHOO_SECTORS_TECH = frozenset({"Technology", "Communication Services"})
YAHOO_SECTORS_MINING = frozenset({"Basic Materials"})
YAHOO_SECTORS_ENERGY = frozenset({"Energy"})
YAHOO_SECTORS_HEALTHCARE = frozenset({"Healthcare"})
YAHOO_SECTORS_DEFENSIVE = frozenset({"Consumer Defensive"})
YAHOO_SECTORS_CYCLICAL = frozenset({"Consumer Cyclical"})
YAHOO_SECTORS_INDUSTRIALS = frozenset({"Industrials"})
YAHOO_SECTORS_FINANCIAL = frozenset({"Financial Services", "Financial"})
TECH_INDUSTRY_KEYWORDS = (
    "software",
    "semiconductor",
    "internet",
    "telecom",
    "communication",
    "information technology",
    "electronic",
)
MINING_INDUSTRY_KEYWORDS = (
    "gold",
    "mining",
    "metal",
    "mineral",
    "steel",
    "copper",
    "silver",
    "uranium",
)
BIOTECH_INDUSTRY_KEYWORDS = ("biotech", "biotechnology", "genomics")
PHARMA_INDUSTRY_KEYWORDS = (
    "pharma",
    "pharmaceutical",
    "drug",
    "medicinal",
    "medical care",
    "medical device",
    "medical instrument",
    "diagnostic",
    "health care",
)
DEFENSE_INDUSTRY_KEYWORDS = (
    "defense",
    "defence",
    "aerospace",
    "armament",
    "military",
    "naval",
)
FOOD_INDUSTRY_KEYWORDS = (
    "food",
    "beverage",
    "restaurant",
    "grocery",
    "agricultural",
    "farm",
    "packaged foods",
    "confectioners",
)
ENERGY_INDUSTRY_KEYWORDS = (
    "oil",
    "gas",
    "petroleum",
    "renewable",
    "solar",
    "wind energy",
    "utilities",
)
CONSUMER_INDUSTRY_KEYWORDS = (
    "retail",
    "apparel",
    "fashion",
    "department store",
    "home improvement",
    "auto dealers",
    "lodging",
    "leisure",
    "travel",
    "resort",
    "gambling",
    "luxury",
    "personal products",
    "footwear",
)

TECH_FAIR_MULTIPLES = {
    "per": 28.0,
    "pb": 5.0,
    "ps": 8.0,
    "div_yield": 0.01,
}

SECTOR_PROFILE_MULTIPLES: dict[str, dict[str, float]] = {
    SECTOR_MINIERES: {"per": 15.0, "pb": 1.35, "ps": 2.0, "div_yield": 0.035},
    SECTOR_TECH: TECH_FAIR_MULTIPLES,
    SECTOR_BIOTECH: {"per": 24.0, "pb": 4.5, "ps": 7.0, "div_yield": 0.0},
    SECTOR_PHARMA: {"per": 18.0, "pb": 3.5, "ps": 4.5, "div_yield": 0.02},
    SECTOR_ENERGY: {"per": 11.0, "pb": 1.25, "ps": 1.2, "div_yield": 0.045},
    SECTOR_DEFENSE: {"per": 17.0, "pb": 2.8, "ps": 1.6, "div_yield": 0.015},
    SECTOR_FOOD: {"per": 17.0, "pb": 2.2, "ps": 1.4, "div_yield": 0.03},
    SECTOR_FINANCE: {"per": 10.0, "pb": 1.05, "ps": 2.5, "div_yield": 0.035},
    SECTOR_CONSUMER: {"per": 20.0, "pb": 3.5, "ps": 2.2, "div_yield": 0.015},
    SECTOR_INDUSTRIAL: {"per": 15.0, "pb": 2.0, "ps": 1.6, "div_yield": 0.025},
}

_LEGACY_SECTOR_PROFILE_MULTIPLES = SECTOR_PROFILE_MULTIPLES

BENCHMARKS_PATH = Path(__file__).resolve().parent / "sector_benchmarks.yaml"
INVESTING_ANCHORS_PATH = Path(__file__).resolve().parent / "investing_anchors.yaml"
_BENCHMARKS_CACHE: dict | None = None


def load_sector_benchmarks(path: str | Path | None = None, *, force_reload: bool = False) -> dict:
    """Charge sector_benchmarks.yaml (Damodaran secteur + pays)."""
    global _BENCHMARKS_CACHE
    benchmark_path = Path(path) if path else BENCHMARKS_PATH
    if not force_reload and path is None and _BENCHMARKS_CACHE is not None:
        return _BENCHMARKS_CACHE

    if not benchmark_path.is_file():
        cfg = {"profiles": {}, "countries": {"default": {"per": 1.0, "pb": 1.0, "ps": 1.0, "div_yield": 1.0}}}
        if path is None:
            _BENCHMARKS_CACHE = cfg
        return cfg

    try:
        import yaml

        with open(benchmark_path, encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}
    except Exception:
        cfg = {"profiles": {}, "countries": {"default": {"per": 1.0, "pb": 1.0, "ps": 1.0, "div_yield": 1.0}}}

    if path is None:
        _BENCHMARKS_CACHE = cfg
    return cfg


def get_base_profile_multiples(benchmarks: Mapping | None = None) -> dict[str, dict[str, float]]:
    """Multiples US de base par profil (sans ajustement pays)."""
    cfg = benchmarks or load_sector_benchmarks()
    profiles = cfg.get("profiles") or {}
    if not profiles:
        return dict(_LEGACY_SECTOR_PROFILE_MULTIPLES)
    out: dict[str, dict[str, float]] = {}
    for name, block in profiles.items():
        multiples = block.get("multiples") if isinstance(block, dict) else None
        if isinstance(multiples, dict) and multiples:
            out[str(name)] = {k: float(v) for k, v in multiples.items()}
    return out or dict(_LEGACY_SECTOR_PROFILE_MULTIPLES)


def _country_for_ticker(ticker: str, cfg: Mapping) -> str:
    baseline = (
        cfg.get("baseline_country")
        or (cfg.get("meta") or {}).get("baseline_country")
        or "United States"
    )
    tk = str(ticker or "").strip().upper()
    if "." not in tk:
        return str(baseline)

    suffix = tk.rsplit(".", 1)[-1]
    suffix_map = cfg.get("ticker_suffix") or {}
    countries = cfg.get("countries") or {}
    country = suffix_map.get(suffix)
    if country and country in countries:
        return str(country)
    if country == "Europe" and "Europe" in countries:
        return "Europe"
    if "default" in countries:
        return "default"
    return str(baseline)


def resolve_profile_multiples(
    profile: str,
    ticker: str | None = None,
    *,
    benchmarks: Mapping | None = None,
) -> dict[str, float]:
    """
    Multiples cibles pour un profil sectoriel, ajustés au pays du ticker (suffixe Yahoo).
    Référence : sector_benchmarks.yaml (Damodaran, janv. 2026).
    """
    cfg = benchmarks or load_sector_benchmarks()
    base_profiles = get_base_profile_multiples(cfg)
    base = dict(
        base_profiles.get(profile)
        or base_profiles.get(SECTOR_MINIERES)
        or _LEGACY_SECTOR_PROFILE_MULTIPLES.get(profile, {})
    )
    if not base:
        return {}

    country_key = _country_for_ticker(ticker or "", cfg)
    countries = cfg.get("countries") or {}
    scales = countries.get(country_key) or countries.get("default") or {"per": 1.0, "pb": 1.0, "ps": 1.0, "div_yield": 1.0}

    out: dict[str, float] = {}
    for key in ("per", "pb", "ps", "div_yield"):
        if key not in base:
            continue
        scale = float(scales.get(key, 1.0))
        val = float(base[key]) * scale
        if key == "div_yield":
            out[key] = min(max(val, 0.0), 0.12)
        else:
            out[key] = round(val, 4)
    return out


def resolve_unified_multiples(ticker: str | None = None) -> dict[str, float]:
    """
    Multiples PER/P/B/P/S/div cibles identiques pour tous les titres en prédiction dashboard.
    Alignés sur le calibrage Ridge (DEFAULT_FAIR_MULTIPLES), sans repères Damodaran par secteur.
    """
    _ = ticker
    return dict(DEFAULT_FAIR_MULTIPLES)


def _normalize_div_yield_decimal(value) -> float | None:
    v = pd.to_numeric(value, errors="coerce")
    if pd.isna(v) or float(v) <= 0:
        return None
    v = float(v)
    if v > 1.5:
        v = v / 100.0
    return v


def _yahoo_profile_sets(
    enrich_indices: Mapping[str, Mapping[str, Mapping]] | None,
) -> tuple[frozenset[str], frozenset[str]]:
    tech: set[str] = set()
    mining: set[str] = set()
    if enrich_indices:
        tech.update((enrich_indices.get(SECTOR_TECH) or {}).keys())
        mining.update((enrich_indices.get(SECTOR_MINIERES) or {}).keys())
    return frozenset(tech), frozenset(mining)


def compute_damodaran_fair_value(
    row: Mapping,
    *,
    tech_yahoo: set[str] | frozenset | None = None,
    mining_yahoo: set[str] | frozenset | None = None,
    benchmarks: Mapping | None = None,
) -> dict[str, float | str | None]:
    """
    Juste valeur indicative : multiples cibles sector_benchmarks.yaml (Damodaran US + ajustement pays).
    Moyenne des prix implicites PER / P/B / P/S / dividende disponibles (médiane).
    """
    mapped = map_fundamentals_row(row)
    price = pd.to_numeric(mapped.get(COL_PRICE), errors="coerce")
    if pd.isna(price) or float(price) <= 0:
        return {
            "Juste valeur Damodaran": None,
            "Upside Damodaran": None,
            "Profil Damodaran": None,
        }

    ticker = str(row.get("Ticker", "")).strip()
    yahoo_sector = row.get("Secteur Yahoo") or row.get("Secteur")
    yahoo_industry = row.get("Industrie Yahoo")
    profile = resolve_fair_value_profile(
        ticker,
        tech_yahoo or frozenset(),
        mining_yahoo or frozenset(),
        yahoo_sector=yahoo_sector,
        yahoo_industry=yahoo_industry,
    )
    fm = resolve_profile_multiples(profile, ticker, benchmarks=benchmarks)
    if not fm:
        return {
            "Juste valeur Damodaran": None,
            "Upside Damodaran": None,
            "Profil Damodaran": profile,
        }

    price_f = float(price)
    per = pd.to_numeric(mapped.get(COL_PER), errors="coerce")
    pb = pd.to_numeric(mapped.get(COL_PB), errors="coerce")
    ps = pd.to_numeric(mapped.get(COL_PS), errors="coerce")
    div_y = _normalize_div_yield_decimal(mapped.get(COL_DIV_YIELD))

    implied: list[float] = []
    if pd.notna(per) and float(per) > 0 and fm.get("per"):
        implied.append(price_f * float(fm["per"]) / float(per))
    if pd.notna(pb) and float(pb) > 0 and fm.get("pb"):
        implied.append(price_f * float(fm["pb"]) / float(pb))
    if pd.notna(ps) and float(ps) > 0 and fm.get("ps"):
        implied.append(price_f * float(fm["ps"]) / float(ps))
    if div_y and fm.get("div_yield") and float(fm["div_yield"]) > 0:
        implied.append(price_f * float(fm["div_yield"]) / div_y)

    if not implied:
        return {
            "Juste valeur Damodaran": None,
            "Upside Damodaran": None,
            "Profil Damodaran": profile,
        }

    jv = float(np.median(implied))
    return {
        "Juste valeur Damodaran": jv,
        "Upside Damodaran": jv / price_f - 1.0,
        "Profil Damodaran": profile,
    }


# Profils US de base (YAML) — repères affichage / scripts ; pas utilisés en prédiction JV dashboard.
SECTOR_PROFILE_MULTIPLES = get_base_profile_multiples()

# Rétrocompat tests / docs : tous les profils utilisent le modèle unifié en prédiction.
PROFILE_BASE_MODEL: dict[str, str] = {
    profile: SECTOR_UNIFIED
    for profile in (
        SECTOR_MINIERES,
        SECTOR_TECH,
        SECTOR_BIOTECH,
        SECTOR_PHARMA,
        SECTOR_ENERGY,
        SECTOR_DEFENSE,
        SECTOR_FOOD,
        SECTOR_FINANCE,
        SECTOR_CONSUMER,
        SECTOR_INDUSTRIAL,
    )
}

NUMERIC_INVESTING_COLUMNS = (
    COL_PRICE,
    COL_FAIR_VALUE,
    COL_RSI,
    COL_PER,
    COL_PB,
    COL_PS,
    COL_EV,
    COL_DEBT_CAP,
    COL_FLOAT,
    COL_DIV_YIELD,
    COL_MCAP,
)

# Seuils observés sur les exports Investing (upside = fair_value / price - 1).
DEFAULT_LABEL_THRESHOLDS = {
    "aubaine_min": 0.50,
    "sous_min": 0.185,
    "sureval_max": -0.19,
}

FUNDAMENTALS_TO_INVESTING = {
    "Dernier cours": COL_PRICE,
    "Prix": COL_PRICE,
    "PER": COL_PER,
    "PER forward": COL_PER_FWD_MA_VUE,
    "Cours / val. comptable": COL_PB,
    "Cours / ventes": COL_PS,
    "Rendement dividende (%)": COL_DIV_YIELD,
    "Rendement dividende": COL_DIV_YIELD,
    "Rendement actionnaire": COL_DIV_YIELD,
    "Dette / Capitaux": COL_DEBT_CAP,
    "RSI (14)": COL_RSI,
    "Capitalisation de marché": COL_MCAP,
    "ROE": "ROE",
    "ROIC": "ROIC",
    "Marge nette": "Marge nette",
    "Upside vs objectif": "Upside vs objectif",
    "Règle des 40": "Règle des 40",
    "Croissance BPA": "EPS growth",
}

# Multiples cibles « neutres » (ajustés par régression sur vos exports).
DEFAULT_FAIR_MULTIPLES = {
    "per": 15.0,
    "pb": 1.35,
    "ps": 2.0,
    "div_yield": 0.035,
}

PILLAR_DESCRIPTIONS = {
    "pillar_graham_log": "Ben Graham : sqrt(22.5 x BPA x val. comptable)",
    "pillar_per_log": "Multiple PER sectoriel (BPA × PER cible / cours)",
    "pillar_pb_log": "Valeur comptable (P/B cible / P/B actuel)",
    "pillar_ps_log": "Multiple ventes (P/S cible / P/S actuel)",
    "pillar_div_log": "Rendement dividende (cible / rendement actuel)",
    "pillar_debt_adj": "Structure financière (dettes / capitaux)",
    "pillar_rsi_adj": "Ajustement technique RSI 14j",
    "pillar_investing_comparable_log": "Juste valeur comparable Investing / cours",
    "pillar_investing_dcf_log": "Juste valeur DCF Investing / cours",
    "pillar_investing_ddm_log": "Juste valeur DDM Investing / cours",
    "pillar_investing_epv_log": "Juste valeur EPV Investing / cours",
    "pillar_investing_graham_export_log": "Formule Ben Graham Investing / cours",
    "pillar_roe_adj": "Ajustement ROE",
    "pillar_roic_adj": "Ajustement ROIC",
    "pillar_rule40_adj": "Règle des 40 (croissance + marge, cible 40 %)",
}

# Piliers toujours présents dans la matrice (NaN → 0 à l'entraînement / prédiction).
CANONICAL_PILLAR_COLUMNS = (
    "pillar_graham_log",
    "pillar_per_log",
    "pillar_pb_log",
    "pillar_ps_log",
    "pillar_div_log",
    "pillar_debt_adj",
    "pillar_rsi_adj",
    "pillar_roe_adj",
    "pillar_margin_adj",
    "pillar_analyst_log",
    "pillar_investing_comparable_log",
    "pillar_investing_dcf_log",
    "pillar_investing_ddm_log",
    "pillar_investing_epv_log",
    "pillar_investing_graham_export_log",
    "pillar_roic_adj",
    "pillar_eps_growth_adj",
    "pillar_rule40_adj",
)

CORE_PILLAR_COLUMNS = (
    "pillar_graham_log",
    "pillar_per_log",
    "pillar_pb_log",
    "pillar_ps_log",
    "pillar_div_log",
    "pillar_debt_adj",
    "pillar_rsi_adj",
)


@dataclass
class FairValueMetrics:
    n: int
    mape: float
    mae: float
    rmse: float
    r2: float
    label_accuracy: float | None = None

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "mape": self.mape,
            "mae": self.mae,
            "rmse": self.rmse,
            "r2": self.r2,
            "label_accuracy": self.label_accuracy,
        }


@dataclass
class FairValueModel:
    """Modèle calibré : prédit log(fair_value / price) puis reconstitue la juste valeur."""

    method: str = "pillars_ridge"
    sector: str = SECTOR_MINIERES
    fair_multiples: dict = field(default_factory=lambda: dict(DEFAULT_FAIR_MULTIPLES))
    random_state: int = 42
    label_thresholds: dict = field(default_factory=lambda: dict(DEFAULT_LABEL_THRESHOLDS))
    feature_columns_: list[str] = field(default_factory=list, init=False)
    pipeline_: Pipeline | None = field(default=None, init=False)
    metrics_: FairValueMetrics | None = field(default=None, init=False)

    def _build_estimator(self):
        if self.method == "gradient_boosting":
            return GradientBoostingRegressor(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.08,
                random_state=self.random_state,
            )
        if self.method in ("ridge", "pillars_ridge"):
            return Ridge(alpha=5.0)
        raise ValueError(f"Méthode inconnue : {self.method}")

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.method == "pillars_ridge":
            return build_pillar_features(df, fair_multiples=self.fair_multiples)
        return build_fair_value_features(df)

    def fit(self, df: pd.DataFrame) -> "FairValueModel":
        X, y, _price, _labels = self._prepare_training(df)
        if self.method == "pillars_ridge":
            self.feature_columns_ = list(CANONICAL_PILLAR_COLUMNS)
            X = align_feature_matrix(X, self.feature_columns_)
        else:
            self.feature_columns_ = list(X.columns)
        self.pipeline_ = Pipeline(
            [("scale", StandardScaler()), ("model", self._build_estimator())]
        )
        y_log_ratio = np.log(y / _price)
        self.pipeline_.fit(X, y_log_ratio)
        pred = self.predict_dataframe(df)
        self.metrics_ = evaluate_predictions(df, pred, label_thresholds=self.label_thresholds)
        return self

    def predict_dataframe(
        self,
        df: pd.DataFrame,
        fair_multiples: Mapping[str, float] | None = None,
    ) -> pd.DataFrame:
        if self.pipeline_ is None:
            raise RuntimeError("Modèle non entraîné : appelez fit() d'abord.")
        work = normalize_investing_frame(df)
        fm = {**self.fair_multiples, **(fair_multiples or {})}
        if self.method == "pillars_ridge":
            X_full = build_pillar_features(work, fair_multiples=fm)
        else:
            X_full = self._build_features(work)
        price_full = work["price"].astype(float)
        out = work[[c for c in (COL_NAME, COL_TICKER) if c in work.columns]].copy()
        out["price"] = price_full
        out["fair_value_pred"] = np.nan
        out["upside_pred"] = np.nan
        out["label_pred"] = ""

        mask = _usable_feature_mask(X_full, self.method) & price_full.notna() & (price_full > 0)
        if not mask.any():
            return out

        feature_cols = (
            list(CANONICAL_PILLAR_COLUMNS)
            if self.method == "pillars_ridge"
            else self.feature_columns_
        )
        X = align_feature_matrix(X_full.loc[mask], feature_cols)
        price = price_full.loc[mask]
        log_ratio = self.pipeline_.predict(X)
        fair_value = price * np.exp(log_ratio)
        upside = fair_value / price - 1.0
        out.loc[mask, "fair_value_pred"] = fair_value
        out.loc[mask, "upside_pred"] = upside
        out.loc[mask, "label_pred"] = classify_upside_many(upside.values, self.label_thresholds)

        blend = _investing_submodel_blend(work.loc[mask], price)
        if blend.notna().any():
            use_blend = blend.notna()
            fair_value = fair_value.copy()
            fair_value.loc[use_blend] = (
                0.35 * fair_value.loc[use_blend] + 0.65 * blend.loc[use_blend]
            )
            upside = fair_value / price - 1.0
            out.loc[mask, "fair_value_pred"] = fair_value
            out.loc[mask, "upside_pred"] = upside
            out.loc[mask, "label_pred"] = classify_upside_many(upside.values, self.label_thresholds)
            out.loc[mask, "fair_value_blend"] = blend

        if "fair_value" in work.columns or COL_FAIR_VALUE in work.columns:
            fv_col = work["fair_value"] if "fair_value" in work.columns else work[COL_FAIR_VALUE]
            out["fair_value_investing"] = pd.to_numeric(fv_col, errors="coerce")
            out["upside_investing"] = out["fair_value_investing"] / price_full - 1.0
            if "label" in work.columns:
                out["label_investing"] = work["label"]
            elif COL_FAIR_LABEL in work.columns:
                out["label_investing"] = work[COL_FAIR_LABEL]
        return out

    def cross_validate(self, df: pd.DataFrame, n_splits: int = 5) -> FairValueMetrics:
        X, y, price, labels = self._prepare_training(df)
        if self.method == "pillars_ridge":
            X = align_feature_matrix(X, CANONICAL_PILLAR_COLUMNS)
        y_log_ratio = np.log(y / price)
        pipe = Pipeline([("scale", StandardScaler()), ("model", self._build_estimator())])
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        pred_log = cross_val_predict(pipe, X, y_log_ratio, cv=cv)
        pred_fv = price * np.exp(pred_log)
        return _metrics_from_arrays(y, pred_fv, labels, pred_log_ratio=pred_log, price=price)

    def _prepare_training(self, df: pd.DataFrame, require_target: bool = True):
        work = normalize_investing_frame(df)
        X = self._build_features(work)
        price = work["price"].astype(float)
        if require_target:
            y = work["fair_value"].astype(float)
            labels = work["label"].astype(str)
            mask = (
                _usable_feature_mask(X, self.method)
                & y.notna()
                & price.notna()
                & (y > 0)
                & (price > 0)
            )
            X_out = X.loc[mask]
            if self.method == "pillars_ridge":
                X_out = X_out.fillna(0.0)
            return X_out, y.loc[mask], price.loc[mask], labels.loc[mask]
        mask = _usable_feature_mask(X, self.method) & price.notna() & (price > 0)
        labels = work.get("label", pd.Series(index=work.index, dtype=str)).loc[mask]
        y = work.get("fair_value", pd.Series(index=work.index, dtype=float)).loc[mask]
        return X.loc[mask], y, price.loc[mask], labels


def glob_ma_vue_exports(directory: str | Path = ".") -> tuple[Path | None, Path | None]:
    """Repère les fichiers ma vue 1 et ma vue 2 dans un dossier."""
    directory = Path(directory)
    vue1 = next(iter(sorted(directory.glob(MA_VUE1_GLOB))), None)
    vue2 = next(iter(sorted(directory.glob(MA_VUE2_GLOB))), None)
    return vue1, vue2


def glob_tech_exports(
    directory: str | Path = ".",
) -> tuple[Path | None, Path | None, Path | None, Path | None]:
    """Repère tech ma vue 2, ma vue 3, Efficiency et Risk."""
    directory = Path(directory)
    vue2 = next(iter(sorted(directory.glob(TECH_MA_VUE2_GLOB))), None)
    vue3 = next(iter(sorted(directory.glob(TECH_MA_VUE3_GLOB))), None)
    efficiency = next(iter(sorted(directory.glob(TECH_EFFICIENCY_GLOB))), None)
    risk = next(iter(sorted(directory.glob(TECH_RISK_GLOB))), None)
    return vue2, efficiency, risk, vue3


def _merge_export_supplement(base: pd.DataFrame, extra: pd.DataFrame | None) -> pd.DataFrame:
    if extra is None or extra.empty or COL_TICKER not in extra.columns:
        return base
    skip = {COL_NAME, COL_TICKER, "Ticker", "Unnamed: 0", "Unnamed: 1", "source_file"}
    cols = [COL_TICKER] + [c for c in extra.columns if c not in skip and c not in base.columns]
    if len(cols) <= 1:
        return base
    return base.merge(extra[cols], on=COL_TICKER, how="left")


def investing_ticker_to_yahoo(full_ticker: str) -> str:
    """Convertit un Full Ticker Investing (ex. ENXTPA:AIR) en symbole Yahoo."""
    tk = str(full_ticker).strip().upper()
    if ":" not in tk:
        return tk.replace(".US", "")
    exchange, symbol = tk.split(":", 1)
    exchange = exchange.upper()
    mapping = {
        "ENXTPA": ".PA",
        "ENXTBR": ".BR",
        "ENXTAM": ".AS",
        "XTRA": ".DE",
        "XETRA": ".DE",
        "DB": ".DE",
        "NYSE": "",
        "NASDAQGS": "",
        "NASDAQGM": "",
        "NASDAQCM": "",
        "NYSEAM": "",
        "OTCPK": "",
        "TSX": ".TO",
        "TSXV": ".V",
    }
    suffix = mapping.get(exchange, "")
    return f"{symbol}{suffix}" if suffix else symbol


def derive_price_from_graham(df: pd.DataFrame) -> pd.Series:
    """Cours implicite : Graham / (1 + upside Graham)."""
    graham = pd.to_numeric(df.get(COL_GRAHAM_VAL), errors="coerce")
    upside = pd.to_numeric(df.get(COL_GRAHAM_UP), errors="coerce")
    denom = 1.0 + upside
    price = graham / denom
    return price.where((graham > 0) & (denom > 0.05))


def derive_price_from_comparable_upside(df: pd.DataFrame) -> pd.Series:
    """Cours implicite : comparable / (1 + upside comparable)."""
    comp = pd.to_numeric(df.get(COL_FV_COMPARABLE), errors="coerce")
    upside = pd.to_numeric(df.get(COL_FV_COMP_UP), errors="coerce")
    denom = 1.0 + upside
    price = comp / denom
    return price.where((comp > 0) & (denom > 0.05))


def _ensure_fair_value_and_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Dérive juste valeur (blend sous-modèles) et libellé si absents (exports tech)."""
    out = df.copy()
    if COL_FAIR_VALUE in out.columns:
        out["fair_value"] = pd.to_numeric(out[COL_FAIR_VALUE], errors="coerce")
    elif "fair_value" not in out.columns:
        out["fair_value"] = np.nan
    else:
        out["fair_value"] = pd.to_numeric(out["fair_value"], errors="coerce")

    if "price" not in out.columns:
        out["price"] = np.nan
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    missing_price = out["price"].isna() | (out["price"] <= 0)
    if missing_price.any():
        implied = derive_price_from_comparable_upside(out)
        out.loc[missing_price, "price"] = implied.loc[missing_price]

    price = out["price"].astype(float)
    needs_fv = out["fair_value"].isna() | (out["fair_value"] <= 0)
    if needs_fv.any():
        blend = _investing_submodel_blend(out, price)
        comp = pd.to_numeric(out.get(COL_FV_COMPARABLE), errors="coerce")
        derived = blend.where(blend.notna(), comp)
        out.loc[needs_fv, "fair_value"] = derived.loc[needs_fv]

    if COL_FAIR_LABEL in out.columns and out[COL_FAIR_LABEL].astype(str).str.strip().ne("").any():
        out["label"] = out[COL_FAIR_LABEL].astype(str).str.strip()
    else:
        upside = out["fair_value"] / price - 1.0
        out["label"] = [
            classify_upside(float(u), DEFAULT_LABEL_THRESHOLDS) if pd.notna(u) else ""
            for u in upside
        ]
    return out


def enrich_price_from_yahoo(df: pd.DataFrame) -> pd.DataFrame:
    """Complète price manquant via yfinance (best effort)."""
    out = df.copy()
    if "price" not in out.columns:
        out["price"] = np.nan
    missing = out["price"].isna() | (out["price"] <= 0)
    if not missing.any() or COL_TICKER not in out.columns:
        return out
    try:
        import yfinance as yf
    except ImportError:
        return out

    for idx in out.index[missing]:
        yahoo_tk = investing_ticker_to_yahoo(out.at[idx, COL_TICKER])
        if not yahoo_tk:
            continue
        try:
            info = yf.Ticker(yahoo_tk).info or {}
            lp = info.get("regularMarketPrice") or info.get("currentPrice")
            if lp and float(lp) > 0:
                out.at[idx, "price"] = float(lp)
        except Exception:
            continue
    return out


def load_ma_vue_export(path: str | Path, sheet: str = INVESTING_SHEET) -> pd.DataFrame:
    """Charge une vue ma vue Investing (.xlsx) sans fusion."""
    path = Path(path)
    df = pd.read_excel(path, sheet_name=sheet, header=INVESTING_HEADER_ROW)
    df = df.dropna(subset=[COL_NAME], how="any")
    df["source_file"] = path.name
    return df


def load_ma_vue(
    vue1_path: str | Path | None = None,
    vue2_path: str | Path | None = None,
    directory: str | Path = ".",
    enrich_yahoo_price: bool = True,
) -> pd.DataFrame:
    """
    Fusionne « Minières - ma vue » et « ma vue 2 », harmonise colonnes
    et dérive le cours (Graham puis Yahoo si demandé).
    """
    directory = Path(directory)
    if vue1_path is None:
        vue1_path, vue2_default = glob_ma_vue_exports(directory)
        if vue2_path is None:
            vue2_path = vue2_default
    if vue1_path is None:
        raise FileNotFoundError(f"Export ma vue 1 introuvable ({MA_VUE1_GLOB}).")

    v1 = load_ma_vue_export(vue1_path)
    if vue2_path and Path(vue2_path).exists():
        v2 = load_ma_vue_export(vue2_path)
        df = v1.merge(v2, on=COL_TICKER, suffixes=("", "_v2"), how="left")
        df["source_file"] = f"{Path(vue1_path).name} + {Path(vue2_path).name}"
    else:
        df = v1.copy()

    df = _harmonize_ma_vue_columns(df)
    df["price"] = derive_price_from_graham(df)
    if enrich_yahoo_price:
        df = enrich_price_from_yahoo(df)
    return normalize_investing_frame(df)


def load_tech_ma_vue(
    vue2_path: str | Path | None = None,
    efficiency_path: str | Path | None = None,
    risk_path: str | Path | None = None,
    vue3_path: str | Path | None = None,
    directory: str | Path = ".",
    enrich_yahoo_price: bool = True,
) -> pd.DataFrame:
    """
    Fusionne tech ma vue 2 + ma vue 3 + Efficiency + Risk, dérive cours et juste valeur
    (blend comparable/DCF/DDM/EPV — pas de colonne « Juste Valeur » dans l'export).
    """
    directory = Path(directory)
    if vue2_path is None:
        vue2_path, efficiency_path, risk_path, vue3_path = glob_tech_exports(directory)
    if vue2_path is None:
        raise FileNotFoundError(f"Export tech ma vue 2 introuvable ({TECH_MA_VUE2_GLOB}).")

    df = load_ma_vue_export(vue2_path)
    if vue3_path and Path(vue3_path).exists():
        df = _merge_export_supplement(df, load_ma_vue_export(vue3_path))
    if efficiency_path and Path(efficiency_path).exists():
        df = _merge_export_supplement(df, load_ma_vue_export(efficiency_path))
    if risk_path and Path(risk_path).exists():
        df = _merge_export_supplement(df, load_ma_vue_export(risk_path))

    df = _harmonize_ma_vue_columns(df)
    df = _harmonize_tech_supplement(df)
    df = _ensure_fair_value_and_labels(df)
    df["price"] = derive_price_from_comparable_upside(df).fillna(df["price"])
    if enrich_yahoo_price:
        df = enrich_price_from_yahoo(df)
    df = _ensure_fair_value_and_labels(df)
    return normalize_investing_frame(df)


def _harmonize_ma_vue_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Mappe les colonnes ma vue vers le schéma standard du module."""
    out = df.copy()
    if COL_FAIR_VALUE in out.columns:
        out["fair_value"] = pd.to_numeric(out[COL_FAIR_VALUE], errors="coerce")

    alias_map = {
        COL_PER: ["PER"],
        COL_PB: ["Ratio P/B"],
        COL_DIV_YIELD: ["Rendement de dividende"],
        COL_DEBT_CAP: ["Dette Totale/Capital Total"],
        "ROE": [COL_ROE_MA_VUE],
        "ROIC": [COL_ROIC_MA_VUE],
        "PER forward": [COL_PER_FWD_MA_VUE],
        "PEG": [COL_PEG_MA_VUE],
        "Beta": [COL_BETA_MA_VUE],
        "EPS growth": [COL_EPS_GROWTH_MA_VUE],
        COL_FV_COMPARABLE: [COL_FV_COMPARABLE],
        COL_FV_DCF: [COL_FV_DCF],
        COL_FV_DDM: [COL_FV_DDM],
        COL_FV_EPV: [COL_FV_EPV],
        COL_GRAHAM_VAL: [COL_GRAHAM_VAL],
        COL_GRAHAM_UP: [COL_GRAHAM_UP],
    }
    for dst, candidates in alias_map.items():
        for src in candidates:
            if src in out.columns:
                out[dst] = pd.to_numeric(out[src], errors="coerce")
                break
    return out


def _find_investing_column(df: pd.DataFrame, *needles: str) -> str | None:
    """Repère une colonne Investing par sous-chaînes (insensible à la casse)."""
    lowered = [n.lower() for n in needles]
    for col in df.columns:
        label = str(col).lower()
        if all(n in label for n in lowered):
            return col
    return None


def _harmonize_tech_supplement(df: pd.DataFrame) -> pd.DataFrame:
    """Mappe colonnes tech ma vue 2/3, Efficiency et Risk vers le schéma standard."""
    out = df.copy()
    alias_map = {
        COL_PER: ["Ratio PER prévu", "PER"],
        COL_PB: ["Ratio P/B"],
        COL_PS: ["Prix / Ventes LTM"],
        COL_DIV_YIELD: ["Ratio de paiement, ordinaire", "Rendement de dividende"],
        COL_DEBT_CAP: ["Dette Totale/Capital Total"],
        "ROE": ["Rendement de Capitaux Propres Ordinaires"],
        "ROIC": ["Rendement des Capitaux Investis"],
        COL_BETA_MA_VUE: ["Bêta (5 ans)", "Beta (5 ans)"],
        COL_PER_FWD_MA_VUE: ["Ratio PER prévu"],
        "PER forward": ["Ratio PER prévu"],
        COL_FV_COMPARABLE: [COL_FV_COMPARABLE],
        COL_FV_DCF: [COL_FV_DCF],
        COL_FV_DDM: [COL_FV_DDM],
        COL_FV_EPV: [COL_FV_EPV],
        COL_FV_COMP_UP: [COL_FV_COMP_UP],
        COL_FV_DCF_UP: [COL_FV_DCF_UP],
        "Objectif analystes": ["Objectif de prix moyen des analystes"],
        "Upside vs objectif": ["Potentiel baissier (Objectif des analystes)"],
    }
    for dst, candidates in alias_map.items():
        for src in candidates:
            if src in out.columns:
                out[dst] = pd.to_numeric(out[src], errors="coerce")
                break

    rule40_col = _find_investing_column(out, "r", "40")
    if rule40_col and "Règle des 40" not in out.columns:
        out["Règle des 40"] = pd.to_numeric(out[rule40_col], errors="coerce")

    secteur_col = _find_investing_column(out, "secteur")
    if secteur_col and "Secteur" not in out.columns:
        out["Secteur"] = out[secteur_col].astype(str).replace({"nan": "", "None": ""})

    equity_col = _find_investing_column(out, "fonds propres", "ordinaires")
    if equity_col and "Fonds propres ordinaires" not in out.columns:
        out["Fonds propres ordinaires"] = pd.to_numeric(out[equity_col], errors="coerce")

    return out


def filter_valid_ma_vue_rows(df: pd.DataFrame, dedupe_ticker: bool = True) -> pd.DataFrame:
    """Titres ma vue avec juste valeur et cours (Graham ou Yahoo)."""
    work = normalize_investing_frame(df)
    work = work[
        work["fair_value"].notna()
        & work["price"].notna()
        & (work["fair_value"] > 0)
        & (work["price"] > 0)
    ].copy()
    if dedupe_ticker and COL_TICKER in work.columns:
        work = work.drop_duplicates(subset=[COL_TICKER], keep="first")
    return work


def load_investing_export(path: str | Path, sheet: str = INVESTING_SHEET) -> pd.DataFrame:
    """Charge un export Premium Investing (.xlsx)."""
    path = Path(path)
    df = pd.read_excel(path, sheet_name=sheet, header=INVESTING_HEADER_ROW)
    df = df.dropna(subset=[COL_NAME], how="any")
    df["source_file"] = path.name
    return normalize_investing_frame(df)


def load_investing_exports(paths: Iterable[str | Path]) -> pd.DataFrame:
    frames = [load_investing_export(p) for p in paths]
    if not frames:
        raise ValueError("Aucun fichier fourni.")
    return pd.concat(frames, ignore_index=True)


def glob_investing_exports(directory: str | Path = ".", pattern: str = "*.xlsx") -> list[Path]:
    directory = Path(directory)
    return sorted(directory.glob(pattern))


def filter_valid_fair_value_rows(df: pd.DataFrame, dedupe_ticker: bool = True) -> pd.DataFrame:
    """Garde les lignes avec cours et juste valeur Investing numériques."""
    work = normalize_investing_frame(df)
    work = work[
        work["fair_value"].notna()
        & work["price"].notna()
        & (work["fair_value"] > 0)
        & (work["price"] > 0)
    ].copy()
    if dedupe_ticker and COL_TICKER in work.columns:
        work = work.drop_duplicates(subset=[COL_TICKER], keep="first")
    return work


def rows_with_complete_features(df: pd.DataFrame, method: str = "pillars_ridge") -> pd.DataFrame:
    """Sous-ensemble utilisable par le modèle (features fondamentaux complètes)."""
    work = filter_valid_fair_value_rows(df)
    builder = build_pillar_features if method == "pillars_ridge" else build_fair_value_features
    X = builder(work)
    mask = _usable_feature_mask(X, method)
    return work.loc[mask].copy()


def load_investing_anchors(path: str | Path | None = None) -> list[dict]:
    """Charge les repères JV Investing manuels (YAML)."""
    anchor_path = Path(path) if path else INVESTING_ANCHORS_PATH
    if not anchor_path.is_file():
        return []
    try:
        import yaml

        with open(anchor_path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except Exception:
        return []
    anchors = data.get("anchors") or []
    return [a for a in anchors if isinstance(a, dict) and a.get("ticker")]


def _anchor_submodel_upside(item: Mapping, key_up: str, key_fv: str, ref_price: float) -> float | None:
    """Hausse sous-modèle Investing : ratio explicite ou FV / cours de référence."""
    if key_up in item and item[key_up] is not None:
        up = pd.to_numeric(item[key_up], errors="coerce")
        if pd.notna(up):
            return float(up)
    if key_fv in item and item[key_fv] is not None and ref_price > 0:
        fv = pd.to_numeric(item[key_fv], errors="coerce")
        if pd.notna(fv) and fv > 0:
            return float(fv) / ref_price - 1.0
    return None


def anchor_item_to_enrichment_dict(item: Mapping) -> dict:
    """
    Repère YAML → champs enrichissement (hausses comparable / DCF / DDM recalculables au cours actuel).
    Aligné sur la méthode InvestingPro : blend des sous-modèles × cours Yahoo.
    """
    ref_price = pd.to_numeric(item.get("price"), errors="coerce")
    if ref_price is None or not np.isfinite(ref_price) or ref_price <= 0:
        return {}
    ref_price = float(ref_price)
    out: dict = {}
    pairs = (
        ("comparable_upside", "comparable_fv", COL_FV_COMP_UP),
        ("dcf_upside", "dcf_fv", COL_FV_DCF_UP),
        ("ddm_upside", "ddm_fv", COL_FV_DDM_UP),
    )
    for key_up, key_fv, col_up in pairs:
        up = _anchor_submodel_upside(item, key_up, key_fv, ref_price)
        if up is not None and up > -0.99:
            out[col_up] = up
    fv = pd.to_numeric(item.get("fair_value"), errors="coerce")
    if pd.notna(fv) and fv > 0:
        out[COL_FAIR_VALUE] = float(fv)
    label = str(item.get("label") or "").strip()
    if label:
        out[COL_FAIR_LABEL] = label
    return out


def build_investing_anchors_index(path: str | Path | None = None) -> dict[str, dict]:
    """Index Yahoo → enrichissement calculable (investing_anchors.yaml)."""
    index: dict[str, dict] = {}
    for item in load_investing_anchors(path):
        ticker = str(item.get("ticker", "")).strip().upper()
        if not ticker:
            continue
        enrich = anchor_item_to_enrichment_dict(item)
        if enrich:
            index[ticker] = enrich
    return index


def merge_prediction_enrichment(
    enrich_indices: Mapping[str, Mapping[str, Mapping]] | None = None,
) -> dict[str, dict]:
    """Fusion ma vue + repères YAML (repères YAML prioritaires si même ticker)."""
    merged = merge_ma_vue_indices(enrich_indices)
    merged.update(build_investing_anchors_index())
    return merged


def investing_anchors_to_training_frame(
    anchors: Sequence[Mapping] | None = None,
    *,
    path: str | Path | None = None,
) -> pd.DataFrame:
    """Convertit les repères Investing en lignes d'entraînement Ridge (cours + JV cible)."""
    items = list(anchors) if anchors is not None else load_investing_anchors(path)
    if not items:
        return pd.DataFrame()

    records: list[dict] = []
    for item in items:
        price = pd.to_numeric(item.get("price"), errors="coerce")
        fair_value = pd.to_numeric(item.get("fair_value"), errors="coerce")
        if pd.isna(price) or pd.isna(fair_value) or price <= 0 or fair_value <= 0:
            continue

        ticker = str(item.get("ticker", "")).strip()
        full_ticker = str(item.get("full_ticker") or ticker).strip()
        label = str(item.get("label") or "").strip()
        if not label:
            label = classify_upside(float(fair_value / price - 1.0))

        row: dict = {
            COL_TICKER: full_ticker,
            COL_NAME: item.get("name") or ticker,
            COL_PRICE: float(price),
            COL_FAIR_VALUE: float(fair_value),
            COL_FAIR_LABEL: label,
            "source_file": "investing_anchors.yaml",
        }

        mapping = (
            ("per", COL_PER),
            ("pb", COL_PB),
            ("ps", COL_PS),
            ("dividend_yield", COL_DIV_YIELD),
            ("debt_cap", COL_DEBT_CAP),
            ("rsi", COL_RSI),
            ("analyst_upside", "Upside vs objectif"),
            ("roe", "ROE"),
            ("roic", "ROIC"),
            ("eps_growth", "EPS growth"),
            ("rule_40", "Règle des 40"),
        )
        for src, dst in mapping:
            if src not in item or item[src] is None:
                continue
            val = pd.to_numeric(item[src], errors="coerce")
            if pd.notna(val):
                row[dst] = float(val)

        enrich = anchor_item_to_enrichment_dict(item)
        ref_p = float(price)
        for up_col, fv_col in (
            (COL_FV_COMP_UP, COL_FV_COMPARABLE),
            (COL_FV_DCF_UP, COL_FV_DCF),
            (COL_FV_DDM_UP, COL_FV_DDM),
        ):
            if up_col in enrich:
                row[fv_col] = ref_p * (1.0 + float(enrich[up_col]))

        records.append(row)

    if not records:
        return pd.DataFrame()

    frame = normalize_investing_frame(pd.DataFrame(records))
    X = build_pillar_features(frame, fair_multiples=DEFAULT_FAIR_MULTIPLES)
    mask = _usable_feature_mask(X, "pillars_ridge")
    return frame.loc[mask].copy()


def append_investing_anchors_to_training(
    train_df: pd.DataFrame,
    directory: str | Path = ".",
) -> tuple[pd.DataFrame, bool]:
    """Fusionne les repères YAML ; priorité aux ancres sur un même Full Ticker."""
    anchor_df = investing_anchors_to_training_frame(path=Path(directory) / "investing_anchors.yaml")
    if anchor_df.empty:
        return train_df, False
    merged = pd.concat([train_df, anchor_df], ignore_index=True)
    if COL_TICKER in merged.columns:
        merged = merged.drop_duplicates(subset=[COL_TICKER], keep="last")
    return merged, True


def _usable_feature_mask(X: pd.DataFrame, method: str) -> pd.Series:
    if method == "pillars_ridge":
        cols = [c for c in X.columns if c.startswith("pillar_")]
        if not cols:
            return pd.Series(False, index=X.index)
        return X[cols].notna().sum(axis=1) >= 3
    return X.notna().all(axis=1)


def align_feature_matrix(X: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    """
    Aligne X sur les colonnes du modèle entraîné.
    Colonnes absentes (ex. piliers ma vue 2) → 0 ; colonnes en trop ignorées.
    """
    return X.reindex(columns=list(feature_columns), fill_value=0.0).fillna(0.0)


def normalize_investing_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Harmonise colonnes Investing et types numériques."""
    out = df.copy()
    for col in NUMERIC_INVESTING_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if COL_PRICE in out.columns:
        price_col = pd.to_numeric(out[COL_PRICE], errors="coerce")
        if "price" in out.columns:
            out["price"] = price_col.fillna(pd.to_numeric(out["price"], errors="coerce"))
        else:
            out["price"] = price_col
    elif "price" in out.columns:
        out["price"] = pd.to_numeric(out["price"], errors="coerce")
    else:
        out["price"] = np.nan
    if COL_FAIR_VALUE in out.columns:
        fv_col = pd.to_numeric(out[COL_FAIR_VALUE], errors="coerce")
        if "fair_value" in out.columns:
            out["fair_value"] = fv_col.fillna(pd.to_numeric(out["fair_value"], errors="coerce"))
        else:
            out["fair_value"] = fv_col
    elif "fair_value" in out.columns:
        out["fair_value"] = pd.to_numeric(out["fair_value"], errors="coerce")
    else:
        out["fair_value"] = np.nan
    if COL_FAIR_LABEL in out.columns:
        label_col = out[COL_FAIR_LABEL].astype(str).str.strip()
        if "label" in out.columns:
            alt = out["label"].astype(str).str.strip()
            out["label"] = label_col.where(label_col.ne("") & label_col.ne("nan"), alt)
        else:
            out["label"] = label_col
    elif "label" in out.columns:
        out["label"] = out["label"].astype(str).str.strip()
    else:
        out["label"] = ""
    out["upside"] = out["fair_value"] / out["price"] - 1.0
    return out


def build_fair_value_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Matrice de features calquée sur les colonnes des exports Investing.

    Combinaison de multiples (PER, P/B, P/S), technique (RSI), dividende,
    structure financière (dettes, flottant) et proxies Graham / PEG.
    """
    price = pd.to_numeric(df["price"] if "price" in df.columns else df[COL_PRICE], errors="coerce")
    per = pd.to_numeric(df.get(COL_PER), errors="coerce").replace(0, np.nan)
    pb = pd.to_numeric(df.get(COL_PB), errors="coerce").replace(0, np.nan)
    ps = pd.to_numeric(df.get(COL_PS), errors="coerce").replace(0, np.nan)
    rsi = pd.to_numeric(df.get(COL_RSI), errors="coerce")

    X = pd.DataFrame(index=df.index)
    X["log_price"] = np.log(price.clip(lower=1e-6))
    X["inv_per"] = 1.0 / per
    X["inv_pb"] = 1.0 / pb
    X["inv_ps"] = 1.0 / ps
    X["per"] = per
    X["pb"] = pb
    X["ps"] = ps
    X["rsi_norm"] = (rsi - 50.0) / 50.0
    X["div_yield"] = pd.to_numeric(df.get(COL_DIV_YIELD), errors="coerce")
    X["debt_cap"] = pd.to_numeric(df.get(COL_DEBT_CAP), errors="coerce")
    X["float_pct"] = pd.to_numeric(df.get(COL_FLOAT), errors="coerce")
    mcap = pd.to_numeric(df.get(COL_MCAP), errors="coerce").clip(lower=1.0)
    X["log_mcap"] = np.log(mcap)

    eps = price / per
    bv = price / pb
    graham = np.sqrt(22.5 * eps.clip(lower=0) * bv.clip(lower=0))
    X["graham_ratio"] = (graham / price).where((eps > 0) & (bv > 0))
    X["peg_proxy"] = per / 15.0
    return X.replace([np.inf, -np.inf], np.nan)


def _as_series(values, index: pd.Index) -> pd.Series:
    """Convertit une valeur ou une série en Series alignée sur l'index du frame."""
    series = pd.to_numeric(values, errors="coerce")
    if not isinstance(series, pd.Series):
        series = pd.Series(series, index=index)
    return series


def build_pillar_features(
    df: pd.DataFrame,
    fair_multiples: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """
    Piliers explicites alignés sur la méthodologie InvestingPro documentée :
    Graham, multiples PER/P/B/P/S, dividende, endettement, RSI.
    Chaque pilier est exprimé en log(juste_valeur / cours) pour combinaison matricielle.
    """
    fm = {**DEFAULT_FAIR_MULTIPLES, **(fair_multiples or {})}
    price = _as_series(
        df["price"] if "price" in df.columns else df.get(COL_PRICE),
        df.index,
    )
    per = _as_series(df.get(COL_PER), df.index).replace(0, np.nan)
    if COL_PER_FWD_MA_VUE in df.columns:
        per_fwd = _as_series(df.get(COL_PER_FWD_MA_VUE), df.index).replace(0, np.nan)
        per = per.fillna(per_fwd.where(per_fwd > 0))
    pb = _as_series(df.get(COL_PB), df.index).replace(0, np.nan)
    ps = _as_series(df.get(COL_PS), df.index).replace(0, np.nan)
    div_y = _as_series(df.get(COL_DIV_YIELD), df.index)
    debt_src = COL_DEBT_CAP
    if debt_src not in df.columns and f"{COL_DEBT_CAP}_y" in df.columns:
        debt_src = f"{COL_DEBT_CAP}_y"
    elif debt_src not in df.columns and f"{COL_DEBT_CAP}_x" in df.columns:
        debt_src = f"{COL_DEBT_CAP}_x"
    debt = _as_series(df.get(debt_src), df.index)
    rsi = _as_series(df.get(COL_RSI), df.index)

    X = pd.DataFrame(index=df.index)
    eps = price / per
    bv = price / pb

    graham = np.sqrt(22.5 * eps.clip(lower=0) * bv.clip(lower=0))
    X["pillar_graham_log"] = np.log(graham / price).where((eps > 0) & (bv > 0) & (price > 0))
    X["pillar_per_log"] = np.log(fm["per"] / per).where(per > 0)
    X["pillar_pb_log"] = np.log(fm["pb"] / pb).where(pb > 0)
    X["pillar_ps_log"] = np.log(fm["ps"] / ps).where(ps > 0)
    X["pillar_div_log"] = np.log(fm["div_yield"] / div_y).where(div_y > 0)
    X["pillar_debt_adj"] = -0.35 * debt.clip(lower=0.0)
    X["pillar_rsi_adj"] = -0.15 * ((rsi - 50.0) / 50.0)

    if "ROE" in df.columns:
        roe = pd.to_numeric(df["ROE"], errors="coerce")
        X["pillar_roe_adj"] = 0.25 * (roe - 0.12).clip(-0.2, 0.3)
    if "Marge nette" in df.columns:
        nm = pd.to_numeric(df["Marge nette"], errors="coerce")
        X["pillar_margin_adj"] = 0.20 * (nm - 0.10).clip(-0.2, 0.3)
    if "Upside vs objectif" in df.columns:
        up = pd.to_numeric(df["Upside vs objectif"], errors="coerce")
        X["pillar_analyst_log"] = np.log1p(up).clip(-0.5, 0.8)

    for col, name in (
        (COL_FV_COMPARABLE, "pillar_investing_comparable_log"),
        (COL_FV_DCF, "pillar_investing_dcf_log"),
        (COL_FV_DDM, "pillar_investing_ddm_log"),
        (COL_FV_EPV, "pillar_investing_epv_log"),
        (COL_GRAHAM_VAL, "pillar_investing_graham_export_log"),
    ):
        if col in df.columns:
            X[name] = _pillar_log_ratio(df[col], price)

    if "ROIC" in df.columns:
        roic = pd.to_numeric(df["ROIC"], errors="coerce")
        X["pillar_roic_adj"] = 0.20 * (roic - 0.10).clip(-0.3, 0.3)
    if "EPS growth" in df.columns:
        eg = pd.to_numeric(df["EPS growth"], errors="coerce")
        X["pillar_eps_growth_adj"] = 0.15 * eg.clip(-0.5, 0.8)
    if "Règle des 40" in df.columns:
        r40 = _as_series(df["Règle des 40"], df.index)
        if r40.notna().any() and r40.abs().max(skipna=True) > 1.5:
            r40 = r40 / 100.0
        X["pillar_rule40_adj"] = 0.20 * (r40 - 0.40).clip(-0.5, 0.5)

    for col in CANONICAL_PILLAR_COLUMNS:
        if col not in X.columns:
            X[col] = np.nan

    return X.replace([np.inf, -np.inf], np.nan)


def _investing_submodel_blend(df: pd.DataFrame, price: pd.Series) -> pd.Series:
    """
    Moyenne pondérée des sous-modèles Investing (ma vue 2) disponibles par titre.
    """
    weights = {
        COL_FV_COMPARABLE: 0.50,
        COL_FV_DCF: 0.30,
        COL_FV_DDM: 0.12,
        COL_FV_EPV: 0.08,
    }
    total_w = pd.Series(0.0, index=df.index)
    total_v = pd.Series(0.0, index=df.index)
    for col, w in weights.items():
        if col not in df.columns:
            continue
        v = pd.to_numeric(df[col], errors="coerce")
        ok = v.notna() & (v > 0)
        total_v = total_v + v.fillna(0.0) * w * ok.astype(float)
        total_w = total_w + w * ok.astype(float)
    return (total_v / total_w).where(total_w > 0)


def _pillar_log_ratio(values, price: pd.Series) -> pd.Series:
    """log(valeur / cours) pour un estimateur Investing absolu."""
    v = pd.to_numeric(values, errors="coerce")
    if not isinstance(v, pd.Series):
        v = pd.Series(v, index=price.index)
    if not isinstance(price, pd.Series):
        price = pd.Series(price, index=v.index)
    return np.log(v / price).where((v > 0) & (price > 0))


def explain_pillar_weights(model: FairValueModel) -> pd.DataFrame:
    """Coefficients Ridge sur les piliers (interprétable)."""
    if model.pipeline_ is None or model.method != "pillars_ridge":
        raise ValueError("Modèle pillars_ridge entraîné requis.")
    ridge = model.pipeline_.named_steps["model"]
    scaler = model.pipeline_.named_steps["scale"]
    coefs = ridge.coef_ / scaler.scale_
    rows = []
    for name, coef in zip(model.feature_columns_, coefs):
        rows.append(
            {
                "Pilier": name,
                "Description": PILLAR_DESCRIPTIONS.get(name, name),
                "Poids": float(coef),
            }
        )
    out = pd.DataFrame(rows).sort_values("Poids", key=abs, ascending=False)
    out.attrs["intercept"] = float(ridge.intercept_)
    return out


def derive_label_thresholds(df: pd.DataFrame) -> dict:
    """Calibre les seuils de libellé à partir des upside Investing observés."""
    upside = pd.to_numeric(df["upside"], errors="coerce")
    labels = df["label"].astype(str)
    aubaine = upside[labels.str.contains("Aubaine", case=False, na=False)]
    sous = upside[labels.str.contains("Sous", case=False, na=False)]
    sure = upside[labels.str.contains("Sur", case=False, na=False)]
    juste = upside[labels == "Juste"]
    thresholds = dict(DEFAULT_LABEL_THRESHOLDS)
    if len(aubaine):
        thresholds["aubaine_min"] = float(aubaine.quantile(0.25))
    if len(sous):
        thresholds["sous_min"] = float(sous.quantile(0.25))
    if len(sure):
        thresholds["sureval_max"] = float(sure.quantile(0.75))
    if len(juste):
        thresholds["juste_low"] = float(juste.quantile(0.25))
        thresholds["juste_high"] = float(juste.quantile(0.75))
    return thresholds


def classify_upside(upside: float, thresholds: Mapping[str, float] | None = None) -> str:
    th = thresholds or DEFAULT_LABEL_THRESHOLDS
    if upside >= th.get("aubaine_min", 0.50):
        return "Aubaine"
    if upside >= th.get("sous_min", 0.185):
        return "Sous-évaluée"
    if upside <= th.get("sureval_max", -0.19):
        return "Surévaluée"
    return "Juste"


def classify_upside_many(upside: Iterable[float], thresholds: Mapping[str, float] | None = None) -> list[str]:
    return [classify_upside(float(u), thresholds) for u in upside]


def map_fundamentals_row(row: Mapping) -> dict:
    """Convertit une ligne fondamentaux dashboard vers champs Investing."""
    out = {}
    for src, dst in FUNDAMENTALS_TO_INVESTING.items():
        if src in row and row[src] is not None and not pd.isna(row[src]):
            out[dst] = row[src]
    if COL_PRICE not in out and "Dernier cours" in row:
        out[COL_PRICE] = row["Dernier cours"]
    return out


def evaluate_predictions(
    df: pd.DataFrame,
    pred: pd.DataFrame,
    label_thresholds: Mapping[str, float] | None = None,
) -> FairValueMetrics:
    y = pred["fair_value_investing"] if "fair_value_investing" in pred.columns else df.get("fair_value")
    if y is None and COL_FAIR_VALUE in df.columns:
        y = df[COL_FAIR_VALUE]
    y = pd.to_numeric(y, errors="coerce")
    yhat = pd.to_numeric(pred["fair_value_pred"], errors="coerce")
    mask = y.notna() & yhat.notna() & (y > 0) & (yhat > 0)
    yv, pv = y[mask].values, yhat[mask].values
    labels_true = pred.loc[mask, "label_investing"] if "label_investing" in pred.columns else None
    labels_pred = pred.loc[mask, "label_pred"] if "label_pred" in pred.columns else None
    label_acc = None
    if labels_true is not None and labels_pred is not None:
        label_acc = float(np.mean(labels_true.astype(str).values == labels_pred.astype(str).values))
    return FairValueMetrics(
        n=int(mask.sum()),
        mape=float(mean_absolute_percentage_error(yv, pv)),
        mae=float(mean_absolute_error(yv, pv)),
        rmse=float(np.sqrt(np.mean((yv - pv) ** 2))),
        r2=float(r2_score(yv, pv)),
        label_accuracy=label_acc,
    )


def _metrics_from_arrays(y, pred_fv, labels, pred_log_ratio=None, price=None):
    mask = np.isfinite(y) & np.isfinite(pred_fv) & (y > 0) & (pred_fv > 0)
    yv, pv = y[mask], pred_fv[mask]
    label_acc = None
    if labels is not None and len(labels) == len(y):
        th = DEFAULT_LABEL_THRESHOLDS
        pred_up = pv / (price[mask] if price is not None else 1.0) - 1.0
        label_acc = float(
            np.mean(
                [
                    classify_upside(u, th) == str(l)
                    for u, l in zip(pred_up, labels[mask])
                ]
            )
        )
    return FairValueMetrics(
        n=int(mask.sum()),
        mape=float(mean_absolute_percentage_error(yv, pv)),
        mae=float(mean_absolute_error(yv, pv)),
        rmse=float(np.sqrt(np.mean((yv - pv) ** 2))),
        r2=float(r2_score(yv, pv)),
        label_accuracy=label_acc,
    )


def calibrate_from_exports(
    paths: Sequence[str | Path],
    method: str = "pillars_ridge",
    calibrate_labels: bool = True,
) -> tuple[FairValueModel, pd.DataFrame, FairValueMetrics, int, int]:
    """
    Charge les exports, entraîne le modèle et renvoie prédictions + métriques CV.
    """
    df = load_investing_exports(paths)
    df = filter_valid_fair_value_rows(df)
    usable = rows_with_complete_features(df, method=method)
    if usable.empty:
        raise ValueError("Aucune ligne avec features complètes pour l'entraînement.")
    model = FairValueModel(method=method)
    if calibrate_labels:
        model.label_thresholds = derive_label_thresholds(df)
    cv_metrics = model.cross_validate(usable)
    model.fit(usable)
    predictions = model.predict_dataframe(df)
    predictions["abs_pct_error"] = np.nan
    ok = predictions["fair_value_investing"].notna() & predictions["fair_value_pred"].notna()
    predictions.loc[ok, "abs_pct_error"] = (
        (predictions.loc[ok, "fair_value_pred"] - predictions.loc[ok, "fair_value_investing"]).abs()
        / predictions.loc[ok, "fair_value_investing"]
    )
    return model, predictions, cv_metrics, len(df), len(usable)


def format_metrics_report(metrics: FairValueMetrics, title: str = "Validation croisée (5 folds)") -> str:
    lines = [
        title,
        f"  Observations : {metrics.n}",
        f"  MAPE         : {metrics.mape:.1%}",
        f"  MAE          : {metrics.mae:.2f}",
        f"  RMSE         : {metrics.rmse:.2f}",
        f"  R²           : {metrics.r2:.3f}",
    ]
    if metrics.label_accuracy is not None:
        lines.append(f"  Labels       : {metrics.label_accuracy:.1%}")
    return "\n".join(lines)


def run_holdout_test(
    train_paths: Sequence[str | Path],
    test_path: str | Path,
    method: str = "pillars_ridge",
) -> tuple[FairValueModel, pd.DataFrame, FairValueMetrics]:
    """Entraîne sur train_paths et évalue sur un export holdout (ex. Warren Buffett)."""
    train_df = filter_valid_fair_value_rows(load_investing_exports(train_paths))
    train_usable = rows_with_complete_features(train_df, method=method)
    test_df = filter_valid_fair_value_rows(load_investing_export(test_path))

    model = FairValueModel(method=method)
    model.label_thresholds = derive_label_thresholds(train_usable)
    model.fit(train_usable)
    predictions = model.predict_dataframe(test_df)
    metrics = evaluate_predictions(test_df, predictions, label_thresholds=model.label_thresholds)
    ok = predictions["fair_value_pred"].notna() & predictions["fair_value_investing"].notna()
    predictions["abs_pct_error"] = np.nan
    predictions.loc[ok, "abs_pct_error"] = (
        (predictions.loc[ok, "fair_value_pred"] - predictions.loc[ok, "fair_value_investing"]).abs()
        / predictions.loc[ok, "fair_value_investing"]
    )
    return model, predictions, metrics


def compare_investing_submodels(df: pd.DataFrame) -> pd.DataFrame:
    """MAPE des sous-modèles Investing (ma vue 2) vs Juste Valeur finale."""
    work = filter_valid_ma_vue_rows(df) if COL_FV_COMPARABLE in df.columns else filter_valid_fair_value_rows(df)
    target = work["fair_value"]
    models = {
        "Juste valeur comparable": work.get(COL_FV_COMPARABLE),
        "Juste valeur DCF": work.get(COL_FV_DCF),
        "Juste valeur DDM": work.get(COL_FV_DDM),
        "Juste valeur EPV": work.get(COL_FV_EPV),
        "Formule Ben Graham": work.get(COL_GRAHAM_VAL),
    }
    rows = []
    for name, pred in models.items():
        if pred is None:
            continue
        p = pd.to_numeric(pred, errors="coerce")
        mask = target.notna() & p.notna() & (target > 0) & (p > 0)
        if not mask.any():
            continue
        yv, pv = target[mask], p[mask]
        rows.append(
            {
                "Modele": name,
                "n": int(mask.sum()),
                "MAPE": float((pv - yv).abs().div(yv).mean()),
                "MAE": float((pv - yv).abs().mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("MAPE")


def calibrate_from_ma_vue(
    vue1_path: str | Path | None = None,
    vue2_path: str | Path | None = None,
    directory: str | Path = ".",
    method: str = "pillars_ridge",
    enrich_yahoo_price: bool = True,
    extra_train_paths: Sequence[str | Path] | None = None,
) -> tuple[FairValueModel, pd.DataFrame, FairValueMetrics, pd.DataFrame]:
    """
    Calibre le modèle sur ma vue (+ exports optionnels) et compare aux Juste Valeur Investing.
    Retourne (modèle, prédictions, métriques CV ma vue, tableau sous-modèles Investing).
    """
    df = load_ma_vue(vue1_path, vue2_path, directory, enrich_yahoo_price=enrich_yahoo_price)
    valid = filter_valid_ma_vue_rows(df)
    if valid.empty:
        raise ValueError("Aucune ligne ma vue avec juste valeur et cours.")

    submodels = compare_investing_submodels(df)

    train_frames = [valid]
    if extra_train_paths:
        extra = rows_with_complete_features(
            filter_valid_fair_value_rows(load_investing_exports(extra_train_paths)),
            method=method,
        )
        if not extra.empty:
            train_frames.append(extra)

    train_df = pd.concat(train_frames, ignore_index=True)
    train_df = train_df.drop_duplicates(subset=[COL_TICKER], keep="first")

    model = FairValueModel(method=method)
    model.sector = SECTOR_MINIERES
    model.fair_multiples = dict(DEFAULT_FAIR_MULTIPLES)
    if "label" in train_df.columns and train_df["label"].astype(str).str.len().gt(0).any():
        model.label_thresholds = derive_label_thresholds(train_df)
    cv_metrics = model.cross_validate(valid)
    model.fit(train_df)
    predictions = model.predict_dataframe(df)
    ok = predictions["fair_value_investing"].notna() & predictions["fair_value_pred"].notna()
    predictions["abs_pct_error"] = np.nan
    predictions.loc[ok, "abs_pct_error"] = (
        (predictions.loc[ok, "fair_value_pred"] - predictions.loc[ok, "fair_value_investing"]).abs()
        / predictions.loc[ok, "fair_value_investing"]
    )
    return model, predictions, cv_metrics, submodels


def calibrate_from_tech_ma_vue(
    vue2_path: str | Path | None = None,
    efficiency_path: str | Path | None = None,
    risk_path: str | Path | None = None,
    vue3_path: str | Path | None = None,
    directory: str | Path = ".",
    method: str = "pillars_ridge",
    enrich_yahoo_price: bool = True,
) -> tuple[FairValueModel, pd.DataFrame, FairValueMetrics, pd.DataFrame]:
    """Calibre le modèle tech sur ma vue 2 + ma vue 3 + Efficiency + Risk."""
    df = load_tech_ma_vue(
        vue2_path,
        efficiency_path,
        risk_path,
        vue3_path,
        directory,
        enrich_yahoo_price=enrich_yahoo_price,
    )
    valid = filter_valid_ma_vue_rows(df)
    if valid.empty:
        raise ValueError("Aucune ligne tech avec juste valeur dérivée et cours.")

    submodels = compare_investing_submodels(df)
    model = FairValueModel(method=method)
    model.sector = SECTOR_TECH
    model.fair_multiples = dict(TECH_FAIR_MULTIPLES)
    if valid["label"].astype(str).str.strip().ne("").any():
        model.label_thresholds = derive_label_thresholds(valid)
    cv_metrics = model.cross_validate(valid)
    model.fit(valid)
    predictions = model.predict_dataframe(df)
    ok = predictions["fair_value_pred"].notna() & predictions["fair_value_investing"].notna()
    predictions["abs_pct_error"] = np.nan
    predictions.loc[ok, "abs_pct_error"] = (
        (predictions.loc[ok, "fair_value_pred"] - predictions.loc[ok, "fair_value_investing"]).abs()
        / predictions.loc[ok, "fair_value_investing"]
    )
    return model, predictions, cv_metrics, submodels


def calibrate_unified_ma_vue(
    directory: str | Path = ".",
    method: str = "pillars_ridge",
    enrich_yahoo_price: bool = True,
) -> tuple[FairValueModel, pd.DataFrame, FairValueMetrics, list[str]]:
    """
    Calibre un modèle Ridge unique sur les exports minières + tech (même logique pour tous les titres).
    """
    directory = Path(directory)
    train_frames: list[pd.DataFrame] = []
    train_labels: list[str] = []

    vue1, vue2 = glob_ma_vue_exports(directory)
    if vue1 is not None:
        df_m = load_ma_vue(vue1, vue2, directory, enrich_yahoo_price=enrich_yahoo_price)
        valid_m = filter_valid_ma_vue_rows(df_m)
        if not valid_m.empty:
            train_frames.append(valid_m)
            label = Path(vue1).name
            if vue2:
                label = f"{label} + {Path(vue2).name}"
            train_labels.append(label)

    t_vue2, t_eff, t_risk, t_vue3 = glob_tech_exports(directory)
    if t_vue2 is not None:
        df_t = load_tech_ma_vue(
            t_vue2, t_eff, t_risk, t_vue3, directory, enrich_yahoo_price=enrich_yahoo_price
        )
        valid_t = filter_valid_ma_vue_rows(df_t)
        if not valid_t.empty:
            train_frames.append(valid_t)
            parts = [Path(t_vue2).name]
            if t_vue3:
                parts.append(Path(t_vue3).name)
            if t_eff:
                parts.append(Path(t_eff).name)
            if t_risk:
                parts.append(Path(t_risk).name)
            train_labels.append(" + ".join(parts))

    if not train_frames:
        raise ValueError("Aucune donnée ma vue pour calibrage unifié.")

    train_df = pd.concat(train_frames, ignore_index=True)
    train_df = train_df.drop_duplicates(subset=[COL_TICKER], keep="first")
    train_df, has_anchors = append_investing_anchors_to_training(train_df, directory)
    if has_anchors:
        train_labels.append("investing_anchors.yaml")

    model = FairValueModel(method=method)
    model.sector = SECTOR_UNIFIED
    model.fair_multiples = dict(DEFAULT_FAIR_MULTIPLES)
    if train_df["label"].astype(str).str.strip().ne("").any():
        model.label_thresholds = derive_label_thresholds(train_df)
    cv_metrics = model.cross_validate(train_df)
    model.fit(train_df)
    predictions = model.predict_dataframe(train_df)
    ok = predictions["fair_value_pred"].notna()
    if "fair_value_investing" in predictions.columns:
        ok = ok & predictions["fair_value_investing"].notna()
        predictions.loc[ok, "abs_pct_error"] = (
            (predictions.loc[ok, "fair_value_pred"] - predictions.loc[ok, "fair_value_investing"]).abs()
            / predictions.loc[ok, "fair_value_investing"]
        )
    return model, predictions, cv_metrics, train_labels


def resolve_prediction_model(
    models: Mapping[str, FairValueModel | None],
) -> FairValueModel | None:
    """Modèle Ridge utilisé en prédiction : unifié, puis repli tech / minières."""
    return (
        models.get(SECTOR_UNIFIED)
        or models.get(SECTOR_TECH)
        or models.get(SECTOR_MINIERES)
    )


# Colonnes ma vue 2 — entraînement / comparaison uniquement (pas injectées au dashboard).
MA_VUE_DASHBOARD_MERGE_COLUMNS = (
    COL_FV_COMPARABLE,
    COL_FV_DCF,
    COL_FV_DDM,
    COL_FV_EPV,
    COL_GRAHAM_VAL,
    COL_GRAHAM_UP,
    COL_PER,
    COL_PB,
    COL_PS,
    COL_DIV_YIELD,
    COL_DEBT_CAP,
    COL_RSI,
    COL_ROE_MA_VUE,
    COL_ROIC_MA_VUE,
    COL_EPS_GROWTH_MA_VUE,
    "ROE",
    "ROIC",
    "EPS growth",
    COL_FAIR_VALUE,
    COL_FAIR_LABEL,
    COL_FV_COMP_UP,
    COL_FV_DCF_UP,
    "Objectif analystes",
    "PER forward",
    COL_PER_FWD_MA_VUE,
    COL_BETA_MA_VUE,
    "Secteur",
    "Règle des 40",
    "Fonds propres ordinaires",
)


def resolve_fair_value_profile(
    ticker: str,
    tech_yahoo: set[str] | frozenset,
    mining_yahoo: set[str] | frozenset | None = None,
    yahoo_sector: str | None = None,
    yahoo_industry: str | None = None,
) -> str:
    """
    Profil sectoriel pour multiples et modèle Ridge (minières / tech + secteurs Yahoo).
    Priorité : exports ma vue → industrie Yahoo → secteur Yahoo.
    """
    tk = str(ticker).strip().upper()
    mining_yahoo = mining_yahoo or frozenset()
    if tk in tech_yahoo:
        return SECTOR_TECH
    if tk in mining_yahoo:
        return SECTOR_MINIERES

    sector = str(yahoo_sector or "").strip()
    industry = str(yahoo_industry or "").lower()

    if any(k in industry for k in BIOTECH_INDUSTRY_KEYWORDS):
        return SECTOR_BIOTECH
    if any(k in industry for k in PHARMA_INDUSTRY_KEYWORDS):
        return SECTOR_PHARMA
    if any(k in industry for k in DEFENSE_INDUSTRY_KEYWORDS):
        return SECTOR_DEFENSE
    if any(k in industry for k in FOOD_INDUSTRY_KEYWORDS):
        return SECTOR_FOOD
    if any(k in industry for k in MINING_INDUSTRY_KEYWORDS):
        return SECTOR_MINIERES
    if any(k in industry for k in ENERGY_INDUSTRY_KEYWORDS):
        return SECTOR_ENERGY
    if any(k in industry for k in CONSUMER_INDUSTRY_KEYWORDS):
        return SECTOR_CONSUMER
    if any(k in industry for k in TECH_INDUSTRY_KEYWORDS):
        return SECTOR_TECH

    if sector in YAHOO_SECTORS_HEALTHCARE:
        return SECTOR_BIOTECH if "biotech" in industry else SECTOR_PHARMA
    if sector in YAHOO_SECTORS_ENERGY:
        return SECTOR_ENERGY
    if sector in YAHOO_SECTORS_MINING:
        return SECTOR_MINIERES
    if sector in YAHOO_SECTORS_TECH:
        return SECTOR_TECH
    if sector in YAHOO_SECTORS_DEFENSIVE:
        return SECTOR_FOOD
    if sector in YAHOO_SECTORS_CYCLICAL:
        return SECTOR_CONSUMER
    if sector in YAHOO_SECTORS_FINANCIAL:
        return SECTOR_FINANCE
    if sector in YAHOO_SECTORS_INDUSTRIALS:
        return SECTOR_DEFENSE if any(k in industry for k in DEFENSE_INDUSTRY_KEYWORDS) else SECTOR_INDUSTRIAL

    return SECTOR_MINIERES


def resolve_fair_value_sector(
    ticker: str,
    tech_yahoo: set[str] | frozenset,
    mining_yahoo: set[str] | frozenset | None = None,
    yahoo_sector: str | None = None,
    yahoo_industry: str | None = None,
) -> str:
    """Rétrocompat : renvoie toujours le modèle unifié (profil = multiples sectoriels)."""
    _ = resolve_fair_value_profile(
        ticker, tech_yahoo, mining_yahoo, yahoo_sector, yahoo_industry
    )
    return SECTOR_UNIFIED


def build_ma_vue_yahoo_index(ma_vue_df: pd.DataFrame) -> dict[str, dict]:
    """Indexe les lignes ma vue par symbole Yahoo (ETL.PA, AA…)."""
    index: dict[str, dict] = {}
    if ma_vue_df is None or ma_vue_df.empty or COL_TICKER not in ma_vue_df.columns:
        return index
    for _, row in ma_vue_df.iterrows():
        yahoo = investing_ticker_to_yahoo(row[COL_TICKER]).upper()
        if yahoo:
            index[yahoo] = row.to_dict()
    return index


def apply_calculable_ma_vue_enrichment(
    mapped: dict,
    mv: Mapping,
    price: float | None,
) -> dict:
    """
    Recalcule les sous-modèles absolus : cours Yahoo × (1 + hausse ma vue).
    N'injecte pas les montants Investing figés — seulement des ratios recalculables.
    """
    p = pd.to_numeric(price, errors="coerce")
    if p is None or not np.isfinite(p) or p <= 0:
        return mapped
    for up_col, fv_col in (
        (COL_FV_COMP_UP, COL_FV_COMPARABLE),
        (COL_FV_DCF_UP, COL_FV_DCF),
        (COL_FV_DDM_UP, COL_FV_DDM),
        (COL_GRAHAM_UP, COL_GRAHAM_VAL),
    ):
        if up_col not in mv:
            continue
        raw = mv.get(up_col)
        if raw is None or (isinstance(raw, str) and raw.strip().upper() in ("NM", "N/A", "")):
            continue
        up = pd.to_numeric(raw, errors="coerce")
        if pd.isna(up) or not np.isfinite(up) or up <= -0.99:
            continue
        val = float(p) * (1.0 + float(up))
        if val > 0:
            mapped[fv_col] = val
    return mapped


def merge_ma_vue_indices(
    enrich_indices: Mapping[str, Mapping[str, Mapping]] | None,
) -> dict[str, dict]:
    """Fusionne les index sectoriels ma vue (clés Yahoo)."""
    merged: dict[str, dict] = {}
    for index in (enrich_indices or {}).values():
        if index:
            merged.update(index)
    return merged


def sector_ticker_sets(
    enrich_indices: Mapping[str, Mapping[str, Mapping]] | None = None,
) -> tuple[frozenset[str], frozenset[str]]:
    """Symboles Yahoo par secteur (routage modèle) — clés d'index ma vue, pas leurs métriques."""
    enrich_indices = enrich_indices or {}
    tech = frozenset((enrich_indices.get(SECTOR_TECH) or {}).keys())
    mining = frozenset((enrich_indices.get(SECTOR_MINIERES) or {}).keys())
    return tech, mining


def fundamentals_to_fair_value_frame(
    rows: Sequence[Mapping],
    ma_vue_index: Mapping[str, Mapping] | None = None,
    *,
    enrich_from_ma_vue: bool = False,
    enrich_calculable_ma_vue: bool = True,
) -> pd.DataFrame:
    """
    Convertit des lignes fondamentaux dashboard en frame prête pour predict_dataframe.
    Par défaut : Yahoo + enrichissement **calculable** ma vue (hausse comparable/DCF × cours).
    """
    ma_vue_index = ma_vue_index or {}
    records: list[dict] = []
    for row in rows:
        mapped = map_fundamentals_row(row)
        ticker = str(row.get("Ticker", "")).strip()
        mapped[COL_TICKER] = ticker
        mapped[COL_NAME] = row.get("Nom") or ticker
        price = mapped.get(COL_PRICE)

        mv = ma_vue_index.get(ticker.upper())
        if mv and enrich_calculable_ma_vue:
            mapped = apply_calculable_ma_vue_enrichment(mapped, mv, price)

        if enrich_from_ma_vue:
            if mv:
                for col in MA_VUE_DASHBOARD_MERGE_COLUMNS:
                    val = mv.get(col)
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        mapped[col] = val
                if COL_FAIR_VALUE in mv and pd.notna(mv.get(COL_FAIR_VALUE)):
                    mapped["fair_value"] = mv[COL_FAIR_VALUE]
                if COL_FAIR_LABEL in mv and pd.notna(mv.get(COL_FAIR_LABEL)):
                    mapped["label"] = mv[COL_FAIR_LABEL]

        records.append(mapped)

    if not records:
        return pd.DataFrame()
    return normalize_investing_frame(pd.DataFrame(records))


def predict_fair_value_for_fundamentals(
    rows: Sequence[Mapping],
    models: Mapping[str, FairValueModel | None],
    enrich_indices: Mapping[str, Mapping[str, Mapping]] | None = None,
) -> pd.DataFrame:
    """
    Prédit juste valeur par titre : modèle Ridge unifié, multiples cibles uniques, enrichissement ma vue.
    """
    empty = pd.DataFrame(
        columns=[
            "Ticker",
            "Juste valeur estimée",
            "Upside juste valeur",
            "Libellé juste valeur",
            "Modèle juste valeur",
            "Juste valeur Damodaran",
            "Upside Damodaran",
            "Profil Damodaran",
        ]
    )
    if not rows:
        return empty

    model = resolve_prediction_model(models)
    if model is None:
        return empty

    ma_vue_merged = merge_prediction_enrichment(enrich_indices)
    fair_multiples = resolve_unified_multiples()
    tech_yahoo, mining_yahoo = _yahoo_profile_sets(enrich_indices)

    records: list[dict] = []
    for row in rows:
        ticker = str(row.get("Ticker", "")).strip()
        fv_in = fundamentals_to_fair_value_frame(
            [row],
            ma_vue_merged,
            enrich_calculable_ma_vue=True,
        )
        if fv_in.empty:
            continue
        pred = model.predict_dataframe(
            fv_in,
            fair_multiples=fair_multiples,
        )
        if pred.empty:
            continue
        jv = pred["fair_value_pred"].iloc[0]
        if pd.isna(jv):
            continue
        damo = compute_damodaran_fair_value(
            row,
            tech_yahoo=tech_yahoo,
            mining_yahoo=mining_yahoo,
        )
        records.append(
            {
                "Ticker": ticker,
                "Juste valeur estimée": jv,
                "Upside juste valeur": pred["upside_pred"].iloc[0],
                "Libellé juste valeur": pred["label_pred"].iloc[0],
                "Modèle juste valeur": SECTOR_UNIFIED,
                **damo,
            }
        )

    if not records:
        return empty
    out = pd.DataFrame(records)
    out["Libellé juste valeur"] = (
        out["Libellé juste valeur"].fillna("").astype(str).replace({"nan": "", "None": ""})
    )
    return out


def load_dashboard_fair_value_models(
    directory: str | Path | None = None,
    enrich_yahoo_price: bool = False,
) -> tuple[dict[str, FairValueModel], dict[str, dict]]:
    """
    Calibre les modèles minières et tech disponibles.
    Retourne ({secteur: modèle}, {secteur: meta}).
    """
    directory = Path(directory or Path(__file__).resolve().parent)
    models: dict[str, FairValueModel] = {}
    meta: dict[str, dict] = {}

    vue1, vue2 = glob_ma_vue_exports(directory)
    if vue1 is not None:
        try:
            model, _pred, cv_metrics, _sub = calibrate_from_ma_vue(
                vue1_path=vue1,
                vue2_path=vue2,
                directory=directory,
                enrich_yahoo_price=enrich_yahoo_price,
            )
            models[SECTOR_MINIERES] = model
            train_label = vue1.name
            if vue2:
                train_label = f"{vue1.name} + {vue2.name}"
            meta[SECTOR_MINIERES] = {
                "status": "ok",
                "cv_mape": cv_metrics.mape,
                "cv_n": cv_metrics.n,
                "train_files": train_label,
                "method": model.method,
            }
        except Exception as exc:
            meta[SECTOR_MINIERES] = {"status": "error", "message": str(exc)}
    else:
        meta[SECTOR_MINIERES] = {
            "status": "missing_ma_vue",
            "message": f"Export ma vue introuvable ({MA_VUE1_GLOB}).",
        }

    t_vue2, t_eff, t_risk, t_vue3 = glob_tech_exports(directory)
    if t_vue2 is not None:
        try:
            model, _pred, cv_metrics, _sub = calibrate_from_tech_ma_vue(
                vue2_path=t_vue2,
                efficiency_path=t_eff,
                risk_path=t_risk,
                vue3_path=t_vue3,
                directory=directory,
                enrich_yahoo_price=enrich_yahoo_price,
            )
            models[SECTOR_TECH] = model
            parts = [t_vue2.name]
            if t_vue3:
                parts.append(t_vue3.name)
            if t_eff:
                parts.append(t_eff.name)
            if t_risk:
                parts.append(t_risk.name)
            meta[SECTOR_TECH] = {
                "status": "ok",
                "cv_mape": cv_metrics.mape,
                "cv_n": cv_metrics.n,
                "train_files": " + ".join(parts),
                "method": model.method,
            }
        except Exception as exc:
            meta[SECTOR_TECH] = {"status": "error", "message": str(exc)}
    else:
        meta[SECTOR_TECH] = {
            "status": "missing_tech",
            "message": f"Export tech ma vue 2 introuvable ({TECH_MA_VUE2_GLOB}).",
        }

    try:
        unified_model, _pred, cv_metrics, train_labels = calibrate_unified_ma_vue(
            directory,
            enrich_yahoo_price=enrich_yahoo_price,
        )
        models[SECTOR_UNIFIED] = unified_model
        meta[SECTOR_UNIFIED] = {
            "status": "ok",
            "cv_mape": cv_metrics.mape,
            "cv_n": cv_metrics.n,
            "train_files": " | ".join(train_labels),
            "method": unified_model.method,
        }
    except Exception as exc:
        meta[SECTOR_UNIFIED] = {"status": "error", "message": str(exc)}

    return models, meta


def load_dashboard_fair_value_model(
    directory: str | Path | None = None,
    enrich_yahoo_price: bool = False,
) -> tuple[FairValueModel | None, dict]:
    """
    Calibre le modèle minières (compatibilité). Préférer load_dashboard_fair_value_models().
    """
    models, meta = load_dashboard_fair_value_models(directory, enrich_yahoo_price)
    model = models.get(SECTOR_MINIERES)
    sector_meta = meta.get(SECTOR_MINIERES, {"status": "missing_ma_vue"})
    return model, sector_meta


def try_load_ma_vue_for_dashboard(
    directory: str | Path | None = None,
    enrich_yahoo_price: bool = False,
) -> pd.DataFrame | None:
    """Charge ma vue minières si les fichiers existent ; None sinon."""
    directory = Path(directory or Path(__file__).resolve().parent)
    vue1, vue2 = glob_ma_vue_exports(directory)
    if vue1 is None:
        return None
    try:
        return load_ma_vue(
            vue1_path=vue1,
            vue2_path=vue2,
            directory=directory,
            enrich_yahoo_price=enrich_yahoo_price,
        )
    except Exception:
        return None


def try_load_tech_ma_vue_for_dashboard(
    directory: str | Path | None = None,
    enrich_yahoo_price: bool = False,
) -> pd.DataFrame | None:
    """Charge export tech (ma vue 2 + ma vue 3 + Efficiency + Risk) ; None si absent."""
    directory = Path(directory or Path(__file__).resolve().parent)
    vue2, efficiency, risk, vue3 = glob_tech_exports(directory)
    if vue2 is None:
        return None
    try:
        return load_tech_ma_vue(
            vue2_path=vue2,
            efficiency_path=efficiency,
            risk_path=risk,
            vue3_path=vue3,
            directory=directory,
            enrich_yahoo_price=enrich_yahoo_price,
        )
    except Exception:
        return None
