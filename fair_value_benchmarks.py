"""
Téléchargement et fusion des repères sectoriels Damodaran → sector_benchmarks.yaml.

Sources : https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datacurrent.html
"""
from __future__ import annotations

import io
import re
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parent
DEFAULT_YAML_PATH = ROOT / "sector_benchmarks.yaml"
CACHE_DIR = ROOT / ".damodaran_cache"

DAMODARAN_DATASETS = {
    "countrystats": "countrystats.xls",
    "pedata": "pedata.xls",
    "pbvdata": "pbvdata.xls",
    "psdata": "psdata.xls",
}
DAMODARAN_BASE_URL = "https://www.stern.nyu.edu/~adamodar/pc/datasets"

# Profils dont les multiples US restent calibrés ma vue (non écrasés par Damodaran).
PRESERVE_PROFILE_MULTIPLES = frozenset({"tech", "minières"})

# Pondération des industries Damodaran (US) par profil dashboard.
PROFILE_INDUSTRY_WEIGHTS: dict[str, list[tuple[str, float]]] = {
    "biotech": [("Drugs (Biotechnology)", 1.0)],
    "pharma": [("Drugs (Pharmaceutical)", 1.0)],
    "énergie": [
        ("Oil/Gas (Integrated)", 0.55),
        ("Utility (General)", 0.45),
    ],
    "défense": [("Aerospace/Defense", 1.0)],
    "alimentation": [
        ("Food Processing", 0.55),
        ("Beverage (Soft)", 0.25),
        ("Beverage (Alcoholic)", 0.20),
    ],
    "finance": [
        ("Banks (Regional)", 0.55),
        ("Insurance (General)", 0.45),
    ],
    "consommation": [
        ("Retail (General)", 0.55),
        ("Apparel", 0.45),
    ],
    "industrie": [
        ("Business & Consumer Services", 0.50),
        ("Steel", 0.25),
        ("Building Materials", 0.25),
    ],
    "minières": [
        ("Precious Metals", 0.55),
        ("Metals & Mining", 0.45),
    ],
}

COUNTRIES_TO_UPDATE = (
    "France",
    "Germany",
    "United Kingdom",
    "Netherlands",
    "Belgium",
    "Switzerland",
    "Italy",
    "Spain",
    "Canada",
    "Australia",
    "Japan",
    "Hong Kong",
    "India",
    "Brazil",
)

PE_AGG_COL = "Aggregate Mkt Cap/ Trailing Net Income (only money making firms)"
DEFAULT_DIV_YIELDS: dict[str, float] = {
    "biotech": 0.0,
    "pharma": 0.02,
    "énergie": 0.045,
    "défense": 0.015,
    "alimentation": 0.03,
    "finance": 0.035,
    "consommation": 0.015,
    "industrie": 0.025,
    "minières": 0.035,
}

YAML_HEADER = """\
# Multiples cibles pour la juste valeur (piliers PER / P/B / P/S / dividende).
# Généré / mis à jour par : python update_sector_benchmarks.py
# Sources : Damodaran NYU Stern — pedata, pbvdata, psdata, countrystats
# https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datacurrent.html
#
# Profils « tech » et « minières » : multiples US conservés (calibrage ma vue).
# Autres profils : médianes agrégées Damodaran (US, sociétés profitables).
# Facteurs pays : médianes countrystats vs États-Unis.

"""


def download_dataset(
    name: str,
    *,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    timeout: int = 90,
) -> bytes:
    """Télécharge un .xls Damodaran (avec cache disque optionnel)."""
    filename = DAMODARAN_DATASETS[name]
    cache_dir = cache_dir or CACHE_DIR
    cache_path = cache_dir / filename

    if use_cache and cache_path.is_file():
        return cache_path.read_bytes()

    url = f"{DAMODARAN_BASE_URL}/{filename}"
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.get(
                url,
                timeout=timeout,
                headers={"User-Agent": "analyse-financiere/1.0"},
            )
            response.raise_for_status()
            content = response.content
            break
        except Exception as exc:
            last_err = exc
            if attempt == 2:
                raise exc
    else:
        raise last_err or RuntimeError(f"Téléchargement échoué : {filename}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(content)
    return content


def _find_header_row(df: pd.DataFrame, label: str) -> int:
    for i in range(len(df)):
        if str(df.iloc[i, 0]).strip() == label:
            return i
    raise ValueError(f"Ligne d'en-tête « {label} » introuvable.")


def parse_industry_averages(content: bytes) -> pd.DataFrame:
    """Parse la feuille « Industry Averages » d'un export Damodaran."""
    xl = pd.ExcelFile(io.BytesIO(content))
    sheet = "Industry Averages" if "Industry Averages" in xl.sheet_names else xl.sheet_names[-1]
    raw = pd.read_excel(xl, sheet_name=sheet, header=None)
    hdr = _find_header_row(raw, "Industry Name")
    data = raw.iloc[hdr + 1 :].copy()
    data.columns = raw.iloc[hdr].tolist()
    data = data[data["Industry Name"].notna()].copy()
    data["Industry Name"] = data["Industry Name"].astype(str).str.strip()
    data = data[~data["Industry Name"].str.startswith("Total")]
    return data.set_index("Industry Name", drop=False)


def parse_countrystats(content: bytes) -> pd.DataFrame:
    raw = pd.read_excel(io.BytesIO(content), header=None)
    hdr = _find_header_row(raw, "Country")
    data = raw.iloc[hdr + 1 :].copy()
    data.columns = raw.iloc[hdr].tolist()
    data = data[data["Country"].notna()].copy()
    data["Country"] = data["Country"].astype(str).str.strip()
    return data.set_index("Country", drop=False)


def _safe_float(value) -> float | None:
    try:
        v = float(value)
        if pd.isna(v) or v <= 0:
            return None
        return v
    except (TypeError, ValueError):
        return None


def _clip_pe(value: float | None) -> float | None:
    if value is None:
        return None
    return float(min(max(value, 4.0), 65.0))


def _clip_pb(value: float | None) -> float | None:
    if value is None:
        return None
    return float(min(max(value, 0.35), 12.0))


def _clip_ps(value: float | None) -> float | None:
    if value is None:
        return None
    return float(min(max(value, 0.25), 15.0))


def _weighted_industry_values(
    pe_df: pd.DataFrame,
    pb_df: pd.DataFrame,
    ps_df: pd.DataFrame,
    weights: list[tuple[str, float]],
) -> dict[str, float | None]:
    per_vals, pb_vals, ps_vals, w_vals = [], [], [], []
    for industry, weight in weights:
        if industry not in pe_df.index:
            continue
        pe_row = pe_df.loc[industry]
        per = _clip_pe(_safe_float(pe_row.get(PE_AGG_COL)) or _safe_float(pe_row.get("Trailing PE")))
        pb = _clip_pb(_safe_float(pb_df.loc[industry, "PBV"]) if industry in pb_df.index else None)
        ps = _clip_ps(_safe_float(ps_df.loc[industry, "Price/Sales"]) if industry in ps_df.index else None)
        if per is None and pb is None and ps is None:
            continue
        w_vals.append(weight)
        per_vals.append(per or 0.0)
        pb_vals.append(pb or 0.0)
        ps_vals.append(ps or 0.0)
    if not w_vals:
        return {"per": None, "pb": None, "ps": None}
    total = sum(w_vals)
    return {
        "per": round(sum(v * w for v, w in zip(per_vals, w_vals)) / total, 2) if any(per_vals) else None,
        "pb": round(sum(v * w for v, w in zip(pb_vals, w_vals)) / total, 2) if any(pb_vals) else None,
        "ps": round(sum(v * w for v, w in zip(ps_vals, w_vals)) / total, 2) if any(ps_vals) else None,
    }


def compute_profile_multiples_from_damodaran(
    pe_df: pd.DataFrame,
    pb_df: pd.DataFrame,
    ps_df: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    for profile, weights in PROFILE_INDUSTRY_WEIGHTS.items():
        vals = _weighted_industry_values(pe_df, pb_df, ps_df, weights)
        industries = ", ".join(ind for ind, _ in weights)
        block: dict[str, Any] = {
            "source_industry": f"{industries} (US Damodaran agg. PE profitable firms)",
            "multiples": {},
        }
        if vals["per"] is not None:
            block["multiples"]["per"] = vals["per"]
        if vals["pb"] is not None:
            block["multiples"]["pb"] = vals["pb"]
        if vals["ps"] is not None:
            block["multiples"]["ps"] = vals["ps"]
        block["multiples"]["div_yield"] = DEFAULT_DIV_YIELDS.get(profile, 0.02)
        profiles[profile] = block
    return profiles


def extract_baseline_market(country_df: pd.DataFrame) -> dict[str, float]:
    us = country_df.loc["United States"]
    per = _safe_float(us.get("median(Trailing PE)"))
    pb = _safe_float(us.get("median(PBV)"))
    ps = _safe_float(us.get("median(PS)"))
    if not all((per, pb, ps)):
        raise ValueError("Médianes US introuvables dans countrystats.")
    return {"per": round(per, 2), "pb": round(pb, 3), "ps": round(ps, 3)}


def _country_scale_row(
    row: pd.Series,
    baseline: Mapping[str, float],
    *,
    div_yield_hint: float | None = None,
) -> dict[str, float]:
    per = _safe_float(row.get("median(Trailing PE)"))
    pb = _safe_float(row.get("median(PBV)"))
    ps = _safe_float(row.get("median(PS)"))
    per_scale = round(per / baseline["per"], 3) if per else 1.0
    pb_scale = round(pb / baseline["pb"], 3) if pb else 1.0
    ps_scale = round(ps / baseline["ps"], 3) if ps else 1.0
    if div_yield_hint is not None:
        div_yield = div_yield_hint
    else:
        div_yield = round(min(max(1.05 / max(per_scale, 0.35), 0.85), 1.30), 3)
    return {
        "per": per_scale,
        "pb": pb_scale,
        "ps": ps_scale,
        "div_yield": div_yield,
    }


def compute_country_scales(
    country_df: pd.DataFrame,
    baseline: Mapping[str, float],
    country_names: list[str],
    *,
    existing_countries: Mapping[str, Mapping] | None = None,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {
        "default": {"per": 1.0, "pb": 1.0, "ps": 1.0, "div_yield": 1.0},
        "United States": {"per": 1.0, "pb": 1.0, "ps": 1.0, "div_yield": 1.0},
    }
    existing_countries = existing_countries or {}
    for name in country_names:
        if name not in country_df.index:
            continue
        row = country_df.loc[name]
        per = _safe_float(row.get("median(Trailing PE)"))
        pb = _safe_float(row.get("median(PBV)"))
        ps = _safe_float(row.get("median(PS)"))
        hint = (existing_countries.get(name) or {}).get("div_yield")
        scales = _country_scale_row(row, baseline, div_yield_hint=hint)
        block = dict(scales)
        if per and pb and ps:
            block["source"] = (
                f"Trailing PE {per:.2f}, PB {pb:.2f}, PS {ps:.2f} "
                f"vs US {baseline['per']} / {baseline['pb']} / {baseline['ps']}"
            )
        out[name] = block

    for key in ("France", "Germany", "United Kingdom"):
        if all(k in out for k in (key,)):
            pass
    if all(k in out for k in ("France", "Germany", "United Kingdom")):
        out["Europe"] = {
            "per": round((out["France"]["per"] + out["Germany"]["per"] + out["United Kingdom"]["per"]) / 3, 3),
            "pb": round((out["France"]["pb"] + out["Germany"]["pb"] + out["United Kingdom"]["pb"]) / 3, 3),
            "ps": round((out["France"]["ps"] + out["Germany"]["ps"] + out["United Kingdom"]["ps"]) / 3, 3),
            "div_yield": round(
                (out["France"]["div_yield"] + out["Germany"]["div_yield"] + out["United Kingdom"]["div_yield"]) / 3,
                3,
            ),
            "source": "Moyenne France / Allemagne / Royaume-Uni — repli suffixes EU",
        }
    return out


def load_existing_config(path: Path | None = None) -> dict:
    path = path or DEFAULT_YAML_PATH
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def merge_benchmark_config(
    existing: Mapping[str, Any],
    pe_df: pd.DataFrame,
    pb_df: pd.DataFrame,
    ps_df: pd.DataFrame,
    country_df: pd.DataFrame,
    *,
    updated: str | None = None,
) -> dict[str, Any]:
    """Fusionne Damodaran dans la config existante (préserve tech/minières et suffixes)."""
    baseline = extract_baseline_market(country_df)
    damodaran_profiles = compute_profile_multiples_from_damodaran(pe_df, pb_df, ps_df)
    existing_profiles = existing.get("profiles") or {}
    merged_profiles: dict[str, Any] = {}

    for profile in sorted(set(existing_profiles) | set(damodaran_profiles)):
        old = existing_profiles.get(profile) or {}
        new = damodaran_profiles.get(profile) or {}
        if profile in PRESERVE_PROFILE_MULTIPLES:
            merged_profiles[profile] = {
                "source_industry": old.get(
                    "source_industry",
                    new.get("source_industry", f"{profile} — base calibrée ma vue"),
                ),
                "multiples": dict((old.get("multiples") or new.get("multiples") or {})),
            }
        elif profile in damodaran_profiles:
            merged_profiles[profile] = new
        else:
            merged_profiles[profile] = old

    if "tech" not in merged_profiles and "tech" in existing_profiles:
        merged_profiles["tech"] = existing_profiles["tech"]

    countries = compute_country_scales(
        country_df,
        baseline,
        list(COUNTRIES_TO_UPDATE),
        existing_countries=existing.get("countries") or {},
    )

    update_label = updated or date.today().strftime("%Y-%m")
    return {
        "meta": {
            "version": int((existing.get("meta") or {}).get("version", 0)) + 1,
            "updated": update_label,
            "baseline_country": "United States",
            "generator": "update_sector_benchmarks.py",
        },
        "baseline_market": {
            **baseline,
            "source": f"Damodaran countrystats — median Trailing PE / PBV / PS, {update_label}",
        },
        "countries": countries,
        "ticker_suffix": dict(existing.get("ticker_suffix") or {}),
        "profiles": merged_profiles,
    }


def write_benchmark_yaml(cfg: Mapping[str, Any], path: Path | None = None) -> Path:
    path = path or DEFAULT_YAML_PATH
    body = yaml.safe_dump(dict(cfg), sort_keys=False, allow_unicode=True, default_flow_style=False)
    path.write_text(YAML_HEADER + body, encoding="utf-8")
    return path


def fetch_and_merge(
    *,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    existing_path: Path | None = None,
) -> dict[str, Any]:
    existing = load_existing_config(existing_path)
    pe_df = parse_industry_averages(download_dataset("pedata", cache_dir=cache_dir, use_cache=use_cache))
    pb_df = parse_industry_averages(download_dataset("pbvdata", cache_dir=cache_dir, use_cache=use_cache))
    ps_df = parse_industry_averages(download_dataset("psdata", cache_dir=cache_dir, use_cache=use_cache))
    country_df = parse_countrystats(download_dataset("countrystats", cache_dir=cache_dir, use_cache=use_cache))
    return merge_benchmark_config(existing, pe_df, pb_df, ps_df, country_df)


def bump_fair_value_engine_version(fundamentals_path: Path | None = None) -> int:
    """Incrémente FAIR_VALUE_ENGINE_VERSION dans fundamentals.py."""
    path = fundamentals_path or ROOT / "fundamentals.py"
    text = path.read_text(encoding="utf-8")
    match = re.search(r"^FAIR_VALUE_ENGINE_VERSION\s*=\s*(\d+)\s*$", text, re.MULTILINE)
    if not match:
        raise ValueError("FAIR_VALUE_ENGINE_VERSION introuvable dans fundamentals.py")
    old = int(match.group(1))
    new = old + 1
    text = re.sub(
        r"^FAIR_VALUE_ENGINE_VERSION\s*=\s*\d+\s*$",
        f"FAIR_VALUE_ENGINE_VERSION = {new}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    path.write_text(text, encoding="utf-8")
    return new
