"""Tests mise à jour sector_benchmarks.yaml (sans réseau)."""
import io

import pandas as pd
import pytest

import fair_value_benchmarks as fb


def _sample_pe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Industry Name": ["Drugs (Pharmaceutical)", "Banks (Regional)"],
            fb.PE_AGG_COL: [24.0, 12.0],
            "Trailing PE": [55.0, 33.0],
        }
    ).set_index("Industry Name", drop=False)


def _sample_pb() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Industry Name": ["Drugs (Pharmaceutical)", "Banks (Regional)"],
            "PBV": [6.6, 1.14],
        }
    ).set_index("Industry Name", drop=False)


def _sample_ps() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Industry Name": ["Drugs (Pharmaceutical)", "Banks (Regional)"],
            "Price/Sales": [5.6, 3.6],
        }
    ).set_index("Industry Name", drop=False)


def _sample_countries() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Country": ["United States", "France"],
            "median(Trailing PE)": [22.0, 16.0],
            "median(PBV)": [2.0, 1.2],
            "median(PS)": [2.0, 1.0],
        }
    ).set_index("Country", drop=False)


def test_compute_profile_multiples():
    profiles = fb.compute_profile_multiples_from_damodaran(_sample_pe(), _sample_pb(), _sample_ps())
    assert profiles["pharma"]["multiples"]["per"] == 24.0
    assert profiles["finance"]["multiples"]["per"] == 12.0


def test_compute_country_scales():
    baseline = {"per": 22.0, "pb": 2.0, "ps": 2.0}
    scales = fb.compute_country_scales(_sample_countries(), baseline, ["France"])
    assert scales["France"]["per"] == pytest.approx(16.0 / 22.0, rel=1e-3)
    assert scales["United States"]["per"] == 1.0


def test_merge_preserves_tech_minieres():
    existing = {
        "meta": {"version": 3},
        "profiles": {
            "tech": {"multiples": {"per": 28.0, "pb": 5.0, "ps": 8.0, "div_yield": 0.01}},
            "minières": {"multiples": {"per": 15.0, "pb": 1.35, "ps": 2.0, "div_yield": 0.035}},
        },
        "ticker_suffix": {"PA": "France"},
    }
    merged = fb.merge_benchmark_config(
        existing,
        _sample_pe(),
        _sample_pb(),
        _sample_ps(),
        _sample_countries(),
        updated="2026-06",
    )
    assert merged["profiles"]["tech"]["multiples"]["per"] == 28.0
    assert merged["profiles"]["minières"]["multiples"]["per"] == 15.0
    assert merged["ticker_suffix"]["PA"] == "France"
    assert merged["meta"]["version"] == 4


def test_parse_industry_header_detection():
    raw = pd.DataFrame(
        [
            ["Date updated:", "2026-01-05"],
            [None, None],
            ["Industry Name", "Trailing PE", fb.PE_AGG_COL],
            ["Drugs (Pharmaceutical)", 55.0, 24.0],
        ]
    )
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        raw.to_excel(writer, sheet_name="Industry Averages", index=False, header=False)
    parsed = fb.parse_industry_averages(buf.getvalue())
    assert parsed.loc["Drugs (Pharmaceutical)", fb.PE_AGG_COL] == 24.0
