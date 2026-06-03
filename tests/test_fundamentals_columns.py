"""Ordre des colonnes fondamentaux (comparaison analystes / juste valeur)."""
import pandas as pd

import fundamentals as fm


def test_valuation_compare_columns_side_by_side():
    df = pd.DataFrame(
        [
            {
                "Nom": "Test SA",
                "Dernier cours": 10.0,
                "PER": 12.0,
                "Objectif analystes": 14.0,
                "Juste valeur estimée": 11.5,
                "Upside vs objectif": 0.4,
                "Upside juste valeur": 0.15,
                "Juste valeur Damodaran": 12.0,
                "Upside Damodaran": 0.2,
                "Profil Damodaran": "tech",
                "Écart juste valeur": 1.5,
                "Libellé juste valeur": "Juste",
                "Modèle juste valeur": "tech",
                "Objectif bas": 12.0,
                "Ticker": "TST",
            }
        ]
    )
    out = fm._reorder_fair_value_columns(df)
    cols = list(out.columns)
    assert cols.index("Objectif analystes") + 1 == cols.index("Juste valeur estimée")
    assert cols.index("Upside vs objectif") + 1 == cols.index("Upside juste valeur")
    assert cols.index("Upside juste valeur") + 1 == cols.index("Juste valeur Damodaran")
    assert cols.index("Juste valeur Damodaran") + 1 == cols.index("Upside Damodaran")
    assert cols.index("Upside Damodaran") + 1 == cols.index("Profil Damodaran")
    assert cols.index("Profil Damodaran") + 1 == cols.index("Écart juste valeur")
    assert cols.index("Dernier cours") < cols.index("Objectif analystes")
    assert cols.index("Upside juste valeur") < cols.index("PER")


def test_valuation_upside_radar_figure():
    row = pd.Series(
        {
            "Upside vs objectif": 0.126,
            "Upside juste valeur": 0.401,
            "Upside Damodaran": 0.375,
        }
    )
    fig = fm.build_valuation_upside_radar(row, "Edenred")
    assert fig is not None
    fills = [t.fillcolor for t in fig.data if getattr(t, "fillcolor", None)]
    assert fm._RADAR_POS_FILL in fills
    polygon = fig.data[-1]
    assert polygon.mode == "lines+markers"
    assert len(polygon.theta) == 4
    assert all(0 <= t < 360 for t in polygon.theta)


def test_radar_axis_angles_normalized():
    angles = fm._radar_axis_angles(3)
    assert angles == [90.0, 330.0, 210.0]


def test_valuation_upside_radar_mixed_signs():
    row = pd.Series(
        {
            "Upside vs objectif": 0.2,
            "Upside juste valeur": -0.1,
            "Upside Damodaran": 0.05,
        }
    )
    fig = fm.build_valuation_upside_radar(row, "Mix")
    assert fig is not None
    fills = [t.fillcolor for t in fig.data if getattr(t, "fillcolor", None)]
    assert fm._RADAR_POS_FILL in fills
    assert fm._RADAR_NEG_FILL in fills


def test_valuation_upside_radar_requires_two_axes():
    row = pd.Series({"Upside vs objectif": 0.1})
    assert fm.build_valuation_upside_radar(row, "X") is None
