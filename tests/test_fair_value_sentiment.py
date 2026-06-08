"""Couleurs vert/rouge colonnes juste valeur et Damodaran."""
import pandas as pd

import fundamentals as fm


def test_damodaran_columns_use_cell_and_upside():
    row = pd.Series(
        {
            "Dernier cours": 100.0,
            "Juste valeur Damodaran": 120.0,
            "Upside Damodaran": 0.20,
            "Profil Damodaran": "tech",
            "Upside juste valeur": -0.10,
        }
    )
    assert fm._fair_value_cell_sentiment(row, "Upside Damodaran", 0.20) == fm._STYLE_FAVORABLE
    assert fm._fair_value_cell_sentiment(row, "Juste valeur Damodaran", 120.0) == fm._STYLE_FAVORABLE
    assert fm._fair_value_cell_sentiment(row, "Profil Damodaran", "tech") == fm._STYLE_FAVORABLE


def test_damodaran_overvalued_red():
    row = pd.Series(
        {
            "Dernier cours": 100.0,
            "Juste valeur Damodaran": 85.0,
            "Upside Damodaran": -0.15,
            "Profil Damodaran": "finance",
        }
    )
    assert fm._fair_value_cell_sentiment(row, "Upside Damodaran", -0.15) == fm._STYLE_UNFAVORABLE
    assert fm._fair_value_cell_sentiment(row, "Juste valeur Damodaran", 85.0) == fm._STYLE_UNFAVORABLE
    assert fm._fair_value_cell_sentiment(row, "Profil Damodaran", "finance") == fm._STYLE_UNFAVORABLE


def test_ridge_columns_use_own_cell_not_damodaran():
    row = pd.Series(
        {
            "Dernier cours": 100.0,
            "Juste valeur estimée": 90.0,
            "Upside juste valeur": -0.10,
            "Écart juste valeur": -10.0,
            "Upside Damodaran": 0.25,
        }
    )
    assert fm._fair_value_cell_sentiment(row, "Upside juste valeur", -0.10) == fm._STYLE_UNFAVORABLE
    assert fm._fair_value_cell_sentiment(row, "Juste valeur estimée", 90.0) == fm._STYLE_UNFAVORABLE
    assert fm._fair_value_cell_sentiment(row, "Écart juste valeur", -10.0) == fm._STYLE_UNFAVORABLE
