"""Poids historiques du portefeuille avec FX USD→EUR."""
import numpy as np
import pandas as pd
import pytest

from analytics import (
    build_portfolio_holdings_eur,
    build_portfolio_weight_pct_history,
)


def test_portfolio_weight_history_with_fx():
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    prices = pd.DataFrame(
        {
            "EUR_T": [100.0, 102.0, 101.0, 103.0, 104.0],
            "USD_T": [50.0, 51.0, 52.0, 51.5, 53.0],
        },
        index=dates,
    )
    fx = pd.Series([0.90, 0.91, 0.92, 0.91, 0.93], index=dates)
    quantities = {"EUR_T": 10.0, "USD_T": 20.0}
    is_usd = lambda t: t == "USD_T"

    values = build_portfolio_holdings_eur(
        prices, quantities, usd_to_eur_series=fx, is_usd_fn=is_usd
    )
    assert values.loc[dates[0], "EUR_T"] == 1000.0
    assert values.loc[dates[0], "USD_T"] == pytest.approx(50 * 20 * 0.90)

    weights = build_portfolio_weight_pct_history(
        prices, quantities, usd_to_eur_series=fx, is_usd_fn=is_usd
    )
    assert len(weights) == 5
    assert np.allclose(weights.sum(axis=1), 100.0, rtol=1e-5)
    assert (weights["EUR_T"] > 0).all()
    assert (weights["USD_T"] > 0).all()


def test_portfolio_weight_history_usd_only():
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    prices = pd.DataFrame({"USD_T": [100.0, 100.0, 100.0]}, index=dates)
    fx = pd.Series([0.5, 0.5, 0.5], index=dates)
    weights = build_portfolio_weight_pct_history(
        prices,
        {"USD_T": 1.0},
        usd_to_eur_series=fx,
        is_usd_fn=lambda t: True,
    )
    assert len(weights) == 3
    assert (weights["USD_T"] == 100.0).all()
