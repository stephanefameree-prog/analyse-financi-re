"""Tests graphiques synthèse & risque (Lot 2–3)."""
import numpy as np
import pandas as pd
import pytest

from analytics import (
    build_portfolio_allocation_treemap,
    build_portfolio_drawdown_figure,
    build_returns_distribution_var_figure,
    build_risk_return_scatter_figure,
    compute_drawdown_series,
)


def test_compute_drawdown_series():
    s = pd.Series([100, 110, 105, 120, 90], index=pd.date_range("2024-01-01", periods=5))
    dd = compute_drawdown_series(s)
    assert float(dd.iloc[1]) == 0.0
    assert float(dd.iloc[2]) == pytest.approx((105 - 110) / 110)
    assert float(dd.min()) == pytest.approx((90 - 120) / 120)


def test_build_portfolio_drawdown_figure():
    s = pd.Series(np.linspace(100, 80, 30), index=pd.date_range("2024-01-01", periods=30))
    fig = build_portfolio_drawdown_figure(s)
    assert fig is not None
    assert fig.layout.height >= 300


def test_build_portfolio_allocation_treemap_with_sector():
    df = pd.DataFrame(
        {
            "Ticker": ["A", "B"],
            "Nom": ["Alpha", "Beta"],
            "Secteur": ["Tech", "Tech"],
            "Valeur Actuelle (€)": [6000.0, 4000.0],
        }
    )
    fig = build_portfolio_allocation_treemap(df)
    assert fig is not None
    assert len(fig.data) == 1


def test_build_risk_return_scatter_figure():
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(
        {
            "A": rng.normal(0.001, 0.02, 100),
            "B": rng.normal(0.0005, 0.01, 100),
        }
    )
    fig = build_risk_return_scatter_figure(returns, weights=pd.Series({"A": 0.7, "B": 0.3}))
    assert fig is not None
    assert fig.data[0].x is not None


def test_build_returns_distribution_var_figure():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0, 0.015, 200))
    fig = build_returns_distribution_var_figure(r, asset_label="Test")
    assert fig is not None
