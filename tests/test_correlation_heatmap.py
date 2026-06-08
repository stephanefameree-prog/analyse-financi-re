"""Tests heatmap corrélation clusterisée."""
import numpy as np
import pandas as pd
import pytest

from analytics import build_correlation_heatmap_figure, cluster_correlation_order


def _sample_corr(n=5, seed=0):
    rng = np.random.default_rng(seed)
    returns = pd.DataFrame(
        rng.normal(0, 0.01, size=(120, n)),
        columns=[f"T{i}" for i in range(n)],
    )
    returns["T1"] = returns["T0"] * 0.9 + rng.normal(0, 0.003, 120)
    returns["T2"] = returns["T0"] * 0.85 + rng.normal(0, 0.003, 120)
    return returns.corr()


def test_cluster_correlation_order_keeps_all_labels():
    corr = _sample_corr(6)
    order = cluster_correlation_order(corr)
    assert len(order) == len(corr)
    assert set(order) == set(corr.index)


def test_triangle_masks_lower_triangle():
    corr = _sample_corr(4)
    fig = build_correlation_heatmap_figure(
        corr, cluster=False, triangle=True, rho_min=-1.0, rho_max=1.0
    )
    z = np.array(fig.data[0].z, dtype=float)
    below = np.tril(np.ones_like(z, dtype=bool), k=-1)
    assert np.all(np.isnan(z[below]))


def test_rho_range_filter_keeps_only_values_in_band():
    corr = _sample_corr(4)
    fig = build_correlation_heatmap_figure(
        corr, cluster=False, triangle=False, rho_min=-0.15, rho_max=0.15
    )
    z = np.array(fig.data[0].z, dtype=float)
    visible = z[~np.isnan(z)]
    assert len(visible) > 0
    assert np.all(visible >= -0.15 - 1e-9)
    assert np.all(visible <= 0.15 + 1e-9)
    assert fig.data[0].zmin == pytest.approx(-0.15)
    assert fig.data[0].zmax == pytest.approx(0.15)


def test_build_correlation_heatmap_figure_uses_diverging_scale():
    corr = _sample_corr(3)
    fig = build_correlation_heatmap_figure(corr, cluster=True, triangle=False)
    heatmap = fig.data[0]
    assert heatmap.zmid == 0
    assert heatmap.zmin == -1
    assert heatmap.zmax == 1
    assert fig.layout.height >= 400
