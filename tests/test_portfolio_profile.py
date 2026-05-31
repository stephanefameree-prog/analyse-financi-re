"""Tests du profil portefeuille (métriques réelles → libellés)."""
import numpy as np
import pytest

from analytics import (
    describe_portfolio_profile,
    metrics_to_pseudo_profile_weights,
)


def _metrics(ret, vol, skew=0.0, kurt=0.0):
    return {
        "Rendement Annuel (moyenne)": ret,
        "Volatilité (Sigma)": vol,
        "Skewness (Asymétrie)": skew,
        "Kurtosis (Aplatissement)": kurt,
    }


def test_pseudo_weights_high_return_low_vol():
    w = metrics_to_pseudo_profile_weights(_metrics(0.12, 0.10), internal_corr=0.4)
    assert w["return"] > 1.0
    assert w["corr_internal"] > 1.0


def test_pseudo_weights_low_diversification_high_corr():
    w = metrics_to_pseudo_profile_weights(_metrics(0.05, 0.15), internal_corr=0.85)
    assert w["corr_internal"] < 0.5


@pytest.mark.parametrize(
    "metrics, corr, expected_fragments",
    [
        (_metrics(0.12, 0.09, skew=0.6), 0.35, ["rendement", "dynamique", "agressif", "opportuniste", "diversificateur"]),
        (_metrics(0.02, 0.18, skew=-0.2, kurt=2.0), 0.75, ["prudent", "conservateur", "défensif"]),
        (_metrics(0.06, 0.14, skew=0.1, kurt=0.5), 0.30, ["diversificateur", "équilibré"]),
    ],
)
def test_portfolio_profile_keywords(metrics, corr, expected_fragments):
    title, desc, _, _ = describe_portfolio_profile(metrics, internal_corr=corr)
    blob = f"{title} {desc}".lower()
    assert any(fragment in blob for fragment in expected_fragments)


def test_portfolio_profile_returns_pseudo_and_scores():
    title, desc, pseudo, scores = describe_portfolio_profile(
        _metrics(0.08, 0.14), internal_corr=0.55
    )
    assert title
    assert desc
    assert set(pseudo.keys()) == {
        "return",
        "vol",
        "kurt",
        "skew",
        "corr_internal",
        "corr_candidate",
    }
    assert scores and max(scores.values()) > 0
