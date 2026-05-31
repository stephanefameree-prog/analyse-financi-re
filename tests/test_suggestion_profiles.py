"""Tests des libellés de profil (Suggestions d'actifs)."""
import itertools

import pytest

from analytics import compute_profile_scores, describe_suggestion_profile

DEFAULT_WEIGHTS = {
    "return": 1.0,
    "vol": 1.0,
    "kurt": 0.7,
    "skew": 0.8,
    "corr_internal": 1.0,
    "corr_candidate": 1.0,
}


@pytest.mark.parametrize(
    "weights, expected_fragments",
    [
        (DEFAULT_WEIGHTS, ["équilibré"]),
        (
            {"return": 1.5, "vol": 0.5, "kurt": 0.5, "skew": 1.5, "corr_internal": 0.5, "corr_candidate": 0.5},
            ["agressif", "dynamique", "opportuniste", "rendement"],
        ),
        (
            {"return": 0.5, "vol": 1.8, "kurt": 1.8, "skew": 0.5, "corr_internal": 1.0, "corr_candidate": 1.0},
            ["prudent", "conservateur", "défensif"],
        ),
        (
            {"return": 1.0, "vol": 1.0, "kurt": 1.0, "skew": 1.0, "corr_internal": 1.6, "corr_candidate": 1.6},
            ["diversificateur"],
        ),
        (
            {"return": 0.0, "vol": 0.0, "kurt": 0.0, "skew": 0.0, "corr_internal": 0.0, "corr_candidate": 0.0},
            ["neutre"],
        ),
        (
            {"return": 1.2, "vol": 1.3, "kurt": 1.2, "skew": 1.4, "corr_internal": 1.5, "corr_candidate": 1.5},
            ["diversificateur", "opportuniste"],
        ),
    ],
)
def test_profile_keywords(weights, expected_fragments):
    title, desc = describe_suggestion_profile(weights)
    blob = f"{title} {desc}".lower()
    assert any(fragment in blob for fragment in expected_fragments)


def test_default_weights_balanced_not_mixte():
    title, _ = describe_suggestion_profile(DEFAULT_WEIGHTS)
    assert "mixte" not in title.lower()
    assert "équilibré" in title.lower()


def test_dual_profile_when_two_axes_close():
    weights = {
        "return": 1.5,
        "vol": 0.5,
        "kurt": 0.5,
        "skew": 1.5,
        "corr_internal": 1.5,
        "corr_candidate": 1.5,
    }
    title, desc = describe_suggestion_profile(weights)
    assert " · " in title
    assert "dynamique" in title.lower() or "opportuniste" in title.lower()


def test_incompatible_profiles_not_paired():
    """Conservateur + prudent ne doivent pas apparaître ensemble (redondant)."""
    weights = {
        "return": 0.6,
        "vol": 1.7,
        "kurt": 1.7,
        "skew": 0.5,
        "corr_internal": 1.55,
        "corr_candidate": 1.55,
    }
    title, _ = describe_suggestion_profile(weights)
    assert "conservateur" in title.lower()
    assert "prudent" not in title.lower()


def test_compute_profile_scores_neutral():
    scores = compute_profile_scores(
        {"return": 0, "vol": 0, "kurt": 0, "skew": 0, "corr_internal": 0, "corr_candidate": 0}
    )
    assert scores == {"Profil neutre": 1.0}


def test_coarse_grid_mixte_rate_below_threshold():
    """Sur une grille grossière, « Profil mixte » doit rester minoritaire."""
    steps = [0.0, 0.5, 1.0, 1.5, 2.0]
    keys = ["return", "vol", "kurt", "skew", "corr_internal", "corr_candidate"]
    mixte = 0
    total = 0
    for vals in itertools.product(steps, repeat=6):
        weights = dict(zip(keys, vals))
        title, _ = describe_suggestion_profile(weights)
        total += 1
        if title == "Profil mixte":
            mixte += 1
    assert mixte / total < 0.06


def test_coarse_grid_balanced_reaches_reasonable_share():
    steps = [0.0, 0.5, 1.0, 1.5, 2.0]
    keys = ["return", "vol", "kurt", "skew", "corr_internal", "corr_candidate"]
    balanced = 0
    total = 0
    for vals in itertools.product(steps, repeat=6):
        weights = dict(zip(keys, vals))
        title, _ = describe_suggestion_profile(weights)
        total += 1
        if "équilibré" in title.lower():
            balanced += 1
    assert balanced / total >= 0.02
