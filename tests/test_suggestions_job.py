"""Job suggestions par étapes."""
from suggestions_job import _build_candidate_pool


def test_build_candidate_pool():
    universe = {"CAC40": ["A.PA", "B.PA"], "US": ["AAPL", "MSFT"]}
    params = {
        "selected_universes": ["CAC40", "US"],
        "portfolio_tickers": ["A.PA"],
        "max_candidates": 10,
    }
    pool = _build_candidate_pool(params, universe)
    assert pool == ["B.PA", "AAPL", "MSFT"]
