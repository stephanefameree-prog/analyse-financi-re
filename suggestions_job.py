"""Génération des suggestions d'actifs par étapes (survit au changement de vue Streamlit)."""
from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd
import streamlit as st

from analytics import merge_technical_columns, merge_suggestion_latent_returns
from dashboard_cache import (
    cached_candidate_market_data,
    cached_candidate_technical_indicators,
    cached_suggest_portfolio_additions,
)
from data_loader import _tickers_signature

FUND_BATCH = 20
DIV_BATCH = 20

_JOB_KEY = "suggestions_job"
_WIP_KEY = "suggestions_wip"


def is_active() -> bool:
    job = st.session_state.get(_JOB_KEY)
    return bool(job and job.get("status") in ("pending", "running"))


def is_done() -> bool:
    job = st.session_state.get(_JOB_KEY)
    return bool(job and job.get("status") == "done")


def start(params: dict) -> None:
    st.session_state[_JOB_KEY] = {
        "status": "pending",
        "phase": "init",
        "message": "Préparation…",
        "params": params,
        "fund_offset": 0,
        "div_offset": 0,
        "fund_rows": [],
        "div_rows": [],
        "fund_stats": {"session": 0, "disk": 0, "fetched": 0, "missing": 0},
        "div_stats": {"session": 0, "disk": 0, "fetched": 0, "missing": 0},
    }
    st.session_state.pop(_WIP_KEY, None)


def cancel() -> None:
    st.session_state.pop(_JOB_KEY, None)
    st.session_state.pop(_WIP_KEY, None)


def dismiss() -> None:
    st.session_state.pop(_JOB_KEY, None)


def _merge_stats(acc: dict, new: dict) -> None:
    for k in acc:
        acc[k] = acc.get(k, 0) + new.get(k, 0)


def _load_index_universe(json_path: str) -> dict:
    if not json_path or not os.path.exists(json_path):
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_candidate_pool(params: dict, index_universe: dict) -> list[str]:
    portfolio_tickers = params["portfolio_tickers"]
    pool: list[str] = []
    for universe in params.get("selected_universes") or []:
        pool.extend(index_universe.get(universe, []))
    max_candidates = int(params.get("max_candidates") or 60)
    return [
        t
        for t in dict.fromkeys(pool)
        if t not in portfolio_tickers
    ][:max_candidates]


def _finalize_job(job: dict, wip: dict) -> None:
    params = job["params"]
    if wip.get("suggestions") is not None and not wip["suggestions"].empty:
        prices = wip.get("candidate_prices")
        if prices is not None and not prices.empty:
            wip["suggestions"] = merge_suggestion_latent_returns(
                wip["suggestions"], prices
            )
    st.session_state["portfolio_suggestions"] = wip["suggestions"]
    st.session_state["portfolio_suggestions_baseline"] = wip["baseline"]
    st.session_state["portfolio_suggestions_internal_corr"] = wip["baseline_internal"]
    st.session_state["portfolio_suggestions_meta"] = {
        "pool_size": len(wip.get("candidate_pool") or []),
        "weight": params.get("candidate_weight"),
        "weights": dict(params.get("objective_weights") or {}),
        "include_fundamentals": params.get("include_fundamentals"),
        "include_technical": params.get("include_technical"),
        "include_dividends": params.get("include_dividends"),
        "selection_mode": params.get("selection_mode"),
        "fund_stats": job.get("fund_stats"),
        "div_stats": job.get("div_stats"),
    }
    job["status"] = "done"
    job["message"] = f"{len(wip['suggestions'])} lignes prêtes"
    st.session_state.pop(_WIP_KEY, None)


def _complete_job(job: dict, wip: dict) -> None:
    _finalize_job(job, wip)
    st.session_state[_JOB_KEY] = job


def run_step(*, returns: pd.DataFrame, ohlcv_sig: str, has_fundamentals: bool, has_dividendes: bool) -> bool:
    """
    Avance d'une étape. Retourne True si d'autres étapes restent (relancer le script).
    """
    job = st.session_state.get(_JOB_KEY)
    if not job or job.get("status") not in ("pending", "running"):
        return False

    job["status"] = "running"
    params = job["params"]
    phase = job.get("phase", "init")
    wip = st.session_state.get(_WIP_KEY) or {}

    try:
        if phase == "init":
            job["message"] = "Construction de la liste de candidats…"
            index_universe = _load_index_universe(params.get("json_path", ""))
            pool = _build_candidate_pool(params, index_universe)
            if not pool:
                job["status"] = "error"
                job["error"] = "Aucun candidat dans les univers sélectionnés."
                st.session_state[_JOB_KEY] = job
                return False
            wip["candidate_pool"] = pool
            st.session_state[_WIP_KEY] = wip
            job["phase"] = "market_data"
            job["message"] = f"Téléchargement des cours ({len(pool)} candidats)…"

        elif phase == "market_data":
            pool = wip["candidate_pool"]
            start_date = params["start_date"]
            candidate_sig = _tickers_signature(pool, start_date)
            candidate_prices, candidate_returns = cached_candidate_market_data(
                candidate_sig,
                tuple(pool),
                str(start_date),
            )
            wip["candidate_prices"] = candidate_prices
            wip["candidate_returns"] = candidate_returns
            st.session_state[_WIP_KEY] = wip
            job["phase"] = "core"
            job["message"] = "Calcul des scores et impacts sur le portefeuille…"

        elif phase == "core":
            pool = wip["candidate_pool"]
            start_date = params["start_date"]
            portfolio_tickers = tuple(params["portfolio_tickers"])
            objective_weights = params.get("objective_weights") or {}
            suggestions, baseline, baseline_internal = cached_suggest_portfolio_additions(
                ohlcv_sig,
                _tickers_signature(pool, start_date),
                portfolio_tickers,
                tuple(pool),
                str(start_date),
                float(params.get("candidate_weight") or 0.1),
                tuple(sorted(objective_weights.items())),
            )
            wip["suggestions"] = suggestions
            wip["baseline"] = baseline
            wip["baseline_internal"] = baseline_internal
            st.session_state[_WIP_KEY] = wip
            if suggestions is None or suggestions.empty:
                job["status"] = "error"
                job["error"] = "Aucune suggestion pertinente calculée."
                st.session_state[_JOB_KEY] = job
                return False
            if params.get("include_technical"):
                job["phase"] = "technical"
                job["message"] = "Indicateurs techniques…"
            elif params.get("include_fundamentals") and has_fundamentals:
                job["phase"] = "fundamentals"
                job["fund_offset"] = 0
                job["fund_rows"] = []
                job["message"] = "Fondamentaux (lot 1)…"
            elif params.get("include_dividends") and has_dividendes:
                job["phase"] = "dividends"
                job["div_offset"] = 0
                job["div_rows"] = []
                job["message"] = "Dividendes (lot 1)…"
            else:
                _complete_job(job, wip)
                return False

        elif phase == "technical":
            start_date = params["start_date"]
            tickers = tuple(wip["suggestions"]["Ticker"].tolist())
            tech_sig = _tickers_signature(tickers, start_date)
            tech_df = cached_candidate_technical_indicators(
                tech_sig,
                tickers,
                str(start_date),
            )
            wip["suggestions"] = merge_technical_columns(wip["suggestions"], tech_df)
            st.session_state[_WIP_KEY] = wip
            if params.get("include_fundamentals") and has_fundamentals:
                job["phase"] = "fundamentals"
                job["fund_offset"] = 0
                job["fund_rows"] = []
                job["message"] = "Fondamentaux (lot 1)…"
            elif params.get("include_dividends") and has_dividendes:
                job["phase"] = "dividends"
                job["div_offset"] = 0
                job["div_rows"] = []
                job["message"] = "Dividendes (lot 1)…"
            else:
                _complete_job(job, wip)
                return False

        elif phase == "fundamentals":
            import fundamentals

            tickers = wip["suggestions"]["Ticker"].tolist()
            offset = int(job.get("fund_offset") or 0)
            batch = tickers[offset : offset + FUND_BATCH]
            cache_key = "suggestions_fundamentals_cache"
            fund_cache = st.session_state.setdefault(cache_key, {})
            job["message"] = (
                f"Fondamentaux : {min(offset + len(batch), len(tickers))}/{len(tickers)}"
            )
            rows, fund_cache, stats = fundamentals.fetch_fundamentals_for_tickers(
                batch,
                wip["candidate_prices"],
                cache=fund_cache,
            )
            st.session_state[cache_key] = fund_cache
            job.setdefault("fund_rows", []).extend(rows)
            _merge_stats(job["fund_stats"], stats)
            job["fund_offset"] = offset + len(batch)
            if job["fund_offset"] >= len(tickers):
                wip["suggestions"] = fundamentals.merge_fundamentals_columns(
                    wip["suggestions"], job["fund_rows"]
                )
                st.session_state[_WIP_KEY] = wip
                job["fund_rows"] = []
                if params.get("include_dividends") and has_dividendes:
                    job["phase"] = "dividends"
                    job["div_offset"] = 0
                    job["div_rows"] = []
                    job["message"] = "Dividendes (lot 1)…"
                else:
                    _complete_job(job, wip)
                    return False
            else:
                job["message"] = (
                    f"Fondamentaux : {job['fund_offset']}/{len(tickers)} — "
                    "vous pouvez changer de page"
                )

        elif phase == "dividends":
            import dividendes

            tickers = wip["suggestions"]["Ticker"].tolist()
            offset = int(job.get("div_offset") or 0)
            batch = tickers[offset : offset + DIV_BATCH]
            cache_key = "suggestions_dividends_cache"
            div_cache = st.session_state.setdefault(cache_key, {})
            job["message"] = (
                f"Dividendes : {min(offset + len(batch), len(tickers))}/{len(tickers)}"
            )
            rows, div_cache, stats = dividendes.fetch_dividends_for_tickers(
                batch,
                wip["candidate_prices"],
                cache=div_cache,
            )
            st.session_state[cache_key] = div_cache
            job.setdefault("div_rows", []).extend(rows)
            _merge_stats(job["div_stats"], stats)
            job["div_offset"] = offset + len(batch)
            if job["div_offset"] >= len(tickers):
                wip["suggestions"] = dividendes.merge_dividend_columns(
                    wip["suggestions"], job["div_rows"]
                )
                st.session_state[_WIP_KEY] = wip
                job["div_rows"] = []
                _complete_job(job, wip)
                return False
            else:
                job["message"] = (
                    f"Dividendes : {job['div_offset']}/{len(tickers)} — "
                    "vous pouvez changer de page"
                )

        elif phase == "finalize":
            _complete_job(job, wip)
            return False

    except Exception as exc:
        job["status"] = "error"
        job["error"] = str(exc)
        st.session_state[_JOB_KEY] = job
        return False

    job["status"] = "pending"
    st.session_state[_JOB_KEY] = job
    return job.get("phase") != "finalize" and job.get("status") != "done"


def render_sidebar_status() -> None:
    job = st.session_state.get(_JOB_KEY)
    if not job:
        return

    status = job.get("status")
    if status == "done":
        st.sidebar.success(f"✅ Suggestions : {job.get('message', 'terminé')}")
        if st.sidebar.button("OK", key="sugg_job_dismiss"):
            dismiss()
            st.rerun()
        return

    if status == "error":
        st.sidebar.error(f"Suggestions : {job.get('error', 'erreur')}")
        if st.sidebar.button("Fermer", key="sugg_job_err_dismiss"):
            dismiss()
            st.rerun()
        return

    if status in ("pending", "running"):
        st.sidebar.warning(f"⏳ {job.get('message', 'Suggestions en cours…')}")
        st.sidebar.caption(
            "Le calcul continue même si vous changez de page. "
            "Revenez sur *Suggestions d'actifs* pour voir le résultat."
        )
        if st.sidebar.button("Annuler la génération", key="sugg_job_cancel"):
            cancel()
            st.rerun()
