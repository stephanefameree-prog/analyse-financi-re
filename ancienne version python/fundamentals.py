import random
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from yahooquery import Ticker


def _safe_float(value):
    if value is None:
        return None
    try:
        if isinstance(value, (float, int, np.floating, np.integer)):
            if np.isnan(value) or np.isinf(value):
                return None
            return float(value)
    except Exception:
        pass
    return None


def _find_row_value(df, patterns):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    for pattern in patterns:
        p = pattern.lower()
        for idx in df.index:
            if p in str(idx).lower():
                row = df.loc[idx]
                if isinstance(row, pd.Series):
                    for val in row.sort_index(ascending=False):
                        v = _safe_float(val)
                        if v is not None:
                            return v
                else:
                    v = _safe_float(row)
                    if v is not None:
                        return v
    return None


def get_fundamentals_yfinance(ticker):
    data = {}
    try:
        info = yf.Ticker(ticker).info or {}
        data["per"] = _safe_float(info.get("trailingPE"))
        data["per_forward"] = _safe_float(info.get("forwardPE"))
        data["debt_to_equity"] = _safe_float(info.get("debtToEquity"))
        data["target_mean"] = _safe_float(info.get("targetMeanPrice"))
        data["target_low"] = _safe_float(info.get("targetLowPrice"))
        data["target_high"] = _safe_float(info.get("targetHighPrice"))
        data["recommendation"] = info.get("recommendationKey")
        data["analyst_count"] = _safe_float(info.get("numberOfAnalystOpinions"))

        total_debt = _safe_float(info.get("totalDebt"))
        fcf = _safe_float(info.get("freeCashflow"))
        if total_debt is not None and fcf not in (None, 0):
            data["debt_fcf"] = total_debt / abs(fcf)
        else:
            bs = yf.Ticker(ticker).balance_sheet
            cf = yf.Ticker(ticker).cashflow
            if bs is not None and not bs.empty:
                total_debt = _find_row_value(
                    bs, ["total debt", "long term debt", "total liabilities"]
                )
            if cf is not None and not cf.empty:
                fcf = _find_row_value(
                    cf,
                    [
                        "free cash flow",
                        "free cashflow",
                        "operating cash flow",
                    ],
                )
            if total_debt is not None and fcf not in (None, 0):
                data["debt_fcf"] = total_debt / abs(fcf)
    except Exception:
        pass
    return data


def get_fundamentals_yahooquery(ticker):
    data = {}
    try:
        t = Ticker(ticker, timeout=20)
        ks = t.key_stats
        if isinstance(ks, dict) and ticker in ks:
            ks = ks[ticker]
        if isinstance(ks, dict):
            data["per"] = _safe_float(ks.get("trailingPE"))
            data["per_forward"] = _safe_float(ks.get("forwardPE"))
            data["debt_to_equity"] = _safe_float(ks.get("debtToEquity"))

        fd = t.financial_data
        if isinstance(fd, dict) and ticker in fd:
            fd = fd[ticker]
        if isinstance(fd, dict):
            data["target_mean"] = _safe_float(fd.get("targetMeanPrice"))
            data["target_low"] = _safe_float(fd.get("targetLowPrice"))
            data["target_high"] = _safe_float(fd.get("targetHighPrice"))
            data["recommendation"] = fd.get("recommendationKey")
            data["analyst_count"] = _safe_float(fd.get("numberOfAnalystOpinions"))

        inc = t.income_statement(frequency="a")
        bs = t.balance_sheet(frequency="a")
        cf = t.cash_flow(frequency="a")
        if isinstance(bs, pd.DataFrame) and not bs.empty:
            bs = bs.sort_index()
            total_debt = _find_row_value(
                bs, ["total debt", "long term debt", "total liabilities"]
            )
        else:
            total_debt = None
        if isinstance(cf, pd.DataFrame) and not cf.empty:
            cf = cf.sort_index()
            fcf = _find_row_value(
                cf, ["free cash flow", "free cashflow", "operating cash flow"]
            )
        else:
            fcf = None
        if total_debt is not None and fcf not in (None, 0):
            data["debt_fcf"] = total_debt / abs(fcf)
    except Exception:
        pass
    return data


def merge_fundamentals(primary, secondary):
    merged = {}
    for src in (primary, secondary):
        for key, val in src.items():
            if key not in merged or merged[key] is None:
                if val is not None:
                    merged[key] = val
    return merged


def fetch_fundamentals_for_ticker(ticker, last_price):
    time.sleep(random.uniform(0.2, 0.5))
    yf_data = get_fundamentals_yfinance(ticker)
    yq_data = get_fundamentals_yahooquery(ticker)
    data = merge_fundamentals(yf_data, yq_data)

    if not data:
        return None

    dte = data.get("debt_to_equity")
    if dte is not None and dte > 10:
        dte = dte / 100

    target = data.get("target_mean")
    upside = None
    if target is not None and last_price not in (None, 0):
        upside = (target - last_price) / last_price

    return {
        "Ticker": ticker,
        "Prix": last_price,
        "PER": data.get("per"),
        "PER forward": data.get("per_forward"),
        "Dette / FCF": data.get("debt_fcf"),
        "Dette / Capitaux": dte,
        "Objectif analystes": target,
        "Objectif bas": data.get("target_low"),
        "Objectif haut": data.get("target_high"),
        "Upside vs objectif": upside,
        "Recommandation": data.get("recommendation"),
        "Nb analystes": data.get("analyst_count"),
    }


def render_fundamentals_dashboard(prices):
    st.subheader("Analyse fondamentale & consensus analystes")
    tickers = [str(t).strip() for t in prices.columns]
    last_prices = prices.iloc[-1]
    signature = "|".join(tickers)
    cache_key = f"data_funda_{abs(hash(signature))}"

    if cache_key not in st.session_state:
        st.session_state[cache_key] = []

    max_default = min(30, len(tickers))
    max_to_analyze = st.slider(
        "Nombre max de tickers a analyser",
        min_value=5 if len(tickers) >= 5 else len(tickers),
        max_value=len(tickers),
        value=max_default,
        step=5 if len(tickers) >= 10 else 1,
    )
    batch_size = st.selectbox("Taille de lot", [5, 10, 15, 20], index=1)
    limited = tickers[:max_to_analyze]
    total_batches = max(1, int(np.ceil(len(limited) / batch_size)))

    batch_key = f"{cache_key}_batch"
    if batch_key not in st.session_state:
        st.session_state[batch_key] = 0
    st.session_state[batch_key] = min(st.session_state[batch_key], total_batches - 1)
    batch_idx = st.session_state[batch_key]

    start_i = batch_idx * batch_size
    end_i = min(start_i + batch_size, len(limited))
    batch = limited[start_i:end_i]
    st.caption(
        f"Lot {batch_idx + 1}/{total_batches} - tickers {start_i + 1} a {end_i} "
        f"sur {len(limited)}."
    )

    c1, c2, c3 = st.columns(3)
    analyze = c1.button("Analyser ce lot", key="funda_analyze")
    next_lot = c2.button("Lot suivant", key="funda_next")
    reset = c3.button("Recommencer", key="funda_reset")

    if next_lot and batch_idx < total_batches - 1:
        st.session_state[batch_key] = batch_idx + 1
        st.rerun()
    if reset:
        if cache_key in st.session_state:
            del st.session_state[cache_key]
        if batch_key in st.session_state:
            del st.session_state[batch_key]
        st.rerun()

    if analyze and batch:
        rows = st.session_state[cache_key]
        known = {r["Ticker"] for r in rows}
        status = st.empty()
        progress = st.progress(0)
        ok, fail = 0, 0

        for i, t in enumerate(batch):
            if t in known:
                progress.progress((i + 1) / len(batch))
                continue
            status.text(f"Collecte fondamentale : {t} ({i + 1}/{len(batch)})")
            lp = float(last_prices[t]) if not np.isnan(last_prices[t]) else None
            row = fetch_fundamentals_for_ticker(t, lp)
            if row is not None:
                rows.append(row)
                ok += 1
            else:
                fail += 1
            progress.progress((i + 1) / len(batch))

        st.session_state[cache_key] = rows
        status.empty()
        progress.empty()
        st.info(f"Lot termine - succes: {ok}, echecs: {fail}")

    data = st.session_state[cache_key]
    if not data:
        st.warning("Aucune donnee fondamentale pour l'instant. Lancez l'analyse d'un lot.")
        return

    df = pd.DataFrame(data)
    sort_col = st.selectbox(
        "Trier par",
        ["PER", "Dette / FCF", "Upside vs objectif", "Objectif analystes"],
        index=2,
    )
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=True, na_position="last")

    st.dataframe(
        df.style.format(
            {
                "Prix": "{:.2f}",
                "PER": "{:.2f}",
                "PER forward": "{:.2f}",
                "Dette / FCF": "{:.2f}",
                "Dette / Capitaux": "{:.2f}",
                "Objectif analystes": "{:.2f}",
                "Objectif bas": "{:.2f}",
                "Objectif haut": "{:.2f}",
                "Upside vs objectif": "{:.2%}",
                "Nb analystes": "{:.0f}",
            },
            na_rep="-",
        ).background_gradient(cmap="RdYlGn", subset=["Upside vs objectif"])
    )

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Telecharger (CSV)", csv, "analyse_fondamentale.csv")

    st.markdown("---")
    detail = st.selectbox("Detail par actif", df["Ticker"])
    row = df[df["Ticker"] == detail].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PER", "-" if pd.isna(row["PER"]) else f"{row['PER']:.2f}")
    c2.metric("Dette / FCF", "-" if pd.isna(row["Dette / FCF"]) else f"{row['Dette / FCF']:.2f}")
    c3.metric(
        "Objectif",
        "-" if pd.isna(row["Objectif analystes"]) else f"{row['Objectif analystes']:.2f}",
    )
    c4.metric(
        "Upside",
        "-"
        if pd.isna(row["Upside vs objectif"])
        else f"{row['Upside vs objectif']:.2%}",
    )
    st.write(f"**Recommandation consensus :** {row.get('Recommandation', '-')}")

    if not pd.isna(row["Objectif bas"]) and not pd.isna(row["Objectif haut"]):
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=["Bas", "Moyen", "Haut"],
                y=[row["Objectif bas"], row["Objectif analystes"], row["Objectif haut"]],
                marker_color=["#d62728", "#1f77b4", "#2ca02c"],
            )
        )
        if not pd.isna(row["Prix"]):
            fig.add_hline(y=row["Prix"], line_dash="dash", annotation_text="Prix actuel")
        fig.update_layout(title=f"Fourchette d'objectifs analystes - {detail}")
        st.plotly_chart(fig, use_container_width=True)
