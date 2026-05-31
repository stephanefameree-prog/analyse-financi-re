import numpy as np
import pandas as pd
import streamlit as st

from analytics import compute_technical_indicators
from display_units import (
    ASSET_SUMMARY_FORMAT,
    ASSET_SUMMARY_LABELS,
    format_map_for_labeled_columns,
    pick_existing_columns,
    rename_columns_for_display,
)
from fundamentals import compute_market_indicators, fetch_fundamentals_for_ticker


def _score_from_votes(votes):
    if not votes:
        return None
    return float(np.mean(votes))


def _label_from_score(score):
    if score is None or pd.isna(score):
        return "N/A"
    if score >= 0.45:
        return "Bon marché"
    if score >= 0.15:
        return "Plutôt bon marché"
    if score <= -0.45:
        return "Cher"
    if score <= -0.15:
        return "Plutôt cher"
    return "Neutre"


def _price_stats_from_series(price_series, last_price, ticker=None):
    """Compatibilité watchlist / synthèse — délègue aux indicateurs marché."""
    market = compute_market_indicators(ticker or "", price_series, last_price)
    return {
        k: market[k]
        for k in (
            "Min 52 sem.",
            "Max 52 sem.",
            "Position 52 sem.",
            "Variation 1 an",
            "Sommet 52 sem.",
            "Creux 52 sem.",
        )
        if market.get(k) is not None
    }


def score_fundamental_valuation(row):
    votes = []

    per = row.get("PER")
    if per is not None and not pd.isna(per) and per > 0:
        votes.append(1 if per < 12 else (-1 if per > 22 else 0))

    per_fwd = row.get("PER forward")
    if per_fwd is not None and not pd.isna(per_fwd) and per_fwd > 0:
        votes.append(1 if per_fwd < 12 else (-1 if per_fwd > 20 else 0))

    ev_ebitda = row.get("VE/EBITDA")
    if ev_ebitda is not None and not pd.isna(ev_ebitda) and ev_ebitda > 0:
        votes.append(1 if ev_ebitda < 8 else (-1 if ev_ebitda > 15 else 0))

    fcf_yield = row.get("Rendement FCF")
    if fcf_yield is not None and not pd.isna(fcf_yield):
        votes.append(1 if fcf_yield >= 0.06 else (-1 if fcf_yield <= 0.02 else 0))

    upside = row.get("Upside vs objectif")
    if upside is not None and not pd.isna(upside):
        votes.append(1 if upside >= 0.15 else (-1 if upside <= -0.05 else 0))

    roic = row.get("ROIC")
    if roic is not None and not pd.isna(roic):
        votes.append(0.5 if roic >= 0.12 else (-0.5 if roic <= 0.05 else 0))

    pb = row.get("Cours / val. comptable")
    if pb is not None and not pd.isna(pb) and pb > 0:
        votes.append(1 if pb < 1.5 else (-1 if pb > 3.5 else 0))

    peg = row.get("Ratio PEG")
    if peg is not None and not pd.isna(peg) and peg > 0:
        votes.append(1 if peg < 1.0 else (-1 if peg > 2.0 else 0))

    pos52 = row.get("Position 52 sem.")
    if pos52 is not None and not pd.isna(pos52):
        votes.append(1 if pos52 <= 0.35 else (-1 if pos52 >= 0.85 else 0))

    debt_fcf = row.get("Dette / FCF")
    if debt_fcf is not None and not pd.isna(debt_fcf) and debt_fcf >= 0:
        votes.append(1 if debt_fcf < 3 else (-1 if debt_fcf > 8 else 0))

    score = _score_from_votes([v for v in votes if v != 0])
    return score, _label_from_score(score), len([v for v in votes if v != 0])


def score_technical_valuation(row, last_price=None):
    votes = []

    rsi = row.get("RSI")
    if rsi is not None and not pd.isna(rsi):
        votes.append(1 if rsi <= 35 else (-1 if rsi >= 65 else 0))

    pct_b = row.get("Bollinger %B")
    if pct_b is not None and not pd.isna(pct_b):
        votes.append(1 if pct_b <= 0.2 else (-1 if pct_b >= 0.8 else 0))

    stoch_k = row.get("Stoch %K")
    if stoch_k is not None and not pd.isna(stoch_k):
        votes.append(1 if stoch_k <= 25 else (-1 if stoch_k >= 75 else 0))

    sma50 = row.get("SMA 50")
    sma200 = row.get("SMA 200")
    price = last_price
    if price is not None and not pd.isna(price):
        if sma200 is not None and not pd.isna(sma200) and sma200 > 0:
            ratio = price / sma200
            votes.append(1 if ratio <= 0.92 else (-1 if ratio >= 1.08 else 0))
        if sma50 is not None and not pd.isna(sma50) and sma50 > 0:
            votes.append(0.5 if price <= sma50 * 0.95 else (-0.5 if price >= sma50 * 1.05 else 0))

    signal_mm = str(row.get("Signal MM", ""))
    if signal_mm == "Baissière":
        votes.append(-0.5)
    elif signal_mm == "Haussière":
        votes.append(0.5)

    score = _score_from_votes([v for v in votes if v != 0])
    return score, _label_from_score(score), len([v for v in votes if v != 0])


def build_synthesis_comment(funda_label, tech_label):
    if funda_label == "N/A" and tech_label == "N/A":
        return "Données insuffisantes pour conclure."
    if funda_label == "N/A":
        return f"Fondamental indisponible — lecture technique : {tech_label.lower()}."
    if tech_label == "N/A":
        return f"Technique indisponible — lecture fondamentale : {funda_label.lower()}."

    cheap_f = funda_label in ("Bon marché", "Plutôt bon marché")
    cheap_t = tech_label in ("Bon marché", "Plutôt bon marché")
    rich_f = funda_label in ("Cher", "Plutôt cher")
    rich_t = tech_label in ("Cher", "Plutôt cher")

    if cheap_f and cheap_t:
        return "Sous-évalué fondamentalement et en zone basse techniquement."
    if rich_f and rich_t:
        return "Surévalué sur les deux plans — prudence."
    if cheap_f and rich_t:
        return "Bon marché en fondamental, prix court terme élevé — patience possible."
    if rich_f and cheap_t:
        return "Cher en fondamental, correction technique en cours."
    if cheap_f:
        return "Attractif en fondamental ; timing technique neutre."
    if rich_f:
        return "Cher en fondamental ; timing technique neutre."
    if cheap_t:
        return "Fondamental neutre avec rebond technique possible."
    if rich_t:
        return "Fondamental neutre avec prix technique étiré."
    return "Ni clairement bon marché ni clairement cher."


def combine_valuation_verdict(funda_score, tech_score):
    if funda_score is None and tech_score is None:
        return None, "N/A"
    if funda_score is None:
        combined = tech_score
    elif tech_score is None:
        combined = funda_score
    else:
        combined = 0.6 * funda_score + 0.4 * tech_score
    return combined, _label_from_score(combined)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_fundamentals_for_summary(ticker, last_price):
    return fetch_fundamentals_for_ticker(ticker, last_price, throttle=False)


def build_asset_summary_table(prices, rsi_period=14):
    last_prices = prices.iloc[-1]
    tech_df = compute_technical_indicators(prices, rsi_period=rsi_period)
    tech_by_ticker = tech_df.set_index("Ticker") if not tech_df.empty else pd.DataFrame()

    rows = []
    for ticker in prices.columns:
        lp = float(last_prices[ticker]) if not np.isnan(last_prices[ticker]) else None
        funda_row = cached_fundamentals_for_summary(ticker, lp) or {}
        funda_row.update(_price_stats_from_series(prices[ticker], lp, ticker=ticker))

        tech_row = tech_by_ticker.loc[ticker].to_dict() if ticker in tech_by_ticker.index else {}

        funda_score, funda_label, funda_votes = score_fundamental_valuation(funda_row)
        tech_score, tech_label, tech_votes = score_technical_valuation(tech_row, last_price=lp)
        combined_score, combined_label = combine_valuation_verdict(funda_score, tech_score)

        rows.append(
            {
                "Ticker": ticker,
                "Prix": lp,
                "Min 52 sem.": funda_row.get("Min 52 sem."),
                "Max 52 sem.": funda_row.get("Max 52 sem."),
                "Verdict fondamental": funda_label,
                "Verdict technique": tech_label,
                "Synthèse globale": combined_label,
                "Score global": combined_score,
                "Commentaire": build_synthesis_comment(funda_label, tech_label),
                "PER": funda_row.get("PER"),
                "Upside vs objectif": funda_row.get("Upside vs objectif"),
                "Position 52 sem.": funda_row.get("Position 52 sem."),
                "RSI": tech_row.get("RSI"),
                "Bollinger %B": tech_row.get("Bollinger %B"),
                "Indicateurs fonda utilisés": funda_votes,
                "Indicateurs tech utilisés": tech_votes,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty and "Score global" in df.columns:
        df = df.sort_values("Score global", ascending=False, na_position="last")
    return df


def render_asset_summary_dashboard(prices):
    st.subheader("Synthèse par actif : bon marché ou cher ?")
    st.caption(
        "Combine valorisation fondamentale (PER, upside analystes, dette/FCF…) "
        "et signaux techniques (RSI, Bollinger, stochastique, moyennes mobiles)."
    )

    rsi_period = st.slider("Période RSI", 7, 21, 14, key="summary_rsi_period")
    run = st.button("Générer la synthèse", type="primary", key="summary_run")

    cache_key = f"asset_summary_{abs(hash('|'.join(prices.columns)))}_{rsi_period}"
    if run:
        with st.spinner("Analyse fondamentale et technique en cours..."):
            st.session_state[cache_key] = build_asset_summary_table(prices, rsi_period=rsi_period)

    df = st.session_state.get(cache_key)
    if df is None or df.empty:
        st.info("Cliquez sur **Générer la synthèse** pour analyser vos actifs.")
        return

    cheap = (df["Synthèse globale"].isin(["Bon marché", "Plutôt bon marché"])).sum()
    rich = (df["Synthèse globale"].isin(["Cher", "Plutôt cher"])).sum()
    neutral = len(df) - cheap - rich
    c1, c2, c3 = st.columns(3)
    c1.metric("Bon marché / plutôt bon marché", int(cheap))
    c2.metric("Neutre", int(neutral))
    c3.metric("Cher / plutôt cher", int(rich))

    df_display = rename_columns_for_display(df, ASSET_SUMMARY_LABELS)
    summary_format = format_map_for_labeled_columns(
        df_display, ASSET_SUMMARY_LABELS, ASSET_SUMMARY_FORMAT
    )

    st.caption(
        "Montants **/ action** · pourcentages en **%** · PER en **×** · "
        "position 52 sem. = % de la fourchette min–max."
    )
    st.dataframe(
        df_display.style.format(summary_format, na_rep="-")
        .background_gradient(
            cmap="RdYlGn",
            subset=pick_existing_columns(df_display, "Score global (pts)", "Score global"),
        )
    )

    detail = st.selectbox("Détail par actif", df["Ticker"], key="summary_detail")
    row = df[df["Ticker"] == detail].iloc[0]
    st.markdown(f"**{detail}** — {row['Synthèse globale']}")
    st.write(row["Commentaire"])
    d1, d2 = st.columns(2)
    d1.write(f"**Fondamental :** {row['Verdict fondamental']}")
    d2.write(f"**Technique :** {row['Verdict technique']}")

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Télécharger la synthèse (CSV)", csv, "synthese_bon_marche_cher.csv")
