# ==========================================
# BLOC 1/4 : Imports & Configuration Initiale
# ==========================================
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from analytics import (
    build_portfolio_value,
    build_portfolio_price_for_fft,
    build_mean_median_mode_pedagogy_chart,
    build_fft_spectral_acf_chart,
    interpret_fft_acf_reading,
    build_fft_cyclic_chart,
    FFT_TREND_OPTIONS,
    FFT_TREND_LOG_LINEAR,
    fft_trend_mode_label,
    build_risk_returns_boxplot,
    build_risk_metrics_boxplot,
    compute_bollinger,
    compute_fibonacci_levels,
    compute_linear_regression_channel,
    compute_macd,
    build_markowitz_assets_figure,
    compute_holdings_weights,
    portfolio_market_value_eur,
    build_markowitz_arbitrage_table,
    portfolio_performance_from_weights,
    _align_weights_to_returns,
    compute_mfi,
    compute_obv,
    compute_portfolio_risk_metrics,
    compute_avg_internal_correlation,
    describe_portfolio_profile,
    compute_returns,
    compute_rsi,
    compute_sma,
    compute_stochastic,
    compute_support_resistance,
    compute_pivot_level,
    compute_volume_sma,
    interpret_fibonacci_comment,
    interpret_macd_comment,
    interpret_mfi_comment,
    interpret_obv_comment,
    interpret_regression_channel_comment,
    interpret_rsi_comment,
    interpret_stochastic_comment,
    interpret_support_resistance_comment,
    interpret_volume_comment,
    describe_suggestion_profile,
    compute_profile_scores,
    DEFAULT_SUGGESTION_OBJECTIVE_WEIGHTS,
    filter_suggestions_by_statistics,
    enrich_suggestions_with_technical,
    filter_suggestions_by_technical,
    style_suggestions_technical,
    SUGGESTION_TECHNICAL_COLUMNS,
)

st.set_page_config(layout="wide", page_title="Financial Dashboard V6.3")

from data_loader import (
    add_company_names,
    candlestick_trace,
    clear_market_cache,
    get_ohlc_for_ticker,
    get_ticker_metadata,
    label_index_with_names,
    load_ohlcv_in_batches,
    load_prices_in_batches,
    ticker_label,
    _tickers_signature,
)
from dashboard_cache import (
    cached_usd_to_eur_rate,
    cached_technical_indicators,
    cached_advanced_risk_metrics,
    cached_corr_matrix,
    cached_markowitz_weights,
    cached_markowitz_frontier_figure,
    cached_fft_periodicity,
    cached_fft_summary,
    cached_suggest_portfolio_additions,
    cached_benchmark_prices,
    cached_candidate_market_data,
    cached_technical_detail_bundle,
    clear_dashboard_compute_cache,
)
from display_units import (
    FFT_FORMAT,
    FFT_LABELS,
    RISK_METRICS_FORMAT,
    SUGGESTIONS_FORMAT,
    SUGGESTIONS_LABELS,
    TECHNICAL_FORMAT,
    TECHNICAL_LABELS,
    format_map_for_labeled_columns,
    pick_existing_columns,
    rename_columns_for_display,
)

try:
    import dividendes
    HAS_DIVIDENDES = True
except Exception:
    HAS_DIVIDENDES = False

try:
    import fundamentals
    HAS_FUNDAMENTALS = True
except Exception:
    HAS_FUNDAMENTALS = False

try:
    import asset_summary
    HAS_ASSET_SUMMARY = True
except Exception:
    HAS_ASSET_SUMMARY = False

try:
    import watchlists
    HAS_WATCHLISTS = True
except Exception:
    HAS_WATCHLISTS = False

FILE_PATH = "mon_portefeuille.csv"
JSON_PATH = "tickers.json"

EUR_SUFFIXES = (".PA", ".DE", ".BR", ".AS", ".MI", ".MC", ".SW", ".L", ".IR", ".HE")
USD_TICKERS = {"HMY", "XPL", "NUTX", "NYXH", "AAPL", "MSFT"}

BENCHMARKS = {
    "CAC 40": "^FCHI",
    "BEL 20": "^BFX",
    "NASDAQ 100": "^NDX",
    "S&P 500": "^GSPC",
    "Euro Stoxx 50": "^STOXX50E",
    "DAX 40": "^GDAXI",
}

VUES_AVEC_SECTEURS = frozenset(
    {
        "Ma watchlist",
        "Synthèse & Plus-values",
        "Optimisation (Markowitz)",
        "Analyse technique (RSI / MACD)",
        "Analyse de Fourier (FFT)",
        "Suggestions d'actifs",
    }
)


def is_usd_ticker(ticker):
    tk = str(ticker).upper()
    if any(tk.endswith(suffix) for suffix in EUR_SUFFIXES):
        return False
    if tk.endswith(".US"):
        return True
    if tk in USD_TICKERS:
        return True
    if "." not in tk:
        return True
    return False


def _render_fft_headline_metrics(result, peaks):
    """Bandeau : cycle dominant, force du cycle et R² de la tendance retirée."""
    c1, c2, c3, c4 = st.columns(4)
    if peaks is not None and len(peaks):
        c1.metric(
            "Cycle principal",
            peaks.iloc[0]["Période"],
            help="Durée estimée du cycle le plus marqué (jours de bourse).",
        )
        c2.metric(
            "Force du cycle",
            f"{peaks.iloc[0]['Puissance relative']:.1%}",
            help=(
                "Part de l'énergie FFT du cycle dominant dans la bande analysée "
                "(même valeur que « Puissance relative » du tableau)."
            ),
        )
        if len(peaks) > 1:
            c3.metric("2e cycle", peaks.iloc[1]["Période"])
        else:
            c3.metric("2e cycle", "—")
    else:
        c1.metric("Cycle principal", "—")
        c2.metric("Force du cycle", "—")
        c3.metric("2e cycle", "—")

    r2 = result.get("trend_r2") if result else None
    rmse = result.get("trend_rmse") if result else None
    if r2 is not None and not (isinstance(r2, float) and np.isnan(r2)):
        r2_help = (
            "Qualité de la tendance retirée avant FFT, mesurée sur le prix réel (€). "
            "1 = parfait · > 0,85 = bon · < 0,5 = tendance mal isolée."
        )
        if rmse is not None and not np.isnan(rmse):
            r2_help += f" RMSE = {rmse:,.2f} (écart moyen prix vs tendance)."
        c4.metric("R² tendance", f"{r2:.3f}", help=r2_help)
    else:
        c4.metric("R² tendance", "—", help="Non calculable sur cette série.")


def _objective_weight_slider(label, low_hint, high_hint, default, key, tech_name=None):
    """Curseur 0–2 : explication simple + nom du paramètre statistique."""
    col_l, col_m, col_r = st.columns([2.5, 5, 2.5])
    with col_l:
        st.caption("**Peu important (0)**")
        st.caption(low_hint)
    with col_m:
        value = st.slider(
            label,
            min_value=0.0,
            max_value=2.0,
            value=default,
            step=0.1,
            key=key,
        )
        if tech_name:
            st.caption(f"Paramètre : **{tech_name}**")
    with col_r:
        st.caption("**Très important (2)**")
        st.caption(high_hint)
    return value


# ==========================================
# BLOC 2/4 : Gestion des Sources (CSV & JSON)
# ==========================================
st.sidebar.title("📈 Financial Dashboard V6.3")

mode_selection = st.sidebar.radio(
    "Source des données :",
    [
        "💼 Mon Portefeuille Réel (CSV)",
        "👁 Mes Watchlists",
        "📊 Analyser un Indice (JSON)",
    ],
)

start_date = st.sidebar.date_input(
    "Date de début des historiques", pd.to_datetime("2020-01-01")
)
if mode_selection == "👁 Mes Watchlists":
    st.sidebar.caption(
        "Pour le min/max sur 52 semaines, un historique d'au moins 1 an est recommandé."
    )
show_company_names = st.sidebar.checkbox(
    "Afficher les noms d'entreprises (Yahoo)",
    value=True,
    help="Ajoute le nom de la société à côté du symbole dans les tableaux et listes.",
)

portefeuille_dict = {}
liste_tickers = []
mode_portfolio_actif = False
mode_watchlist_actif = False
indice_choisi = None
active_watchlist_name = None
INDEX_TICKERS = {}

if mode_selection == "💼 Mon Portefeuille Réel (CSV)":
    mode_portfolio_actif = True
    if os.path.exists(FILE_PATH) and os.path.getsize(FILE_PATH) > 0:
        try:
            df_portefeuille = pd.read_csv(FILE_PATH)
            df_portefeuille.columns = df_portefeuille.columns.str.strip()
            portefeuille_dict = df_portefeuille.set_index("Ticker").to_dict(orient="index")
            liste_tickers = list(portefeuille_dict.keys())
        except Exception:
            portefeuille_dict = {}
            liste_tickers = []

    st.sidebar.subheader("🛒 Gestion du Portefeuille")
    with st.sidebar.expander("➕ Ajouter / Modifier une ligne", expanded=False):
        add_ticker = st.text_input(
            "Ticker Yahoo (ex: AIR.PA, AAPL, HMY)", key="add_tk"
        ).upper().strip()
        add_quantite = st.number_input(
            "Nombre d'actions", min_value=0.0, step=1.0, value=1.0
        )
        add_pru = st.number_input(
            "PRU (Dans la devise d'origine)", min_value=0.0, step=0.1, value=10.0
        )

        if st.button("Enregistrer la position"):
            if add_ticker:
                portefeuille_dict[add_ticker] = {
                    "Quantite": add_quantite,
                    "PRU": add_pru,
                }
                df_save = pd.DataFrame.from_dict(
                    portefeuille_dict, orient="index"
                ).reset_index()
                df_save.columns = ["Ticker", "Quantite", "PRU"]
                df_save.to_csv(FILE_PATH, index=False)
                st.success(f"{add_ticker} enregistré !")
                st.rerun()
            else:
                st.error("Saisissez un Ticker valide.")

    if not portefeuille_dict:
        st.sidebar.info("Votre portefeuille CSV est vide. Ajoutez un ticker ci-dessus.")

elif mode_selection == "👁 Mes Watchlists":
    mode_watchlist_actif = True
    if os.path.exists(JSON_PATH):
        try:
            with open(JSON_PATH, "r", encoding="utf-8") as f:
                INDEX_TICKERS = json.load(f)
        except Exception:
            INDEX_TICKERS = {}
    if HAS_WATCHLISTS:
        liste_tickers, active_watchlist_name = watchlists.render_watchlist_sidebar_controls(
            index_tickers=INDEX_TICKERS or None
        )
    else:
        st.sidebar.error("Le module watchlists.py n'a pas pu être chargé.")
        liste_tickers = []

else:
    if os.path.exists(JSON_PATH):
        try:
            with open(JSON_PATH, "r", encoding="utf-8") as f:
                INDEX_TICKERS = json.load(f)
            cles_disponibles = list(INDEX_TICKERS.keys())
            indice_choisi = st.sidebar.selectbox(
                "Sélectionnez l'indice à analyser :", cles_disponibles
            )
            liste_tickers = INDEX_TICKERS[indice_choisi]
            st.sidebar.success(
                f"{len(liste_tickers)} tickers chargés depuis {indice_choisi}."
            )
        except Exception as e:
            st.sidebar.error(f"Erreur de lecture du fichier JSON : {e}")
            liste_tickers = []
    else:
        st.sidebar.error("Fichier tickers.json introuvable dans le dossier.")
        liste_tickers = []

# ==========================================
# Sélection de vue (avant métadonnées — secteurs chargés selon la vue)
# ==========================================
vue_options = [
    "Synthèse & Plus-values",
    "Matrice de corrélation",
    "Analyse des risques",
    "Optimisation (Markowitz)",
    "Dividendes & Qualité",
    "Analyse fondamentale",
    "Analyse technique (RSI / MACD)",
    "Analyse de Fourier (FFT)",
    "Synthèse bon marché / cher",
    "Suggestions d'actifs",
]
if mode_watchlist_actif:
    vue_options = ["Ma watchlist"] + vue_options

vue = st.sidebar.radio(
    "Sélectionnez une analyse :",
    vue_options,
)

# ==========================================
# Noms d'entreprises et secteurs Yahoo (hors fragment — cache 24 h)
# ==========================================
ticker_names = {}
ticker_sectors = {}
need_ticker_sectors = bool(liste_tickers) and (
    (mode_portfolio_actif and portefeuille_dict)
    or vue in VUES_AVEC_SECTEURS
)
if liste_tickers:
    uniq_tickers = sorted(set(liste_tickers))
    need_names = show_company_names
    if need_names or need_ticker_sectors:
        spinner_label = "Récupération des métadonnées Yahoo..."
        if need_names and not need_ticker_sectors:
            spinner_label = "Récupération des noms d'entreprises..."
        with st.spinner(spinner_label):
            ticker_names, ticker_sectors = get_ticker_metadata(
                uniq_tickers,
                need_names=need_names,
                need_sectors=need_ticker_sectors,
            )

if mode_portfolio_actif and portefeuille_dict:
    st.sidebar.write("### Vos Positions :")
    df_visu = pd.DataFrame.from_dict(portefeuille_dict, orient="index").reset_index()
    df_visu.columns = ["Ticker", "Quantite", "PRU"]
    df_visu = add_company_names(
        df_visu,
        ticker_names,
        show_names=show_company_names,
        sectors=ticker_sectors if need_ticker_sectors else None,
    )
    st.sidebar.dataframe(df_visu, hide_index=True)

    ticker_to_delete = st.sidebar.selectbox(
        "Supprimer une ligne :",
        ["-"] + list(portefeuille_dict.keys()),
        format_func=lambda t: "-"
        if t == "-"
        else ticker_label(t, ticker_names, show_company_names),
    )
    if ticker_to_delete != "-":
        if st.sidebar.button("❌ Confirmer la suppression"):
            del portefeuille_dict[ticker_to_delete]
            df_save = pd.DataFrame.from_dict(
                portefeuille_dict, orient="index"
            ).reset_index()
            if not df_save.empty:
                df_save.columns = ["Ticker", "Quantite", "PRU"]
                df_save.to_csv(FILE_PATH, index=False)
            elif os.path.exists(FILE_PATH):
                os.remove(FILE_PATH)
            st.sidebar.success(f"{ticker_to_delete} supprimé.")
            st.rerun()

if mode_watchlist_actif and HAS_WATCHLISTS:
    watchlists.render_watchlist_sidebar_table(
        liste_tickers,
        active_watchlist_name or "Watchlist",
        show_company_names,
        wl_names=ticker_names if show_company_names else {},
        wl_sectors=ticker_sectors if need_ticker_sectors else None,
    )

if liste_tickers and (show_company_names and ticker_names or ticker_sectors):
    with st.sidebar.expander("📋 Symboles Yahoo → noms d'entreprises", expanded=False):
        ref_df = pd.DataFrame(
            [
                {
                    "Ticker": t,
                    "Nom": ticker_names.get(t, t) if show_company_names else t,
                    "Secteur": ticker_sectors.get(t, "—"),
                }
                for t in liste_tickers
            ]
        )
        st.dataframe(ref_df, hide_index=True, use_container_width=True)

# ==========================================
# Rafraîchissement auto (sidebar stable)
# ==========================================
auto_refresh_enabled = st.sidebar.checkbox(
    "🔄 Rafraîchissement auto des cours",
    value=False,
    help="Met à jour uniquement les derniers jours de cotation (rapide). Désactivé par défaut.",
)
refresh_interval = st.sidebar.slider(
    "Intervalle (secondes)",
    min_value=60,
    max_value=180,
    value=90,
    step=15,
    disabled=not auto_refresh_enabled,
)
if st.sidebar.button("Vider le cache des cours", help="Force un rechargement complet au prochain affichage"):
    if liste_tickers:
        clear_market_cache(_tickers_signature(liste_tickers, start_date))
    else:
        clear_market_cache()
    clear_dashboard_compute_cache()
    st.session_state.pop("ohlcv_sig", None)
    st.sidebar.success("Cache cours vidé.")
run_every = refresh_interval if auto_refresh_enabled and liste_tickers else None


@st.fragment(run_every=run_every)
def render_live_dashboard():
    """Fragment dynamique : cours + vues (refresh auto optionnel, cache disque)."""

    # --- Chargement des cours (fragment) ---
    prices = pd.DataFrame()
    volumes = pd.DataFrame()
    highs = pd.DataFrame()
    lows = pd.DataFrame()
    opens = pd.DataFrame()
    ohlc_closes = pd.DataFrame()
    returns = pd.DataFrame()
    missing_tickers = []
    ohlcv_sig = _tickers_signature(liste_tickers, start_date) if liste_tickers else ""

    status_col, refresh_col = st.columns([5, 1])
    refresh_clicked = False
    with refresh_col:
        refresh_clicked = st.button("↻", help="Actualiser les cours (derniers jours)", key="refresh_prices_now")
    with status_col:
        if auto_refresh_enabled and liste_tickers:
            st.caption(f"🔄 Rafraîchissement auto actif ({refresh_interval} s, mise à jour légère)")

    if liste_tickers:
        try:
            ohlcv_sig = _tickers_signature(liste_tickers, start_date)
            if st.session_state.get("ohlcv_sig") != ohlcv_sig:
                refresh_mode = "full"
                st.session_state.ohlcv_sig = ohlcv_sig
            elif refresh_clicked:
                refresh_mode = "incremental"
            elif auto_refresh_enabled:
                refresh_mode = "incremental"
            else:
                refresh_mode = "cache_only"

            prices, volumes, highs, lows, opens, ohlc_closes = load_ohlcv_in_batches(
                liste_tickers, start=start_date, refresh=refresh_mode
            )
            if prices.empty and refresh_mode == "cache_only":
                prices, volumes, highs, lows, opens, ohlc_closes = load_ohlcv_in_batches(
                    liste_tickers, start=start_date, refresh="full"
                )
            if not prices.empty:
                returns = compute_returns(prices)
                missing_tickers = [t for t in liste_tickers if t not in prices.columns]
                if missing_tickers:
                    st.warning(
                        "Tickers non chargés depuis Yahoo Finance : "
                        + ", ".join(missing_tickers)
                    )
                st.caption(f"🟢 Dernière mise à jour des cours : **{datetime.now():%H:%M:%S}**")
        except Exception as e:
            st.error(f"Erreur lors de la récupération des prix : {e}")
    else:
        st.warning("Aucun ticker sélectionné ou disponible.")

    if mode_watchlist_actif and vue == "Ma watchlist":
        if not liste_tickers:
            st.header("👁 Mes Watchlists")
            st.info("Créez une watchlist et ajoutez des tickers dans la barre latérale.")
        elif prices.empty:
            st.warning("Impossible de charger les cours pour les tickers de la watchlist.")
        elif HAS_WATCHLISTS:
            watchlists.render_watchlist_dashboard(
                prices,
                ticker_names,
                show_company_names,
                active_watchlist_name or "Watchlist",
                ticker_sectors=ticker_sectors,
            )
        else:
            st.error("Module watchlists indisponible.")

    elif not prices.empty and not returns.empty:

        if vue == "Synthèse & Plus-values":
            if mode_portfolio_actif:
                st.header("💼 Valorisation & Performance du Portefeuille (Converti en EUR)")

                with st.spinner("Mise à jour du taux de change EUR/USD..."):
                    usd_to_eur = cached_usd_to_eur_rate()

                last_prices = prices.iloc[-1]
                synthese_rows = []
                total_valeur_actuelle = 0.0
                total_cout_achat = 0.0
                tickers_sans_cours = []
                quantities = {}

                for tk in liste_tickers:
                    if tk not in last_prices or pd.isna(last_prices[tk]):
                        tickers_sans_cours.append(tk)
                        continue

                    q = portefeuille_dict[tk]["Quantite"]
                    pru_devise = portefeuille_dict[tk]["PRU"]
                    current_p_devise = float(last_prices[tk])
                    quantities[tk] = q

                    if is_usd_ticker(tk):
                        taux = usd_to_eur
                        devise_origine = "$"
                    else:
                        taux = 1.0
                        devise_origine = "€"

                    pru_eur = pru_devise * taux
                    current_p_eur = current_p_devise * taux
                    cout_eur = q * pru_eur
                    valeur_eur = q * current_p_eur
                    pvh_eur = valeur_eur - cout_eur
                    pvh_pct = (pvh_eur / cout_eur) if cout_eur > 0 else 0.0

                    total_valeur_actuelle += valeur_eur
                    total_cout_achat += cout_eur

                    synthese_rows.append(
                        {
                            "Ticker": tk,
                            "Quantité": q,
                            "PRU (Origine)": f"{pru_devise:.2f} {devise_origine}",
                            "PRU (€)": pru_eur,
                            "Cours (€)": current_p_eur,
                            "Coût d'Achat (€)": cout_eur,
                            "Valeur Actuelle (€)": valeur_eur,
                            "Plus/Moins-Value (€)": pvh_eur,
                            "Perf (%)": pvh_pct,
                        }
                    )

                if tickers_sans_cours:
                    st.error(
                        "Positions sans cours disponible : "
                        + ", ".join(tickers_sans_cours)
                    )

                if synthese_rows:
                    df_synthese = pd.DataFrame(synthese_rows)
                    df_synthese = add_company_names(
                        df_synthese, ticker_names, show_names=show_company_names, sectors=ticker_sectors
                    )
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Valeur Totale (EUR)", f"{total_valeur_actuelle:,.2f} €")
                    c2.metric("Coût d'Achat Total (EUR)", f"{total_cout_achat:,.2f} €")
                    tot_pvh = total_valeur_actuelle - total_cout_achat
                    tot_pvh_pct = (
                        (tot_pvh / total_cout_achat) if total_cout_achat > 0 else 0.0
                    )
                    c3.metric(
                        "Plus-Value Globale (EUR)",
                        f"{tot_pvh:,.2f} €",
                        f"{tot_pvh_pct:.2%}",
                    )

                    st.write("---")
                    st.caption(
                        "Montants en **€** (conversion USD→EUR si applicable). **Perf (%)** = plus-value relative."
                    )
                    st.dataframe(
                        df_synthese.style.format(
                            {
                                "PRU (€)": "{:.2f} €",
                                "Cours (€)": "{:.2f} €",
                                "Coût d'Achat (€)": "{:.2f} €",
                                "Valeur Actuelle (€)": "{:.2f} €",
                                "Plus/Moins-Value (€)": "{:.2f} €",
                                "Perf (%)": "{:.2%}",
                            }
                        ).background_gradient(
                            cmap="RdYlGn", subset=["Plus/Moins-Value (€)"]
                        )
                    )

                    fig_pie = px.pie(
                        df_synthese,
                        values="Valeur Actuelle (€)",
                        names="Nom" if show_company_names and "Nom" in df_synthese.columns else "Ticker",
                        title="Répartition réelle du Capital (€)",
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                    port_value = build_portfolio_value(
                        prices, quantities, usd_to_eur=usd_to_eur, is_usd_fn=is_usd_ticker
                    )
                    port_risk = compute_portfolio_risk_metrics(port_value)
                    if port_risk is not None and len(port_value) >= 2:
                        rendement_periode = port_value.iloc[-1] / port_value.iloc[0] - 1
                        port_tickers_risk = [
                            t for t in liste_tickers if t in returns.columns and quantities.get(t, 0) > 0
                        ]
                        internal_corr = compute_avg_internal_correlation(
                            returns, port_tickers_risk
                        )

                        profile_title, profile_desc, pseudo_weights, profile_scores = (
                            describe_portfolio_profile(
                                port_risk,
                                internal_corr=internal_corr,
                            )
                        )

                        st.markdown("### Profil de votre portefeuille")
                        st.markdown(f"**{profile_title}**")
                        st.caption(profile_desc)
                        with st.expander(
                            "📖 Légende — comment lire le profil du portefeuille ?",
                            expanded=False,
                        ):
                            st.markdown(
                                """
Le profil est déduit de **vos métriques réelles** (rendement, volatilité, skewness,
kurtosis, corrélation entre vos lignes) — pas de vos curseurs dans « Suggestions d'actifs ».

| Dimension observée | Ce qu'elle indique |
|--------------------|-------------------|
| **Rendement annuel** | Performance moyenne journalière annualisée (× 252). |
| **Volatilité** | Ampleur des variations quotidiennes. Faible = portefeuille calme. |
| **Skewness** | Asymétrie des rendements. **> 0** = journées de forte hausse plus fréquentes. |
| **Kurtosis** | Risque d'événements extrêmes. **Élevé** = queues épaisses (grosses secousses). |
| **Corrélation interne** | Les titres montent-ils **ensemble** ? **Faible** = meilleure diversification. |

Les libellés (Prudent, Dynamique, Diversificateur…) sont les **mêmes** que dans
« Suggestions d'actifs », mais ici ils décrivent **ce que votre portefeuille a fait**
sur la période sélectionnée.
                                """
                            )
                        with st.expander("Détail du profil (scores & traduction)", expanded=False):
                            st.caption(
                                "Les métriques sont converties en pseudo-poids (0–2), puis scores "
                                "de profil — même moteur que les suggestions d'actifs."
                            )
                            c_m1, c_m2 = st.columns(2)
                            with c_m1:
                                st.markdown("**Métriques observées**")
                                metrics_view = pd.DataFrame(
                                    [
                                        {
                                            "Indicateur": "Rendement annuel (moy.)",
                                            "Valeur": f"{port_risk['Rendement Annuel (moyenne)']:.2%}",
                                        },
                                        {
                                            "Indicateur": "Volatilité",
                                            "Valeur": f"{port_risk['Volatilité (Sigma)']:.2%}",
                                        },
                                        {
                                            "Indicateur": "Skewness",
                                            "Valeur": f"{port_risk['Skewness (Asymétrie)']:.2f}",
                                        },
                                        {
                                            "Indicateur": "Kurtosis",
                                            "Valeur": f"{port_risk['Kurtosis (Aplatissement)']:.2f}",
                                        },
                                        {
                                            "Indicateur": "Corr. interne moy.",
                                            "Valeur": (
                                                f"{internal_corr:.2f}"
                                                if not np.isnan(internal_corr)
                                                else "—"
                                            ),
                                        },
                                    ]
                                )
                                st.dataframe(metrics_view, hide_index=True, use_container_width=True)
                            with c_m2:
                                st.markdown("**Pseudo-poids dérivés (0–2)**")
                                st.write(
                                    pd.Series(
                                        {
                                            "rendement ↑": pseudo_weights["return"],
                                            "volatilité ↓": pseudo_weights["vol"],
                                            "kurtosis ↓": pseudo_weights["kurt"],
                                            "skewness ↑": pseudo_weights["skew"],
                                            "corrélation interne ↓": pseudo_weights[
                                                "corr_internal"
                                            ],
                                        }
                                    ).map(lambda x: f"{x:.2f}")
                                )
                            score_df = pd.DataFrame(
                                [
                                    {"Profil": name, "Score": score}
                                    for name, score in sorted(
                                        profile_scores.items(),
                                        key=lambda item: -item[1],
                                    )
                                ]
                            )
                            st.dataframe(
                                score_df.style.format({"Score": "{:.0%}"}),
                                hide_index=True,
                                use_container_width=True,
                            )
                            if " · " in profile_title:
                                st.caption(
                                    "Deux profils affichés car le portefeuille combine "
                                    "plusieurs caractéristiques marquées."
                                )

                        st.markdown("### Métriques de risque du portefeuille")
                        st.caption(
                            f"Calculées sur la période sélectionnée ({start_date} → aujourd'hui), "
                            "à partir de la valorisation quotidienne en €. "
                            "Moyenne, médiane et mode des rendements journaliers (annualisés) "
                            "aident à lire la skewness : asymétrie droite si moyenne > médiane > mode."
                        )
                        c1, c2, c3, c4, c5, c6 = st.columns(6)
                        c1.metric("Rendement (période)", f"{rendement_periode:.2%}")
                        c2.metric("Moyenne annuelle", f"{port_risk['Rendement Annuel (moyenne)']:.2%}")
                        c3.metric("Médiane annuelle", f"{port_risk['Médiane annuelle']:.2%}")
                        c4.metric("Mode annuelle", f"{port_risk['Mode annuelle']:.2%}")
                        c5.metric("Skewness", f"{port_risk['Skewness (Asymétrie)']:.2f}")
                        c6.metric("Volatilité annuelle", f"{port_risk['Volatilité (Sigma)']:.2%}")
                        c7, c8 = st.columns(2)
                        c7.metric("Ratio de Sharpe", f"{port_risk['Ratio de Sharpe']:.2f}")
                        c8.metric("Kurtosis", f"{port_risk['Kurtosis (Aplatissement)']:.2f}")

                        st.markdown("### Performance vs indice de référence")

                        @st.fragment
                        def _benchmark_comparison():
                            bench_label = st.selectbox(
                                "Indice de comparaison",
                                list(BENCHMARKS.keys()),
                                index=0,
                                key="synth_bench_label",
                            )
                            bench_ticker = BENCHMARKS[bench_label]
                            try:
                                bench_prices = cached_benchmark_prices(
                                    bench_ticker, str(start_date)
                                )
                                if not bench_prices.empty and bench_ticker in bench_prices.columns:
                                    bench_series = bench_prices[bench_ticker].dropna()
                                    common_idx = port_value.index.intersection(bench_series.index)
                                    if len(common_idx) >= 2:
                                        port_norm = port_value.loc[common_idx]
                                        bench_norm = bench_series.loc[common_idx]
                                        port_base100 = port_norm / port_norm.iloc[0] * 100
                                        bench_base100 = bench_norm / bench_norm.iloc[0] * 100
                                        perf_port = port_base100.iloc[-1] / 100 - 1
                                        perf_bench = bench_base100.iloc[-1] / 100 - 1
                                        alpha = perf_port - perf_bench

                                        m1, m2, m3 = st.columns(3)
                                        m1.metric("Perf. portefeuille", f"{perf_port:.2%}")
                                        m2.metric(f"Perf. {bench_label}", f"{perf_bench:.2%}")
                                        m3.metric("Écart (alpha)", f"{alpha:.2%}")

                                        fig_bench = go.Figure()
                                        fig_bench.add_trace(
                                            go.Scatter(
                                                x=port_base100.index,
                                                y=port_base100.values,
                                                mode="lines",
                                                name="Portefeuille",
                                            )
                                        )
                                        fig_bench.add_trace(
                                            go.Scatter(
                                                x=bench_base100.index,
                                                y=bench_base100.values,
                                                mode="lines",
                                                name=bench_label,
                                            )
                                        )
                                        fig_bench.update_layout(
                                            title=f"Portefeuille vs {bench_label} (base 100)",
                                            yaxis_title="Base 100",
                                        )
                                        st.plotly_chart(fig_bench, use_container_width=True)
                                    else:
                                        st.info("Pas assez de dates communes pour la comparaison.")
                                else:
                                    st.warning(f"Impossible de charger l'indice {bench_label}.")
                            except Exception as e:
                                st.warning(f"Comparaison benchmark indisponible : {e}")

                        _benchmark_comparison()
                elif not tickers_sans_cours:
                    st.warning("Aucune position valide à afficher.")
            else:
                st.header(f"📊 Aperçu des Derniers Cours de l'Indice : {indice_choisi}")
                last_prices = prices.iloc[-1].reset_index()
                last_prices.columns = ["Ticker", "Dernier Cours (Devise d'origine)"]
                last_prices = add_company_names(
                    last_prices, ticker_names, show_names=show_company_names, sectors=ticker_sectors
                )
                st.dataframe(
                    last_prices.style.format(
                        {"Dernier Cours (Devise d'origine)": "{:.2f}"}
                    )
                )

        elif vue == "Matrice de corrélation":
            st.header("📊 Matrice de Corrélation")
            corr = cached_corr_matrix(ohlcv_sig, returns)
            if show_company_names:
                labels = {
                    t: ticker_label(t, ticker_names, True) for t in corr.index
                }
                corr = corr.rename(index=labels, columns=labels)
            fig_corr = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="RdYlGn_r",
                zmin=-1,
                zmax=1,
            )
            st.plotly_chart(fig_corr, use_container_width=True)

        elif vue == "Analyse des risques":
            st.header("⚠️ Analyse des Risques")
            with st.expander("📖 Légende des mesures — que signifient ces chiffres ?", expanded=False):
                st.markdown(
                    """
    Les valeurs sont calculées à partir des **rendements journaliers** de chaque actif sur la période
    sélectionnée (sauf mention contraire).

    **Centre de la distribution (rendement typique)**

    | Mesure | Signification concrète |
    |--------|------------------------|
    | **Rendement annuel (moyenne)** | Gain (ou perte) moyen par jour, annualisé (× 252 jours). Sensible aux journées extrêmes. |
    | **Médiane annuelle** | Rendement « du milieu » : 50 % des journées sont au-dessus, 50 % en dessous. Plus robuste que la moyenne. |
    | **Mode annuelle** | Rendement le plus fréquent (classe la plus courante). Indique le régime le plus habituel. |

    **Lecture conjointe :** si *moyenne > médiane > mode*, la distribution est étirée vers les **gains**
    (asymétrie droite). Si *moyenne < médiane < mode*, elle est étirée vers les **pertes** (asymétrie gauche).
                    """
                )
                st.plotly_chart(build_mean_median_mode_pedagogy_chart(), use_container_width=True)
                st.caption(
                    "Graphique d'exemple : quelques journées très haussières tirent la **moyenne** (vert) "
                    "vers la droite ; la **médiane** (orange) reste au centre ; le **mode** (violet) indique "
                    "le rendement journalier le plus fréquent. Les valeurs entre parenthèses sont annualisées (× 252)."
                )
                st.markdown(
                    """
    **Risque et performance ajustée**

    | Mesure | Signification concrète |
    |--------|------------------------|
    | **Volatilité (Sigma)** | Amplitude des variations quotidiennes, annualisée. Plus elle est élevée, plus le titre bouge. |
    | **Ratio de Sharpe** | Rendement annualisé ÷ volatilité. Au-dessus de **1** = bon rapport rendement/risque ; en dessous de **0** = rendement négatif. |
    | **Ratio de Sortino** | Comme le Sharpe, mais ne pénalise que la volatilité **à la baisse** (pertes). |

    **Pertes extrêmes**

    | Mesure | Signification concrète |
    |--------|------------------------|
    | **VaR 95 % (jour)** | Perte journalière maximale « normale » : dans 95 % des cas, la baisse du jour ne dépasse pas ce niveau. |
    | **CVaR 95 % (jour)** | Perte moyenne les jours où l'on dépasse la VaR (queue de distribution — les pires journées). |

    **Forme de la distribution**

    | Mesure | Signification concrète |
    |--------|------------------------|
    | **Skewness (asymétrie)** | **> 0** : risque de grosses hausses ; **< 0** : risque de grosses baisses (krachs). Proche de **0** : distribution symétrique. |
    | **Kurtosis (aplatissement)** | **> 0** : queues épaisses — événements extrêmes plus fréquents qu'une cloche normale. **< 0** : distribution plus « plate » au centre. |
                    """
                )
            metrics = cached_advanced_risk_metrics(ohlcv_sig, returns)
            metrics = label_index_with_names(metrics, ticker_names, show_company_names)
            st.caption(
                "Rendements et volatilité en **%** (annualisés) · VaR/CVaR en **%** journalier · "
                "Sharpe/Sortino en **×** (sans unité monétaire)."
            )
            st.dataframe(
                metrics.style.format(RISK_METRICS_FORMAT)
                .background_gradient(
                    cmap="RdYlGn",
                    subset=["Rendement Annuel (moyenne)", "Médiane annuelle", "Ratio de Sharpe", "Ratio de Sortino"],
                ).background_gradient(
                    cmap="RdYlGn_r",
                    subset=["Volatilité (Sigma)", "VaR 95% (Jour)", "CVaR 95% (Jour)"],
                )
            )

            st.markdown("### Boîtes à moustaches — rendements journaliers")
            st.caption(
                "Pour chaque actif, un **trait horizontal** va du rendement journalier **minimum** "
                "au **maximum** ; les repères indiquent **Q1**, la **médiane** (losange) et **Q3**."
            )
            label_map = {}
            if show_company_names:
                label_map = {
                    col: ticker_label(col, ticker_names, True)
                    for col in returns.columns
                }
            st.plotly_chart(
                build_risk_returns_boxplot(returns, column_labels=label_map),
                use_container_width=True,
            )

            st.markdown("### Boîtes à moustaches — métriques agrégées")
            st.caption(
                "Compare la **dispersion entre actifs** pour une même mesure : trait **min–max**, "
                "repères **Q1 / médiane / Q3**, et **points violets** (un par titre). "
                "**Survolez un point** pour afficher le nom de l'actif et sa valeur."
            )

            @st.fragment
            def _risk_metrics_chart():
                metric_groups = {
                    "Rendement (annualisé)": [
                        "Rendement Annuel (moyenne)",
                        "Médiane annuelle",
                        "Mode annuelle",
                    ],
                    "Volatilité & pertes extrêmes": [
                        "Volatilité (Sigma)",
                        "VaR 95% (Jour)",
                        "CVaR 95% (Jour)",
                    ],
                    "Ratios ajustés du risque": ["Ratio de Sharpe", "Ratio de Sortino"],
                    "Forme de la distribution": [
                        "Skewness (Asymétrie)",
                        "Kurtosis (Aplatissement)",
                    ],
                }
                selected_group = st.selectbox(
                    "Groupe de mesures",
                    list(metric_groups.keys()),
                    key="risk_box_metric_group",
                )
                st.plotly_chart(
                    build_risk_metrics_boxplot(metrics, metric_groups[selected_group]),
                    use_container_width=True,
                )

            _risk_metrics_chart()

        elif vue == "Optimisation (Markowitz)":
            st.header("🎯 Allocation Optimale (Poids Théoriques)")
            st.caption(
                "Optimisation **long-only** (poids ≥ 0, somme = 100 %) — maximisation du ratio de Sharpe "
                "sur l'historique de rendements sélectionné."
            )

            @st.fragment
            def _markowitz_view():
                if returns.shape[1] < 2:
                    st.warning("Au moins **2 actifs** avec des rendements sont nécessaires pour Markowitz.")
                    return
                c_mk1, c_mk2, c_mk3 = st.columns(3)
                n_random = c_mk1.slider(
                    "Portefeuilles aléatoires (nuage)",
                    min_value=500,
                    max_value=5000,
                    value=2000,
                    step=500,
                    key="markowitz_n_random",
                )
                rf_pct = c_mk2.number_input(
                    "Taux sans risque (% / an)",
                    min_value=0.0,
                    max_value=10.0,
                    value=2.0,
                    step=0.25,
                    key="markowitz_rf",
                )
                show_cal = c_mk3.checkbox(
                    "Afficher la CAL",
                    value=True,
                    help="Capital Allocation Line — droite tangente au portefeuille max Sharpe.",
                    key="markowitz_show_cal",
                )
                risk_free_rate = rf_pct / 100.0 if show_cal else 0.0

                weights = cached_markowitz_weights(ohlcv_sig, returns)

                current_weights = None
                total_valeur_eur = None
                if mode_portfolio_actif and portefeuille_dict:
                    usd_to_eur = cached_usd_to_eur_rate()
                    quantities_mk = {
                        tk: portefeuille_dict[tk]["Quantite"] for tk in portefeuille_dict
                    }
                    held = compute_holdings_weights(
                        prices,
                        quantities_mk,
                        usd_to_eur=usd_to_eur,
                        is_usd_fn=is_usd_ticker,
                        universe=list(returns.columns),
                    )
                    current_weights = _align_weights_to_returns(returns, held)
                    total_valeur_eur = portfolio_market_value_eur(
                        prices, quantities_mk, usd_to_eur=usd_to_eur, is_usd_fn=is_usd_ticker
                    )

                df_w = pd.DataFrame({"Poids Optimal": weights}).sort_values(
                    by="Poids Optimal", ascending=False
                )
                df_w = label_index_with_names(df_w, ticker_names, show_company_names)

                asset_labels = [
                    ticker_label(t, ticker_names, show_company_names)
                    for t in returns.columns
                ]
                label_map = {t: asset_labels[i] for i, t in enumerate(returns.columns)}

                if current_weights is not None and current_weights.sum() > 0:
                    st.markdown("### Situation actuelle vs après arbitrage Markowitz")
                    perf_cur = portfolio_performance_from_weights(returns, current_weights)
                    perf_tgt = portfolio_performance_from_weights(returns, weights)
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.markdown("**Portefeuille actuel**")
                        if perf_cur:
                            st.metric("Rendement espéré", f"{perf_cur['Rendement']:.2%}")
                            st.metric("Volatilité", f"{perf_cur['Volatilité']:.2%}")
                            st.metric("Sharpe", f"{perf_cur['Sharpe']:.2f}")
                    with m2:
                        st.markdown("**Après arbitrage (Markowitz)**")
                        if perf_tgt:
                            st.metric(
                                "Rendement espéré",
                                f"{perf_tgt['Rendement']:.2%}",
                                delta=(
                                    f"{(perf_tgt['Rendement'] - perf_cur['Rendement']):+.2%}"
                                    if perf_cur
                                    else None
                                ),
                            )
                            st.metric(
                                "Volatilité",
                                f"{perf_tgt['Volatilité']:.2%}",
                                delta=(
                                    f"{(perf_tgt['Volatilité'] - perf_cur['Volatilité']):+.2%}"
                                    if perf_cur
                                    else None
                                ),
                            )
                            st.metric(
                                "Sharpe",
                                f"{perf_tgt['Sharpe']:.2f}",
                                delta=(
                                    f"{(perf_tgt['Sharpe'] - perf_cur['Sharpe']):+.2f}"
                                    if perf_cur
                                    else None
                                ),
                            )
                    with m3:
                        st.markdown("**Arbitrage indicatif**")
                        if total_valeur_eur:
                            st.metric("Valorisation totale", f"{total_valeur_eur:,.0f} €")
                        top_cur = current_weights.idxmax()
                        top_tgt = weights.idxmax()
                        st.caption(
                            f"Poids max **actuel** : {ticker_label(top_cur, ticker_names, show_company_names)} "
                            f"({current_weights[top_cur]:.1%})"
                        )
                        st.caption(
                            f"Poids max **cible** : {ticker_label(top_tgt, ticker_names, show_company_names)} "
                            f"({weights[top_tgt]:.1%})"
                        )

                    arb_df = build_markowitz_arbitrage_table(
                        current_weights, weights, labels=label_map
                    )
                    if ticker_sectors:
                        arb_df.insert(
                            2,
                            "Secteur",
                            arb_df["Ticker"].map(lambda t: ticker_sectors.get(t, "—")),
                        )
                    if total_valeur_eur:
                        arb_df["Arbitrage (€)"] = arb_df["Écart (pts)"] * total_valeur_eur
                    st.dataframe(
                        arb_df.style.format(
                            {
                                "Poids actuel": "{:.2%}",
                                "Poids cible (Markowitz)": "{:.2%}",
                                "Écart (pts)": "{:+.2%}",
                                **(
                                    {"Arbitrage (€)": "{:+,.0f}"}
                                    if total_valeur_eur
                                    else {}
                                ),
                            }
                        ).background_gradient(
                            cmap="RdYlGn",
                            subset=["Écart (pts)"],
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                    st.caption(
                        "Écart **positif** = renforcer · **négatif** = alléger. "
                        "Montants indicatifs (réallocation sans frais ni fiscalité)."
                    )

                    col_cur, col_tgt = st.columns(2)
                    with col_cur:
                        st.plotly_chart(
                            build_markowitz_assets_figure(
                                returns,
                                current_weights,
                                labels=asset_labels,
                                title="Situation actuelle (taille = poids réel)",
                                portfolio_label="Portefeuille actuel",
                                portfolio_color="#7b1fa2",
                                colorscale="Purples",
                            ),
                            use_container_width=True,
                        )
                    with col_tgt:
                        st.plotly_chart(
                            build_markowitz_assets_figure(
                                returns,
                                weights,
                                labels=asset_labels,
                                title="Après arbitrage Markowitz (taille = poids cible)",
                                portfolio_label="Portefeuille cible",
                                portfolio_color="#ffd600",
                                colorscale="Viridis",
                            ),
                            use_container_width=True,
                        )
                else:
                    st.info(
                        "Chargez votre **portefeuille CSV** (mode « Mon Portefeuille Réel ») "
                        "pour comparer la situation actuelle et l'allocation Markowitz."
                    )
                    st.dataframe(df_w.style.format({"Poids Optimal": "{:.2%}"}))
                    st.caption("Poids théoriques Markowitz en **%** du portefeuille.")

                with st.expander("📖 Légende — comment lire ces graphiques ?", expanded=False):
                    st.markdown(
                        """
| Élément | Signification |
|---------|---------------|
| **Points rouges** | Portefeuilles **aléatoires** (poids tirés au hasard, long-only). |
| **Courbe bleue** | **Frontière efficiente** — meilleur rendement possible pour chaque niveau de risque. |
| **Étoile violette** | **Portefeuille actuel** (poids réels du CSV). |
| **Étoile jaune** | **Après arbitrage Markowitz** (max Sharpe). |
| **Graphiques actifs** | Taille du point ∝ poids · gauche = **actuel**, droite = **cible**. |
| **CAL (noire)** | Avec actif sans risque : combinaisons optimales risque/rendement le long de la tangente. |

**Précautions :** rendements passés, covariance stable et normalité implicite — repère théorique, pas recommandation.
                        """
                    )

                st.markdown("### Frontière efficiente")
                has_current = current_weights is not None and current_weights.sum() > 0
                fig_frontier = cached_markowitz_frontier_figure(
                    ohlcv_sig,
                    returns,
                    weights,
                    risk_free_rate,
                    n_random,
                    has_current,
                    current_weights if has_current else None,
                )
                st.plotly_chart(fig_frontier, use_container_width=True)

                if not has_current:
                    st.markdown("### Actifs — taille = poids dans le portefeuille optimal")
                    st.plotly_chart(
                        build_markowitz_assets_figure(
                            returns,
                            weights,
                            labels=asset_labels,
                            title="Allocation Markowitz (taille = poids cible)",
                        ),
                        use_container_width=True,
                    )

                csv = df_w.reset_index().rename(columns={"index": "Ticker"}).to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Télécharger les poids (CSV)",
                    csv,
                    "markowitz_poids.csv",
                    key="markowitz_csv",
                )

            _markowitz_view()

        elif vue == "Dividendes & Qualité":
            if HAS_DIVIDENDES:
                clean_prices = prices.copy()
                clean_returns = returns.copy()
                if isinstance(clean_prices.columns, pd.MultiIndex):
                    clean_prices.columns = clean_prices.columns.get_level_values(0)
                if isinstance(clean_returns.columns, pd.MultiIndex):
                    clean_returns.columns = clean_returns.columns.get_level_values(0)
                try:
                    dividendes.render_dividendes_dashboard(clean_prices, clean_returns)
                except Exception as e:
                    st.error(f"Erreur d'analyse dans le module externe de dividendes : {e}")
            else:
                st.error("Le module 'dividendes.py' n'a pas pu être chargé.")

        elif vue == "Analyse fondamentale":
            if HAS_FUNDAMENTALS:
                st.caption(
                    "Ratios de valorisation, endettement et consensus analystes via Yahoo Finance / YahooQuery."
                )
                fundamentals.render_fundamentals_dashboard(prices)
            else:
                st.error("Le module 'fundamentals.py' n'a pas pu être chargé.")

        elif vue == "Analyse technique (RSI / MACD)":
            st.header("📈 Analyse technique")
            st.caption(
                "Indicateurs calculés localement à partir des prix Yahoo (graphiques en chandeliers japonais) : "
                "RSI, MACD, stochastique, moyennes mobiles 20/50/200, Bollinger, ATR, "
                "volumes, OBV, MFI, régression linéaire, supports/résistances et Fibonacci."
            )
            @st.fragment
            def _technical_summary():
                rsi_period = st.slider("Période RSI", 7, 21, 14, key="tech_rsi_period")
                tech_df = cached_technical_indicators(
                    ohlcv_sig,
                    rsi_period,
                    prices,
                    volumes,
                    highs,
                    lows,
                )
                if tech_df.empty:
                    st.warning("Pas assez de données pour calculer les indicateurs techniques.")
                    st.session_state.pop("technical_df", None)
                    return
                st.session_state["technical_df"] = tech_df
                st.session_state["technical_rsi_period"] = rsi_period
                tech_display_df = add_company_names(
                    tech_df, ticker_names, show_names=show_company_names, sectors=ticker_sectors
                )
                tech_display = rename_columns_for_display(tech_display_df, TECHNICAL_LABELS)
                tech_format = format_map_for_labeled_columns(
                    tech_display, TECHNICAL_LABELS, TECHNICAL_FORMAT
                )
                st.caption(
                    "Prix et moyennes en **/ action** · volumes en **titres** · "
                    "RSI/MFI/stochastique sur **0–100** · ATR/Prix en **%**."
                )
                st.dataframe(
                    tech_display.style.format(tech_format, na_rep="-")
                )

            @st.fragment
            def _technical_detail():
                tech_df = st.session_state.get("technical_df")
                rsi_period_detail = st.session_state.get("technical_rsi_period", 14)
                if tech_df is None or tech_df.empty:
                    return
                detail_ticker = st.selectbox(
                    "Graphiques détaillés",
                    tech_df["Ticker"],
                    format_func=lambda t: ticker_label(t, ticker_names, show_company_names),
                )
                detail_label = ticker_label(detail_ticker, ticker_names, show_company_names)
                bundle = cached_technical_detail_bundle(
                    ohlcv_sig,
                    detail_ticker,
                    rsi_period_detail,
                    prices,
                    volumes,
                    highs,
                    lows,
                    opens,
                    ohlc_closes,
                )
                s = bundle["s"]
                rsi_series = bundle["rsi"]
                macd_line, signal_line, histogram = bundle["macd"]
                sma50 = bundle["sma50"]
                sma200 = bundle["sma200"]
                bb_upper, bb_mid, bb_lower = bundle["bollinger"]
                stoch_k, stoch_d = bundle["stochastic"]
                vol_s = bundle["vol_s"]
                has_volume = bundle["has_volume"]
                ohlc = bundle["ohlc"]
                has_ohlc = bundle["has_ohlc"]

                fig = make_subplots(
                    rows=2,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.06,
                    row_heights=[0.72, 0.28],
                    subplot_titles=(f"{detail_label} — Chandeliers & MM", "Volume échangé"),
                )
                if has_ohlc:
                    fig.add_trace(candlestick_trace(ohlc), row=1, col=1)
                else:
                    fig.add_trace(
                        go.Scatter(x=s.index, y=s, mode="lines", name="Prix"),
                        row=1,
                        col=1,
                    )
                if not sma50.empty:
                    fig.add_trace(
                        go.Scatter(x=sma50.index, y=sma50, mode="lines", name="SMA 50"),
                        row=1,
                        col=1,
                    )
                if not sma200.empty:
                    fig.add_trace(
                        go.Scatter(x=sma200.index, y=sma200, mode="lines", name="SMA 200"),
                        row=1,
                        col=1,
                    )
                if not bb_upper.empty:
                    if not bb_mid.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=bb_mid.index,
                                y=bb_mid,
                                mode="lines",
                                name="SMA 20 (Bollinger)",
                                line=dict(color="orange", width=1.5),
                            ),
                            row=1,
                            col=1,
                        )
                    fig.add_trace(
                        go.Scatter(
                            x=bb_upper.index,
                            y=bb_upper,
                            mode="lines",
                            name="Bollinger haut",
                            line=dict(dash="dot"),
                        ),
                        row=1,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=bb_lower.index,
                            y=bb_lower,
                            mode="lines",
                            name="Bollinger bas",
                            line=dict(dash="dot"),
                        ),
                        row=1,
                        col=1,
                    )
                if has_volume:
                    vol_aligned = volumes[detail_ticker].reindex(s.index)
                    vol_sma20 = compute_volume_sma(vol_aligned, 20)
                    vol_idx = vol_aligned.dropna().index
                    bar_colors = []
                    for d in vol_idx:
                        if has_ohlc and d in ohlc.index:
                            bar_colors.append(
                                "#26a69a" if ohlc.loc[d, "close"] >= ohlc.loc[d, "open"] else "#ef5350"
                            )
                        elif d in s.index and pd.notna(s.loc[d]) and pd.notna(s.shift(1).loc[d]):
                            bar_colors.append(
                                "#26a69a" if s.loc[d] >= s.shift(1).loc[d] else "#ef5350"
                            )
                        else:
                            bar_colors.append("#42a5f5")
                    fig.add_trace(
                        go.Bar(
                            x=vol_idx,
                            y=vol_aligned.loc[vol_idx],
                            name="Volume",
                            marker_color=bar_colors,
                            opacity=0.7,
                        ),
                        row=2,
                        col=1,
                    )
                    if not vol_sma20.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=vol_sma20.index,
                                y=vol_sma20,
                                mode="lines",
                                name="Moy. volume 20j",
                                line=dict(color="orange", width=2),
                            ),
                            row=2,
                            col=1,
                        )
                fig.update_layout(height=620, showlegend=True, xaxis_rangeslider_visible=False)
                fig.update_yaxes(title_text="Prix", row=1, col=1)
                fig.update_yaxes(title_text="Titres", row=2, col=1)
                st.plotly_chart(fig, use_container_width=True)

                c1, c2 = st.columns(2)
                with c1:
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(
                        go.Scatter(x=rsi_series.index, y=rsi_series, mode="lines", name=f"RSI ({rsi_period_detail})")
                    )
                    fig_rsi.add_hline(y=70, line_dash="dot", line_color="red")
                    fig_rsi.add_hline(y=30, line_dash="dot", line_color="green")
                    fig_rsi.update_layout(title=f"{detail_label} — RSI", yaxis=dict(range=[0, 100]))
                    st.plotly_chart(fig_rsi, use_container_width=True)
                    st.info(interpret_rsi_comment(rsi_series, rsi_period_detail))
                with c2:
                    fig_stoch = go.Figure()
                    fig_stoch.add_trace(go.Scatter(x=stoch_k.index, y=stoch_k, mode="lines", name="%K"))
                    fig_stoch.add_trace(go.Scatter(x=stoch_d.index, y=stoch_d, mode="lines", name="%D"))
                    fig_stoch.add_hline(y=80, line_dash="dot", line_color="red")
                    fig_stoch.add_hline(y=20, line_dash="dot", line_color="green")
                    fig_stoch.update_layout(title=f"{detail_label} — Stochastique", yaxis=dict(range=[0, 100]))
                    st.plotly_chart(fig_stoch, use_container_width=True)
                    st.info(interpret_stochastic_comment(stoch_k, stoch_d))

                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(x=macd_line.index, y=macd_line, mode="lines", name="MACD"))
                fig_macd.add_trace(go.Scatter(x=signal_line.index, y=signal_line, mode="lines", name="Signal"))
                fig_macd.add_trace(go.Bar(x=histogram.index, y=histogram, name="Histogramme", opacity=0.4))
                fig_macd.update_layout(title=f"{detail_label} — MACD")
                st.plotly_chart(fig_macd, use_container_width=True)
                st.info(interpret_macd_comment(macd_line, signal_line, histogram))

                st.subheader("📐 Tendance, supports & Fibonacci")
                reg_channel = compute_linear_regression_channel(s)
                supports, resistances = compute_support_resistance(s, window=10, n_levels=2)
                h_series = (
                    highs[detail_ticker].reindex(s.index)
                    if not highs.empty and detail_ticker in highs.columns
                    else None
                )
                l_series = (
                    lows[detail_ticker].reindex(s.index)
                    if not lows.empty and detail_ticker in lows.columns
                    else None
                )
                pivot = compute_pivot_level(s, highs=h_series, lows=l_series)
                fib_data = compute_fibonacci_levels(s, highs=h_series, lows=l_series)
                fib_colors = {
                    "0.0%": "#6a1b9a",
                    "23.6%": "#7b1fa2",
                    "38.2%": "#8e24aa",
                    "50.0%": "#9c27b0",
                    "61.8%": "#ab47bc",
                    "78.6%": "#ba68c8",
                    "100.0%": "#ce93d8",
                }

                tab_reg, tab_sr, tab_fib = st.tabs(
                    ["Régression & canal σ", "Supports / Résistances", "Fibonacci"]
                )

                with tab_reg:
                    fig_reg = go.Figure()
                    if has_ohlc:
                        fig_reg.add_trace(candlestick_trace(ohlc))
                    else:
                        fig_reg.add_trace(
                            go.Scatter(x=s.index, y=s, mode="lines", name="Prix", line=dict(color="#1f77b4", width=2))
                        )
                    if reg_channel:
                        fig_reg.add_trace(
                            go.Scatter(
                                x=reg_channel["regression"].index,
                                y=reg_channel["regression"],
                                mode="lines",
                                name="Régression linéaire",
                                line=dict(color="#1565c0", width=2),
                            )
                        )
                        for key, label, dash, width in [
                            ("plus_1std", "+1σ", "dash", 1.2),
                            ("minus_1std", "−1σ", "dash", 1.2),
                            ("plus_2std", "+2σ", "dot", 1),
                            ("minus_2std", "−2σ", "dot", 1),
                        ]:
                            fig_reg.add_trace(
                                go.Scatter(
                                    x=reg_channel[key].index,
                                    y=reg_channel[key],
                                    mode="lines",
                                    name=label,
                                    line=dict(color="#64b5f6", width=width, dash=dash),
                                )
                            )
                    fig_reg.update_layout(
                        title=f"{detail_label} — Régression & bandes ±1σ / ±2σ",
                        height=450,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        hovermode="x unified",
                        xaxis_rangeslider_visible=False,
                    )
                    st.plotly_chart(fig_reg, use_container_width=True)
                    if reg_channel:
                        st.info(interpret_regression_channel_comment(s, reg_channel))

                with tab_sr:
                    fig_sr = go.Figure()
                    if has_ohlc:
                        fig_sr.add_trace(candlestick_trace(ohlc))
                    else:
                        fig_sr.add_trace(
                            go.Scatter(x=s.index, y=s, mode="lines", name="Prix", line=dict(color="#1f77b4", width=2))
                        )
                    for i, sup in enumerate(supports, start=1):
                        fig_sr.add_hline(
                            y=sup,
                            line_dash="longdash",
                            line_color="#2e7d32",
                            line_width=1.5,
                            annotation_text=f"Support {i} ({sup:.2f})",
                            annotation_position="bottom left",
                            annotation_font_size=10,
                            annotation_font_color="#2e7d32",
                        )
                    for i, res in enumerate(resistances, start=1):
                        fig_sr.add_hline(
                            y=res,
                            line_dash="longdash",
                            line_color="#c62828",
                            line_width=1.5,
                            annotation_text=f"Résistance {i} ({res:.2f})",
                            annotation_position="top left",
                            annotation_font_size=10,
                            annotation_font_color="#c62828",
                        )
                    if pivot is not None:
                        fig_sr.add_hline(
                            y=pivot,
                            line_dash="solid",
                            line_color="#1565c0",
                            line_width=2,
                            annotation_text=f"Pivot ({pivot:.2f})",
                            annotation_position="right",
                            annotation_font_size=10,
                            annotation_font_color="#1565c0",
                        )
                    last_price = float(s.iloc[-1])
                    max_res = max(resistances) if resistances else last_price
                    min_sup = min(supports) if supports else last_price
                    y_lo = min_sup * 0.96
                    y_hi = max(max_res, min(last_price, max_res * 1.25)) * 1.06
                    if pivot is not None:
                        y_lo = min(y_lo, pivot * 0.96)
                        y_hi = max(y_hi, pivot * 1.04)
                    if last_price > max_res * 1.4:
                        st.caption(
                            "Échelle centrée sur la zone support/résistance — le pic récent "
                            "est hors champ pour mieux lire les planchers et plafonds."
                        )
                    else:
                        y_hi = max(y_hi, last_price * 1.04)
                    fig_sr.update_layout(
                        title=f"{detail_label} — Supports & résistances",
                        height=450,
                        yaxis=dict(range=[y_lo, y_hi]),
                        hovermode="x unified",
                        xaxis_rangeslider_visible=False,
                    )
                    st.plotly_chart(fig_sr, use_container_width=True)
                    st.info(interpret_support_resistance_comment(s, supports, resistances, pivot=pivot))

                with tab_fib:
                    fig_fib = go.Figure()
                    if has_ohlc:
                        fig_fib.add_trace(candlestick_trace(ohlc))
                    else:
                        fig_fib.add_trace(
                            go.Scatter(x=s.index, y=s, mode="lines", name="Prix", line=dict(color="#1f77b4", width=2))
                        )
                    if fib_data:
                        for fib_label, fib_price in fib_data["levels"].items():
                            fig_fib.add_hline(
                                y=fib_price,
                                line_dash="dashdot",
                                line_color=fib_colors.get(fib_label, "#9c27b0"),
                                line_width=1,
                                annotation_text=f"Fib {fib_label}",
                                annotation_position="right",
                                annotation_font_size=9,
                                annotation_font_color=fib_colors.get(fib_label, "#9c27b0"),
                            )
                        fib_lo = min(fib_data["levels"].values())
                        fib_hi = max(fib_data["levels"].values())
                        fib_pad = (fib_hi - fib_lo) * 0.04 or 0.1
                        fig_fib.update_layout(
                            title=f"{detail_label} — Niveaux de Fibonacci",
                            height=450,
                            yaxis=dict(range=[fib_lo - fib_pad, fib_hi + fib_pad]),
                            hovermode="x unified",
                            xaxis_rangeslider_visible=False,
                        )
                    else:
                        fig_fib.update_layout(
                            title=f"{detail_label} — Niveaux de Fibonacci",
                            height=450,
                            hovermode="x unified",
                            xaxis_rangeslider_visible=False,
                        )
                    st.plotly_chart(fig_fib, use_container_width=True)
                    if fib_data:
                        st.info(interpret_fibonacci_comment(s, fib_data))

                if has_volume:
                    st.subheader("📊 Volumes & indicateurs associés")
                    vol_aligned = volumes[detail_ticker].reindex(s.index)
                    vol_sma20 = compute_volume_sma(vol_aligned, 20)
                    st.info(interpret_volume_comment(vol_aligned, s, period=20))

                    obv = compute_obv(s, vol_aligned)
                    mfi_series = pd.Series(dtype=float)
                    if (
                        not highs.empty
                        and not lows.empty
                        and detail_ticker in highs.columns
                        and detail_ticker in lows.columns
                    ):
                        h = highs[detail_ticker].reindex(s.index)
                        l = lows[detail_ticker].reindex(s.index)
                        mfi_series = compute_mfi(h, l, s, vol_aligned)

                    c3, c4 = st.columns(2)
                    with c3:
                        fig_obv = go.Figure()
                        fig_obv.add_trace(
                            go.Scatter(x=obv.index, y=obv, mode="lines", name="OBV", line=dict(color="#7e57c2"))
                        )
                        fig_obv.update_layout(title=f"{detail_label} — OBV (On-Balance Volume)")
                        st.plotly_chart(fig_obv, use_container_width=True)
                        st.info(interpret_obv_comment(obv, s))
                    with c4:
                        if not mfi_series.empty:
                            fig_mfi = go.Figure()
                            fig_mfi.add_trace(
                                go.Scatter(x=mfi_series.index, y=mfi_series, mode="lines", name="MFI")
                            )
                            fig_mfi.add_hline(y=80, line_dash="dot", line_color="red")
                            fig_mfi.add_hline(y=20, line_dash="dot", line_color="green")
                            fig_mfi.update_layout(
                                title=f"{detail_label} — MFI (Money Flow Index)",
                                yaxis=dict(range=[0, 100]),
                            )
                            st.plotly_chart(fig_mfi, use_container_width=True)
                            st.info(interpret_mfi_comment(mfi_series))
                        else:
                            st.caption("MFI indisponible (plus hauts/bas manquants pour ce titre).")
                else:
                    st.info(
                        "Volumes non disponibles pour ce titre sur Yahoo Finance. "
                        "Les indicateurs OBV et MFI ne peuvent pas être calculés."
                    )

            _technical_summary()
            _technical_detail()

        elif vue == "Analyse de Fourier (FFT)":
            st.header("🌊 Analyse de Fourier (FFT) — cycles périodiques")
            st.caption(
                "Décomposition fréquentielle des cours (prix logarithmiques dé-trendés) pour repérer "
                "d'éventuelles **périodicités** (semaines, mois, trimestre…). "
                "Indicateur exploratoire — les marchés ne sont pas parfaitement périodiques."
            )

            with st.expander("📖 Légende — comment lire cette analyse ?", expanded=False):
                st.markdown(
                    """
| Terme | Signification |
|-------|----------------|
| **FFT** | *Fast Fourier Transform* — transforme l'historique des prix en **fréquences** (cycles). |
| **Période (jours)** | Durée estimée d'un cycle complet (jours de bourse, ~5 j/semaine). |
| **Force du cycle** | **Même chose que « Puissance relative »** : part (en **%**) de l'énergie FFT portée par ce cycle dans la bande de périodes analysée (sliders min/max). Ex. **18 %** = motif modéré ; **> 25 %** = cycle assez visible ; **< 10 %** = faible, à prendre avec prudence. Affichée en métrique et dans le tableau des pics. |
| **Puissance relative** | Synonyme technique de **force du cycle** (colonne du tableau des pics dominants). |
| **Densité spectrale** | Courbe bleue (graphique du haut) : répartition de l'énergie par **période** (amplitude FFT² normalisée). Les **points rouges** marquent les pics retenus. |
| **Autocorrélation (ACF)** | Courbe violette (graphique du bas) : à quel point le signal **ressemble à lui-même** décalé de *n* jours (théorème de Wiener-Khinchin — même information que le spectre, autre vue). |
| **Comment lire l'ACF** | **Proche de +1** au décalage *n* → le motif tend à se **répéter** tous les *n* jours. **Proche de 0** → pas de régularité à ce décalage. **Négatif** → tendance à **alterner** (hausse puis baisse). **Entre les pointillés gris (±95 %)** → compatible avec le **bruit** (pas de cycle clair à ce lag). |
| **Lignes orange (ACF)** | Décalages (en jours) correspondant aux **périodes dominantes** détectées par FFT : si l'ACF est aussi élevée à ces lags, le cycle FFT est **confirmé** ; si l'ACF reste proche de 0, le pic FFT peut être **accidentel**. |
| **Prix dé-trendé** | On retire la tendance longue pour isoler les oscillations analysées par FFT. |
| **Tendance de fond** | Ligne grise en **€** : tendance retirée avant FFT ; type choisi sous le graphique cyclique. |
| **R² tendance** | Coefficient affiché en **métrique** (4ᵉ colonne) et sous le graphique cyclique : qualité de l'ajustement de la tendance au **prix réel**. **1,000** = parfait · **> 0,85** = bon · **< 0,5** = la tendance choisie ne décrit pas bien le cours (cycles moins fiables). |
| **RMSE** | Écart-type moyen (€) entre prix réel et tendance — complète le R² (affiché sous le graphique cyclique). |
| **Modèle FFT** | Tendance + cycles combinés, reprojetés en **prix** (cyan) ; prolongation **+30 %** en pointillé après la ligne « Fin historique ». |
| **Prix** | Courbe **bleue** : historique réel du titre. |

**Tendances disponibles :** **Log-linéaire** (croissance % constante) · **Linéaire €** (droite sur le prix)
· **Hodrick-Prescott** (courbe lisse). Comparez les **R²** affichés en changeant la tendance sous le graphique cyclique.

**Extrapolation :** le modèle mathématique (tendance + harmoniques) est prolongé de **30 %** de la durée
observée (jours de bourse). Zone pointillée = **scénario exploratoire**, pas une prévision garantie.

**Précautions :** un pic FFT peut venir du hasard, d'un dividende récurrent mal lissé ou d'une
fenêtre trop courte. Croisez avec l'analyse technique et fondamentale.
                    """
                )

            if prices.empty or len(prices.dropna(how="all")) < 40:
                st.warning(
                    "Historique insuffisant pour une FFT fiable (minimum ~40 séances). "
                    "Élargissez la date de début dans la barre latérale."
                )
            else:
                c_fft1, c_fft2, c_fft3 = st.columns(3)
                min_period = c_fft1.slider(
                    "Période min. recherchée (jours)",
                    min_value=3,
                    max_value=30,
                    value=5,
                    step=1,
                )
                max_period = c_fft2.slider(
                    "Période max. recherchée (jours)",
                    min_value=30,
                    max_value=min(504, len(prices) // 2),
                    value=min(252, max(60, len(prices) // 2)),
                    step=5,
                )
                top_peaks = c_fft3.slider("Nombre de pics affichés", 3, 10, 5)

                def _fft_trend_quality_caption(result):
                    if not result:
                        return
                    r2 = result.get("trend_r2")
                    rmse = result.get("trend_rmse")
                    r2_txt = f"{r2:.3f}" if r2 is not None and not np.isnan(r2) else "—"
                    rmse_txt = f"{rmse:,.2f}" if rmse is not None and not np.isnan(rmse) else "—"
                    st.caption(
                        f"Tendance : **{fft_trend_mode_label(result.get('trend_mode', trend_mode))}** · "
                        f"R² = **{r2_txt}** · RMSE = **{rmse_txt}** € "
                        "(qualité de la tendance retirée avant FFT — voir légende)"
                    )

                portfolio_tickers_fft = [t for t in liste_tickers if t in prices.columns]
                quantities_fft = None
                portfolio_fft_label = "Portefeuille (indice équipondéré, base 100)"
                if mode_portfolio_actif and portefeuille_dict:
                    quantities_fft = {
                        t: portefeuille_dict[t]["Quantite"]
                        for t in portfolio_tickers_fft
                        if t in portefeuille_dict
                    }
                    portfolio_fft_label = "Portefeuille complet (valorisation €, pondéré par quantités)"

                usd_to_eur_fft = cached_usd_to_eur_rate() if mode_portfolio_actif else 1.0
                portfolio_series = build_portfolio_price_for_fft(
                    prices,
                    tickers=portfolio_tickers_fft,
                    quantities=quantities_fft,
                    usd_to_eur=usd_to_eur_fft,
                    is_usd_fn=is_usd_ticker,
                )

                st.markdown("### Portefeuille complet")
                st.caption(portfolio_fft_label)

                if "fft_trend_mode" not in st.session_state:
                    st.session_state.fft_trend_mode = FFT_TREND_LOG_LINEAR
                trend_mode = st.session_state.fft_trend_mode

                if portfolio_series is None or len(portfolio_series.dropna()) < 40:
                    st.warning(
                        "Historique insuffisant pour la FFT du portefeuille agrégé "
                        "(minimum ~40 séances communes)."
                    )
                else:
                    port_fft = cached_fft_periodicity(
                        f"{ohlcv_sig}|portfolio|{portfolio_fft_label}|{trend_mode}",
                        min_period,
                        max_period,
                        top_peaks,
                        trend_mode,
                        portfolio_series,
                    )
                    if port_fft is None:
                        st.warning("FFT portefeuille indisponible sur la période.")
                    else:
                        port_peaks = port_fft["peaks"].drop(columns=["_freq_idx"], errors="ignore")
                        _render_fft_headline_metrics(port_fft, port_peaks)
                        st.plotly_chart(
                            build_fft_spectral_acf_chart(
                                port_fft,
                                title="Densité spectrale & autocorrélation — portefeuille complet",
                            ),
                            use_container_width=True,
                        )
                        _acf_reading = interpret_fft_acf_reading(port_fft)
                        if _acf_reading:
                            st.info(_acf_reading)
                        port_peaks_disp = rename_columns_for_display(port_peaks, FFT_LABELS)
                        st.dataframe(
                            port_peaks_disp.style.format(
                                format_map_for_labeled_columns(
                                    port_peaks_disp, FFT_LABELS, FFT_FORMAT
                                )
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )
                        _trend_ctrl, _trend_metrics = st.columns([1, 2])
                        with _trend_ctrl:
                            st.selectbox(
                                "Tendance de fond",
                                options=list(FFT_TREND_OPTIONS.keys()),
                                format_func=fft_trend_mode_label,
                                key="fft_trend_mode",
                                help="Méthode pour isoler les cycles avant FFT.",
                            )
                        with _trend_metrics:
                            _fft_trend_quality_caption(port_fft)
                        st.plotly_chart(
                            build_fft_cyclic_chart(
                                port_fft,
                                n_components=min(top_peaks, len(port_fft["peaks"])),
                                title="Cycle estimé vs portefeuille complet",
                            ),
                            use_container_width=True,
                        )
                        if len(port_peaks):
                            st.success(
                                f"Sur l'ensemble du portefeuille, le cycle le plus visible est "
                                f"**{port_peaks.iloc[0]['Période']}** "
                                f"(~{port_peaks.iloc[0]['Puissance relative']:.0%} de la puissance "
                                "dans la bande analysée)."
                            )

                st.markdown("---")
                st.markdown("### Détail par actif")

                fft_summary = cached_fft_summary(
                    ohlcv_sig,
                    min_period,
                    max_period,
                    3,
                    trend_mode,
                    prices,
                )
                if fft_summary.empty:
                    st.warning("Aucun ticker avec assez de données pour l'analyse FFT.")
                else:
                    fft_summary = add_company_names(
                        fft_summary, ticker_names, show_names=show_company_names, sectors=ticker_sectors
                    )
                    fft_display = rename_columns_for_display(fft_summary, FFT_LABELS)
                    fmt_cols = format_map_for_labeled_columns(
                        fft_display, FFT_LABELS, FFT_FORMAT
                    )
                    st.markdown("### Synthèse — périodes dominantes par actif")
                    st.caption(
                        "Périodes en **jours de bourse** (colonne numérique) ou **libellé** "
                        "(ex. « 21 j (~4,2 sem.) ») · poids/force en **%** de la puissance FFT."
                    )
                    st.dataframe(
                        fft_display.style.format(
                            {k: v for k, v in fmt_cols.items() if k in fft_display.columns},
                            na_rep="—",
                        ),
                        use_container_width=True,
                    )

                    fft_ticker = st.selectbox(
                        "Détail FFT par actif",
                        fft_summary["Ticker"],
                        format_func=lambda t: ticker_label(t, ticker_names, show_company_names),
                    )
                    fft_result = cached_fft_periodicity(
                        f"{ohlcv_sig}|{fft_ticker}|{trend_mode}",
                        min_period,
                        max_period,
                        top_peaks,
                        trend_mode,
                        prices[fft_ticker],
                    )
                    if fft_result is None:
                        st.warning("FFT indisponible pour ce titre.")
                    else:
                        detail_label = ticker_label(
                            fft_ticker, ticker_names, show_company_names
                        )
                        peaks = fft_result["peaks"].drop(columns=["_freq_idx"], errors="ignore")
                        _render_fft_headline_metrics(fft_result, peaks)
                        st.plotly_chart(
                            build_fft_spectral_acf_chart(
                                fft_result,
                                title=f"Densité spectrale & autocorrélation — {detail_label}",
                            ),
                            use_container_width=True,
                        )
                        _acf_reading = interpret_fft_acf_reading(fft_result)
                        if _acf_reading:
                            st.info(_acf_reading)
                        peaks_disp = rename_columns_for_display(peaks, FFT_LABELS)
                        st.dataframe(
                            peaks_disp.style.format(
                                format_map_for_labeled_columns(
                                    peaks_disp, FFT_LABELS, FFT_FORMAT
                                )
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )
                        _fft_trend_quality_caption(fft_result)
                        st.plotly_chart(
                            build_fft_cyclic_chart(
                                fft_result,
                                n_components=min(top_peaks, len(fft_result["peaks"])),
                                title=f"Cycle estimé vs prix — {detail_label}",
                            ),
                            use_container_width=True,
                        )
                        if len(peaks):
                            p1 = peaks.iloc[0]["Période"]
                            w1 = peaks.iloc[0]["Puissance relative"]
                            st.info(
                                f"Cycle le plus marqué pour **{detail_label}** : **{p1}** "
                                f"(~{w1:.0%} de la puissance dans la bande analysée). "
                                "À interpréter comme une **tendance cyclique possible**, pas une règle stricte."
                            )

        elif vue == "Synthèse bon marché / cher":
            if HAS_ASSET_SUMMARY:
                asset_summary.render_asset_summary_dashboard(
                    prices,
                    ticker_names=ticker_names,
                    show_company_names=show_company_names,
                    ticker_sectors=ticker_sectors,
                )
            else:
                st.error("Le module 'asset_summary.py' n'a pas pu être chargé.")

        elif vue == "Suggestions d'actifs":
            st.header("💡 Suggestions d'actifs pour améliorer le portefeuille")
            st.caption(
                "Compare votre portefeuille actuel à l'ajout simulé de chaque candidat "
                "(poids configurable). Choisissez le **filtrage par seuils** ou le "
                "**score pondéré** pour la sélection statistique, puis affinez avec "
                "fondamentaux, technique et dividendes."
            )

            if not mode_portfolio_actif:
                st.info(
                    "Mode portefeuille CSV recommandé. En mode indice, les positions actuelles "
                    "sont traitées avec un poids égal entre les composants de l'indice."
                )

            portfolio_tickers = [t for t in liste_tickers if t in returns.columns]
            if len(portfolio_tickers) < 2:
                st.warning("Il faut au moins 2 actifs dans le portefeuille pour générer des suggestions.")
            else:
                index_universe = {}
                if os.path.exists(JSON_PATH):
                    try:
                        with open(JSON_PATH, "r", encoding="utf-8") as f:
                            index_universe = json.load(f)
                    except Exception as e:
                        st.error(f"Impossible de lire tickers.json : {e}")

                if not index_universe:
                    st.error("Fichier tickers.json introuvable ou vide — impossible de proposer des candidats.")
                else:
                    default_universes = [
                        k
                        for k in [
                            "CAC40",
                            "PARIS_HORS_CAC40",
                            "BRUSSELS_HORS_BEL20",
                            "NYSE_EXTRA",
                            "NASDAQ_EXTRA",
                            "RUSSELL2000",
                            "DAX40",
                            "EU_DIVIDEND_STABLE",
                            "EU_DIVIDEND_GROWTH",
                            "NASDAQ100",
                        ]
                        if k in index_universe
                    ] or list(index_universe.keys())[:5]

                    selected_universes = st.multiselect(
                        "Univers de candidats (tickers.json)",
                        list(index_universe.keys()),
                        default=default_universes,
                    )

                    c1, c2, c3 = st.columns(3)
                    candidate_weight = c1.slider("Poids simulé du nouvel actif", 0.05, 0.25, 0.10, 0.01)
                    max_candidates = c2.slider("Nombre max de candidats à tester", 20, 150, 60, 10)
                    top_n = c3.slider("Nombre de suggestions affichées", 5, 30, 15, 1)

                    include_fundamentals = False
                    min_piotroski = 0
                    min_roe_pct = 0.0
                    max_per_filter = None
                    max_debt_equity_filter = None
                    include_technical = True
                    max_rsi_filter = None
                    min_rsi_filter = None
                    max_bollinger_b_filter = None
                    min_bollinger_b_filter = None
                    include_dividends = False
                    dividend_active_only = False
                    min_yield_filter = None
                    max_yield_filter = None
                    min_coverage_filter = None
                    min_cagr_filter = None
                    min_growth_years_filter = None
                    max_payout_filter = None
                    max_fcf_payout_filter = None
                    if HAS_FUNDAMENTALS:
                        with st.expander(
                            "Indicateurs fondamentaux (qualité des entreprises)",
                            expanded=True,
                        ):
                            st.markdown(
                                "Enrichit chaque candidat avec des ratios de **rentabilité**, "
                                "**valorisation**, **endettement** et **consensus analystes**. "
                                "Les données sont **mises en cache sur le disque 24 h** "
                                f"(`{fundamentals.FUNDAMENTALS_DISK_CACHE_FILE}`). "
                                "Le code couleur reprend les repères de la page *Dividendes & Qualité*."
                            )
                            include_fundamentals = st.checkbox(
                                "Inclure les indicateurs fondamentaux",
                                value=True,
                                key="sugg_include_fundamentals",
                            )
                            f1, f2, f3, f4 = st.columns(4)
                            min_piotroski = f1.slider(
                                "Score Piotroski min.",
                                0,
                                9,
                                0,
                                1,
                                help="0 = pas de filtre. ≥ 7 = entreprise financièrement solide.",
                            )
                            min_roe_pct = f2.slider(
                                "ROE min. (%)",
                                0.0,
                                30.0,
                                0.0,
                                1.0,
                                help="0 = pas de filtre. Ex. 15 % = rentabilité des capitaux propres.",
                            ) / 100.0
                            use_max_per = f3.checkbox(
                                "Filtrer PER max.",
                                value=False,
                                key="sugg_use_max_per",
                            )
                            max_per_filter = (
                                f3.number_input(
                                    "PER max.",
                                    min_value=1.0,
                                    max_value=80.0,
                                    value=25.0,
                                    step=1.0,
                                    disabled=not use_max_per,
                                )
                                if use_max_per
                                else None
                            )
                            use_max_debt = f4.checkbox(
                                "Filtrer Dette / Capitaux max.",
                                value=False,
                                key="sugg_use_max_debt",
                            )
                            max_debt_equity_filter = (
                                f4.number_input(
                                    "Dette / Capitaux max.",
                                    min_value=0.0,
                                    max_value=5.0,
                                    value=1.5,
                                    step=0.1,
                                    disabled=not use_max_debt,
                                )
                                if use_max_debt
                                else None
                            )
                    elif not HAS_FUNDAMENTALS:
                        st.caption(
                            "Module `fundamentals.py` indisponible — indicateurs fondamentaux désactivés."
                        )

                    with st.expander(
                        "Indicateurs techniques (RSI, Bollinger, stochastique)",
                        expanded=True,
                    ):
                        st.markdown(
                            "Calculés à partir des **cours déjà téléchargés** (sans appel API "
                            "supplémentaire). **Vert** = zone basse (survente / bas de bande) · "
                            "**Rouge** = zone haute (surachat / haut de bande)."
                        )
                        include_technical = st.checkbox(
                            "Inclure les indicateurs techniques",
                            value=True,
                            key="sugg_include_technical",
                        )
                        t1, t2, t3, t4 = st.columns(4)
                        use_max_rsi = t1.checkbox(
                            "Filtrer RSI max.",
                            value=False,
                            key="sugg_use_max_rsi",
                            help="Ex. 65 = exclure les titres en surachat.",
                        )
                        max_rsi_filter = (
                            t1.slider(
                                "RSI max.",
                                30,
                                90,
                                65,
                                1,
                                disabled=not use_max_rsi,
                            )
                            if use_max_rsi
                            else None
                        )
                        use_min_rsi = t2.checkbox(
                            "Filtrer RSI min.",
                            value=False,
                            key="sugg_use_min_rsi",
                            help="Ex. 35 = ne garder que les titres en zone de survente.",
                        )
                        min_rsi_filter = (
                            t2.slider(
                                "RSI min.",
                                10,
                                50,
                                35,
                                1,
                                disabled=not use_min_rsi,
                            )
                            if use_min_rsi
                            else None
                        )
                        use_max_bb = t3.checkbox(
                            "Filtrer Bollinger %B max.",
                            value=False,
                            key="sugg_use_max_bb",
                            help="Ex. 0,8 = exclure les titres proches de la bande haute.",
                        )
                        max_bollinger_b_filter = (
                            t3.slider(
                                "Bollinger %B max.",
                                0.0,
                                1.0,
                                0.8,
                                0.05,
                                disabled=not use_max_bb,
                            )
                            if use_max_bb
                            else None
                        )
                        use_min_bb = t4.checkbox(
                            "Filtrer Bollinger %B min.",
                            value=False,
                            key="sugg_use_min_bb",
                            help="Ex. 0,2 = ne garder que les titres proches de la bande basse.",
                        )
                        min_bollinger_b_filter = (
                            t4.slider(
                                "Bollinger %B min.",
                                0.0,
                                1.0,
                                0.2,
                                0.05,
                                disabled=not use_min_bb,
                            )
                            if use_min_bb
                            else None
                        )

                    if HAS_DIVIDENDES:
                        with st.expander(
                            "Indicateurs dividendes (rendement, couverture, croissance)",
                            expanded=True,
                        ):
                            st.markdown(
                                "Données **Yahoo Finance** (dividendes TTM, payout, croissance). "
                                "Cache disque **24 h** "
                                f"(`{dividendes.DIVIDENDS_DISK_CACHE_FILE}`). "
                                "Le **ratio de couverture** = bénéfices ÷ dividendes (> 1 = dividende "
                                "couvert). Code couleur aligné sur la page *Dividendes & Qualité*."
                            )
                            include_dividends = st.checkbox(
                                "Inclure les indicateurs dividendes",
                                value=True,
                                key="sugg_include_dividends",
                            )
                            dividend_active_only = st.checkbox(
                                "Dividendes actifs uniquement",
                                value=False,
                                key="sugg_dividend_active_only",
                                help="Exclut les titres sans versement sur les 12 derniers mois.",
                            )
                            d1, d2, d3, d4 = st.columns(4)
                            use_min_yield = d1.checkbox(
                                "Filtrer rendement min.",
                                value=False,
                                key="sugg_use_min_yield",
                            )
                            min_yield_filter = (
                                d1.slider(
                                    "Rendement min. (%)",
                                    0.0,
                                    12.0,
                                    2.0,
                                    0.25,
                                    disabled=not use_min_yield,
                                )
                                / 100.0
                                if use_min_yield
                                else None
                            )
                            use_min_coverage = d2.checkbox(
                                "Filtrer couverture min.",
                                value=False,
                                key="sugg_use_min_coverage",
                            )
                            min_coverage_filter = (
                                d2.number_input(
                                    "Ratio couverture min. (×)",
                                    min_value=0.5,
                                    max_value=5.0,
                                    value=1.2,
                                    step=0.1,
                                    disabled=not use_min_coverage,
                                    help="Ex. 1,5 = le bénéfice couvre 1,5× le dividende versé.",
                                )
                                if use_min_coverage
                                else None
                            )
                            use_min_cagr = d3.checkbox(
                                "Filtrer croissance min.",
                                value=False,
                                key="sugg_use_min_cagr",
                            )
                            min_cagr_filter = (
                                d3.slider(
                                    "CAGR dividende 5 ans min. (%)",
                                    -10.0,
                                    20.0,
                                    0.0,
                                    0.5,
                                    disabled=not use_min_cagr,
                                )
                                / 100.0
                                if use_min_cagr
                                else None
                            )
                            use_min_growth_years = d4.checkbox(
                                "Filtrer années croissance min.",
                                value=False,
                                key="sugg_use_min_growth_years",
                            )
                            min_growth_years_filter = (
                                d4.slider(
                                    "Années de croissance min.",
                                    0,
                                    20,
                                    3,
                                    1,
                                    disabled=not use_min_growth_years,
                                )
                                if use_min_growth_years
                                else None
                            )
                            d5, d6, d7 = st.columns(3)
                            use_max_payout = d5.checkbox(
                                "Filtrer payout max.",
                                value=False,
                                key="sugg_use_max_payout",
                            )
                            max_payout_filter = (
                                d5.slider(
                                    "Payout max. (%)",
                                    20.0,
                                    120.0,
                                    80.0,
                                    5.0,
                                    disabled=not use_max_payout,
                                )
                                / 100.0
                                if use_max_payout
                                else None
                            )
                            use_max_fcf_payout = d6.checkbox(
                                "Filtrer FCF payout max.",
                                value=False,
                                key="sugg_use_max_fcf_payout",
                            )
                            max_fcf_payout_filter = (
                                d6.slider(
                                    "FCF payout max. (%)",
                                    20.0,
                                    150.0,
                                    80.0,
                                    5.0,
                                    disabled=not use_max_fcf_payout,
                                )
                                / 100.0
                                if use_max_fcf_payout
                                else None
                            )
                            use_max_yield = d7.checkbox(
                                "Filtrer rendement max.",
                                value=False,
                                key="sugg_use_max_yield",
                            )
                            max_yield_filter = (
                                d7.slider(
                                    "Rendement max. (%)",
                                    2.0,
                                    15.0,
                                    8.0,
                                    0.25,
                                    disabled=not use_max_yield,
                                    help="Exclut les rendements « trop beaux » (risque de trappe).",
                                )
                                / 100.0
                                if use_max_yield
                                else None
                            )
                    elif not HAS_DIVIDENDES:
                        st.caption(
                            "Module `dividendes.py` indisponible — filtres dividendes désactivés."
                        )

                    selection_mode = st.radio(
                        "Mode de sélection statistique",
                        ["Filtrage par seuils", "Score pondéré"],
                        horizontal=True,
                        key="sugg_selection_mode",
                        help=(
                            "**Filtrage** : exclut les candidats hors de vos critères (comme les "
                            "filtres fondamentaux). **Score pondéré** : classe par un score "
                            "combiné réglé avec les curseurs."
                        ),
                    )

                    min_candidate_return = None
                    max_candidate_vol = None
                    max_candidate_kurtosis = None
                    min_candidate_skewness = None
                    max_corr_portfolio = None
                    min_delta_return = None
                    min_delta_vol = None
                    min_delta_kurtosis = None
                    min_delta_skewness = None
                    min_delta_corr_internal = None

                    if selection_mode == "Filtrage par seuils":
                        with st.expander(
                            "Filtres statistiques (risque, rendement, diversification)",
                            expanded=True,
                        ):
                            st.markdown(
                                "Exclut les candidats qui ne respectent pas vos seuils. "
                                "Les candidats restants sont **triés par score équilibré**. "
                                "Δ = impact simulé sur le portefeuille si vous ajoutez le titre "
                                f"avec le poids réglé ci-dessus ({candidate_weight:.0%})."
                            )
                            st.markdown("**Propriétés du candidat**")
                            s1, s2, s3, s4, s5 = st.columns(5)
                            if s1.checkbox(
                                "Rendement candidat min.",
                                key="sugg_use_min_cand_return",
                            ):
                                min_candidate_return = (
                                    s1.slider(
                                        "Rendement min. (%)",
                                        -20.0,
                                        40.0,
                                        5.0,
                                        1.0,
                                        key="sugg_min_cand_return",
                                    )
                                    / 100.0
                                )
                            if s2.checkbox(
                                "Volatilité candidat max.",
                                key="sugg_use_max_cand_vol",
                            ):
                                max_candidate_vol = (
                                    s2.slider(
                                        "Volatilité max. (%)",
                                        5.0,
                                        80.0,
                                        25.0,
                                        1.0,
                                        key="sugg_max_cand_vol",
                                    )
                                    / 100.0
                                )
                            if s3.checkbox(
                                "Kurtosis candidat max.",
                                key="sugg_use_max_cand_kurt",
                            ):
                                max_candidate_kurtosis = s3.slider(
                                    "Kurtosis max.",
                                    0.0,
                                    15.0,
                                    5.0,
                                    0.5,
                                    key="sugg_max_cand_kurt",
                                )
                            if s4.checkbox(
                                "Skewness candidat min.",
                                key="sugg_use_min_cand_skew",
                            ):
                                min_candidate_skewness = s4.slider(
                                    "Skewness min.",
                                    -2.0,
                                    2.0,
                                    0.0,
                                    0.1,
                                    key="sugg_min_cand_skew",
                                )
                            if s5.checkbox(
                                "Corrélation portef. max.",
                                key="sugg_use_max_corr_cand",
                            ):
                                max_corr_portfolio = s5.slider(
                                    "Corr. moy. max. (0–1)",
                                    0.0,
                                    1.0,
                                    0.65,
                                    0.05,
                                    key="sugg_max_corr_cand",
                                )
                            st.markdown("**Impact simulé sur le portefeuille**")
                            d1, d2, d3, d4, d5 = st.columns(5)
                            if d1.checkbox(
                                "Δ Rendement min.",
                                key="sugg_use_min_delta_return",
                            ):
                                min_delta_return = (
                                    d1.slider(
                                        "Δ Rendement min. (pts %)",
                                        -5.0,
                                        10.0,
                                        0.0,
                                        0.5,
                                        key="sugg_min_delta_return",
                                    )
                                    / 100.0
                                )
                            if d2.checkbox(
                                "Δ Volatilité min.",
                                key="sugg_use_min_delta_vol",
                                help="Positif = le portefeuille devient moins volatile.",
                            ):
                                min_delta_vol = (
                                    d2.slider(
                                        "Δ Volatilité min. (pts %)",
                                        -10.0,
                                        10.0,
                                        0.0,
                                        0.5,
                                        key="sugg_min_delta_vol",
                                    )
                                    / 100.0
                                )
                            if d3.checkbox(
                                "Δ Kurtosis min.",
                                key="sugg_use_min_delta_kurt",
                                help="Positif = moins de risque de chutes brutales.",
                            ):
                                min_delta_kurtosis = d3.slider(
                                    "Δ Kurtosis min.",
                                    -5.0,
                                    5.0,
                                    0.0,
                                    0.25,
                                    key="sugg_min_delta_kurt",
                                )
                            if d4.checkbox(
                                "Δ Skewness min.",
                                key="sugg_use_min_delta_skew",
                            ):
                                min_delta_skewness = d4.slider(
                                    "Δ Skewness min.",
                                    -2.0,
                                    2.0,
                                    0.0,
                                    0.1,
                                    key="sugg_min_delta_skew",
                                )
                            if d5.checkbox(
                                "Δ Corr. interne min.",
                                key="sugg_use_min_delta_corr_int",
                                help="Positif = les titres du portefeuille divergent davantage.",
                            ):
                                min_delta_corr_internal = d5.slider(
                                    "Δ Corr. interne min.",
                                    -0.5,
                                    0.5,
                                    0.0,
                                    0.05,
                                    key="sugg_min_delta_corr_int",
                                )
                        objective_weights = dict(DEFAULT_SUGGESTION_OBJECTIVE_WEIGHTS)
                    else:
                        with st.expander("Poids des objectifs du score"):
                            st.markdown(
                                "Réglez ce qui compte **le plus** pour vous. Chaque curseur va de **0** "
                                "(on s'en fiche) à **2** (c'est prioritaire). "
                                "**À gauche** : si vous mettez peu ; **à droite** : si vous mettez beaucoup."
                            )
                            w_return = _objective_weight_slider(
                                "Rendement (gains)",
                                "On cherche surtout à **limiter le risque**, pas à maximiser les gains.",
                                "On privilégie les titres qui **rapportent le plus** — en acceptant "
                                "souvent plus de risque.",
                                1.0,
                                "sugg_w_return",
                                tech_name="rendement ↑",
                            )
                            w_vol = _objective_weight_slider(
                                "Stabilité (moins de variations)",
                                "On accepte des titres qui **montent et descendent fort** d'un jour "
                                "à l'autre.",
                                "On préfère un portefeuille **plus calme**, avec moins de surprises "
                                "— souvent un peu moins rentable.",
                                1.0,
                                "sugg_w_vol",
                                tech_name="volatilité ↓",
                            )
                            w_kurt = _objective_weight_slider(
                                "Éviter les chutes brutales",
                                "On tolère des titres qui peuvent **perdre beaucoup d'un coup** "
                                "(gros revers en une séance).",
                                "On évite les titres où les **grosses chutes** arrivent souvent.",
                                0.7,
                                "sugg_w_kurt",
                                tech_name="kurtosis ↓",
                            )
                            w_skew = _objective_weight_slider(
                                "Potentiel de grosses hausses",
                                "Peu importe si le titre peut parfois **monter très vite**.",
                                "On favorise les titres qui ont parfois de **très belles journées** "
                                "à la hausse.",
                                0.8,
                                "sugg_w_skew",
                                tech_name="skewness ↑",
                            )
                            w_corr_int = _objective_weight_slider(
                                "Vos titres ne bougent pas tous pareil",
                                "Ça ne gêne pas si **plusieurs lignes de votre portefeuille** "
                                "montent ou baissent en même temps.",
                                "On veut que vos titres actuels **ne réagissent pas tous pareil** "
                                "— moins de mauvaises surprises groupées.",
                                1.0,
                                "sugg_w_corr_int",
                                tech_name="corrélation interne ↓",
                            )
                            w_corr_cand = _objective_weight_slider(
                                "Nouveau titre différent du reste",
                                "Le candidat peut ressembler à ce que vous avez **déjà** "
                                "(même type d'entreprise, même région).",
                                "On cherche un titre qui **ne suit pas** le reste de votre "
                                "portefeuille — pour mieux répartir le risque.",
                                1.0,
                                "sugg_w_corr_cand",
                                tech_name="corrélation au portefeuille ↓",
                            )

                        objective_weights = {
                            "return": w_return,
                            "vol": w_vol,
                            "kurt": w_kurt,
                            "skew": w_skew,
                            "corr_internal": w_corr_int,
                            "corr_candidate": w_corr_cand,
                        }

                    run_suggestions = st.button("Générer les suggestions", type="primary")

                    if run_suggestions:
                        candidate_pool = []
                        for universe in selected_universes:
                            candidate_pool.extend(index_universe.get(universe, []))
                        candidate_pool = [
                            t for t in dict.fromkeys(candidate_pool)
                            if t not in portfolio_tickers
                        ][:max_candidates]

                        if not candidate_pool:
                            st.warning("Aucun candidat disponible dans les univers sélectionnés.")
                        else:
                            with st.spinner(
                                f"Téléchargement et analyse de {len(candidate_pool)} candidats..."
                            ):
                                candidate_sig = _tickers_signature(candidate_pool, start_date)
                                candidate_prices, candidate_returns = cached_candidate_market_data(
                                    candidate_sig,
                                    tuple(candidate_pool),
                                    str(start_date),
                                )

                            suggestions, baseline, baseline_internal = cached_suggest_portfolio_additions(
                                ohlcv_sig,
                                _tickers_signature(candidate_pool, start_date),
                                returns,
                                tuple(portfolio_tickers),
                                candidate_returns,
                                candidate_weight,
                                tuple(sorted(objective_weights.items())),
                            )

                            if (
                                include_fundamentals
                                and HAS_FUNDAMENTALS
                                and not suggestions.empty
                            ):
                                cache_key = "suggestions_fundamentals_cache"
                                fund_cache = st.session_state.setdefault(cache_key, {})
                                status_fund = st.empty()
                                progress_fund = st.progress(0)

                                def _fund_progress(ticker, current, total, stats=None):
                                    stats = stats or {}
                                    if stats.get("fetched", 0) == 0 and (
                                        stats.get("disk", 0) + stats.get("session", 0)
                                    ) > 0:
                                        label = "Fondamentaux (cache)"
                                    else:
                                        label = "Fondamentaux"
                                    status_fund.text(
                                        f"{label} : {ticker} ({current}/{total})"
                                    )
                                    progress_fund.progress(current / total)

                                fund_rows, fund_cache, fund_stats = (
                                    fundamentals.fetch_fundamentals_for_tickers(
                                        suggestions["Ticker"].tolist(),
                                        candidate_prices,
                                        cache=fund_cache,
                                        progress_cb=_fund_progress,
                                    )
                                )
                                st.session_state[cache_key] = fund_cache
                                suggestions = fundamentals.merge_fundamentals_columns(
                                    suggestions, fund_rows
                                )
                                status_fund.empty()
                                progress_fund.empty()
                                st.caption(
                                    "Fondamentaux — "
                                    f"**{fund_stats['disk']}** depuis le disque · "
                                    f"**{fund_stats['session']}** en session · "
                                    f"**{fund_stats['fetched']}** téléchargés · "
                                    f"**{fund_stats['missing']}** sans données. "
                                    f"Cache : `{fundamentals._fundamentals_cache_path()}` (24 h)."
                                )

                            if include_technical and not suggestions.empty:
                                suggestions = enrich_suggestions_with_technical(
                                    suggestions, candidate_prices
                                )

                            if (
                                include_dividends
                                and HAS_DIVIDENDES
                                and not suggestions.empty
                            ):
                                cache_key = "suggestions_dividends_cache"
                                div_cache = st.session_state.setdefault(cache_key, {})
                                status_div = st.empty()
                                progress_div = st.progress(0)

                                def _div_progress(ticker, current, total, stats=None):
                                    stats = stats or {}
                                    if stats.get("fetched", 0) == 0 and (
                                        stats.get("disk", 0) + stats.get("session", 0)
                                    ) > 0:
                                        label = "Dividendes (cache)"
                                    else:
                                        label = "Dividendes"
                                    status_div.text(
                                        f"{label} : {ticker} ({current}/{total})"
                                    )
                                    progress_div.progress(current / total)

                                div_rows, div_cache, div_stats = (
                                    dividendes.fetch_dividends_for_tickers(
                                        suggestions["Ticker"].tolist(),
                                        candidate_prices,
                                        cache=div_cache,
                                        progress_cb=_div_progress,
                                    )
                                )
                                st.session_state[cache_key] = div_cache
                                suggestions = dividendes.merge_dividend_columns(
                                    suggestions, div_rows
                                )
                                status_div.empty()
                                progress_div.empty()
                                st.caption(
                                    "Dividendes — "
                                    f"**{div_stats['disk']}** depuis le disque · "
                                    f"**{div_stats['session']}** en session · "
                                    f"**{div_stats['fetched']}** téléchargés · "
                                    f"**{div_stats['missing']}** sans données. "
                                    f"Cache : `{dividendes._dividends_cache_path()}` (24 h)."
                                )

                            st.session_state["portfolio_suggestions"] = suggestions
                            st.session_state["portfolio_suggestions_baseline"] = baseline
                            st.session_state["portfolio_suggestions_internal_corr"] = baseline_internal
                            st.session_state["portfolio_suggestions_meta"] = {
                                "pool_size": len(candidate_pool),
                                "weight": candidate_weight,
                                "weights": dict(objective_weights),
                                "include_fundamentals": include_fundamentals,
                                "include_technical": include_technical,
                                "include_dividends": include_dividends,
                                "selection_mode": selection_mode,
                            }

                    if "portfolio_suggestions_baseline" in st.session_state:
                        baseline = st.session_state["portfolio_suggestions_baseline"]
                        baseline_internal = st.session_state.get(
                            "portfolio_suggestions_internal_corr", np.nan
                        )
                        meta = st.session_state.get("portfolio_suggestions_meta", {})
                        st.markdown("### Référence portefeuille actuel")
                        if baseline:
                            b1, b2, b3, b4 = st.columns(4)
                            b1.metric("Rendement annuel", f"{baseline['Rendement Annuel']:.2%}")
                            b2.metric("Volatilité", f"{baseline['Volatilité (Sigma)']:.2%}")
                            b3.metric("Skewness", f"{baseline['Skewness (Asymétrie)']:.2f}")
                            b4.metric("Corr. interne moy.", f"{baseline_internal:.2f}")
                        if meta:
                            st.caption(
                                f"Candidats testés : {meta.get('pool_size', '?')} | "
                                f"Poids simulé : {meta.get('weight', 0.1):.0%}"
                            )

                    suggestions = st.session_state.get("portfolio_suggestions")
                    if suggestions is not None and not suggestions.empty:
                        suggestions_view = suggestions
                        filter_parts = []

                        if selection_mode == "Filtrage par seuils":
                            before_stats = len(suggestions_view)
                            suggestions_view = filter_suggestions_by_statistics(
                                suggestions_view,
                                min_candidate_return=min_candidate_return,
                                max_candidate_vol=max_candidate_vol,
                                max_candidate_kurtosis=max_candidate_kurtosis,
                                min_candidate_skewness=min_candidate_skewness,
                                max_corr_portfolio=max_corr_portfolio,
                                min_delta_return=min_delta_return,
                                min_delta_vol=min_delta_vol,
                                min_delta_kurtosis=min_delta_kurtosis,
                                min_delta_skewness=min_delta_skewness,
                                min_delta_corr_internal=min_delta_corr_internal,
                            )
                            if len(suggestions_view) < before_stats:
                                filter_parts.append("statistiques")

                        if HAS_FUNDAMENTALS and any(
                            c in suggestions.columns
                            for c in fundamentals.SUGGESTION_QUALITY_COLUMNS
                        ):
                            suggestions_view = fundamentals.filter_suggestions_by_quality(
                                suggestions_view,
                                min_piotroski=min_piotroski,
                                min_roe=min_roe_pct,
                                max_per=max_per_filter,
                                max_debt_equity=max_debt_equity_filter,
                            )
                            if len(suggestions_view) < len(suggestions):
                                filter_parts.append("fondamentaux")

                        if any(
                            c in suggestions_view.columns
                            for c in SUGGESTION_TECHNICAL_COLUMNS
                        ):
                            before_technical = len(suggestions_view)
                            suggestions_view = filter_suggestions_by_technical(
                                suggestions_view,
                                max_rsi=max_rsi_filter,
                                min_rsi=min_rsi_filter,
                                max_bollinger_b=max_bollinger_b_filter,
                                min_bollinger_b=min_bollinger_b_filter,
                            )
                            if len(suggestions_view) < before_technical:
                                filter_parts.append("techniques")

                        if HAS_DIVIDENDES and any(
                            c in suggestions_view.columns
                            for c in dividendes.SUGGESTION_DIVIDEND_COLUMNS
                        ):
                            before_dividends = len(suggestions_view)
                            suggestions_view = dividendes.filter_suggestions_by_dividends(
                                suggestions_view,
                                min_yield=min_yield_filter,
                                max_yield=max_yield_filter,
                                min_coverage=min_coverage_filter,
                                min_cagr_5y=min_cagr_filter,
                                min_growth_years=min_growth_years_filter,
                                max_payout=max_payout_filter,
                                max_fcf_payout=max_fcf_payout_filter,
                                active_only=dividend_active_only,
                            )
                            if len(suggestions_view) < before_dividends:
                                filter_parts.append("dividendes")

                        if filter_parts:
                            st.caption(
                                f"Filtre {' + '.join(filter_parts)} : "
                                f"**{len(suggestions_view)}** / {len(suggestions)} "
                                "candidats conservés."
                            )
                        if suggestions_view.empty:
                            st.warning(
                                "Aucun candidat ne passe les filtres actifs. "
                                "Assouplissez les seuils ou regénérez les suggestions."
                            )

                        if selection_mode == "Filtrage par seuils":
                            st.markdown("**Mode : filtrage par seuils statistiques**")
                            st.caption(
                                "Les candidats affichés passent vos filtres actifs et sont triés "
                                "par score équilibré (poids identiques sur rendement, volatilité, "
                                "kurtosis, skewness et corrélations)."
                            )
                        else:
                            profile_scores = compute_profile_scores(objective_weights)
                            st.markdown(f"**Profil de sélection : {profile_title}**")
                            st.caption(profile_desc)
                            with st.expander("Détail du profil (scores)", expanded=False):
                                score_df = pd.DataFrame(
                                    [
                                        {"Profil": name, "Score": score}
                                        for name, score in sorted(
                                            profile_scores.items(),
                                            key=lambda item: -item[1],
                                        )
                                    ]
                                )
                                st.dataframe(
                                    score_df.style.format({"Score": "{:.0%}"}),
                                    hide_index=True,
                                    use_container_width=True,
                                )
                                if " · " in profile_title:
                                    st.caption(
                                        "Deux profils affichés car leurs scores sont proches "
                                        "(priorités combinées)."
                                    )
                        if (
                            selection_mode == "Score pondéré"
                            and st.session_state.get("portfolio_suggestions_meta", {}).get(
                                "weights"
                            )
                            != objective_weights
                        ):
                            st.caption(
                                "ℹ️ Les curseurs ont changé depuis la dernière génération — "
                                "cliquez **Générer les suggestions** pour recalculer avec ce profil."
                            )
                        elif (
                            selection_mode == "Filtrage par seuils"
                            and st.session_state.get("portfolio_suggestions_meta", {}).get(
                                "selection_mode"
                            )
                            != selection_mode
                        ):
                            st.caption(
                                "ℹ️ Le mode de sélection a changé — regénérez si besoin."
                            )

                        if suggestions_view.empty:
                            pass
                        else:
                            display_df = suggestions_view.head(top_n).copy()
                            if "Ticker" in display_df.columns and (
                                show_company_names or need_ticker_sectors
                            ):
                                sugg_names, sugg_sectors = get_ticker_metadata(
                                    display_df["Ticker"].tolist(),
                                    need_names=show_company_names,
                                    need_sectors=need_ticker_sectors,
                                )
                                display_df = add_company_names(
                                    display_df,
                                    sugg_names,
                                    show_names=show_company_names,
                                    sectors=sugg_sectors if need_ticker_sectors else None,
                                )
                            elif "Ticker" in display_df.columns:
                                display_df = add_company_names(
                                    display_df,
                                    {},
                                    show_names=False,
                                    sectors=None,
                                )
                            display_view = rename_columns_for_display(
                                display_df, SUGGESTIONS_LABELS
                            )
                            sugg_format = format_map_for_labeled_columns(
                                display_view, SUGGESTIONS_LABELS, SUGGESTIONS_FORMAT
                            )
                            st.caption(
                                "Rendements et volatilités en **%** · corrélations sur **0–1** · "
                                "Δ = impact simulé sur le portefeuille."
                            )
                            if HAS_FUNDAMENTALS and any(
                                c in display_df.columns
                                for c in fundamentals.SUGGESTION_QUALITY_COLUMNS
                            ):
                                st.caption(
                                    "Indicateurs fondamentaux : **vert** = repère favorable · "
                                    "**rouge** = repère défavorable (voir page *Dividendes & Qualité*)."
                                )
                            if any(
                                c in display_df.columns
                                for c in SUGGESTION_TECHNICAL_COLUMNS
                            ):
                                st.caption(
                                    "Indicateurs techniques : **vert** = survente / bas de bande · "
                                    "**rouge** = surachat / haut de bande."
                                )
                            if HAS_DIVIDENDES and any(
                                c in display_df.columns
                                for c in dividendes.SUGGESTION_DIVIDEND_COLUMNS
                            ):
                                st.caption(
                                    "Indicateurs dividendes : **vert** = rendement / croissance / "
                                    "couverture favorables · **rouge** = signaux faibles."
                                )
                            styled_sugg = display_view.style.format(sugg_format, na_rep="-")
                            grad_pos = pick_existing_columns(
                                display_view,
                                "Score composite (pts)",
                                "Δ Rendement portef. (%)",
                            )
                            if grad_pos:
                                styled_sugg = styled_sugg.background_gradient(
                                    cmap="RdYlGn", subset=grad_pos
                                )
                            grad_neg = pick_existing_columns(
                                display_view,
                                "Δ Volatilité portef. (%)",
                                "Δ Kurtosis portef.",
                                "Δ Corr. interne (0–1)",
                            )
                            if grad_neg:
                                styled_sugg = styled_sugg.background_gradient(
                                    cmap="RdYlGn", subset=grad_neg
                                )
                            if HAS_FUNDAMENTALS:
                                styled_sugg = fundamentals.style_fundamentals_sentiment(
                                    styled_sugg
                                )
                            if any(
                                c in display_df.columns
                                for c in SUGGESTION_TECHNICAL_COLUMNS
                            ):
                                styled_sugg = style_suggestions_technical(styled_sugg)
                            if HAS_DIVIDENDES and any(
                                c in display_df.columns
                                for c in dividendes.SUGGESTION_DIVIDEND_COLUMNS
                            ):
                                styled_sugg = dividendes.style_dividend_sentiment(styled_sugg)
                            st.dataframe(styled_sugg)

                            st.markdown("### Visualisation des compromis")
                            plot_df = suggestions_view.head(min(50, len(suggestions_view))).copy()
                            if show_company_names and "Ticker" in plot_df.columns:
                                plot_names, _ = get_ticker_metadata(
                                    plot_df["Ticker"].tolist(),
                                    need_names=True,
                                    need_sectors=False,
                                )
                                plot_df["Libellé"] = plot_df["Ticker"].map(
                                    lambda t: ticker_label(t, plot_names, True)
                                )
                            score_min = plot_df["Score composite"].min()
                            plot_df["Taille score"] = plot_df["Score composite"] - score_min + 0.01
                            fig_sugg = px.scatter(
                                plot_df,
                                x="Corr. moy. portefeuille",
                                y="Rendement candidat",
                                size="Taille score",
                                color="Score composite",
                                hover_name="Libellé" if show_company_names and "Libellé" in plot_df.columns else "Ticker",
                                color_continuous_scale="RdYlGn",
                                title="Rendement vs diversification (corrélation au portefeuille)",
                            )
                            fig_sugg.update_yaxes(tickformat=".0%")
                            fig_sugg.update_layout(coloraxis_colorbar_title="Score composite")
                            st.plotly_chart(fig_sugg, use_container_width=True)

                            csv_sugg = suggestions_view.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                "Télécharger les suggestions (CSV)",
                                csv_sugg,
                                "suggestions_actifs.csv",
                            )
                    elif run_suggestions:
                        st.warning("Aucune suggestion pertinente n'a pu être calculée.")


render_live_dashboard()
