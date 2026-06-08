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

import suggestions_job
from analytics import (
    build_portfolio_value,
    build_portfolio_weight_pct_history,
    build_portfolio_weight_history_chart,
    build_portfolio_allocation_treemap,
    build_portfolio_allocation_pie_figure,
    build_portfolio_drawdown_figure,
    build_risk_return_scatter_figure,
    build_returns_distribution_var_figure,
    build_suggestions_tradeoff_scatter,
    build_technical_overview_figure,
    build_technical_stochastic_figure,
    build_technical_macd_figure,
    build_portfolio_price_for_fft,
    build_mean_median_mode_pedagogy_chart,
    build_fft_spectral_acf_chart,
    interpret_fft_acf_reading,
    build_fft_cyclic_chart,
    compute_fft_model_correlations,
    summarize_fft_trend_quality,
    interpret_fft_trend_r2,
    format_fft_rmse_display,
    FFT_TREND_OPTIONS,
    FFT_TREND_LOG_LINEAR,
    FFT_RECON_OPTIONS,
    FFT_RECON_FFT,
    FFT_PRICE_UNIT_EUR,
    FFT_PRICE_UNIT_PORTFOLIO,
    FFT_PRICE_UNIT_INDEX,
    fft_trend_mode_label,
    fft_recon_mode_label,
    build_risk_returns_boxplot,
    build_risk_metrics_boxplot,
    build_correlation_heatmap_figure,
    cluster_correlation_order,
    compute_bollinger,
    compute_fibonacci_levels,
    compute_linear_regression_channel,
    build_regression_channel_figure,
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
    interpret_supertrend_comment,
    interpret_support_resistance_comment,
    interpret_volume_comment,
    describe_suggestion_profile,
    compute_profile_scores,
    DEFAULT_SUGGESTION_OBJECTIVE_WEIGHTS,
    filter_suggestions_by_statistics,
    filter_suggestions_by_technical,
    style_suggestions_technical,
    style_technical_table,
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
    cached_usd_to_eur_series,
    cached_compute_returns,
    cached_technical_indicators,
    cached_advanced_risk_metrics,
    cached_corr_matrix,
    cached_markowitz_weights,
    cached_markowitz_frontier_figure,
    cached_fft_periodicity_ticker,
    cached_fft_periodicity_portfolio,
    cached_fft_summary,
    cached_benchmark_prices,
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


def _render_fft_headline_metrics(
    result,
    peaks,
    n_components=3,
    price_unit=FFT_PRICE_UNIT_EUR,
    recon_mode=FFT_RECON_FFT,
    recency_weighted=True,
    recency_calibrate=True,
):
    """Bandeau : cycles, R² tendance et corrélations cours / modèle."""
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
    rmse_pct = result.get("trend_rmse_pct") if result else None
    if r2 is not None and not (isinstance(r2, float) and np.isnan(r2)):
        reliability = interpret_fft_trend_r2(r2)
        abs_rmse, pct_rmse = format_fft_rmse_display(rmse, rmse_pct, price_unit)
        r2_help = (
            "Qualité de la tendance retirée avant FFT, mesurée sur le prix réel. "
            "R² = part de la variance du cours expliquée par la tendance seule. "
            "1 = parfait · ≥ 0,85 = excellente · 0,65–0,85 = bonne · "
            "0,50–0,65 = modérée · 0,30–0,50 = faible · < 0,30 = très faible."
        )
        if abs_rmse != "—":
            r2_help += f" RMSE = {abs_rmse}"
            if pct_rmse != "—":
                r2_help += f" ({pct_rmse} du prix moyen)."
        r2_help += f" {reliability['detail']}"
        c4.metric(
            "R² tendance",
            f"{r2:.3f}",
            delta=f"Fiabilité {reliability['label'].lower()}",
            delta_color="off",
            help=r2_help,
        )
    else:
        c4.metric("R² tendance", "—", help="Non calculable sur cette série.")

    peak_df = result.get("peaks") if result else None
    if peak_df is not None and len(peak_df):
        indices = peak_df.head(n_components)["_freq_idx"].tolist()
        corrs = compute_fft_model_correlations(
            result,
            indices,
            n_components=n_components,
            recency_weighted=recency_weighted,
            recency_calibrate=recency_calibrate,
        )
    else:
        corrs = None

    has_smooth = result is not None and result.get("smooth_prices") is not None
    smooth_label = "MM41" if result and result.get("trend_mode", "").startswith("ma41") else "lissage"

    c5, c6, c7, c8 = st.columns(4)
    if corrs:
        r_fft = corrs.get("corr_with_trend_filter")
        r_harm = corrs.get("corr_harmonic")
        r_smooth = corrs.get("corr_vs_smooth")
        r_harm_smooth = corrs.get("corr_harmonic_vs_smooth")

        if r_fft is not None and not np.isnan(r_fft):
            c5.metric(
                "Corr. modèle FFT vs cours",
                f"{r_fft:.3f}",
                help=(
                    "Corrélation de Pearson entre le cours réel et le modèle "
                    "tendance + cycles (pics FFT), sur l'historique."
                ),
            )
        else:
            c5.metric("Corr. modèle FFT vs cours", "—")

        if r_harm is not None and not np.isnan(r_harm):
            c6.metric(
                "Corr. harmonique vs cours",
                f"{r_harm:.3f}",
                help=(
                    "Corrélation entre le cours et une régression sin/cos "
                    "aux périodes dominantes détectées par FFT."
                ),
            )
        else:
            c6.metric("Corr. harmonique vs cours", "—")

        if has_smooth and r_smooth is not None and not np.isnan(r_smooth):
            c7.metric(
                f"Corr. FFT vs {smooth_label}",
                f"{r_smooth:.3f}",
                help=(
                    f"Fidélité du modèle FFT à la série lissée ({smooth_label}), "
                    "sans le bruit haute fréquence du cours brut."
                ),
            )
        elif has_smooth and r_harm_smooth is not None and not np.isnan(r_harm_smooth):
            c7.metric(
                f"Corr. harmonique vs {smooth_label}",
                f"{r_harm_smooth:.3f}",
                help=f"Qualité de l'ajustement harmonique sur la série {smooth_label}.",
            )
        else:
            r_without = corrs.get("corr_without_trend_filter")
            if r_without is not None and not np.isnan(r_without):
                c7.metric(
                    "Corr. FFT sans dé-trend",
                    f"{r_without:.3f}",
                    help="FFT sur log(prix) sans retirer la tendance (comparaison).",
                )
            else:
                c7.metric("Corr. vs lissage", "—")

        if has_smooth and r_harm_smooth is not None and not np.isnan(r_harm_smooth):
            c8.metric(
                f"Harmonique vs {smooth_label}",
                f"{r_harm_smooth:.3f}",
                help="Régression harmonique comparée à la série lissée.",
            )
        else:
            c8.metric("Méthode graphique", fft_recon_mode_label(recon_mode))
    else:
        c5.metric("Corr. modèle FFT vs cours", "—")
        c6.metric("Corr. harmonique vs cours", "—")
        c7.metric("Corr. vs lissage", "—")
        c8.metric("Méthode graphique", "—")


def _suggestion_range_filter(
    container,
    label,
    min_value,
    max_value,
    default_range,
    step,
    *,
    key_prefix,
    help=None,
    scale=1.0,
):
    """Case à cocher + curseur double borne ; retourne (min, max) ou (None, None)."""
    with container:
        active = st.checkbox(
            f"Filtrer {label}",
            value=False,
            key=f"{key_prefix}_use",
            help=help,
        )
        if not active:
            return None, None
        lo, hi = st.slider(
            f"Plage {label}",
            min_value=min_value,
            max_value=max_value,
            value=default_range,
            step=step,
            key=f"{key_prefix}_range",
        )
        if scale != 1.0:
            return lo * scale, hi * scale
        return lo, hi


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
    "Date de début des historiques",
    (pd.Timestamp.today().normalize() - pd.DateOffset(years=1)).date(),
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
        clear_market_cache(
            _tickers_signature(liste_tickers, start_date),
            tickers=liste_tickers,
            start=start_date,
        )
    else:
        clear_market_cache()
    clear_dashboard_compute_cache()
    st.session_state.pop("ohlcv_sig", None)
    st.sidebar.success("Cache cours vidé.")
run_every = refresh_interval if auto_refresh_enabled and liste_tickers else None


@st.fragment
def _render_synthese_plus_values(prices, returns, ohlcv_sig):
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

            st.markdown("### Répartition du capital")
            alloc_mode = st.radio(
                "Vue allocation",
                options=["Treemap (secteurs)", "Donut"],
                horizontal=True,
                key="synth_alloc_mode",
                label_visibility="collapsed",
            )
            if alloc_mode.startswith("Treemap"):
                fig_alloc = build_portfolio_allocation_treemap(
                    df_synthese,
                    label_col="Nom" if show_company_names and "Nom" in df_synthese.columns else "Ticker",
                )
            else:
                fig_alloc = build_portfolio_allocation_pie_figure(
                    df_synthese,
                    label_col="Nom" if show_company_names and "Nom" in df_synthese.columns else "Ticker",
                )
            if fig_alloc is not None:
                st.plotly_chart(fig_alloc, use_container_width=True)
            st.caption(
                "**Treemap** : hiérarchie par secteur si disponible · **Donut** : vue circulaire classique."
            )

            fx_hist = cached_usd_to_eur_series(str(start_date))
            weight_hist = build_portfolio_weight_pct_history(
                prices,
                quantities,
                usd_to_eur_series=fx_hist,
                usd_to_eur=usd_to_eur,
                is_usd_fn=is_usd_ticker,
            )
            st.markdown("### Évolution des poids dans le portefeuille")
            if weight_hist.empty or len(weight_hist) < 2:
                st.info(
                    "Historique insuffisant pour afficher l'évolution des poids "
                    "(vérifiez la date de début et les cours disponibles)."
                )
            else:
                weight_labels = {
                    row["Ticker"]: (
                        row["Nom"]
                        if show_company_names and "Nom" in row and row["Nom"]
                        else row["Ticker"]
                    )
                    for _, row in df_synthese.iterrows()
                }
                fig_weights = build_portfolio_weight_history_chart(
                    weight_hist,
                    labels=weight_labels,
                    title="Répartition du capital par titre (%)",
                )
                if fig_weights is not None:
                    st.plotly_chart(fig_weights, use_container_width=True)
                st.caption(
                    "Part de chaque ligne en **%** de la valorisation totale (€), "
                    "recalculée chaque jour. Titres en **USD** : conversion au "
                    "**taux EUR/USD historique** du jour (pas le taux actuel). "
                    "Quantités = positions actuelles du CSV (vue rétrospective)."
                )

            port_value = build_portfolio_value(
                prices, quantities, usd_to_eur=usd_to_eur, is_usd_fn=is_usd_ticker
            )
            if port_value is not None and len(port_value.dropna()) >= 2:
                st.markdown("### Drawdown du portefeuille")
                fig_dd = build_portfolio_drawdown_figure(
                    port_value,
                    title="Baisse depuis le dernier pic (valorisation €)",
                )
                if fig_dd is not None:
                    st.plotly_chart(fig_dd, use_container_width=True)
                st.caption(
                    "Drawdown = **(valeur − pic cumulé) / pic**. "
                    "Mesure la baisse maximale vécue sur la période."
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
| **Rendement annuel** | Performance moyenne journalière annualisée (├ù 252). |
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
                        "de profil — m├¬me moteur que les suggestions d'actifs."
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
                    if " ┬À " in profile_title:
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

                st.markdown("### Tendance & canal σ — valorisation portefeuille")
                port_val_clean = port_value.dropna() if port_value is not None else pd.Series(dtype=float)
                if len(port_val_clean) >= 10:
                    fig_reg = build_regression_channel_figure(
                        port_val_clean,
                        scale="log",
                        title="Valorisation portefeuille — régression log & bandes ±1σ / ±2σ",
                        series_name="Portefeuille (€)",
                        yaxis_title="Valorisation (€)",
                    )
                    if fig_reg is not None:
                        st.plotly_chart(fig_reg, use_container_width=True)
                    st.caption(
                        "Échelle **logarithmique** : la droite bleue est une tendance de "
                        "**croissance en %** (régression sur log(valorisation)). "
                        "Pointillés = bandes parallèles à ±**1σ** et ±**2σ** "
                        "(écarts relatifs autour de la tendance). Adapté aux portefeuilles "
                        "dont la valeur évolue de façon composée dans le temps."
                    )
                else:
                    st.info(
                        "Canal de régression indisponible (minimum ~10 séances de valorisation)."
                    )
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

def _resolve_ticker_metadata(
    tickers,
    base_names,
    base_sectors,
    *,
    need_names=True,
    need_sectors=False,
):
    """Réutilise les métadonnées déjà chargées ; ne fetch que les tickers manquants."""
    tickers = [t for t in dict.fromkeys(tickers or []) if t]
    names = dict(base_names or {})
    sectors = dict(base_sectors or {})
    missing = []
    for t in tickers:
        if need_names and t not in names:
            missing.append(t)
        elif need_sectors and t not in sectors:
            missing.append(t)
    missing = sorted(set(missing))
    if missing:
        extra_names, extra_sectors = get_ticker_metadata(
            missing,
            need_names=need_names,
            need_sectors=need_sectors,
        )
        names.update(extra_names)
        sectors.update(extra_sectors)
    return names, sectors


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
                refresh_mode = "auto"
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
                tickers_tuple = tuple(sorted(liste_tickers))
                returns = cached_compute_returns(ohlcv_sig, tickers_tuple, str(start_date))
                if returns.empty:
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

    if suggestions_job.is_active():
        _sugg_job = st.session_state.get("suggestions_job") or {}
        _sugg_phase = _sugg_job.get("phase", "init")
        if _sugg_phase == "core" and (prices.empty or returns.empty):
            _sugg_job["status"] = "error"
            _sugg_job["error"] = (
                "Cours du portefeuille indisponibles — vérifiez les tickers et réessayez."
            )
            st.session_state["suggestions_job"] = _sugg_job
        else:
            _sugg_sig = ohlcv_sig or (
                _tickers_signature(liste_tickers, start_date) if liste_tickers else ""
            )
            _sugg_returns = returns if not returns.empty else pd.DataFrame()
            if suggestions_job.run_step(
                returns=_sugg_returns,
                ohlcv_sig=_sugg_sig,
                has_fundamentals=HAS_FUNDAMENTALS,
                has_dividendes=HAS_DIVIDENDES,
            ):
                st.rerun()

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
        tickers_tuple = tuple(sorted(liste_tickers))
        start_date_str = str(start_date)

        if vue == "Synthèse & Plus-values":
            _render_synthese_plus_values(
                prices=prices,
                returns=returns,
                ohlcv_sig=ohlcv_sig,
            )

        elif vue == "Matrice de corrélation":
            st.header("📊 Matrice de Corrélation")
            corr = cached_corr_matrix(ohlcv_sig, tickers_tuple, start_date_str)
            if show_company_names:
                labels = {
                    t: ticker_label(t, ticker_names, True) for t in corr.index
                }
                corr = corr.rename(index=labels, columns=labels)

            @st.fragment
            def _correlation_view():
                c_corr1, c_corr2 = st.columns(2)
                cluster_corr = c_corr1.checkbox(
                    "Regrouper par similarité",
                    value=True,
                    help="Clustering hiérarchique : titres corrélés côte à côte.",
                    key="corr_cluster",
                )
                triangle_corr = c_corr2.checkbox(
                    "Triangle supérieur",
                    value=True,
                    help="Masque la moitié symétrique pour alléger la lecture.",
                    key="corr_triangle",
                )
                rho_lo, rho_hi = st.slider(
                    "Plage de ρ affichée (borne inférieure → supérieure)",
                    min_value=-1.0,
                    max_value=1.0,
                    value=(-1.0, 1.0),
                    step=0.05,
                    help=(
                        "Seules les corrélations dans cet intervalle sont colorées. "
                        "Ex. −0,15 à +0,15 pour repérer les paires **peu corrélées**."
                    ),
                    key="corr_rho_range",
                )
                fig_corr = build_correlation_heatmap_figure(
                    corr,
                    cluster=cluster_corr,
                    triangle=triangle_corr,
                    rho_min=rho_lo,
                    rho_max=rho_hi,
                )
                st.plotly_chart(fig_corr, use_container_width=True)
                if rho_lo <= -0.95 and rho_hi >= 0.95:
                    st.caption(
                        "Échelle **RdBu** : bleu = corrélation négative, rouge = positive, blanc ≈ 0. "
                        "ρ proche de **+1** : les titres bougent ensemble ; **−1** : mouvements opposés."
                    )
                else:
                    st.caption(
                        f"Filtre actif : **{rho_lo:+.2f} ≤ ρ ≤ {rho_hi:+.2f}** — "
                        "cellules hors plage masquées (diagonale incluse si ρ = 1,00 sort de la plage). "
                        "Réduisez la plage autour de **0** pour isoler les actifs diversifiants."
                    )

            _correlation_view()

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
            metrics = cached_advanced_risk_metrics(ohlcv_sig, tickers_tuple, start_date_str)
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

            st.markdown("### Carte rendement / volatilité")
            risk_label_map = {
                col: ticker_label(col, ticker_names, show_company_names)
                for col in returns.columns
            }
            scatter_weights = None
            if mode_portfolio_actif and portefeuille_dict:
                scatter_weights = compute_holdings_weights(
                    prices,
                    {tk: portefeuille_dict[tk]["Quantite"] for tk in portefeuille_dict},
                    usd_to_eur=cached_usd_to_eur_rate(),
                    is_usd_fn=is_usd_ticker,
                    universe=list(returns.columns),
                )
            fig_rr = build_risk_return_scatter_figure(
                returns,
                weights=scatter_weights,
                column_labels=risk_label_map,
            )
            if fig_rr is not None:
                st.plotly_chart(fig_rr, use_container_width=True)
            st.caption(
                "Chaque point = un actif (annualisé). **Couleur** = Sharpe · "
                "**Taille** = poids portefeuille si mode CSV actif."
            )

            @st.fragment
            def _return_distribution_view():
                dist_ticker = st.selectbox(
                    "Distribution des rendements journaliers",
                    list(returns.columns),
                    format_func=lambda t: risk_label_map.get(t, t),
                    key="risk_return_dist_ticker",
                )
                fig_dist = build_returns_distribution_var_figure(
                    returns[dist_ticker],
                    asset_label=risk_label_map.get(dist_ticker, dist_ticker),
                )
                if fig_dist is not None:
                    st.plotly_chart(fig_dist, use_container_width=True)
                st.caption(
                    "Trait rouge = **VaR 95 %** (5 % pires journées). "
                    "Histogramme en densité de probabilité."
                )

            _return_distribution_view()

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

                weights = cached_markowitz_weights(ohlcv_sig, tickers_tuple, start_date_str)

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
                current_weights_items = (
                    tuple(current_weights.items()) if has_current else None
                )
                fig_frontier = cached_markowitz_frontier_figure(
                    ohlcv_sig,
                    tickers_tuple,
                    start_date_str,
                    risk_free_rate,
                    n_random,
                    has_current,
                    current_weights_items,
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
                "RSI, MACD, stochastique, SuperTrend, moyennes mobiles 20/50/200, Bollinger, ATR, "
                "volumes, OBV, MFI, régression linéaire, supports/résistances et Fibonacci."
            )
            @st.fragment
            def _technical_summary():
                c_rsi, c_st_p, c_st_m = st.columns(3)
                with c_rsi:
                    rsi_period = st.slider("Période RSI", 7, 21, 14, key="tech_rsi_period")
                with c_st_p:
                    st_period = st.slider("SuperTrend — période ATR", 7, 21, 10, key="tech_st_period")
                with c_st_m:
                    st_multiplier = st.slider(
                        "SuperTrend — multiplicateur ATR",
                        1.0,
                        5.0,
                        3.0,
                        0.1,
                        key="tech_st_multiplier",
                    )
                tech_df = cached_technical_indicators(
                    ohlcv_sig,
                    rsi_period,
                    tickers_tuple,
                    start_date_str,
                    supertrend_period=st_period,
                    supertrend_multiplier=st_multiplier,
                )
                if tech_df.empty:
                    st.warning("Pas assez de données pour calculer les indicateurs techniques.")
                    st.session_state.pop("technical_df", None)
                    return
                st.session_state["technical_df"] = tech_df
                st.session_state["technical_rsi_period"] = rsi_period
                st.session_state["technical_st_period"] = st_period
                st.session_state["technical_st_multiplier"] = st_multiplier
                tech_display_df = add_company_names(
                    tech_df, ticker_names, show_names=show_company_names, sectors=ticker_sectors
                )
                tech_display = rename_columns_for_display(tech_display_df, TECHNICAL_LABELS)
                tech_format = format_map_for_labeled_columns(
                    tech_display, TECHNICAL_LABELS, TECHNICAL_FORMAT
                )
                st.caption(
                    "Prix et moyennes en **/ action** · volumes en **titres** · "
                    "RSI/MFI/stochastique sur **0–100** · ATR/Prix en **%**. "
                    "**Vert** = survente · **rouge** = surachat (colonnes signal et indicateurs)."
                )
                styled_tech = tech_display.style.format(tech_format, na_rep="-")
                styled_tech = style_technical_table(styled_tech)
                st.dataframe(styled_tech, use_container_width=True)

            @st.fragment
            def _technical_detail():
                tech_df = st.session_state.get("technical_df")
                rsi_period_detail = st.session_state.get("technical_rsi_period", 14)
                st_period_detail = st.session_state.get("technical_st_period", 10)
                st_multiplier_detail = st.session_state.get("technical_st_multiplier", 3.0)
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
                    tickers_tuple,
                    start_date_str,
                    supertrend_period=st_period_detail,
                    supertrend_multiplier=st_multiplier_detail,
                )
                if not bundle or "s" not in bundle:
                    st.warning(
                        f"Données insuffisantes pour les graphiques de "
                        f"{detail_label} — actualisez les cours."
                    )
                    return
                s = bundle["s"]
                rsi_series = bundle["rsi"]
                macd_line, signal_line, histogram = bundle["macd"]
                sma50 = bundle["sma50"]
                sma200 = bundle["sma200"]
                bb_upper, bb_mid, bb_lower = bundle["bollinger"]
                stoch_k, stoch_d = bundle["stochastic"]
                st_line = bundle["supertrend"]
                st_dir = bundle["supertrend_direction"]
                vol_s = bundle["vol_s"]
                has_volume = bundle["has_volume"]
                ohlc = bundle["ohlc"]
                has_ohlc = bundle["has_ohlc"]

                supports, resistances = compute_support_resistance(s, window=10, n_levels=2)
                vol_ticker = (
                    volumes[detail_ticker].reindex(s.index)
                    if not volumes.empty and detail_ticker in volumes.columns
                    else vol_s
                )

                fig_overview = build_technical_overview_figure(
                    s,
                    detail_label=detail_label,
                    ohlc=ohlc if has_ohlc else None,
                    volume_series=vol_ticker if has_volume else None,
                    sma50=sma50,
                    sma200=sma200,
                    bb_upper=bb_upper,
                    bb_mid=bb_mid,
                    bb_lower=bb_lower,
                    rsi_series=rsi_series,
                    rsi_period=rsi_period_detail,
                    supertrend_series=st_line,
                    supertrend_direction=st_dir,
                    supports=supports,
                    resistances=resistances,
                )
                if fig_overview is not None:
                    st.plotly_chart(fig_overview, use_container_width=True)
                    st.caption(
                        "Survol synchronisé sur les **3 panneaux** (prix, volume, RSI). "
                        "**SuperTrend** : vert = tendance haussière · rouge = baissière. "
                        "Zones **vertes** RSI 0–30 (survente), **rouges** 70–100 (surachat). "
                        "Traits **S** / **R** = supports et résistances locaux."
                    )
                    st.info(interpret_rsi_comment(rsi_series, rsi_period_detail))
                    st.info(
                        interpret_supertrend_comment(
                            s,
                            st_line,
                            st_dir,
                            period=st_period_detail,
                            multiplier=st_multiplier_detail,
                        )
                    )

                c1, c2 = st.columns(2)
                with c1:
                    fig_stoch = build_technical_stochastic_figure(
                        stoch_k, stoch_d, detail_label=detail_label
                    )
                    if fig_stoch is not None:
                        st.plotly_chart(fig_stoch, use_container_width=True)
                    st.info(interpret_stochastic_comment(stoch_k, stoch_d))
                with c2:
                    fig_macd = build_technical_macd_figure(
                        macd_line, signal_line, histogram, detail_label=detail_label
                    )
                    if fig_macd is not None:
                        st.plotly_chart(fig_macd, use_container_width=True)
                    st.info(interpret_macd_comment(macd_line, signal_line, histogram))

                st.subheader("📐 Tendance, supports & Fibonacci")
                reg_channel = compute_linear_regression_channel(s)
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
| **R² tendance** | Part de la **variance du cours** expliquée par la tendance seule (avant FFT). Affiché avec un **niveau de fiabilité** : ≥ 0,85 excellente · 0,65–0,85 bonne · 0,50–0,65 modérée · 0,30–0,50 faible · < 0,30 très faible. |
| **RMSE** | Écart quadratique moyen entre cours et tendance, en **€ / action**, **€ (portefeuille)** ou **pts (indice base 100)** selon la série ; complété par un **% du prix moyen** (comparable entre titres). |
| **Corr. modèle (avec dé-trend)** | Corrélation de Pearson entre le **cours** et le modèle FFT **tendance + cycles** (courbe cyan), sur l'historique. Tendance retirée avant FFT puis réinjectée. |
| **Corr. harmonique vs cours** | Régression sin/cos aux périodes FFT — souvent plus fidèle que la reconstruction par pics seuls. |
| **Corr. vs MM41 / lissage** | Modes MM41 ou STL : qualité du modèle sur la **série lissée** (sans bruit journalier). |
| **Corr. modèle (sans dé-trend)** | (Modes classiques) FFT sur log(prix) sans retirer la tendance — comparaison. |
| **Modèle FFT** | Tendance + cycles combinés, reprojetés en **prix** (cyan) ; prolongation **+30 %** en pointillé après la ligne « Fin historique ». |
| **Prix** | Courbe **bleue** : historique réel du titre. |

**Tendances disponibles :** **Log-linéaire** · **Linéaire €** · **Hodrick-Prescott**
· **MM centrée 41 j** (analyse, ±20 séances — regard futur) · **MM causale 41 j** (sans look-ahead, extrapolation)
· **STL 21 j** (décomposition trend + saison, FFT sur la saison). Comparez les **R²** et corrélations en changeant la tendance.

**Reconstruction du modèle :** **FFT (pics)** · **Harmonique (périodes FFT)** · **Harmonique (21–126 j fixes)**.

**Extrapolation :** le modèle mathématique (tendance + harmoniques) est prolongé de **30 %** de la durée
observée (jours de bourse). Zone pointillée = **scénario exploratoire**, pas une prévision garantie.

**Précautions :** un pic FFT peut venir du hasard, d'un dividende récurrent mal lissé ou d'une
fenêtre trop courte. Croisez avec l'analyse technique et fondamentale.

**Amplitude en fin de série :** cochez **Priorité aux valeurs récentes** pour éviter l'atténuation
due à la fenêtre Hanning (reconstruction harmonique pondérée + tendance locale).
                    """
                )

            if prices.empty or len(prices.dropna(how="all")) < 40:
                st.warning(
                    "Historique insuffisant pour une FFT fiable (minimum ~40 séances). "
                    "Élargissez la date de début dans la barre latérale."
                )
            else:
                @st.fragment
                def _fft_interactive():
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
                    recency_fit = st.checkbox(
                        "Priorité aux valeurs récentes (amplitude)",
                        value=True,
                        key="fft_recency_fit",
                        help=(
                            "Reconstruction sans fenêtre Hanning + pondération récente "
                            "et tendance locale en fin de série — réduit l'écart en bout d'historique."
                        ),
                    )

                    def _fft_trend_quality_caption(result, price_unit=FFT_PRICE_UNIT_EUR):
                        if not result:
                            return
                        summary = summarize_fft_trend_quality(result, price_unit=price_unit)
                        if summary.get("caption_md"):
                            st.caption(summary["caption_md"])

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
                    port_price_unit = (
                        FFT_PRICE_UNIT_PORTFOLIO if quantities_fft else FFT_PRICE_UNIT_INDEX
                    )

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
                    if "fft_recon_mode" not in st.session_state:
                        st.session_state.fft_recon_mode = FFT_RECON_FFT
                    trend_mode = st.session_state.fft_trend_mode
                    recon_mode = st.session_state.fft_recon_mode

                    if portfolio_series is None or len(portfolio_series.dropna()) < 40:
                        st.warning(
                            "Historique insuffisant pour la FFT du portefeuille agrégé "
                            "(minimum ~40 séances communes)."
                        )
                    else:
                        port_fft = cached_fft_periodicity_portfolio(
                            ohlcv_sig,
                            tickers_tuple,
                            start_date_str,
                            tuple(sorted(quantities_fft.items())) if quantities_fft else (),
                            bool(mode_portfolio_actif),
                            min_period,
                            max_period,
                            top_peaks,
                            trend_mode,
                        )
                        if port_fft is None:
                            st.warning("FFT portefeuille indisponible sur la période.")
                        else:
                            port_peaks = port_fft["peaks"].drop(columns=["_freq_idx"], errors="ignore")
                            _render_fft_headline_metrics(
                                port_fft,
                                port_peaks,
                                n_components=min(top_peaks, len(port_fft["peaks"])),
                                price_unit=port_price_unit,
                                recon_mode=recon_mode,
                                recency_weighted=recency_fit,
                                recency_calibrate=recency_fit,
                            )
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
                            _trend_ctrl, _recon_ctrl, _trend_metrics = st.columns([1, 1, 2])
                            with _trend_ctrl:
                                st.selectbox(
                                    "Tendance de fond",
                                    options=list(FFT_TREND_OPTIONS.keys()),
                                    format_func=fft_trend_mode_label,
                                    key="fft_trend_mode",
                                    help="Méthode pour isoler les cycles avant FFT.",
                                )
                            with _recon_ctrl:
                                st.selectbox(
                                    "Reconstruction du modèle",
                                    options=list(FFT_RECON_OPTIONS.keys()),
                                    format_func=fft_recon_mode_label,
                                    key="fft_recon_mode",
                                    help=(
                                        "Comment reconstruire le prix modèle : pics FFT ou "
                                        "régression harmonique sin/cos."
                                    ),
                                )
                            with _trend_metrics:
                                _fft_trend_quality_caption(port_fft, price_unit=port_price_unit)
                            st.plotly_chart(
                                build_fft_cyclic_chart(
                                    port_fft,
                                    n_components=min(top_peaks, len(port_fft["peaks"])),
                                    title="Cycle estimé vs portefeuille complet",
                                    recon_mode=recon_mode,
                                    recency_weighted=recency_fit,
                                    recency_calibrate=recency_fit,
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
                        tickers_tuple,
                        start_date_str,
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
                        fft_result = cached_fft_periodicity_ticker(
                            ohlcv_sig,
                            tickers_tuple,
                            start_date_str,
                            fft_ticker,
                            min_period,
                            max_period,
                            top_peaks,
                            trend_mode,
                        )
                        if fft_result is None:
                            st.warning("FFT indisponible pour ce titre.")
                        else:
                            detail_label = ticker_label(
                                fft_ticker, ticker_names, show_company_names
                            )
                            peaks = fft_result["peaks"].drop(columns=["_freq_idx"], errors="ignore")
                            _render_fft_headline_metrics(
                                fft_result,
                                peaks,
                                n_components=min(top_peaks, len(fft_result["peaks"])),
                                recon_mode=recon_mode,
                                recency_weighted=recency_fit,
                                recency_calibrate=recency_fit,
                            )
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
                                    recon_mode=recon_mode,
                                    recency_weighted=recency_fit,
                                    recency_calibrate=recency_fit,
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

                _fft_interactive()
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
            if suggestions_job.is_active():
                st.info(
                    "Génération en cours — vous pouvez consulter d'autres pages "
                    "(ex. *Dividendes & Qualité*). Suivez la progression dans la "
                    "**barre latérale** ; le résultat apparaîtra ici à la fin."
                )
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
                    piotroski_lo = piotroski_hi = None
                    roe_lo = roe_hi = None
                    per_lo = per_hi = None
                    debt_lo = debt_hi = None
                    include_technical = True
                    rsi_lo = rsi_hi = None
                    bb_lo = bb_hi = None
                    include_dividends = False
                    dividend_active_only = False
                    yield_lo = yield_hi = None
                    coverage_lo = coverage_hi = None
                    cagr_lo = cagr_hi = None
                    growth_years_lo = growth_years_hi = None
                    payout_lo = payout_hi = None
                    fcf_payout_lo = fcf_payout_hi = None
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
                            piotroski_lo, piotroski_hi = _suggestion_range_filter(
                                f1,
                                "Piotroski",
                                0,
                                9,
                                (6, 9),
                                1,
                                key_prefix="sugg_piotroski",
                                help="Score 0–9 · ≥ 7 = solide financièrement.",
                            )
                            roe_lo, roe_hi = _suggestion_range_filter(
                                f2,
                                "ROE (%)",
                                0.0,
                                30.0,
                                (10.0, 30.0),
                                1.0,
                                key_prefix="sugg_roe",
                                help="Rentabilité des capitaux propres.",
                                scale=1 / 100.0,
                            )
                            per_lo, per_hi = _suggestion_range_filter(
                                f3,
                                "PER",
                                1.0,
                                80.0,
                                (5.0, 25.0),
                                1.0,
                                key_prefix="sugg_per",
                                help="PER > 0 uniquement (titres profitables).",
                            )
                            debt_lo, debt_hi = _suggestion_range_filter(
                                f4,
                                "Dette / Capitaux",
                                0.0,
                                5.0,
                                (0.0, 1.5),
                                0.1,
                                key_prefix="sugg_debt",
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
                        t1, t2 = st.columns(2)
                        rsi_lo, rsi_hi = _suggestion_range_filter(
                            t1,
                            "RSI",
                            10,
                            90,
                            (30, 65),
                            1,
                            key_prefix="sugg_rsi",
                            help="Ex. 30–65 = hors zones extrêmes de surachat.",
                        )
                        bb_lo, bb_hi = _suggestion_range_filter(
                            t2,
                            "Bollinger %B",
                            0.0,
                            1.0,
                            (0.2, 0.8),
                            0.05,
                            key_prefix="sugg_bb",
                            help="0 = bande basse · 1 = bande haute.",
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
                            yield_lo, yield_hi = _suggestion_range_filter(
                                d1,
                                "rendement (%)",
                                0.0,
                                12.0,
                                (2.0, 8.0),
                                0.25,
                                key_prefix="sugg_yield",
                                scale=1 / 100.0,
                            )
                            coverage_lo, coverage_hi = _suggestion_range_filter(
                                d2,
                                "couverture (×)",
                                0.5,
                                5.0,
                                (1.2, 5.0),
                                0.1,
                                key_prefix="sugg_coverage",
                                help="Bénéfices ÷ dividendes (> 1 = couvert).",
                            )
                            cagr_lo, cagr_hi = _suggestion_range_filter(
                                d3,
                                "CAGR 5 ans (%)",
                                -10.0,
                                20.0,
                                (0.0, 15.0),
                                0.5,
                                key_prefix="sugg_cagr",
                                scale=1 / 100.0,
                            )
                            growth_years_lo, growth_years_hi = _suggestion_range_filter(
                                d4,
                                "années croissance",
                                0,
                                20,
                                (3, 20),
                                1,
                                key_prefix="sugg_growth_years",
                            )
                            d5, d6 = st.columns(2)
                            payout_lo, payout_hi = _suggestion_range_filter(
                                d5,
                                "payout (%)",
                                20.0,
                                120.0,
                                (20.0, 80.0),
                                5.0,
                                key_prefix="sugg_payout",
                                scale=1 / 100.0,
                            )
                            fcf_payout_lo, fcf_payout_hi = _suggestion_range_filter(
                                d6,
                                "FCF payout (%)",
                                20.0,
                                150.0,
                                (20.0, 80.0),
                                5.0,
                                key_prefix="sugg_fcf_payout",
                                scale=1 / 100.0,
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

                    cand_return_lo = cand_return_hi = None
                    cand_vol_lo = cand_vol_hi = None
                    cand_kurt_lo = cand_kurt_hi = None
                    cand_skew_lo = cand_skew_hi = None
                    cand_sharpe_lo = cand_sharpe_hi = None
                    corr_pf_lo = corr_pf_hi = None
                    delta_return_lo = delta_return_hi = None
                    delta_vol_lo = delta_vol_hi = None
                    delta_kurt_lo = delta_kurt_hi = None
                    delta_skew_lo = delta_skew_hi = None
                    delta_sharpe_lo = delta_sharpe_hi = None
                    delta_corr_lo = delta_corr_hi = None

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
                            s1, s2, s3 = st.columns(3)
                            s4, s5, s6 = st.columns(3)
                            cand_return_lo, cand_return_hi = _suggestion_range_filter(
                                s1,
                                "rendement candidat (%)",
                                -20.0,
                                40.0,
                                (5.0, 40.0),
                                1.0,
                                key_prefix="sugg_cand_return",
                                scale=1 / 100.0,
                            )
                            cand_vol_lo, cand_vol_hi = _suggestion_range_filter(
                                s2,
                                "volatilité candidat (%)",
                                5.0,
                                80.0,
                                (5.0, 25.0),
                                1.0,
                                key_prefix="sugg_cand_vol",
                                scale=1 / 100.0,
                            )
                            cand_kurt_lo, cand_kurt_hi = _suggestion_range_filter(
                                s3,
                                "kurtosis candidat",
                                0.0,
                                15.0,
                                (0.0, 5.0),
                                0.5,
                                key_prefix="sugg_cand_kurt",
                            )
                            cand_skew_lo, cand_skew_hi = _suggestion_range_filter(
                                s4,
                                "skewness candidat",
                                -2.0,
                                2.0,
                                (0.0, 2.0),
                                0.1,
                                key_prefix="sugg_cand_skew",
                            )
                            corr_pf_lo, corr_pf_hi = _suggestion_range_filter(
                                s5,
                                "corr. portefeuille",
                                0.0,
                                1.0,
                                (0.0, 0.65),
                                0.05,
                                key_prefix="sugg_corr_pf",
                                help="Corrélation moyenne avec le portefeuille (0–1).",
                            )
                            cand_sharpe_lo, cand_sharpe_hi = _suggestion_range_filter(
                                s6,
                                "Sharpe candidat",
                                -1.0,
                                3.0,
                                (0.5, 3.0),
                                0.1,
                                key_prefix="sugg_cand_sharpe",
                                help="Ratio de Sharpe annualisé du candidat seul.",
                            )
                            st.markdown("**Impact simulé sur le portefeuille**")
                            d1, d2, d3 = st.columns(3)
                            d4, d5, d6 = st.columns(3)
                            delta_return_lo, delta_return_hi = _suggestion_range_filter(
                                d1,
                                "Δ rendement (pts %)",
                                -5.0,
                                10.0,
                                (0.0, 10.0),
                                0.5,
                                key_prefix="sugg_delta_return",
                                scale=1 / 100.0,
                            )
                            delta_vol_lo, delta_vol_hi = _suggestion_range_filter(
                                d2,
                                "Δ volatilité (pts %)",
                                -10.0,
                                10.0,
                                (0.0, 10.0),
                                0.5,
                                key_prefix="sugg_delta_vol",
                                help="Positif = le portefeuille devient moins volatile.",
                                scale=1 / 100.0,
                            )
                            delta_kurt_lo, delta_kurt_hi = _suggestion_range_filter(
                                d3,
                                "Δ kurtosis",
                                -5.0,
                                5.0,
                                (0.0, 5.0),
                                0.25,
                                key_prefix="sugg_delta_kurt",
                                help="Positif = moins de risque de chutes brutales.",
                            )
                            delta_skew_lo, delta_skew_hi = _suggestion_range_filter(
                                d4,
                                "Δ skewness",
                                -2.0,
                                2.0,
                                (0.0, 2.0),
                                0.1,
                                key_prefix="sugg_delta_skew",
                            )
                            delta_corr_lo, delta_corr_hi = _suggestion_range_filter(
                                d5,
                                "Δ corr. interne",
                                -0.5,
                                0.5,
                                (0.0, 0.5),
                                0.05,
                                key_prefix="sugg_delta_corr",
                                help="Positif = les titres divergent davantage.",
                            )
                            delta_sharpe_lo, delta_sharpe_hi = _suggestion_range_filter(
                                d6,
                                "Δ Sharpe",
                                -1.0,
                                2.0,
                                (0.0, 2.0),
                                0.05,
                                key_prefix="sugg_delta_sharpe",
                                help="Positif = le Sharpe du portefeuille simulé s'améliore.",
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

                    run_suggestions = st.button(
                        "Générer les suggestions",
                        type="primary",
                        disabled=suggestions_job.is_active(),
                    )

                    if run_suggestions:
                        if not selected_universes:
                            st.warning("Sélectionnez au moins un univers de candidats.")
                        else:
                            suggestions_job.start(
                                {
                                    "selected_universes": list(selected_universes),
                                    "portfolio_tickers": list(portfolio_tickers),
                                    "candidate_weight": candidate_weight,
                                    "max_candidates": max_candidates,
                                    "objective_weights": dict(objective_weights),
                                    "include_fundamentals": include_fundamentals,
                                    "include_technical": include_technical,
                                    "include_dividends": include_dividends,
                                    "selection_mode": selection_mode,
                                    "start_date": str(start_date),
                                    "ohlcv_sig": ohlcv_sig,
                                    "json_path": JSON_PATH,
                                }
                            )
                            st.rerun()

                    if "portfolio_suggestions_baseline" in st.session_state:
                        baseline = st.session_state["portfolio_suggestions_baseline"]
                        baseline_internal = st.session_state.get(
                            "portfolio_suggestions_internal_corr", np.nan
                        )
                        meta = st.session_state.get("portfolio_suggestions_meta", {})
                        st.markdown("### Référence portefeuille actuel")
                        if baseline:
                            b1, b2, b3, b4, b5 = st.columns(5)
                            b1.metric("Rendement annuel", f"{baseline['Rendement Annuel']:.2%}")
                            b2.metric("Volatilité", f"{baseline['Volatilité (Sigma)']:.2%}")
                            b3.metric("Sharpe", f"{baseline.get('Ratio de Sharpe', 0):.2f}")
                            b4.metric("Skewness", f"{baseline['Skewness (Asymétrie)']:.2f}")
                            b5.metric("Corr. interne moy.", f"{baseline_internal:.2f}")
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
                                min_candidate_return=cand_return_lo,
                                max_candidate_return=cand_return_hi,
                                min_candidate_vol=cand_vol_lo,
                                max_candidate_vol=cand_vol_hi,
                                min_candidate_kurtosis=cand_kurt_lo,
                                max_candidate_kurtosis=cand_kurt_hi,
                                min_candidate_skewness=cand_skew_lo,
                                max_candidate_skewness=cand_skew_hi,
                                min_candidate_sharpe=cand_sharpe_lo,
                                max_candidate_sharpe=cand_sharpe_hi,
                                min_corr_portfolio=corr_pf_lo,
                                max_corr_portfolio=corr_pf_hi,
                                min_delta_return=delta_return_lo,
                                max_delta_return=delta_return_hi,
                                min_delta_vol=delta_vol_lo,
                                max_delta_vol=delta_vol_hi,
                                min_delta_kurtosis=delta_kurt_lo,
                                max_delta_kurtosis=delta_kurt_hi,
                                min_delta_skewness=delta_skew_lo,
                                max_delta_skewness=delta_skew_hi,
                                min_delta_sharpe=delta_sharpe_lo,
                                max_delta_sharpe=delta_sharpe_hi,
                                min_delta_corr_internal=delta_corr_lo,
                                max_delta_corr_internal=delta_corr_hi,
                            )
                            if len(suggestions_view) < before_stats:
                                filter_parts.append("statistiques")

                        if HAS_FUNDAMENTALS and any(
                            c in suggestions.columns
                            for c in fundamentals.SUGGESTION_QUALITY_COLUMNS
                        ):
                            suggestions_view = fundamentals.filter_suggestions_by_quality(
                                suggestions_view,
                                min_piotroski=piotroski_lo,
                                max_piotroski=piotroski_hi,
                                min_roe=roe_lo,
                                max_roe=roe_hi,
                                min_per=per_lo,
                                max_per=per_hi,
                                min_debt_equity=debt_lo,
                                max_debt_equity=debt_hi,
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
                                min_rsi=rsi_lo,
                                max_rsi=rsi_hi,
                                min_bollinger_b=bb_lo,
                                max_bollinger_b=bb_hi,
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
                                min_yield=yield_lo,
                                max_yield=yield_hi,
                                min_coverage=coverage_lo,
                                max_coverage=coverage_hi,
                                min_cagr_5y=cagr_lo,
                                max_cagr_5y=cagr_hi,
                                min_growth_years=growth_years_lo,
                                max_growth_years=growth_years_hi,
                                min_payout=payout_lo,
                                max_payout=payout_hi,
                                min_fcf_payout=fcf_payout_lo,
                                max_fcf_payout=fcf_payout_hi,
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
                                sugg_names, sugg_sectors = _resolve_ticker_metadata(
                                    display_df["Ticker"].tolist(),
                                    ticker_names,
                                    ticker_sectors,
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
                                plot_names, _ = _resolve_ticker_metadata(
                                    plot_df["Ticker"].tolist(),
                                    ticker_names,
                                    ticker_sectors,
                                    need_names=True,
                                    need_sectors=False,
                                )
                                plot_df["Libellé"] = plot_df["Ticker"].map(
                                    lambda t: ticker_label(t, plot_names, True)
                                )
                            score_min = plot_df["Score composite"].min()
                            plot_df["Taille score"] = plot_df["Score composite"] - score_min + 0.01
                            fig_sugg = build_suggestions_tradeoff_scatter(
                                plot_df,
                                label_col=(
                                    "Libellé"
                                    if show_company_names and "Libellé" in plot_df.columns
                                    else "Ticker"
                                ),
                            )
                            if fig_sugg is not None:
                                st.plotly_chart(fig_sugg, use_container_width=True)
                                st.caption(
                                    "Survolez les points pour le détail — "
                                    "bas à gauche = diversifiant et performant."
                                )

                            csv_sugg = suggestions_view.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                "Télécharger les suggestions (CSV)",
                                csv_sugg,
                                "suggestions_actifs.csv",
                            )
                    job_state = st.session_state.get("suggestions_job")
                    if job_state and job_state.get("status") == "error":
                        st.error(f"Échec : {job_state.get('error', 'erreur inconnue')}")
                    meta = st.session_state.get("portfolio_suggestions_meta") or {}
                    if meta.get("fund_stats"):
                        fs = meta["fund_stats"]
                        st.caption(
                            "Fondamentaux — "
                            f"**{fs.get('disk', 0)}** disque · "
                            f"**{fs.get('session', 0)}** session · "
                            f"**{fs.get('fetched', 0)}** téléchargés · "
                            f"**{fs.get('missing', 0)}** sans données."
                        )
                    if meta.get("div_stats"):
                        ds = meta["div_stats"]
                        st.caption(
                            "Dividendes — "
                            f"**{ds.get('disk', 0)}** disque · "
                            f"**{ds.get('session', 0)}** session · "
                            f"**{ds.get('fetched', 0)}** téléchargés · "
                            f"**{ds.get('missing', 0)}** sans données."
                        )


if vue == "Dividendes & Qualité" and HAS_DIVIDENDES:
    pf_tickers = sorted({str(t).strip() for t in liste_tickers}) if liste_tickers else None
    with st.expander("Univers dividendes — tickers.json (5000 actions)", expanded=True):
        dividendes.render_dividend_universe_builder(portfolio_tickers=pf_tickers)
    st.markdown("---")

suggestions_job.render_sidebar_status()
render_live_dashboard()
