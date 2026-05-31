# ==========================
# BLOC 1/4 : Imports & indices
# ==========================
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
from analytics import (
    compute_advanced_risk_metrics,
    compute_clusters,
    compute_corr,
    compute_cov,
    compute_markowitz,
    compute_max_drawdown,
    compute_pca,
    compute_returns,
)
from data_loader import filter_valid_tickers, load_prices_in_batches

# Configuration de la page (OBLIGATOIREMENT EN PREMIER)
st.set_page_config(layout="wide")

# Chargement du fichier JSON
try:
    with open("tickers.json", "r") as f:
        INDEX_TICKERS = json.load(f)
except FileNotFoundError:
    st.error(
        "Fichier 'tickers.json' introuvable. Veuillez le placer au même endroit que ce script."
    )
    st.stop()

# Importation directe du module dividendes pour intercepter explicitement les erreurs
try:
    import dividendes
    HAS_DIVIDENDES = True
except Exception as e:
    HAS_DIVIDENDES = False
    st.sidebar.error(f"⚠️ Erreur de chargement dans 'dividendes.py' : {e}")

try:
    import fundamentals
    HAS_FUNDAMENTALS = True
except Exception as e:
    HAS_FUNDAMENTALS = False
    st.sidebar.error(f"⚠️ Erreur de chargement dans 'fundamentals.py' : {e}")

# ==========================
# BLOC 2/3 : Interface & chargement
# ==========================

st.sidebar.title("Dashboard v6.2 — Multi-indices")
universe_mode = st.sidebar.radio(
    "Source de l'univers",
    ["Indice", "Personnalisé"],
    horizontal=True,
)
index_choice = st.sidebar.selectbox("Indice", list(INDEX_TICKERS.keys()))
start = st.sidebar.date_input("Date de début", pd.to_datetime("2015-01-01"))

vue = st.sidebar.selectbox(
    "Vue",
    [
        "Vue globale",
        "Corrélations avancées",
        "Covariance",
        "PCA / Facteurs",
        "Analyse par actif",
        "Portefeuille",
        "Optimisation / Markowitz",
        "Diversification / Clusters",
        "Rendement / Volatilité / Sharpe",
        "Distribution des gains/pertes",
        "Dividendes & Qualité",
        "Analyse fondamentale",
    ],
)

if universe_mode == "Indice":
    with st.spinner("Vérification des tickers valides..."):
        tickers = filter_valid_tickers(INDEX_TICKERS[index_choice], start)
    if len(tickers) < 2:
        st.error("Pas assez de tickers valides pour cet indice.")
        st.stop()
    st.sidebar.write(f"{len(tickers)} tickers détectés dans {index_choice}")
else:
    tickers = []
    st.sidebar.caption(
        "Mode personnalisé: ajoute tes tickers/noms ci-dessous sans dépendre d'un indice."
    )

NAME_TO_TICKER = {
    "edenred": "EDEN.PA",
    "nyxoah": "NYXH",
    "solitario xpl": "XPL",
    "solitario": "XPL",
    "netgem": "ALNTG.PA",
    "zalando": "ZAL.DE",
    "nutex": "NUTX",
    "harmony gold": "HMY",
    "eutelsat": "ETL.PA",
    "etl": "ETL.PA",
    "ipsos": "IPS.PA",
}
if "extra_tickers" not in st.session_state:
    st.session_state["extra_tickers"] = []

name_input = st.sidebar.text_input(
    "Ajouter par nom (virgules)",
    value="",
    key="name_input_field",
    help="Exemple: edenred, zalando, ipsos",
)
if st.sidebar.button("Ajouter noms"):
    name_tokens = [
        t.strip().lower() for t in name_input.replace(";", ",").split(",") if t.strip()
    ]
    mapped_tickers = [NAME_TO_TICKER[n] for n in name_tokens if n in NAME_TO_TICKER]
    unknown_names = [n for n in name_tokens if n not in NAME_TO_TICKER]
    if mapped_tickers:
        st.session_state["extra_tickers"] = list(
            dict.fromkeys(st.session_state["extra_tickers"] + mapped_tickers)
        )
        st.sidebar.caption("Noms reconnus -> " + ", ".join(mapped_tickers))
    if unknown_names:
        st.sidebar.warning(
            "Noms non reconnus: "
            + ", ".join(unknown_names)
            + ". Ajoute le ticker manuel."
        )

custom_tickers_input = st.sidebar.text_input(
    "Ajouter des tickers perso (virgules)",
    value="",
    key="custom_ticker_field",
    help="Exemple: EDEN.PA, ZAL.DE, IPS.PA",
)
if st.sidebar.button("Ajouter tickers"):
    custom_tickers = [
        t.strip().upper()
        for t in custom_tickers_input.replace(";", ",").split(",")
        if t.strip()
    ]
    st.session_state["extra_tickers"] = list(
        dict.fromkeys(st.session_state["extra_tickers"] + custom_tickers)
    )

if st.session_state["extra_tickers"]:
    st.sidebar.caption("Tickers ajoutés: " + ", ".join(st.session_state["extra_tickers"]))
if st.sidebar.button("Vider tickers ajoutés"):
    st.session_state["extra_tickers"] = []
    st.rerun()

combined_universe = list(dict.fromkeys(list(tickers) + st.session_state["extra_tickers"]))
if not combined_universe:
    st.info(
        "Ajoute des tickers via 'Ajouter par nom' ou 'Ajouter des tickers perso' pour commencer."
    )
    st.stop()

selected_tickers = st.sidebar.multiselect(
    "Univers analysé",
    options=combined_universe,
    default=combined_universe,
    help="Limitez le nombre d'actifs pour accélérer les calculs.",
)
if len(selected_tickers) < 2:
    st.error("Sélectionnez au moins 2 tickers dans 'Univers analysé'.")
    st.stop()

prices = load_prices_in_batches(selected_tickers, start)
if prices.empty:
    st.error(
        "Impossible de charger les prix. Vérifiez votre connexion internet ou le format des tickers."
    )
    st.stop()

returns = compute_returns(prices)
if returns.empty:
    st.error("Pas assez de données de rendements générées.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.subheader("Santé des données")
coverage = (1 - prices.isna().mean()).sort_values(ascending=True)
st.sidebar.caption(
    f"Période disponible : {prices.index.min().date()} -> {prices.index.max().date()}"
)
st.sidebar.caption(f"Actifs chargés : {prices.shape[1]} / {len(selected_tickers)}")
if not coverage.empty:
    worst = coverage.index[0]
    st.sidebar.caption(f"Couverture minimale : {coverage.iloc[0]:.1%} ({worst})")


# =========================================================
# FILTRES DYNAMIQUES (Barre latérale conditionnelle)
# =========================================================
if vue == "Rendement / Volatilité / Sharpe":
    # On génère les métriques en tâche de fond pour calibrer les min/max des sliders
    raw_perf = compute_advanced_risk_metrics(returns)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Filtres de sélection")
    
    # 1. Filtre Sharpe
    min_sh, max_sh = float(raw_perf["Ratio de Sharpe"].min()), float(raw_perf["Ratio de Sharpe"].max())
    f_sharpe = st.sidebar.slider("Ratio de Sharpe minimum", min_sh, max_sh, max(min_sh, 0.0), step=0.1)
    
    # 2. Filtre Volatilité
    min_vol, max_vol = float(raw_perf["Volatilité (Sigma)"].min()), float(raw_perf["Volatilité (Sigma)"].max())
    f_vol = st.sidebar.slider("Volatilité maximum", min_vol, max_vol, max_vol, format="%.1f")
    
    # 3. Filtre Skewness
    min_sk, max_sk = float(raw_perf["Skewness (Asymétrie)"].min()), float(raw_perf["Skewness (Asymétrie)"].max())
    f_skew = st.sidebar.slider("Skewness minimum (Éviter les krachs)", min_sk, max_sk, min_sk, step=0.1)


# ==========================
# BLOC 4/4 : Vues du dashboard
# ==========================

if vue == "Vue globale":
    st.subheader(f"Évolution des prix (Base 100) — {index_choice}")
    prices_norm = (prices / prices.iloc[0].fillna(1)) * 100
    st.line_chart(prices_norm)

elif vue == "Corrélations avancées":
    st.subheader("Matrice de corrélation")
    corr = compute_corr(returns)
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
    )
    st.plotly_chart(fig, use_container_width=True)

elif vue == "Covariance":
    st.subheader("Matrice de covariance (Annualisée)")
    cov = compute_cov(returns) * 252
    fig = px.imshow(
        cov, text_auto=".4f", aspect="auto", color_continuous_scale="Blues"
    )
    st.plotly_chart(fig, use_container_width=True)

elif vue == "PCA / Facteurs":
    st.subheader("Analyse en Composantes Principales")
    pca, X = compute_pca(returns)
    exp = pca.explained_variance_ratio_
    fig = px.bar(
        x=list(range(1, len(exp) + 1)),
        y=exp,
        color=exp,
        labels={"x": "Composante", "y": "Variance expliquée"},
        color_continuous_scale="Viridis",
    )
    st.plotly_chart(fig, use_container_width=True)

elif vue == "Analyse par actif":
    st.subheader("Analyse individuelle")
    t = st.selectbox("Choisir un ticker", prices.columns)
    s = prices[t].dropna()
    r = s.pct_change().dropna()

    col1, col2, col3 = st.columns(3)
    col1.metric("Perf. cumulée", f"{(s.iloc[-1] / s.iloc[0] - 1):.2%}")
    col2.metric("Volatilité annualisée", f"{(r.std() * np.sqrt(252)):.2%}")
    col3.metric("Max Drawdown", f"{compute_max_drawdown(s):.2%}")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=s.index, y=s, mode="lines", line=dict(color="blue"), name=t
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[s.idxmin()],
            y=[s.min()],
            mode="markers",
            marker=dict(size=12, color="red"),
            name="Min",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[s.idxmax()],
            y=[s.max()],
            mode="markers",
            marker=dict(size=12, color="green"),
            name="Max",
        )
    )
    st.plotly_chart(fig, use_container_width=True)

elif vue == "Portefeuille":
    st.subheader("Portefeuille personnalisé")
    mode_portefeuille = st.radio(
        "Mode de construction",
        ["Pondérations (%)", "Positions (quantité + prix moyen)"],
        horizontal=True,
    )

    if mode_portefeuille == "Pondérations (%)":
        default_n = min(5, len(prices.columns))
        default_assets = list(prices.columns[:default_n])
        portfolio_assets = st.multiselect(
            "Sélectionner les actifs du portefeuille",
            options=list(prices.columns),
            default=default_assets,
        )

        if len(portfolio_assets) < 2:
            st.warning("Sélectionnez au moins 2 actifs pour construire un portefeuille.")
        else:
            st.markdown("### Pondérations (normalisées automatiquement)")
            raw_weights = []
            cols = st.columns(min(4, len(portfolio_assets)))
            for i, asset in enumerate(portfolio_assets):
                with cols[i % len(cols)]:
                    w = st.number_input(
                        f"{asset} (%)",
                        min_value=0.0,
                        max_value=100.0,
                        value=float(100.0 / len(portfolio_assets)),
                        step=1.0,
                        key=f"weight_{asset}",
                    )
                raw_weights.append(w / 100.0)

            raw_weights = np.array(raw_weights, dtype=float)
            if raw_weights.sum() <= 0:
                st.error("La somme des pondérations doit être supérieure à 0.")
            else:
                weights = raw_weights / raw_weights.sum()
                weights_s = pd.Series(weights, index=portfolio_assets, name="Poids normalisés")
                st.dataframe((weights_s * 100).to_frame().style.format("{:.2f}%"))

                sub_returns = returns[portfolio_assets].dropna(how="any")
                if sub_returns.empty:
                    st.error("Pas assez de données communes entre ces actifs.")
                else:
                    portfolio_r = sub_returns.dot(weights)
                    portfolio_curve = (1 + portfolio_r).cumprod() * 100

                    annual_return = portfolio_r.mean() * 252
                    annual_vol = portfolio_r.std() * np.sqrt(252)
                    downside_std = portfolio_r.clip(upper=0).std() * np.sqrt(252)
                    sharpe = annual_return / (annual_vol + 1e-8)
                    sortino = annual_return / (downside_std + 1e-8)
                    var95 = portfolio_r.quantile(0.05)
                    cvar95 = portfolio_r[portfolio_r <= var95].mean()
                    max_dd = compute_max_drawdown(portfolio_curve)

                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("Rendement annualisé", f"{annual_return:.2%}")
                    c2.metric("Volatilité annualisée", f"{annual_vol:.2%}")
                    c3.metric("Sharpe", f"{sharpe:.2f}")
                    c4.metric("Sortino", f"{sortino:.2f}")
                    c5.metric("Max Drawdown", f"{max_dd:.2%}")
                    st.caption(f"VaR 95% (jour) : {var95:.2%} | CVaR 95% (jour) : {cvar95:.2%}")

                    fig = px.line(
                        portfolio_curve,
                        labels={"value": "Base 100", "index": "Date"},
                        title="Performance cumulée du portefeuille (Base 100)",
                    )
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption(
            "Renseigne le ticker Yahoo Finance, la quantité et ton prix moyen d'achat."
        )
        with st.expander("Aide: correspondance nom -> ticker (à adapter si besoin)"):
            st.markdown(
                "- Edenred -> `EDEN.PA`\n"
                "- Nyxoah -> `NYXH`\n"
                "- Solitario Xpl -> `XPL`\n"
                "- Netgem -> `ALNTG.PA`\n"
                "- Zalando -> `ZAL.DE`\n"
                "- Nutex -> `NUTX`\n"
                "- Harmony Gold -> `HMY`\n"
                "- Eutelsat -> `ETL.PA`\n"
                "- Ipsos -> `IPS.PA`"
            )

        default_positions = pd.DataFrame(
            [
                {"Ticker": "EDEN.PA", "Quantité": 0.0, "Prix moyen": 0.0},
                {"Ticker": "NYXH", "Quantité": 0.0, "Prix moyen": 0.0},
                {"Ticker": "XPL", "Quantité": 0.0, "Prix moyen": 0.0},
                {"Ticker": "ALNTG.PA", "Quantité": 0.0, "Prix moyen": 0.0},
                {"Ticker": "ZAL.DE", "Quantité": 0.0, "Prix moyen": 0.0},
                {"Ticker": "NUTX", "Quantité": 0.0, "Prix moyen": 0.0},
                {"Ticker": "HMY", "Quantité": 0.0, "Prix moyen": 0.0},
                {"Ticker": "ETL.PA", "Quantité": 0.0, "Prix moyen": 0.0},
                {"Ticker": "IPS.PA", "Quantité": 0.0, "Prix moyen": 0.0},
            ]
        )
        positions_input = st.data_editor(
            default_positions,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key="positions_editor",
        )
        positions_df = positions_input.copy()
        positions_df["Ticker"] = positions_df["Ticker"].astype(str).str.strip().str.upper()
        positions_df = positions_df.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["Ticker", "Quantité", "Prix moyen"]
        )
        positions_df = positions_df[
            (positions_df["Ticker"] != "")
            & (positions_df["Quantité"] > 0)
            & (positions_df["Prix moyen"] > 0)
        ]

        if positions_df.empty:
            st.info("Ajoute au moins une ligne valide avec ticker, quantité et prix moyen.")
        else:
            pos_tickers = positions_df["Ticker"].tolist()
            pos_prices = load_prices_in_batches(pos_tickers, start, batch_size=20)
            if pos_prices.empty:
                st.error("Impossible de charger les prix de ces tickers. Vérifie les symboles Yahoo.")
            else:
                latest = pos_prices.ffill().bfill().iloc[-1]
                positions_df["Prix actuel"] = positions_df["Ticker"].map(latest.to_dict())
                positions_df = positions_df.dropna(subset=["Prix actuel"]).copy()
                if positions_df.empty:
                    st.error("Aucun ticker n'a retourné de prix exploitable.")
                else:
                    positions_df["Coût total"] = positions_df["Quantité"] * positions_df["Prix moyen"]
                    positions_df["Valeur actuelle"] = positions_df["Quantité"] * positions_df["Prix actuel"]
                    positions_df["P/L latent"] = (
                        positions_df["Valeur actuelle"] - positions_df["Coût total"]
                    )
                    positions_df["P/L latent %"] = (
                        positions_df["P/L latent"] / (positions_df["Coût total"] + 1e-8)
                    )
                    total_value = positions_df["Valeur actuelle"].sum()
                    positions_df["Poids actuel"] = positions_df["Valeur actuelle"] / (
                        total_value + 1e-8
                    )

                    st.dataframe(
                        positions_df.style.format(
                            {
                                "Quantité": "{:.2f}",
                                "Prix moyen": "{:.2f}",
                                "Prix actuel": "{:.2f}",
                                "Coût total": "{:.2f}",
                                "Valeur actuelle": "{:.2f}",
                                "P/L latent": "{:.2f}",
                                "P/L latent %": "{:.2%}",
                                "Poids actuel": "{:.2%}",
                            }
                        ).background_gradient(cmap="RdYlGn", subset=["P/L latent %"])
                    )

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Valeur portefeuille", f"{total_value:,.2f}")
                    total_cost = positions_df["Coût total"].sum()
                    total_pnl = total_value - total_cost
                    c2.metric("P/L total", f"{total_pnl:,.2f}")
                    c3.metric("P/L total %", f"{(total_pnl / (total_cost + 1e-8)):.2%}")

                    avail_cols = [t for t in positions_df["Ticker"] if t in pos_prices.columns]
                    hist = pos_prices[avail_cols].ffill().bfill().dropna(how="all")
                    qty_map = positions_df.set_index("Ticker")["Quantité"].to_dict()
                    hist = hist.mul(pd.Series(qty_map), axis=1)
                    portfolio_value = hist.sum(axis=1).dropna()
                    port_r = portfolio_value.pct_change().dropna()

                    if not port_r.empty:
                        annual_return = port_r.mean() * 252
                        annual_vol = port_r.std() * np.sqrt(252)
                        downside_std = port_r.clip(upper=0).std() * np.sqrt(252)
                        sharpe = annual_return / (annual_vol + 1e-8)
                        sortino = annual_return / (downside_std + 1e-8)
                        var95 = port_r.quantile(0.05)
                        cvar95 = port_r[port_r <= var95].mean()
                        max_dd = compute_max_drawdown(portfolio_value)

                        c1, c2, c3, c4, c5 = st.columns(5)
                        c1.metric("Rendement annualisé", f"{annual_return:.2%}")
                        c2.metric("Volatilité annualisée", f"{annual_vol:.2%}")
                        c3.metric("Sharpe", f"{sharpe:.2f}")
                        c4.metric("Sortino", f"{sortino:.2f}")
                        c5.metric("Max Drawdown", f"{max_dd:.2%}")
                        st.caption(
                            f"VaR 95% (jour) : {var95:.2%} | CVaR 95% (jour) : {cvar95:.2%}"
                        )

                        base100 = portfolio_value / portfolio_value.iloc[0] * 100
                        fig = px.line(
                            base100,
                            labels={"value": "Base 100", "index": "Date"},
                            title="Performance historique basée sur les positions",
                        )
                        st.plotly_chart(fig, use_container_width=True)

elif vue == "Optimisation / Markowitz":
    st.subheader("Optimisation de portefeuille (Markowitz - Long Only)")
    w = compute_markowitz(returns)
    df = pd.DataFrame({"Ticker": returns.columns, "Poids optimaux": w})
    st.dataframe(df.style.format({"Poids optimaux": "{:.2%}"}))

elif vue == "Diversification / Clusters":
    st.subheader("Clustering des actifs (K-Means)")
    k = st.slider("Nombre de clusters", 2, min(10, len(returns.columns)), 4)
    labels = compute_clusters(returns, k)
    df = pd.DataFrame({"Ticker": returns.columns, "Cluster": labels})

    st.dataframe(df.sort_values(by="Cluster"))

    fig = px.scatter(
        x=returns.mean() * 252,
        y=returns.std() * np.sqrt(252),
        color=labels.astype(str),
        text=returns.columns,
        labels={"x": "Rendement annuel attendu", "y": "Volatilité annuelle"},
        title="Groupement des actifs Risque vs Rendement",
    )
    st.plotly_chart(fig, use_container_width=True)

elif vue == "Rendement / Volatilité / Sharpe":
    st.subheader("Analyse Comparative Filtre : Performance, Risque & Forme")
    
    # Application des filtres sur la table de données
    advanced_perf = raw_perf.copy()
    
    filtered_df = advanced_perf[
        (advanced_perf["Ratio de Sharpe"] >= f_sharpe) & 
        (advanced_perf["Volatilité (Sigma)"] <= f_vol) & 
        (advanced_perf["Skewness (Asymétrie)"] >= f_skew)
    ]
    
    # Affichage du nombre d'actifs restants
    st.write(f"🔍 **{len(filtered_df)}** actifs correspondent actuellement à vos critères sur un total de {len(advanced_perf)}.")

    if filtered_df.empty:
        st.warning("⚠️ Aucun actif ne correspond à cette combinaison de filtres. Essayez d'élargir vos critères.")
    else:
        # Affichage du tableau filtré
        st.dataframe(
            filtered_df.style.format(
                {
                    "Rendement Annuel": "{:.2%}",
                    "Volatilité (Sigma)": "{:.2%}",
                    "Ratio de Sharpe": "{:.2f}",
                    "Ratio de Sortino": "{:.2f}",
                    "VaR 95% (Jour)": "{:.2%}",
                    "CVaR 95% (Jour)": "{:.2%}",
                    "Skewness (Asymétrie)": "{:.2f}",
                    "Kurtosis (Aplatissement)": "{:.2f}",
                }
            ).background_gradient(
                cmap="RdYlGn", 
                subset=["Rendement Annuel", "Ratio de Sharpe", "Ratio de Sortino"]
            ).background_gradient(
                cmap="RdYlGn_r", 
                subset=["Volatilité (Sigma)", "VaR 95% (Jour)", "CVaR 95% (Jour)"]
            )
        )

        # Bouton pour exporter uniquement la sélection filtrée
        csv_stats = filtered_df.to_csv().encode("utf-8")
        st.download_button("Télécharger la liste réduite (CSV)", csv_stats, "liste_actifs_filtrés.csv")

        # Graphique dynamique mis à jour uniquement avec la liste réduite
        fig = px.bar(
            filtered_df.reset_index(),
            x="index",
            y="Ratio de Sharpe",
            color="Ratio de Sharpe",
            color_continuous_scale="RdYlGn",
            labels={"index": "Ticker"},
            title="Ratio de Sharpe des actifs sélectionnés",
        )
        st.plotly_chart(fig, use_container_width=True)

elif vue == "Distribution des gains/pertes":
    st.subheader("Distribution des rendements journaliers (%)")

    t = st.selectbox("Choisir un ticker", returns.columns)
    r = returns[t].dropna() * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Moyenne (%)", f"{r.mean():.2f}%")
    col2.metric("Médiane (%)", f"{r.median():.2f}%")
    col3.metric("Skewness", f"{r.skew():.2f}")
    col4.metric("Kurtosis", f"{r.kurtosis():.2f}")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Histogram(
            x=r,
            nbinsx=80,
            histnorm="probability density",
            marker_color="lightgreen" if r.mean() > 0 else "lightcoral",
            opacity=0.6,
            name="Densité Histogramme",
        ),
        secondary_y=False,
    )

    x_grid = np.linspace(r.min(), r.max(), 300)
    kde = gaussian_kde(r)
    y_kde = kde(x_grid)

    fig.add_trace(
        go.Scatter(
            x=x_grid,
            y=y_kde,
            mode="lines",
            line=dict(color="black", width=2),
            name="Courbe Densité (KDE)",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Box(
            x=r,
            name="Boxplot",
            marker_color="darkgreen" if r.median() > 0 else "darkred",
            boxpoints="outliers",
            orientation="h",
        ),
        secondary_y=True,
    )

    fig.update_layout(title_text=f"Analyse de la distribution pour {t}")
    fig.update_yaxes(title_text="Densité de probabilité", secondary_y=False)
    fig.update_yaxes(showticklabels=False, secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)

elif vue == "Dividendes & Qualité":
    if HAS_DIVIDENDES:
        try:
            st.caption(
                "Mode robuste active: analyse par lots recommandee pour limiter les erreurs de saturation."
            )
            # Appel direct de la fonction du module externe
            dividendes.render_dividendes_dashboard(prices, returns)
        except Exception as e:
            st.error(f"Erreur lors de l'exécution du module dividendes : {e}")
    else:
        st.info(
            "Le fichier 'dividendes.py' contient des erreurs de structure ou est introuvable. "
            "Vérifiez le message d'erreur rouge affiché dans la barre latérale gauche."
        )

elif vue == "Analyse fondamentale":
    if HAS_FUNDAMENTALS:
        try:
            st.caption(
                "Ratios (PER, dette/FCF) et consensus analystes via Yahoo Finance / YahooQuery."
            )
            fundamentals.render_fundamentals_dashboard(prices)
        except Exception as e:
            st.error(f"Erreur lors de l'exécution du module fundamentals : {e}")
    else:
        st.info(
            "Le fichier 'fundamentals.py' est introuvable ou contient une erreur. "
            "Vérifiez le message d'erreur dans la barre latérale."
        )