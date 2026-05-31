import streamlit as st
from yahooquery import Ticker
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from sklearn.decomposition import PCA

import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

import requests
from bs4 import BeautifulSoup
import re

# ------------------------------------------------------------
# Config Streamlit
# ------------------------------------------------------------
st.set_page_config(page_title="Analyse de portefeuille", layout="wide")
st.title("📊 Dashboard d'analyse de portefeuille — v5 (corrélations & diversification)")

# ------------------------------------------------------------
# Extraction robuste des tickers via BeautifulSoup + cache
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_tickers_from_wikipedia(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        raise Exception(f"HTTP {response.status_code} en accédant à {url}")

    soup = BeautifulSoup(response.text, "html.parser")
    tables = soup.find_all("table")

    if not tables:
        raise Exception("Aucun tableau trouvé sur la page Wikipedia.")

    tickers = []
    ticker_regex = re.compile(r"^[A-Z0-9.\-]{2,10}$")

    for table in tables:
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all(["td", "th"])
            for cell in cells:
                text = cell.get_text(strip=True)
                if ticker_regex.match(text):
                    tickers.append(text)

    tickers = list(dict.fromkeys(tickers))

    if len(tickers) == 0:
        raise Exception("Impossible d'extraire des tickers — structure inattendue.")

    return tickers

def load_cac40():
    return load_tickers_from_wikipedia("https://en.wikipedia.org/wiki/CAC_40")

def load_bel20():
    return load_tickers_from_wikipedia("https://en.wikipedia.org/wiki/BEL_20")

def load_dax():
    return load_tickers_from_wikipedia("https://en.wikipedia.org/wiki/DAX")

def load_sp500():
    return load_tickers_from_wikipedia("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")

def load_djia():
    return load_tickers_from_wikipedia("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average")

def load_nasdaq100():
    return load_tickers_from_wikipedia("https://en.wikipedia.org/wiki/NASDAQ-100")

def load_stoxx600():
    return load_tickers_from_wikipedia("https://en.wikipedia.org/wiki/STOXX_Europe_600")

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
st.sidebar.header("Paramètres")

default_tickers = ["AAPL", "MSFT", "GOOGL"]

sources = st.sidebar.multiselect(
    "Sources des tickers",
    [
        "Manuel",
        "CAC40",
        "DAX",
        "BEL20",
        "NASDAQ100",
        "S&P500",
        "Dow Jones (DJIA)",
        "STOXX Europe 600"
    ],
    default=["CAC40"]
)

tickers = []

for source in sources:
    if source == "CAC40":
        tickers += load_cac40()
    elif source == "DAX":
        tickers += load_dax()
    elif source == "BEL20":
        tickers += load_bel20()
    elif source == "NASDAQ100":
        tickers += load_nasdaq100()
    elif source == "S&P500":
        tickers += load_sp500()
    elif source == "Dow Jones (DJIA)":
        tickers += load_djia()
    elif source == "STOXX Europe 600":
        tickers += load_stoxx600()
    elif source == "Manuel":
        tickers += st.sidebar.text_area(
            "Tickers manuels",
            ", ".join(default_tickers)
        ).replace(" ", "").split(",")

tickers = list(dict.fromkeys(tickers))

start = st.sidebar.date_input("Date de début", datetime(2022, 1, 1))
end = st.sidebar.date_input("Date de fin", datetime.today())

vue = st.sidebar.selectbox(
    "Vue",
    [
        "Vue globale",
        "Corrélations avancées",
        "PCA / Facteurs",
        "Analyse par actif",
        "Optimisation / Markowitz",
        "Diversification / Clusters"
    ]
)

# ------------------------------------------------------------
# Chargement des prix avec cache + progress bar
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_single_ticker_history(ticker, start, end):
    df = Ticker(ticker).history(start=start, end=end)
    if isinstance(df, pd.DataFrame) and "close" in df.columns:
        s = df["close"].droplevel(0)
        if len(s) > 0:
            return s
    return None

st.subheader("📥 Téléchargement des données")

prices_dict = {}
if len(tickers) == 0:
    st.error("Aucun ticker sélectionné.")
    st.stop()

progress = st.progress(0)
status = st.empty()

with st.spinner(f"Chargement des données Yahoo Finance pour {len(tickers)} tickers…"):
    for i, t in enumerate(tickers):
        status.text(f"Chargement {i+1}/{len(tickers)} : {t}")
        s = load_single_ticker_history(t, start, end)
        if s is not None:
            prices_dict[t] = s
        progress.progress((i + 1) / len(tickers))

status.empty()
progress.empty()

if len(prices_dict) == 0:
    st.error("Aucune donnée de prix disponible.")
    st.stop()

prices = pd.DataFrame(prices_dict).dropna()
returns = prices.pct_change().dropna()
corr = returns.corr()

# ------------------------------------------------------------
# Fonctions utilitaires
# ------------------------------------------------------------
def compute_markowitz(returns, n_portfolios=3000):
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    results = np.zeros((3, n_portfolios))
    weights_record = []
    for i in range(n_portfolios):
        weights = np.random.random(len(returns.columns))
        weights /= np.sum(weights)
        weights_record.append(weights)
        ret = np.sum(mean_returns * weights) * 252
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        sharpe = ret / vol if vol > 0 else 0
        results[0, i] = vol
        results[1, i] = ret
        results[2, i] = sharpe
    return results, weights_record

def corr_network_figure(corr, threshold=0.6, clusters=None):
    G = nx.Graph()
    for col in corr.columns:
        G.add_node(col)
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            c = corr.iloc[i, j]
            if abs(c) >= threshold:
                G.add_edge(corr.index[i], corr.columns[j], weight=c)

    pos = nx.spring_layout(G, seed=42)
    edge_x, edge_y = [], []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(color="lightgray"))

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]

    if clusters is not None:
        colors = [clusters.get(n, 0) for n in G.nodes()]
    else:
        colors = "blue"

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=list(G.nodes()),
        textposition="top center",
        marker=dict(
            size=12,
            color=colors,
            colorscale="Turbo",
            showscale=True if clusters is not None else False
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(title="Graphe des corrélations")
    return fig

def compute_clusters_from_corr(corr, max_clusters=8):
    dist = 1 - corr
    link = linkage(dist, method='ward')
    # nombre de clusters déterminé automatiquement mais borné
    k = min(max_clusters, max(2, int(np.sqrt(len(corr.columns)))))
    labels = fcluster(link, k, criterion='maxclust')
    cluster_map = {ticker: int(label) for ticker, label in zip(corr.columns, labels)}
    return cluster_map, link

# ------------------------------------------------------------
# VUES
# ------------------------------------------------------------

# --- Vue globale ---
if vue == "Vue globale":
    st.subheader("🔥 Heatmap triangulaire + clustering")
    link = linkage(corr, method='ward')
    idx = leaves_list(link)
    corr_clustered = corr.iloc[idx, :].iloc[:, idx]

    fig = px.imshow(
        corr_clustered,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📈 Évolution normalisée")
    norm = prices / prices.iloc[0] * 100
    fig = px.line(norm)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Histogrammes des rendements")
    fig = px.histogram(returns, nbins=50, opacity=0.7)
    st.plotly_chart(fig, use_container_width=True)

# --- Corrélations avancées ---
elif vue == "Corrélations avancées":
    st.subheader("🔗 Graphe des corrélations")
    threshold = st.slider("Seuil |corr|", 0.3, 0.99, 0.9, 0.05)
    fig = corr_network_figure(corr, threshold=threshold)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📐 Matrice complète")
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    st.plotly_chart(fig, use_container_width=True)

# --- PCA ---
elif vue == "PCA / Facteurs":
    st.subheader("🧩 PCA")
    pca = PCA(n_components=2)
    X = returns.values
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    comps = pca.fit_transform(X_std)

    df_pca = pd.DataFrame({"PC1": comps[:, 0], "PC2": comps[:, 1]})
    fig = px.scatter(df_pca, x="PC1", y="PC2", opacity=0.6)
    st.plotly_chart(fig, use_container_width=True)

    st.write("Variance expliquée :")
    st.write(f"PC1 : {pca.explained_variance_ratio_[0]:.2%}")
    st.write(f"PC2 : {pca.explained_variance_ratio_[1]:.2%}")

# --- Analyse par actif ---
elif vue == "Analyse par actif":
    st.subheader("🔍 Analyse individuelle")
    ticker_sel = st.selectbox("Choisir un ticker", tickers)

    fig = px.line(prices[ticker_sel], title=f"Prix de {ticker_sel}")
    st.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(returns[ticker_sel], nbins=50)
    st.plotly_chart(fig, use_container_width=True)

    st.write("📌 Statistiques")
    st.write(returns[ticker_sel].describe())

# --- Optimisation ---
elif vue == "Optimisation / Markowitz":
    st.subheader("🧭 Frontière efficiente")
    n_port = st.slider("Nombre de portefeuilles", 500, 10000, 3000, 500)
    results, weights_record = compute_markowitz(returns, n_port)

    df_mc = pd.DataFrame({
        "Volatilité": results[0],
        "Rendement": results[1],
        "Sharpe": results[2]
    })

    fig = px.scatter(df_mc, x="Volatilité", y="Rendement", color="Sharpe")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("⭐ Portefeuille max Sharpe")
    idx = np.argmax(results[2])
    best_weights = weights_record[idx]

    df_w = pd.DataFrame({"Ticker": tickers, "Poids": best_weights})
    df_w = df_w.sort_values("Poids", ascending=False)
    st.dataframe(df_w.style.format({"Poids": "{:.2%}"}))

    csv = df_w.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Télécharger CSV", csv, "max_sharpe.csv")

# --- Diversification / Clusters ---
elif vue == "Diversification / Clusters":
    st.subheader("🧬 Clusters de corrélation & diversification")

    cluster_map, link = compute_clusters_from_corr(corr)
    fig = corr_network_figure(corr, threshold=0.6, clusters=cluster_map)
    st.plotly_chart(fig, use_container_width=True)

    df_clusters = pd.DataFrame({
        "Ticker": list(cluster_map.keys()),
        "Cluster": list(cluster_map.values())
    }).sort_values(["Cluster", "Ticker"])

    st.subheader("📋 Répartition par cluster")
    st.dataframe(df_clusters)

    cluster_counts = df_clusters["Cluster"].value_counts().sort_index()
    fig_bar = px.bar(
        x=cluster_counts.index.astype(str),
        y=cluster_counts.values,
        labels={"x": "Cluster", "y": "Nombre de titres"},
        title="Nombre de titres par cluster"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Score simple de diversification
    n_clusters = cluster_counts.shape[0]
    n_assets = df_clusters.shape[0]
    max_cluster = cluster_counts.max()

    diversification_score = n_clusters / n_assets
    concentration_score = max_cluster / n_assets

    st.markdown("### 🔎 Diagnostic de diversification")
    st.write(f"- Nombre total de titres : **{n_assets}**")
    st.write(f"- Nombre de clusters détectés : **{n_clusters}**")
    st.write(f"- Taille du plus gros cluster : **{max_cluster}** titres")
    st.write(f"- Score de diversification (plus proche de 1 = mieux) : **{diversification_score:.2f}**")
    st.write(f"- Score de concentration (plus proche de 0 = mieux) : **{concentration_score:.2f}**")

    st.markdown("### 🧭 Interprétation rapide")
    if concentration_score > 0.5:
        st.warning("Ton risque est fortement concentré dans un seul cluster.")
    elif concentration_score > 0.3:
        st.info("Ton portefeuille présente une concentration modérée dans certains clusters.")
    else:
        st.success("Ton portefeuille est relativement bien diversifié entre plusieurs clusters.")
