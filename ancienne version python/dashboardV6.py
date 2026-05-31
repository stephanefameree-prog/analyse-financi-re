import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from dividendes import render_dividendes_dashboard

st.set_page_config(layout="wide")

@st.cache_data
def load_prices(tickers, start):
    data = yf.download(tickers, start=start)["Adj Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    return data.dropna(how="all")

def compute_returns(prices):
    return prices.pct_change().dropna()

def compute_corr(returns):
    return returns.corr()

def compute_pca(returns):
    scaler = StandardScaler()
    X = scaler.fit_transform(returns)
    pca = PCA()
    pca.fit(X)
    return pca, X

def compute_markowitz(returns):
    mu = returns.mean() * 252
    cov = returns.cov() * 252
    inv = np.linalg.inv(cov)
    w = inv @ mu
    w = w / w.sum()
    return w

def compute_clusters(returns, k):
    scaler = StandardScaler()
    X = scaler.fit_transform(returns.T)
    km = KMeans(n_clusters=k, n_init=10)
    labels = km.fit_predict(X)
    return labels

st.sidebar.title("Dashboard v6")
tickers = st.sidebar.text_input("Tickers (séparés par des espaces)", "AAPL MSFT NVDA META AMZN JPM V")
start = st.sidebar.date_input("Date de début", pd.to_datetime("2015-01-01"))
vue = st.sidebar.selectbox(
    "Vue",
    [
        "Vue globale",
        "Corrélations avancées",
        "PCA / Facteurs",
        "Analyse par actif",
        "Optimisation / Markowitz",
        "Diversification / Clusters",
        "Dividendes & Qualité"
    ]
)

tickers = tickers.split()
prices = load_prices(tickers, start)
returns = compute_returns(prices)

if vue == "Vue globale":
    st.subheader("Vue globale")
    st.line_chart(prices)

elif vue == "Corrélations avancées":
    st.subheader("Corrélations")
    corr = compute_corr(returns)
    fig = px.imshow(corr, text_auto=True, aspect="auto")
    st.plotly_chart(fig, use_container_width=True)

elif vue == "PCA / Facteurs":
    st.subheader("PCA")
    pca, X = compute_pca(returns)
    exp = pca.explained_variance_ratio_
    fig = px.bar(x=list(range(1, len(exp)+1)), y=exp)
    st.plotly_chart(fig, use_container_width=True)

elif vue == "Analyse par actif":
    st.subheader("Analyse par actif")
    t = st.selectbox("Choisir un ticker", tickers)
    st.line_chart(prices[t])
    st.line_chart(returns[t])

elif vue == "Optimisation / Markowitz":
    st.subheader("Optimisation de portefeuille")
    w = compute_markowitz(returns)
    df = pd.DataFrame({"Ticker": tickers, "Poids": w})
    st.dataframe(df)

elif vue == "Diversification / Clusters":
    st.subheader("Clusters")
    k = st.slider("Nombre de clusters", 2, 10, 4)
    labels = compute_clusters(returns, k)
    df = pd.DataFrame({"Ticker": tickers, "Cluster": labels})
    st.dataframe(df)
    fig = px.scatter(x=returns.mean(), y=returns.std(), color=labels, text=tickers)
    st.plotly_chart(fig, use_container_width=True)

elif vue == "Dividendes & Qualité":
    render_dividendes_dashboard(prices, returns)
