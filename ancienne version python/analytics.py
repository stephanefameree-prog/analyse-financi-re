import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def compute_returns(prices):
    return prices.pct_change().dropna(how="all")


def compute_corr(returns):
    return returns.corr()


def compute_cov(returns):
    return returns.cov()


def compute_pca(returns):
    df_filled = returns.fillna(0)
    scaler = StandardScaler()
    X = scaler.fit_transform(df_filled)
    pca = PCA()
    pca.fit(X)
    return pca, X


def compute_markowitz(returns):
    df_filled = returns.fillna(0)
    mu = df_filled.mean() * 252
    cov = df_filled.cov() * 252
    noa = len(returns.columns)

    def min_func_sharpe(weights):
        p_return = np.sum(mu * weights)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        return -p_return / (p_vol + 1e-8)

    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bounds = tuple((0.0, 1.0) for _ in range(noa))
    init_guess = noa * [1.0 / noa]

    opts = minimize(
        min_func_sharpe,
        init_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not opts.success:
        return pd.Series(index=returns.columns, data=1.0 / noa)

    return pd.Series(opts.x, index=returns.columns)


def compute_clusters(returns, k):
    df_filled = returns.fillna(0).T
    scaler = StandardScaler()
    X = scaler.fit_transform(df_filled)
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    return labels


def compute_advanced_risk_metrics(returns):
    annual_return = returns.mean() * 252
    annual_vol = returns.std() * np.sqrt(252)
    sharpe = annual_return / (annual_vol + 1e-8)
    downside_std = returns.clip(upper=0).std() * np.sqrt(252)
    sortino = annual_return / (downside_std + 1e-8)
    var_95 = returns.quantile(0.05)
    cvar_95 = returns.where(returns.le(var_95), np.nan).mean()

    skew = returns.skew()
    kurt = returns.kurtosis()

    df = pd.DataFrame(
        {
            "Rendement Annuel": annual_return,
            "Volatilité (Sigma)": annual_vol,
            "Ratio de Sharpe": sharpe,
            "Ratio de Sortino": sortino,
            "VaR 95% (Jour)": var_95,
            "CVaR 95% (Jour)": cvar_95,
            "Skewness (Asymétrie)": skew,
            "Kurtosis (Aplatissement)": kurt,
        }
    )
    return df


def compute_max_drawdown(price_series):
    s = price_series.dropna()
    if s.empty:
        return np.nan
    running_max = s.cummax()
    drawdown = s / running_max - 1
    return float(drawdown.min())
