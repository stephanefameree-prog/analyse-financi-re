import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def compute_returns(prices):
    return prices.pct_change().dropna(how="all")


def compute_corr(returns):
    return returns.corr()


def cluster_correlation_order(corr: pd.DataFrame) -> list:
    """Ordre des tickers par clustering hiérarchique (corrélations proches regroupées)."""
    if corr is None or corr.empty or len(corr) < 3:
        return list(corr.index) if corr is not None and not corr.empty else []
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform

    mat = corr.values.astype(float).copy()
    np.fill_diagonal(mat, 1.0)
    dist = np.sqrt(np.clip(2.0 * (1.0 - mat), 0.0, None))
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method="ward")
    order = leaves_list(link)
    return [corr.index[i] for i in order]


def build_correlation_heatmap_figure(
    corr: pd.DataFrame,
    *,
    cluster: bool = True,
    triangle: bool = True,
    rho_min: float = -1.0,
    rho_max: float = 1.0,
    title: str = "Matrice de corrélation (rendements journaliers)",
):
    """
    Heatmap ρ avec palette divergente, clustering optionnel et masque triangle supérieur.
    Seules les cellules avec rho_min ≤ ρ ≤ rho_max sont affichées (diagonale incluse).
    """
    from chart_theme import CHART_HEIGHT, CORRELATION_COLORSCALE, apply_chart_theme

    if corr is None or corr.empty:
        return apply_chart_theme(go.Figure(), height=CHART_HEIGHT["heatmap"], title=title)

    rho_min = float(max(-1.0, min(1.0, rho_min)))
    rho_max = float(max(-1.0, min(1.0, rho_max)))
    if rho_min > rho_max:
        rho_min, rho_max = rho_max, rho_min

    display = corr.astype(float).copy()
    if cluster and len(display) >= 3:
        order = cluster_correlation_order(display)
        display = display.loc[order, order]

    z = display.values.copy()
    labels = [str(x) for x in display.index]

    if triangle and len(display) >= 2:
        lower = np.tril(np.ones_like(z, dtype=bool), k=-1)
        z = np.where(lower, np.nan, z)

    filtered = rho_min > -1.0 + 1e-9 or rho_max < 1.0 - 1e-9
    if filtered:
        out_of_range = (z < rho_min) | (z > rho_max)
        z = np.where(out_of_range, np.nan, z)

    scale_min, scale_max = (-1.0, 1.0)
    if filtered:
        scale_min, scale_max = rho_min, rho_max
        if scale_min == scale_max:
            scale_min -= 0.05
            scale_max += 0.05

    text = [
        [f"{val:.2f}" if not np.isnan(val) else "" for val in row]
        for row in z
    ]

    zmid = 0.0 if scale_min < 0 < scale_max else (scale_min + scale_max) / 2.0

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            colorscale=CORRELATION_COLORSCALE,
            zmid=zmid,
            zmin=scale_min,
            zmax=scale_max,
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=10),
            hovertemplate="%{y} ↔ %{x}<br>ρ = %{z:.2f}<extra></extra>",
            colorbar=dict(title="ρ", tickformat=".2f", len=0.75),
        )
    )
    fig.update_xaxes(tickangle=-45, side="bottom", tickfont=dict(size=10))
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=10))
    apply_chart_theme(
        fig,
        height=max(CHART_HEIGHT["heatmap"], 36 * len(labels) + 120),
        title=title,
        legend_horizontal=False,
    )
    return fig


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

    if noa == 0:
        return pd.Series()

    def min_func_sharpe(weights):
        p_return = np.sum(mu * weights)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        return -p_return / (p_vol + 1e-8)

    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bounds = tuple((0.0, 1.0) for _ in range(noa))
    init_guess = np.array(noa * [1.0 / noa])

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


def _annualized_mu_cov(returns):
    """Rendements moyens et covariance annualisés (252 séances)."""
    df_filled = returns.fillna(0)
    mu = df_filled.mean().values * 252
    cov = df_filled.cov().values * 252
    tickers = list(returns.columns)
    return mu, cov, tickers


def _portfolio_performance(weights, mu, cov):
    ret = float(np.dot(weights, mu))
    vol = float(np.sqrt(np.dot(weights.T, np.dot(cov, weights))))
    sharpe = ret / (vol + 1e-12)
    return ret, vol, sharpe


def _align_weights_to_returns(returns, weights):
    """Aligne une Series de poids sur les colonnes de returns (0 si absent)."""
    tickers = list(returns.columns)
    w = pd.Series(0.0, index=tickers)
    if weights is None or len(weights) == 0:
        return w
    for tk, val in weights.items():
        if tk in w.index:
            w[tk] = float(val or 0)
    total = w.sum()
    if total > 0:
        w = w / total
    return w


def compute_holdings_weights(prices, quantities, usd_to_eur=1.0, is_usd_fn=None, universe=None):
    """Poids de marché actuels (valorisation / total) en EUR."""
    is_usd_fn = is_usd_fn or (lambda _t: False)
    if prices is None or prices.empty:
        return pd.Series(dtype=float)
    last = prices.iloc[-1]
    tickers = universe or [t for t in quantities if t in last.index]
    values = {}
    for tk in tickers:
        q = float(quantities.get(tk, 0) or 0)
        if q <= 0 or tk not in last.index or pd.isna(last[tk]):
            continue
        rate = usd_to_eur if is_usd_fn(tk) else 1.0
        values[tk] = q * float(last[tk]) * rate
    total = sum(values.values())
    if total <= 0:
        return pd.Series(dtype=float)
    return pd.Series({k: v / total for k, v in values.items()})


def portfolio_market_value_eur(prices, quantities, usd_to_eur=1.0, is_usd_fn=None):
    """Valorisation totale du portefeuille en EUR (dernière séance)."""
    is_usd_fn = is_usd_fn or (lambda _t: False)
    if prices is None or prices.empty:
        return 0.0
    last = prices.iloc[-1]
    total = 0.0
    for tk, q in quantities.items():
        q = float(q or 0)
        if q <= 0 or tk not in last.index or pd.isna(last[tk]):
            continue
        rate = usd_to_eur if is_usd_fn(tk) else 1.0
        total += q * float(last[tk]) * rate
    return total


def build_markowitz_arbitrage_table(current_weights, target_weights, labels=None):
    """Tableau comparatif poids actuels vs cibles Markowitz."""
    tickers = list(target_weights.index)
    if labels is None:
        labels = {tk: tk for tk in tickers}
    rows = []
    for tk in tickers:
        cur = float(current_weights.get(tk, 0) or 0)
        tgt = float(target_weights.get(tk, 0) or 0)
        rows.append(
            {
                "Ticker": tk,
                "Libellé": labels.get(tk, tk),
                "Poids actuel": cur,
                "Poids cible (Markowitz)": tgt,
                "Écart (pts)": tgt - cur,
            }
        )
    df = pd.DataFrame(rows).sort_values("Poids actuel", ascending=False)
    return df


def portfolio_performance_from_weights(returns, weights):
    """Rendement, volatilité et Sharpe annualisés pour un vecteur de poids."""
    mu, cov, tickers = _annualized_mu_cov(returns)
    w = _align_weights_to_returns(returns, weights).values
    if w.sum() <= 0:
        return None
    ret, vol, sharpe = _portfolio_performance(w, mu, cov)
    return {"Rendement": ret, "Volatilité": vol, "Sharpe": sharpe}


def simulate_random_portfolios(returns, n_portfolios=2000, seed=42):
    """Portefeuilles aléatoires long-only (somme des poids = 1)."""
    mu, cov, tickers = _annualized_mu_cov(returns)
    n_assets = len(tickers)
    if n_assets == 0:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_portfolios):
        w = rng.random(n_assets)
        w /= w.sum()
        ret, vol, sharpe = _portfolio_performance(w, mu, cov)
        rows.append({"Volatilité": vol, "Rendement": ret, "Sharpe": sharpe})
    return pd.DataFrame(rows)


def compute_efficient_frontier(returns, n_points=50):
    """Frontière efficiente (min variance pour rendements cibles)."""
    mu, cov, tickers = _annualized_mu_cov(returns)
    n_assets = len(tickers)
    if n_assets < 2:
        return pd.DataFrame()

    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    init = np.array(n_assets * [1.0 / n_assets])
    eq_sum = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    def _min_variance(extra_cons=()):
        def objective(w):
            return np.sqrt(np.dot(w.T, np.dot(cov, w)))

        res = minimize(
            objective,
            init,
            method="SLSQP",
            bounds=bounds,
            constraints=(eq_sum,) + extra_cons,
        )
        if not res.success:
            return None, None, None
        w = res.x
        return w, *_portfolio_performance(w, mu, cov)[:2]

    w_min, ret_min, vol_min = _min_variance()
    if w_min is None:
        return pd.DataFrame()

    def _max_return():
        res = minimize(
            lambda w: -np.dot(w, mu),
            init,
            method="SLSQP",
            bounds=bounds,
            constraints=(eq_sum,),
        )
        if not res.success:
            return float(np.max(mu))
        return float(np.dot(res.x, mu))

    ret_max = _max_return()
    targets = np.linspace(ret_min, ret_max, n_points)
    frontier_rows = []
    for target in targets:
        ret_con = {
            "type": "eq",
            "fun": lambda w, t=target: np.dot(w, mu) - t,
        }
        _, ret, vol = _min_variance(extra_cons=(ret_con,))
        if ret is not None and vol is not None:
            frontier_rows.append({"Volatilité": vol, "Rendement": ret})

    return pd.DataFrame(frontier_rows)


def build_markowitz_frontier_figure(
    returns,
    optimal_weights,
    risk_free_rate=0.0,
    n_random=2000,
    title="Frontière efficiente et portefeuilles aléatoires",
    current_weights=None,
):
    """Nuage Monte Carlo, frontière efficiente, portefeuille max Sharpe et CAL."""
    mu, cov, tickers = _annualized_mu_cov(returns)
    if len(tickers) == 0:
        return go.Figure()

    random_df = simulate_random_portfolios(returns, n_portfolios=n_random)
    frontier_df = compute_efficient_frontier(returns)

    w_opt = _align_weights_to_returns(returns, optimal_weights).values
    opt_ret, opt_vol, opt_sharpe = _portfolio_performance(w_opt, mu, cov)

    fig = go.Figure()

    if not random_df.empty:
        fig.add_trace(
            go.Scatter(
                x=random_df["Volatilité"],
                y=random_df["Rendement"],
                mode="markers",
                name="Portefeuilles aléatoires",
                marker=dict(size=5, color="#ef5350", opacity=0.35, symbol="diamond"),
                hovertemplate="σ=%{x:.2%}<br>R=%{y:.2%}<extra>Aléatoire</extra>",
            )
        )

    if not frontier_df.empty:
        frontier_df = frontier_df.sort_values("Volatilité")
        fig.add_trace(
            go.Scatter(
                x=frontier_df["Volatilité"],
                y=frontier_df["Rendement"],
                mode="lines",
                name="Frontière efficiente",
                line=dict(color="#1565c0", width=2.5),
                hovertemplate="σ=%{x:.2%}<br>R=%{y:.2%}<extra>Efficient</extra>",
            )
        )

    if current_weights is not None and _align_weights_to_returns(returns, current_weights).sum() > 0:
        w_cur = _align_weights_to_returns(returns, current_weights).values
        cur_ret, cur_vol, cur_sharpe = _portfolio_performance(w_cur, mu, cov)
        fig.add_trace(
            go.Scatter(
                x=[cur_vol],
                y=[cur_ret],
                mode="markers",
                name=f"Portefeuille actuel (Sharpe {cur_sharpe:.2f})",
                marker=dict(size=16, color="#7b1fa2", symbol="star", line=dict(color="#333", width=1)),
                hovertemplate=(
                    f"σ={cur_vol:.2%}<br>R={cur_ret:.2%}<br>Sharpe={cur_sharpe:.2f}"
                    "<extra>Actuel</extra>"
                ),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[opt_vol],
            y=[opt_ret],
            mode="markers",
            name=f"Après arbitrage (Sharpe {opt_sharpe:.2f})",
            marker=dict(size=16, color="#ffd600", symbol="star", line=dict(color="#333", width=1)),
            hovertemplate=(
                f"σ={opt_vol:.2%}<br>R={opt_ret:.2%}<br>Sharpe={opt_sharpe:.2f}"
                "<extra>Optimal</extra>"
            ),
        )
    )

    if risk_free_rate > 0 and opt_vol > 0:
        cal_x = np.linspace(0, max(opt_vol * 1.6, frontier_df["Volatilité"].max() if not frontier_df.empty else opt_vol * 1.6), 50)
        cal_y = risk_free_rate + (opt_ret - risk_free_rate) / opt_vol * cal_x
        fig.add_trace(
            go.Scatter(
                x=cal_x,
                y=cal_y,
                mode="lines",
                name="CAL (actif sans risque)",
                line=dict(color="#212121", width=3),
                hovertemplate="R=%{y:.2%}<br>σ=%{x:.2%}<extra>CAL</extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[risk_free_rate],
                mode="markers",
                name="Actif sans risque",
                marker=dict(size=10, color="#212121", symbol="circle"),
                hovertemplate=f"R={risk_free_rate:.2%}<br>σ=0%<extra>Sans risque</extra>",
            )
        )

    fig.update_layout(
        xaxis_title="Risque (écart-type annualisé)",
        yaxis_title="Rendement espéré (annualisé)",
        hovermode="closest",
    )
    fig.update_xaxes(tickformat=".1%")
    fig.update_yaxes(tickformat=".1%")
    from chart_theme import CHART_HEIGHT, apply_chart_theme

    apply_chart_theme(fig, height=CHART_HEIGHT["frontier"], title=title)
    return fig


def build_markowitz_assets_figure(
    returns,
    weights,
    labels=None,
    title="Actifs — rendement vs risque (taille = poids)",
    portfolio_label="Portefeuille",
    portfolio_color="#ffd600",
    colorscale="Viridis",
):
    """Chaque actif : σ, rendement annualisés ; taille du point = poids du portefeuille."""
    mu, cov, tickers = _annualized_mu_cov(returns)
    if len(tickers) == 0:
        return go.Figure()

    asset_vol = returns.fillna(0).std().values * np.sqrt(252)
    asset_ret = returns.fillna(0).mean().values * 252
    w_aligned = _align_weights_to_returns(returns, weights)
    weights_arr = w_aligned.values
    if weights_arr.max() > 0:
        size = np.where(weights_arr > 0, 12 + 48 * (weights_arr / weights_arr.max()), 8)
    else:
        size = np.full(len(tickers), 12)

    if labels is None:
        labels = tickers
    hover = [
        f"{lab}<br>σ={vol:.2%}<br>R={ret:.2%}<br>Poids={wt:.2%}"
        for lab, vol, ret, wt in zip(labels, asset_vol, asset_ret, weights_arr)
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=asset_vol,
            y=asset_ret,
            mode="markers+text",
            text=labels,
            textposition="top center",
            textfont=dict(size=9),
            name="Actifs",
            marker=dict(
                size=size,
                color=weights_arr,
                colorscale=colorscale,
                cmin=0,
                cmax=max(weights_arr.max(), 1e-6),
                showscale=True,
                colorbar=dict(title="Poids"),
                line=dict(color="#333", width=0.8),
                opacity=0.9,
            ),
            hovertext=hover,
            hoverinfo="text",
        )
    )

    if weights_arr.sum() > 0:
        port_ret, port_vol, _ = _portfolio_performance(weights_arr, mu, cov)
        fig.add_trace(
            go.Scatter(
                x=[port_vol],
                y=[port_ret],
                mode="markers",
                name=portfolio_label,
                marker=dict(
                    size=14,
                    color=portfolio_color,
                    symbol="star",
                    line=dict(color="#333", width=1),
                ),
                hovertemplate=f"σ={port_vol:.2%}<br>R={port_ret:.2%}<extra>{portfolio_label}</extra>",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Risque (écart-type annualisé)",
        yaxis_title="Rendement espéré (annualisé)",
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="closest",
    )
    fig.update_xaxes(tickformat=".1%")
    fig.update_yaxes(tickformat=".1%")
    return fig


def compute_clusters(returns, k):
    df_filled = returns.fillna(0).T
    scaler = StandardScaler()
    X = scaler.fit_transform(df_filled)
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    return labels


def estimate_mode(series):
    """Mode estimé par histogramme (rendements journaliers continus)."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 5:
        return np.nan
    n_bins = min(40, max(8, int(np.sqrt(len(s)))))
    counts, edges = np.histogram(s, bins=n_bins)
    if counts.sum() == 0:
        return np.nan
    idx = int(np.argmax(counts))
    return float((edges[idx] + edges[idx + 1]) / 2)


def build_mean_median_mode_pedagogy_chart():
    """Histogramme illustratif : moyenne, médiane et mode (distribution asymétrique)."""
    rng = np.random.default_rng(42)
    daily = np.concatenate(
        [
            rng.normal(0.0002, 0.007, 2200),
            rng.normal(0.028, 0.014, 70),
        ]
    )
    s = pd.Series(daily)
    mean_d = float(s.mean())
    median_d = float(s.median())
    mode_d = estimate_mode(s)
    if np.isnan(mode_d):
        mode_d = median_d

    pct = daily * 100
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=pct,
            nbinsx=45,
            name="Fréquence",
            marker_color="rgba(100, 181, 246, 0.75)",
            marker_line=dict(color="white", width=0.5),
        )
    )

    markers = [
        (mean_d, "#2e7d32", "Moyenne"),
        (median_d, "#ef6c00", "Médiane"),
        (mode_d, "#7b1fa2", "Mode"),
    ]
    for val, color, label in markers:
        x_pct = val * 100
        ann = f"{label}<br>({val * 252:.1%} ann.)"
        fig.add_vline(x=x_pct, line_width=2.5, line_dash="dash", line_color=color)
        fig.add_annotation(
            x=x_pct,
            y=1.02,
            yref="paper",
            text=ann,
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor=color,
            ax=0,
            ay=-28,
            font=dict(color=color, size=11),
            bgcolor="rgba(255,255,255,0.85)",
        )

    fig.update_layout(
        xaxis_title="Rendement journalier (%)",
        yaxis_title="Nombre de journées",
        bargap=0.04,
        showlegend=False,
    )
    from chart_theme import CHART_HEIGHT, apply_chart_theme

    apply_chart_theme(
        fig,
        height=CHART_HEIGHT["pedagogy"],
        title="Où se placent la moyenne, la médiane et le mode ? (exemple pédagogique)",
        legend_horizontal=False,
    )
    return fig


TRADING_DAYS_PER_YEAR = 252

FFT_MA_WINDOW = 41

FFT_TREND_LOG_LINEAR = "log_linear"
FFT_TREND_LINEAR_PRICE = "linear_price"
FFT_TREND_HP = "hp"
FFT_TREND_MA41_CENTER = "ma41_center"
FFT_TREND_MA41_CAUSAL = "ma41_causal"
FFT_TREND_STL = "stl"

FFT_TREND_OPTIONS = {
    FFT_TREND_LOG_LINEAR: "Log-linéaire (croissance % constante)",
    FFT_TREND_LINEAR_PRICE: "Linéaire sur le prix (€)",
    FFT_TREND_HP: "Filtre Hodrick-Prescott",
    FFT_TREND_MA41_CENTER: "MM centrée 41 j (±20 séances)",
    FFT_TREND_MA41_CAUSAL: "MM causale 41 j (sans regard futur)",
    FFT_TREND_STL: "STL (trend + saison 21 j)",
}

HP_LAMBDA_DAILY = 6.25e6
STL_SEASONAL_PERIOD = 21

FFT_RECON_FFT = "fft_peaks"
FFT_RECON_HARMONIC = "harmonic"
FFT_RECON_HARMONIC_FIXED = "harmonic_fixed"

FFT_RECON_OPTIONS = {
    FFT_RECON_FFT: "FFT — pics dominants",
    FFT_RECON_HARMONIC: "Régression harmonique (périodes FFT)",
    FFT_RECON_HARMONIC_FIXED: "Régression harmonique (21–126 j)",
}

HARMONIC_FIXED_PERIODS = [21, 42, 63, 84, 105, 126]


FFT_TREND_LEGEND = {
    FFT_TREND_LOG_LINEAR: "log-linéaire",
    FFT_TREND_LINEAR_PRICE: "linéaire €",
    FFT_TREND_HP: "Hodrick-Prescott",
    FFT_TREND_MA41_CENTER: "MM41 centrée",
    FFT_TREND_MA41_CAUSAL: "MM41 causale",
    FFT_TREND_STL: "STL 21 j",
}


def fft_recon_mode_label(recon_mode):
    return FFT_RECON_OPTIONS.get(recon_mode, str(recon_mode))


def fft_trend_mode_label(trend_mode):
    return FFT_TREND_OPTIONS.get(trend_mode, str(trend_mode))


def fft_trend_legend_label(trend_mode):
    return FFT_TREND_LEGEND.get(trend_mode, str(trend_mode))


def _hodrick_prescott(y, lamb=HP_LAMBDA_DAILY):
    """Filtre HP : sépare tendance lisse et composante cyclique (additif sur le prix)."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 4:
        return y.copy(), np.zeros(n)
    d = np.zeros((n - 2, n))
    for i in range(n - 2):
        d[i, i] = 1.0
        d[i, i + 1] = -2.0
        d[i, i + 2] = 1.0
    trend = np.linalg.solve(np.eye(n) + lamb * d.T @ d, y)
    return trend, y - trend


def _ma41_series(prices, *, centered):
    """Moyenne mobile 41 séances (centrée ±20 ou causale)."""
    s = pd.Series(np.asarray(prices, dtype=float))
    if centered:
        return s.rolling(
            window=FFT_MA_WINDOW,
            center=True,
            min_periods=FFT_MA_WINDOW // 2 + 1,
        ).mean()
    return s.rolling(window=FFT_MA_WINDOW, min_periods=FFT_MA_WINDOW).mean()


def _stl_log_decompose(log_prices, period=STL_SEASONAL_PERIOD):
    """Décomposition STL sur log(prix) ; retourne trend_log, seasonal_log."""
    y = np.asarray(log_prices, dtype=float)
    n = len(y)
    if n < period * 2:
        return None
    try:
        from statsmodels.tsa.seasonal import STL

        fit = STL(y, period=period, robust=True).fit()
        return fit.trend, fit.seasonal
    except ImportError:
        s = pd.Series(y)
        trend = s.rolling(window=period * 2 + 1, center=True, min_periods=period).mean()
        trend = trend.bfill().ffill()
        detrended = s - trend
        seasonal = detrended.groupby(np.arange(n) % period).transform("mean")
        return trend.values, seasonal.values


def _positive_trend(trend_p, ref_p):
    floor = float(np.max(ref_p)) * 1e-8 + 1e-12
    return np.maximum(np.asarray(trend_p, dtype=float), floor)


def _fit_price_trend(prices, t, trend_mode):
    p = np.asarray(prices, dtype=float)
    if trend_mode == FFT_TREND_LOG_LINEAR:
        log_p = np.log(p)
        log_tr = np.polyval(np.polyfit(t, log_p, 1), t)
        return _positive_trend(np.exp(log_tr), p)
    if trend_mode == FFT_TREND_LINEAR_PRICE:
        return _positive_trend(np.polyval(np.polyfit(t, p, 1), t), p)
    if trend_mode == FFT_TREND_HP:
        trend_p, _ = _hodrick_prescott(p)
        return _positive_trend(trend_p, p)
    raise ValueError(f"Mode de tendance inconnu: {trend_mode}")


def _extend_log_trend(log_trend_hist, n_total, trend_mode):
    """Prolonge la tendance log (log-linéaire ou extrapolation linéaire sur log)."""
    n = len(log_trend_hist)
    t_hist = np.arange(n)
    t_ext = np.arange(n_total)
    log_trend_hist = np.asarray(log_trend_hist, dtype=float)
    if trend_mode == FFT_TREND_LOG_LINEAR or trend_mode in (
        FFT_TREND_MA41_CENTER,
        FFT_TREND_MA41_CAUSAL,
        FFT_TREND_STL,
    ):
        return np.polyval(np.polyfit(t_hist, log_trend_hist, 1), t_ext)
    if trend_mode == FFT_TREND_LINEAR_PRICE:
        return np.log(
            np.maximum(
                np.polyval(np.polyfit(t_hist, np.exp(log_trend_hist), 1), t_ext),
                1e-12,
            )
        )
    if trend_mode == FFT_TREND_HP:
        return np.polyval(np.polyfit(t_hist, log_trend_hist, 1), t_ext)
    return np.polyval(np.polyfit(t_hist, log_trend_hist, 1), t_ext)


def _extend_price_trend(trend_p, n_total, trend_mode):
    n = len(trend_p)
    t_hist = np.arange(n)
    t_ext = np.arange(n_total)
    trend_p = np.asarray(trend_p, dtype=float)
    if trend_mode in (FFT_TREND_MA41_CENTER, FFT_TREND_MA41_CAUSAL):
        last = float(trend_p[-1])
        ext = np.full(n_total, last)
        ext[:n] = trend_p
        return _positive_trend(ext, trend_p)
    if trend_mode == FFT_TREND_STL:
        log_tr = np.log(np.maximum(trend_p, 1e-12))
        return _positive_trend(np.exp(_extend_log_trend(log_tr, n_total, trend_mode)), trend_p)
    if trend_mode == FFT_TREND_LOG_LINEAR:
        log_tr = np.log(trend_p)
        log_ext = np.polyval(np.polyfit(t_hist, log_tr, 1), t_ext)
        return np.exp(log_ext)
    return np.polyval(np.polyfit(t_hist, trend_p, 1), t_ext)


FFT_PRICE_UNIT_EUR = "eur"
FFT_PRICE_UNIT_PORTFOLIO = "portfolio_eur"
FFT_PRICE_UNIT_INDEX = "index"


def _trend_fit_metrics(prices, trend_p):
    p = np.asarray(prices, dtype=float)
    trend_p = np.asarray(trend_p, dtype=float)
    ss_res = float(np.sum((p - trend_p) ** 2))
    ss_tot = float(np.sum((p - p.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.mean((p - trend_p) ** 2)))
    mean_p = float(np.mean(p))
    rmse_pct = (rmse / mean_p * 100.0) if mean_p > 0 else np.nan
    return r2, rmse, mean_p, rmse_pct


def interpret_fft_trend_r2(r2):
    """
    Fiabilité de l'isolation de tendance (avant FFT) selon R² sur le prix réel.

    R² mesure la part de variance du cours expliquée par la tendance seule.
    """
    if r2 is None or (isinstance(r2, float) and np.isnan(r2)):
        return {
            "label": "—",
            "cycles_reliability": "—",
            "detail": "Non calculable sur cette série.",
        }
    if r2 >= 0.85:
        return {
            "label": "Excellente",
            "cycles_reliability": "Élevée",
            "detail": (
                "La tendance décrit très bien le cours ; l'isolation des cycles est fiable."
            ),
        }
    if r2 >= 0.65:
        return {
            "label": "Bonne",
            "cycles_reliability": "Correcte",
            "detail": (
                "La tendance capture une large part du mouvement ; les cycles sont interprétables."
            ),
        }
    if r2 >= 0.50:
        return {
            "label": "Modérée",
            "cycles_reliability": "Moyenne",
            "detail": (
                "La tendance n'explique qu'une partie du cours ; interpréter les cycles avec prudence."
            ),
        }
    if r2 >= 0.30:
        return {
            "label": "Faible",
            "cycles_reliability": "Limitée",
            "detail": (
                "La tendance isole mal le fond ; les cycles FFT sont peu fiables."
            ),
        }
    return {
        "label": "Très faible",
        "cycles_reliability": "Très limitée",
        "detail": (
            "La tendance n'explique qu'une faible part du cours ; cycles et extrapolation peu fiables."
        ),
    }


def format_fft_rmse_display(rmse, rmse_pct, price_unit=FFT_PRICE_UNIT_EUR):
    """RMSE absolu (unité adaptée) et relatif (% du prix moyen)."""
    if rmse is None or (isinstance(rmse, float) and np.isnan(rmse)):
        return "—", "—"
    if price_unit == FFT_PRICE_UNIT_INDEX:
        abs_txt = f"{rmse:.2f} pts"
    elif price_unit == FFT_PRICE_UNIT_PORTFOLIO:
        abs_txt = f"{rmse:,.2f} € (valorisation portefeuille)"
    else:
        abs_txt = f"{rmse:,.2f} € / action"
    if rmse_pct is None or (isinstance(rmse_pct, float) and np.isnan(rmse_pct)):
        pct_txt = "—"
    else:
        pct_txt = f"{rmse_pct:.1f} %"
    return abs_txt, pct_txt


def summarize_fft_trend_quality(result, price_unit=FFT_PRICE_UNIT_EUR):
    """Texte d'interprétation R² / RMSE pour le dashboard FFT."""
    if not result:
        return {"caption_md": "", "reliability": interpret_fft_trend_r2(None)}

    r2 = result.get("trend_r2")
    rmse = result.get("trend_rmse")
    rmse_pct = result.get("trend_rmse_pct")
    reliability = interpret_fft_trend_r2(r2)
    abs_txt, pct_txt = format_fft_rmse_display(rmse, rmse_pct, price_unit)
    trend_label = fft_trend_mode_label(result.get("trend_mode", FFT_TREND_LOG_LINEAR))

    if r2 is None or (isinstance(r2, float) and np.isnan(r2)):
        r2_txt = "—"
    else:
        r2_txt = f"{r2:.3f}"

    caption = (
        f"Tendance : **{trend_label}** · "
        f"R² = **{r2_txt}** — fiabilité **{reliability['label'].lower()}** "
        f"(cycles : **{reliability['cycles_reliability'].lower()}**) · "
        f"RMSE = **{abs_txt}** "
    )
    if pct_txt != "—":
        caption += f"(**{pct_txt}** du prix moyen) · "
    caption += reliability["detail"]

    return {"caption_md": caption, "reliability": reliability}


def format_fft_period_label(days):
    """Libellé lisible pour une période en jours de bourse."""
    if days is None or np.isnan(days):
        return "—"
    days = float(days)
    if days < 15:
        return f"{days:.0f} jours"
    if days < 70:
        weeks = days / 5
        return f"{days:.0f} j (~{weeks:.1f} sem.)"
    if days < 400:
        months = days / 21
        return f"{days:.0f} j (~{months:.1f} mois)"
    years = days / TRADING_DAYS_PER_YEAR
    return f"{days:.0f} j (~{years:.1f} an(s))"


def _prepare_fft_detrended_series(price_series, trend_mode=FFT_TREND_LOG_LINEAR):
    """
    Prépare le signal analysé par FFT selon le mode de tendance choisi.
    Le signal cyclique = log(component) − log(tendance lente) ou composante STL.
    """
    if trend_mode not in FFT_TREND_OPTIONS:
        trend_mode = FFT_TREND_LOG_LINEAR

    s = pd.Series(price_series).dropna().astype(float)
    min_len = max(40, FFT_MA_WINDOW + 5)
    if trend_mode == FFT_TREND_STL:
        min_len = max(min_len, STL_SEASONAL_PERIOD * 2 + 5)
    if len(s) < min_len:
        return None

    smooth_prices = None

    if trend_mode in (FFT_TREND_MA41_CENTER, FFT_TREND_MA41_CAUSAL):
        centered = trend_mode == FFT_TREND_MA41_CENTER
        ma = _ma41_series(s.values, centered=centered)
        valid = np.isfinite(ma.values)
        if valid.sum() < min_len:
            return None
        idx = np.where(valid)[0]
        i0, i1 = int(idx[0]), int(idx[-1]) + 1
        s = s.iloc[i0:i1]
        p = s.values.astype(float)
        ma_trim = ma.iloc[i0:i1].values.astype(float)
        t = np.arange(len(p))
        log_ma = np.log(ma_trim)
        log_slow = np.polyval(np.polyfit(t, log_ma, 1), t)
        detrended = log_ma - log_slow
        trend_p = ma_trim
        log_trend = log_slow
        smooth_prices = pd.Series(ma_trim, index=s.index)
        trend_r2, trend_rmse, trend_mean_price, trend_rmse_pct = _trend_fit_metrics(p, trend_p)
    elif trend_mode == FFT_TREND_STL:
        p = s.values.astype(float)
        log_p = np.log(p)
        stl_parts = _stl_log_decompose(log_p)
        if stl_parts is None:
            return None
        trend_log, seasonal_log = stl_parts
        detrended = seasonal_log - np.mean(seasonal_log)
        log_trend = trend_log
        trend_p = np.exp(trend_log)
        smooth_prices = pd.Series(np.exp(trend_log + seasonal_log), index=s.index)
        trend_r2, trend_rmse, trend_mean_price, trend_rmse_pct = _trend_fit_metrics(p, trend_p)
    else:
        p = s.values
        t = np.arange(len(p))
        trend_p = _fit_price_trend(p, t, trend_mode)
        log_p = np.log(p)
        log_trend = np.log(trend_p)
        detrended = log_p - log_trend
        trend_r2, trend_rmse, trend_mean_price, trend_rmse_pct = _trend_fit_metrics(p, trend_p)

    prep = {
        "series": s,
        "dates": s.index,
        "prices": s,
        "trend_mode": trend_mode,
        "trend_prices": pd.Series(trend_p, index=s.index),
        "log_trend": pd.Series(log_trend, index=s.index),
        "detrended_log": pd.Series(detrended, index=s.index),
        "detrended": detrended,
        "trend_r2": trend_r2,
        "trend_rmse": trend_rmse,
        "trend_mean_price": trend_mean_price,
        "trend_rmse_pct": trend_rmse_pct,
        "n_obs": len(s),
    }
    if smooth_prices is not None:
        prep["smooth_prices"] = smooth_prices
    return prep


def _prepare_log_detrended_series(price_series):
    """Alias rétrocompatible — tendance log-linéaire."""
    return _prepare_fft_detrended_series(price_series, trend_mode=FFT_TREND_LOG_LINEAR)


def compute_fft_periodicity(
    price_series,
    min_period_days=5,
    max_period_days=None,
    top_n=8,
    trend_mode=FFT_TREND_LOG_LINEAR,
):
    """
    FFT sur série de prix (log + dé-trend) pour détecter des cycles périodiques.

    Retourne None si historique trop court.
    """
    prep = _prepare_fft_detrended_series(price_series, trend_mode=trend_mode)
    if prep is None:
        return None

    s = prep["series"]
    detrended = prep["detrended"]
    trend = prep["log_trend"].values
    n = prep["n_obs"]

    if max_period_days is None:
        max_period_days = min(n // 2, TRADING_DAYS_PER_YEAR)

    window = np.hanning(len(detrended))
    windowed = detrended * window

    fft_vals = np.fft.rfft(windowed)
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0)

    valid = freqs > 0
    if min_period_days > 0:
        valid &= freqs <= 1.0 / min_period_days
    if max_period_days and max_period_days > 0:
        valid &= freqs >= 1.0 / max_period_days

    idx = np.where(valid)[0]
    if len(idx) == 0:
        return None

    periods = 1.0 / freqs[idx]
    powers = power[idx]
    order = np.argsort(powers)[::-1]
    idx_sorted = idx[order]
    periods_sorted = periods[order]
    powers_sorted = powers[order]

    total_power = float(powers_sorted.sum()) or 1.0
    peak_rows = []
    used_periods = []
    for rank, (fi, period, pwr) in enumerate(
        zip(idx_sorted[: top_n * 2], periods_sorted[: top_n * 2], powers_sorted[: top_n * 2]),
        start=1,
    ):
        if len(peak_rows) >= top_n:
            break
        if any(abs(period - up) / up < 0.08 for up in used_periods if up > 0):
            continue
        used_periods.append(period)
        peak_rows.append(
            {
                "Rang": len(peak_rows) + 1,
                "Période (jours)": float(period),
                "Période": format_fft_period_label(period),
                "Puissance relative": float(pwr / total_power),
                "_freq_idx": int(fi),
            }
        )

    peaks_df = pd.DataFrame(peak_rows)
    if peaks_df.empty:
        return None

    out = {
        "dates": s.index,
        "prices": s,
        "trend_mode": prep["trend_mode"],
        "trend_prices": prep["trend_prices"],
        "trend_r2": prep["trend_r2"],
        "trend_rmse": prep["trend_rmse"],
        "trend_mean_price": prep["trend_mean_price"],
        "trend_rmse_pct": prep["trend_rmse_pct"],
        "log_trend": pd.Series(trend, index=s.index),
        "detrended_log": pd.Series(detrended, index=s.index),
        "periods_days": periods,
        "power": powers,
        "freqs": freqs[idx],
        "peaks": peaks_df,
        "n_obs": n,
        "min_period_days": min_period_days,
        "max_period_days": max_period_days,
    }
    if prep.get("smooth_prices") is not None:
        out["smooth_prices"] = prep["smooth_prices"]
    return out


def reconstruct_fft_cycles(detrended_values, peak_freq_indices, n_obs):
    """Reconstruit la composante cyclique à partir des pics FFT sélectionnés."""
    windowed = np.asarray(detrended_values, dtype=float)
    fft_vals = np.fft.rfft(windowed)
    filtered = np.zeros_like(fft_vals)
    for fi in peak_freq_indices:
        if 0 < fi < len(fft_vals):
            filtered[fi] = fft_vals[fi]
    cyclic = np.fft.irfft(filtered, n=n_obs)
    return cyclic


def _harmonic_design_matrix(t, periods_days):
    """Matrice sin/cos pour régression harmonique."""
    t = np.asarray(t, dtype=float)
    cols = [np.ones(len(t))]
    for period in periods_days:
        p = float(period)
        if p <= 0:
            continue
        cols.append(np.cos(2.0 * np.pi * t / p))
        cols.append(np.sin(2.0 * np.pi * t / p))
    return np.column_stack(cols)


def reconstruct_harmonic_cycles(
    detrended_values,
    periods_days,
    n_obs=None,
    *,
    weights=None,
):
    """Régression sin/cos aux périodes données (OLS ou pondéré récence)."""
    y = np.asarray(detrended_values, dtype=float)
    n = len(y) if n_obs is None else int(n_obs)
    y = y[:n]
    periods = [float(p) for p in periods_days if float(p) > 0]
    if not periods or len(y) < 3:
        return np.zeros(n)
    x = _harmonic_design_matrix(np.arange(n), periods)
    if weights is not None:
        w = np.sqrt(np.asarray(weights, dtype=float)[:n])
        coef, _, _, _ = np.linalg.lstsq(x * w[:, None], y * w, rcond=None)
    else:
        coef, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    return x @ coef


def reconstruct_cycles_from_fft_peaks(
    detrended_values,
    peak_freq_indices,
    n_obs,
    *,
    weights=None,
):
    """
    Reconstruction aux périodes des pics FFT par régression (sans fenêtre Hanning).
    Corrige l'atténuation d'amplitude en début/fin de série.
    """
    periods = _periods_from_peak_indices(n_obs, peak_freq_indices)
    if not periods:
        return np.zeros(int(n_obs))
    return reconstruct_harmonic_cycles(
        detrended_values, periods, n_obs, weights=weights
    )


def _extend_harmonic_cycles(
    detrended_hist,
    periods_days,
    n_hist,
    n_total,
    *,
    weights=None,
):
    """Prolonge les harmoniques ajustées sur l'historique (signal non fenêtré)."""
    y = np.asarray(detrended_hist, dtype=float)[:n_hist]
    x_hist = _harmonic_design_matrix(np.arange(n_hist), periods_days)
    x_full = _harmonic_design_matrix(np.arange(n_total), periods_days)
    if weights is not None:
        w = np.sqrt(np.asarray(weights, dtype=float)[:n_hist])
        coef, _, _, _ = np.linalg.lstsq(x_hist * w[:, None], y * w, rcond=None)
    else:
        coef, _, _, _ = np.linalg.lstsq(x_hist, y, rcond=None)
    return x_full @ coef


def harmonic_periods_from_result(result, recon_mode=FFT_RECON_HARMONIC, n_components=5):
    """Périodes utilisées pour la régression harmonique."""
    if recon_mode == FFT_RECON_HARMONIC_FIXED:
        return list(HARMONIC_FIXED_PERIODS)
    peaks = result.get("peaks")
    if peaks is None or peaks.empty:
        return list(HARMONIC_FIXED_PERIODS[:3])
    periods = peaks.head(n_components)["Période (jours)"].astype(float).tolist()
    return periods or list(HARMONIC_FIXED_PERIODS[:3])


def _reconstruct_cyclic_component(
    result,
    peak_freq_indices,
    *,
    recon_mode=FFT_RECON_FFT,
    n_components=5,
    n_obs=None,
    recency_weighted=True,
):
    """Composante cyclique — détection FFT (Hanning) / reconstruction sans fenêtre."""
    n = int(n_obs or result["n_obs"])
    detrended = np.asarray(result["detrended_log"].values, dtype=float)
    weights = _recency_weights(n) if recency_weighted else None
    if recon_mode == FFT_RECON_FFT:
        return reconstruct_cycles_from_fft_peaks(
            detrended, peak_freq_indices, n, weights=weights
        )
    periods = harmonic_periods_from_result(result, recon_mode, n_components)
    return reconstruct_harmonic_cycles(detrended, periods, n, weights=weights)


def _safe_pearson(a, b):
    """Corrélation de Pearson ; NaN si série trop courte ou variance nulle."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if len(a) < 3:
        return np.nan
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def compute_fft_model_correlations(
    result,
    peak_freq_indices,
    n_components=5,
    *,
    recency_weighted=True,
    recency_calibrate=True,
):
    """
    Corrélation Pearson entre le cours et les modèles sur l'historique.

    - corr_with_trend_filter : modèle FFT/harmonique (tendance + cycles)
    - corr_without_trend_filter : FFT sur log(prix) sans retirer la tendance
    - corr_vs_smooth : vs MM41 ou trend+saison STL si disponible
    - corr_harmonic : régression harmonique (périodes FFT) vs cours
    """
    if result is None or not peak_freq_indices:
        return None

    n = int(result["n_obs"])
    prices = np.asarray(result["prices"].values, dtype=float)
    log_trend, detrended, weights = _fft_model_series(
        result,
        recency_calibrate=recency_calibrate,
        recency_weighted=recency_weighted,
    )
    hann = np.hanning(n)

    cyclic_fft = reconstruct_cycles_from_fft_peaks(
        detrended, peak_freq_indices, n, weights=weights
    )
    model_fft = np.exp(log_trend + cyclic_fft)
    corr_with = _safe_pearson(prices, model_fft)

    log_p = np.log(prices)
    windowed_log = log_p * hann
    fft_raw = np.fft.rfft(windowed_log)
    filtered = np.zeros_like(fft_raw)
    filtered[0] = fft_raw[0]
    for fi in peak_freq_indices:
        fi = int(fi)
        if 0 < fi < len(filtered):
            filtered[fi] = fft_raw[fi]
    recon_log = np.fft.irfft(filtered, n=n)
    model_without_trend_filter = np.exp(recon_log)
    corr_without = _safe_pearson(prices, model_without_trend_filter)

    smooth = result.get("smooth_prices")
    corr_vs_smooth = None
    if smooth is not None:
        smooth_vals = np.asarray(smooth.values, dtype=float)
        corr_vs_smooth = _safe_pearson(smooth_vals, model_fft)

    periods = harmonic_periods_from_result(result, FFT_RECON_HARMONIC, n_components)
    cyclic_harm = reconstruct_harmonic_cycles(detrended, periods, n, weights=weights)
    model_harmonic = np.exp(log_trend + cyclic_harm)
    corr_harmonic = _safe_pearson(prices, model_harmonic)

    corr_harm_vs_smooth = None
    if smooth is not None:
        corr_harm_vs_smooth = _safe_pearson(smooth_vals, model_harmonic)

    return {
        "corr_with_trend_filter": corr_with,
        "corr_without_trend_filter": corr_without,
        "corr_vs_smooth": corr_vs_smooth,
        "corr_harmonic": corr_harmonic,
        "corr_harmonic_vs_smooth": corr_harm_vs_smooth,
    }


FFT_FORECAST_RATIO = 0.30
FFT_RECENCY_HALFLIFE = 63
FFT_RECENT_TREND_TAIL = 42
FFT_RECENT_TREND_BLEND = 0.22


def _recency_weights(n, halflife=FFT_RECENCY_HALFLIFE):
    """Poids exponentiels — les séances récentes comptent davantage."""
    n = int(n)
    if n < 2:
        return np.ones(max(n, 1), dtype=float)
    ages = np.arange(n - 1, -1, -1, dtype=float)
    w = np.exp(-ages / float(halflife))
    return w / (w.mean() + 1e-12)


def _periods_from_peak_indices(n_obs, peak_freq_indices):
    """Convertit indices FFT en périodes (jours de bourse)."""
    periods = []
    n = int(n_obs)
    for fi in peak_freq_indices:
        fi = int(fi)
        if fi <= 0:
            continue
        periods.append(float(n) / fi)
    return periods


def _blend_recent_log_trend(
    log_trend,
    log_prices,
    *,
    tail_days=FFT_RECENT_TREND_TAIL,
    blend_pct=FFT_RECENT_TREND_BLEND,
):
    """Remplace progressivement la tendance globale par une droite locale en fin de série."""
    log_trend = np.asarray(log_trend, dtype=float).copy()
    log_prices = np.asarray(log_prices, dtype=float)
    n = len(log_trend)
    if n < 10:
        return log_trend
    tail = min(int(tail_days), max(n // 4, 5))
    blend_start = max(0, int(round(n * (1.0 - blend_pct))))
    if blend_start >= n - 2:
        return log_trend
    t = np.arange(n, dtype=float)
    local = np.polyval(np.polyfit(t[-tail:], log_prices[-tail:], 1), t)
    for i in range(blend_start, n):
        alpha = (i - blend_start) / max(n - blend_start - 1, 1)
        log_trend[i] = (1.0 - alpha) * log_trend[i] + alpha * local[i]
    return log_trend


def _calibrated_log_trend_for_model(result, *, recency_calibrate=True):
    """Tendance log utilisée pour le modèle (ajustée en fin de série si demandé)."""
    log_trend = np.asarray(result["log_trend"].values, dtype=float)
    if not recency_calibrate:
        return log_trend
    log_prices = np.log(np.asarray(result["prices"].values, dtype=float))
    return _blend_recent_log_trend(log_trend, log_prices)


def _fft_model_series(result, *, recency_calibrate=True, recency_weighted=True):
    """Tendance, résidu et poids cohérents pour la reconstruction du modèle."""
    n = int(result["n_obs"])
    log_prices = np.log(np.asarray(result["prices"].values, dtype=float))
    log_trend = _calibrated_log_trend_for_model(
        result, recency_calibrate=recency_calibrate
    )
    detrended = log_prices - log_trend
    weights = _recency_weights(n) if recency_weighted else None
    return log_trend, detrended, weights


def _synthesize_rfft_peaks(fft_vals, peak_indices, n_hist, n_total):
    """Prolonge la somme des harmoniques FFT au-delà de l'historique."""
    t = np.arange(n_total, dtype=float)
    out = np.zeros(n_total, dtype=float)
    for fi in peak_indices:
        fi = int(fi)
        if fi <= 0 or fi >= len(fft_vals):
            continue
        coef = fft_vals[fi]
        if n_hist % 2 == 0 and fi == len(fft_vals) - 1:
            out += (1.0 / n_hist) * np.real(coef) * np.cos(np.pi * t)
        else:
            out += (2.0 / n_hist) * np.real(coef * np.exp(2j * np.pi * fi * t / n_hist))
    return out


def build_fft_extended_series(
    result,
    peak_indices,
    extend_ratio=FFT_FORECAST_RATIO,
    recon_mode=FFT_RECON_FFT,
    n_components=5,
    *,
    recency_weighted=True,
    recency_calibrate=True,
):
    """Extrapole tendance log + cycles (FFT ou harmonique) sur extend_ratio × l'historique."""
    if not peak_indices or extend_ratio <= 0:
        return None
    if recon_mode not in (FFT_RECON_FFT, FFT_RECON_HARMONIC, FFT_RECON_HARMONIC_FIXED):
        recon_mode = FFT_RECON_FFT

    n = int(result["n_obs"])
    n_extra = max(1, int(round(n * extend_ratio)))
    n_total = n + n_extra

    log_trend_hist, detrended_model, weights = _fft_model_series(
        result,
        recency_calibrate=recency_calibrate,
        recency_weighted=recency_weighted,
    )

    trend_mode = result.get("trend_mode", FFT_TREND_LOG_LINEAR)
    trend_p_hist = np.exp(log_trend_hist)
    trend_p_ext = _extend_price_trend(trend_p_hist, n_total, trend_mode)
    trend_p_ext = _positive_trend(trend_p_ext, result["prices"].values)
    log_trend_ext = _extend_log_trend(log_trend_hist, n_total, trend_mode)

    if recon_mode == FFT_RECON_FFT:
        periods = _periods_from_peak_indices(n, peak_indices)
        cyclic_ext = _extend_harmonic_cycles(
            detrended_model, periods, n, n_total, weights=weights
        )
    else:
        periods = harmonic_periods_from_result(result, recon_mode, n_components)
        cyclic_ext = _extend_harmonic_cycles(
            detrended_model, periods, n, n_total, weights=weights
        )

    log_model_ext = log_trend_ext + cyclic_ext
    price_model_ext = np.exp(log_model_ext)

    prices = result["prices"]
    mean_p = float(prices.mean())
    std_p = float(prices.std()) + 1e-8
    trend_norm_ext = (trend_p_ext - mean_p) / std_p
    cyclic_norm_ext = (cyclic_ext - cyclic_ext[:n].mean()) / (float(cyclic_ext[:n].std()) + 1e-8)

    hist_dates = pd.DatetimeIndex(result["dates"])
    future_dates = pd.bdate_range(start=hist_dates[-1] + pd.offsets.BDay(1), periods=n_extra)
    all_dates = hist_dates.union(future_dates)

    return {
        "dates": all_dates,
        "n_hist": n,
        "n_extra": n_extra,
        "extend_ratio": extend_ratio,
        "trend_norm": trend_norm_ext,
        "trend_prices_ext": trend_p_ext,
        "cyclic_norm": cyclic_norm_ext,
        "price_model": price_model_ext,
        "prices_hist": np.asarray(prices.values, dtype=float),
        "hist_end": hist_dates[-1],
    }


def _add_fft_split_trace(
    fig,
    dates,
    values,
    n_hist,
    name,
    line_color,
    line_width,
    dash_forecast="dash",
    secondary_y=False,
    forecast_suffix=None,
):
    """Trace historique (plein) + extrapolation (pointillé) avec point de jonction."""
    forecast_suffix = forecast_suffix or f" (+{int(round(FFT_FORECAST_RATIO * 100))}%)"
    trace_kwargs = {"secondary_y": secondary_y} if secondary_y else {}
    fig.add_trace(
        go.Scatter(
            x=dates[:n_hist],
            y=values[:n_hist],
            mode="lines",
            name=name,
            line=dict(color=line_color, width=line_width),
        ),
        **trace_kwargs,
    )
    if n_hist < len(dates):
        fig.add_trace(
            go.Scatter(
                x=dates[n_hist - 1 :],
                y=values[n_hist - 1 :],
                mode="lines",
                name=f"{name}{forecast_suffix}",
                line=dict(color=line_color, width=line_width, dash=dash_forecast),
            ),
            **trace_kwargs,
        )


def compute_fft_summary_for_prices(
    prices, min_period_days=5, max_period_days=None, top_n=3, trend_mode=FFT_TREND_LOG_LINEAR
):
    """Tableau récapitulatif : périodes dominantes par ticker."""
    rows = []
    for ticker in prices.columns:
        result = compute_fft_periodicity(
            prices[ticker],
            min_period_days=min_period_days,
            max_period_days=max_period_days,
            top_n=top_n,
            trend_mode=trend_mode,
        )
        if result is None:
            continue
        peaks = result["peaks"]
        row = {"Ticker": ticker, "Observations": result["n_obs"]}
        for i in range(top_n):
            col_p = f"Période #{i + 1}"
            col_w = f"Poids #{i + 1}"
            if i < len(peaks):
                row[col_p] = peaks.iloc[i]["Période"]
                row[col_w] = peaks.iloc[i]["Puissance relative"]
            else:
                row[col_p] = None
                row[col_w] = None
        if len(peaks):
            row["Période dominante"] = peaks.iloc[0]["Période"]
            row["Force cycle principal"] = peaks.iloc[0]["Puissance relative"]
        rows.append(row)
    return pd.DataFrame(rows)


def build_fft_spectrum_chart(result, title="Spectre de puissance (FFT)"):
    """Graphique période (jours) vs puissance relative."""
    periods = result["periods_days"]
    power = result["power"]
    rel = power / (power.sum() or 1.0)
    peaks = result["peaks"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=periods,
            y=rel,
            mode="lines",
            name="Puissance",
            line=dict(color="steelblue", width=1.5),
        )
    )
    if not peaks.empty:
        peak_periods = peaks["Période (jours)"].values
        peak_power = peaks["Puissance relative"].values
        fig.add_trace(
            go.Scatter(
                x=peak_periods,
                y=peak_power,
                mode="markers+text",
                name="Pics dominants",
                marker=dict(color="crimson", size=10),
                text=[f"#{int(r)}" for r in peaks["Rang"]],
                textposition="top center",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Période (jours de bourse)",
        yaxis_title="Puissance relative",
        height=420,
        hovermode="x unified",
    )
    fig.update_xaxes(type="log")
    return fig


def _fft_windowed_signal(result):
    """Signal log dé-trendé × fenêtre Hanning (identique à l'entrée FFT)."""
    detrended = np.asarray(result["detrended_log"].values, dtype=float)
    return detrended * np.hanning(len(detrended))


def compute_normalized_acf(signal, max_lag=None):
    """Autocorrélation normalisée (lag 0 = 1) du signal centré."""
    x = np.asarray(signal, dtype=float)
    x = x - np.mean(x)
    n = len(x)
    if n < 2:
        return np.array([0.0]), np.array([1.0])
    if max_lag is None:
        max_lag = min(n // 2, 252)
    max_lag = int(min(max(max_lag, 1), n - 1))
    corr_full = np.correlate(x, x, mode="full")
    mid = n - 1
    acf = corr_full[mid : mid + max_lag + 1]
    acf = acf / (acf[0] + 1e-12)
    return np.arange(max_lag + 1, dtype=float), acf


def build_fft_spectral_acf_chart(
    result,
    title="Densité spectrale & autocorrélation",
    max_acf_lag=None,
):
    """
    Graphique dual : densité spectrale (période) et fonction d'autocorrélation (décalage).
    Les deux vues proviennent du même signal dé-trendé (théorème de Wiener-Khinchin).
    """
    periods = result["periods_days"]
    power = result["power"]
    density = power / (float(power.sum()) or 1.0)
    peaks = result["peaks"]
    n = int(result["n_obs"])

    windowed = _fft_windowed_signal(result)
    if max_acf_lag is None:
        max_acf_lag = int(min(n // 2, result.get("max_period_days") or 252, 504))
    lags, acf = compute_normalized_acf(windowed, max_lag=max_acf_lag)
    conf = 1.96 / np.sqrt(n)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.11,
        subplot_titles=(
            "Densité spectrale — |FFT|² normalisée vs période",
            "Autocorrélation — signal log dé-trendé (fenêtre Hanning)",
        ),
        row_heights=[0.52, 0.48],
    )

    fig.add_trace(
        go.Scatter(
            x=periods,
            y=density,
            mode="lines",
            name="Densité spectrale",
            line=dict(color="#2563eb", width=1.5),
            hovertemplate="Période: %{x:.1f} j<br>Densité: %{y:.2%}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    if not peaks.empty:
        fig.add_trace(
            go.Scatter(
                x=peaks["Période (jours)"],
                y=peaks["Puissance relative"],
                mode="markers+text",
                name="Pics dominants",
                marker=dict(color="#dc2626", size=9),
                text=[f"#{int(r)}" for r in peaks["Rang"]],
                textposition="top center",
                hovertemplate="Période: %{x:.1f} j<br>Part: %{y:.1%}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=lags,
            y=acf,
            mode="lines",
            name="Autocorrélation",
            line=dict(color="#7c3aed", width=1.5),
            hovertemplate="Lag: %{x:.0f} j<br>ACF: %{y:.3f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_hline(
        y=conf,
        line_dash="dot",
        line_color="#94a3b8",
        annotation_text="95 % (bruit blanc)",
        annotation_position="right",
        row=2,
        col=1,
    )
    fig.add_hline(y=-conf, line_dash="dot", line_color="#94a3b8", row=2, col=1)

    for _, peak in peaks.head(3).iterrows():
        period = float(peak["Période (jours)"])
        if 1 <= period <= max_acf_lag:
            fig.add_vline(
                x=period,
                line_dash="dash",
                line_color="#f97316",
                opacity=0.55,
                row=2,
                col=1,
            )

    fig.update_layout(
        title=title,
        height=580,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(title_text="Période (jours)", type="log", row=1, col=1)
    fig.update_yaxes(title_text="Densité relative", tickformat=".1%", row=1, col=1)
    fig.update_xaxes(title_text="Décalage / lag (jours)", row=2, col=1)
    fig.update_yaxes(title_text="Autocorrélation", range=[-1.05, 1.05], row=2, col=1)
    return fig


def interpret_fft_acf_reading(result, max_acf_lag=None):
    """
    Commentaire en langage clair pour la lecture du graphique d'autocorrélation (ACF).
    """
    if result is None:
        return None

    n = int(result["n_obs"])
    peaks = result.get("peaks")
    if peaks is None:
        peaks = pd.DataFrame()

    windowed = _fft_windowed_signal(result)
    if max_acf_lag is None:
        max_acf_lag = int(min(n // 2, result.get("max_period_days") or 252, 504))
    lags, acf = compute_normalized_acf(windowed, max_lag=max_acf_lag)
    conf = 1.96 / np.sqrt(n)

    parts = [
        "**Comment lire le graphique du bas :** l'axe horizontal = nombre de jours de décalage ; "
        "la courbe violette mesure si le cours dé-trendé **ressemble** à ce qu'il était il y a "
        f"*n* jours. Au point **0 j**, la valeur est toujours **1** (série comparée à elle-même). "
        f"Les pointillés gris (±{conf:.2f}) délimitent la zone de **bruit** : en dehors, un pic "
        "peut signaler une répétition réelle."
    ]

    confirmed = []
    weak = []
    if not peaks.empty:
        for _, peak in peaks.head(3).iterrows():
            period = float(peak["Période (jours)"])
            period_label = str(peak.get("Période") or format_fft_period_label(period))
            strength = float(peak["Puissance relative"])
            lag_idx = int(min(max(round(period), 1), len(acf) - 1))
            acf_val = float(acf[lag_idx])
            if abs(acf_val) > conf:
                if acf_val > 0:
                    confirmed.append(
                        f"à **{period_label}** (ligne orange) l'ACF vaut **{acf_val:+.2f}** : "
                        f"le motif a tendance à **se répéter** (cycle FFT ≈ {strength:.0%} de force)"
                    )
                else:
                    confirmed.append(
                        f"à **{period_label}** (ligne orange) l'ACF vaut **{acf_val:+.2f}** : "
                        f"alternance hausse/baisse à cet intervalle (cycle FFT ≈ {strength:.0%})"
                    )
            else:
                weak.append(
                    f"à **{period_label}** l'ACF vaut **{acf_val:+.2f}** (dans le bruit) : "
                    f"le pic FFT ({strength:.0%}) **n'est pas confirmé** ici"
                )

    if confirmed:
        parts.append("**Ce qui ressort :** " + " · ".join(confirmed) + ".")
    if weak:
        parts.append("**À nuancer :** " + " · ".join(weak) + ".")

    if len(acf) > 1:
        acf_rest = acf[1:]
        lags_rest = lags[1:]
        best_i = int(np.argmax(np.abs(acf_rest)))
        best_lag = int(lags_rest[best_i])
        best_val = float(acf_rest[best_i])
        if abs(best_val) > conf and not confirmed:
            if best_val > 0:
                parts.append(
                    f"**Répétition la plus nette** : vers **{best_lag} jours** "
                    f"(ACF = **{best_val:+.2f}**) — intervalle où le signal se ressemble le plus, "
                    "hors bruit."
                )
            else:
                parts.append(
                    f"**Alternance la plus nette** : vers **{best_lag} jours** "
                    f"(ACF = **{best_val:+.2f}**) — le signal tend à s'inverser à cet intervalle."
                )
        elif not confirmed and not weak:
            parts.append(
                "**Lecture globale :** la courbe reste surtout **entre les pointillés gris** — "
                "pas de cycle clairement répétitif sur la fenêtre ; les pics du graphique du haut "
                "peuvent venir du hasard ou d'événements ponctuels."
            )
        elif confirmed and best_lag > 0 and abs(best_val) > conf:
            fft_lags = {int(round(float(p))) for p in peaks.head(3)["Période (jours)"]}
            if best_lag not in fft_lags and abs(best_val) > conf + 0.05:
                parts.append(
                    f"**Autre décalage notable** : **{best_lag} jours** (ACF = **{best_val:+.2f}**), "
                    "non marqué en orange — à comparer avec les cycles FFT."
                )

    parts.append(
        "**En pratique :** cherchez un **rebond** de la courbe violette près des lignes orange ; "
        "s'il dépasse la zone grise, le cycle du haut gagne en crédibilité. Sinon, restez prudent."
    )
    return "\n\n".join(parts)


def _fft_fond_trend_normalized(prices, log_trend=None, trend_prices=None):
    """
    Tendance de fond reprojetée dans l'espace normalisé du prix (centré / réduit).
    """
    prices = pd.Series(prices).astype(float)
    if trend_prices is not None:
        trend_prices = np.asarray(trend_prices, dtype=float)
    else:
        log_trend = np.asarray(log_trend, dtype=float)
        trend_prices = np.exp(log_trend)
    mean = float(prices.mean())
    std = float(prices.std()) + 1e-8
    trend_norm = (trend_prices - mean) / std
    return trend_norm


def build_fft_cyclic_chart(
    result,
    n_components=3,
    title="Composante cyclique estimée",
    extend_ratio=FFT_FORECAST_RATIO,
    recon_mode=FFT_RECON_FFT,
    *,
    recency_weighted=True,
    recency_calibrate=True,
):
    """Prix, tendance de fond, lissage (optionnel) et modèle FFT/harmonique extrapolé."""
    peaks = result["peaks"].head(n_components)
    indices = peaks["_freq_idx"].tolist()
    n_hist = result["n_obs"]
    trend_mode = result.get("trend_mode", FFT_TREND_LOG_LINEAR)
    trend_name = f"Tendance de fond ({fft_trend_legend_label(trend_mode)})"
    recon_label = fft_recon_mode_label(recon_mode)

    extended = build_fft_extended_series(
        result,
        indices,
        extend_ratio=extend_ratio,
        recon_mode=recon_mode,
        n_components=n_components,
        recency_weighted=recency_weighted,
        recency_calibrate=recency_calibrate,
    )
    if extended is not None:
        dates = extended["dates"]
        n_hist = extended["n_hist"]
        trend_line = extended["trend_prices_ext"]
        price_model = extended["price_model"]
        prices_hist = extended["prices_hist"]
        hist_end = extended["hist_end"]
    else:
        dates = result["dates"]
        prices_hist = np.asarray(result["prices"].values, dtype=float)
        trend_line = np.asarray(result["trend_prices"].values, dtype=float)
        price_model = None
        hist_end = pd.Timestamp(dates[-1])

    smooth = result.get("smooth_prices")
    smooth_hist = np.asarray(smooth.values, dtype=float) if smooth is not None else None

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dates[:n_hist],
            y=prices_hist,
            mode="lines",
            name="Prix (historique)",
            line=dict(color="#1e40af", width=2.5),
        )
    )

    if smooth_hist is not None:
        fig.add_trace(
            go.Scatter(
                x=dates[:n_hist],
                y=smooth_hist,
                mode="lines",
                name="Série lissée (MM41 / STL)",
                line=dict(color="#9333ea", width=1.8, dash="dot"),
            )
        )

    if price_model is not None:
        _add_fft_split_trace(
            fig,
            dates,
            price_model,
            n_hist,
            f"Modèle ({recon_label})",
            "#0ea5e9",
            2.2,
            dash_forecast="dash",
            secondary_y=False,
        )

    _add_fft_split_trace(
        fig,
        dates,
        trend_line,
        n_hist,
        trend_name,
        "#64748b",
        2.5,
        dash_forecast="dash",
        secondary_y=False,
    )

    if extended is not None:
        pct = int(round(extended["extend_ratio"] * 100))
        fig.add_shape(
            type="line",
            x0=hist_end,
            x1=hist_end,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="#94a3b8", dash="dot", width=1),
        )
        fig.add_annotation(
            x=hist_end,
            y=1.02,
            yref="paper",
            text="Fin historique",
            showarrow=False,
            font=dict(size=10, color="#64748b"),
        )
        title_suffix = f" · extrapolation +{pct} %"
    else:
        title_suffix = ""

    fig.update_layout(
        title=f"{title}{title_suffix}",
        xaxis_title="Date",
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        hovermode="x unified",
        yaxis_title="Prix / modèle / tendance",
    )
    return fig


def compute_advanced_risk_metrics(returns):
    annual_return = returns.mean() * 252
    median_annual = returns.median() * 252
    mode_annual = returns.apply(estimate_mode, axis=0) * 252
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
            "Rendement Annuel (moyenne)": annual_return,
            "Médiane annuelle": median_annual,
            "Mode annuelle": mode_annual,
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


def build_risk_returns_boxplot(returns, column_labels=None):
    """Trait horizontal min–max des rendements journaliers avec Q1, médiane et Q3."""
    column_labels = column_labels or {}
    stats_rows = []
    for col in returns.columns:
        series = returns[col].dropna()
        if series.empty:
            continue
        name = str(column_labels.get(col, col))
        stats_rows.append(
            (
                name,
                float(series.min()),
                float(series.quantile(0.25)),
                float(series.median()),
                float(series.quantile(0.75)),
                float(series.max()),
            )
        )

    fig = go.Figure()
    if not stats_rows:
        return fig

    for name, vmin, q1, med, q3, vmax in stats_rows:
        fig.add_trace(
            go.Scatter(
                x=[vmin, vmax],
                y=[name, name],
                mode="lines",
                line=dict(color="#94a3b8", width=2),
                showlegend=False,
                hovertemplate=(
                    f"{name}<br>Min: {vmin:.2%}<br>Max: {vmax:.2%}"
                    "<extra></extra>"
                ),
            )
        )

    q1_x, q1_y, med_x, med_y, q3_x, q3_y = [], [], [], [], [], []
    for name, _vmin, q1, med, q3, _vmax in stats_rows:
        q1_x.append(q1)
        q1_y.append(name)
        med_x.append(med)
        med_y.append(name)
        q3_x.append(q3)
        q3_y.append(name)

    fig.add_trace(
        go.Scatter(
            x=q1_x,
            y=q1_y,
            mode="markers",
            name="Q1",
            marker=dict(symbol="line-ns-open", size=14, color="#64748b", line=dict(width=2)),
            hovertemplate="Q1: %{x:.2%}<extra>%{y}</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=med_x,
            y=med_y,
            mode="markers",
            name="Médiane",
            marker=dict(symbol="diamond", size=11, color="#1e40af", line=dict(width=1, color="white")),
            hovertemplate="Médiane: %{x:.2%}<extra>%{y}</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=q3_x,
            y=q3_y,
            mode="markers",
            name="Q3",
            marker=dict(symbol="line-ns-open", size=14, color="#64748b", line=dict(width=2)),
            hovertemplate="Q3: %{x:.2%}<extra>%{y}</extra>",
        )
    )

    n_assets = len(stats_rows)
    fig.update_layout(
        title="Rendements journaliers — min, quartiles et médiane par actif",
        xaxis_title="Rendement journalier",
        yaxis_title="Actif",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        height=max(420, 36 * n_assets + 140),
    )
    fig.update_xaxes(tickformat=".2%")
    fig.update_yaxes(autorange="reversed")
    return fig


def build_risk_metrics_boxplot(metrics_df, columns):
    """Répartition d'une métrique entre actifs : min–max, quartiles et points identifiés au survol."""
    columns = [c for c in columns if c in metrics_df.columns]
    if not columns:
        return go.Figure()

    percent_cols = {
        "Rendement Annuel (moyenne)",
        "Médiane annuelle",
        "Mode annuelle",
        "Volatilité (Sigma)",
        "VaR 95% (Jour)",
        "CVaR 95% (Jour)",
    }
    ratio_cols = {"Ratio de Sharpe", "Ratio de Sortino"}

    def _fmt(col, val):
        if col in percent_cols:
            return f"{val:.2%}"
        if col in ratio_cols:
            return f"{val:.2f}"
        return f"{val:.3f}"

    fig = go.Figure()
    q1_x, q1_y, med_x, med_y, q3_x, q3_y = [], [], [], [], [], []
    pt_x, pt_y, pt_hover = [], [], []

    for col in columns:
        series = metrics_df[col].dropna()
        if series.empty:
            continue

        vmin = float(series.min())
        vmax = float(series.max())
        q1 = float(series.quantile(0.25))
        med = float(series.median())
        q3 = float(series.quantile(0.75))

        fig.add_trace(
            go.Scatter(
                x=[vmin, vmax],
                y=[col, col],
                mode="lines",
                line=dict(color="#94a3b8", width=2),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        q1_x.append(q1)
        q1_y.append(col)
        med_x.append(med)
        med_y.append(col)
        q3_x.append(q3)
        q3_y.append(col)

        for asset, val in series.items():
            fv = float(val)
            pt_x.append(fv)
            pt_y.append(col)
            pt_hover.append(f"<b>{asset}</b><br>{col}: {_fmt(col, fv)}")

    if not pt_x:
        return fig

    all_pct = all(c in percent_cols for c in columns)
    all_ratio = all(c in ratio_cols for c in columns)
    if all_pct:
        q_hover = lambda label: f"{label}: %{{x:.2%}}<extra>%{{y}}</extra>"
    elif all_ratio:
        q_hover = lambda label: f"{label}: %{{x:.2f}}<extra>%{{y}}</extra>"
    else:
        q_hover = lambda label: f"{label}: %{{x:.3f}}<extra>%{{y}}</extra>"

    fig.add_trace(
        go.Scatter(
            x=q1_x,
            y=q1_y,
            mode="markers",
            name="Q1",
            marker=dict(symbol="line-ns-open", size=14, color="#64748b", line=dict(width=2)),
            hovertemplate=q_hover("Q1"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=med_x,
            y=med_y,
            mode="markers",
            name="Médiane",
            marker=dict(symbol="diamond", size=11, color="#1e40af", line=dict(width=1, color="white")),
            hovertemplate=q_hover("Médiane"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=q3_x,
            y=q3_y,
            mode="markers",
            name="Q3",
            marker=dict(symbol="line-ns-open", size=14, color="#64748b", line=dict(width=2)),
            hovertemplate=q_hover("Q3"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pt_x,
            y=pt_y,
            mode="markers",
            name="Actifs",
            marker=dict(size=9, color="#6366f1", line=dict(width=1, color="white")),
            customdata=pt_hover,
            hovertemplate="%{customdata}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Métriques de risque — étendue entre actifs (survol = nom du titre)",
        xaxis_title="Valeur",
        yaxis_title="Mesure",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        height=max(360, 72 * len(columns) + 120),
    )
    fig.update_yaxes(autorange="reversed")
    if all_pct:
        fig.update_xaxes(tickformat=".2%")
    elif all_ratio:
        fig.update_xaxes(tickformat=".2f")

    return fig


def compute_max_drawdown(price_series):
    s = price_series.dropna()
    if s.empty:
        return np.nan
    running_max = s.cummax()
    drawdown = s / running_max - 1
    return float(drawdown.min())


def compute_rsi(series, period=14):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < period + 1:
        return pd.Series(dtype=float)
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))


def compute_macd(series, fast=12, slow=26, signal=9):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < slow + signal:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
    ema_fast = s.ewm(span=fast, adjust=False).mean()
    ema_slow = s.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_sma(series, period):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < period:
        return pd.Series(dtype=float)
    return s.rolling(period).mean()


def compute_bollinger(series, period=20, num_std=2):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < period:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
    mid = s.rolling(period).mean()
    std = s.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def compute_stochastic(series, period=14, smooth=3):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < period + smooth:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    low_min = s.rolling(period).min()
    high_max = s.rolling(period).max()
    k = 100 * (s - low_min) / (high_max - low_min + 1e-8)
    d = k.rolling(smooth).mean()
    return k, d


def compute_atr(price_series, period=14):
    s = pd.to_numeric(price_series, errors="coerce").dropna()
    if len(s) < period + 1:
        return pd.Series(dtype=float)
    tr = s.diff().abs()
    return tr.rolling(period).mean()


def compute_true_range(high, low, close):
    """True Range (OHLC) — max(H-L, |H-Cp|, |L-Cp|)."""
    h = pd.to_numeric(high, errors="coerce")
    l = pd.to_numeric(low, errors="coerce")
    c = pd.to_numeric(close, errors="coerce")
    aligned = pd.concat([h, l, c], axis=1, keys=["high", "low", "close"]).dropna()
    if len(aligned) < 2:
        return pd.Series(dtype=float)
    prev_close = aligned["close"].shift(1)
    tr = pd.concat(
        [
            aligned["high"] - aligned["low"],
            (aligned["high"] - prev_close).abs(),
            (aligned["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    if len(tr):
        tr.iloc[0] = aligned["high"].iloc[0] - aligned["low"].iloc[0]
    return tr


def compute_wilder_atr(high, low, close, period=10):
    """ATR de Wilder (lissage RMA) — standard SuperTrend."""
    tr = compute_true_range(high, low, close)
    if len(tr) < period:
        return pd.Series(dtype=float)
    atr_vals = np.full(len(tr), np.nan)
    atr_vals[period - 1] = float(tr.iloc[:period].mean())
    for i in range(period, len(tr)):
        atr_vals[i] = (atr_vals[i - 1] * (period - 1) + float(tr.iloc[i])) / period
    return pd.Series(atr_vals, index=tr.index)


def compute_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    SuperTrend (ATR Wilder + bandes hl2 ± mult×ATR).
    Retourne (ligne SuperTrend, direction) avec direction 1 = haussier, -1 = baissier.
    """
    h = pd.to_numeric(high, errors="coerce")
    l = pd.to_numeric(low, errors="coerce")
    c = pd.to_numeric(close, errors="coerce")
    aligned = pd.concat([h, l, c], axis=1, keys=["high", "low", "close"]).dropna()
    if len(aligned) < period + 2:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    high_s = aligned["high"].to_numpy()
    low_s = aligned["low"].to_numpy()
    close_s = aligned["close"].to_numpy()
    atr = compute_wilder_atr(aligned["high"], aligned["low"], aligned["close"], period)
    if atr.empty or atr.notna().sum() < 2:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    hl2 = (high_s + low_s) / 2.0
    atr_np = atr.reindex(aligned.index).to_numpy()
    basic_ub = hl2 + multiplier * atr_np
    basic_lb = hl2 - multiplier * atr_np

    n = len(aligned)
    final_ub = np.full(n, np.nan)
    final_lb = np.full(n, np.nan)
    st = np.full(n, np.nan)
    direction = np.full(n, np.nan)

    start = int(np.argmax(~np.isnan(atr_np)))
    if np.isnan(atr_np[start]):
        return pd.Series(dtype=float), pd.Series(dtype=float)

    final_ub[start] = basic_ub[start]
    final_lb[start] = basic_lb[start]
    st[start] = final_lb[start]
    direction[start] = 1

    for i in range(start + 1, n):
        if np.isnan(basic_ub[i]) or np.isnan(basic_lb[i]):
            st[i] = st[i - 1]
            direction[i] = direction[i - 1]
            final_ub[i] = final_ub[i - 1]
            final_lb[i] = final_lb[i - 1]
            continue

        if basic_ub[i] < final_ub[i - 1] or close_s[i - 1] > final_ub[i - 1]:
            final_ub[i] = basic_ub[i]
        else:
            final_ub[i] = final_ub[i - 1]

        if basic_lb[i] > final_lb[i - 1] or close_s[i - 1] < final_lb[i - 1]:
            final_lb[i] = basic_lb[i]
        else:
            final_lb[i] = final_lb[i - 1]

        if close_s[i] > final_ub[i - 1]:
            direction[i] = 1
        elif close_s[i] < final_lb[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]

        st[i] = final_lb[i] if direction[i] == 1 else final_ub[i]

    idx = aligned.index
    return pd.Series(st, index=idx), pd.Series(direction, index=idx)


def supertrend_signal(direction_val):
    if direction_val is None or pd.isna(direction_val):
        return "N/A"
    if float(direction_val) > 0:
        return "Haussier"
    if float(direction_val) < 0:
        return "Baissier"
    return "Neutre"


def rsi_signal(rsi_val):
    if rsi_val is None or pd.isna(rsi_val):
        return "N/A"
    if rsi_val >= 70:
        return "Surachat"
    if rsi_val <= 30:
        return "Survente"
    return "Neutre"


def stochastic_signal(k_val):
    if k_val is None or pd.isna(k_val):
        return "N/A"
    if k_val >= 80:
        return "Surachat"
    if k_val <= 20:
        return "Survente"
    return "Neutre"


def _last_valid(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.iloc[-1])


def _prev_valid(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 2:
        return None
    return float(s.iloc[-2])


def interpret_rsi_comment(rsi_series, period=14):
    """Commentaire pédagogique sur le RSI (dernière valeur + tendance récente)."""
    val = _last_valid(rsi_series)
    if val is None:
        return "Données RSI insuffisantes pour commenter ce graphique."

    prev = _prev_valid(rsi_series)
    trend = ""
    if prev is not None:
        if val > prev + 1:
            trend = " Il remonte légèrement."
        elif val < prev - 1:
            trend = " Il descend légèrement."

    if val >= 70:
        situation = (
            f"**Situation :** RSI à **{val:.0f}** — zone de **surachat** (≥ 70). "
            "Le titre a beaucoup monté récemment ; une pause ou une correction est possible."
        )
    elif val <= 30:
        situation = (
            f"**Situation :** RSI à **{val:.0f}** — zone de **survente** (≤ 30). "
            "Le titre a beaucoup baissé ; un rebond technique est possible, sans garantie."
        )
    else:
        situation = (
            f"**Situation :** RSI à **{val:.0f}** — zone **neutre** (entre 30 et 70). "
            "Pas de signal extrême de surachat ou survente."
        )

    guide = (
        f"**Comment lire :** le RSI mesure la force des hausses vs des baisses sur **{period} jours**, "
        "sur une échelle de 0 à 100. Au-dessus de 70 = prix « tendu » à la hausse ; "
        "en dessous de 30 = prix « tendu » à la baisse. Ce n'est pas un ordre d'achat ou de vente."
    )
    return f"{situation}{trend}\n\n{guide}"


def interpret_stochastic_comment(stoch_k, stoch_d):
    """Commentaire pédagogique sur le stochastique (%K et %D)."""
    k = _last_valid(stoch_k)
    d = _last_valid(stoch_d)
    if k is None:
        return "Données stochastiques insuffisantes pour commenter ce graphique."

    if k >= 80:
        zone = f"**Situation :** %K à **{k:.0f}** — zone de **surachat** (≥ 80)."
    elif k <= 20:
        zone = f"**Situation :** %K à **{k:.0f}** — zone de **survente** (≤ 20)."
    else:
        zone = f"**Situation :** %K à **{k:.0f}** — zone **intermédiaire**."

    cross = ""
    if d is not None:
        k_prev = _prev_valid(stoch_k)
        d_prev = _prev_valid(stoch_d)
        if k_prev is not None and d_prev is not None:
            if k_prev <= d_prev and k > d:
                cross = " %K vient de **croiser %D vers le haut** (signal haussier possible)."
            elif k_prev >= d_prev and k < d:
                cross = " %K vient de **croiser %D vers le bas** (signal baissier possible)."
        if k > d:
            cross += " %K est au-dessus de %D (élan haussier court terme)." if not cross else ""
        elif k < d:
            cross += " %K est sous %D (élan baissier court terme)." if not cross else ""

    guide = (
        "**Comment lire :** le stochastique compare le cours actuel au plus haut / plus bas "
        "des **14 derniers jours**. %K (ligne rapide) réagit vite ; %D (ligne lente) lisse le signal. "
        "Au-dessus de 80 = prix proche du haut récent ; en dessous de 20 = proche du bas récent."
    )
    return f"{zone}{cross}\n\n{guide}"


def interpret_macd_comment(macd_line, signal_line, histogram):
    """Commentaire pédagogique sur le MACD."""
    macd = _last_valid(macd_line)
    signal = _last_valid(signal_line)
    hist = _last_valid(histogram)
    if macd is None or signal is None:
        return "Données MACD insuffisantes pour commenter ce graphique."

    if macd > signal:
        momentum = (
            f"**Situation :** la ligne MACD (**{macd:.3f}**) est **au-dessus** du signal (**{signal:.3f}**) — "
            "élan **haussier** sur la période analysée."
        )
    elif macd < signal:
        momentum = (
            f"**Situation :** la ligne MACD (**{macd:.3f}**) est **sous** le signal (**{signal:.3f}**) — "
            "élan **baissier** sur la période analysée."
        )
    else:
        momentum = "**Situation :** MACD et signal sont très proches — élan **neutre**."

    hist_txt = ""
    if hist is not None:
        if hist > 0:
            hist_txt = f" L'histogramme est **positif** ({hist:.3f}) : l'élan haussier **accélère**."
        elif hist < 0:
            hist_txt = f" L'histogramme est **négatif** ({hist:.3f}) : l'élan baissier **domine**."
        else:
            hist_txt = " L'histogramme est proche de zéro."

    macd_prev = _prev_valid(macd_line)
    signal_prev = _prev_valid(signal_line)
    cross = ""
    if macd_prev is not None and signal_prev is not None:
        if macd_prev <= signal_prev and macd > signal:
            cross = " Croisement récent MACD **au-dessus** du signal (souvent interprété comme signal d'achat)."
        elif macd_prev >= signal_prev and macd < signal:
            cross = " Croisement récent MACD **sous** le signal (souvent interprété comme signal de vente)."

    guide = (
        "**Comment lire :** le MACD compare deux moyennes mobiles (12 et 26 jours). "
        "Quand la ligne MACD passe **au-dessus** de la ligne Signal, la dynamique s'améliore ; "
        "quand elle passe **en dessous**, elle se dégrade. "
        "L'histogramme (barres) montre l'écart entre les deux : barres qui grandissent = mouvement qui gagne en force."
    )
    return f"{momentum}{hist_txt}{cross}\n\n{guide}"


def interpret_supertrend_comment(close_series, supertrend_series, direction_series, *, period=10, multiplier=3.0):
    """Commentaire pédagogique sur le SuperTrend."""
    price = _last_valid(close_series)
    st_val = _last_valid(supertrend_series)
    direction = _last_valid(direction_series)
    if price is None or st_val is None or direction is None:
        return "Données SuperTrend insuffisantes pour commenter ce graphique."

    sig = supertrend_signal(direction)
    dist_pct = (price - st_val) / price * 100 if price else 0.0

    if sig == "Haussier":
        situation = (
            f"**Situation :** tendance **haussière** — le cours (**{price:.2f}**) est "
            f"au-dessus du SuperTrend (**{st_val:.2f}**, +{dist_pct:.1f} %). "
            "La ligne agit comme un **support dynamique** ; tant qu'elle reste sous le prix, "
            "la dynamique est considérée favorable."
        )
    elif sig == "Baissier":
        situation = (
            f"**Situation :** tendance **baissière** — le cours (**{price:.2f}**) est "
            f"sous le SuperTrend (**{st_val:.2f}**, {dist_pct:.1f} %). "
            "La ligne agit comme une **résistance dynamique** ; tant qu'elle reste au-dessus du prix, "
            "la pression vendeuse domine."
        )
    else:
        situation = "**Situation :** signal SuperTrend **neutre** ou en transition."

    dir_prev = _prev_valid(direction_series)
    flip = ""
    if dir_prev is not None and np.sign(direction) != np.sign(dir_prev):
        if direction > 0:
            flip = " **Retournement récent** vers le signal haussier (cassure au-dessus de la bande)."
        else:
            flip = " **Retournement récent** vers le signal baissier (cassure sous la bande)."

    guide = (
        f"**Comment lire :** le SuperTrend combine la médiane **(H+L)/2** et l'**ATR Wilder ({period} j)** "
        f"× **{multiplier:g}**. Ligne **verte** = tendance haussière (support) · **rouge** = baissière (résistance). "
        "Un changement de couleur indique souvent un **retournement de tendance** — à croiser avec le contexte global."
    )
    return f"{situation}{flip}\n\n{guide}"


def _add_supertrend_traces(fig, supertrend_series, direction_series, *, row=1, col=1):
    """SuperTrend bicolore sur le panneau prix (vert haussier, rouge baissier)."""
    st = pd.to_numeric(supertrend_series, errors="coerce")
    direction = pd.to_numeric(direction_series, errors="coerce")
    aligned = pd.concat([st, direction], axis=1, keys=["st", "dir"]).dropna()
    if aligned.empty:
        return

    bull = aligned["st"].where(aligned["dir"] > 0)
    bear = aligned["st"].where(aligned["dir"] < 0)
    fig.add_trace(
        go.Scatter(
            x=bull.index,
            y=bull,
            mode="lines",
            name="SuperTrend (haussier)",
            line=dict(color="#2e7d32", width=2),
            connectgaps=False,
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=bear.index,
            y=bear,
            mode="lines",
            name="SuperTrend (baissier)",
            line=dict(color="#c62828", width=2),
            connectgaps=False,
        ),
        row=row,
        col=col,
    )


def compute_obv(close, volume):
    """On-Balance Volume : cumule le volume selon la direction du cours."""
    close = pd.to_numeric(close, errors="coerce")
    volume = pd.to_numeric(volume, errors="coerce")
    aligned = pd.concat([close, volume], axis=1, keys=["close", "volume"]).dropna()
    if len(aligned) < 2:
        return pd.Series(dtype=float)
    c = aligned["close"]
    v = aligned["volume"]
    direction = np.sign(c.diff()).fillna(0)
    return (direction * v).cumsum()


def compute_volume_sma(volume, period=20):
    v = pd.to_numeric(volume, errors="coerce").dropna()
    if len(v) < period:
        return pd.Series(dtype=float)
    return v.rolling(period).mean()


def compute_mfi(high, low, close, volume, period=14):
    """Money Flow Index : RSI pondéré par le volume (0–100)."""
    h = pd.to_numeric(high, errors="coerce")
    l = pd.to_numeric(low, errors="coerce")
    c = pd.to_numeric(close, errors="coerce")
    v = pd.to_numeric(volume, errors="coerce")
    aligned = pd.concat([h, l, c, v], axis=1, keys=["high", "low", "close", "volume"]).dropna()
    if len(aligned) < period + 1:
        return pd.Series(dtype=float)
    tp = (aligned["high"] + aligned["low"] + aligned["close"]) / 3
    mf = tp * aligned["volume"]
    delta = tp.diff()
    pos_mf = mf.where(delta > 0, 0.0).rolling(period).sum()
    neg_mf = mf.where(delta < 0, 0.0).rolling(period).sum()
    return 100 - (100 / (1 + pos_mf / (neg_mf + 1e-8)))


def volume_ratio_signal(ratio):
    if ratio is None or pd.isna(ratio):
        return "N/A"
    if ratio >= 1.5:
        return "Volume élevé"
    if ratio <= 0.7:
        return "Volume faible"
    return "Volume normal"


def mfi_signal(mfi_val):
    if mfi_val is None or pd.isna(mfi_val):
        return "N/A"
    if mfi_val >= 80:
        return "Surachat"
    if mfi_val <= 20:
        return "Survente"
    return "Neutre"


def obv_trend_signal(obv_series, lookback=5):
    obv = pd.to_numeric(obv_series, errors="coerce").dropna()
    if len(obv) < lookback + 1:
        return "N/A"
    delta = float(obv.iloc[-1] - obv.iloc[-1 - lookback])
    if delta > 0:
        return "Accumulation"
    if delta < 0:
        return "Distribution"
    return "Neutre"


def interpret_volume_comment(volume_series, close_series, period=20):
    """Commentaire pédagogique sur les volumes échangés."""
    vol = pd.to_numeric(volume_series, errors="coerce").dropna()
    close = pd.to_numeric(close_series, errors="coerce").dropna()
    if vol.empty:
        return "Données de volume indisponibles pour ce titre."

    last_vol = float(vol.iloc[-1])
    vol_sma = compute_volume_sma(vol, period)
    avg = _last_valid(vol_sma)
    ratio = last_vol / avg if avg and avg > 0 else None

    price_chg = None
    if len(close) >= 2:
        price_chg = float(close.iloc[-1] - close.iloc[-2])

    if ratio is not None:
        if ratio >= 1.5:
            intensity = (
                f"**Situation :** volume du jour **élevé** ({last_vol:,.0f} titres, "
                f"**{ratio:.1f}×** la moyenne sur {period} jours)."
            )
        elif ratio <= 0.7:
            intensity = (
                f"**Situation :** volume du jour **faible** ({last_vol:,.0f} titres, "
                f"**{ratio:.1f}×** la moyenne sur {period} jours)."
            )
        else:
            intensity = (
                f"**Situation :** volume du jour **dans la normale** ({last_vol:,.0f} titres, "
                f"**{ratio:.1f}×** la moyenne sur {period} jours)."
            )
    else:
        intensity = f"**Situation :** volume du jour : **{last_vol:,.0f}** titres échangés."

    confirmation = ""
    if price_chg is not None and ratio is not None and ratio >= 1.2:
        if price_chg > 0:
            confirmation = (
                " Fort volume sur une **journée haussière** : le mouvement à la hausse "
                "est **confirmé** par l'intérêt des investisseurs."
            )
        elif price_chg < 0:
            confirmation = (
                " Fort volume sur une **journée baissière** : la baisse est **accompagnée** "
                "d'une forte activité (ventes marquées)."
            )
        else:
            confirmation = " Volume élevé mais cours stable : possible **hésitation** ou **changement de mains**."

    guide = (
        "**Comment lire :** le volume compte **combien de titres** ont été échangés. "
        "Un mouvement de prix avec un volume **supérieur à la moyenne** est souvent jugé plus "
        "« solide » qu'un mouvement sur volume faible. Comparez les barres au trait orange "
        f"(moyenne mobile sur **{period} jours**)."
    )
    return f"{intensity}{confirmation}\n\n{guide}"


def interpret_obv_comment(obv_series, close_series, lookback=5):
    """Commentaire pédagogique sur l'OBV."""
    obv = pd.to_numeric(obv_series, errors="coerce").dropna()
    close = pd.to_numeric(close_series, errors="coerce").dropna()
    if obv.empty:
        return "Données OBV insuffisantes pour commenter ce graphique."

    obv_delta = float(obv.iloc[-1] - obv.iloc[-1 - lookback]) if len(obv) > lookback else 0.0
    price_delta = float(close.iloc[-1] - close.iloc[-1 - lookback]) if len(close) > lookback else 0.0

    if obv_delta > 0 and price_delta >= 0:
        situation = (
            "**Situation :** l'OBV **monte** en même temps que le cours — "
            "signal d'**accumulation** (achats progressifs)."
        )
    elif obv_delta < 0 and price_delta <= 0:
        situation = (
            "**Situation :** l'OBV **baisse** avec le cours — "
            "signal de **distribution** (ventes progressives)."
        )
    elif obv_delta > 0 and price_delta < 0:
        situation = (
            "**Situation :** l'OBV **monte** alors que le cours **recule** — "
            "possible **accumulation cachée** (achats malgré la baisse affichée)."
        )
    elif obv_delta < 0 and price_delta > 0:
        situation = (
            "**Situation :** l'OBV **baisse** alors que le cours **monte** — "
            "possible **faiblesse cachée** (ventes malgré la hausse affichée)."
        )
    else:
        situation = "**Situation :** OBV et cours évoluent de façon **neutre** sur la période récente."

    guide = (
        "**Comment lire :** l'OBV (On-Balance Volume) **additionne le volume** les jours de hausse "
        "et le **soustrait** les jours de baisse. Une ligne qui **monte** suggère que les acheteurs "
        "dominent ; une ligne qui **descend**, que les vendeurs dominent. "
        "Regardez si l'OBV **confirme** ou **contredit** le mouvement du prix."
    )
    return f"{situation}\n\n{guide}"


def interpret_mfi_comment(mfi_series, period=14):
    """Commentaire pédagogique sur le MFI (Money Flow Index)."""
    val = _last_valid(mfi_series)
    if val is None:
        return "Données MFI insuffisantes pour commenter ce graphique."

    prev = _prev_valid(mfi_series)
    trend = ""
    if prev is not None:
        if val > prev + 2:
            trend = " Il remonte."
        elif val < prev - 2:
            trend = " Il descend."

    if val >= 80:
        situation = (
            f"**Situation :** MFI à **{val:.0f}** — zone de **surachat** (≥ 80). "
            "L'argent entre fortement ; le titre peut être « surchauffé » à court terme."
        )
    elif val <= 20:
        situation = (
            f"**Situation :** MFI à **{val:.0f}** — zone de **survente** (≤ 20). "
            "Les flux monétaires sont faibles ; un rebond technique est parfois observé."
        )
    else:
        situation = (
            f"**Situation :** MFI à **{val:.0f}** — zone **intermédiaire**. "
            "Pas de signal extrême côté flux monétaires."
        )

    guide = (
        f"**Comment lire :** le MFI ressemble au RSI mais intègre le **volume** et le **prix typique** "
        f"(moyenne haut/bas/clôture) sur **{period} jours**. "
        "Au-dessus de 80, l'argent « pousse » fortement le titre ; en dessous de 20, il se retire. "
        "Utile pour voir si un mouvement de prix est soutenu par de **vrais échanges**."
    )
    return f"{situation}{trend}\n\n{guide}"


FIBONACCI_RATIOS = {
    "0.0%": 0.0,
    "23.6%": 0.236,
    "38.2%": 0.382,
    "50.0%": 0.5,
    "61.8%": 0.618,
    "78.6%": 0.786,
    "100.0%": 1.0,
}


def compute_linear_regression_channel(price_series):
    """Régression linéaire + bandes parallèles à ±1σ et ±2σ (écart-type des résidus)."""
    s = pd.to_numeric(price_series, errors="coerce").dropna()
    if len(s) < 10:
        return None

    x = np.arange(len(s), dtype=float)
    y = s.values.astype(float)
    slope, intercept = np.polyfit(x, y, 1)
    reg = intercept + slope * x
    residuals = y - reg
    std = float(np.std(residuals, ddof=1))

    idx = s.index
    reg_s = pd.Series(reg, index=idx)
    return {
        "regression": reg_s,
        "plus_1std": pd.Series(reg + std, index=idx),
        "minus_1std": pd.Series(reg - std, index=idx),
        "plus_2std": pd.Series(reg + 2 * std, index=idx),
        "minus_2std": pd.Series(reg - 2 * std, index=idx),
        "slope": float(slope),
        "std": std,
        "slope_pct_annualized": float(slope / y.mean() * 252) if y.mean() else 0.0,
    }


def compute_log_regression_channel(price_series):
    """Régression log-linéaire (log prix ~ temps) + bandes ±1σ / ±2σ en espace log."""
    s = pd.to_numeric(price_series, errors="coerce").dropna()
    if len(s) < 10 or (s <= 0).any():
        return None

    x = np.arange(len(s), dtype=float)
    log_y = np.log(s.values.astype(float))
    slope, intercept = np.polyfit(x, log_y, 1)
    reg_log = intercept + slope * x
    residuals = log_y - reg_log
    std = float(np.std(residuals, ddof=1))

    idx = s.index
    return {
        "regression": pd.Series(np.exp(reg_log), index=idx),
        "plus_1std": pd.Series(np.exp(reg_log + std), index=idx),
        "minus_1std": pd.Series(np.exp(reg_log - std), index=idx),
        "plus_2std": pd.Series(np.exp(reg_log + 2 * std), index=idx),
        "minus_2std": pd.Series(np.exp(reg_log - 2 * std), index=idx),
        "slope": float(slope),
        "std": std,
        "slope_pct_annualized": float(np.exp(slope * 252) - 1.0),
    }


_REGRESSION_BAND_STYLE = (
    ("plus_1std", "+1σ", "dash", 1.2),
    ("minus_1std", "−1σ", "dash", 1.2),
    ("plus_2std", "+2σ", "dot", 1.0),
    ("minus_2std", "−2σ", "dot", 1.0),
)


def build_regression_channel_figure(
    price_series,
    *,
    scale="linear",
    title="Régression & bandes ±1σ / ±2σ",
    series_name="Série",
    yaxis_title="Valeur",
):
    """Graphique prix + droite de régression et bandes parallèles (linéaire ou log)."""
    s = pd.to_numeric(price_series, errors="coerce").dropna()
    if len(s) < 10:
        return None

    if scale == "log":
        channel = compute_log_regression_channel(s)
    else:
        channel = compute_linear_regression_channel(s)
    if channel is None:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=s.index,
            y=s.values,
            mode="lines",
            name=series_name,
            line=dict(color="#1e40af", width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=channel["regression"].index,
            y=channel["regression"],
            mode="lines",
            name="Régression",
            line=dict(color="#1565c0", width=2),
        )
    )
    for key, label, dash, width in _REGRESSION_BAND_STYLE:
        fig.add_trace(
            go.Scatter(
                x=channel[key].index,
                y=channel[key],
                mode="lines",
                name=label,
                line=dict(color="#64b5f6", width=width, dash=dash),
            )
        )

    yaxis = dict(title=yaxis_title)
    if scale == "log":
        yaxis["type"] = "log"

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis=yaxis,
        height=460,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis_rangeslider_visible=False,
    )
    return fig


def compute_support_resistance(price_series, window=10, n_levels=2):
    """Supports (plans) et résistances (plafonds) via extrema locaux."""
    s = pd.to_numeric(price_series, errors="coerce").dropna()
    if len(s) < window * 2 + 1:
        return [], []

    vals = s.values
    idx = s.index
    highs = []
    lows = []
    for i in range(window, len(s) - window):
        segment = vals[i - window : i + window + 1]
        if vals[i] == segment.max():
            highs.append(float(vals[i]))
        if vals[i] == segment.min():
            lows.append(float(vals[i]))

    def _pick_levels(candidates, fallback, reverse=False):
        unique = sorted(set(candidates), reverse=reverse)
        picked = []
        for price in unique:
            if not picked or all(abs(price - p) / max(p, 1e-8) > 0.015 for p in picked):
                picked.append(price)
            if len(picked) >= n_levels:
                break
        if not picked:
            picked = [float(fallback)]
        return sorted(picked)

    supports = _pick_levels(lows, s.min(), reverse=False)
    resistances = _pick_levels(highs, s.max(), reverse=True)
    return supports, resistances


def _apply_plotly_crosshair(fig, n_rows: int = 1, *, subplot: bool = True) -> None:
    """Spikes synchronisés entre sous-graphiques (survol date commune)."""
    fig.update_layout(hovermode="x unified")
    if subplot and n_rows > 1:
        for row in range(1, n_rows + 1):
            fig.update_xaxes(
                showspikes=True,
                spikemode="across+marker",
                spikesnap="cursor",
                spikedash="dot",
                spikethickness=1,
                row=row,
                col=1,
            )
            fig.update_yaxes(showspikes=True, spikemode="across", row=row, col=1)
    else:
        fig.update_xaxes(
            showspikes=True,
            spikemode="across+marker",
            spikesnap="cursor",
            spikedash="dot",
            spikethickness=1,
        )
        fig.update_yaxes(showspikes=True, spikemode="across")


def _add_rsi_zones(fig, *, row: int | None = None, col: int = 1) -> None:
    kwargs = {"row": row, "col": col} if row else {}
    fig.add_hrect(
        y0=0,
        y1=30,
        fillcolor="rgba(46, 125, 50, 0.18)",
        line_width=0,
        layer="below",
        **kwargs,
    )
    fig.add_hrect(
        y0=70,
        y1=100,
        fillcolor="rgba(198, 40, 40, 0.18)",
        line_width=0,
        layer="below",
        **kwargs,
    )
    fig.add_hline(y=30, line_dash="dot", line_color="#2e7d32", line_width=1, **kwargs)
    fig.add_hline(y=70, line_dash="dot", line_color="#c62828", line_width=1, **kwargs)


def _add_stoch_zones(fig) -> None:
    fig.add_hrect(y0=0, y1=20, fillcolor="rgba(46, 125, 50, 0.18)", line_width=0, layer="below")
    fig.add_hrect(y0=80, y1=100, fillcolor="rgba(198, 40, 40, 0.18)", line_width=0, layer="below")
    fig.add_hline(y=20, line_dash="dot", line_color="#2e7d32", line_width=1)
    fig.add_hline(y=80, line_dash="dot", line_color="#c62828", line_width=1)


def _add_support_resistance_lines(fig, supports, resistances, *, row: int = 1, col: int = 1) -> None:
    supports = supports or []
    resistances = resistances or []
    for i, level in enumerate(supports, start=1):
        fig.add_hline(
            y=level,
            line_dash="longdash",
            line_color="#2e7d32",
            line_width=1.2,
            annotation_text=f"S{i}",
            annotation_position="bottom left",
            annotation_font_size=9,
            annotation_font_color="#2e7d32",
            row=row,
            col=col,
        )
    for i, level in enumerate(resistances, start=1):
        fig.add_hline(
            y=level,
            line_dash="longdash",
            line_color="#c62828",
            line_width=1.2,
            annotation_text=f"R{i}",
            annotation_position="top left",
            annotation_font_size=9,
            annotation_font_color="#c62828",
            row=row,
            col=col,
        )


def build_technical_overview_figure(
    price_series,
    *,
    detail_label="Actif",
    ohlc=None,
    volume_series=None,
    sma50=None,
    sma200=None,
    bb_upper=None,
    bb_mid=None,
    bb_lower=None,
    rsi_series=None,
    rsi_period=14,
    supertrend_series=None,
    supertrend_direction=None,
    supports=None,
    resistances=None,
):
    """
    Vue liée : chandeliers (+ MM, Bollinger, S/R), volume, RSI avec zones survente/surachat.
    """
    from chart_theme import CHART_HEIGHT, apply_chart_theme
    from data_loader import candlestick_trace

    s = pd.to_numeric(price_series, errors="coerce").dropna()
    if s.empty:
        return None

    has_ohlc = ohlc is not None and not ohlc.empty
    vol = pd.to_numeric(volume_series, errors="coerce").reindex(s.index) if volume_series is not None else None
    has_volume = vol is not None and vol.dropna().size >= 5

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.54, 0.18, 0.28],
        subplot_titles=(
            f"{detail_label} — Prix, moyennes, SuperTrend & niveaux S/R",
            "Volume échangé",
            f"RSI ({rsi_period})",
        ),
    )

    if has_ohlc:
        fig.add_trace(candlestick_trace(ohlc), row=1, col=1)
    else:
        fig.add_trace(
            go.Scatter(x=s.index, y=s, mode="lines", name="Prix", line=dict(color="#1e40af", width=2)),
            row=1,
            col=1,
        )

    if sma50 is not None and not sma50.empty:
        fig.add_trace(
            go.Scatter(x=sma50.index, y=sma50, mode="lines", name="SMA 50", line=dict(width=1.2)),
            row=1,
            col=1,
        )
    if sma200 is not None and not sma200.empty:
        fig.add_trace(
            go.Scatter(x=sma200.index, y=sma200, mode="lines", name="SMA 200", line=dict(width=1.2)),
            row=1,
            col=1,
        )
    if bb_upper is not None and not bb_upper.empty:
        if bb_mid is not None and not bb_mid.empty:
            fig.add_trace(
                go.Scatter(
                    x=bb_mid.index,
                    y=bb_mid,
                    mode="lines",
                    name="SMA 20 (Bollinger)",
                    line=dict(color="orange", width=1.2),
                ),
                row=1,
                col=1,
            )
        fig.add_trace(
            go.Scatter(x=bb_upper.index, y=bb_upper, mode="lines", name="Bollinger haut", line=dict(dash="dot")),
            row=1,
            col=1,
        )
        if bb_lower is not None and not bb_lower.empty:
            fig.add_trace(
                go.Scatter(x=bb_lower.index, y=bb_lower, mode="lines", name="Bollinger bas", line=dict(dash="dot")),
                row=1,
                col=1,
            )

    if supertrend_series is not None and supertrend_direction is not None:
        if not supertrend_series.empty:
            _add_supertrend_traces(
                fig,
                supertrend_series.reindex(s.index),
                supertrend_direction.reindex(s.index),
                row=1,
                col=1,
            )

    _add_support_resistance_lines(fig, supports, resistances, row=1, col=1)

    if has_volume:
        vol_idx = vol.dropna().index
        bar_colors = []
        for d in vol_idx:
            if has_ohlc and d in ohlc.index:
                bar_colors.append(
                    "#26a69a" if ohlc.loc[d, "close"] >= ohlc.loc[d, "open"] else "#ef5350"
                )
            elif d in s.index and pd.notna(s.loc[d]) and pd.notna(s.shift(1).loc[d]):
                bar_colors.append("#26a69a" if s.loc[d] >= s.shift(1).loc[d] else "#ef5350")
            else:
                bar_colors.append("#42a5f5")
        fig.add_trace(
            go.Bar(x=vol_idx, y=vol.loc[vol_idx], name="Volume", marker_color=bar_colors, opacity=0.7),
            row=2,
            col=1,
        )
        vol_sma20 = compute_volume_sma(vol, 20)
        if not vol_sma20.empty:
            fig.add_trace(
                go.Scatter(
                    x=vol_sma20.index,
                    y=vol_sma20,
                    mode="lines",
                    name="Moy. volume 20j",
                    line=dict(color="orange", width=1.5),
                ),
                row=2,
                col=1,
            )

    if rsi_series is not None and not rsi_series.empty:
        fig.add_trace(
            go.Scatter(x=rsi_series.index, y=rsi_series, mode="lines", name="RSI", line=dict(width=1.5)),
            row=3,
            col=1,
        )
        _add_rsi_zones(fig, row=3, col=1)
        fig.update_yaxes(range=[0, 100], row=3, col=1)

    fig.update_yaxes(title_text="Prix", row=1, col=1)
    fig.update_yaxes(title_text="Titres", row=2, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False)
    _apply_plotly_crosshair(fig, n_rows=3, subplot=True)
    apply_chart_theme(fig, height=CHART_HEIGHT.get("technical_overview", 720), legend_horizontal=True)
    return fig


def build_technical_stochastic_figure(stoch_k, stoch_d, *, detail_label="Actif"):
    from chart_theme import CHART_HEIGHT, apply_chart_theme

    if stoch_k is None or stoch_k.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stoch_k.index, y=stoch_k, mode="lines", name="%K"))
    if stoch_d is not None and not stoch_d.empty:
        fig.add_trace(go.Scatter(x=stoch_d.index, y=stoch_d, mode="lines", name="%D"))
    _add_stoch_zones(fig)
    fig.update_layout(yaxis=dict(range=[0, 100]))
    _apply_plotly_crosshair(fig, subplot=False)
    apply_chart_theme(
        fig,
        height=CHART_HEIGHT.get("technical_osc", 320),
        title=f"{detail_label} — Stochastique",
        legend_horizontal=True,
    )
    return fig


def build_technical_macd_figure(macd_line, signal_line, histogram, *, detail_label="Actif"):
    from chart_theme import CHART_HEIGHT, apply_chart_theme

    if macd_line is None or macd_line.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=macd_line.index, y=macd_line, mode="lines", name="MACD"))
    if signal_line is not None and not signal_line.empty:
        fig.add_trace(go.Scatter(x=signal_line.index, y=signal_line, mode="lines", name="Signal"))
    if histogram is not None and not histogram.empty:
        colors = ["#26a69a" if v >= 0 else "#ef5350" for v in histogram.fillna(0)]
        fig.add_trace(
            go.Bar(x=histogram.index, y=histogram, name="Histogramme", marker_color=colors, opacity=0.5)
        )
    _apply_plotly_crosshair(fig, subplot=False)
    apply_chart_theme(
        fig,
        height=CHART_HEIGHT.get("technical_osc", 340),
        title=f"{detail_label} — MACD",
        legend_horizontal=True,
    )
    return fig


def compute_pivot_level(price_series, highs=None, lows=None):
    """
    Point pivot (PP) : (H + L + C) / 3 sur la période analysée.
    H/L = plus haut / plus bas (OHLC si disponible, sinon clôtures).
    """
    s = pd.to_numeric(price_series, errors="coerce").dropna()
    if len(s) < 2:
        return None

    h = pd.to_numeric(highs.reindex(s.index), errors="coerce") if highs is not None else s
    l = pd.to_numeric(lows.reindex(s.index), errors="coerce") if lows is not None else s
    h = h.dropna()
    l = l.dropna()
    if h.empty or l.empty:
        return None

    swing_high = float(h.max())
    swing_low = float(l.min())
    close = float(s.iloc[-1])
    if swing_high < swing_low:
        return None
    return (swing_high + swing_low + close) / 3.0


def compute_fibonacci_levels(price_series, highs=None, lows=None):
    """Niveaux de Fibonacci entre le plus bas et le plus haut de la période."""
    s = pd.to_numeric(price_series, errors="coerce").dropna()
    if len(s) < 5:
        return None

    h = pd.to_numeric(highs.reindex(s.index), errors="coerce") if highs is not None else s
    l = pd.to_numeric(lows.reindex(s.index), errors="coerce") if lows is not None else s

    swing_high = float(h.max())
    swing_low = float(l.min())
    high_date = h.idxmax()
    low_date = l.idxmin()
    diff = swing_high - swing_low
    if diff <= 0:
        return None

    levels = {label: swing_low + ratio * diff for label, ratio in FIBONACCI_RATIOS.items()}
    return {
        "levels": levels,
        "swing_high": swing_high,
        "swing_low": swing_low,
        "high_date": high_date,
        "low_date": low_date,
    }


def interpret_regression_channel_comment(price_series, channel):
    if channel is None:
        return "Données insuffisantes pour la régression linéaire."

    s = pd.to_numeric(price_series, errors="coerce").dropna()
    price = float(s.iloc[-1])
    reg_val = _last_valid(channel["regression"])
    p1 = _last_valid(channel["plus_1std"])
    m1 = _last_valid(channel["minus_1std"])
    p2 = _last_valid(channel["plus_2std"])
    m2 = _last_valid(channel["minus_2std"])
    slope = channel["slope"]

    if reg_val is None:
        return "Données insuffisantes pour la régression linéaire."

    trend_word = "**haussière**" if slope > 0 else "**baissière**" if slope < 0 else "**plate**"
    vs_reg = price - reg_val
    if abs(vs_reg) / reg_val < 0.005:
        position = "sur la **droite de régression** (tendance « juste prix »)."
    elif vs_reg > 0:
        position = f"**{vs_reg:.2f} € au-dessus** de la droite de régression."
    else:
        position = f"**{abs(vs_reg):.2f} € en dessous** de la droite de régression."

    band = ""
    if p2 is not None and price >= p2:
        band = " Le cours dépasse **+2σ** : zone statistiquement ** très haute** (≈ 2,5 % des cas en distribution normale)."
    elif p1 is not None and price >= p1:
        band = " Le cours est entre **+1σ et +2σ** : au-dessus de la tendance moyenne."
    elif m2 is not None and price <= m2:
        band = " Le cours est sous **−2σ** : zone statistiquement ** très basse**."
    elif m1 is not None and price <= m1:
        band = " Le cours est entre **−1σ et −2σ** : en dessous de la tendance moyenne."
    else:
        band = " Le cours évolue dans le **canal central** (entre −1σ et +1σ) : proche de la tendance linéaire."

    situation = (
        f"**Situation :** tendance linéaire **{trend_word}** sur la période affichée. "
        f"Cours actuel **{price:.2f} €**, {position}{band}"
    )
    guide = (
        "**Comment lire :** la **droite bleue** est la meilleure tendance linéaire sur la période. "
        "Les bandes **±1σ** et **±2σ** (écarts-types des écarts au prix) forment un **canal** : "
        "environ 68 % des cours restent entre −1σ et +1σ, 95 % entre −2σ et +2σ "
        "(hypothèse de dispersion « normale »). Un prix aux extrêmes du canal peut signaler "
        "un **écart temporaire** par rapport à la tendance — pas forcément un retournement."
    )
    return f"{situation}\n\n{guide}"


def interpret_support_resistance_comment(price_series, supports, resistances, pivot=None):
    s = pd.to_numeric(price_series, errors="coerce").dropna()
    if s.empty:
        return "Données insuffisantes pour supports et résistances."

    price = float(s.iloc[-1])
    supports_below = sorted([lv for lv in supports if lv <= price], reverse=True)
    resistances_above = sorted([lv for lv in resistances if lv >= price])

    nearest_support = supports_below[0] if supports_below else (min(supports) if supports else None)
    nearest_resistance = resistances_above[0] if resistances_above else (max(resistances) if resistances else None)

    parts = [f"**Situation :** cours actuel **{price:.2f} €**."]
    if pivot is not None:
        dist_p = (price - pivot) / price * 100
        if abs(dist_p) < 0.5:
            parts.append(
                f" Cours **sur le pivot** (**{pivot:.2f} €**) : zone d'équilibre entre acheteurs et vendeurs."
            )
        elif price > pivot:
            parts.append(
                f" Cours **{dist_p:.1f} % au-dessus du pivot** (**{pivot:.2f} €**) : biais plutôt haussier."
            )
        else:
            parts.append(
                f" Cours **{abs(dist_p):.1f} % sous le pivot** (**{pivot:.2f} €**) : biais plutôt baissier."
            )
    if nearest_support is not None:
        dist_s = (price - nearest_support) / price * 100
        parts.append(
            f" **Plancher (support)** le plus proche : **{nearest_support:.2f} €** "
            f"({dist_s:.1f} % en dessous) — zone où le titre a déjà **rebondi** par le passé."
        )
    if nearest_resistance is not None:
        dist_r = (nearest_resistance - price) / price * 100
        parts.append(
            f" **Plafond (résistance)** le plus proche : **{nearest_resistance:.2f} €** "
            f"({dist_r:.1f} % au-dessus) — zone où le titre a déjà **buté** par le passé."
        )

    if nearest_support and nearest_resistance and nearest_resistance > nearest_support:
        range_pct = (nearest_resistance - nearest_support) / price * 100
        parts.append(
            f" Le titre évolue dans un **couloir** d'environ **{range_pct:.1f} %** "
            "entre ces deux niveaux."
        )

    guide = (
        "**Comment lire :** les lignes **vertes** (supports / planchers) marquent des prix "
        "où les acheteurs sont revenus ; les lignes **rouges** (résistances / plafonds) "
        "marquent des prix où les vendeurs ont freiné la hausse ; la ligne **bleue (Pivot)** "
        "est le point pivot **PP = (H + L + C) / 3** sur la période affichée. "
        "Un **franchissement durable** d'une résistance peut la transformer en support (et inversement)."
    )
    return " ".join(parts) + f"\n\n{guide}"


def _nearest_fib_level(price, fib_levels):
    if not fib_levels:
        return None, None
    best_label = None
    best_dist = None
    for label, level in fib_levels.items():
        dist = abs(price - level)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_label = label
    return best_label, fib_levels.get(best_label)


def interpret_fibonacci_comment(price_series, fib_data):
    if fib_data is None or not fib_data.get("levels"):
        return "Données insuffisantes pour les niveaux de Fibonacci."

    s = pd.to_numeric(price_series, errors="coerce").dropna()
    price = float(s.iloc[-1])
    levels = fib_data["levels"]
    swing_high = fib_data["swing_high"]
    swing_low = fib_data["swing_low"]

    label, nearest = _nearest_fib_level(price, levels)
    dist_pct = abs(price - nearest) / price * 100 if nearest else 0

    move = (
        "**hausse**"
        if fib_data["low_date"] < fib_data["high_date"]
        else "**baisse**"
    )
    situation = (
        f"**Situation :** sur la période, le titre est passé de **{swing_low:.2f} €** "
        f"(bas) à **{swing_high:.2f} €** (haut) — dernier grand mouvement en {move}. "
        f"Le cours **{price:.2f} €** est proche du retracement Fibonacci **{label}** "
        f"(**{nearest:.2f} €**, écart {dist_pct:.1f} %)."
    )

    if label in ("61.8%", "50.0%", "38.2%"):
        situation += (
            f" Le niveau **{label}** est **parmi les plus suivis** : zone possible de "
            "**pause**, **rebond** ou **accélération** selon le contexte."
        )
    elif label in ("0.0%", "100.0%"):
        situation += " Le cours teste un **extrême** de la fourchette récente (sommet ou plancher)."

    guide = (
        "**Comment lire :** les retracements de **Fibonacci** découpent le mouvement entre le plus bas "
        "et le plus haut en ratios (23,6 %, 38,2 %, 50 %, 61,8 %…). "
        "Après une hausse, un retracement à 61,8 % indique une correction d'environ 60 % du mouvement ; "
        "beaucoup de traders **surveillent** ces niveaux comme zones de décision. "
        "Ce ne sont pas des prédictions — des **repères visuels** sur l'historique récent."
    )
    return f"{situation}\n\n{guide}"


def ma_signal(price, sma50, sma200):
    if any(v is None or pd.isna(v) for v in [price, sma50, sma200]):
        return "N/A"
    if price > sma50 > sma200:
        return "Haussière"
    if price < sma50 < sma200:
        return "Baissière"
    return "Neutre"


def compute_technical_indicators(
    prices,
    rsi_period=14,
    volumes=None,
    highs=None,
    lows=None,
    supertrend_period=10,
    supertrend_multiplier=3.0,
):
    rows = []
    for ticker in prices.columns:
        s = prices[ticker].dropna()
        if len(s) < 30:
            continue
        rsi = compute_rsi(s, rsi_period)
        macd_line, signal_line, histogram = compute_macd(s)
        sma50 = compute_sma(s, 50)
        sma200 = compute_sma(s, 200)
        bb_upper, bb_mid, bb_lower = compute_bollinger(s)
        stoch_k, stoch_d = compute_stochastic(s)
        atr = compute_atr(s)

        h_s = s
        l_s = s
        if highs is not None and lows is not None and ticker in highs.columns and ticker in lows.columns:
            h_s = highs[ticker].reindex(s.index).fillna(s)
            l_s = lows[ticker].reindex(s.index).fillna(s)
        st_line, st_dir = compute_supertrend(
            h_s,
            l_s,
            s,
            period=supertrend_period,
            multiplier=supertrend_multiplier,
        )

        vol_s = None
        if volumes is not None and ticker in volumes.columns:
            vol_s = volumes[ticker].reindex(s.index)

        last_vol = np.nan
        vol_ratio = np.nan
        last_mfi = np.nan
        obv_sig = "N/A"
        if vol_s is not None and vol_s.dropna().size >= 20:
            last_vol = float(vol_s.iloc[-1])
            vol_avg = compute_volume_sma(vol_s, 20)
            if not vol_avg.empty and vol_avg.iloc[-1] > 0:
                vol_ratio = last_vol / float(vol_avg.iloc[-1])
            obv = compute_obv(s, vol_s)
            if not obv.empty:
                obv_sig = obv_trend_signal(obv)
            if highs is not None and lows is not None and ticker in highs.columns and ticker in lows.columns:
                h = highs[ticker].reindex(s.index)
                l = lows[ticker].reindex(s.index)
                mfi = compute_mfi(h, l, s, vol_s)
                if not mfi.empty:
                    last_mfi = float(mfi.iloc[-1])

        last_price = float(s.iloc[-1])
        last_rsi = float(rsi.iloc[-1]) if not rsi.empty else np.nan
        last_macd = float(macd_line.iloc[-1]) if not macd_line.empty else np.nan
        last_signal = float(signal_line.iloc[-1]) if not signal_line.empty else np.nan
        last_hist = float(histogram.iloc[-1]) if not histogram.empty else np.nan
        last_sma50 = float(sma50.iloc[-1]) if not sma50.empty else np.nan
        last_sma200 = float(sma200.iloc[-1]) if not sma200.empty else np.nan
        last_stoch_k = float(stoch_k.iloc[-1]) if not stoch_k.empty else np.nan
        last_stoch_d = float(stoch_d.iloc[-1]) if not stoch_d.empty else np.nan
        last_atr = float(atr.iloc[-1]) if not atr.empty else np.nan
        last_st = float(st_line.iloc[-1]) if not st_line.empty else np.nan
        last_st_dir = float(st_dir.iloc[-1]) if not st_dir.empty else np.nan

        bb_pct = np.nan
        if not bb_upper.empty and not bb_lower.empty:
            u, l = float(bb_upper.iloc[-1]), float(bb_lower.iloc[-1])
            if u > l:
                bb_pct = (last_price - l) / (u - l)

        rows.append(
            {
                "Ticker": ticker,
                "RSI": last_rsi,
                "Signal RSI": rsi_signal(last_rsi),
                "Stoch %K": last_stoch_k,
                "Stoch %D": last_stoch_d,
                "Signal Stoch": stochastic_signal(last_stoch_k),
                "MACD": last_macd,
                "Signal MACD": last_signal,
                "Histogramme": last_hist,
                "Volume": last_vol,
                "Vol / Moy. 20j": vol_ratio,
                "Signal Volume": volume_ratio_signal(vol_ratio),
                "MFI": last_mfi,
                "Signal MFI": mfi_signal(last_mfi),
                "Tendance OBV": obv_sig,
                "SMA 50": last_sma50,
                "SMA 200": last_sma200,
                "Signal MM": ma_signal(last_price, last_sma50, last_sma200),
                "SuperTrend": last_st,
                "Signal SuperTrend": supertrend_signal(last_st_dir),
                "Bollinger %B": bb_pct,
                "ATR": last_atr,
                "ATR/Prix": last_atr / last_price if last_price else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_portfolio_value(prices, quantities, usd_to_eur=1.0, is_usd_fn=None):
    is_usd_fn = is_usd_fn or (lambda _t: False)
    cols = [c for c in quantities if c in prices.columns]
    if not cols:
        return pd.Series(dtype=float)
    sub = prices[cols].copy()
    for col in cols:
        if is_usd_fn(col):
            sub[col] = sub[col] * usd_to_eur
    weights = pd.Series({c: float(quantities[c]) for c in cols})
    return sub.mul(weights, axis=1).sum(axis=1).dropna()


def _align_fx_series_to_index(fx_series, index):
    """Aligne un taux USD→EUR sur l'index des cours (forward-fill)."""
    s = pd.to_numeric(fx_series, errors="coerce").sort_index()
    if s.empty:
        return pd.Series(float("nan"), index=index)
    if hasattr(index, "tz") and index.tz is not None and s.index.tz is None:
        s.index = s.index.tz_localize(index.tz)
    elif hasattr(s.index, "tz") and s.index.tz is not None and getattr(index, "tz", None) is None:
        s.index = s.index.tz_localize(None)
    aligned = s.reindex(index).ffill().bfill()
    return aligned


def build_portfolio_holdings_eur(
    prices,
    quantities,
    usd_to_eur_series=None,
    usd_to_eur=1.0,
    is_usd_fn=None,
):
    """
    Valorisation journalière de chaque ligne du portefeuille en EUR.

    Titres USD : cours × quantité × taux USD→EUR du jour (série alignée sur les dates).
    """
    is_usd_fn = is_usd_fn or (lambda _t: False)
    cols = [
        c
        for c in quantities
        if c in prices.columns and float(quantities.get(c, 0) or 0) > 0
    ]
    if not cols:
        return pd.DataFrame()

    sub = prices[cols].copy().ffill()
    fx = None
    if usd_to_eur_series is not None and len(usd_to_eur_series):
        fx = _align_fx_series_to_index(usd_to_eur_series, sub.index)

    out = pd.DataFrame(index=sub.index, columns=cols, dtype=float)
    for col in cols:
        q = float(quantities[col])
        px = pd.to_numeric(sub[col], errors="coerce")
        if is_usd_fn(col):
            if fx is not None and fx.notna().any():
                rate = fx
            else:
                rate = float(usd_to_eur or 1.0)
            out[col] = q * px * rate
        else:
            out[col] = q * px
    return out.dropna(how="all")


def build_portfolio_weight_pct_history(
    prices,
    quantities,
    usd_to_eur_series=None,
    usd_to_eur=1.0,
    is_usd_fn=None,
):
    """Part de chaque titre dans le portefeuille (%) par date de cotation."""
    values = build_portfolio_holdings_eur(
        prices,
        quantities,
        usd_to_eur_series=usd_to_eur_series,
        usd_to_eur=usd_to_eur,
        is_usd_fn=is_usd_fn,
    )
    if values.empty or len(values) < 2:
        return pd.DataFrame()
    total = values.sum(axis=1).replace(0, np.nan)
    return values.div(total, axis=0) * 100.0


def build_portfolio_weight_history_chart(
    weight_pct,
    labels=None,
    title="Répartition du capital par titre (%)",
):
    """Graphique area stack : poids % de chaque titre dans le temps."""
    from chart_theme import CHART_HEIGHT, apply_chart_theme

    if weight_pct is None or weight_pct.empty:
        return None
    labels = labels or {}
    fig = go.Figure()
    for col in weight_pct.columns:
        fig.add_trace(
            go.Scatter(
                x=weight_pct.index,
                y=weight_pct[col],
                mode="lines",
                name=str(labels.get(col, col)),
                stackgroup="one",
                line=dict(width=0.6),
                hovertemplate="%{fullData.name}: %{y:.1f} %<extra></extra>",
            )
        )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Part du portefeuille (%)",
        yaxis=dict(range=[0, 100], ticksuffix=" %"),
        hovermode="x unified",
    )
    apply_chart_theme(
        fig,
        height=CHART_HEIGHT["stacked_area"],
        title=title,
        legend_horizontal=True,
    )
    return fig


def compute_drawdown_series(value_series):
    """Drawdown relatif (0 = pic, négatif = baisse depuis le pic)."""
    s = pd.to_numeric(value_series, errors="coerce").dropna()
    if len(s) < 2:
        return pd.Series(dtype=float)
    peak = s.cummax()
    return (s - peak) / peak.replace(0, np.nan)


def build_portfolio_drawdown_figure(
    value_series,
    title="Drawdown du portefeuille",
):
    """Courbe de drawdown (%) sous le pic cumulé."""
    from chart_theme import CHART_COLORS, CHART_HEIGHT, apply_chart_theme

    dd = compute_drawdown_series(value_series)
    if dd.empty:
        return None
    dd_pct = dd * 100.0
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dd_pct.index,
            y=dd_pct.values,
            mode="lines",
            name="Drawdown",
            fill="tozeroy",
            line=dict(color=CHART_COLORS["negative"], width=1.5),
            fillcolor="rgba(198, 40, 40, 0.25)",
            hovertemplate="%{x|%Y-%m-%d}<br>Drawdown: %{y:.2f} %<extra></extra>",
        )
    )
    max_dd = float(dd_pct.min())
    fig.add_hline(
        y=max_dd,
        line_dash="dash",
        line_color=CHART_COLORS["negative"],
        annotation_text=f"Max {max_dd:.1f} %",
        annotation_position="bottom left",
    )
    fig.update_layout(yaxis_title="Drawdown (%)", hovermode="x unified")
    apply_chart_theme(
        fig,
        height=CHART_HEIGHT["drawdown"],
        title=title,
        legend_horizontal=False,
    )
    return fig


def build_portfolio_allocation_treemap(
    df,
    *,
    value_col="Valeur Actuelle (€)",
    label_col="Nom",
    ticker_col="Ticker",
    sector_col="Secteur",
    title="Répartition du capital (€)",
):
    """Treemap hiérarchique secteur → titre (ou flat si secteur absent)."""
    import plotly.express as px
    from chart_theme import CHART_HEIGHT, apply_chart_theme

    if df is None or df.empty or value_col not in df.columns:
        return None

    plot_df = df.copy()
    if label_col in plot_df.columns:
        plot_df["_label"] = plot_df[label_col].astype(str)
    else:
        plot_df["_label"] = plot_df.get(ticker_col, plot_df.index).astype(str)

    use_sector = (
        sector_col in plot_df.columns
        and plot_df[sector_col].notna().any()
        and (plot_df[sector_col].astype(str).str.strip() != "—").any()
    )
    if use_sector:
        plot_df[sector_col] = plot_df[sector_col].fillna("Autre").astype(str)
        path = [sector_col, "_label"]
    else:
        path = ["_label"]

    fig = px.treemap(
        plot_df,
        path=path,
        values=value_col,
        color=value_col,
        color_continuous_scale="Blues",
        hover_data={ticker_col: True, value_col: ":,.0f"} if ticker_col in plot_df.columns else None,
    )
    fig.update_traces(
        textinfo="label+percent entry",
        hovertemplate="%{label}<br>%{value:,.0f} €<br>%{percentEntry:.1%}<extra></extra>",
    )
    apply_chart_theme(
        fig,
        height=CHART_HEIGHT["treemap"],
        title=title,
        legend_horizontal=False,
    )
    return fig


def build_portfolio_allocation_pie_figure(
    df,
    *,
    value_col="Valeur Actuelle (€)",
    label_col="Nom",
    ticker_col="Ticker",
    title="Répartition du capital (€)",
):
    """Donut de répartition (alternative au treemap)."""
    import plotly.express as px
    from chart_theme import CHART_HEIGHT, apply_chart_theme

    if df is None or df.empty or value_col not in df.columns:
        return None
    names = (
        df[label_col].astype(str)
        if label_col in df.columns
        else df.get(ticker_col, df.index).astype(str)
    )
    fig = px.pie(
        df,
        values=value_col,
        names=names,
        hole=0.45,
        title=title,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    apply_chart_theme(
        fig,
        height=CHART_HEIGHT["treemap"],
        title=title,
        legend_horizontal=True,
    )
    return fig


def build_risk_return_scatter_figure(
    returns,
    *,
    weights=None,
    column_labels=None,
    title="Rendement vs volatilité (annualisés)",
):
    """Nuage rendement × volatilité — couleur = Sharpe, taille = poids optionnel."""
    from chart_theme import CHART_HEIGHT, apply_chart_theme

    if returns is None or returns.empty:
        return None
    column_labels = column_labels or {}
    mu = returns.mean() * TRADING_DAYS_PER_YEAR
    vol = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = mu / (vol + 1e-8)

    tickers = list(returns.columns)
    names = [str(column_labels.get(t, t)) for t in tickers]
    if weights is not None:
        w = pd.Series(weights).reindex(tickers).fillna(0).astype(float)
        w_max = float(w.max()) or 1.0
        sizes = (12 + 28 * (w / w_max)).tolist()
    else:
        sizes = [14] * len(tickers)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=vol,
            y=mu,
            mode="markers",
            marker=dict(
                size=sizes,
                color=sharpe,
                colorscale="RdYlGn",
                cmin=-0.5,
                cmax=2.0,
                colorbar=dict(title="Sharpe"),
                line=dict(width=0.5, color="#334155"),
            ),
            customdata=np.column_stack([sharpe.values, names]),
            hovertemplate=(
                "%{customdata[1]}<br>Rendement: %{y:.2%}<br>Volatilité: %{x:.2%}"
                "<br>Sharpe: %{customdata[0]:.2f}<extra></extra>"
            ),
        )
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#94a3b8", line_width=1)
    fig.update_layout(
        xaxis_title="Volatilité (annualisée)",
        yaxis_title="Rendement (annualisé)",
    )
    fig.update_xaxes(tickformat=".1%")
    fig.update_yaxes(tickformat=".1%")
    apply_chart_theme(fig, height=CHART_HEIGHT["scatter"], title=title)
    return fig


def build_returns_distribution_var_figure(
    returns_series,
    *,
    var_quantile=0.05,
    title="Distribution des rendements journaliers",
    asset_label="Actif",
):
    """Histogramme + KDE approximative et ligne VaR."""
    from chart_theme import CHART_COLORS, CHART_HEIGHT, apply_chart_theme

    r = pd.to_numeric(returns_series, errors="coerce").dropna()
    if r.empty:
        return None
    r_pct = r * 100.0
    var_val = float(r.quantile(var_quantile))
    var_pct = var_val * 100.0

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=r_pct,
            nbinsx=min(60, max(20, len(r) // 5)),
            name="Fréquence",
            marker_color="rgba(37, 99, 235, 0.65)",
            marker_line=dict(color="white", width=0.5),
            histnorm="probability density",
        )
    )
    fig.add_vline(
        x=var_pct,
        line_dash="dash",
        line_color=CHART_COLORS["negative"],
        line_width=2,
        annotation_text=f"VaR {int(var_quantile * 100)} % ({var_pct:.2f} %)",
        annotation_position="top left",
    )
    fig.update_layout(
        xaxis_title="Rendement journalier (%)",
        yaxis_title="Densité",
        bargap=0.04,
    )
    apply_chart_theme(
        fig,
        height=CHART_HEIGHT["pedagogy"],
        title=f"{title} — {asset_label}",
        legend_horizontal=False,
    )
    return fig


def build_suggestions_tradeoff_scatter(
    plot_df,
    *,
    x_col="Corr. moy. portefeuille",
    y_col="Rendement candidat",
    score_col="Score composite",
    label_col="Ticker",
    title="Rendement vs diversification (corrélation au portefeuille)",
):
    """Nuage suggestions — couleur/taille = score, détails au survol."""
    from chart_theme import CHART_HEIGHT, apply_chart_theme

    if plot_df is None or plot_df.empty:
        return None
    needed = [x_col, y_col, score_col, label_col]
    if any(c not in plot_df.columns for c in needed):
        return None
    plot = plot_df.dropna(subset=[x_col, y_col, score_col]).copy()
    if plot.empty:
        return None

    score_min = float(plot[score_col].min())
    sizes = np.clip(plot[score_col].astype(float) - score_min + 8.0, 8.0, 28.0)
    labels = plot[label_col].astype(str)
    tickers = plot["Ticker"].astype(str) if "Ticker" in plot.columns else labels

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot[x_col],
            y=plot[y_col],
            mode="markers",
            marker=dict(
                size=sizes,
                color=plot[score_col],
                colorscale="RdYlGn",
                colorbar=dict(title="Score"),
                line=dict(width=0.5, color="#334155"),
            ),
            customdata=np.column_stack([labels, tickers, plot[score_col]]),
            hovertemplate=(
                "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                "Corrélation : %{x:.2f}<br>Rendement : %{y:.2%}<br>"
                "Score : %{customdata[2]:.3f}<extra></extra>"
            ),
        )
    )
    fig.update_xaxes(title="Corrélation moy. portefeuille")
    fig.update_yaxes(title="Rendement candidat", tickformat=".1%")
    apply_chart_theme(fig, height=CHART_HEIGHT.get("scatter", 480), title=title)
    return fig


def build_equal_weight_portfolio_price(prices, tickers=None):
    """Indice équipondéré rebasé à 100 (1re date commune) pour analyse globale."""
    cols = [t for t in (tickers or prices.columns) if t in prices.columns]
    if not cols:
        return pd.Series(dtype=float)
    sub = prices[cols].dropna(how="all")
    if sub.empty or len(sub) < 2:
        return pd.Series(dtype=float)
    first_valid = sub.apply(lambda col: col.first_valid_index())
    start = first_valid.max()
    if start is None:
        return pd.Series(dtype=float)
    sub = sub.loc[start:].ffill().bfill()
    if sub.empty or sub.iloc[0].isna().any():
        return pd.Series(dtype=float)
    normalized = sub.div(sub.iloc[0]) * 100.0
    return normalized.mean(axis=1).dropna()


def build_portfolio_price_for_fft(
    prices,
    tickers=None,
    quantities=None,
    usd_to_eur=1.0,
    is_usd_fn=None,
):
    """
    Série agrégée pour FFT portefeuille :
    - valorisation pondérée par quantités si disponibles ;
    - sinon indice équipondéré rebasé à 100.
    """
    is_usd_fn = is_usd_fn or (lambda _t: False)
    tickers = tickers or list(prices.columns)
    if quantities:
        q = {
            str(t).strip(): float(quantities[t])
            for t in quantities
            if str(t).strip() in prices.columns and float(quantities[t] or 0) > 0
        }
        if q:
            return build_portfolio_value(prices, q, usd_to_eur=usd_to_eur, is_usd_fn=is_usd_fn)
    return build_equal_weight_portfolio_price(prices, tickers=tickers)


def compute_portfolio_risk_metrics(portfolio_value):
    s = portfolio_value.dropna()
    if len(s) < 2:
        return None
    r = s.pct_change().dropna()
    if r.empty:
        return None
    annual_return = r.mean() * 252
    annual_vol = r.std() * np.sqrt(252)
    sharpe = annual_return / (annual_vol + 1e-8)
    median_annual = float(r.median() * 252)
    mode_d = estimate_mode(r)
    mode_annual = float(mode_d * 252) if not np.isnan(mode_d) else np.nan
    return pd.Series(
        {
            "Rendement Annuel (moyenne)": annual_return,
            "Médiane annuelle": median_annual,
            "Mode annuelle": mode_annual,
            "Volatilité (Sigma)": annual_vol,
            "Ratio de Sharpe": sharpe,
            "Skewness (Asymétrie)": r.skew(),
            "Kurtosis (Aplatissement)": r.kurtosis(),
        }
    )


MAX_DAILY_RETURN = 0.50


def sanitize_return_series(series, max_daily=MAX_DAILY_RETURN):
    r = series.dropna().copy()
    return r[r.abs() <= max_daily]


def metrics_from_returns(returns_series):
    r = sanitize_return_series(returns_series)
    if len(r) < 30:
        return None
    annual_return = float(r.mean() * 252)
    annual_vol = float(r.std() * np.sqrt(252))
    return {
        "Rendement Annuel": annual_return,
        "Volatilité (Sigma)": annual_vol,
        "Ratio de Sharpe": annual_return / (annual_vol + 1e-8),
        "Skewness (Asymétrie)": float(r.skew()),
        "Kurtosis (Aplatissement)": float(r.kurtosis()),
    }


def compute_equal_weight_portfolio_returns(returns, tickers=None):
    cols = [t for t in (tickers or returns.columns) if t in returns.columns]
    if not cols:
        return pd.Series(dtype=float)
    return returns[cols].mean(axis=1).dropna()


def compute_avg_internal_correlation(returns, tickers=None):
    cols = [t for t in (tickers or returns.columns) if t in returns.columns]
    if len(cols) < 2:
        return np.nan
    corr = returns[cols].corr()
    values = corr.values[np.triu_indices(len(cols), k=1)]
    return float(np.nanmean(values))


def compute_avg_pairwise_corr(candidate_series, returns, tickers):
    cols = [t for t in tickers if t in returns.columns]
    corrs = []
    for ticker in cols:
        pair = pd.concat([candidate_series, returns[ticker]], axis=1, join="inner").dropna()
        if len(pair) >= 20:
            corrs.append(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
    return float(np.nanmean(corrs)) if corrs else np.nan


def blend_portfolio_returns(base_returns, candidate_returns, candidate_weight):
    combined = pd.concat([base_returns, candidate_returns], axis=1, join="inner").dropna()
    if combined.empty:
        return pd.Series(dtype=float)
    alpha = candidate_weight
    return (1 - alpha) * combined.iloc[:, 0] + alpha * combined.iloc[:, 1]


_PROFILE_PHRASES = {
    "Profil neutre": (
        "Peu de critères sont activés — le score différenciera peu les candidats."
    ),
    "Profil équilibré": (
        "Rendement, stabilité et diversification sont pondérés de façon similaire."
    ),
    "Profil diversificateur": (
        "Priorité à la diversification : titres qui ne réagissent pas comme votre portefeuille."
    ),
    "Profil très prudent": (
        "Stabilité et limitation des chutes brutales avant tout — rendement secondaire."
    ),
    "Profil prudent": (
        "Vous favorisez la stabilité et limitez les fortes variations."
    ),
    "Profil agressif": (
        "Rendement maximisé : la volatilité compte peu dans la sélection."
    ),
    "Profil orienté rendement": (
        "Vous acceptez davantage de risque pour viser de meilleurs gains."
    ),
    "Profil opportuniste": (
        "Vous visez des titres avec un fort potentiel de très belles hausses (skewness)."
    ),
    "Profil défensif": (
        "Calme et protection recherchés ; gains modestes assumés."
    ),
    "Profil dynamique": (
        "Préférence pour le rendement et les opportunités à la hausse."
    ),
    "Profil conservateur": (
        "Préférence pour la stabilité et la limitation des risques."
    ),
    "Profil mixte": (
        "Combinaison atypique — affinez les curseurs pour clarifier vos priorités."
    ),
}

_PORTFOLIO_PROFILE_PHRASES = {
    "Profil neutre": (
        "Données insuffisantes pour caractériser le portefeuille sur la période."
    ),
    "Profil équilibré": (
        "Rendement, volatilité et corrélation interne restent modérés — profil équilibré."
    ),
    "Profil diversificateur": (
        "Vos lignes bougent relativement peu ensemble : le risque est bien réparti entre les titres."
    ),
    "Profil très prudent": (
        "Portefeuille très stable, peu de variations et peu de queues épaisses sur la période."
    ),
    "Profil prudent": (
        "Volatilité contenue — comportement plutôt calme au jour le jour."
    ),
    "Profil agressif": (
        "Rendement élevé avec volatilité encore acceptable — profil offensif."
    ),
    "Profil orienté rendement": (
        "Performance annualisée solide — le portefeuille a bien progressé sur la période."
    ),
    "Profil opportuniste": (
        "Asymétrie positive : le portefeuille a connu des journées de forte hausse."
    ),
    "Profil défensif": (
        "Rendement modeste mais risque contenu — profil défensif observé."
    ),
    "Profil dynamique": (
        "Bon compromis rendement / risque avec une légère asymétrie haussière."
    ),
    "Profil conservateur": (
        "Priorité implicite à la préservation du capital — faible volatilité observée."
    ),
    "Profil mixte": (
        "Profil atypique — plusieurs dimensions se compensent (voir le détail des scores)."
    ),
}

# Paires redondantes à ne pas afficher ensemble.
_INCOMPATIBLE_DUAL = {
    frozenset({"Profil dynamique", "Profil orienté rendement"}),
    frozenset({"Profil agressif", "Profil orienté rendement"}),
    frozenset({"Profil conservateur", "Profil prudent"}),
    frozenset({"Profil conservateur", "Profil très prudent"}),
    frozenset({"Profil prudent", "Profil très prudent"}),
    frozenset({"Profil défensif", "Profil prudent"}),
    frozenset({"Profil défensif", "Profil très prudent"}),
}


def _profiles_compatible(existing, candidate):
    for profile in existing:
        if frozenset({profile, candidate}) in _INCOMPATIBLE_DUAL:
            return False
    return True


def _clamp01(value):
    return float(max(0.0, min(1.0, value)))


def _clamp02(value):
    return float(max(0.0, min(2.0, value)))


def metrics_to_pseudo_profile_weights(metrics, internal_corr=None):
    """
    Traduit les métriques observées du portefeuille en pseudo-poids 0–2
    (même échelle que les curseurs « Suggestions d'actifs »).
    """
    ret = float(
        metrics.get("Rendement Annuel (moyenne)", metrics.get("Rendement Annuel", 0)) or 0
    )
    vol = float(metrics.get("Volatilité (Sigma)", 0.15) or 0.15)
    skew = float(metrics.get("Skewness (Asymétrie)", 0) or 0)
    kurt = float(metrics.get("Kurtosis (Aplatissement)", 0) or 0)

    # Repères : rendement annuel −2 % → 0, +8 % → 1, +18 % → 2
    w_return = _clamp02((ret + 0.02) / 0.10)
    # Volatilité 22 % → 0 (peu stable), 14 % → 1, 6 % → 2 (très stable)
    w_vol = _clamp02((0.22 - vol) / 0.08)
    # Kurtosis excès faible → 2 (peu de chutes extrêmes), élevé → 0
    w_kurt = _clamp02((2.5 - kurt) / 1.25)
    # Skewness −0,25 → 0, +0,5 → 1, +1,25 → 2
    w_skew = _clamp02((skew + 0.25) / 0.75)

    if ret >= 0.07:
        # Fort rendement : évite un libellé « conservateur » uniquement parce que la vol est faible
        w_vol *= _clamp01(1 - (ret - 0.07) / 0.12)
    if vol >= 0.18:
        w_return *= 0.75

    if internal_corr is None or (isinstance(internal_corr, float) and np.isnan(internal_corr)):
        w_corr_int = 1.0
    else:
        # Corrélation interne 0,90 → 0, 0,55 → 1, 0,20 → 2
        w_corr_int = _clamp02((0.90 - float(internal_corr)) / 0.35)

    return {
        "return": w_return,
        "vol": w_vol,
        "kurt": w_kurt,
        "skew": w_skew,
        "corr_internal": w_corr_int,
        "corr_candidate": w_corr_int,
    }


def _pick_profile_from_scores(scores, phrase_map, dual_ratio=0.80, min_secondary=0.26):
    if scores.get("Profil neutre") == 1.0:
        return "Profil neutre", phrase_map["Profil neutre"]

    ranked = sorted(scores.items(), key=lambda item: -item[1])
    top_name, top_score = ranked[0]

    if top_score < 0.12:
        return "Profil mixte", phrase_map["Profil mixte"]

    picked = [top_name]
    for name, score in ranked[1:]:
        if score < dual_ratio * top_score or score < min_secondary:
            break
        if name in picked:
            continue
        if _profiles_compatible(picked, name):
            picked.append(name)
            break

    title = " · ".join(picked)
    description = " · ".join(phrase_map[name] for name in picked)
    return title, description


def compute_profile_scores(weights):
    """
    Score 0–1 par profil pour les curseurs « Poids des objectifs ».
    Plusieurs profils peuvent avoir un score élevé (profil composite possible).
    """
    ret = float(weights.get("return", 1.0))
    vol = float(weights.get("vol", 1.0))
    kurt = float(weights.get("kurt", 1.0))
    skew = float(weights.get("skew", 1.0))
    corr_i = float(weights.get("corr_internal", 1.0))
    corr_c = float(weights.get("corr_candidate", 1.0))

    all_w = [ret, vol, kurt, skew, corr_i, corr_c]
    if max(all_w) < 0.35:
        return {"Profil neutre": 1.0}

    safety = (vol + kurt) / 2
    diversify = (corr_i + corr_c) / 2
    performance = (ret + skew) / 2
    tilt = (ret + skew * 0.7) - (vol + kurt * 0.7)
    spread = max(all_w) - min(all_w)
    avg = sum(all_w) / len(all_w)

    in_band = sum(0.55 <= w <= 1.45 for w in all_w) / len(all_w)
    avg_closeness = _clamp01(1 - abs(avg - 1) / 0.55)
    spread_closeness = _clamp01(1 - spread / 1.6)

    scores = {
        "Profil équilibré": _clamp01(
            0.35 * in_band + 0.35 * avg_closeness + 0.3 * spread_closeness
        ),
        "Profil diversificateur": _clamp01(
            0.55 * _clamp01((diversify - 0.35) / 1.65)
            + 0.45 * _clamp01((diversify - max(safety, performance) + 0.35) / 0.85)
        ),
        "Profil très prudent": _clamp01(
            _clamp01((safety - 0.55) / 1.45)
            * (0.45 + 0.55 * _clamp01((0.9 - performance) / 0.9))
        )
        if performance <= 0.9
        else 0.0,
        "Profil prudent": _clamp01(
            _clamp01((safety - 0.45) / 1.55)
            * _clamp01((safety - performance + 0.25) / 1.1)
        ),
        "Profil agressif": _clamp01(
            _clamp01((ret - 0.65) / 1.35)
            * (0.45 + 0.55 * _clamp01((1.05 - vol) / 1.05))
        ),
        "Profil orienté rendement": _clamp01(
            0.5 * _clamp01((performance - 0.45) / 1.55)
            + 0.5 * _clamp01((performance - safety + 0.35) / 0.95)
        ),
        "Profil opportuniste": _clamp01(
            0.5 * _clamp01((skew - 0.45) / 1.55)
            + 0.5 * _clamp01((skew - max(vol, kurt) * 0.88 + 0.25) / 0.75)
        ),
        "Profil défensif": _clamp01(
            _clamp01((safety - 0.5) / 1.5) * _clamp01((0.95 - performance) / 0.75)
        )
        if performance <= 1.0
        else 0.0,
        "Profil dynamique": _clamp01((tilt + 0.12) / 0.88),
        "Profil conservateur": _clamp01((-tilt + 0.12) / 0.88),
    }

    # Réduit les chevauchements : profils « légers » affaiblis si un voisin domine fortement.
    ranked = sorted(scores.items(), key=lambda item: -item[1])
    if ranked[0][1] > 0.55:
        dominant = ranked[0][0]
        light = {"Profil dynamique", "Profil conservateur", "Profil défensif"}
        if dominant not in light:
            for name in light:
                scores[name] *= 0.65

    return scores


def describe_suggestion_profile(weights, dual_ratio=0.80, min_secondary=0.26):
    """
    Résume les curseurs en un libellé (1 ou 2 adjectifs) + phrase explicative.

    dual_ratio : le 2e profil apparaît s'il atteint ce ratio du score max.
    """
    scores = compute_profile_scores(weights)
    return _pick_profile_from_scores(
        scores, _PROFILE_PHRASES, dual_ratio=dual_ratio, min_secondary=min_secondary
    )


def describe_portfolio_profile(metrics, internal_corr=None, dual_ratio=0.80, min_secondary=0.26):
    """
    Caractérise le portefeuille réel (métriques + corrélation interne)
    avec les mêmes libellés que les suggestions d'actifs.
    """
    pseudo = metrics_to_pseudo_profile_weights(metrics, internal_corr=internal_corr)
    scores = compute_profile_scores(pseudo)
    title, description = _pick_profile_from_scores(
        scores, _PORTFOLIO_PROFILE_PHRASES, dual_ratio=dual_ratio, min_secondary=min_secondary
    )
    return title, description, pseudo, scores


def suggest_portfolio_additions(
    returns,
    portfolio_tickers,
    candidate_returns,
    candidate_weight=0.10,
    objective_weights=None,
):
    objective_weights = objective_weights or {
        "return": 1.0,
        "vol": 1.0,
        "kurt": 0.7,
        "skew": 0.8,
        "corr_internal": 1.0,
        "corr_candidate": 1.0,
    }

    port_returns = compute_equal_weight_portfolio_returns(returns, portfolio_tickers)
    baseline = metrics_from_returns(port_returns)
    baseline_internal = compute_avg_internal_correlation(returns, portfolio_tickers)
    if baseline is None:
        return pd.DataFrame(), baseline, baseline_internal

    rows = []
    for candidate in candidate_returns.columns:
        if candidate in portfolio_tickers:
            continue

        candidate_series = sanitize_return_series(candidate_returns[candidate])
        aligned = pd.concat([port_returns, candidate_series], axis=1, join="inner").dropna()
        if len(aligned) < 30:
            continue

        base_aligned = aligned.iloc[:, 0]
        cand_aligned = aligned.iloc[:, 1]
        blended = blend_portfolio_returns(base_aligned, cand_aligned, candidate_weight)
        blended_metrics = metrics_from_returns(blended)
        candidate_metrics = metrics_from_returns(cand_aligned)
        if blended_metrics is None or candidate_metrics is None:
            continue

        avg_corr = compute_avg_pairwise_corr(cand_aligned, returns, portfolio_tickers)
        extended_returns = returns.join(candidate_returns[[candidate]], how="inner")
        new_internal = compute_avg_internal_correlation(
            extended_returns, portfolio_tickers + [candidate]
        )

        delta_return = blended_metrics["Rendement Annuel"] - baseline["Rendement Annuel"]
        delta_vol = baseline["Volatilité (Sigma)"] - blended_metrics["Volatilité (Sigma)"]
        delta_kurt = baseline["Kurtosis (Aplatissement)"] - blended_metrics["Kurtosis (Aplatissement)"]
        delta_skew = blended_metrics["Skewness (Asymétrie)"] - baseline["Skewness (Asymétrie)"]
        delta_sharpe = blended_metrics["Ratio de Sharpe"] - baseline["Ratio de Sharpe"]
        delta_corr_internal = (
            baseline_internal - new_internal
            if not np.isnan(baseline_internal) and not np.isnan(new_internal)
            else 0.0
        )
        diversification_bonus = -avg_corr if not np.isnan(avg_corr) else 0.0

        score = (
            objective_weights["return"] * delta_return
            + objective_weights["vol"] * delta_vol
            + objective_weights["kurt"] * delta_kurt
            + objective_weights["skew"] * delta_skew
            + objective_weights["corr_internal"] * delta_corr_internal
            + objective_weights["corr_candidate"] * diversification_bonus
        )

        rows.append(
            {
                "Ticker": candidate,
                "Score composite": score,
                "Rendement candidat": candidate_metrics["Rendement Annuel"],
                "Volatilité candidat": candidate_metrics["Volatilité (Sigma)"],
                "Skewness candidat": candidate_metrics["Skewness (Asymétrie)"],
                "Kurtosis candidat": candidate_metrics["Kurtosis (Aplatissement)"],
                "Sharpe candidat": candidate_metrics["Ratio de Sharpe"],
                "Corr. moy. portefeuille": avg_corr,
                "Δ Rendement portef.": delta_return,
                "Δ Volatilité portef.": delta_vol,
                "Δ Kurtosis portef.": delta_kurt,
                "Δ Skewness portef.": delta_skew,
                "Δ Sharpe portef.": delta_sharpe,
                "Δ Corr. interne": delta_corr_internal,
            }
        )

    if not rows:
        return pd.DataFrame(), baseline, baseline_internal

    df = pd.DataFrame(rows)
    return df.sort_values("Score composite", ascending=False), baseline, baseline_internal


DEFAULT_SUGGESTION_OBJECTIVE_WEIGHTS = {
    "return": 1.0,
    "vol": 1.0,
    "kurt": 1.0,
    "skew": 1.0,
    "corr_internal": 1.0,
    "corr_candidate": 1.0,
}


def filter_suggestions_by_statistics(
    df,
    min_candidate_return=None,
    max_candidate_return=None,
    min_candidate_vol=None,
    max_candidate_vol=None,
    min_candidate_kurtosis=None,
    max_candidate_kurtosis=None,
    min_candidate_skewness=None,
    max_candidate_skewness=None,
    min_candidate_sharpe=None,
    max_candidate_sharpe=None,
    min_corr_portfolio=None,
    max_corr_portfolio=None,
    min_delta_return=None,
    max_delta_return=None,
    min_delta_vol=None,
    max_delta_vol=None,
    min_delta_kurtosis=None,
    max_delta_kurtosis=None,
    min_delta_skewness=None,
    max_delta_skewness=None,
    min_delta_sharpe=None,
    max_delta_sharpe=None,
    min_delta_corr_internal=None,
    max_delta_corr_internal=None,
):
    """Filtre les suggestions selon des plages risque / rendement / diversification."""
    if df is None or df.empty:
        return df
    out = df.copy()

    if min_candidate_return is not None and "Rendement candidat" in out.columns:
        out = out[out["Rendement candidat"].fillna(-1) >= min_candidate_return]
    if max_candidate_return is not None and "Rendement candidat" in out.columns:
        out = out[
            out["Rendement candidat"].isna()
            | (out["Rendement candidat"] <= max_candidate_return)
        ]
    if min_candidate_vol is not None and "Volatilité candidat" in out.columns:
        out = out[
            out["Volatilité candidat"].isna()
            | (out["Volatilité candidat"] >= min_candidate_vol)
        ]
    if max_candidate_vol is not None and "Volatilité candidat" in out.columns:
        out = out[
            out["Volatilité candidat"].isna()
            | (out["Volatilité candidat"] <= max_candidate_vol)
        ]
    if min_candidate_kurtosis is not None and "Kurtosis candidat" in out.columns:
        out = out[
            out["Kurtosis candidat"].isna()
            | (out["Kurtosis candidat"] >= min_candidate_kurtosis)
        ]
    if max_candidate_kurtosis is not None and "Kurtosis candidat" in out.columns:
        out = out[
            out["Kurtosis candidat"].isna()
            | (out["Kurtosis candidat"] <= max_candidate_kurtosis)
        ]
    if min_candidate_skewness is not None and "Skewness candidat" in out.columns:
        out = out[out["Skewness candidat"].fillna(-1) >= min_candidate_skewness]
    if max_candidate_skewness is not None and "Skewness candidat" in out.columns:
        out = out[
            out["Skewness candidat"].isna()
            | (out["Skewness candidat"] <= max_candidate_skewness)
        ]
    if min_candidate_sharpe is not None and "Sharpe candidat" in out.columns:
        out = out[out["Sharpe candidat"].fillna(-1) >= min_candidate_sharpe]
    if max_candidate_sharpe is not None and "Sharpe candidat" in out.columns:
        out = out[
            out["Sharpe candidat"].isna()
            | (out["Sharpe candidat"] <= max_candidate_sharpe)
        ]
    if min_corr_portfolio is not None and "Corr. moy. portefeuille" in out.columns:
        out = out[
            out["Corr. moy. portefeuille"].isna()
            | (out["Corr. moy. portefeuille"] >= min_corr_portfolio)
        ]
    if max_corr_portfolio is not None and "Corr. moy. portefeuille" in out.columns:
        out = out[
            out["Corr. moy. portefeuille"].isna()
            | (out["Corr. moy. portefeuille"] <= max_corr_portfolio)
        ]
    if min_delta_return is not None and "Δ Rendement portef." in out.columns:
        out = out[out["Δ Rendement portef."].fillna(-1) >= min_delta_return]
    if max_delta_return is not None and "Δ Rendement portef." in out.columns:
        out = out[
            out["Δ Rendement portef."].isna()
            | (out["Δ Rendement portef."] <= max_delta_return)
        ]
    if min_delta_vol is not None and "Δ Volatilité portef." in out.columns:
        out = out[out["Δ Volatilité portef."].fillna(-1) >= min_delta_vol]
    if max_delta_vol is not None and "Δ Volatilité portef." in out.columns:
        out = out[
            out["Δ Volatilité portef."].isna()
            | (out["Δ Volatilité portef."] <= max_delta_vol)
        ]
    if min_delta_kurtosis is not None and "Δ Kurtosis portef." in out.columns:
        out = out[out["Δ Kurtosis portef."].fillna(-1) >= min_delta_kurtosis]
    if max_delta_kurtosis is not None and "Δ Kurtosis portef." in out.columns:
        out = out[
            out["Δ Kurtosis portef."].isna()
            | (out["Δ Kurtosis portef."] <= max_delta_kurtosis)
        ]
    if min_delta_skewness is not None and "Δ Skewness portef." in out.columns:
        out = out[out["Δ Skewness portef."].fillna(-1) >= min_delta_skewness]
    if max_delta_skewness is not None and "Δ Skewness portef." in out.columns:
        out = out[
            out["Δ Skewness portef."].isna()
            | (out["Δ Skewness portef."] <= max_delta_skewness)
        ]
    if min_delta_sharpe is not None and "Δ Sharpe portef." in out.columns:
        out = out[out["Δ Sharpe portef."].fillna(-1) >= min_delta_sharpe]
    if max_delta_sharpe is not None and "Δ Sharpe portef." in out.columns:
        out = out[
            out["Δ Sharpe portef."].isna()
            | (out["Δ Sharpe portef."] <= max_delta_sharpe)
        ]
    if min_delta_corr_internal is not None and "Δ Corr. interne" in out.columns:
        out = out[out["Δ Corr. interne"].fillna(-1) >= min_delta_corr_internal]
    if max_delta_corr_internal is not None and "Δ Corr. interne" in out.columns:
        out = out[
            out["Δ Corr. interne"].isna()
            | (out["Δ Corr. interne"] <= max_delta_corr_internal)
        ]

    return out


_STYLE_FAVORABLE = "background-color: #d4edda; color: #155724"
_STYLE_UNFAVORABLE = "background-color: #f8d7da; color: #721c24"

SUGGESTION_TECHNICAL_COLUMNS = (
    "RSI",
    "Signal RSI",
    "Bollinger %B",
    "Stoch %K",
    "Signal MM",
)


def merge_technical_columns(df, tech_df, columns=None):
    """Joint les indicateurs techniques au tableau de suggestions (clé Ticker)."""
    if df is None or df.empty or tech_df is None or tech_df.empty:
        return df
    columns = columns or SUGGESTION_TECHNICAL_COLUMNS
    keep = ["Ticker"] + [c for c in columns if c in tech_df.columns]
    tech_df = tech_df[keep].drop_duplicates(subset=["Ticker"], keep="last")
    return df.merge(tech_df, on="Ticker", how="left")


def enrich_suggestions_with_technical(df, prices, rsi_period=14, columns=None):
    """Calcule RSI, Bollinger %B, etc. à partir des cours candidats."""
    if df is None or df.empty or prices is None or prices.empty:
        return df
    tickers = [t for t in df["Ticker"] if t in prices.columns]
    if not tickers:
        return df
    tech_df = compute_technical_indicators(prices[tickers], rsi_period=rsi_period)
    return merge_technical_columns(df, tech_df, columns=columns)


def filter_suggestions_by_technical(
    df,
    max_rsi=None,
    min_rsi=None,
    max_bollinger_b=None,
    min_bollinger_b=None,
):
    """Filtre les suggestions selon des seuils techniques (RSI, Bollinger %B)."""
    if df is None or df.empty:
        return df
    out = df.copy()
    if max_rsi is not None and "RSI" in out.columns:
        out = out[out["RSI"].isna() | (out["RSI"] <= max_rsi)]
    if min_rsi is not None and "RSI" in out.columns:
        out = out[out["RSI"].isna() | (out["RSI"] >= min_rsi)]
    if max_bollinger_b is not None and "Bollinger %B" in out.columns:
        out = out[
            out["Bollinger %B"].isna()
            | (out["Bollinger %B"] <= max_bollinger_b)
        ]
    if min_bollinger_b is not None and "Bollinger %B" in out.columns:
        out = out[
            out["Bollinger %B"].isna()
            | (out["Bollinger %B"] >= min_bollinger_b)
        ]
    return out


def _technical_metric_sentiment(column, val):
    """Vert = zone basse (potentiel rebond) · rouge = zone haute (surachat)."""
    v = _safe_float_technical(val)
    if v is None:
        return ""
    col = str(column)
    if col in ("RSI", "RSI (0–100)"):
        if v <= 35:
            return _STYLE_FAVORABLE
        if v >= 65:
            return _STYLE_UNFAVORABLE
    elif col in ("Bollinger %B", "Bollinger %B (0–1)"):
        if v <= 0.2:
            return _STYLE_FAVORABLE
        if v >= 0.8:
            return _STYLE_UNFAVORABLE
    elif col in ("Stoch %K", "Stochastique %K (0–100)"):
        if v <= 25:
            return _STYLE_FAVORABLE
        if v >= 75:
            return _STYLE_UNFAVORABLE
    elif col in ("MFI", "MFI (0–100)"):
        if v <= 20:
            return _STYLE_FAVORABLE
        if v >= 80:
            return _STYLE_UNFAVORABLE
    return ""


_TECHNICAL_NUMERIC_COLUMNS = {
    "RSI",
    "RSI (0–100)",
    "Bollinger %B",
    "Bollinger %B (0–1)",
    "Stoch %K",
    "Stochastique %K (0–100)",
    "MFI",
    "MFI (0–100)",
}

_TECHNICAL_SIGNAL_COLUMNS = {
    "Signal RSI",
    "Signal Stoch",
    "Signal MFI",
    "Signal SuperTrend",
}


def _technical_signal_cell_style(val):
    """Fond vert = survente · fond rouge = surachat."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    text = str(val).strip()
    if text == "Surachat":
        return _STYLE_UNFAVORABLE
    if text == "Survente":
        return _STYLE_FAVORABLE
    if text == "Haussier":
        return _STYLE_FAVORABLE
    if text == "Baissier":
        return _STYLE_UNFAVORABLE
    return ""


def _apply_technical_column_styles(series, include_signals=True):
    name = series.name
    if include_signals and name in _TECHNICAL_SIGNAL_COLUMNS:
        return [_technical_signal_cell_style(v) for v in series]
    if name in _TECHNICAL_NUMERIC_COLUMNS:
        return [_technical_metric_sentiment(name, v) for v in series]
    return [""] * len(series)


def style_technical_table(styler):
    """Code couleur surachat/survente dans le tableau analyse technique."""
    return styler.apply(
        lambda s: _apply_technical_column_styles(s, include_signals=True), axis=0
    )


def _safe_float_technical(value):
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


def style_suggestions_technical(styler):
    """Code couleur RSI / Bollinger / stochastique dans le tableau suggestions."""
    return styler.apply(
        lambda s: _apply_technical_column_styles(s, include_signals=False), axis=0
    )
