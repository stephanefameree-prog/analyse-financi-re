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
        title=title,
        xaxis_title="Risque (écart-type annualisé)",
        yaxis_title="Rendement espéré (annualisé)",
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right"),
        hovermode="closest",
    )
    fig.update_xaxes(tickformat=".1%")
    fig.update_yaxes(tickformat=".1%")
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
        title="Où se placent la moyenne, la médiane et le mode ? (exemple pédagogique)",
        xaxis_title="Rendement journalier (%)",
        yaxis_title="Nombre de journées",
        bargap=0.04,
        height=400,
        margin=dict(t=60, b=50),
        showlegend=False,
    )
    return fig


TRADING_DAYS_PER_YEAR = 252

FFT_TREND_LOG_LINEAR = "log_linear"
FFT_TREND_LINEAR_PRICE = "linear_price"
FFT_TREND_HP = "hp"

FFT_TREND_OPTIONS = {
    FFT_TREND_LOG_LINEAR: "Log-linéaire (croissance % constante)",
    FFT_TREND_LINEAR_PRICE: "Linéaire sur le prix (€)",
    FFT_TREND_HP: "Filtre Hodrick-Prescott",
}

HP_LAMBDA_DAILY = 6.25e6


FFT_TREND_LEGEND = {
    FFT_TREND_LOG_LINEAR: "log-linéaire",
    FFT_TREND_LINEAR_PRICE: "linéaire €",
    FFT_TREND_HP: "Hodrick-Prescott",
}


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


def _extend_price_trend(trend_p, n_total, trend_mode):
    n = len(trend_p)
    t_hist = np.arange(n)
    t_ext = np.arange(n_total)
    trend_p = np.asarray(trend_p, dtype=float)
    if trend_mode == FFT_TREND_LOG_LINEAR:
        log_tr = np.log(trend_p)
        log_ext = np.polyval(np.polyfit(t_hist, log_tr, 1), t_ext)
        return np.exp(log_ext)
    return np.polyval(np.polyfit(t_hist, trend_p, 1), t_ext)


def _trend_fit_metrics(prices, trend_p):
    p = np.asarray(prices, dtype=float)
    trend_p = np.asarray(trend_p, dtype=float)
    ss_res = float(np.sum((p - trend_p) ** 2))
    ss_tot = float(np.sum((p - p.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.mean((p - trend_p) ** 2)))
    return r2, rmse


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
    Retire une tendance de fond (log-linéaire, linéaire € ou HP) avant FFT.
    Le signal cyclique analysé = log(prix) − log(tendance).
    """
    if trend_mode not in FFT_TREND_OPTIONS:
        trend_mode = FFT_TREND_LOG_LINEAR

    s = pd.Series(price_series).dropna().astype(float)
    if len(s) < 40:
        return None

    p = s.values
    t = np.arange(len(p))
    trend_p = _fit_price_trend(p, t, trend_mode)
    log_p = np.log(p)
    log_trend = np.log(trend_p)
    detrended = log_p - log_trend
    trend_r2, trend_rmse = _trend_fit_metrics(p, trend_p)

    return {
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
        "n_obs": len(s),
    }


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

    return {
        "dates": s.index,
        "prices": s,
        "trend_mode": prep["trend_mode"],
        "trend_prices": prep["trend_prices"],
        "trend_r2": prep["trend_r2"],
        "trend_rmse": prep["trend_rmse"],
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


FFT_FORECAST_RATIO = 0.30


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


def build_fft_extended_series(result, peak_indices, extend_ratio=FFT_FORECAST_RATIO):
    """Extrapole tendance log + harmoniques sur extend_ratio × la durée historique."""
    if not peak_indices or extend_ratio <= 0:
        return None

    n = int(result["n_obs"])
    n_extra = max(1, int(round(n * extend_ratio)))
    n_total = n + n_extra

    detrended = np.asarray(result["detrended_log"].values, dtype=float)
    windowed = detrended * np.hanning(n)
    fft_vals = np.fft.rfft(windowed)

    log_trend_hist = np.asarray(result["log_trend"].values, dtype=float)
    trend_mode = result.get("trend_mode", FFT_TREND_LOG_LINEAR)
    trend_p_hist = np.asarray(result["trend_prices"].values, dtype=float)
    trend_p_ext = _extend_price_trend(trend_p_hist, n_total, trend_mode)
    trend_p_ext = _positive_trend(trend_p_ext, result["prices"].values)
    log_trend_ext = np.log(trend_p_ext)
    cyclic_ext = _synthesize_rfft_peaks(fft_vals, peak_indices, n, n_total)
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
):
    """Prix, tendance de fond (€) et extrapolation du modèle FFT (+30 % par défaut)."""
    peaks = result["peaks"].head(n_components)
    indices = peaks["_freq_idx"].tolist()
    n_hist = result["n_obs"]
    trend_mode = result.get("trend_mode", FFT_TREND_LOG_LINEAR)
    trend_name = f"Tendance de fond ({fft_trend_legend_label(trend_mode)})"

    extended = build_fft_extended_series(result, indices, extend_ratio=extend_ratio)
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

    if price_model is not None:
        _add_fft_split_trace(
            fig,
            dates,
            price_model,
            n_hist,
            "Modèle FFT (tendance + cycles)",
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


def compute_technical_indicators(prices, rsi_period=14, volumes=None, highs=None, lows=None):
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
    return {
        "Rendement Annuel": float(r.mean() * 252),
        "Volatilité (Sigma)": float(r.std() * np.sqrt(252)),
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
                "Corr. moy. portefeuille": avg_corr,
                "Δ Rendement portef.": delta_return,
                "Δ Volatilité portef.": delta_vol,
                "Δ Kurtosis portef.": delta_kurt,
                "Δ Skewness portef.": delta_skew,
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
    max_candidate_vol=None,
    max_candidate_kurtosis=None,
    min_candidate_skewness=None,
    max_corr_portfolio=None,
    min_delta_return=None,
    min_delta_vol=None,
    min_delta_kurtosis=None,
    min_delta_skewness=None,
    min_delta_corr_internal=None,
):
    """Filtre les suggestions selon des seuils risque / rendement / diversification."""
    if df is None or df.empty:
        return df
    out = df.copy()

    if min_candidate_return is not None and "Rendement candidat" in out.columns:
        out = out[out["Rendement candidat"].fillna(-1) >= min_candidate_return]
    if max_candidate_vol is not None and "Volatilité candidat" in out.columns:
        out = out[
            out["Volatilité candidat"].isna()
            | (out["Volatilité candidat"] <= max_candidate_vol)
        ]
    if max_candidate_kurtosis is not None and "Kurtosis candidat" in out.columns:
        out = out[
            out["Kurtosis candidat"].isna()
            | (out["Kurtosis candidat"] <= max_candidate_kurtosis)
        ]
    if min_candidate_skewness is not None and "Skewness candidat" in out.columns:
        out = out[out["Skewness candidat"].fillna(-1) >= min_candidate_skewness]
    if max_corr_portfolio is not None and "Corr. moy. portefeuille" in out.columns:
        out = out[
            out["Corr. moy. portefeuille"].isna()
            | (out["Corr. moy. portefeuille"] <= max_corr_portfolio)
        ]
    if min_delta_return is not None and "Δ Rendement portef." in out.columns:
        out = out[out["Δ Rendement portef."].fillna(-1) >= min_delta_return]
    if min_delta_vol is not None and "Δ Volatilité portef." in out.columns:
        out = out[out["Δ Volatilité portef."].fillna(-1) >= min_delta_vol]
    if min_delta_kurtosis is not None and "Δ Kurtosis portef." in out.columns:
        out = out[out["Δ Kurtosis portef."].fillna(-1) >= min_delta_kurtosis]
    if min_delta_skewness is not None and "Δ Skewness portef." in out.columns:
        out = out[out["Δ Skewness portef."].fillna(-1) >= min_delta_skewness]
    if min_delta_corr_internal is not None and "Δ Corr. interne" in out.columns:
        out = out[out["Δ Corr. interne"].fillna(-1) >= min_delta_corr_internal]

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
    return ""


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
    tech_cols = {
        "RSI",
        "RSI (0–100)",
        "Bollinger %B",
        "Bollinger %B (0–1)",
        "Stoch %K",
        "Stochastique %K (0–100)",
    }

    def _color_column(series):
        if series.name not in tech_cols:
            return [""] * len(series)
        return [_technical_metric_sentiment(series.name, v) for v in series]

    return styler.apply(_color_column, axis=0)
