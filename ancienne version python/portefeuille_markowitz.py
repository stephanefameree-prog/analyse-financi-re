import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# =========================
# 1. Paramètres du portefeuille
# =========================

tickers = [
    "HMY", "XPL", "NYXH.BR", "NUTX",
    "ALNTG.PA", "IPS.PA", "EDEN.PA",
    "ZAL.DE", "ETL.PA"
]

# Poids ACTUELS de ton portefeuille (en fraction, pas en %)
weights_actual = np.array([
    0.0688,  # HMY
    0.0996,  # XPL
    0.0516,  # NYXH
    0.0939,  # NUTX
    0.0531,  # ALNTG
    0.0674,  # IPS
    0.1631,  # EDEN
    0.0764,  # ZAL
    0.3260   # ETL
])

risk_free = 0.025  # taux sans risque annualisé (≈ 2,5%)

# =========================
# 2. Données de marché (6 derniers mois)
# =========================

print("Téléchargement des données (6 mois)...")
data = yf.download(tickers, period="6mo")["Close"]
returns = data.pct_change().dropna()

mean_returns = returns.mean() * 252          # rendement annualisé
cov_matrix = returns.cov() * 252             # covariance annualisée

print("\nRendements annualisés par actif :")
print(mean_returns)

# =========================
# 3. Fonctions de base
# =========================

def portfolio_stats(weights):
    """Retourne (rendement, volatilité, Sharpe) pour un vecteur de poids."""
    ret = np.dot(weights, mean_returns)
    vol = np.sqrt(weights.T @ cov_matrix @ weights)
    sharpe = (ret - risk_free) / vol
    return ret, vol, sharpe

# =========================
# 4. Sharpe du portefeuille ACTUEL
# =========================

ret_p, vol_p, sharpe_p = portfolio_stats(weights_actual)

print("\n=== Portefeuille ACTUEL ===")
print(f"Rendement annualisé : {ret_p:.4f}")
print(f"Volatilité annualisée : {vol_p:.4f}")
print(f"Sharpe : {sharpe_p:.4f}")

# =========================
# 5. Portefeuille max Sharpe (optimisation)
# =========================

num_assets = len(tickers)
bounds = tuple((0, 1) for _ in range(num_assets))
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

def neg_sharpe(weights):
    return -portfolio_stats(weights)[2]

print("\nOptimisation du portefeuille max Sharpe...")
opt_sharpe = minimize(
    neg_sharpe,
    x0=np.array(num_assets * [1/num_assets]),
    bounds=bounds,
    constraints=constraints
)

weights_max_sharpe = opt_sharpe.x
ret_ms, vol_ms, sharpe_ms = portfolio_stats(weights_max_sharpe)

print("\n=== Portefeuille MAX SHARPE ===")
print(f"Rendement annualisé : {ret_ms:.4f}")
print(f"Volatilité annualisée : {vol_ms:.4f}")
print(f"Sharpe : {sharpe_ms:.4f}")
print("\nPoids du portefeuille max Sharpe :")
for t, w in zip(tickers, weights_max_sharpe):
    print(f"{t:8s} : {w:.3f}")

# =========================
# 6. Frontière efficiente
# =========================

print("\nCalcul de la frontière efficiente...")

target_returns = np.linspace(mean_returns.min(), mean_returns.max(), 50)
frontier_vol = []

for tr in target_returns:
    cons = (
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'eq', 'fun': lambda w, tr=tr: np.dot(w, mean_returns) - tr}
    )
    res = minimize(
        lambda w: np.sqrt(w.T @ cov_matrix @ w),
        x0=np.array(num_assets * [1/num_assets]),
        bounds=bounds,
        constraints=cons
    )
    frontier_vol.append(res.fun)

plt.figure(figsize=(10, 6))
plt.plot(frontier_vol, target_returns, 'r--', label="Frontière efficiente")
plt.scatter(vol_p, ret_p, c='blue', marker='o', label='Portefeuille actuel')
plt.scatter(vol_ms, ret_ms, c='green', marker='*', s=200, label='Max Sharpe')
plt.xlabel("Volatilité annualisée")
plt.ylabel("Rendement annualisé")
plt.title("Frontière efficiente (6 mois)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# 7. Contribution au risque du portefeuille ACTUEL
# =========================

print("\n=== Contribution au risque (portefeuille ACTUEL) ===")

portfolio_vol = vol_p
marginal_contrib = cov_matrix @ weights_actual
risk_contrib = weights_actual * marginal_contrib / portfolio_vol

for t, rc, w in zip(tickers, risk_contrib, weights_actual):
    print(f"{t:8s} | poids = {w:5.3f} | contrib. risque = {rc:7.4f}")

print("\nTerminé.")
