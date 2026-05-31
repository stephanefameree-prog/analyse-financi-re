# ============================================================
#  ANALYSE DE PORTEFEUILLE — VERSION COMPLÈTE
#  YahooQuery + Heatmaps + Clustering + Frontière efficiente
# ============================================================

from yahooquery import Ticker
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.cluster.hierarchy import linkage, leaves_list

# ------------------------------------------------------------
# 1. LISTE DES TICKERS
# ------------------------------------------------------------
tickers = [
    "HMY", "XPL", "NYXH.BR", "NUTX",
    "ALNTG.PA", "IPS.PA", "EDEN.PA",
    "ZAL.DE", "ETL.PA",
    "GTT.PA", "CAP.PA", "RMS.PA",
    "GBLB.BR", "GIMB.BR", "AI.PA",
    "CA.PA", "TEP.PA",
    "FRVIA.PA", "BOI.PA", "LTA.PA",
    "TRI.PA"
]

start = "2022-01-01"
end = datetime.today().strftime("%Y-%m-%d")

# ------------------------------------------------------------
# 2. TÉLÉCHARGEMENT DES DONNÉES
# ------------------------------------------------------------
print("=== Téléchargement des données ===")

data = {}
for t in tickers:
    try:
        df = Ticker(t).history(start=start, end=end)
        if isinstance(df, pd.DataFrame) and "close" in df.columns:
            s = df["close"].droplevel(0)
            if len(s) > 0:
                print(f"✔ {t} OK")
                data[t] = s
            else:
                print(f"✖ {t} vide")
        else:
            print(f"✖ {t} pas de données")
    except Exception as e:
        print(f"✖ {t} erreur : {e}")

if len(data) == 0:
    print("❌ Aucune donnée exploitable.")
    exit()

prices = pd.DataFrame(data).dropna()
returns = prices.pct_change().dropna()

# ------------------------------------------------------------
# 3. HEATMAP CLASSIQUE (PALETTE PRO)
# ------------------------------------------------------------
plt.figure(figsize=(14, 10))
sns.heatmap(
    returns.corr(),
    cmap="RdBu_r",
    annot=True,
    fmt=".2f",
    center=0,
    linewidths=0.5
)
plt.title("Heatmap de corrélation (palette pro)", fontsize=16)
plt.show()

# ------------------------------------------------------------
# 4. HEATMAP TRIANGULAIRE + CLUSTERING HIÉRARCHIQUE (STYLE BLOOMBERG)
# ------------------------------------------------------------
corr = returns.corr()

# Clustering hiérarchique
link = linkage(corr, method='ward')
idx = leaves_list(link)

# Réorganisation selon les clusters
corr_clustered = corr.iloc[idx, :].iloc[:, idx]

# Masque triangulaire
mask = np.triu(np.ones_like(corr_clustered, dtype=bool))

plt.figure(figsize=(14, 10))
sns.heatmap(
    corr_clustered,
    mask=mask,
    cmap="RdBu_r",
    annot=True,
    fmt=".2f",
    center=0,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)
plt.title("Heatmap triangulaire + clustering hiérarchique (style Bloomberg)", fontsize=16)
plt.show()

# ------------------------------------------------------------
# 5. COURBES DE PRIX NORMALISÉS
# ------------------------------------------------------------
plt.figure(figsize=(14, 8))
(prices / prices.iloc[0] * 100).plot(figsize=(14, 8))
plt.title("Évolution normalisée des prix (base 100)", fontsize=16)
plt.ylabel("Indice (base 100)")
plt.legend(loc="upper left", ncol=2)
plt.show()

# ------------------------------------------------------------
# 6. HISTOGRAMMES DES RENDEMENTS
# ------------------------------------------------------------
returns.hist(bins=50, figsize=(14, 10), grid=False)
plt.suptitle("Distribution des rendements journaliers", fontsize=16)
plt.show()

# ------------------------------------------------------------
# 7. FRONTIÈRE EFFICIENTE (SIMULATION MONTE CARLO)
# ------------------------------------------------------------
mean_returns = returns.mean()
cov_matrix = returns.cov()

def portfolio_performance(weights):
    ret = np.sum(mean_returns * weights) * 252
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe = ret / vol
    return ret, vol, sharpe

def random_portfolios(n=5000):
    results = np.zeros((3, n))
    weights_record = []
    for i in range(n):
        weights = np.random.random(len(tickers))
        weights /= np.sum(weights)
        weights_record.append(weights)
        ret, vol, sharpe = portfolio_performance(weights)
        results[0,i] = vol
        results[1,i] = ret
        results[2,i] = sharpe
    return results, weights_record

results, weights_record = random_portfolios()

plt.figure(figsize=(12, 8))
plt.scatter(results[0], results[1], c=results[2], cmap='viridis', s=10)
plt.colorbar(label='Sharpe ratio')
plt.xlabel('Volatilité')
plt.ylabel('Rendement annuel')
plt.title('Frontière efficiente (simulation Monte Carlo)')
plt.show()

# ------------------------------------------------------------
# 8. PORTFEUILLE MAX SHARPE
# ------------------------------------------------------------
max_sharpe_idx = np.argmax(results[2])
best_weights = weights_record[max_sharpe_idx]

print("\n=== Portefeuille max Sharpe ===")
for t, w in zip(tickers, best_weights):
    print(f"{t} : {w:.2%}")
