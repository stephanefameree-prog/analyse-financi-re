import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
tickers = [
    "HMY", "XPL", "NYXH.BR", "NUTX",
    "ALNTG.PA", "IPS.PA", "EDEN.PA",
    "ZAL.DE", "ETL.PA",
    "GTT.PA", "CAP.PA", "RMS.PA",
    "GBLB.BR", "GIMB.BR", "AI.PA",
    "CA.PA", "TEP.PA",
    "FRVIA.PA", "BOI.PA", "LTA.PA",
    "RIB.PA", "VUS.PA", "TRI.PA"
]




print("\n=== Téléchargement des données ===\n")

# Téléchargement groupé (beaucoup plus fiable)
df = yf.download(tickers, start="2020-01-01")

# Extraction automatique de Close ou Adj Close
if ("Adj Close" in df.columns):
    data = df["Adj Close"]
elif ("Close" in df.columns):
    data = df["Close"]
else:
    print("Aucune colonne Close/Adj Close trouvée.")
    exit()

# Nettoyage
data = data.dropna(how="all")  # supprime les lignes vides
data = data.dropna(axis=1, how="all")  # supprime les colonnes vides

if data.empty:
    print("Aucune donnée exploitable.")
    exit()

returns = data.pct_change().dropna()

corr = returns.corr()
cov = returns.cov()

print("\n=== MATRICE DE CORRÉLATION ===\n")
print(corr)

print("\n=== MATRICE DE COVARIANCE ===\n")
print(cov)

plt.figure(figsize=(10, 7))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
plt.title("Heatmap des corrélations")
plt.tight_layout()
plt.show()
