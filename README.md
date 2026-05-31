# Dashboard financier V6.3

Application **Streamlit** d'analyse de portefeuille et de marchés : synthèse P&L, risque, Markowitz, dividendes, fondamentaux, technique, FFT, watchlists.

## Lancement

```bash
cd "analyse financiere"
pip install -r requirements.txt
streamlit run dashboardV6.3.py
```

Sous Windows : double-clic sur `lancerdashboard.bat`.

Depuis le dossier parent `Documenten`, vous pouvez aussi lancer `dashboardV6.3.py` (redirecteur).

## Configuration

| Fichier | Rôle |
|---------|------|
| `mon_portefeuille.csv` | Vos positions (Ticker, Quantite, PRU) — **local, non versionné** |
| `mon_portefeuille.csv.example` | Modèle à copier |
| `tickers.json` | Univers de tickers (indices, listes BFM…) |
| `watchlists.json` | Watchlists personnelles — **local, non versionné** |

```bash
copy mon_portefeuille.csv.example mon_portefeuille.csv
```

## Modules principaux

- `dashboardV6.3.py` — interface (9 vues)
- `analytics.py` — risque, Markowitz, technique, FFT
- `fundamentals.py` — ratios & consensus analystes
- `dividendes.py` — qualité dividendes & univers
- `data_loader.py` — prix Yahoo, cache OHLCV
- `display_units.py` — libellés avec unités dans les tableaux

## Tests

```bash
pytest tests/
```

## Publier sur GitHub

1. Installer [Git for Windows](https://git-scm.com/download/win)
2. Créer un dépôt **vide** sur [github.com/new](https://github.com/new) (ex. `dashboard-financier`)
3. Exécuter :

```powershell
cd "chemin\vers\analyse financiere"
.\publier_github.ps1 -RemoteUrl "https://github.com/VOTRE_COMPTE/dashboard-financier.git"
```

Ou suivre les commandes affichées par le script.

## Confidentialité

Ne commitez **jamais** `mon_portefeuille.csv`, `watchlists.json` ni les fichiers `*_cache.json` (déjà exclus via `.gitignore`).
