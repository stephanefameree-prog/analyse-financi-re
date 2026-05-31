"""Libellés de colonnes avec unités explicites pour les tableaux du dashboard."""

import pandas as pd

# --- Dividendes (univers) ---

DIVIDEND_UNIVERSE_LABELS = {
    "Ratio couverture": "Ratio couverture (×)",
    "Série versement (ans)": "Série versement (ans)",
    "Série croissance (ans)": "Série croissance (ans)",
    "Dividende / action": "Dividende / action (/ action)",
    "Dividende TTM": "Dividende TTM (/ action, 12 mois)",
    "Dividende total historique": "Dividende total hist. (/ action, cumul)",
    "Prix": "Prix (/ action)",
}

DIVIDEND_UNIVERSE_FORMAT = {
    "Rendement (%)": "{:.2%}",
    "Croissance 1 an (%)": "{:.2%}",
    "Croissance 3 ans (%)": "{:.2%}",
    "Croissance 5 ans (%)": "{:.2%}",
    "Rendement moy. 5 ans (%)": "{:.2%}",
    "Taux versement (%)": "{:.2%}",
    "Ratio couverture (×)": "{:.2f}×",
    "Ratio couverture": "{:.2f}×",
    "Série versement (ans)": "{:.0f} ans",
    "Série croissance (ans)": "{:.0f} ans",
    "Dividende / action (/ action)": "{:.2f}",
    "Dividende / action": "{:.2f}",
    "Dividende TTM (/ action, 12 mois)": "{:.2f}",
    "Dividende TTM": "{:.2f}",
    "Dividende total hist. (/ action, cumul)": "{:.2f}",
    "Dividende total historique": "{:.2f}",
    "Prix (/ action)": "{:.2f}",
    "Prix": "{:.2f}",
}

DIVIDEND_UNIVERSE_CAPTION = (
    "Montants **par action** : unité monétaire = colonne **Devise** (EUR, USD…). "
    "Rendements et croissances en **%**. Ratio couverture en **×** (ex. 1,5× = marge de sécurité)."
)

# --- Dividendes (portefeuille) ---

DIVIDEND_PORTFOLIO_LABELS = {
    "Prix": "Prix (/ action)",
    "Rendement": "Rendement (%)",
    "CAGR 5 ans": "CAGR 5 ans (%)",
    "Payout": "Payout (%)",
    "FCF Payout": "FCF Payout (%)",
    "Dividende annuel (TTM)": "Dividende annuel TTM (/ action)",
    "Dividende annuel": "Dividende annuel (/ action)",
    "DPS dernier versement": "DPS dernier versement (/ action)",
    "Années de croissance": "Années de croissance (ans)",
    "Score": "Score qualité (pts / ~100)",
    "Score stabilité": "Score stabilité (pts / 30)",
    "Score croissance": "Score croissance (pts / 25)",
    "Score payout": "Score payout (pts / 20)",
    "Score FCF": "Score FCF (pts / 20)",
    "Score rendement": "Score rendement (pts / 5)",
}

DIVIDEND_PORTFOLIO_FORMAT_BASE = {
    "Rendement (%)": "{:.2%}",
    "Rendement": "{:.2%}",
    "CAGR 5 ans (%)": "{:.2%}",
    "CAGR 5 ans": "{:.2%}",
    "Payout (%)": "{:.2%}",
    "Payout": "{:.2%}",
    "FCF Payout (%)": "{:.2%}",
    "FCF Payout": "{:.2%}",
    "Années de croissance (ans)": "{:.0f}",
    "Années de croissance": "{:.0f}",
    "Score qualité (pts / ~100)": "{:.0f}",
    "Score": "{:.0f}",
    "Score stabilité (pts / 30)": "{:.0f}",
    "Score stabilité": "{:.0f}",
    "Score croissance (pts / 25)": "{:.0f}",
    "Score croissance": "{:.0f}",
    "Score payout (pts / 20)": "{:.0f}",
    "Score payout": "{:.0f}",
    "Score FCF (pts / 20)": "{:.0f}",
    "Score FCF": "{:.0f}",
    "Score rendement (pts / 5)": "{:.0f}",
    "Score rendement": "{:.0f}",
}

DIVIDEND_PORTFOLIO_CAPTION = (
    "Montants **par action** (voir **Devise** si affichée). Scores en **points** "
    "(max indicatif indiqué dans l'en-tête). Pourcentages au format **%**."
)

# --- Fondamentaux ---

FUNDAMENTALS_LABELS = {
    "Dernier cours": "Dernier cours (/ action)",
    "Prix": "Prix (/ action)",
    "Var. journalière (%)": "Var. journalière (%)",
    "Sommet 52 sem.": "Sommet 52 sem. (/ action)",
    "Creux 52 sem.": "Creux 52 sem. (/ action)",
    "Var. vs sommet 52 sem. (%)": "Var. vs sommet 52 sem. (%)",
    "Var. vs creux 52 sem. (%)": "Var. vs creux 52 sem. (%)",
    "Prix % MA50": "Prix / MA50 (%, 100 = sur la MM)",
    "Prix % MA200": "Prix / MA200 (%, 100 = sur la MM)",
    "Float / actions": "Float / actions (%)",
    "Actions en circulation": "Actions en circulation (nb)",
    "Volat. réalisée 30j": "Volat. réalisée 30j (%, ann.)",
    "Volat. réalisée 90j": "Volat. réalisée 90j (%, ann.)",
    "Volat. réalisée 1 an": "Volat. réalisée 1 an (%, ann.)",
    "Variation 1 an": "Variation 1 an (%)",
    "Position 52 sem.": "Position 52 sem. (0–1)",
    "Min 52 sem.": "Min 52 sem. (/ action)",
    "Max 52 sem.": "Max 52 sem. (/ action)",
    "ROIC": "ROIC (%)",
    "ROIC (moy. 5 ans)": "ROIC moy. 5 ans (%)",
    "ROE": "ROE (%)",
    "ROA": "ROA (%)",
    "ROCE": "ROCE (%)",
    "Marge brute": "Marge brute (%)",
    "Marge brute (moy. 5 ans)": "Marge brute moy. 5 ans (%)",
    "Marge exploitation": "Marge exploitation (%)",
    "Marge exploitation (moy. 5 ans)": "Marge exploitation moy. 5 ans (%)",
    "Marge avant impôt": "Marge avant impôt (%)",
    "Marge avant impôt (moy. 5 ans)": "Marge avant impôt moy. 5 ans (%)",
    "Marge nette": "Marge nette (%)",
    "Marge nette (moy. 5 ans)": "Marge nette moy. 5 ans (%)",
    "Marge EBITDA": "Marge EBITDA (%)",
    "Rotation actif": "Rotation actif (×)",
    "Rotation stocks": "Rotation stocks (×)",
    "Rotation créances": "Rotation créances (×)",
    "Revenu / employé": "Revenu / employé (mon.)",
    "Bn net / employé": "Bn net / employé (mon.)",
    "Capitalisation de marché": "Capitalisation (mon.)",
    "PER": "PER (×)",
    "PER forward": "PER forward (×)",
    "Ratio PEG": "PEG (×)",
    "PEG forward": "PEG forward (×)",
    "VE/EBITDA": "VE/EBITDA (×)",
    "VE/EBITDA forward": "VE/EBITDA forward (×)",
    "Cours / ventes": "Cours / ventes (×)",
    "Cours / flux exploitation": "Cours / flux exploitation (×)",
    "Cours / flux disponible": "Cours / flux disponible (×)",
    "Cours / val. comptable": "Cours / val. comptable (×)",
    "Cours / val. comptable tangible": "Cours / val. comptable tangible (×)",
    "Rendement bénéfices": "Rendement bénéfices (%)",
    "Rendement actionnaire": "Rendement actionnaire (%)",
    "Rendement rachat": "Rendement rachat (%)",
    "Rendement FCF": "Rendement FCF (%)",
    "PER % PER moy. 3 ans": "PER / PER moy. 3 ans (%, 100 = moyenne)",
    "Valeur entreprise": "Valeur entreprise (VE, mon.)",
    "VE/EBIT": "VE/EBIT (×)",
    "VE/FCF": "VE/FCF (×)",
    "Dette / FCF": "Dette / FCF (×)",
    "Dette / Capitaux": "Dette / Capitaux (×)",
    "Upside vs objectif": "Upside vs objectif (%)",
    "Objectif analystes": "Objectif analystes (/ action)",
    "Objectif bas": "Objectif bas (/ action)",
    "Objectif haut": "Objectif haut (/ action)",
    "Score Piotroski": "Score Piotroski (/ 9)",
    "Nb analystes": "Nb analystes (nb)",
}

FUNDAMENTALS_EXTRA_FORMAT = {
    "Prix / MA50 (%, 100 = sur la MM)": "{:.1f}",
    "Prix % MA50": "{:.1f}",
    "Prix / MA200 (%, 100 = sur la MM)": "{:.1f}",
    "Prix % MA200": "{:.1f}",
    "PER / PER moy. 3 ans (%, 100 = moyenne)": "{:.0%}",
    "Rotation actif (×)": "{:.2f}×",
    "Rotation stocks (×)": "{:.2f}×",
    "Rotation créances (×)": "{:.2f}×",
    "Score Piotroski (/ 9)": "{:.0f}",
    "Position 52 sem. (0–1)": "{:.0%}",
}

FUNDAMENTALS_CAPTION = (
    "**%** = pourcentage · **×** = multiple · **mon.** = montant en devise du titre "
    "(colonne Devise si présente, sinon devise Yahoo) · **/ action** = par titre · "
    "**nb** = nombre · **ann.** = annualisé."
)

# --- Watchlist / synthèse ---

WATCHLIST_LABELS = {
    "Prix": "Prix (/ action)",
    "Min 52 sem.": "Min 52 sem. (/ action)",
    "Max 52 sem.": "Max 52 sem. (/ action)",
    "Position 52 sem.": "Position 52 sem. (% fourchette)",
    "Variation 1 an": "Variation 1 an (%)",
    "PER": "PER (×)",
    "Upside vs objectif": "Upside vs objectif (%)",
    "RSI": "RSI (0–100)",
    "Bollinger %B": "Bollinger %B (0–1)",
    "Score global": "Score global (pts)",
}

WATCHLIST_FORMAT = {
    "Prix (/ action)": "{:.2f}",
    "Prix": "{:.2f}",
    "Min 52 sem. (/ action)": "{:.2f}",
    "Min 52 sem.": "{:.2f}",
    "Max 52 sem. (/ action)": "{:.2f}",
    "Max 52 sem.": "{:.2f}",
    "Position 52 sem. (% fourchette)": "{:.0%}",
    "Position 52 sem.": "{:.0%}",
    "Variation 1 an (%)": "{:.2%}",
    "Variation 1 an": "{:.2%}",
    "PER (×)": "{:.2f}×",
    "PER": "{:.2f}×",
    "Upside vs objectif (%)": "{:.2%}",
    "Upside vs objectif": "{:.2%}",
    "RSI (0–100)": "{:.1f}",
    "RSI": "{:.1f}",
    "Bollinger %B (0–1)": "{:.2f}",
    "Bollinger %B": "{:.2f}",
    "Score global (pts)": "{:.2f}",
    "Score global": "{:.2f}",
}

# --- Technique (dashboard) ---

TECHNICAL_LABELS = {
    "RSI": "RSI (0–100)",
    "Stoch %K": "Stochastique %K (0–100)",
    "Stoch %D": "Stochastique %D (0–100)",
    "MACD": "MACD (pts)",
    "Signal MACD": "Signal MACD (pts)",
    "Histogramme": "Histogramme MACD (pts)",
    "Volume": "Volume (titres)",
    "Vol / Moy. 20j": "Volume / moy. 20j (×)",
    "MFI": "MFI (0–100)",
    "SMA 50": "SMA 50 (/ action)",
    "SMA 200": "SMA 200 (/ action)",
    "Bollinger %B": "Bollinger %B (0–1)",
    "ATR": "ATR (/ action)",
    "ATR/Prix": "ATR / Prix (%)",
}

TECHNICAL_FORMAT = {
    "RSI (0–100)": "{:.1f}",
    "RSI": "{:.1f}",
    "Stochastique %K (0–100)": "{:.1f}",
    "Stoch %K": "{:.1f}",
    "Stochastique %D (0–100)": "{:.1f}",
    "Stoch %D": "{:.1f}",
    "MACD (pts)": "{:.4f}",
    "MACD": "{:.4f}",
    "Signal MACD (pts)": "{:.4f}",
    "Signal MACD": "{:.4f}",
    "Histogramme MACD (pts)": "{:.4f}",
    "Histogramme": "{:.4f}",
    "Volume (titres)": "{:,.0f}",
    "Volume": "{:,.0f}",
    "Volume / moy. 20j (×)": "{:.2f}×",
    "Vol / Moy. 20j": "{:.2f}×",
    "MFI (0–100)": "{:.1f}",
    "MFI": "{:.1f}",
    "SMA 50 (/ action)": "{:.2f}",
    "SMA 50": "{:.2f}",
    "SMA 200 (/ action)": "{:.2f}",
    "SMA 200": "{:.2f}",
    "Bollinger %B (0–1)": "{:.2f}",
    "Bollinger %B": "{:.2f}",
    "ATR (/ action)": "{:.2f}",
    "ATR": "{:.2f}",
    "ATR / Prix (%)": "{:.2%}",
    "ATR/Prix": "{:.2%}",
}


def rename_columns_for_display(df, label_map):
    """Renomme les colonnes présentes dans label_map (affichage + export CSV)."""
    if df is None or df.empty:
        return df
    mapping = {}
    target_names = set(df.columns)
    for col in df.columns:
        if col not in label_map:
            continue
        new_name = label_map[col]
        if new_name == col or new_name in target_names:
            continue
        mapping[col] = new_name
        target_names.discard(col)
        target_names.add(new_name)
    if not mapping:
        return df
    return df.rename(columns=mapping)


def pick_format(df, format_map):
    """Ne garde que les formats dont la colonne existe (noms déjà renommés ou bruts)."""
    return {k: v for k, v in format_map.items() if k in df.columns}


def merge_format(*dicts):
    out = {}
    for d in dicts:
        out.update(d)
    return out


def format_map_for_labeled_columns(df, label_map, format_map):
    """Applique format_map en suivant les renommages de label_map."""
    out = pick_format(df, format_map)
    reverse = {v: k for k, v in label_map.items()}
    for col in df.columns:
        if col in out:
            continue
        src = reverse.get(col)
        if src and src in format_map:
            out[col] = format_map[src]
    return out


def internal_column_name(col, *label_maps):
    """Nom interne si col est un libellé d'affichage."""
    for label_map in label_maps:
        for internal, display in label_map.items():
            if col == display:
                return internal
    return col


def expand_sentiment_columns(base_cols, *label_maps):
    """Inclut les libellés affichés pour le code couleur."""
    expanded = set(base_cols)
    for label_map in label_maps:
        for internal, display in label_map.items():
            if internal in base_cols:
                expanded.add(display)
    return frozenset(expanded)


# --- Risque / portefeuille (dashboard) ---

RISK_METRICS_FORMAT = {
    "Rendement Annuel (moyenne)": "{:.2%}",
    "Médiane annuelle": "{:.2%}",
    "Mode annuelle": "{:.2%}",
    "Volatilité (Sigma)": "{:.2%}",
    "Ratio de Sharpe": "{:.2f}×",
    "Ratio de Sortino": "{:.2f}×",
    "VaR 95% (Jour)": "{:.2%}",
    "CVaR 95% (Jour)": "{:.2%}",
    "Skewness (Asymétrie)": "{:.2f}",
    "Kurtosis (Aplatissement)": "{:.2f}",
}

SUGGESTIONS_LABELS = {
    "Score composite": "Score composite (pts)",
    "Rendement candidat": "Rendement candidat (%)",
    "Volatilité candidat": "Volatilité candidat (%)",
    "Skewness candidat": "Skewness candidat",
    "Kurtosis candidat": "Kurtosis candidat",
    "Corr. moy. portefeuille": "Corr. moy. portefeuille (0–1)",
    "Δ Rendement portef.": "Δ Rendement portef. (%)",
    "Δ Volatilité portef.": "Δ Volatilité portef. (%)",
    "Δ Kurtosis portef.": "Δ Kurtosis portef.",
    "Δ Skewness portef.": "Δ Skewness portef.",
    "Δ Corr. interne": "Δ Corr. interne (0–1)",
}

SUGGESTIONS_FORMAT = {
    "Score composite (pts)": "{:.4f}",
    "Score composite": "{:.4f}",
    "Rendement candidat (%)": "{:.2%}",
    "Rendement candidat": "{:.2%}",
    "Volatilité candidat (%)": "{:.2%}",
    "Volatilité candidat": "{:.2%}",
    "Skewness candidat": "{:.2f}",
    "Kurtosis candidat": "{:.2f}",
    "Corr. moy. portefeuille (0–1)": "{:.2f}",
    "Corr. moy. portefeuille": "{:.2f}",
    "Δ Rendement portef. (%)": "{:.2%}",
    "Δ Rendement portef.": "{:.2%}",
    "Δ Volatilité portef. (%)": "{:.2%}",
    "Δ Volatilité portef.": "{:.2%}",
    "Δ Kurtosis portef.": "{:.2f}",
    "Δ Skewness portef.": "{:.2f}",
    "Δ Corr. interne (0–1)": "{:.2f}",
    "Δ Corr. interne": "{:.2f}",
}

ASSET_SUMMARY_LABELS = WATCHLIST_LABELS
ASSET_SUMMARY_FORMAT = WATCHLIST_FORMAT

# --- FFT (dashboard) ---

FFT_LABELS = {
    "Observations": "Observations (jours)",
    "Période dominante": "Période dominante (libellé)",
    "Force cycle principal": "Force cycle principal (%)",
    "Poids #1": "Poids #1 (%)",
    "Poids #2": "Poids #2 (%)",
    "Poids #3": "Poids #3 (%)",
    "Période #1": "Période #1 (libellé)",
    "Période #2": "Période #2 (libellé)",
    "Période #3": "Période #3 (libellé)",
    "Puissance relative": "Puissance relative (%)",
    "Période": "Période (libellé)",
}

FFT_FORMAT = {
    "Observations (jours)": "{:.0f}",
    "Observations": "{:.0f}",
    "Période (jours)": "{:.0f}",
    "Force cycle principal (%)": "{:.1%}",
    "Force cycle principal": "{:.1%}",
    "Poids #1 (%)": "{:.1%}",
    "Poids #1": "{:.1%}",
    "Poids #2 (%)": "{:.1%}",
    "Poids #2": "{:.1%}",
    "Poids #3 (%)": "{:.1%}",
    "Poids #3": "{:.1%}",
    "Puissance relative (%)": "{:.1%}",
    "Puissance relative": "{:.1%}",
}
