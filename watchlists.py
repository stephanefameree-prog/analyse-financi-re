import json
import os

import pandas as pd
import streamlit as st

from asset_summary import build_asset_summary_table
from data_loader import add_company_names, get_ticker_names, ticker_label
from display_units import (
    WATCHLIST_FORMAT,
    WATCHLIST_LABELS,
    format_map_for_labeled_columns,
    rename_columns_for_display,
)

WATCHLISTS_PATH = "watchlists.json"
DEFAULT_WATCHLIST_NAME = "Ma watchlist"


def _empty_store():
    return {"watchlists": {DEFAULT_WATCHLIST_NAME: []}}


def load_watchlists_data():
    if not os.path.exists(WATCHLISTS_PATH):
        return _empty_store()
    try:
        with open(WATCHLISTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "watchlists" not in data:
            return _empty_store()
        if not isinstance(data["watchlists"], dict):
            data["watchlists"] = {DEFAULT_WATCHLIST_NAME: []}
        if not data["watchlists"]:
            data["watchlists"] = {DEFAULT_WATCHLIST_NAME: []}
        return data
    except Exception:
        return _empty_store()


def save_watchlists_data(data):
    with open(WATCHLISTS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _normalize_ticker(ticker):
    return str(ticker).upper().strip()


def get_watchlist_names(data):
    return list(data.get("watchlists", {}).keys())


def get_watchlist_tickers(data, name):
    tickers = data.get("watchlists", {}).get(name, [])
    return [_normalize_ticker(t) for t in tickers if str(t).strip()]


def create_watchlist(data, name):
    name = str(name).strip()
    if not name:
        return False, "Nom de watchlist invalide."
    if name in data["watchlists"]:
        return False, f"La watchlist « {name} » existe déjà."
    data["watchlists"][name] = []
    save_watchlists_data(data)
    return True, f"Watchlist « {name} » créée."


def delete_watchlist(data, name):
    if name not in data["watchlists"]:
        return False, "Watchlist introuvable."
    if len(data["watchlists"]) <= 1:
        return False, "Impossible de supprimer la dernière watchlist."
    del data["watchlists"][name]
    save_watchlists_data(data)
    return True, f"Watchlist « {name} » supprimée."


def rename_watchlist(data, old_name, new_name):
    new_name = str(new_name).strip()
    if not new_name:
        return False, "Nouveau nom invalide."
    if old_name not in data["watchlists"]:
        return False, "Watchlist introuvable."
    if new_name in data["watchlists"] and new_name != old_name:
        return False, f"« {new_name} » existe déjà."
    data["watchlists"][new_name] = data["watchlists"].pop(old_name)
    save_watchlists_data(data)
    return True, f"Watchlist renommée en « {new_name} »."


def add_ticker_to_watchlist(data, name, ticker):
    ticker = _normalize_ticker(ticker)
    if not ticker:
        return False, "Ticker invalide."
    if name not in data["watchlists"]:
        return False, "Watchlist introuvable."
    if ticker in data["watchlists"][name]:
        return False, f"{ticker} est déjà dans la watchlist."
    data["watchlists"][name].append(ticker)
    save_watchlists_data(data)
    return True, f"{ticker} ajouté."


def remove_ticker_from_watchlist(data, name, ticker):
    if name not in data["watchlists"]:
        return False, "Watchlist introuvable."
    tickers = [_normalize_ticker(t) for t in data["watchlists"][name]]
    ticker = _normalize_ticker(ticker)
    if ticker not in tickers:
        return False, f"{ticker} absent de la watchlist."
    data["watchlists"][name] = [t for t in tickers if t != ticker]
    save_watchlists_data(data)
    return True, f"{ticker} retiré."


def import_tickers_to_watchlist(data, name, tickers):
    if name not in data["watchlists"]:
        return False, "Watchlist introuvable."
    existing = {_normalize_ticker(t) for t in data["watchlists"][name]}
    added = 0
    for ticker in tickers:
        tk = _normalize_ticker(ticker)
        if tk and tk not in existing:
            data["watchlists"][name].append(tk)
            existing.add(tk)
            added += 1
    if added:
        save_watchlists_data(data)
    return True, f"{added} ticker(s) importé(s)."


WATCHLIST_DISPLAY_COLUMNS = [
    "Ticker",
    "Nom",
    "Prix",
    "Min 52 sem.",
    "Max 52 sem.",
    "Position 52 sem.",
    "Variation 1 an",
    "Verdict fondamental",
    "Verdict technique",
    "Synthèse globale",
    "Commentaire",
    "PER",
    "Upside vs objectif",
    "RSI",
    "Bollinger %B",
    "Score global",
]


def build_watchlist_table(prices, rsi_period=14):
    df = build_asset_summary_table(prices, rsi_period=rsi_period)
    if df.empty:
        return df
    for col in WATCHLIST_DISPLAY_COLUMNS:
        if col not in df.columns and col != "Nom":
            df[col] = None
    return df


def render_watchlist_sidebar(show_company_names, index_tickers=None):
    """Gère les watchlists dans la sidebar. Retourne (liste_tickers, nom_watchlist_active)."""
    data = load_watchlists_data()
    names = get_watchlist_names(data)
    if not names:
        data = _empty_store()
        save_watchlists_data(data)
        names = get_watchlist_names(data)

    st.sidebar.subheader("👁 Mes Watchlists")

    active_name = st.sidebar.selectbox(
        "Watchlist active",
        names,
        key="wl_active_select",
    )

    with st.sidebar.expander("➕ Nouvelle watchlist", expanded=False):
        new_wl_name = st.text_input("Nom", key="wl_new_name")
        if st.button("Créer", key="wl_create_btn"):
            ok, msg = create_watchlist(data, new_wl_name)
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    with st.sidebar.expander("✏️ Renommer / supprimer", expanded=False):
        rename_to = st.text_input("Nouveau nom", key="wl_rename_to")
        c1, c2 = st.columns(2)
        if c1.button("Renommer", key="wl_rename_btn"):
            data = load_watchlists_data()
            ok, msg = rename_watchlist(data, active_name, rename_to)
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)
        if c2.button("Supprimer", key="wl_delete_btn"):
            data = load_watchlists_data()
            ok, msg = delete_watchlist(data, active_name)
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    with st.sidebar.expander("➕ Ajouter un ticker", expanded=False):
        add_tk = st.text_input(
            "Ticker Yahoo (ex: AIR.PA, NVDA)", key="wl_add_ticker"
        ).upper().strip()
        if st.button("Ajouter à la watchlist", key="wl_add_btn"):
            if add_tk:
                data = load_watchlists_data()
                ok, msg = add_ticker_to_watchlist(data, active_name, add_tk)
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
            else:
                st.error("Saisissez un ticker.")

    tickers = get_watchlist_tickers(load_watchlists_data(), active_name)

    if index_tickers:
        with st.sidebar.expander("📥 Importer depuis tickers.json", expanded=False):
            import_list = st.selectbox(
                "Liste prédéfinie",
                [""] + list(index_tickers.keys()),
                key="wl_import_list",
            )
            if import_list and st.button("Importer", key="wl_import_btn"):
                data = load_watchlists_data()
                ok, msg = import_tickers_to_watchlist(
                    data, active_name, index_tickers.get(import_list, [])
                )
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

    portfolio_path = "mon_portefeuille.csv"
    if os.path.exists(portfolio_path):
        with st.sidebar.expander("📥 Importer depuis mon portefeuille", expanded=False):
            st.caption("Copie les tickers sans quantité ni PRU.")
            if st.button("Importer le portefeuille CSV", key="wl_import_pf_btn"):
                try:
                    pf = pd.read_csv(portfolio_path)
                    pf.columns = pf.columns.str.strip()
                    pf_tickers = pf["Ticker"].astype(str).str.upper().str.strip().tolist()
                    data = load_watchlists_data()
                    ok, msg = import_tickers_to_watchlist(data, active_name, pf_tickers)
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
                except Exception as e:
                    st.error(f"Erreur lecture portefeuille : {e}")

    if tickers:
        wl_names = get_ticker_names(tickers) if show_company_names else {}
        df_wl = pd.DataFrame({"Ticker": tickers})
        df_wl = add_company_names(df_wl, wl_names, show_names=show_company_names)
        st.sidebar.write(f"**{len(tickers)} titres** dans « {active_name} »")
        st.sidebar.dataframe(df_wl, hide_index=True, use_container_width=True)

        remove_tk = st.sidebar.selectbox(
            "Retirer un ticker :",
            ["-"] + tickers,
            format_func=lambda t: "-"
            if t == "-"
            else ticker_label(t, wl_names, show_company_names),
            key="wl_remove_select",
        )
        if remove_tk != "-" and st.sidebar.button("Retirer", key="wl_remove_btn"):
            data = load_watchlists_data()
            ok, msg = remove_ticker_from_watchlist(data, active_name, remove_tk)
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)
    else:
        st.sidebar.info(
            f"La watchlist « {active_name} » est vide. Ajoutez des tickers ci-dessus."
        )

    return tickers, active_name


def render_watchlist_dashboard(prices, ticker_names, show_company_names, watchlist_name):
    st.header(f"👁 Watchlist : {watchlist_name}")
    st.caption(
        "Suivi sans quantité ni PRU : cours actuel, fourchette 52 semaines, "
        "verdicts fondamental et technique (Yahoo Finance)."
    )

    rsi_period = st.slider("Période RSI", 7, 21, 14, key="wl_rsi_period")
    run = st.button("Actualiser l'analyse", type="primary", key="wl_refresh")

    cache_key = f"watchlist_{watchlist_name}_{abs(hash('|'.join(sorted(prices.columns))))}"
    rsi_key = f"{cache_key}_rsi"

    if run or cache_key not in st.session_state:
        with st.spinner("Analyse fondamentale et technique en cours..."):
            st.session_state[cache_key] = build_watchlist_table(
                prices, rsi_period=rsi_period
            )
            st.session_state[rsi_key] = rsi_period
    elif st.session_state.get(rsi_key) != rsi_period:
        st.caption("Période RSI modifiée — cliquez **Actualiser l'analyse** pour recalculer.")

    df = st.session_state.get(cache_key)
    if df is None or df.empty:
        st.info("Cliquez sur **Actualiser l'analyse** pour analyser les titres de la watchlist.")
        return

    if show_company_names:
        df = add_company_names(df, ticker_names, show_names=True)

    display_cols = [c for c in WATCHLIST_DISPLAY_COLUMNS if c in df.columns]
    view_df = df[display_cols].copy()
    view_display = rename_columns_for_display(view_df, WATCHLIST_LABELS)
    watchlist_format = format_map_for_labeled_columns(
        view_display, WATCHLIST_LABELS, WATCHLIST_FORMAT
    )

    cheap = (view_df["Synthèse globale"].isin(["Bon marché", "Plutôt bon marché"])).sum()
    rich = (view_df["Synthèse globale"].isin(["Cher", "Plutôt cher"])).sum()
    neutral = len(view_df) - cheap - rich
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Titres suivis", len(view_df))
    c2.metric("Bon marché / plutôt bon marché", int(cheap))
    c3.metric("Neutre", int(neutral))
    c4.metric("Cher / plutôt cher", int(rich))

    st.caption(
        "Montants **/ action** · pourcentages en **%** · PER en **×** · "
        "position 52 sem. = % de la fourchette min–max."
    )
    st.dataframe(
        view_display.style.format(watchlist_format, na_rep="-")
        .background_gradient(cmap="RdYlGn", subset=["Score global (pts)", "Score global"]),
        use_container_width=True,
        height=min(600, 38 + 35 * len(view_display)),
    )

    detail = st.selectbox(
        "Détail par titre",
        view_df["Ticker"],
        format_func=lambda t: ticker_label(t, ticker_names, show_company_names),
        key="wl_detail",
    )
    row = view_df[view_df["Ticker"] == detail].iloc[0]
    label = ticker_label(detail, ticker_names, show_company_names)
    st.markdown(f"### {label}")
    d1, d2, d3 = st.columns(3)
    d1.metric("Prix", f"{row['Prix']:.2f}" if pd.notna(row.get("Prix")) else "-")
    d2.metric(
        "Min / Max 52 sem.",
        f"{row['Min 52 sem.']:.2f} / {row['Max 52 sem.']:.2f}"
        if pd.notna(row.get("Min 52 sem.")) and pd.notna(row.get("Max 52 sem."))
        else "-",
    )
    d3.metric("Synthèse", row.get("Synthèse globale", "N/A"))

    st.markdown(f"**Fondamental :** {row.get('Verdict fondamental', 'N/A')}")
    st.markdown(f"**Technique :** {row.get('Verdict technique', 'N/A')}")
    st.info(row.get("Commentaire", ""))

    csv = view_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Télécharger la watchlist (CSV)",
        csv,
        f"watchlist_{watchlist_name.replace(' ', '_')}.csv",
        key="wl_csv_dl",
    )
