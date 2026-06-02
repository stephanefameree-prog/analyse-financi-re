import random
import time
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from yahooquery import Ticker

from analytics import compute_sma
from display_units import (
    FUNDAMENTALS_CAPTION,
    FUNDAMENTALS_EXTRA_FORMAT,
    FUNDAMENTALS_LABELS,
    expand_sentiment_columns,
    format_map_for_labeled_columns,
    internal_column_name,
    pick_format,
    rename_columns_for_display,
)

FUNDAMENTALS_DISK_CACHE_FILE = "fundamentals_cache.json"
DISK_CACHE_TTL_SECONDS = 86400


def _fundamentals_cache_path():
    return Path(__file__).resolve().parent / FUNDAMENTALS_DISK_CACHE_FILE


def _load_fundamentals_disk_store():
    path = _fundamentals_cache_path()
    if not path.is_file():
        return {"version": 1, "portfolios": {}}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("portfolios"), dict):
            data.setdefault("tickers", {})
            return data
    except Exception:
        pass
    return {"version": 1, "portfolios": {}, "tickers": {}}


def _save_fundamentals_disk_store(data):
    path = _fundamentals_cache_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def load_fundamentals_rows_from_disk(ticker_signature):
    """Charge les lignes fondamentales depuis le disque (TTL 24 h)."""
    entry = _load_fundamentals_disk_store().get("portfolios", {}).get(ticker_signature)
    if not entry:
        return None, None
    updated_at = entry.get("updated_at")
    if updated_at:
        try:
            ts = datetime.fromisoformat(updated_at)
            if (datetime.now() - ts).total_seconds() > DISK_CACHE_TTL_SECONDS:
                return None, updated_at
        except Exception:
            pass
    rows = entry.get("rows")
    if not rows:
        return None, updated_at
    return rows, updated_at


def save_fundamentals_rows_to_disk(ticker_signature, rows):
    store = _load_fundamentals_disk_store()
    store.setdefault("portfolios", {})[ticker_signature] = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "rows": rows,
    }
    _save_fundamentals_disk_store(store)


def load_fundamental_ticker_from_disk(ticker):
    """Charge les fondamentaux d'un ticker depuis le cache disque (TTL 24 h)."""
    entry = _load_fundamentals_disk_store().get("tickers", {}).get(ticker)
    if not entry:
        return None
    row = entry.get("row") if isinstance(entry, dict) else entry
    if row and _row_is_fresh(row):
        return row
    return None


def save_fundamental_ticker_to_disk(ticker, row):
    """Enregistre les fondamentaux d'un ticker sur le disque."""
    if not row:
        return
    store = _load_fundamentals_disk_store()
    store.setdefault("tickers", {})[ticker] = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "row": row,
    }
    _save_fundamentals_disk_store(store)


def clear_fundamentals_disk_cache(ticker_signature=None):
    if ticker_signature is None:
        path = _fundamentals_cache_path()
        if path.is_file():
            path.unlink()
        return
    store = _load_fundamentals_disk_store()
    portfolios = store.get("portfolios", {})
    if ticker_signature in portfolios:
        del portfolios[ticker_signature]
        _save_fundamentals_disk_store(store)


def _hydrate_fundamentals_from_disk(cache_key, ticker_signature):
    if st.session_state.get(cache_key):
        return False
    rows, updated_at = load_fundamentals_rows_from_disk(ticker_signature)
    if not rows:
        return False
    st.session_state[cache_key] = rows
    st.session_state[f"{cache_key}_disk_loaded"] = updated_at
    return True


def _fundamentals_rows_index(rows):
    return {r["Ticker"]: r for r in rows if r.get("Ticker")}


def _row_is_fresh(row, ttl_seconds=DISK_CACHE_TTL_SECONDS):
    ts = row.get("_fetched_at")
    if not ts:
        return False
    try:
        return (datetime.now() - datetime.fromisoformat(ts)).total_seconds() <= ttl_seconds
    except Exception:
        return False


def _safe_float(value):
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


def _find_row_value(df, patterns):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    for pattern in patterns:
        p = pattern.lower()
        for idx in df.index:
            if p in str(idx).lower():
                row = df.loc[idx]
                if isinstance(row, pd.Series):
                    for val in row.sort_index(ascending=False):
                        v = _safe_float(val)
                        if v is not None:
                            return v
                else:
                    v = _safe_float(row)
                    if v is not None:
                        return v
    return None


def _safe_div(num, den):
    n = _safe_float(num)
    d = _safe_float(den)
    if n is None or d in (None, 0):
        return None
    return n / d


def _avg_valid(values):
    vals = [_safe_float(v) for v in values]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def _prep_annual_statement(df, ticker):
    """Extrait les exercices annuels (12M) triés du plus récent au plus ancien."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    out = df.reset_index() if df.index.name == "symbol" or "symbol" in getattr(df.index, "names", []) else df.copy()
    if "symbol" in out.columns:
        out = out[out["symbol"].astype(str) == str(ticker)]
    if "periodType" in out.columns:
        out = out[out["periodType"].astype(str).str.upper().isin(["12M", "ANNUAL"])]
    if "asOfDate" not in out.columns:
        return pd.DataFrame()
    out = out.copy()
    out["asOfDate"] = pd.to_datetime(out["asOfDate"], errors="coerce")
    out = out.dropna(subset=["asOfDate"])
    out = out.sort_values("asOfDate", ascending=False).drop_duplicates("asOfDate", keep="first")
    return out.reset_index(drop=True)


def _row_get(row, *keys):
    if row is None:
        return None
    for key in keys:
        if key in row.index:
            v = _safe_float(row.get(key))
            if v is not None:
                return v
    return None


def _inventory_value(bs_row):
    if bs_row is None:
        return None
    for key in (
        "Inventory",
        "InventoryFinishedGoods",
        "Inventories",
        "TotalInventory",
        "MerchandiseInventory",
    ):
        v = _row_get(bs_row, key)
        if v is not None and v > 0:
            return v
    return None


def _year_profitability_ratios(inc_row, bs_row):
    """Ratios rentabilité / rotation pour un exercice annuel."""
    revenue = _row_get(inc_row, "TotalRevenue", "OperatingRevenue")
    cogs = _row_get(inc_row, "CostOfRevenue", "ReconciledCostOfRevenue")
    gross_profit = _row_get(inc_row, "GrossProfit")
    if gross_profit is None and revenue is not None and cogs is not None:
        gross_profit = revenue - cogs

    operating_income = _row_get(inc_row, "OperatingIncome", "TotalOperatingIncomeAsReported")
    ebit = _row_get(inc_row, "EBIT") or operating_income
    ebitda = _row_get(inc_row, "EBITDA", "NormalizedEBITDA")
    pretax = _row_get(inc_row, "PretaxIncome")
    net_income = _row_get(
        inc_row,
        "NetIncomeCommonStockholders",
        "NetIncome",
        "NetIncomeContinuousOperations",
    )

    total_assets = _row_get(bs_row, "TotalAssets")
    equity = _row_get(bs_row, "CommonStockEquity", "StockholdersEquity", "TotalEquityGrossMinorityInterest")
    total_debt = _row_get(bs_row, "TotalDebt")
    cash = _row_get(bs_row, "CashAndCashEquivalents", "CashCashEquivalentsAndShortTermInvestments")
    current_assets = _row_get(bs_row, "CurrentAssets")
    current_liabilities = _row_get(bs_row, "CurrentLiabilities")
    receivables = _row_get(bs_row, "AccountsReceivable", "GrossAccountsReceivable", "NetReceivables")
    inventory = _inventory_value(bs_row)

    tax_rate = _row_get(inc_row, "TaxRateForCalcs")
    if tax_rate is not None and (tax_rate < 0 or tax_rate > 1):
        tax_rate = None
    nopat = None
    if ebit is not None:
        if tax_rate is not None:
            nopat = ebit * (1 - tax_rate)
        elif pretax not in (None, 0) and net_income is not None:
            eff = max(0.0, min(0.5, (pretax - net_income) / pretax))
            nopat = ebit * (1 - eff)
        else:
            nopat = ebit * 0.75

    invested_capital = None
    if equity is not None:
        invested_capital = equity + (total_debt or 0) - (cash or 0)

    capital_employed = None
    if total_assets is not None and current_liabilities is not None:
        capital_employed = total_assets - current_liabilities

    return {
        "gross_margin": _safe_div(gross_profit, revenue),
        "operating_margin": _safe_div(operating_income, revenue),
        "pretax_margin": _safe_div(pretax, revenue),
        "net_margin": _safe_div(net_income, revenue),
        "ebitda_margin": _safe_div(ebitda, revenue),
        "roa": _safe_div(net_income, total_assets),
        "roe": _safe_div(net_income, equity),
        "roic": _safe_div(nopat, invested_capital) if invested_capital and invested_capital > 0 else None,
        "roce": _safe_div(ebit, capital_employed) if capital_employed and capital_employed > 0 else None,
        "asset_turnover": _safe_div(revenue, total_assets),
        "inventory_turnover": _safe_div(cogs, inventory),
        "receivables_turnover": _safe_div(revenue, receivables),
        "revenue": revenue,
        "net_income": net_income,
        "total_assets": total_assets,
        "equity": equity,
        "long_term_debt": _row_get(bs_row, "LongTermDebt", "LongTermDebtAndCapitalLeaseObligation"),
        "current_assets": current_assets,
        "current_liabilities": current_liabilities,
        "operating_cash_flow": None,
        "shares": _row_get(inc_row, "DilutedAverageShares", "BasicAverageShares"),
        "gross_profit": gross_profit,
    }


def _attach_cashflow(ratios, cf_row):
    if ratios is None or cf_row is None:
        return ratios
    ratios = dict(ratios)
    ratios["operating_cash_flow"] = _row_get(
        cf_row,
        "OperatingCashFlow",
        "CashFlowFromContinuingOperatingActivities",
        "TotalCashFromOperatingActivities",
    )
    return ratios


def _piotroski_score(cur, prev):
    """Score Piotroski F (0–9) sur les deux derniers exercices annuels."""
    if cur is None or prev is None:
        return None
    score = 0

    if (cur.get("roa") or 0) > 0:
        score += 1
    if (cur.get("operating_cash_flow") or 0) > 0:
        score += 1
    if cur.get("roa") is not None and prev.get("roa") is not None and cur["roa"] > prev["roa"]:
        score += 1
    ocf = cur.get("operating_cash_flow")
    ni = cur.get("net_income")
    if ocf is not None and ni is not None and ocf > ni:
        score += 1

    cur_ltd = cur.get("long_term_debt") or 0
    prev_ltd = prev.get("long_term_debt") or 0
    cur_ta = cur.get("total_assets")
    prev_ta = prev.get("total_assets")
    if cur_ta and prev_ta and cur_ta > 0 and prev_ta > 0:
        if (cur_ltd / cur_ta) < (prev_ltd / prev_ta):
            score += 1

    cur_cr = _safe_div(cur.get("current_assets"), cur.get("current_liabilities"))
    prev_cr = _safe_div(prev.get("current_assets"), prev.get("current_liabilities"))
    if cur_cr is not None and prev_cr is not None and cur_cr > prev_cr:
        score += 1

    cur_sh = cur.get("shares")
    prev_sh = prev.get("shares")
    if cur_sh is not None and prev_sh is not None and cur_sh <= prev_sh:
        score += 1

    if (
        cur.get("gross_profit") is not None
        and prev.get("gross_profit") is not None
        and cur.get("revenue") not in (None, 0)
        and prev.get("revenue") not in (None, 0)
    ):
        if (cur["gross_profit"] / cur["revenue"]) > (prev["gross_profit"] / prev["revenue"]):
            score += 1

    if (
        cur.get("asset_turnover") is not None
        and prev.get("asset_turnover") is not None
        and cur["asset_turnover"] > prev["asset_turnover"]
    ):
        score += 1

    return score


def _build_yearly_ratios(inc, bs, cf, max_years=5):
    """Construit la liste des ratios annuels alignés sur la date d'exercice."""
    if inc is None or inc.empty or bs is None or bs.empty:
        return []

    bs_by_date = {row["asOfDate"]: row for _, row in bs.iterrows()}
    cf_by_date = {row["asOfDate"]: row for _, row in cf.iterrows()} if cf is not None and not cf.empty else {}

    yearly = []
    for _, inc_row in inc.iterrows():
        dt = inc_row.get("asOfDate")
        bs_row = bs_by_date.get(dt)
        if bs_row is None:
            continue
        ratios = _year_profitability_ratios(inc_row, bs_row)
        ratios = _attach_cashflow(ratios, cf_by_date.get(dt))
        yearly.append(ratios)
        if len(yearly) >= max_years:
            break
    return yearly


def compute_profitability_indicators(ticker):
    """Indicateurs rentabilité / efficacité (capture Investing.com / BFM)."""
    metrics = {}
    try:
        t = Ticker(ticker, timeout=25)
        inc = _prep_annual_statement(t.income_statement(frequency="a"), ticker)
        bs = _prep_annual_statement(t.balance_sheet(frequency="a"), ticker)
        cf = _prep_annual_statement(t.cash_flow(frequency="a"), ticker)

        yearly = _build_yearly_ratios(inc, bs, cf, max_years=5)

        if yearly:
            latest = yearly[0]
            metrics["ROIC"] = latest.get("roic")
            metrics["ROE"] = latest.get("roe")
            metrics["ROA"] = latest.get("roa")
            metrics["ROCE"] = latest.get("roce")
            metrics["Marge brute"] = latest.get("gross_margin")
            metrics["Marge exploitation"] = latest.get("operating_margin")
            metrics["Marge avant impôt"] = latest.get("pretax_margin")
            metrics["Marge nette"] = latest.get("net_margin")
            metrics["Marge EBITDA"] = latest.get("ebitda_margin")
            metrics["Rotation actif"] = latest.get("asset_turnover")
            metrics["Rotation stocks"] = latest.get("inventory_turnover")
            metrics["Rotation créances"] = latest.get("receivables_turnover")

            metrics["ROIC (moy. 5 ans)"] = _avg_valid([y.get("roic") for y in yearly])
            metrics["Marge brute (moy. 5 ans)"] = _avg_valid([y.get("gross_margin") for y in yearly])
            metrics["Marge exploitation (moy. 5 ans)"] = _avg_valid([y.get("operating_margin") for y in yearly])
            metrics["Marge avant impôt (moy. 5 ans)"] = _avg_valid([y.get("pretax_margin") for y in yearly])
            metrics["Marge nette (moy. 5 ans)"] = _avg_valid([y.get("net_margin") for y in yearly])

            if len(yearly) >= 2:
                metrics["Score Piotroski"] = _piotroski_score(yearly[0], yearly[1])

        info = yf.Ticker(ticker).info or {}
        employees = _safe_float(info.get("fullTimeEmployees"))
        if employees and employees > 0:
            rev = _safe_float(info.get("totalRevenue")) or (yearly[0].get("revenue") if yearly else None)
            ni = _safe_float(info.get("netIncomeToCommon")) or (yearly[0].get("net_income") if yearly else None)
            if rev is not None:
                metrics["Revenu / employé"] = rev / employees
            if ni is not None:
                metrics["Bn net / employé"] = ni / employees

        # Compléter avec les marges / ROE TTM Yahoo si les états annuels sont incomplets
        ttm_map = {
            "ROE": info.get("returnOnEquity"),
            "ROA": info.get("returnOnAssets"),
            "Marge brute": info.get("grossMargins"),
            "Marge exploitation": info.get("operatingMargins"),
            "Marge nette": info.get("profitMargins"),
            "Marge EBITDA": info.get("ebitdaMargins"),
        }
        for key, val in ttm_map.items():
            if metrics.get(key) is None:
                metrics[key] = _safe_float(val)

    except Exception:
        pass

    return metrics


def _growth_as_decimal(value):
    g = _safe_float(value)
    if g is None:
        return None
    if abs(g) > 1:
        return g / 100.0
    return g


def compute_valuation_indicators(ticker, last_price=None):
    """Multiples et rendements de valorisation (capture BFM / Investing)."""
    metrics = {}
    try:
        info = yf.Ticker(ticker).info or {}
        t = Ticker(ticker, timeout=25)
        ks = t.key_stats
        fd = t.financial_data
        if isinstance(ks, dict) and ticker in ks:
            ks = ks[ticker]
        if isinstance(fd, dict) and ticker in fd:
            fd = fd[ticker]
        ks = ks if isinstance(ks, dict) else {}
        fd = fd if isinstance(fd, dict) else {}

        lp = _safe_float(last_price) or _safe_float(
            info.get("currentPrice") or info.get("regularMarketPrice") or fd.get("currentPrice")
        )
        market_cap = _safe_float(info.get("marketCap")) or _safe_float(fd.get("marketCap"))
        ev = _safe_float(info.get("enterpriseValue")) or _safe_float(ks.get("enterpriseValue"))
        if ev is not None and ev <= 0:
            ev = None

        per = _safe_float(info.get("trailingPE")) or _safe_float(ks.get("trailingPE"))
        per_fwd = _safe_float(info.get("forwardPE")) or _safe_float(ks.get("forwardPE"))
        peg = _safe_float(info.get("pegRatio")) or _safe_float(ks.get("pegRatio"))
        pb = _safe_float(info.get("priceToBook")) or _safe_float(ks.get("priceToBook"))
        ps = _safe_float(info.get("priceToSalesTrailing12Months")) or _safe_float(
            ks.get("priceToSalesTrailing12Months")
        )
        ev_ebitda = _safe_float(info.get("enterpriseToEbitda")) or _safe_float(
            ks.get("enterpriseToEbitda")
        )

        ebitda = _safe_float(fd.get("ebitda"))
        fcf = _safe_float(fd.get("freeCashflow")) or _safe_float(info.get("freeCashflow"))
        ocf = _safe_float(fd.get("operatingCashflow")) or _safe_float(info.get("operatingCashflow"))
        shares = _safe_float(info.get("sharesOutstanding"))
        growth = _growth_as_decimal(fd.get("earningsGrowth") or info.get("earningsGrowth"))

        inc = _prep_annual_statement(t.income_statement(frequency="a"), ticker)
        bs = _prep_annual_statement(t.balance_sheet(frequency="a"), ticker)

        metrics["Capitalisation de marché"] = market_cap
        metrics["PER"] = per
        metrics["Ratio PEG"] = peg
        metrics["VE/EBITDA"] = ev_ebitda
        metrics["Cours / ventes"] = ps
        metrics["Cours / val. comptable"] = pb
        metrics["PER forward"] = per_fwd

        if market_cap and ocf and ocf > 0:
            metrics["Cours / flux exploitation"] = market_cap / ocf
        if market_cap and fcf and fcf > 0:
            metrics["Cours / flux disponible"] = market_cap / fcf

        if lp and not bs.empty:
            bs_row = bs.iloc[0]
            equity = _row_get(bs_row, "CommonStockEquity", "StockholdersEquity")
            goodwill = _row_get(bs_row, "Goodwill") or 0.0
            intang = _row_get(bs_row, "OtherIntangibleAssets", "IntangibleAssets") or 0.0
            sh = shares or _row_get(
                inc.iloc[0] if not inc.empty else bs_row,
                "DilutedAverageShares",
                "BasicAverageShares",
            )
            if equity and sh and sh > 0:
                tangible_bps = (equity - goodwill - intang) / sh
                if tangible_bps > 0:
                    metrics["Cours / val. comptable tangible"] = lp / tangible_bps

        if per_fwd and growth and growth > 0:
            metrics["PEG forward"] = per_fwd / (growth * 100.0)
        if ev and ebitda and ebitda > 0 and growth is not None:
            metrics["VE/EBITDA forward"] = ev / (ebitda * (1.0 + growth))

        if per and per > 0:
            metrics["Rendement bénéfices"] = 1.0 / per
        elif lp:
            eps = _safe_float(info.get("trailingEps"))
            if eps and lp > 0:
                metrics["Rendement bénéfices"] = eps / lp

        div_yield = _normalize_yahoo_pct(
            info.get("dividendYield") or info.get("trailingAnnualDividendYield")
        )
        buyback_yield = None
        if len(inc) >= 2:
            sh_cur = _row_get(inc.iloc[0], "DilutedAverageShares", "BasicAverageShares")
            sh_prev = _row_get(inc.iloc[1], "DilutedAverageShares", "BasicAverageShares")
            if sh_cur and sh_prev and sh_prev > 0 and sh_cur < sh_prev:
                buyback_yield = (sh_prev - sh_cur) / sh_prev

        metrics["Rendement rachat"] = buyback_yield
        if div_yield is not None or buyback_yield is not None:
            metrics["Rendement actionnaire"] = (div_yield or 0.0) + (buyback_yield or 0.0)

        if market_cap and fcf and fcf > 0:
            fcf_yield = fcf / market_cap
            metrics["Rendement FCF"] = fcf_yield

        if lp and not inc.empty:
            eps_vals = []
            for i in range(min(3, len(inc))):
                eps = _row_get(inc.iloc[i], "DilutedEPS", "BasicEPS")
                if eps and eps > 0:
                    eps_vals.append(eps)
            if eps_vals and per and per > 0:
                avg_eps = float(np.mean(eps_vals))
                avg_per = lp / avg_eps if avg_eps > 0 else None
                if avg_per and avg_per > 0:
                    metrics["PER % PER moy. 3 ans"] = per / avg_per

        metrics["Valeur entreprise"] = ev
        if not inc.empty:
            ebit = _row_get(inc.iloc[0], "EBIT", "OperatingIncome")
            if ev and ebit and ebit > 0:
                metrics["VE/EBIT"] = ev / ebit
        if ev and fcf and fcf > 0:
            metrics["VE/FCF"] = ev / fcf

    except Exception:
        pass

    return metrics


def get_fundamentals_yfinance(ticker):
    data = {}
    try:
        info = yf.Ticker(ticker).info or {}
        data["per"] = _safe_float(info.get("trailingPE"))
        data["per_forward"] = _safe_float(info.get("forwardPE"))
        data["debt_to_equity"] = _safe_float(info.get("debtToEquity"))
        data["target_mean"] = _safe_float(info.get("targetMeanPrice"))
        data["target_low"] = _safe_float(info.get("targetLowPrice"))
        data["target_high"] = _safe_float(info.get("targetHighPrice"))
        data["recommendation"] = info.get("recommendationKey")
        data["analyst_count"] = _safe_float(info.get("numberOfAnalystOpinions"))

        total_debt = _safe_float(info.get("totalDebt"))
        fcf = _safe_float(info.get("freeCashflow"))
        if total_debt is not None and fcf not in (None, 0):
            data["debt_fcf"] = total_debt / abs(fcf)
        else:
            bs = yf.Ticker(ticker).balance_sheet
            cf = yf.Ticker(ticker).cashflow
            if bs is not None and not bs.empty:
                total_debt = _find_row_value(
                    bs, ["total debt", "long term debt", "total liabilities"]
                )
            if cf is not None and not cf.empty:
                fcf = _find_row_value(
                    cf,
                    [
                        "free cash flow",
                        "free cashflow",
                        "operating cash flow",
                    ],
                )
            if total_debt is not None and fcf not in (None, 0):
                data["debt_fcf"] = total_debt / abs(fcf)
    except Exception:
        pass
    return data


def get_fundamentals_yahooquery(ticker):
    data = {}
    try:
        t = Ticker(ticker, timeout=20)
        ks = t.key_stats
        if isinstance(ks, dict) and ticker in ks:
            ks = ks[ticker]
        if isinstance(ks, dict):
            data["per"] = _safe_float(ks.get("trailingPE"))
            data["per_forward"] = _safe_float(ks.get("forwardPE"))
            data["debt_to_equity"] = _safe_float(ks.get("debtToEquity"))

        fd = t.financial_data
        if isinstance(fd, dict) and ticker in fd:
            fd = fd[ticker]
        if isinstance(fd, dict):
            data["target_mean"] = _safe_float(fd.get("targetMeanPrice"))
            data["target_low"] = _safe_float(fd.get("targetLowPrice"))
            data["target_high"] = _safe_float(fd.get("targetHighPrice"))
            data["recommendation"] = fd.get("recommendationKey")
            data["analyst_count"] = _safe_float(fd.get("numberOfAnalystOpinions"))

        inc = t.income_statement(frequency="a")
        bs = t.balance_sheet(frequency="a")
        cf = t.cash_flow(frequency="a")
        if isinstance(bs, pd.DataFrame) and not bs.empty:
            bs = bs.sort_index()
            total_debt = _find_row_value(
                bs, ["total debt", "long term debt", "total liabilities"]
            )
        else:
            total_debt = None
        if isinstance(cf, pd.DataFrame) and not cf.empty:
            cf = cf.sort_index()
            fcf = _find_row_value(
                cf, ["free cash flow", "free cashflow", "operating cash flow"]
            )
        else:
            fcf = None
        if total_debt is not None and fcf not in (None, 0):
            data["debt_fcf"] = total_debt / abs(fcf)
    except Exception:
        pass
    return data


def merge_fundamentals(primary, secondary):
    merged = {}
    for src in (primary, secondary):
        for key, val in src.items():
            if key not in merged or merged[key] is None:
                if val is not None:
                    merged[key] = val
    return merged


def _normalize_yahoo_pct(value):
    """Convertit une variation Yahoo en décimal (0.015 = 1,5 %)."""
    v = _safe_float(value)
    if v is None:
        return None
    # Yahoo renvoie souvent 1,5 pour +1,5 % (pas 0,015)
    if abs(v) >= 0.5:
        return v / 100.0
    return v


def _normalize_yahoo_change_pct(value):
    """
    regularMarketChangePercent Yahoo : toujours en points de % (-0,37 = -0,37 %, 2,5 = +2,5 %).
    Ne pas réutiliser _normalize_yahoo_pct (seuil 0,5) : les petites variations seraient ×100.
    """
    v = _safe_float(value)
    if v is None:
        return None
    return v / 100.0


def _fix_legacy_daily_var(value):
    """Corrige d'anciennes valeurs stockées ×100 (ex. -0,37 lues comme -37 %)."""
    v = _safe_float(value)
    if v is None:
        return None
    if 0.005 <= abs(v) < 0.5:
        return v / 100.0
    return v


@st.cache_data(ttl=300, show_spinner=False)
def _cached_yahoo_info(ticker):
    try:
        return dict(yf.Ticker(str(ticker)).info or {})
    except Exception:
        return {}


def _daily_var_from_yahoo_or_prices(ticker, price_series=None, last_price=None):
    """Variation du jour : Yahoo (prioritaire) puis clôture veille, puis historique OHLCV."""
    info = _cached_yahoo_info(str(ticker)) if ticker else {}

    daily = _normalize_yahoo_change_pct(info.get("regularMarketChangePercent"))
    if daily is not None:
        return daily

    lp = _safe_float(last_price)
    if lp is None:
        lp = _safe_float(info.get("regularMarketPrice") or info.get("currentPrice"))

    prev_close = _safe_float(info.get("regularMarketPreviousClose") or info.get("previousClose"))
    if lp is not None and prev_close and prev_close > 0:
        return (lp - prev_close) / prev_close

    s = pd.Series(dtype=float)
    if price_series is not None:
        s = pd.to_numeric(price_series, errors="coerce").dropna()
    if lp is None and not s.empty:
        lp = float(s.iloc[-1])
    if len(s) >= 2 and lp is not None:
        prev = float(s.iloc[-2])
        if prev > 0 and (s.index[-1] - s.index[-2]).days <= 4:
            return (lp - prev) / prev
    return None


def _realized_volatility(price_series, trading_days):
    s = pd.to_numeric(price_series, errors="coerce").dropna()
    if len(s) < trading_days + 2:
        return None
    rets = s.pct_change().dropna().tail(trading_days)
    if len(rets) < max(5, trading_days // 2):
        return None
    return float(rets.std() * np.sqrt(252))


def compute_market_indicators(ticker, price_series=None, last_price=None):
    """
    Indicateurs marché / technique (capture BFM) : cours, 52 sem., MA, float, volatilité.
    """
    metrics = {}
    info = {}
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        pass

    lp = _safe_float(last_price)
    if lp is None:
        lp = _safe_float(info.get("regularMarketPrice") or info.get("currentPrice"))

    s = pd.Series(dtype=float)
    if price_series is not None:
        s = pd.to_numeric(price_series, errors="coerce").dropna()

    metrics["Dernier cours"] = lp
    metrics["Prix"] = lp

    daily = _daily_var_from_yahoo_or_prices(ticker, price_series=s, last_price=lp)
    metrics["Var. journalière (%)"] = daily

    window = s.tail(252) if len(s) >= 60 else s
    high_52 = _safe_float(info.get("fiftyTwoWeekHigh"))
    low_52 = _safe_float(info.get("fiftyTwoWeekLow"))
    if not window.empty:
        high_52 = float(window.max()) if len(window) >= 60 else (high_52 or float(window.max()))
        low_52 = float(window.min()) if len(window) >= 60 else (low_52 or float(window.min()))

    metrics["Sommet 52 sem."] = high_52
    metrics["Creux 52 sem."] = low_52
    metrics["Max 52 sem."] = high_52
    metrics["Min 52 sem."] = low_52

    if lp is not None and high_52 not in (None, 0):
        metrics["Var. vs sommet 52 sem. (%)"] = (lp - high_52) / high_52
    if lp is not None and low_52 not in (None, 0):
        metrics["Var. vs creux 52 sem. (%)"] = (lp - low_52) / low_52
    if (
        lp is not None
        and high_52 is not None
        and low_52 is not None
        and high_52 > low_52
    ):
        metrics["Position 52 sem."] = (lp - low_52) / (high_52 - low_52)
    if len(s) >= 252 and lp is not None:
        metrics["Variation 1 an"] = float(lp / float(s.iloc[-252]) - 1)

    if not s.empty:
        sma50 = compute_sma(s, 50)
        sma200 = compute_sma(s, 200)
        if lp is not None and not sma50.empty:
            m50 = float(sma50.iloc[-1])
            if m50 > 0:
                metrics["Prix % MA50"] = lp / m50 * 100.0
        if lp is not None and not sma200.empty:
            m200 = float(sma200.iloc[-1])
            if m200 > 0:
                metrics["Prix % MA200"] = lp / m200 * 100.0

        metrics["Volat. réalisée 30j"] = _realized_volatility(s, 30)
        metrics["Volat. réalisée 90j"] = _realized_volatility(s, 90)
        metrics["Volat. réalisée 1 an"] = _realized_volatility(s, 252)

    shares = _safe_float(info.get("sharesOutstanding"))
    float_sh = _safe_float(info.get("floatShares"))
    metrics["Actions en circulation"] = shares
    if shares and float_sh and shares > 0:
        metrics["Float / actions"] = float_sh / shares

    return metrics


def _refresh_market_fields_from_prices(row, price_series, last_price=None, ticker=None):
    """
    Recalcule les colonnes marché depuis l'historique OHLCV + Yahoo (var. jour).
    Corrige les valeurs en cache obsolètes ou erronées (×100).
    """
    row = dict(row)
    ticker = ticker or row.get("Ticker")
    if price_series is None:
        daily = _daily_var_from_yahoo_or_prices(ticker, last_price=last_price)
        if daily is None:
            daily = _fix_legacy_daily_var(row.get("Var. journalière (%)"))
        if daily is not None:
            row["Var. journalière (%)"] = daily
        return row

    s = pd.to_numeric(price_series, errors="coerce").dropna()
    if s.empty:
        return row

    lp = _safe_float(last_price)
    if lp is None:
        lp = float(s.iloc[-1])
    row["Dernier cours"] = lp
    row["Prix"] = lp

    daily = _daily_var_from_yahoo_or_prices(ticker, price_series=s, last_price=lp)
    if daily is None:
        daily = _fix_legacy_daily_var(row.get("Var. journalière (%)"))
    if daily is not None:
        row["Var. journalière (%)"] = daily

    window = s.tail(252) if len(s) >= 60 else s
    if not window.empty:
        high_52 = float(window.max())
        low_52 = float(window.min())
        row["Sommet 52 sem."] = high_52
        row["Creux 52 sem."] = low_52
        row["Max 52 sem."] = high_52
        row["Min 52 sem."] = low_52
        if high_52 > low_52:
            row["Position 52 sem."] = (lp - low_52) / (high_52 - low_52)
        if high_52 > 0:
            row["Var. vs sommet 52 sem. (%)"] = (lp - high_52) / high_52
        if low_52 > 0:
            row["Var. vs creux 52 sem. (%)"] = (lp - low_52) / low_52

    if len(s) >= 252:
        row["Variation 1 an"] = float(lp / float(s.iloc[-252]) - 1)

    sma50 = compute_sma(s, 50)
    sma200 = compute_sma(s, 200)
    if not sma50.empty:
        m50 = float(sma50.iloc[-1])
        if m50 > 0:
            row["Prix % MA50"] = lp / m50 * 100.0
    if not sma200.empty:
        m200 = float(sma200.iloc[-1])
        if m200 > 0:
            row["Prix % MA200"] = lp / m200 * 100.0

    for days, col in ((30, "Volat. réalisée 30j"), (90, "Volat. réalisée 90j"), (252, "Volat. réalisée 1 an")):
        vol = _realized_volatility(s, days)
        if vol is not None:
            row[col] = vol

    return row


def fetch_fundamentals_for_ticker(ticker, last_price, price_series=None, throttle=True):
    if throttle:
        time.sleep(random.uniform(0.2, 0.5))
    yf_data = get_fundamentals_yfinance(ticker)
    yq_data = get_fundamentals_yahooquery(ticker)
    data = merge_fundamentals(yf_data, yq_data)
    profitability = compute_profitability_indicators(ticker)
    market = compute_market_indicators(ticker, price_series=price_series, last_price=last_price)
    lp = market.get("Dernier cours") or last_price
    valuation = compute_valuation_indicators(ticker, lp)

    if (
        not data
        and not profitability
        and not any(v is not None for v in market.values())
        and not any(v is not None for v in valuation.values())
    ):
        return None

    dte = data.get("debt_to_equity")
    if dte is not None and dte > 10:
        dte = dte / 100

    target = data.get("target_mean")
    upside = None
    if target is not None and lp not in (None, 0):
        upside = (target - lp) / lp

    per = valuation.get("PER") or data.get("per")
    per_fwd = valuation.get("PER forward") or data.get("per_forward")

    row = {
        "Ticker": ticker,
        "Dernier cours": lp,
        "Prix": lp,
        "Var. journalière (%)": market.get("Var. journalière (%)"),
        "Sommet 52 sem.": market.get("Sommet 52 sem."),
        "Creux 52 sem.": market.get("Creux 52 sem."),
        "Var. vs sommet 52 sem. (%)": market.get("Var. vs sommet 52 sem. (%)"),
        "Var. vs creux 52 sem. (%)": market.get("Var. vs creux 52 sem. (%)"),
        "Prix % MA50": market.get("Prix % MA50"),
        "Prix % MA200": market.get("Prix % MA200"),
        "Float / actions": market.get("Float / actions"),
        "Actions en circulation": market.get("Actions en circulation"),
        "Volat. réalisée 30j": market.get("Volat. réalisée 30j"),
        "Volat. réalisée 90j": market.get("Volat. réalisée 90j"),
        "Volat. réalisée 1 an": market.get("Volat. réalisée 1 an"),
        "Max 52 sem.": market.get("Max 52 sem."),
        "Min 52 sem.": market.get("Min 52 sem."),
        "Position 52 sem.": market.get("Position 52 sem."),
        "Variation 1 an": market.get("Variation 1 an"),
        "ROIC": profitability.get("ROIC"),
        "ROIC (moy. 5 ans)": profitability.get("ROIC (moy. 5 ans)"),
        "Score Piotroski": profitability.get("Score Piotroski"),
        "ROE": profitability.get("ROE"),
        "Marge brute": profitability.get("Marge brute"),
        "ROA": profitability.get("ROA"),
        "ROCE": profitability.get("ROCE"),
        "Marge EBITDA": profitability.get("Marge EBITDA"),
        "Rotation actif": profitability.get("Rotation actif"),
        "Rotation stocks": profitability.get("Rotation stocks"),
        "Revenu / employé": profitability.get("Revenu / employé"),
        "Bn net / employé": profitability.get("Bn net / employé"),
        "Rotation créances": profitability.get("Rotation créances"),
        "Marge brute (moy. 5 ans)": profitability.get("Marge brute (moy. 5 ans)"),
        "Marge exploitation": profitability.get("Marge exploitation"),
        "Marge exploitation (moy. 5 ans)": profitability.get("Marge exploitation (moy. 5 ans)"),
        "Marge avant impôt": profitability.get("Marge avant impôt"),
        "Marge avant impôt (moy. 5 ans)": profitability.get("Marge avant impôt (moy. 5 ans)"),
        "Marge nette": profitability.get("Marge nette"),
        "Marge nette (moy. 5 ans)": profitability.get("Marge nette (moy. 5 ans)"),
        "Capitalisation de marché": valuation.get("Capitalisation de marché"),
        "PER": per,
        "Ratio PEG": valuation.get("Ratio PEG"),
        "VE/EBITDA": valuation.get("VE/EBITDA"),
        "Cours / ventes": valuation.get("Cours / ventes"),
        "Cours / flux exploitation": valuation.get("Cours / flux exploitation"),
        "Cours / flux disponible": valuation.get("Cours / flux disponible"),
        "Cours / val. comptable": valuation.get("Cours / val. comptable"),
        "Cours / val. comptable tangible": valuation.get("Cours / val. comptable tangible"),
        "PER forward": per_fwd,
        "PEG forward": valuation.get("PEG forward"),
        "VE/EBITDA forward": valuation.get("VE/EBITDA forward"),
        "Rendement bénéfices": valuation.get("Rendement bénéfices"),
        "Rendement actionnaire": valuation.get("Rendement actionnaire"),
        "Rendement rachat": valuation.get("Rendement rachat"),
        "Rendement FCF": valuation.get("Rendement FCF"),
        "PER % PER moy. 3 ans": valuation.get("PER % PER moy. 3 ans"),
        "Valeur entreprise": valuation.get("Valeur entreprise"),
        "VE/EBIT": valuation.get("VE/EBIT"),
        "VE/FCF": valuation.get("VE/FCF"),
        "Dette / FCF": data.get("debt_fcf"),
        "Dette / Capitaux": dte,
        "Objectif analystes": target,
        "Objectif bas": data.get("target_low"),
        "Objectif haut": data.get("target_high"),
        "Upside vs objectif": upside,
        "Recommandation": data.get("recommendation"),
        "Nb analystes": data.get("analyst_count"),
        "_fetched_at": datetime.now().isoformat(timespec="seconds"),
    }
    return row


@st.cache_data(ttl=86400, show_spinner=False)
def cached_fetch_fundamentals_for_ticker(ticker, last_price_key):
    """Cache mémoire par ticker (complète le cache disque). last_price_key = round(prix, 4) ou -1."""
    lp = None if last_price_key is None or last_price_key < 0 else float(last_price_key)
    return fetch_fundamentals_for_ticker(ticker, lp, throttle=False)


FUNDAMENTALS_PERCENT_COLUMNS = frozenset({
    "ROIC",
    "ROIC (moy. 5 ans)",
    "ROE",
    "ROA",
    "ROCE",
    "Marge brute",
    "Marge brute (moy. 5 ans)",
    "Marge exploitation",
    "Marge exploitation (moy. 5 ans)",
    "Marge avant impôt",
    "Marge avant impôt (moy. 5 ans)",
    "Marge nette",
    "Marge nette (moy. 5 ans)",
    "Marge EBITDA",
    "Upside vs objectif",
    "Var. journalière (%)",
    "Var. vs sommet 52 sem. (%)",
    "Var. vs creux 52 sem. (%)",
    "Volat. réalisée 30j",
    "Volat. réalisée 90j",
    "Volat. réalisée 1 an",
    "Variation 1 an",
    "Float / actions",
    "Rendement bénéfices",
    "Rendement actionnaire",
    "Rendement rachat",
    "Rendement FCF",
    "PER % PER moy. 3 ans",
})

FUNDAMENTALS_LARGE_MONEY_COLUMNS = frozenset({
    "Capitalisation de marché",
    "Valeur entreprise",
})

FUNDAMENTALS_PRICE_COLUMNS = frozenset({
    "Dernier cours",
    "Prix",
    "Sommet 52 sem.",
    "Creux 52 sem.",
    "Objectif analystes",
    "Objectif bas",
    "Objectif haut",
})

FUNDAMENTALS_MA_PCT_COLUMNS = frozenset({"Prix % MA50", "Prix % MA200"})

FUNDAMENTALS_SHARE_COLUMNS = frozenset({"Actions en circulation"})

FUNDAMENTALS_RATIO_COLUMNS = frozenset({
    "Rotation actif",
    "Rotation stocks",
    "Rotation créances",
    "PER",
    "PER forward",
    "Ratio PEG",
    "PEG forward",
    "VE/EBITDA",
    "VE/EBITDA forward",
    "Cours / ventes",
    "Cours / flux exploitation",
    "Cours / flux disponible",
    "Cours / val. comptable",
    "Cours / val. comptable tangible",
    "VE/EBIT",
    "VE/FCF",
    "Dette / FCF",
    "Dette / Capitaux",
})

FUNDAMENTALS_MONEY_COLUMNS = frozenset({"Revenu / employé", "Bn net / employé"})


def _fundamentals_style_format():
    fmt = {
        "Nb analystes": "{:.0f}",
        "Score Piotroski": "{:.0f}",
    }
    for col in FUNDAMENTALS_PRICE_COLUMNS:
        fmt[col] = "{:.2f}"
    for col in FUNDAMENTALS_PERCENT_COLUMNS:
        fmt[col] = "{:.2%}"
    for col in FUNDAMENTALS_MA_PCT_COLUMNS:
        fmt[col] = "{:.1f}"
    for col in FUNDAMENTALS_RATIO_COLUMNS:
        fmt[col] = "{:.2f}"
    for col in FUNDAMENTALS_MONEY_COLUMNS:
        fmt[col] = "{:,.0f}"
    for col in FUNDAMENTALS_LARGE_MONEY_COLUMNS:
        fmt[col] = "{:,.0f}"
    for col in FUNDAMENTALS_SHARE_COLUMNS:
        fmt[col] = "{:,.0f}"
    return fmt


_STYLE_FAVORABLE = "background-color: #d4edda; color: #155724"
_STYLE_UNFAVORABLE = "background-color: #f8d7da; color: #721c24"

_VALUATION_LOWER_IS_BETTER = frozenset({
    "PER",
    "PER forward",
    "Ratio PEG",
    "PEG forward",
    "VE/EBITDA",
    "VE/EBITDA forward",
    "Cours / ventes",
    "Cours / flux exploitation",
    "Cours / flux disponible",
    "Cours / val. comptable",
    "Cours / val. comptable tangible",
    "VE/EBIT",
    "VE/FCF",
})

_VALUATION_YIELD_COLUMNS = frozenset({
    "Rendement bénéfices",
    "Rendement actionnaire",
    "Rendement rachat",
    "Rendement FCF",
})

_FUNDAMENTALS_SENTIMENT_COLUMNS = expand_sentiment_columns(
    frozenset(
        FUNDAMENTALS_PERCENT_COLUMNS
        | FUNDAMENTALS_RATIO_COLUMNS
        | FUNDAMENTALS_MONEY_COLUMNS
        | FUNDAMENTALS_MA_PCT_COLUMNS
        | _VALUATION_YIELD_COLUMNS
        | {"Score Piotroski", "PER % PER moy. 3 ans"}
    ),
    FUNDAMENTALS_LABELS,
)


def _fundamentals_display_format(df):
    fmt = format_map_for_labeled_columns(df, FUNDAMENTALS_LABELS, _fundamentals_style_format())
    fmt.update(pick_format(df, FUNDAMENTALS_EXTRA_FORMAT))
    for col in df.columns:
        internal = internal_column_name(col, FUNDAMENTALS_LABELS)
        if internal in FUNDAMENTALS_RATIO_COLUMNS:
            if col not in fmt or fmt[col] == "{:.2f}":
                fmt[col] = "{:.2f}×"
    return fmt


def _fundamentals_metric_sentiment(column, val):
    """Vert = favorable, rouge = défavorable (repères pédagogiques par colonne)."""
    column = internal_column_name(column, FUNDAMENTALS_LABELS)
    v = _safe_float(val)
    if v is None:
        return ""

    if column in ("ROIC", "ROIC (moy. 5 ans)", "ROCE"):
        if v >= 0.12:
            return _STYLE_FAVORABLE
        if v < 0.05:
            return _STYLE_UNFAVORABLE

    elif column in ("ROE", "ROA"):
        if v >= 0.15 if column == "ROE" else 0.06:
            return _STYLE_FAVORABLE
        if v < 0.02:
            return _STYLE_UNFAVORABLE

    elif column == "Score Piotroski":
        if v >= 7:
            return _STYLE_FAVORABLE
        if v <= 3:
            return _STYLE_UNFAVORABLE

    elif "Marge" in column:
        if v >= 0.15:
            return _STYLE_FAVORABLE
        if v < 0.05:
            return _STYLE_UNFAVORABLE

    elif column in ("Rotation actif", "Rotation stocks", "Rotation créances"):
        if v >= 1.0:
            return _STYLE_FAVORABLE
        if v < 0.5:
            return _STYLE_UNFAVORABLE

    elif column == "Revenu / employé":
        if v >= 250_000:
            return _STYLE_FAVORABLE
        if v < 50_000:
            return _STYLE_UNFAVORABLE

    elif column == "Bn net / employé":
        if v > 0:
            return _STYLE_FAVORABLE
        if v < 0:
            return _STYLE_UNFAVORABLE

    elif column in ("PER", "PER forward"):
        if 0 < v <= 12:
            return _STYLE_FAVORABLE
        if v > 25 or v < 0:
            return _STYLE_UNFAVORABLE

    elif column in ("Ratio PEG", "PEG forward"):
        if 0 < v <= 1.0:
            return _STYLE_FAVORABLE
        if v > 2.0 or v < 0:
            return _STYLE_UNFAVORABLE

    elif column in ("VE/EBITDA", "VE/EBITDA forward"):
        if 0 < v <= 8:
            return _STYLE_FAVORABLE
        if v > 15:
            return _STYLE_UNFAVORABLE

    elif column == "Cours / ventes":
        if 0 < v <= 2:
            return _STYLE_FAVORABLE
        if v > 5:
            return _STYLE_UNFAVORABLE

    elif column in ("Cours / flux exploitation", "Cours / flux disponible"):
        if 0 < v <= 15:
            return _STYLE_FAVORABLE
        if v > 30:
            return _STYLE_UNFAVORABLE

    elif column in ("Cours / val. comptable", "Cours / val. comptable tangible"):
        if 0 < v <= 1.5:
            return _STYLE_FAVORABLE
        if v > 3.5:
            return _STYLE_UNFAVORABLE

    elif column == "VE/EBIT":
        if 0 < v <= 12:
            return _STYLE_FAVORABLE
        if v > 20:
            return _STYLE_UNFAVORABLE

    elif column == "VE/FCF":
        if 0 < v <= 15:
            return _STYLE_FAVORABLE
        if v > 25:
            return _STYLE_UNFAVORABLE

    elif column in _VALUATION_YIELD_COLUMNS:
        if v >= 0.06:
            return _STYLE_FAVORABLE
        if v <= 0.02:
            return _STYLE_UNFAVORABLE

    elif column == "PER % PER moy. 3 ans":
        if v <= 0.90:
            return _STYLE_FAVORABLE
        if v >= 1.10:
            return _STYLE_UNFAVORABLE

    elif column == "Dette / FCF":
        if v <= 3:
            return _STYLE_FAVORABLE
        if v > 8:
            return _STYLE_UNFAVORABLE

    elif column == "Dette / Capitaux":
        if v <= 0.8:
            return _STYLE_FAVORABLE
        if v > 1.5:
            return _STYLE_UNFAVORABLE

    elif column == "Upside vs objectif":
        if v >= 0.10:
            return _STYLE_FAVORABLE
        if v < 0:
            return _STYLE_UNFAVORABLE

    elif column == "Var. journalière (%)":
        if v >= 0.01:
            return _STYLE_FAVORABLE
        if v <= -0.01:
            return _STYLE_UNFAVORABLE

    elif column == "Var. vs sommet 52 sem. (%)":
        if v <= -0.15:
            return _STYLE_FAVORABLE
        if v >= -0.02:
            return _STYLE_UNFAVORABLE

    elif column == "Var. vs creux 52 sem. (%)":
        if v >= 0.30:
            return _STYLE_FAVORABLE
        if v <= 0.05:
            return _STYLE_UNFAVORABLE

    elif column in FUNDAMENTALS_MA_PCT_COLUMNS:
        if 100 <= v <= 105:
            return _STYLE_FAVORABLE
        if v < 95 or v > 115:
            return _STYLE_UNFAVORABLE

    elif column.startswith("Volat. réalisée"):
        if v <= 0.20:
            return _STYLE_FAVORABLE
        if v >= 0.45:
            return _STYLE_UNFAVORABLE

    elif column == "Variation 1 an":
        if v >= 0.10:
            return _STYLE_FAVORABLE
        if v < 0:
            return _STYLE_UNFAVORABLE

    return ""


_NEGATIVE_RATIO_HINTS = {
    "PER": "PER négatif : bénéfice net < 0 (pertes). Le multiple n'est pas comparable à un PER classique.",
    "PER forward": "PER forward négatif : bénéfice anticipé < 0.",
    "VE/EBITDA": "VE/EBITDA négatif : EBITDA ≤ 0 (pas de cash opérationnel avant amortissements).",
    "VE/EBITDA forward": "VE/EBITDA forward négatif : EBITDA anticipé ≤ 0.",
    "VE/EBIT": "VE/EBIT négatif : résultat opérationnel ≤ 0.",
    "VE/FCF": "VE/FCF négatif : free cash-flow ≤ 0.",
    "Ratio PEG": "PEG négatif ou peu fiable : croissance ou bénéfice négatif.",
    "PEG forward": "PEG forward négatif : idem sur anticipations.",
    "Cours / val. comptable": "P/B négatif : fonds propres comptables < 0.",
    "Cours / val. comptable tangible": "P/B tangible négatif : fonds propres tangibles < 0.",
}

_FUNDAMENTALS_COLUMN_HELP = {
    "Collecte fondamentaux": "Date/heure du dernier téléchargement des ratios comptables (Yahoo). Cache disque 24 h.",
    "Notes interprétation": "Alertes sur ratios non comparables (pertes, EBITDA négatif, etc.).",
    "Var. journalière (%)": "Variation vs clôture veille (Yahoo). Distinct de la volatilité 1 an.",
    "Volat. réalisée 1 an (%, ann.)": "Amplitude des mouvements quotidiens annualisée — ce n'est pas la variation du jour.",
    "PER (×)": "Cours ÷ bénéfice par action. Négatif si l'entreprise est déficitaire.",
    "PER forward (×)": "Cours ÷ bénéfice anticipé. Négatif si pertes anticipées.",
    "VE/EBITDA (×)": "Valeur d'entreprise ÷ EBITDA. Négatif si EBITDA ≤ 0.",
    "VE/EBITDA forward (×)": "Idem sur EBITDA anticipé.",
}


def _format_collecte_timestamp(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    try:
        return datetime.fromisoformat(str(value)).strftime("%d/%m/%Y %H:%M")
    except (TypeError, ValueError):
        return str(value)[:16]


def _fundamental_metric_hint(column, raw_value):
    column = internal_column_name(column, FUNDAMENTALS_LABELS)
    v = _safe_float(raw_value)
    if v is None:
        return None
    if column in _NEGATIVE_RATIO_HINTS and v < 0:
        return _NEGATIVE_RATIO_HINTS[column]
    if "Marge" in column and v < -0.5:
        return "Marge très négative : pertes nettes ou opérationnelles >> chiffre d'affaires."
    if column == "ROE" and v < 0:
        return "ROE négatif : fonds propres ou résultat net négatif sur la période."
    return None


def _fundamental_row_notes(row):
    notes = []
    if hasattr(row, "get"):
        getter = row.get
    else:
        getter = lambda k, default=None: row[k] if k in row.index else default

    for col, hint in _NEGATIVE_RATIO_HINTS.items():
        v = _safe_float(getter(col))
        if v is not None and v < 0:
            short = hint.split(" : ", 1)[0]
            notes.append(short)

    for col in ("Marge nette", "Marge EBITDA", "Marge exploitation"):
        v = _safe_float(getter(col))
        if v is not None and v < -0.5:
            notes.append("Marges très négatives (pertes >> CA)")
            break

    roe = _safe_float(getter("ROE"))
    if roe is not None and roe < 0:
        notes.append("ROE négatif")

    if not notes:
        return "—"
    return " · ".join(dict.fromkeys(notes))


def _fundamentals_table_column_config(df_display):
    config = {}
    for col in df_display.columns:
        help_text = _FUNDAMENTALS_COLUMN_HELP.get(col)
        if help_text:
            config[col] = st.column_config.TextColumn(help=help_text)
    return config


def _format_fundamental_value(col, val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "-"
    if col in FUNDAMENTALS_PERCENT_COLUMNS:
        return f"{val:.2%}"
    if col in FUNDAMENTALS_LARGE_MONEY_COLUMNS:
        return f"{val:,.0f}"
    if col in FUNDAMENTALS_MONEY_COLUMNS:
        return f"{val:,.0f}"
    if col in FUNDAMENTALS_SHARE_COLUMNS:
        return f"{val:,.0f}"
    if col in FUNDAMENTALS_PRICE_COLUMNS:
        return f"{val:.2f}"
    if col in FUNDAMENTALS_MA_PCT_COLUMNS:
        return f"{val:.1f}"
    if col == "Score Piotroski":
        return f"{val:.0f} / 9"
    if col in FUNDAMENTALS_RATIO_COLUMNS:
        return f"{val:.2f}"
    return f"{val:.2f}"


def _render_fundamental_metric(label, column, raw_value, formatted=None):
    """Affiche une métrique Streamlit avec code couleur vert / rouge."""
    formatted = formatted if formatted is not None else _format_fundamental_value(column, raw_value)
    style = _fundamentals_metric_sentiment(column, raw_value)
    if not style:
        st.metric(label, formatted)
        hint = _fundamental_metric_hint(column, raw_value)
        if hint:
            st.caption(hint)
        return
    st.markdown(
        f'<div style="margin-bottom: 0.35rem;">'
        f'<div style="font-size: 0.875rem; line-height: 1.25; color: rgb(49, 51, 63); opacity: 0.8;">'
        f"{label}</div>"
        f'<div style="font-size: 1.75rem; font-weight: 600; line-height: 1.2; {style} '
        f'padding: 0.2rem 0.55rem; border-radius: 0.35rem; display: inline-block;">'
        f"{formatted}</div></div>",
        unsafe_allow_html=True,
    )
    hint = _fundamental_metric_hint(column, raw_value)
    if hint:
        st.caption(hint)


def _render_fundamentals_detail_table(row, cols):
    """Tableau détaillé avec code couleur par indicateur."""
    entries = []
    for col in cols:
        if col not in row.index:
            continue
        raw = row[col]
        hint = _fundamental_metric_hint(col, raw)
        entries.append(
            {
                "Indicateur": FUNDAMENTALS_LABELS.get(col, col),
                "Valeur": _format_fundamental_value(col, raw),
                "Interprétation": hint or "—",
                "_col": col,
                "_raw": raw,
            }
        )
    if not entries:
        return
    detail_df = pd.DataFrame(entries)

    def _color_detail_row(series):
        idx = series.name
        style = _fundamentals_metric_sentiment(
            detail_df.loc[idx, "_col"],
            detail_df.loc[idx, "_raw"],
        )
        return ["", style, ""]

    styled = detail_df[["Indicateur", "Valeur", "Interprétation"]].style.apply(
        _color_detail_row, axis=1
    )
    st.dataframe(styled, hide_index=True, use_container_width=True)


def _style_fundamentals_sentiment(styler):
    def _color_column(series):
        if series.name not in _FUNDAMENTALS_SENTIMENT_COLUMNS:
            return [""] * len(series)
        return [_fundamentals_metric_sentiment(series.name, v) for v in series]

    return styler.apply(_color_column, axis=0)


style_fundamentals_sentiment = _style_fundamentals_sentiment

SUGGESTION_QUALITY_COLUMNS = (
    "Score Piotroski",
    "ROE",
    "ROIC",
    "Marge nette (moy. 5 ans)",
    "PER",
    "Ratio PEG",
    "Dette / Capitaux",
    "Dette / FCF",
    "Upside vs objectif",
    "Recommandation",
)


def fetch_fundamentals_for_tickers(
    tickers,
    prices,
    cache=None,
    progress_cb=None,
    use_disk_cache=True,
):
    """Collecte les fondamentaux (session + disque 24 h, puis Yahoo)."""
    cache = cache if cache is not None else {}
    rows = []
    stats = {"session": 0, "disk": 0, "fetched": 0, "missing": 0}
    tickers = [t for t in dict.fromkeys(tickers) if t]
    for i, t in enumerate(tickers):
        if progress_cb:
            progress_cb(t, i + 1, len(tickers), stats)
        cached = cache.get(t)
        if cached and _row_is_fresh(cached):
            rows.append(cached)
            stats["session"] += 1
            continue
        if use_disk_cache:
            disk_row = load_fundamental_ticker_from_disk(t)
            if disk_row is not None:
                cache[t] = disk_row
                rows.append(disk_row)
                stats["disk"] += 1
                continue
        lp = None
        price_series = None
        if prices is not None and t in prices.columns:
            price_series = prices[t]
            s = price_series.dropna()
            if len(s):
                lp = float(s.iloc[-1])
        row = fetch_fundamentals_for_ticker(t, lp, price_series=price_series)
        if row is not None:
            cache[t] = row
            if use_disk_cache:
                save_fundamental_ticker_to_disk(t, row)
            rows.append(row)
            stats["fetched"] += 1
        else:
            stats["missing"] += 1
    return rows, cache, stats


def merge_fundamentals_columns(df, fundamentals_rows, columns=None):
    """Joint les indicateurs fondamentaux au tableau de suggestions (clé Ticker)."""
    if df is None or df.empty or not fundamentals_rows:
        return df
    columns = columns or SUGGESTION_QUALITY_COLUMNS
    fund_df = pd.DataFrame(fundamentals_rows)
    if fund_df.empty or "Ticker" not in fund_df.columns:
        return df
    keep = ["Ticker"] + [c for c in columns if c in fund_df.columns]
    fund_df = fund_df[keep].drop_duplicates(subset=["Ticker"], keep="last")
    merged = df.merge(fund_df, on="Ticker", how="left")
    return merged


def filter_suggestions_by_quality(
    df,
    min_piotroski=0,
    min_roe=0.0,
    max_per=None,
    max_debt_equity=None,
):
    """Filtre les suggestions selon des seuils de qualité fondamentale."""
    if df is None or df.empty:
        return df
    out = df.copy()
    if min_piotroski > 0 and "Score Piotroski" in out.columns:
        out = out[out["Score Piotroski"].fillna(-1) >= min_piotroski]
    if min_roe > 0 and "ROE" in out.columns:
        out = out[out["ROE"].fillna(-1) >= min_roe]
    if max_per is not None and "PER" in out.columns:
        out = out[
            out["PER"].isna()
            | ((out["PER"] > 0) & (out["PER"] <= max_per))
        ]
    if max_debt_equity is not None and "Dette / Capitaux" in out.columns:
        out = out[
            out["Dette / Capitaux"].isna()
            | (out["Dette / Capitaux"] <= max_debt_equity)
        ]
    return out


def _render_fundamentals_legend():
    """Légende pédagogique des ratios et du consensus analystes."""
    with st.expander("📖 Légende — que signifient ces indicateurs ?", expanded=False):
        st.markdown(
            """
Données issues de **Yahoo Finance** / **YahooQuery** (sans compte). Les seuils ci-dessous
sont des **repères pédagogiques**, pas des règles absolues — à adapter au secteur
(utilités, tech, banques…).

---

### Marché & technique (cours, 52 semaines, volatilité)

| Terme | Signification | Lecture simplifiée |
|-------|---------------|-------------------|
| **Dernier cours** | Dernier prix de clôture disponible | Base de tous les ratios « au cours actuel ». |
| **Var. journalière (%)** | Variation du jour | Hausse/baisse par rapport à la veille. |
| **Sommet / Creux 52 sem.** | Plus haut / plus bas sur ~1 an | Fourchette récente de négociation (historique chargé ou Yahoo). |
| **Var. vs sommet / creux 52 sem.** | Écart du cours actuel vs extrêmes | Proche du **sommet** = titre déjà fort ; au-dessus du **creux** = rebond depuis le plancher. |
| **Prix % MA50 / MA200** | Cours ÷ moyenne mobile (× 100) | **100** = sur la MM ; **> 100** = au-dessus (tendance haussière) ; **< 100** = en dessous. |
| **Float / actions** | Part des titres réellement échangeables | Proche de **100 %** = flottant élevé ; faible = titres « bloqués » (fondateurs, État…). |
| **Actions en circulation** | Nombre total d'actions émises | Utile pour la capitalisation (cours × actions). |
| **Volat. réalisée 30j / 90j / 1 an** | Écart-type des rendements quotidiens, annualisé | Mesure l'**amplitude réelle** des mouvements ; faible = titre calme. **≠ var. journalière.** |

| **Collecte fondamentaux** | Date du dernier téléchargement Yahoo (ratios comptables) | Cache **24 h** ; cours / var. jour recalculés à chaque affichage. |
| **Notes interprétation** | Alertes par titre | Signale PER, VE/EBITDA ou marges **non comparables** (souvent pertes). |

### Ratios négatifs — comment les lire

| Ratio négatif | Signification habituelle |
|---------------|-------------------------|
| **PER / PER forward < 0** | Bénéfice (ou anticipé) **négatif** — pas « bon marché » au sens classique. |
| **VE/EBITDA < 0** | **EBITDA ≤ 0** : pas de base de cash opérationnel avant D&A. |
| **P/B < 0** | **Fonds propres comptables négatifs** (passif > actif côté capitaux propres). |
| **Marge << −100 %** | Pertes **supérieures au chiffre d'affaires** sur la période (fréquent en biotech / early stage). |

*Historique recommandé : ≥ 1 an pour les extrêmes 52 semaines et la volatilité 1 an ; ≥ 200 séances pour la MM200.*

---

### Rentabilité et efficacité (écran fondamental)

| Terme | Signification | Lecture simplifiée |
|-------|---------------|-------------------|
| **ROIC** | *Return On Invested Capital* — rentabilité du capital investi | **> 12 %** souvent bon ; mesure la création de valeur économique. |
| **ROIC (moy. 5 ans)** | Moyenne du ROIC sur 5 exercices | Lisse les effets d'un mauvais ou bonne année isolée. |
| **Score Piotroski** | Score **F** (0 à 9) : solidité financière | **≥ 7** = profil robuste ; **≤ 3** = signaux faibles (9 critères comptables). |
| **ROE** | Bénéfice net ÷ capitaux propres | **> 15 %** souvent solide pour l'actionnaire. |
| **ROA** | Bénéfice net ÷ actif total | Efficacité globale de l'actif. |
| **ROCE** | EBIT ÷ capital employé | Rentabilité du capital exploité (hors dette fournisseurs). |
| **Marge brute / exploitation / nette / EBITDA** | Profit ÷ chiffre d'affaires | Plus la marge est haute, plus le modèle économique est « rentable ». |
| **Marges (moy. 5 ans)** | Moyennes sur 5 ans | Stabilité du modèle économique dans le temps. |
| **Rotation actif** | CA ÷ actif total | Efficacité d'utilisation des actifs (×/an). |
| **Rotation stocks** | Coût des ventes ÷ stocks | Vitesse d'écoulement des stocks ; **N/A** fréquent (services sans stock). |
| **Rotation créances** | CA ÷ créances clients | Rapidité d'encaissement des clients. |
| **Revenu / employé** | CA ÷ effectif | Productivité commerciale (approx. TTM / dernier exercice). |
| **Bn net / employé** | Résultat net ÷ effectif | Profitabilité par salarié. |

---

### Code couleur du tableau (vert / rouge)

Les cellules sont colorées automatiquement selon des **seuils indicatifs** (à adapter au secteur) :

| Couleur | Signification |
|---------|---------------|
| **Vert** | Profil **favorable** sur cet indicateur |
| **Rouge** | Signal **défavorable** ou de vigilance |
| *Blanc* | Zone intermédiaire ou donnée absente |

| Indicateur | Vert si… | Rouge si… |
|------------|----------|-----------|
| ROIC / ROIC moy. 5 ans / ROCE | ≥ 12 % | < 5 % |
| ROE | ≥ 15 % | < 2 % |
| ROA | ≥ 6 % | < 2 % |
| Score Piotroski | ≥ 7 / 9 | ≤ 3 / 9 |
| Marges (brute, exploitation, nette…) | ≥ 15 % | < 5 % |
| Rotations (actif, stocks, créances) | ≥ 1,0× | < 0,5× |
| Revenu / employé | ≥ 250 k | < 50 k |
| Bn net / employé | > 0 | < 0 |
| Dette / FCF | ≤ 3 | > 8 |
| Dette / Capitaux | ≤ 0,8 | > 1,5 |
| Upside vs objectif | ≥ +10 % | < 0 % |
| Var. journalière | ≥ +1 % | ≤ −1 % |
| Var. vs sommet 52 sem. | ≤ −15 % (éloigné du plus haut) | ≥ −2 % (proche du plus haut) |
| Var. vs creux 52 sem. | ≥ +30 % | ≤ +5 % (proche du plus bas) |
| Prix % MA50 / MA200 | 100–105 | < 95 ou > 115 |
| Volat. réalisée | ≤ 20 % | ≥ 45 % |
| PER / PER forward | 0 < PER ≤ 12 | PER > 25 |
| Ratio / PEG forward | ≤ 1 | > 2 |
| VE/EBITDA (et forward) | ≤ 8 | > 15 |
| Cours / ventes | ≤ 2 | > 5 |
| Cours / flux (expl. ou dispo.) | ≤ 15 | > 30 |
| Cours / val. comptable (tangible) | ≤ 1,5 | > 3,5 |
| VE/EBIT | ≤ 12 | > 20 |
| VE/FCF | ≤ 15 | > 25 |
| Rendements (bénéfices, FCF, actionnaire…) | ≥ 6 % | ≤ 2 % |
| PER % PER moy. 3 ans | ≤ 90 % du moyen | ≥ 110 % |

---

### Valorisation — multiples et rendements

| Terme | Formule / idée | Lecture simplifiée |
|-------|----------------|-------------------|
| **Capitalisation de marché** | Cours × actions en circulation | Taille boursière de l'entreprise. |
| **PER** | Prix ÷ bénéfice par action | **< 12** souvent bas ; **> 25** élevé. |
| **Ratio PEG** | PER ÷ croissance des bénéfices (%) | **< 1** = croissance « bon marché ». |
| **VE/EBITDA** | Valeur d'entreprise ÷ EBITDA | **< 8** attractif ; **> 15** cher (dette incluse). |
| **Cours / ventes** | Capitalisation ÷ chiffre d'affaires | Multiple de revenus — varie fortement par secteur. |
| **Cours / flux exploitation** | Cap. boursière ÷ flux opérationnel | Combien le marché paie pour le cash opérationnel. |
| **Cours / flux disponible** | Cap. boursière ÷ free cash flow | Variante « cash réel » après investissements. |
| **Cours / val. comptable** | Prix ÷ valeur comptable par action (*P/B*) | **< 1,5** peut signaler décote comptable. |
| **Cours / val. comptable tangible** | Idem hors goodwill & incorporels | Plus strict pour les entreprises acquisitives. |
| **PER forward / PEG forward / VE/EBITDA forward** | Multiples basés sur estimations | Vision **prospective** (croissance attendue). |
| **Rendement bénéfices** | 1 ÷ PER (ou EPS ÷ prix) | Inverse du PER — **> 6 %** peut être attractif. |
| **Rendement actionnaire** | Dividende + rachats d'actions (approx.) | Cash total retourné à l'actionnaire. |
| **Rendement rachat** | Baisse du nombre d'actions (approx.) | Politique de rachat — accroît le BPA. |
| **Rendement FCF** | Free cash flow ÷ capitalisation | Rendement « cash » de l'action. |
| **PER % PER moy. 3 ans** | PER actuel ÷ PER moyen (3 ans d'EPS) | **< 90 %** = moins cher que sa moyenne récente. |
| **Valeur entreprise (VE)** | Cap. boursière + dette nette | Valeur totale pour un repreneur. |
| **VE/EBIT** | VE ÷ résultat d'exploitation | Multiple opérationnel hors amortissements. |
| **VE/FCF** | VE ÷ free cash flow | Prix payé pour le cash disponible (dette incluse). |

---

### Endettement (le risque financier)

| Terme | Signification | Lecture simplifiée |
|-------|---------------|-------------------|
| **Dette / FCF** | Dette totale ÷ free cash flow | Combien d'années de FCF pour rembourser la dette. **< 3** confortable ; **> 8** risque élevé si le FCF baisse. |
| **Dette / Capitaux** (*Debt to Equity*) | Dette ÷ capitaux propres | Levier financier. **> 1** = plus de dette que de fonds propres ; comparer au secteur. |

---

### Consensus analystes (opinion du marché)

| Terme | Signification |
|-------|---------------|
| **Objectif analystes** | Cours cible **moyen** sur 12 mois (consensus). |
| **Objectif bas / haut** | Fourchette des estimations les plus pessimistes / optimistes. |
| **Upside vs objectif** | `(Objectif moyen − prix actuel) ÷ prix actuel`. **+20 %** = les analystes voient **20 %** de hausse potentielle ; **négatif** = cours au-dessus du consensus. |
| **Recommandation** | Consensus textuel Yahoo : *strong buy*, *buy*, *hold*, *sell*… |
| **Nb analystes** | Nombre d'analystes ayant contribué — plus il est élevé, plus le consensus est **robuste**. |

---

### Graphique « fourchette d'objectifs »

Le graphique en barres compare le **prix actuel** (trait pointillé) aux objectifs **bas / moyen / haut**.
Si le prix est **au-dessus** de l'objectif moyen, le titre est déjà « dans les prix » selon le consensus.
            """
        )
        st.caption(
            "Marché : historique des cours chargé + Yahoo (float, 52 sem.). "
            "Rentabilité : états financiers annuels (moyennes 5 ans + Piotroski). "
            "Valorisation / consensus : ratios et objectifs analystes. "
            "Couleurs : repères automatiques — section « Code couleur ». "
            "Valeurs absentes (-) si historique ou donnée insuffisants."
        )


@st.fragment
def render_fundamentals_dashboard(prices):
    st.subheader("Analyse fondamentale & consensus analystes")
    _render_fundamentals_legend()
    tickers = [str(t).strip() for t in prices.columns]
    last_prices = prices.iloc[-1]
    signature = "|".join(tickers)
    cache_key = f"data_funda_{abs(hash(signature))}"

    if cache_key not in st.session_state:
        st.session_state[cache_key] = []

    if _hydrate_fundamentals_from_disk(cache_key, signature):
        updated = st.session_state.get(f"{cache_key}_disk_loaded", "")
        st.success(
            f"📁 {len(st.session_state[cache_key])} résultat(s) rechargé(s) depuis le cache local "
            f"(dernière analyse : {updated})."
        )
        st.caption(f"Fichier cache : `{_fundamentals_cache_path()}`")

    max_default = min(30, len(tickers))
    max_to_analyze = st.slider(
        "Nombre max de tickers a analyser",
        min_value=5 if len(tickers) >= 5 else len(tickers),
        max_value=len(tickers),
        value=max_default,
        step=5 if len(tickers) >= 10 else 1,
    )
    batch_size = st.selectbox("Taille de lot", [5, 10, 15, 20], index=1)
    limited = tickers[:max_to_analyze]
    total_batches = max(1, int(np.ceil(len(limited) / batch_size)))

    batch_key = f"{cache_key}_batch"
    if batch_key not in st.session_state:
        st.session_state[batch_key] = 0
    st.session_state[batch_key] = min(st.session_state[batch_key], total_batches - 1)
    batch_idx = st.session_state[batch_key]

    start_i = batch_idx * batch_size
    end_i = min(start_i + batch_size, len(limited))
    batch = limited[start_i:end_i]
    rows = st.session_state[cache_key]
    known = _fundamentals_rows_index(rows)
    cached_in_batch = sum(1 for t in batch if t in known and _row_is_fresh(known[t]))
    stale_in_batch = sum(1 for t in batch if t in known and not _row_is_fresh(known[t]))
    missing_in_batch = sum(1 for t in batch if t not in known)

    st.caption(
        f"Lot {batch_idx + 1}/{total_batches} — tickers {start_i + 1} à {end_i} sur {len(limited)} · "
        f"cache : {len(rows)}/{len(limited)} titres · "
        f"lot : {cached_in_batch} à jour, {stale_in_batch} obsolètes (>24 h), {missing_in_batch} manquants."
    )
    st.caption(
        "Les données sont **conservées en session et sur disque (24 h)**. "
        "Relancez seulement les titres **manquants ou obsolètes**, sauf si vous forcez la mise à jour."
    )

    force_refresh = st.checkbox(
        "Forcer la mise à jour du lot (ignorer le cache)",
        value=False,
        help="Re-télécharge Yahoo pour tous les tickers du lot, même s'ils sont déjà en cache.",
        key="funda_force_refresh",
    )

    c1, c2, c3 = st.columns(3)
    analyze = c1.button("Analyser / compléter ce lot", key="funda_analyze")
    next_lot = c2.button("Lot suivant", key="funda_next")
    reset = c3.button("Vider le cache", key="funda_reset")

    if next_lot and batch_idx < total_batches - 1:
        st.session_state[batch_key] = batch_idx + 1
        st.rerun()
    if reset:
        clear_fundamentals_disk_cache(signature)
        try:
            from dashboard_cache import clear_dashboard_compute_cache

            clear_dashboard_compute_cache()
        except ImportError:
            pass
        cached_fetch_fundamentals_for_ticker.clear()
        if cache_key in st.session_state:
            del st.session_state[cache_key]
        disk_loaded_key = f"{cache_key}_disk_loaded"
        if disk_loaded_key in st.session_state:
            del st.session_state[disk_loaded_key]
        if batch_key in st.session_state:
            del st.session_state[batch_key]
        st.rerun()

    if analyze and batch:
        rows = list(st.session_state[cache_key])
        known = _fundamentals_rows_index(rows)
        status = st.empty()
        progress = st.progress(0)
        ok, fail, skipped = 0, 0, 0

        for i, t in enumerate(batch):
            existing = known.get(t)
            if existing and not force_refresh and _row_is_fresh(existing):
                skipped += 1
                progress.progress((i + 1) / len(batch))
                continue
            status.text(f"Collecte fondamentale : {t} ({i + 1}/{len(batch)})")
            lp = float(last_prices[t]) if not np.isnan(last_prices[t]) else None
            row = fetch_fundamentals_for_ticker(
                t, lp, price_series=prices[t] if t in prices.columns else None
            )
            if row is not None:
                rows = [r for r in rows if r.get("Ticker") != t]
                rows.append(row)
                known[t] = row
                ok += 1
            else:
                fail += 1
            progress.progress((i + 1) / len(batch))

        st.session_state[cache_key] = rows
        save_fundamentals_rows_to_disk(signature, rows)
        status.empty()
        progress.empty()
        st.info(
            f"Lot terminé — récupérés : {ok}, ignorés (cache valide) : {skipped}, échecs : {fail}. "
            f"💾 Enregistré : `{_fundamentals_cache_path()}`"
        )

    data = st.session_state[cache_key]
    if not data:
        st.warning(
            "Aucune donnée fondamentale pour l'instant. Cliquez sur **Analyser / compléter ce lot** "
            "pour lancer la collecte (une seule fois suffit — le cache est réutilisé ensuite)."
        )
        return

    refreshed = []
    for row in data:
        ticker = row.get("Ticker")
        if ticker in prices.columns:
            lp = last_prices.get(ticker)
            lp = float(lp) if lp is not None and not pd.isna(lp) else None
            refreshed.append(
                _refresh_market_fields_from_prices(
                    row, prices[ticker], last_price=lp, ticker=ticker
                )
            )
        else:
            refreshed.append(dict(row))
    data = refreshed

    df = pd.DataFrame(data)
    if "_fetched_at" in df.columns:
        df["Collecte Yahoo"] = df["_fetched_at"].apply(_format_collecte_timestamp)
        df = df.drop(columns=["_fetched_at"])
    else:
        df["Collecte Yahoo"] = "—"
    df["Notes interprétation"] = df.apply(_fundamental_row_notes, axis=1)
    meta_cols = ["Ticker", "Collecte Yahoo", "Notes interprétation"]
    other_cols = [c for c in df.columns if c not in meta_cols]
    df = df[meta_cols + other_cols]
    sort_options = [
        "PER",
        "VE/EBITDA",
        "Rendement FCF",
        "Ratio PEG",
        "Var. journalière (%)",
        "Var. vs creux 52 sem. (%)",
        "ROIC",
        "ROE",
        "Score Piotroski",
        "Marge nette",
        "Volat. réalisée 1 an",
        "Dette / FCF",
        "Upside vs objectif",
        "Objectif analystes",
    ]
    sort_options = [c for c in sort_options if c in df.columns]
    sort_col = st.selectbox("Trier par", sort_options, index=min(2, len(sort_options) - 1))
    ascending = sort_col in (
        "PER",
        "PER forward",
        "Ratio PEG",
        "PEG forward",
        "VE/EBITDA",
        "VE/EBITDA forward",
        "Cours / ventes",
        "Cours / flux exploitation",
        "Cours / flux disponible",
        "Cours / val. comptable",
        "Cours / val. comptable tangible",
        "VE/EBIT",
        "VE/FCF",
        "Dette / FCF",
        "Dette / Capitaux",
        "Volat. réalisée 30j",
        "Volat. réalisée 90j",
        "Volat. réalisée 1 an",
        "PER % PER moy. 3 ans",
    )
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=ascending, na_position="last")

    df_display = rename_columns_for_display(df, FUNDAMENTALS_LABELS)
    styled = _style_fundamentals_sentiment(
        df_display.style.format(_fundamentals_display_format(df_display), na_rep="-")
    )
    st.caption(FUNDAMENTALS_CAPTION)
    st.caption(
        "Couleurs : vert = favorable, rouge = défavorable (seuils détaillés dans la légende 📖). "
        "Survolez les en-têtes **PER**, **VE/EBITDA**, **Collecte** pour l'aide contextuelle."
    )
    st.dataframe(
        styled,
        use_container_width=True,
        column_config=_fundamentals_table_column_config(df_display),
    )

    csv = df_display.to_csv(index=False).encode("utf-8")
    st.download_button("Telecharger (CSV)", csv, "analyse_fondamentale.csv")

    st.markdown("---")
    detail = st.selectbox("Detail par actif", df["Ticker"])
    row = df[df["Ticker"] == detail].iloc[0]
    if row.get("Notes interprétation") not in (None, "—", ""):
        st.info(f"**{detail}** — {row['Notes interprétation']}")
    if row.get("Collecte Yahoo") not in (None, "—", ""):
        st.caption(
            f"Collecte fondamentaux Yahoo : **{row['Collecte Yahoo']}** · "
            "Cours, var. journalière et volatilités : recalculés à l'affichage."
        )
    st.caption(
        "Code couleur par indicateur : **vert** = repère favorable · **rouge** = repère "
        "défavorable (seuils dans la légende 📖). Sans couleur = neutre ou non applicable."
    )

    st.markdown("#### Marché & technique")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        _render_fundamental_metric("Dernier cours", "Dernier cours", row.get("Dernier cours"))
    with m2:
        _render_fundamental_metric(
            "Var. jour", "Var. journalière (%)", row.get("Var. journalière (%)")
        )
    with m3:
        _render_fundamental_metric("Prix % MA200", "Prix % MA200", row.get("Prix % MA200"))
    with m4:
        _render_fundamental_metric(
            "Volat. 1 an", "Volat. réalisée 1 an", row.get("Volat. réalisée 1 an")
        )

    with st.expander("Tous les indicateurs marché", expanded=False):
        market_cols = [
            "Dernier cours",
            "Var. journalière (%)",
            "Sommet 52 sem.",
            "Creux 52 sem.",
            "Var. vs sommet 52 sem. (%)",
            "Var. vs creux 52 sem. (%)",
            "Prix % MA50",
            "Prix % MA200",
            "Float / actions",
            "Actions en circulation",
            "Volat. réalisée 30j",
            "Volat. réalisée 90j",
            "Volat. réalisée 1 an",
        ]
        _render_fundamentals_detail_table(row, market_cols)

    st.markdown("#### Rentabilité & efficacité")
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        _render_fundamental_metric("ROIC", "ROIC", row.get("ROIC"))
    with r2:
        _render_fundamental_metric("ROE", "ROE", row.get("ROE"))
    with r3:
        _render_fundamental_metric(
            "Score Piotroski", "Score Piotroski", row.get("Score Piotroski")
        )
    with r4:
        _render_fundamental_metric("Marge nette", "Marge nette", row.get("Marge nette"))

    with st.expander("Tous les ratios rentabilité", expanded=False):
        prof_cols = [
            "ROIC",
            "ROIC (moy. 5 ans)",
            "Score Piotroski",
            "ROE",
            "ROA",
            "ROCE",
            "Marge brute",
            "Marge brute (moy. 5 ans)",
            "Marge exploitation",
            "Marge exploitation (moy. 5 ans)",
            "Marge avant impôt",
            "Marge avant impôt (moy. 5 ans)",
            "Marge nette",
            "Marge nette (moy. 5 ans)",
            "Marge EBITDA",
            "Rotation actif",
            "Rotation stocks",
            "Rotation créances",
            "Revenu / employé",
            "Bn net / employé",
        ]
        _render_fundamentals_detail_table(row, prof_cols)

    st.markdown("#### Valorisation (multiples & rendements)")
    v1, v2, v3, v4 = st.columns(4)
    with v1:
        _render_fundamental_metric("PER", "PER", row.get("PER"))
    with v2:
        _render_fundamental_metric("VE/EBITDA", "VE/EBITDA", row.get("VE/EBITDA"))
    with v3:
        _render_fundamental_metric("Rendement FCF", "Rendement FCF", row.get("Rendement FCF"))
    with v4:
        _render_fundamental_metric("Ratio PEG", "Ratio PEG", row.get("Ratio PEG"))

    with st.expander("Tous les multiples de valorisation", expanded=False):
        val_cols = [
            "Capitalisation de marché",
            "PER",
            "Ratio PEG",
            "VE/EBITDA",
            "Cours / ventes",
            "Cours / flux exploitation",
            "Cours / flux disponible",
            "Cours / val. comptable",
            "Cours / val. comptable tangible",
            "PER forward",
            "PEG forward",
            "VE/EBITDA forward",
            "Rendement bénéfices",
            "Rendement actionnaire",
            "Rendement rachat",
            "Rendement FCF",
            "PER % PER moy. 3 ans",
            "Valeur entreprise",
            "VE/EBIT",
            "VE/FCF",
        ]
        _render_fundamentals_detail_table(row, val_cols)

    st.markdown("#### Consensus analystes")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _render_fundamental_metric("PER", "PER", row.get("PER"))
    with c2:
        _render_fundamental_metric("Dette / FCF", "Dette / FCF", row.get("Dette / FCF"))
    with c3:
        _render_fundamental_metric(
            "Objectif", "Objectif analystes", row.get("Objectif analystes")
        )
    with c4:
        _render_fundamental_metric(
            "Upside", "Upside vs objectif", row.get("Upside vs objectif")
        )
    st.write(f"**Recommandation consensus :** {row.get('Recommandation', '-')}")

    if not pd.isna(row["Objectif bas"]) and not pd.isna(row["Objectif haut"]):
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=["Bas", "Moyen", "Haut"],
                y=[row["Objectif bas"], row["Objectif analystes"], row["Objectif haut"]],
                marker_color=["#d62728", "#1f77b4", "#2ca02c"],
            )
        )
        if not pd.isna(row["Prix"]):
            fig.add_hline(y=row["Prix"], line_dash="dash", annotation_text="Prix actuel")
        fig.update_layout(title=f"Fourchette d'objectifs analystes - {detail}")
        st.plotly_chart(fig, use_container_width=True)
