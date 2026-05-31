import random
import re
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from bs4 import BeautifulSoup
import yfinance as yf
from yahooquery import Ticker

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def safe_get(url, max_retries=3, sleep_base=1.0):
    for attempt in range(max_retries):
        try:
            time.sleep(random.uniform(0.5, 1.5))
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                return r.text
            elif r.status_code == 429:
                time.sleep(sleep_base * 4)
        except:
            pass
        time.sleep(sleep_base * (2**attempt) + random.random() * 0.5)
    return None


def get_isin_from_yahoo(ticker):
    try:
        t = Ticker(ticker, timeout=15)
        p = t.price
        if isinstance(p, dict) and ticker in p:
            d = p[ticker]
            for k in ["isin", "ISIN", "isinCode"]:
                if k in d and d[k]:
                    return d[k]
    except:
        pass
    return None


def get_dividends_yahoo(ticker):
    try:
        t = Ticker(ticker, timeout=20)
        d = t.dividends
        if isinstance(d, pd.DataFrame) and not d.empty:
            if isinstance(d.index, pd.MultiIndex):
                if ticker in d.index.get_level_values(0):
                    d = d.xs(ticker, level=0)
                else:
                    return None

            if "dividends" in d.columns:
                d.index = pd.to_datetime(d.index)
                s = d["dividends"].sort_index()
                return s
    except:
        pass
    return None


def get_dividends_yfinance_native(ticker):
    def _extract_dividends_from_history(hist, ticker_symbol):
        if hist is None or hist.empty:
            return None

        if isinstance(hist.columns, pd.MultiIndex):
            if ticker_symbol in hist.columns.get_level_values(1):
                try:
                    div_col = ("Dividends", ticker_symbol)
                    if div_col in hist.columns:
                        s = hist[div_col]
                    else:
                        return None
                except Exception:
                    return None
            elif ticker_symbol in hist.columns.get_level_values(0):
                try:
                    div_col = (ticker_symbol, "Dividends")
                    if div_col in hist.columns:
                        s = hist[div_col]
                    else:
                        return None
                except Exception:
                    return None
            else:
                level0 = hist.columns.get_level_values(0)
                if "Dividends" not in level0:
                    return None
                div_frame = hist.xs("Dividends", axis=1, level=0)
                if div_frame.empty:
                    return None
                s = div_frame.iloc[:, 0]
        else:
            if "Dividends" not in hist.columns:
                return None
            s = hist["Dividends"]

        s = pd.to_numeric(s, errors="coerce").dropna()
        s = s[s > 0]
        if s.empty:
            return None
        s.index = pd.to_datetime(s.index).tz_localize(None)
        return s.sort_index()

    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="max", auto_adjust=False, raise_errors=False)
        s = _extract_dividends_from_history(hist, ticker)
        if s is not None:
            return s

        hist2 = yf.download(ticker, period="max", auto_adjust=False, actions=True, progress=False, threads=False)
        s = _extract_dividends_from_history(hist2, ticker)
        if s is not None:
            return s
    except:
        pass
    return None


def normalize_boursorama_ticker(t):
    code = t.split(".")[0]
    if ".PA" in t:
        return f"1rP{code}"
    return code


def get_dividends_boursorama(ticker):
    code = normalize_boursorama_ticker(ticker)
    url = f"https://www.boursorama.com/cours/dividendes/{code}/"
    html = safe_get(url)
    if html is None:
        return None

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", {"class": "c-table"})
    if table is None:
        table = soup.find("table")

    if table is None:
        return None

    rows = table.find_all("tr")
    data = []
    for r in rows[1:]:
        c = r.find_all("td")
        if len(c) < 2:
            continue
        date_txt = c[0].get_text(strip=True)
        div_txt = c[1].get_text(strip=True).replace(",", ".")

        if len(date_txt) == 4 and date_txt.isdigit():
            date_txt = f"{date_txt}-06-01"

        try:
            v = float(re.findall(r"[-+]?\d*\.?\d+", div_txt)[0])
        except:
            continue
        data.append((date_txt, v))

    if not data:
        return None

    idx = pd.to_datetime([d[0] for d in data], errors="coerce")
    vals = [d[1] for d in data]
    s = pd.Series(vals, index=idx).dropna().sort_index()
    return s if not s.empty else None


def get_dividends_lecho(isin):
    if isin is None:
        return None
    url = f"https://www.lecho.be/cours/{isin}"
    html = safe_get(url)
    if html is None:
        return None
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)
    matches = re.findall(r"Dividende\s+([\d,\.]+)", text)
    if not matches:
        return None
    vals = []
    for m in matches:
        try:
            vals.append(float(m.replace(",", ".")))
        except:
            pass
    if not vals:
        return None
    years = list(range(len(vals)))
    idx = pd.to_datetime(["2020-01-01"] * len(vals)) + pd.to_timedelta(years, unit="Y")
    return pd.Series(vals, index=idx).sort_index()


def compute_dividend_metrics(s, last_price):
    if s is None or len(s) == 0 or last_price is None or last_price <= 0:
        return None

    # --- SÉCURITÉ IMPORTANTE (EUTELSAT CORRECTIF) ---
    derniere_date_versement = s.index.max()
    jours_depuis_dernier_div = (pd.Timestamp.now() - derniere_date_versement).days
    if jours_depuis_dernier_div > 450:
        return {
            "last_year_div": 0.0,
            "current_yield": 0.0,
            "cagr_5y": 0.0,
            "growth_years": 0,
            "years_available": 0,
        }
    # -----------------------------------------------

    df = s.groupby(s.index.year).sum().sort_index()
    if df.empty:
        return None
    last_div = df.iloc[-1]
    y = last_div / last_price

    if len(df) >= 5:
        start = df.iloc[-5]
        end = df.iloc[-1]
        if start