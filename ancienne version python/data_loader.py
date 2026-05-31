import pandas as pd
import streamlit as st
import yfinance as yf


def filter_valid_tickers(tickers, start_date):
    try:
        start_str = pd.to_datetime(start_date).strftime("%Y-%m-%d")
        end_str = (pd.to_datetime(start_date) + pd.Timedelta(days=10)).strftime(
            "%Y-%m-%d"
        )

        ticker_test = tickers[0]
        df = yf.download(ticker_test, start=start_str, end=end_str, progress=False)

        if df.empty:
            return tickers
        return tickers
    except Exception:
        return tickers


@st.cache_data
def load_prices_in_batches(tickers, start, batch_size=50):
    all_data = []
    failed = []

    start_str = pd.to_datetime(start).strftime("%Y-%m-%d")

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        success = False
        attempts = 0
        data = pd.DataFrame()

        while not success and attempts < 3:
            try:
                data = yf.download(batch, start=start_str, group_by="ticker", progress=False)
                if not data.empty:
                    success = True
                else:
                    attempts += 1
            except Exception:
                attempts += 1

        if not success or data.empty:
            failed.extend(batch)
            continue

        for ticker in batch:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    if ticker in data.columns.get_level_values(0):
                        ticker_df = data[ticker]
                        if "Adj Close" in ticker_df.columns:
                            series = ticker_df["Adj Close"]
                        elif "Close" in ticker_df.columns:
                            series = ticker_df["Close"]
                        else:
                            continue

                        clean_df = pd.DataFrame({ticker: series})
                        if not clean_df.dropna().empty:
                            all_data.append(clean_df)
                else:
                    if "Adj Close" in data.columns:
                        clean_df = pd.DataFrame({ticker: data["Adj Close"]})
                    elif "Close" in data.columns:
                        clean_df = pd.DataFrame({ticker: data["Close"]})
                    else:
                        continue
                    if not clean_df.dropna().empty:
                        all_data.append(clean_df)
            except Exception:
                continue

    if failed:
        st.warning(f"{len(failed)} tickers ignorés (pas de données Yahoo) : {failed}")

    if not all_data:
        return pd.DataFrame()

    prices = pd.concat(all_data, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices = prices.ffill().bfill()
    return prices
