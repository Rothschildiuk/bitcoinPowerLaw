import contextlib
import io

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from core.constants import CURRENCY_DOLLAR, CURRENCY_EURO, CURRENCY_GOLD, GENESIS_DATE

BTC_HISTORY_CSV_URL = (
    "https://raw.githubusercontent.com/Habrador/Bitcoin-price-visualization/main/Bitcoin-price-USD.csv"
)


def _extract_close_series(download_df):
    if download_df is None or download_df.empty:
        return pd.Series(dtype=float)
    close_series = download_df["Close"]
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.iloc[:, 0]
    close_series = pd.to_numeric(close_series, errors="coerce").dropna()
    if close_series.empty:
        return pd.Series(dtype=float)
    close_series.index = pd.to_datetime(close_series.index).tz_localize(None)
    return close_series.astype(float)


def _safe_download_close_series(symbol, start_date):
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            downloaded_df = yf.download(symbol, start=start_date, progress=False, threads=False)
    except Exception:
        return pd.Series(dtype=float)
    return _extract_close_series(downloaded_df)


@st.cache_data(ttl=3600)
def load_reference_series(start_date):
    eur_usd = _safe_download_close_series("EURUSD=X", start_date)
    # GC=F is usually more stable than XAUUSD=X on hosted environments.
    xau_usd = _safe_download_close_series("GC=F", start_date)
    if xau_usd.empty:
        xau_usd = _safe_download_close_series("XAUUSD=X", start_date)
    return eur_usd, xau_usd


def build_currency_close_series(raw_df, selected_currency):
    close_usd = raw_df["Close"].astype(float)
    if selected_currency == CURRENCY_DOLLAR:
        return close_usd

    start_date = str(raw_df.index.min().date())
    eur_usd, xau_usd = load_reference_series(start_date)

    if selected_currency == CURRENCY_EURO and not eur_usd.empty:
        eur_usd_aligned = eur_usd.reindex(close_usd.index).interpolate("time").ffill().bfill()
        return close_usd / eur_usd_aligned

    if selected_currency == CURRENCY_GOLD and not xau_usd.empty:
        xau_usd_aligned = xau_usd.reindex(close_usd.index).interpolate("time").ffill().bfill()
        return close_usd / xau_usd_aligned

    return close_usd


@st.cache_data(ttl=3600)
def load_prepared_price_data(price_history_url=BTC_HISTORY_CSV_URL, stale_after_days=3):
    full_df = pd.read_csv(price_history_url)
    full_df["Date"] = pd.to_datetime(full_df["Date"])
    full_df.set_index("Date", inplace=True)
    full_df.rename(columns={"Price": "Close"}, inplace=True)
    full_df["Close"] = pd.to_numeric(full_df["Close"], errors="coerce")
    full_df = full_df.dropna(subset=["Close"]).sort_index()

    latest_csv_date = full_df.index.max()
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    if stale_after_days is not None and (today - latest_csv_date.normalize()).days > int(stale_after_days):
        btc_tail = _safe_download_close_series(
            "BTC-USD",
            (latest_csv_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        )
        if not btc_tail.empty:
            tail_df = btc_tail.to_frame(name="Close")
            tail_df.index.name = "Date"
            full_df = pd.concat([full_df[["Close"]], tail_df], axis=0)
            full_df = full_df[~full_df.index.duplicated(keep="last")].sort_index()

    full_df = full_df[full_df["Close"] > 0]
    full_df["AbsDays"] = (full_df.index - GENESIS_DATE).days
    full_df["LogClose"] = np.log10(full_df["Close"])
    return full_df
