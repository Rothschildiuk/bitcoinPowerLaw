import contextlib
import io
import json
import time
import urllib.request

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from core.constants import CURRENCY_DOLLAR, CURRENCY_EURO, CURRENCY_GOLD, GENESIS_DATE

BTC_HISTORY_CSV_URL = (
    "https://raw.githubusercontent.com/Habrador/Bitcoin-price-visualization/main/Bitcoin-price-USD.csv"
)
MINER_REVENUE_CSV_URL = (
    "https://api.blockchain.info/charts/miners-revenue"
    "?timespan=all&sampled=false&metadata=false&cors=true&format=csv"
)
DIFFICULTY_CSV_URL = (
    "https://api.blockchain.info/charts/difficulty"
    "?timespan=all&sampled=false&metadata=false&cors=true&format=csv"
)
HASHRATE_CSV_URL = (
    "https://api.blockchain.info/charts/hash-rate"
    "?timespan=all&sampled=false&metadata=false&cors=true&format=csv"
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


def _normalize_chart_csv(data_df, value_column_name):
    if data_df is None or data_df.empty:
        return pd.DataFrame(columns=["Close", "AbsDays", "LogClose"])

    date_column = None
    for candidate in ("Date", "Timestamp", "date", "timestamp"):
        if candidate in data_df.columns:
            date_column = candidate
            break
    if date_column is None:
        date_column = data_df.columns[0]

    value_column = None
    for candidate in ("Value", "value", "Price", "price", "Close", "close"):
        if candidate in data_df.columns:
            value_column = candidate
            break
    if value_column is None:
        value_column = data_df.columns[-1]

    prepared_df = data_df[[date_column, value_column]].copy()
    prepared_df.columns = ["Date", value_column_name]
    prepared_df["Date"] = pd.to_datetime(prepared_df["Date"], errors="coerce")
    prepared_df[value_column_name] = pd.to_numeric(prepared_df[value_column_name], errors="coerce")
    prepared_df = prepared_df.dropna(subset=["Date", value_column_name])
    prepared_df = prepared_df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    prepared_df = prepared_df.set_index("Date")
    prepared_df = prepared_df[prepared_df[value_column_name] > 0]
    prepared_df["AbsDays"] = (prepared_df.index - GENESIS_DATE).days
    prepared_df["LogClose"] = np.log10(prepared_df[value_column_name])
    return prepared_df


def _safe_download_close_series(symbol, start_date):
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            downloaded_df = yf.download(symbol, start=start_date, progress=False, threads=False)
    except Exception:
        return pd.Series(dtype=float)
    return _extract_close_series(downloaded_df)


def _safe_download_btc_tail_from_coingecko(start_date):
    start_ts = int(pd.Timestamp(start_date).timestamp())
    end_ts = int(pd.Timestamp.utcnow().timestamp())
    url = (
        "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
        f"?vs_currency=usd&from={start_ts}&to={end_ts}"
    )
    payload = None
    for _ in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "PowerLaw/1.0"})
            with urllib.request.urlopen(request, timeout=15) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except Exception:
            time.sleep(0.4)
    if payload is None:
        return pd.Series(dtype=float)

    prices = payload.get("prices", [])
    if not prices:
        return pd.Series(dtype=float)

    tail_df = pd.DataFrame(prices, columns=["ts_ms", "Close"])
    tail_df["Date"] = pd.to_datetime(tail_df["ts_ms"], unit="ms", utc=True).dt.tz_localize(None)
    # CoinGecko can return multiple points per day; keep daily close-like last point.
    tail_df = tail_df.sort_values("Date")
    tail_df["Day"] = tail_df["Date"].dt.normalize()
    daily_tail = tail_df.groupby("Day", as_index=True)["Close"].last()
    daily_tail.index = pd.to_datetime(daily_tail.index)
    daily_tail = pd.to_numeric(daily_tail, errors="coerce").dropna()
    return daily_tail.astype(float)


def _safe_download_btc_tail_from_coincap(start_date):
    try:
        start_ms = int(pd.Timestamp(start_date).timestamp() * 1000)
        end_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
        url = (
            "https://api.coincap.io/v2/assets/bitcoin/history"
            f"?interval=d1&start={start_ms}&end={end_ms}"
        )
        request = urllib.request.Request(url, headers={"User-Agent": "PowerLaw/1.0"})
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return pd.Series(dtype=float)

    rows = payload.get("data", [])
    if not rows:
        return pd.Series(dtype=float)

    history_df = pd.DataFrame(rows)
    if "time" not in history_df.columns or "priceUsd" not in history_df.columns:
        return pd.Series(dtype=float)

    history_df["Date"] = pd.to_datetime(history_df["time"], unit="ms", utc=True).dt.tz_localize(None)
    history_df["Close"] = pd.to_numeric(history_df["priceUsd"], errors="coerce")
    history_df = history_df.dropna(subset=["Date", "Close"]).sort_values("Date")
    history_df["Day"] = history_df["Date"].dt.normalize()
    daily_tail = history_df.groupby("Day", as_index=True)["Close"].last()
    daily_tail.index = pd.to_datetime(daily_tail.index)
    return daily_tail.astype(float)


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
        if btc_tail.empty:
            btc_tail = _safe_download_btc_tail_from_coingecko(
                (latest_csv_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            )
        if btc_tail.empty:
            btc_tail = _safe_download_btc_tail_from_coincap(
                (latest_csv_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
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


@st.cache_data(ttl=3600)
def load_prepared_miner_revenue_data(revenue_history_url=MINER_REVENUE_CSV_URL):
    revenue_df = pd.read_csv(revenue_history_url)
    return _normalize_chart_csv(revenue_df, "Close")


@st.cache_data(ttl=3600)
def load_prepared_difficulty_data(difficulty_history_url=DIFFICULTY_CSV_URL):
    difficulty_df = pd.read_csv(difficulty_history_url)
    prepared_df = _normalize_chart_csv(difficulty_df, "Close")
    return prepared_df[prepared_df.index >= pd.Timestamp("2010-01-01")]


@st.cache_data(ttl=3600)
def load_prepared_hashrate_data(hashrate_history_url=HASHRATE_CSV_URL):
    hashrate_df = pd.read_csv(hashrate_history_url)
    prepared_df = _normalize_chart_csv(hashrate_df, "Close")
    return prepared_df[prepared_df.index >= pd.Timestamp("2010-01-01")]
