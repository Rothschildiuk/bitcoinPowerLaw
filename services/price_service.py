import contextlib
import io
import json
import re
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from core.constants import CURRENCY_DOLLAR, CURRENCY_EURO, CURRENCY_GOLD, GENESIS_DATE

BTC_HISTORY_CSV_URL = "https://raw.githubusercontent.com/Habrador/Bitcoin-price-visualization/main/Bitcoin-price-USD.csv"
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
BITCOIN_VISUALS_DAILY_CSV_URL = "https://bitcoinvisuals.com/static/data/data_daily.csv"
LIQUID_RESERVES_URL = "https://liquid.network/api/v1/liquid/reserves"
LIQUID_RESERVES_MONTH_URL = "https://liquid.network/api/v1/liquid/reserves/month"
LIQUID_CHARTS_DATA_URL = "https://liquid.net/api/getChartsData"
FRED_M2SL_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=M2SL"
LOCAL_DATA_CACHE_DIR = Path("output/data_cache")
SNAPSHOT_DATA_DIR = Path("data/snapshots")
LOCAL_CACHE_SCHEMA_VERSION = 4
FAST_REFRESH_SECONDS = 3600
SLOW_REFRESH_SECONDS = 6 * 3600
REFERENCE_REFRESH_SECONDS = 12 * 3600
COINLORE_MONERO_START_DATE = "2014-05-21"
COINLORE_CRYPTO_META = {
    "FIL": {"slug": "filecoin", "start_date": "2017-12-13"},
    "XMR": {"slug": "monero", "start_date": COINLORE_MONERO_START_DATE},
    "LTC": {"slug": "litecoin", "start_date": "2013-04-28"},
    "DOGE": {"slug": "dogecoin", "start_date": "2013-12-15"},
}


@dataclass(frozen=True)
class DataFrameSourceAdapter:
    cache_key: str
    refresh_seconds: int
    fetch_fn: Callable[[], pd.DataFrame]
    validator_fn: Callable[[pd.DataFrame], bool] | None = None

    def load(self):
        return _load_or_refresh_dataframe_cache(
            self.cache_key,
            self.fetch_fn,
            min_check_interval_seconds=self.refresh_seconds,
            validator_fn=self.validator_fn,
        )


def _ensure_local_data_cache_dir():
    LOCAL_DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _ensure_snapshot_data_dir():
    SNAPSHOT_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _get_cache_frame_path(cache_key):
    return LOCAL_DATA_CACHE_DIR / f"{cache_key}.csv"


def _get_cache_meta_path(cache_key):
    return LOCAL_DATA_CACHE_DIR / f"{cache_key}.meta.json"


def _get_snapshot_frame_path(snapshot_key):
    return SNAPSHOT_DATA_DIR / f"{snapshot_key}.csv"


def _read_cache_meta(cache_key):
    meta_path = _get_cache_meta_path(cache_key)
    if not meta_path.exists():
        return {}

    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_cache_meta(cache_key, *, error_message=None, store_index=None):
    _ensure_local_data_cache_dir()
    meta_path = _get_cache_meta_path(cache_key)
    payload = {
        "version": LOCAL_CACHE_SCHEMA_VERSION,
        "checked_at": pd.Timestamp.utcnow().tz_localize(None).isoformat(),
    }
    if store_index is not None:
        payload["store_index"] = bool(store_index)
    if error_message:
        payload["last_error"] = str(error_message)
    meta_path.write_text(json.dumps(payload), encoding="utf-8")


def _read_cached_dataframe(cache_key):
    frame_path = _get_cache_frame_path(cache_key)
    if not frame_path.exists():
        return None

    try:
        cached_df = pd.read_csv(frame_path)
    except Exception:
        return None

    cache_meta = _read_cache_meta(cache_key)
    if cache_meta.get("store_index") and "Date" in cached_df.columns:
        cached_df["Date"] = pd.to_datetime(cached_df["Date"], errors="coerce")
        cached_df = cached_df.dropna(subset=["Date"]).set_index("Date").sort_index()
    return cached_df


def _write_cached_dataframe(cache_key, data_df):
    if data_df is None or data_df.empty:
        return

    _ensure_local_data_cache_dir()
    frame_path = _get_cache_frame_path(cache_key)
    data_to_store = data_df.copy()
    store_index = not isinstance(data_to_store.index, pd.RangeIndex)
    if store_index:
        data_to_store.to_csv(frame_path, index_label="Date")
    else:
        data_to_store.to_csv(frame_path, index=False)
    _write_cache_meta(cache_key, store_index=store_index)


def _read_snapshot_dataframe(snapshot_key):
    frame_path = _get_snapshot_frame_path(snapshot_key)
    if not frame_path.exists():
        return None

    try:
        snapshot_df = pd.read_csv(frame_path)
    except Exception:
        return None

    if "Date" in snapshot_df.columns:
        snapshot_df["Date"] = pd.to_datetime(snapshot_df["Date"], errors="coerce")
        snapshot_df = snapshot_df.dropna(subset=["Date"]).set_index("Date").sort_index()
    return snapshot_df


def write_snapshot_dataframe(snapshot_key, data_df):
    if data_df is None or data_df.empty:
        return

    _ensure_snapshot_data_dir()
    frame_path = _get_snapshot_frame_path(snapshot_key)
    data_to_store = data_df.copy()
    if isinstance(data_to_store.index, pd.DatetimeIndex):
        data_to_store.to_csv(frame_path, index_label="Date")
    else:
        data_to_store.to_csv(frame_path, index=False)


def _is_cache_refresh_due(cache_key, min_check_interval_seconds):
    cache_meta = _read_cache_meta(cache_key)
    if int(cache_meta.get("version", 0)) != LOCAL_CACHE_SCHEMA_VERSION:
        return True
    checked_at = cache_meta.get("checked_at")
    if not checked_at:
        return True

    checked_ts = pd.to_datetime(checked_at, errors="coerce")
    if pd.isna(checked_ts):
        return True

    age_seconds = (
        pd.Timestamp.utcnow().tz_localize(None) - checked_ts.tz_localize(None)
    ).total_seconds()
    return age_seconds >= float(min_check_interval_seconds)


def _is_cache_valid(cache_key, cached_df, validator_fn=None):
    if cached_df is None or cached_df.empty:
        return False

    cache_meta = _read_cache_meta(cache_key)
    if int(cache_meta.get("version", 0)) != LOCAL_CACHE_SCHEMA_VERSION:
        return False

    if validator_fn is not None:
        try:
            return bool(validator_fn(cached_df))
        except Exception:
            return False

    return True


def _load_or_refresh_dataframe_cache(
    cache_key,
    fetch_fn,
    *,
    min_check_interval_seconds,
    validator_fn=None,
):
    cached_df = _read_cached_dataframe(cache_key)
    has_valid_cache = _is_cache_valid(cache_key, cached_df, validator_fn=validator_fn)
    if has_valid_cache and not _is_cache_refresh_due(cache_key, min_check_interval_seconds):
        return cached_df

    try:
        fresh_df = fetch_fn()
    except Exception as exc:
        if has_valid_cache:
            _write_cache_meta(cache_key, error_message=exc)
            return cached_df
        raise

    if fresh_df is None or fresh_df.empty:
        if has_valid_cache:
            _write_cache_meta(cache_key, error_message="Empty refresh result")
            return cached_df
        return fresh_df

    _write_cached_dataframe(cache_key, fresh_df)
    return fresh_df


def _load_source_adapter(adapter):
    return adapter.load()


def _load_cached_dataframe_snapshot(cache_keys, validator_fn=None):
    for cache_key in cache_keys:
        cached_df = _read_cached_dataframe(cache_key)
        if _is_cache_valid(cache_key, cached_df, validator_fn=validator_fn):
            return cached_df
    return None


def _load_snapshot_or_live(snapshot_key, validator_fn, live_loader, *, source="auto"):
    if source not in {"auto", "snapshot", "live"}:
        raise ValueError(f"Unsupported source mode: {source}")

    snapshot_df = None
    if source in {"auto", "snapshot"}:
        snapshot_df = _read_snapshot_dataframe(snapshot_key)
        if snapshot_df is not None and validator_fn(snapshot_df):
            if source == "snapshot":
                return snapshot_df
            try:
                live_df = live_loader()
            except Exception:
                return snapshot_df
            if live_df is not None and validator_fn(live_df):
                return live_df
            return snapshot_df
        if source == "snapshot":
            raise ValueError(f"Local snapshot is missing or invalid: {snapshot_key}")

    return live_loader()


def _validate_reference_frame(data_df):
    return (
        isinstance(data_df, pd.DataFrame)
        and len(data_df) >= 100
        and ("EURUSD" in data_df.columns or "XAUUSD" in data_df.columns)
    )


def _validate_prepared_price_data(data_df):
    return (
        isinstance(data_df, pd.DataFrame)
        and len(data_df) >= 1000
        and "Close" in data_df.columns
        and data_df.index.min() <= pd.Timestamp("2011-01-01")
    )


def _validate_prepared_chart_data(data_df):
    return (
        isinstance(data_df, pd.DataFrame)
        and len(data_df) >= 100
        and "Close" in data_df.columns
        and "AbsDays" in data_df.columns
        and "LogClose" in data_df.columns
    )


def _validate_bitcoin_visuals_daily_data(data_df):
    return isinstance(data_df, pd.DataFrame) and len(data_df) >= 100 and "day" in data_df.columns


def _validate_prepared_liquid_btc_data(data_df):
    return (
        isinstance(data_df, pd.DataFrame)
        and len(data_df) >= 30
        and "Close" in data_df.columns
        and data_df.index.min() <= pd.Timestamp("2024-12-31")
    )


def _validate_prepared_liquid_transactions_data(data_df):
    return (
        isinstance(data_df, pd.DataFrame)
        and len(data_df) >= 100
        and "Close" in data_df.columns
        and data_df.index.min() <= pd.Timestamp("2019-01-31")
    )


def _validate_prepared_filecoin_btc_data(data_df):
    return (
        isinstance(data_df, pd.DataFrame)
        and len(data_df) >= 100
        and "Close" in data_df.columns
        and data_df.index.min() <= pd.Timestamp("2020-01-01")
    )


def _validate_prepared_monero_btc_data(data_df):
    return (
        isinstance(data_df, pd.DataFrame)
        and len(data_df) >= 100
        and "Close" in data_df.columns
        and data_df.index.min() <= pd.Timestamp("2015-02-01")
    )


def _validate_prepared_litecoin_btc_data(data_df):
    return (
        isinstance(data_df, pd.DataFrame)
        and len(data_df) >= 100
        and "Close" in data_df.columns
        and data_df.index.min() <= pd.Timestamp("2013-10-01")
    )


def _validate_prepared_dogecoin_btc_data(data_df):
    return (
        isinstance(data_df, pd.DataFrame)
        and len(data_df) >= 100
        and "Close" in data_df.columns
        and data_df.index.min() <= pd.Timestamp("2014-02-15")
    )


def _validate_prepared_us_m2_data(data_df):
    return (
        isinstance(data_df, pd.DataFrame)
        and len(data_df) >= 500
        and "Close" in data_df.columns
        and data_df.index.min() <= pd.Timestamp("1959-01-01")
    )


def _append_btc_live_tail(base_df, *, stale_after_days):
    if base_df is None or base_df.empty:
        return base_df

    full_df = base_df.copy()
    full_df.index = pd.to_datetime(full_df.index)
    full_df = full_df.sort_index()
    if stale_after_days is None:
        return full_df

    latest_base_date = full_df.index.max()
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    if (today - latest_base_date.normalize()).days <= int(stale_after_days):
        return full_df

    tail_start = (latest_base_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    btc_tail = _safe_download_close_series("BTC-USD", tail_start)
    if btc_tail.empty:
        btc_tail = _safe_download_btc_tail_from_coingecko(tail_start)
    if btc_tail.empty:
        btc_tail = _safe_download_btc_tail_from_coincap(tail_start)
    if btc_tail.empty:
        return full_df

    tail_df = btc_tail.to_frame(name="Close")
    tail_df.index.name = "Date"
    full_df = pd.concat([full_df[["Close"]], tail_df], axis=0)
    full_df = full_df[~full_df.index.duplicated(keep="last")].sort_index()
    return full_df


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


def _build_normalized_csv_adapter(
    cache_key,
    source_url,
    *,
    refresh_seconds,
    validator_fn,
    value_column_name="Close",
    postprocess_fn=None,
    read_csv_kwargs=None,
):
    def fetch_frame():
        prepared_df = _normalize_chart_csv(
            pd.read_csv(source_url, **(read_csv_kwargs or {})),
            value_column_name,
        )
        if callable(postprocess_fn):
            prepared_df = postprocess_fn(prepared_df)
        return prepared_df

    return DataFrameSourceAdapter(
        cache_key=cache_key,
        refresh_seconds=refresh_seconds,
        fetch_fn=fetch_frame,
        validator_fn=validator_fn,
    )


def _build_blockchain_chart_adapter(cache_key, source_url):
    return _build_normalized_csv_adapter(
        cache_key,
        source_url,
        refresh_seconds=SLOW_REFRESH_SECONDS,
        validator_fn=_validate_prepared_chart_data,
        read_csv_kwargs={"header": None, "names": ["Timestamp", "Value"]},
    )


def _safe_download_close_series(symbol, start_date):
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            downloaded_df = yf.download(symbol, start=start_date, progress=False, threads=False)
    except Exception:
        return pd.Series(dtype=float)
    return _extract_close_series(downloaded_df)


def _fetch_json_with_retry(
    url,
    *,
    retries=3,
    timeout=15,
    initial_backoff_seconds=0.4,
    backoff_multiplier=2.0,
):
    headers = {"User-Agent": "PowerLaw/1.0"}
    last_exception = None
    for attempt in range(max(1, int(retries))):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            last_exception = exc
            if attempt == retries - 1:
                break
            sleep_seconds = initial_backoff_seconds * (backoff_multiplier**attempt)
            time.sleep(sleep_seconds)
    if last_exception is not None:
        return None
    return None


def _fetch_csv_with_retry(
    url,
    *,
    retries=3,
    timeout=20,
    initial_backoff_seconds=0.4,
    backoff_multiplier=2.0,
):
    headers = {"User-Agent": "PowerLaw/1.0"}
    for attempt in range(max(1, int(retries))):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload_text = response.read().decode("utf-8")
            return pd.read_csv(io.StringIO(payload_text))
        except Exception:
            if attempt == retries - 1:
                break
            sleep_seconds = initial_backoff_seconds * (backoff_multiplier**attempt)
            time.sleep(sleep_seconds)
    return pd.DataFrame()


def _fetch_text_with_retry(
    url,
    *,
    retries=3,
    timeout=20,
    initial_backoff_seconds=0.4,
    backoff_multiplier=2.0,
):
    headers = {"User-Agent": "Mozilla/5.0"}
    last_exception = None
    for attempt in range(max(1, int(retries))):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8")
        except Exception as exc:
            last_exception = exc
            if attempt == retries - 1:
                break
            sleep_seconds = initial_backoff_seconds * (backoff_multiplier**attempt)
            time.sleep(sleep_seconds)
    if last_exception is not None:
        return ""
    return ""


def _safe_download_cryptocompare_histoday(fsym, tsym):
    cache_key = f"cryptocompare_histoday_{fsym.lower()}_{tsym.lower()}"

    def fetch_histoday_frame():
        url = (
            "https://min-api.cryptocompare.com/data/v2/histoday"
            f"?fsym={fsym}&tsym={tsym}&limit=5000&allData=true"
        )
        payload = _fetch_json_with_retry(url, retries=3, timeout=20)
        if not isinstance(payload, dict) or payload.get("Response") != "Success":
            raise ValueError(f"Unable to load CryptoCompare history for {fsym}/{tsym}.")

        rows = payload.get("Data", {}).get("Data", [])
        if not isinstance(rows, list) or len(rows) == 0:
            raise ValueError(f"CryptoCompare returned no rows for {fsym}/{tsym}.")

        history_df = pd.DataFrame(rows)
        if "time" not in history_df.columns or "close" not in history_df.columns:
            raise ValueError(f"CryptoCompare payload is missing columns for {fsym}/{tsym}.")

        history_df["Date"] = pd.to_datetime(history_df["time"], unit="s", utc=True).dt.tz_localize(
            None
        )
        history_df["Close"] = pd.to_numeric(history_df["close"], errors="coerce")
        history_df = history_df.dropna(subset=["Date", "Close"]).sort_values("Date")
        history_df = history_df[history_df["Close"] > 0]
        if history_df.empty:
            raise ValueError(f"CryptoCompare returned only non-positive rows for {fsym}/{tsym}.")

        return history_df[["Date", "Close"]].copy()

    try:
        history_df = _load_or_refresh_dataframe_cache(
            cache_key,
            fetch_histoday_frame,
            min_check_interval_seconds=FAST_REFRESH_SECONDS,
            validator_fn=lambda data_df: (
                isinstance(data_df, pd.DataFrame)
                and len(data_df) >= 100
                and "Date" in data_df.columns
                and "Close" in data_df.columns
            ),
        )
    except Exception:
        return pd.Series(dtype=float)
    if history_df is None or history_df.empty:
        return pd.Series(dtype=float)

    history_df = history_df.copy()
    history_df["Date"] = pd.to_datetime(history_df["Date"], errors="coerce")
    history_df["Close"] = pd.to_numeric(history_df["Close"], errors="coerce")
    history_df = history_df.dropna(subset=["Date", "Close"]).sort_values("Date")
    history_df = history_df[history_df["Close"] > 0]
    if history_df.empty:
        return pd.Series(dtype=float)

    close_series = history_df.groupby("Date", as_index=True)["Close"].last()
    close_series.index = pd.to_datetime(close_series.index)
    return close_series.astype(float)


def _safe_download_coinlore_crypto_usd(coin_slug, cache_key, start_date, end_date=None):
    if not coin_slug:
        return pd.Series(dtype=float)

    def fetch_coinlore_frame():
        start_ts = int(pd.Timestamp(start_date).timestamp())
        end_ts = int(pd.Timestamp(end_date or pd.Timestamp.utcnow().normalize()).timestamp())
        url = f"https://www.coinlore.com/coin/{coin_slug}/historical-data" f"/{start_ts}/{end_ts}"
        payload_text = _fetch_text_with_retry(url, retries=3, timeout=20)
        if not payload_text:
            raise ValueError(f"Unable to load CoinLore history for {coin_slug}.")

        row_pattern = re.compile(
            r'<tr class="txtr">\s*<td class="nwt txtl font-bold">\s*(.*?)\s*</td>\s*'
            r'<td class="nwt">\$(.*?)</td>\s*'
            r'<td class="nwt">\$(.*?)</td>\s*'
            r'<td class="nwt">\$(.*?)</td>\s*'
            r'<td class="nwt">\$(.*?)</td>',
            re.DOTALL,
        )
        rows = row_pattern.findall(payload_text)
        if not rows:
            raise ValueError(f"CoinLore returned no rows for {coin_slug}.")

        history_df = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close"])
        history_df["Date"] = pd.to_datetime(history_df["Date"], errors="coerce")
        history_df["Close"] = (
            history_df["Close"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .pipe(pd.to_numeric, errors="coerce")
        )
        history_df = history_df.dropna(subset=["Date", "Close"]).sort_values("Date")
        history_df = history_df[history_df["Close"] > 0]
        if history_df.empty:
            raise ValueError(f"CoinLore returned only non-positive rows for {coin_slug}.")

        return history_df[["Date", "Close"]].copy()

    try:
        history_df = _load_or_refresh_dataframe_cache(
            cache_key,
            fetch_coinlore_frame,
            min_check_interval_seconds=SLOW_REFRESH_SECONDS,
            validator_fn=lambda data_df: (
                isinstance(data_df, pd.DataFrame)
                and len(data_df) >= 100
                and "Date" in data_df.columns
                and "Close" in data_df.columns
            ),
        )
    except Exception:
        return pd.Series(dtype=float)
    if history_df is None or history_df.empty:
        return pd.Series(dtype=float)

    history_df = history_df.copy()
    history_df["Date"] = pd.to_datetime(history_df["Date"], errors="coerce")
    history_df["Close"] = pd.to_numeric(history_df["Close"], errors="coerce")
    history_df = history_df.dropna(subset=["Date", "Close"]).sort_values("Date")
    history_df = history_df[history_df["Close"] > 0]
    if history_df.empty:
        return pd.Series(dtype=float)

    close_series = history_df.groupby("Date", as_index=True)["Close"].last()
    close_series.index = pd.to_datetime(close_series.index)
    return close_series.astype(float)


def _safe_download_coinlore_monero_usd(start_date=COINLORE_MONERO_START_DATE, end_date=None):
    return _safe_download_coinlore_crypto_usd(
        "monero",
        "coinlore_monero_usd",
        start_date,
        end_date=end_date,
    )


def _derive_crypto_btc_from_usd_series(crypto_usd, btc_usd, start_date):
    if crypto_usd.empty or btc_usd.empty:
        return pd.Series(dtype=float)

    combined_index = crypto_usd.index.union(btc_usd.index)
    crypto_usd = crypto_usd.reindex(combined_index).interpolate("time").ffill()
    btc_usd = btc_usd.reindex(combined_index).interpolate("time").ffill()

    first_valid_date = max(
        pd.Timestamp(crypto_usd.first_valid_index()),
        pd.Timestamp(btc_usd.first_valid_index()),
        pd.Timestamp(start_date),
    )

    ratio = crypto_usd / btc_usd
    ratio = pd.to_numeric(ratio, errors="coerce")
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    ratio = ratio[ratio.index >= first_valid_date]
    ratio = ratio[ratio > 0]
    return ratio.astype(float)


def _safe_download_crypto_btc_via_usd(fsym, start_date):
    crypto_usd = _safe_download_cryptocompare_histoday(fsym, "USD")
    btc_usd = _safe_download_cryptocompare_histoday("BTC", "USD")
    return _derive_crypto_btc_from_usd_series(crypto_usd, btc_usd, start_date)


def _safe_download_monero_btc_via_coinlore(start_date):
    monero_usd = _safe_download_coinlore_monero_usd(start_date=COINLORE_MONERO_START_DATE)
    if monero_usd.empty:
        return pd.Series(dtype=float)

    btc_price_df = load_prepared_price_data(source="auto")
    if btc_price_df is None or btc_price_df.empty or "Close" not in btc_price_df.columns:
        return pd.Series(dtype=float)

    btc_usd = pd.to_numeric(btc_price_df["Close"], errors="coerce").dropna()
    btc_usd.index = pd.to_datetime(btc_usd.index)
    return _derive_crypto_btc_from_usd_series(monero_usd, btc_usd, start_date)


def _safe_download_crypto_btc_via_coinlore(fsym, start_date):
    meta = COINLORE_CRYPTO_META.get(str(fsym).upper())
    if meta is None:
        return pd.Series(dtype=float)

    crypto_usd = _safe_download_coinlore_crypto_usd(
        meta["slug"],
        f"coinlore_{str(fsym).lower()}_usd",
        meta["start_date"],
    )
    if crypto_usd.empty:
        return pd.Series(dtype=float)

    btc_price_df = load_prepared_price_data(source="auto")
    if btc_price_df is None or btc_price_df.empty or "Close" not in btc_price_df.columns:
        return pd.Series(dtype=float)

    btc_usd = pd.to_numeric(btc_price_df["Close"], errors="coerce").dropna()
    btc_usd.index = pd.to_datetime(btc_usd.index)
    return _derive_crypto_btc_from_usd_series(crypto_usd, btc_usd, start_date)


def _safe_download_btc_tail_from_coingecko(start_date):
    start_ts = int(pd.Timestamp(start_date).timestamp())
    end_ts = int(pd.Timestamp.utcnow().timestamp())
    url = (
        "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
        f"?vs_currency=usd&from={start_ts}&to={end_ts}"
    )
    payload = _fetch_json_with_retry(url, retries=3, timeout=15)
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
    start_ms = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
    url = (
        "https://api.coincap.io/v2/assets/bitcoin/history"
        f"?interval=d1&start={start_ms}&end={end_ms}"
    )
    payload = _fetch_json_with_retry(url, retries=3, timeout=15)
    if payload is None:
        return pd.Series(dtype=float)

    rows = payload.get("data", [])
    if not rows:
        return pd.Series(dtype=float)

    history_df = pd.DataFrame(rows)
    if "time" not in history_df.columns or "priceUsd" not in history_df.columns:
        return pd.Series(dtype=float)

    history_df["Date"] = pd.to_datetime(history_df["time"], unit="ms", utc=True).dt.tz_localize(
        None
    )
    history_df["Close"] = pd.to_numeric(history_df["priceUsd"], errors="coerce")
    history_df = history_df.dropna(subset=["Date", "Close"]).sort_values("Date")
    history_df["Day"] = history_df["Date"].dt.normalize()
    daily_tail = history_df.groupby("Day", as_index=True)["Close"].last()
    daily_tail.index = pd.to_datetime(daily_tail.index)
    return daily_tail.astype(float)


@st.cache_data(ttl=3600)
def load_reference_series(start_date, source="auto"):
    def fetch_reference_frame():
        eur_usd = _safe_download_close_series("EURUSD=X", start_date)
        # GC=F is usually more stable than XAUUSD=X on hosted environments.
        xau_usd = _safe_download_close_series("GC=F", start_date)
        if xau_usd.empty:
            xau_usd = _safe_download_close_series("XAUUSD=X", start_date)

        reference_df = pd.concat(
            [
                eur_usd.rename("EURUSD"),
                xau_usd.rename("XAUUSD"),
            ],
            axis=1,
        ).sort_index()
        return reference_df

    reference_df = _load_snapshot_or_live(
        "reference_series",
        _validate_reference_frame,
        lambda: _load_source_adapter(
            DataFrameSourceAdapter(
                cache_key="reference_series",
                refresh_seconds=REFERENCE_REFRESH_SECONDS,
                fetch_fn=fetch_reference_frame,
                validator_fn=_validate_reference_frame,
            )
        ),
        source=source,
    )
    eur_usd = (
        pd.to_numeric(reference_df["EURUSD"], errors="coerce").dropna()
        if "EURUSD" in reference_df.columns
        else pd.Series(dtype=float)
    )
    xau_usd = (
        pd.to_numeric(reference_df["XAUUSD"], errors="coerce").dropna()
        if "XAUUSD" in reference_df.columns
        else pd.Series(dtype=float)
    )
    eur_usd.index = pd.to_datetime(eur_usd.index)
    xau_usd.index = pd.to_datetime(xau_usd.index)
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
def load_prepared_price_data(
    price_history_url=BTC_HISTORY_CSV_URL, stale_after_days=0, source="auto"
):
    def fetch_price_data():
        full_df = pd.read_csv(price_history_url)
        full_df["Date"] = pd.to_datetime(full_df["Date"])
        full_df.set_index("Date", inplace=True)
        full_df.rename(columns={"Price": "Close"}, inplace=True)
        full_df["Close"] = pd.to_numeric(full_df["Close"], errors="coerce")
        full_df = full_df.dropna(subset=["Close"]).sort_index()
        full_df = _append_btc_live_tail(full_df, stale_after_days=stale_after_days)
        full_df = full_df[full_df["Close"] > 0]
        full_df["AbsDays"] = (full_df.index - GENESIS_DATE).days
        full_df["LogClose"] = np.log10(full_df["Close"])
        return full_df

    if source == "auto":
        snapshot_df = _read_snapshot_dataframe("prepared_price_data")
        if snapshot_df is not None and _validate_prepared_price_data(snapshot_df):
            merged_df = _append_btc_live_tail(
                snapshot_df[["Close"]].copy(),
                stale_after_days=stale_after_days,
            )
            merged_df = merged_df[merged_df["Close"] > 0]
            merged_df["AbsDays"] = (merged_df.index - GENESIS_DATE).days
            merged_df["LogClose"] = np.log10(merged_df["Close"])
            return merged_df

    return _load_snapshot_or_live(
        "prepared_price_data",
        _validate_prepared_price_data,
        lambda: _load_source_adapter(
            DataFrameSourceAdapter(
                cache_key="prepared_price_data",
                refresh_seconds=FAST_REFRESH_SECONDS,
                fetch_fn=fetch_price_data,
                validator_fn=_validate_prepared_price_data,
            )
        ),
        source=source,
    )


@st.cache_data(ttl=3600)
def load_prepared_miner_revenue_data(revenue_history_url=MINER_REVENUE_CSV_URL, source="auto"):
    return _load_snapshot_or_live(
        "prepared_miner_revenue_data",
        _validate_prepared_chart_data,
        lambda: _load_source_adapter(
            _build_blockchain_chart_adapter("prepared_miner_revenue_data", revenue_history_url)
        ),
        source=source,
    )


@st.cache_data(ttl=3600)
def load_prepared_difficulty_data(difficulty_history_url=DIFFICULTY_CSV_URL, source="auto"):
    return _load_snapshot_or_live(
        "prepared_difficulty_data",
        _validate_prepared_chart_data,
        lambda: _load_source_adapter(
            _build_blockchain_chart_adapter("prepared_difficulty_data", difficulty_history_url)
        ),
        source=source,
    )


@st.cache_data(ttl=3600)
def load_prepared_hashrate_data(hashrate_history_url=HASHRATE_CSV_URL, source="auto"):
    return _load_snapshot_or_live(
        "prepared_hashrate_data",
        _validate_prepared_chart_data,
        lambda: _load_source_adapter(
            _build_blockchain_chart_adapter("prepared_hashrate_data", hashrate_history_url)
        ),
        source=source,
    )


@st.cache_data(ttl=3600)
def load_prepared_us_m2_data(m2_history_url=FRED_M2SL_CSV_URL, source="auto"):
    return _load_snapshot_or_live(
        "prepared_us_m2_data",
        _validate_prepared_us_m2_data,
        lambda: _load_source_adapter(
            _build_normalized_csv_adapter(
                "prepared_us_m2_data",
                m2_history_url,
                refresh_seconds=REFERENCE_REFRESH_SECONDS,
                validator_fn=_validate_prepared_us_m2_data,
            )
        ),
        source=source,
    )


@st.cache_data(ttl=3600)
def load_bitcoin_visuals_daily_data(data_url=BITCOIN_VISUALS_DAILY_CSV_URL, source="auto"):
    def fetch_daily_data():
        data_df = _fetch_csv_with_retry(data_url)
        if data_df.empty:
            raise ValueError("Unable to load Bitcoin Visuals daily data.")
        return data_df

    return _load_snapshot_or_live(
        "bitcoin_visuals_daily_data",
        _validate_bitcoin_visuals_daily_data,
        lambda: _load_source_adapter(
            DataFrameSourceAdapter(
                cache_key="bitcoin_visuals_daily_data",
                refresh_seconds=SLOW_REFRESH_SECONDS,
                fetch_fn=fetch_daily_data,
                validator_fn=_validate_bitcoin_visuals_daily_data,
            )
        ),
        source=source,
    )


def _load_prepared_lightning_series(
    value_column_name, data_url=BITCOIN_VISUALS_DAILY_CSV_URL, *, source="auto"
):
    daily_df = load_bitcoin_visuals_daily_data(data_url, source=source)
    if "day" not in daily_df.columns or value_column_name not in daily_df.columns:
        raise ValueError(f"Missing required columns for Lightning series: day, {value_column_name}")

    series_df = daily_df[["day", value_column_name]].copy()
    series_df.columns = ["Date", "Close"]
    return _normalize_chart_csv(series_df, "Close")


@st.cache_data(ttl=3600)
def load_prepared_lightning_nodes_data(
    lightning_data_url=BITCOIN_VISUALS_DAILY_CSV_URL, source="auto"
):
    return _load_snapshot_or_live(
        "prepared_lightning_nodes_data",
        _validate_prepared_chart_data,
        lambda: _load_prepared_lightning_series(
            "nodes_with_channels", lightning_data_url, source=source
        ),
        source=source,
    )


@st.cache_data(ttl=3600)
def load_prepared_lightning_capacity_data(
    lightning_data_url=BITCOIN_VISUALS_DAILY_CSV_URL, source="auto"
):
    return _load_snapshot_or_live(
        "prepared_lightning_capacity_data",
        _validate_prepared_chart_data,
        lambda: _load_prepared_lightning_series(
            "capacity_total", lightning_data_url, source=source
        ),
        source=source,
    )


@st.cache_data(ttl=3600)
def load_prepared_filecoin_btc_data(start_date="2020-01-01", source="auto"):
    def fetch_filecoin_btc_data():
        return _fetch_prepared_crypto_btc_data("FIL", start_date, "Filecoin/BTC")

    return _load_snapshot_or_live(
        "prepared_filecoin_btc_data",
        _validate_prepared_filecoin_btc_data,
        lambda: _load_source_adapter(
            DataFrameSourceAdapter(
                cache_key="prepared_filecoin_btc_data",
                refresh_seconds=FAST_REFRESH_SECONDS,
                fetch_fn=fetch_filecoin_btc_data,
                validator_fn=_validate_prepared_filecoin_btc_data,
            )
        ),
        source=source,
    )


def _fetch_prepared_crypto_btc_data(fsym, start_date, label):
    close_series = _safe_download_cryptocompare_histoday(fsym, "BTC")
    if not close_series.empty:
        close_series = close_series[close_series.index >= pd.Timestamp(start_date)]
    if close_series.empty:
        close_series = _safe_download_crypto_btc_via_usd(fsym, start_date)
    if close_series.empty:
        close_series = _safe_download_crypto_btc_via_coinlore(fsym, start_date)
    if close_series.empty:
        raise ValueError(f"Unable to load {label} history.")

    points_df = close_series.to_frame(name="Close").reset_index()
    points_df.columns = ["Date", "Close"]
    return _normalize_chart_csv(points_df, "Close")


@st.cache_data(ttl=3600)
def load_prepared_monero_btc_data(start_date="2014-01-01", source="auto"):
    def fetch_monero_btc_data():
        try:
            close_series = _safe_download_cryptocompare_histoday("XMR", "BTC")
            if not close_series.empty:
                close_series = close_series[close_series.index >= pd.Timestamp(start_date)]
            if close_series.empty:
                close_series = _safe_download_crypto_btc_via_usd("XMR", start_date)
            if close_series.empty:
                close_series = _safe_download_monero_btc_via_coinlore(start_date)
            if close_series.empty:
                raise ValueError("Unable to load Monero/BTC history.")

            points_df = close_series.to_frame(name="Close").reset_index()
            points_df.columns = ["Date", "Close"]
            return _normalize_chart_csv(points_df, "Close")
        except Exception:
            cached_df = _load_cached_dataframe_snapshot(
                ["prepared_monero_btc_data_v2", "prepared_monero_btc_data"],
                validator_fn=_validate_prepared_chart_data,
            )
            if cached_df is not None:
                return cached_df
            raise

    return _load_snapshot_or_live(
        "prepared_monero_btc_data",
        _validate_prepared_monero_btc_data,
        lambda: _load_source_adapter(
            DataFrameSourceAdapter(
                cache_key="prepared_monero_btc_data_v3",
                refresh_seconds=FAST_REFRESH_SECONDS,
                fetch_fn=fetch_monero_btc_data,
                validator_fn=_validate_prepared_monero_btc_data,
            )
        ),
        source=source,
    )


@st.cache_data(ttl=3600)
def load_prepared_litecoin_btc_data(start_date="2013-01-01", source="auto"):
    def fetch_litecoin_btc_data():
        try:
            return _fetch_prepared_crypto_btc_data("LTC", start_date, "Litecoin/BTC")
        except Exception:
            cached_df = _load_cached_dataframe_snapshot(
                ["prepared_litecoin_btc_data_v2", "prepared_litecoin_btc_data"],
                validator_fn=_validate_prepared_chart_data,
            )
            if cached_df is not None:
                return cached_df
            raise

    return _load_snapshot_or_live(
        "prepared_litecoin_btc_data",
        _validate_prepared_litecoin_btc_data,
        lambda: _load_source_adapter(
            DataFrameSourceAdapter(
                cache_key="prepared_litecoin_btc_data_v3",
                refresh_seconds=FAST_REFRESH_SECONDS,
                fetch_fn=fetch_litecoin_btc_data,
                validator_fn=_validate_prepared_litecoin_btc_data,
            )
        ),
        source=source,
    )


@st.cache_data(ttl=3600)
def load_prepared_dogecoin_btc_data(start_date="2014-01-01", source="auto"):
    def fetch_dogecoin_btc_data():
        try:
            return _fetch_prepared_crypto_btc_data("DOGE", start_date, "Dogecoin/BTC")
        except Exception:
            cached_df = _load_cached_dataframe_snapshot(
                [
                    "prepared_dogecoin_btc_data_v3",
                    "prepared_dogecoin_btc_data_v2",
                    "prepared_dogecoin_btc_data",
                ],
                validator_fn=_validate_prepared_chart_data,
            )
            if cached_df is not None:
                return cached_df
            raise

    return _load_snapshot_or_live(
        "prepared_dogecoin_btc_data",
        _validate_prepared_dogecoin_btc_data,
        lambda: _load_source_adapter(
            DataFrameSourceAdapter(
                cache_key="prepared_dogecoin_btc_data_v4",
                refresh_seconds=FAST_REFRESH_SECONDS,
                fetch_fn=fetch_dogecoin_btc_data,
                validator_fn=_validate_prepared_dogecoin_btc_data,
            )
        ),
        source=source,
    )


@st.cache_data(ttl=3600)
def load_prepared_liquid_btc_data(
    liquid_reserves_month_url=LIQUID_RESERVES_MONTH_URL,
    liquid_reserves_url=LIQUID_RESERVES_URL,
    source="auto",
):
    def fetch_liquid_btc_data():
        month_payload = _fetch_json_with_retry(liquid_reserves_month_url)
        if not isinstance(month_payload, list) or len(month_payload) == 0:
            raise ValueError("Unable to load Liquid monthly reserve history.")

        month_df = pd.DataFrame(month_payload)
        if "date" not in month_df.columns or "amount" not in month_df.columns:
            raise ValueError("Liquid monthly reserve payload is missing required columns.")

        month_df = month_df[["date", "amount"]].copy()
        month_df.columns = ["Date", "Sats"]
        month_df["Date"] = pd.to_datetime(month_df["Date"], errors="coerce")
        month_df["Sats"] = pd.to_numeric(month_df["Sats"], errors="coerce")
        month_df = month_df.dropna(subset=["Date", "Sats"])
        month_df["Close"] = month_df["Sats"] / 1e8

        reserves_payload = _fetch_json_with_retry(liquid_reserves_url)
        if isinstance(reserves_payload, dict) and "amount" in reserves_payload:
            current_btc = float(
                pd.to_numeric(reserves_payload.get("amount"), errors="coerce") / 1e8
            )
            if current_btc > 0:
                month_df = pd.concat(
                    [
                        month_df,
                        pd.DataFrame(
                            {
                                "Date": [pd.Timestamp.utcnow().tz_localize(None).normalize()],
                                "Sats": [current_btc * 1e8],
                                "Close": [current_btc],
                            }
                        ),
                    ],
                    ignore_index=True,
                )

        points_df = month_df[["Date", "Close"]].copy()
        prepared_df = _normalize_chart_csv(points_df, "Close")
        if prepared_df.empty:
            raise ValueError("Liquid BTC reserve series is empty after normalization.")

        last_known_day = prepared_df.index.max().normalize()
        utc_today = pd.Timestamp.utcnow().tz_localize(None).normalize()
        full_daily_index = pd.date_range(
            prepared_df.index.min().normalize(), max(last_known_day, utc_today), freq="D"
        )

        daily_df = prepared_df[["Close"]].copy()
        daily_df.index = daily_df.index.normalize()
        daily_df = daily_df[~daily_df.index.duplicated(keep="last")]
        daily_df = daily_df.reindex(full_daily_index).ffill().dropna()

        daily_df["AbsDays"] = (daily_df.index - GENESIS_DATE).days
        daily_df["LogClose"] = np.log10(daily_df["Close"])
        return daily_df

    return _load_snapshot_or_live(
        "prepared_liquid_btc_data",
        _validate_prepared_liquid_btc_data,
        lambda: _load_source_adapter(
            DataFrameSourceAdapter(
                cache_key="prepared_liquid_btc_data",
                refresh_seconds=SLOW_REFRESH_SECONDS,
                fetch_fn=fetch_liquid_btc_data,
                validator_fn=_validate_prepared_liquid_btc_data,
            )
        ),
        source=source,
    )


@st.cache_data(ttl=3600)
def load_prepared_liquid_transactions_data(
    liquid_charts_data_url=LIQUID_CHARTS_DATA_URL, source="auto"
):
    def fetch_liquid_transactions_data():
        payload = _fetch_json_with_retry(liquid_charts_data_url)
        if not isinstance(payload, dict) or "data" not in payload:
            raise ValueError("Unable to load Liquid charts data.")

        charts_data = payload["data"]
        weekly_transactions = charts_data.get("Weekly Transactions")
        if not isinstance(weekly_transactions, list) or len(weekly_transactions) == 0:
            raise ValueError("Liquid weekly transactions payload is missing data.")

        transactions_df = pd.DataFrame(weekly_transactions)
        if "Start Time" not in transactions_df.columns or "TX Total" not in transactions_df.columns:
            raise ValueError("Liquid weekly transactions payload is missing required columns.")

        transactions_df = transactions_df[["Start Time", "TX Total"]].copy()
        transactions_df.columns = ["Date", "Close"]
        return _normalize_chart_csv(transactions_df, "Close")

    return _load_snapshot_or_live(
        "prepared_liquid_transactions_data",
        _validate_prepared_liquid_transactions_data,
        lambda: _load_source_adapter(
            DataFrameSourceAdapter(
                cache_key="prepared_liquid_transactions_data",
                refresh_seconds=SLOW_REFRESH_SECONDS,
                fetch_fn=fetch_liquid_transactions_data,
                validator_fn=_validate_prepared_liquid_transactions_data,
            )
        ),
        source=source,
    )
