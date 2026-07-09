"""Lazy loading and normalization of the time series used by the UI."""

import numpy as np
import streamlit as st

from core.constants import (
    CURRENCY_DOLLAR,
    CURRENCY_OPTIONS,
    KEY_BITCOIN_NETWORK_SIMULATION_RESOLUTION,
    KEY_BITCOIN_NETWORK_SIMULATION_SEED,
    KEY_CURRENCY_SELECTOR,
    POWERLAW_SERIES_BITCOIN_MARKET_CAP,
    POWERLAW_SERIES_BITCOIN_NETWORK_SIMULATION,
    POWERLAW_SERIES_BITCOIN_VOLATILITY,
    POWERLAW_SERIES_DIFFICULTY,
    POWERLAW_SERIES_DOGECOIN_BTC,
    POWERLAW_SERIES_FILECOIN_BTC,
    POWERLAW_SERIES_HASHRATE,
    POWERLAW_SERIES_LIGHTNING_CAPACITY,
    POWERLAW_SERIES_LIGHTNING_NODES,
    POWERLAW_SERIES_LIQUID_BTC,
    POWERLAW_SERIES_LIQUID_TRANSACTIONS,
    POWERLAW_SERIES_LITECOIN_BTC,
    POWERLAW_SERIES_MONERO_BTC,
    POWERLAW_SERIES_PRICE,
    POWERLAW_SERIES_REVENUE,
    POWERLAW_SERIES_US_M2,
    POWERLAW_SERIES_USDT_SUPPLY,
)
from core.simulation import build_bitcoin_network_simulation
from services.price_service import (
    build_currency_close_series,
    build_prepared_bitcoin_market_cap_data,
    build_prepared_bitcoin_volatility_data,
    get_runtime_data_source,
    load_prepared_bitcoin_supply_data,
    load_prepared_difficulty_data,
    load_prepared_dogecoin_btc_data,
    load_prepared_filecoin_btc_data,
    load_prepared_hashrate_data,
    load_prepared_lightning_capacity_data,
    load_prepared_lightning_nodes_data,
    load_prepared_litecoin_btc_data,
    load_prepared_liquid_btc_data,
    load_prepared_liquid_transactions_data,
    load_prepared_miner_revenue_data,
    load_prepared_monero_btc_data,
    load_prepared_price_data,
    load_prepared_us_m2_data,
    load_prepared_usdt_supply_data,
)

SERIES_LOADERS = {
    POWERLAW_SERIES_PRICE: ("BTC price", load_prepared_price_data),
    POWERLAW_SERIES_REVENUE: ("miner revenue", load_prepared_miner_revenue_data),
    POWERLAW_SERIES_DIFFICULTY: ("difficulty", load_prepared_difficulty_data),
    POWERLAW_SERIES_HASHRATE: ("hashrate", load_prepared_hashrate_data),
    POWERLAW_SERIES_LIGHTNING_NODES: ("Lightning node", load_prepared_lightning_nodes_data),
    POWERLAW_SERIES_LIGHTNING_CAPACITY: (
        "Lightning capacity",
        load_prepared_lightning_capacity_data,
    ),
    POWERLAW_SERIES_LIQUID_BTC: ("Liquid BTC", load_prepared_liquid_btc_data),
    POWERLAW_SERIES_LIQUID_TRANSACTIONS: (
        "Liquid transactions",
        load_prepared_liquid_transactions_data,
    ),
    POWERLAW_SERIES_FILECOIN_BTC: ("Filecoin/BTC", load_prepared_filecoin_btc_data),
    POWERLAW_SERIES_MONERO_BTC: ("Monero/BTC", load_prepared_monero_btc_data),
    POWERLAW_SERIES_LITECOIN_BTC: ("Litecoin/BTC", load_prepared_litecoin_btc_data),
    POWERLAW_SERIES_DOGECOIN_BTC: ("Dogecoin/BTC", load_prepared_dogecoin_btc_data),
    POWERLAW_SERIES_US_M2: ("U.S. M2", load_prepared_us_m2_data),
    POWERLAW_SERIES_USDT_SUPPLY: ("USDT supply", load_prepared_usdt_supply_data),
}


def normalize_close_frame(data_df):
    normalized_df = data_df[data_df["Close"] > 0].copy()
    normalized_df["LogClose"] = np.log10(normalized_df["Close"])
    return normalized_df


@st.cache_data(ttl=3600)
def _prepare_bitcoin_network_simulation(base_df, seed, resolution_days):
    return build_bitcoin_network_simulation(
        base_df,
        seed=int(seed),
        resolution_days=float(resolution_days),
    )


class SeriesFrameStore:
    """Keeps one normalized DataFrame per selected series for a Streamlit rerun."""

    def __init__(self, data_source=None):
        self._frames = {}
        self._data_source = data_source or get_runtime_data_source()

    @property
    def data_source(self):
        return self._data_source

    def get(self, series_name):
        if series_name not in self._frames:
            self._frames[series_name] = self._load(series_name)
        return self._frames[series_name]

    def _load(self, series_name):
        derived_series = {
            POWERLAW_SERIES_BITCOIN_VOLATILITY,
            POWERLAW_SERIES_BITCOIN_MARKET_CAP,
            POWERLAW_SERIES_BITCOIN_NETWORK_SIMULATION,
        }
        if series_name not in SERIES_LOADERS and series_name not in derived_series:
            st.error(f"Unknown data series: {series_name}")
            st.stop()

        try:
            if series_name == POWERLAW_SERIES_BITCOIN_VOLATILITY:
                return normalize_close_frame(
                    build_prepared_bitcoin_volatility_data(self.get(POWERLAW_SERIES_PRICE))
                )
            if series_name == POWERLAW_SERIES_BITCOIN_MARKET_CAP:
                return normalize_close_frame(
                    build_prepared_bitcoin_market_cap_data(
                        self.get(POWERLAW_SERIES_PRICE),
                        load_prepared_bitcoin_supply_data(source=self._data_source),
                    )
                )
            if series_name == POWERLAW_SERIES_BITCOIN_NETWORK_SIMULATION:
                return normalize_close_frame(
                    _prepare_bitcoin_network_simulation(
                        self.get(POWERLAW_SERIES_PRICE),
                        seed=int(st.session_state.get(KEY_BITCOIN_NETWORK_SIMULATION_SEED, 1)),
                        resolution_days=float(
                            st.session_state.get(KEY_BITCOIN_NETWORK_SIMULATION_RESOLUTION, 0.00001)
                        ),
                    )
                )

            label, loader = SERIES_LOADERS[series_name]
            return normalize_close_frame(loader(source=self._data_source))
        except Exception as exc:
            label = SERIES_LOADERS.get(series_name, (series_name, None))[0]
            st.error(f"Error loading {label} data: {exc}")
            st.stop()


class SidebarSeriesData:
    def __init__(self, series_store):
        self._series_store = series_store

    def __getitem__(self, series_name):
        data_df = self._series_store.get(series_name)
        if series_name == POWERLAW_SERIES_PRICE:
            currency = st.session_state.get(KEY_CURRENCY_SELECTOR, CURRENCY_DOLLAR)
            if currency not in CURRENCY_OPTIONS:
                currency = CURRENCY_DOLLAR
            price_close = build_currency_close_series(
                data_df,
                currency,
                source=self._series_store.data_source,
            )
            price_close = price_close[price_close > 0]
            return {
                "absolute_days": data_df.loc[price_close.index, "AbsDays"].values,
                "close": price_close.values,
                "log_close": np.log10(price_close.values),
                "date_index": price_close.index,
            }

        return {
            "absolute_days": data_df["AbsDays"].values,
            "close": data_df["Close"].values,
            "log_close": data_df["LogClose"].values,
            "date_index": data_df.index,
        }
