import unittest
from contextlib import ExitStack
from unittest.mock import patch

from core.constants import (
    CURRENCY_ALUMINUM,
    CURRENCY_CHF,
    CURRENCY_COPPER,
    CURRENCY_IRON,
    CURRENCY_NDAQ,
    CURRENCY_OIL,
    CURRENCY_RUB,
    CURRENCY_SILVER,
    CURRENCY_SP500,
    CURRENCY_UAH,
    CURRENCY_US_HOUSING,
    POWERLAW_SERIES_BITCOIN_VOLATILITY,
    POWERLAW_SERIES_BITCOIN_MARKET_CAP,
    POWERLAW_SERIES_PRICE,
)
from scripts import update_powerlaw_defaults
from scripts.update_powerlaw_defaults import DEFAULT_CASES, update_constants_content


class TestUpdateDefaultsScript(unittest.TestCase):
    def test_default_cases_include_bitcoin_volatility(self):
        volatility_cases = [
            case for case in DEFAULT_CASES if case[0] == POWERLAW_SERIES_BITCOIN_VOLATILITY
        ]

        self.assertEqual(len(volatility_cases), 1)
        self.assertEqual(volatility_cases[0][2], "DEFAULT_BITCOIN_VOLATILITY_A")
        self.assertEqual(volatility_cases[0][3], "DEFAULT_BITCOIN_VOLATILITY_B")

    def test_default_cases_include_bitcoin_market_cap(self):
        self.assertIn(
            (
                POWERLAW_SERIES_BITCOIN_MARKET_CAP,
                update_powerlaw_defaults.CURRENCY_DOLLAR,
                "DEFAULT_BITCOIN_MARKET_CAP_A",
                "DEFAULT_BITCOIN_MARKET_CAP_B",
            ),
            DEFAULT_CASES,
        )

    def test_default_cases_include_uah_price_model(self):
        self.assertIn(
            (POWERLAW_SERIES_PRICE, CURRENCY_UAH, "DEFAULT_UAH_A", "DEFAULT_UAH_B"),
            DEFAULT_CASES,
        )

    def test_default_cases_include_chf_price_model(self):
        self.assertIn(
            (POWERLAW_SERIES_PRICE, CURRENCY_CHF, "DEFAULT_CHF_A", "DEFAULT_CHF_B"),
            DEFAULT_CASES,
        )

    def test_default_cases_include_rub_price_model(self):
        self.assertIn(
            (POWERLAW_SERIES_PRICE, CURRENCY_RUB, "DEFAULT_RUB_A", "DEFAULT_RUB_B"),
            DEFAULT_CASES,
        )

    def test_default_cases_include_silver_price_model(self):
        self.assertIn(
            (
                POWERLAW_SERIES_PRICE,
                CURRENCY_SILVER,
                "DEFAULT_SILVER_A",
                "DEFAULT_SILVER_B",
            ),
            DEFAULT_CASES,
        )

    def test_default_cases_include_copper_price_model(self):
        self.assertIn(
            (
                POWERLAW_SERIES_PRICE,
                CURRENCY_COPPER,
                "DEFAULT_COPPER_A",
                "DEFAULT_COPPER_B",
            ),
            DEFAULT_CASES,
        )

    def test_default_cases_include_iron_price_model(self):
        self.assertIn(
            (POWERLAW_SERIES_PRICE, CURRENCY_IRON, "DEFAULT_IRON_A", "DEFAULT_IRON_B"),
            DEFAULT_CASES,
        )

    def test_default_cases_include_aluminum_price_model(self):
        self.assertIn(
            (
                POWERLAW_SERIES_PRICE,
                CURRENCY_ALUMINUM,
                "DEFAULT_ALUMINUM_A",
                "DEFAULT_ALUMINUM_B",
            ),
            DEFAULT_CASES,
        )

    def test_default_cases_include_oil_price_model(self):
        self.assertIn(
            (POWERLAW_SERIES_PRICE, CURRENCY_OIL, "DEFAULT_OIL_A", "DEFAULT_OIL_B"),
            DEFAULT_CASES,
        )

    def test_default_cases_include_us_housing_price_model(self):
        self.assertIn(
            (
                POWERLAW_SERIES_PRICE,
                CURRENCY_US_HOUSING,
                "DEFAULT_US_HOUSING_A",
                "DEFAULT_US_HOUSING_B",
            ),
            DEFAULT_CASES,
        )

    def test_default_cases_include_sp500_price_model(self):
        self.assertIn(
            (
                POWERLAW_SERIES_PRICE,
                CURRENCY_SP500,
                "DEFAULT_SP500_A",
                "DEFAULT_SP500_B",
            ),
            DEFAULT_CASES,
        )

    def test_default_cases_include_ndaq_price_model(self):
        self.assertIn(
            (POWERLAW_SERIES_PRICE, CURRENCY_NDAQ, "DEFAULT_NDAQ_A", "DEFAULT_NDAQ_B"),
            DEFAULT_CASES,
        )

    def test_load_series_frames_builds_bitcoin_volatility_from_price_data(self):
        update_powerlaw_defaults.load_prepared_price_data.clear()
        price_df = update_powerlaw_defaults.load_prepared_price_data(source="snapshot").head(120)

        loader_names = [
            "load_prepared_price_data",
            "load_prepared_miner_revenue_data",
            "load_prepared_bitcoin_supply_data",
            "load_prepared_difficulty_data",
            "load_prepared_hashrate_data",
            "load_prepared_lightning_nodes_data",
            "load_prepared_lightning_capacity_data",
            "load_prepared_liquid_btc_data",
            "load_prepared_liquid_transactions_data",
            "load_prepared_filecoin_btc_data",
            "load_prepared_monero_btc_data",
            "load_prepared_litecoin_btc_data",
            "load_prepared_dogecoin_btc_data",
            "load_prepared_us_m2_data",
            "load_prepared_usdt_supply_data",
        ]
        with ExitStack() as stack:
            loader_mocks = {}
            for loader_name in loader_names:
                loader_mocks[loader_name] = stack.enter_context(
                    patch.object(update_powerlaw_defaults, loader_name, return_value=price_df)
                )
            series_frames = update_powerlaw_defaults._load_series_frames()

        for loader_mock in loader_mocks.values():
            loader_mock.assert_called_once_with(source="snapshot")
        self.assertIn(POWERLAW_SERIES_BITCOIN_VOLATILITY, series_frames)
        self.assertIn(POWERLAW_SERIES_BITCOIN_MARKET_CAP, series_frames)
        self.assertGreater(len(series_frames[POWERLAW_SERIES_BITCOIN_VOLATILITY]), 0)
        self.assertGreater(len(series_frames[POWERLAW_SERIES_BITCOIN_MARKET_CAP]), 0)

    def test_prepare_fit_frame_uses_snapshot_currency_references(self):
        update_powerlaw_defaults.load_prepared_price_data.clear()
        price_df = update_powerlaw_defaults.load_prepared_price_data(source="snapshot").head(120)

        with patch.object(
            update_powerlaw_defaults,
            "build_currency_close_series",
            return_value=price_df["Close"],
        ) as build_currency_close_series:
            update_powerlaw_defaults._prepare_fit_frame(
                POWERLAW_SERIES_PRICE,
                update_powerlaw_defaults.CURRENCY_EURO,
                {POWERLAW_SERIES_PRICE: price_df},
            )

        self.assertEqual(build_currency_close_series.call_args.kwargs["source"], "snapshot")

    def test_update_constants_content_updates_scalars(self):
        original = "DEFAULT_A = -16.511\n"
        updated = update_constants_content(
            original,
            {"DEFAULT_A": "-16.400"},
        )

        self.assertIn("DEFAULT_A = -16.400", updated)


if __name__ == "__main__":
    unittest.main()
