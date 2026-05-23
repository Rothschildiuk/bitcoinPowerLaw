import unittest
from contextlib import ExitStack
from unittest.mock import patch

from core.constants import (
    CURRENCY_ALUMINUM,
    CURRENCY_COPPER,
    CURRENCY_IRON,
    CURRENCY_RUB,
    CURRENCY_SILVER,
    CURRENCY_UAH,
    CURRENCY_US_HOUSING,
    POWERLAW_SERIES_BITCOIN_VOLATILITY,
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

    def test_default_cases_include_uah_price_model(self):
        self.assertIn(
            (POWERLAW_SERIES_PRICE, CURRENCY_UAH, "DEFAULT_UAH_A", "DEFAULT_UAH_B"),
            DEFAULT_CASES,
        )

    def test_default_cases_include_rub_price_model(self):
        self.assertIn(
            (POWERLAW_SERIES_PRICE, CURRENCY_RUB, "DEFAULT_RUB_A", "DEFAULT_RUB_B"),
            DEFAULT_CASES,
        )

    def test_default_cases_include_silver_price_model(self):
        self.assertIn(
            (POWERLAW_SERIES_PRICE, CURRENCY_SILVER, "DEFAULT_SILVER_A", "DEFAULT_SILVER_B"),
            DEFAULT_CASES,
        )

    def test_default_cases_include_copper_price_model(self):
        self.assertIn(
            (POWERLAW_SERIES_PRICE, CURRENCY_COPPER, "DEFAULT_COPPER_A", "DEFAULT_COPPER_B"),
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

    def test_oscillator_default_cases_include_us_housing_price_model(self):
        self.assertIn(
            (
                POWERLAW_SERIES_PRICE,
                CURRENCY_US_HOUSING,
                "OSC_DEFAULTS_US_HOUSING",
                update_powerlaw_defaults.OSC_DEFAULTS_US_HOUSING,
            ),
            update_powerlaw_defaults.OSCILLATOR_DEFAULT_CASES,
        )

    def test_load_series_frames_builds_bitcoin_volatility_from_price_data(self):
        update_powerlaw_defaults.load_prepared_price_data.clear()
        price_df = update_powerlaw_defaults.load_prepared_price_data(source="snapshot").head(120)

        loader_names = [
            "load_prepared_price_data",
            "load_prepared_miner_revenue_data",
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
            for loader_name in loader_names:
                stack.enter_context(
                    patch.object(update_powerlaw_defaults, loader_name, return_value=price_df)
                )
            series_frames = update_powerlaw_defaults._load_series_frames()

        self.assertIn(POWERLAW_SERIES_BITCOIN_VOLATILITY, series_frames)
        self.assertGreater(len(series_frames[POWERLAW_SERIES_BITCOIN_VOLATILITY]), 0)

    def test_update_constants_content_updates_scalars_and_mapping_entries(self):
        original = """DEFAULT_A = -16.511
OSC_DEFAULTS_HASHRATE = {
    "lambda_val": 4.71,
    "t1_age": 1.69,
    "harmonic_count": 1,
}
"""
        updated = update_constants_content(
            original,
            {
                "DEFAULT_A": "-16.400",
                "OSC_DEFAULTS_HASHRATE.lambda_val": "5.32",
                "OSC_DEFAULTS_HASHRATE.t1_age": "1.51",
                "OSC_DEFAULTS_HASHRATE.harmonic_count": "3",
            },
        )

        self.assertIn("DEFAULT_A = -16.400", updated)
        self.assertIn('"lambda_val": 5.32,', updated)
        self.assertIn('"t1_age": 1.51,', updated)
        self.assertIn('"harmonic_count": 3,', updated)


if __name__ == "__main__":
    unittest.main()
