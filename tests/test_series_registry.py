import unittest

from core.constants import (
    CURRENCY_DOLLAR,
    CURRENCY_EURO,
    DIFFICULTY_HASHRATE_ANALYSIS_START_ABS_DAYS,
    DOGECOIN_BTC_MODEL_ORIGIN_ABS_DAYS,
    FILECOIN_BTC_MODEL_ORIGIN_ABS_DAYS,
    LIGHTNING_MODEL_ORIGIN_ABS_DAYS,
    LITECOIN_BTC_MODEL_ORIGIN_ABS_DAYS,
    LIQUID_BTC_MODEL_ORIGIN_ABS_DAYS,
    LIQUID_TRANSACTIONS_MODEL_ORIGIN_ABS_DAYS,
    MODE_LOGPERIODIC,
    MODE_PORTFOLIO,
    MODE_POWERLAW,
    POWERLAW_SERIES_DOGECOIN_BTC,
    POWERLAW_SERIES_BITCOIN_NETWORK_SIMULATION,
    POWERLAW_SERIES_DIFFICULTY,
    POWERLAW_SERIES_FILECOIN_BTC,
    POWERLAW_SERIES_HASHRATE,
    POWERLAW_SERIES_LITECOIN_BTC,
    POWERLAW_SERIES_LIGHTNING_CAPACITY,
    POWERLAW_SERIES_LIGHTNING_NODES,
    POWERLAW_SERIES_LIQUID_BTC,
    POWERLAW_SERIES_LIQUID_TRANSACTIONS,
    POWERLAW_SERIES_MONERO_BTC,
    POWERLAW_SERIES_PRICE,
    POWERLAW_SERIES_REVENUE,
    POWERLAW_SERIES_RUSSIAN_M2,
    POWERLAW_SERIES_US_M2,
    MONERO_BTC_MODEL_ORIGIN_ABS_DAYS,
    RUSSIAN_M2_MODEL_ORIGIN_ABS_DAYS,
    US_M2_MODEL_ORIGIN_ABS_DAYS,
)
from core.series_registry import (
    get_active_model_config,
    get_powerlaw_series_group_for_series,
    get_powerlaw_series_group_map,
    get_logperiodic_series_options,
    get_powerlaw_series_groups,
    get_powerlaw_series_options,
    get_selected_series_name,
    iter_session_model_defaults,
    series_supports_currency_selector,
)


class TestSeriesRegistry(unittest.TestCase):
    def assert_default_params_are_within_powerlaw_bounds(self, config):
        self.assertIsNotNone(config.powerlaw_intercept_bounds)
        self.assertIsNotNone(config.powerlaw_slope_bounds)
        self.assertLessEqual(config.powerlaw_intercept_bounds[0], config.default_a)
        self.assertGreaterEqual(config.powerlaw_intercept_bounds[1], config.default_a)
        self.assertLessEqual(config.powerlaw_slope_bounds[0], config.default_b)
        self.assertGreaterEqual(config.powerlaw_slope_bounds[1], config.default_b)

    def test_options_follow_registry_capabilities(self):
        self.assertIn(POWERLAW_SERIES_REVENUE, get_powerlaw_series_options())
        self.assertIn(POWERLAW_SERIES_LIQUID_TRANSACTIONS, get_powerlaw_series_options())
        self.assertIn(POWERLAW_SERIES_DIFFICULTY, get_logperiodic_series_options())
        self.assertIn(POWERLAW_SERIES_HASHRATE, get_logperiodic_series_options())
        self.assertNotIn(POWERLAW_SERIES_LIQUID_TRANSACTIONS, get_logperiodic_series_options())

    def test_powerlaw_series_groups_cover_all_powerlaw_options(self):
        grouped_series = [
            series_name
            for _, series_options in get_powerlaw_series_groups()
            for series_name in series_options
        ]
        self.assertCountEqual(grouped_series, get_powerlaw_series_options())

    def test_powerlaw_series_group_mapping_resolves_group_for_series(self):
        group_map = get_powerlaw_series_group_map()

        self.assertEqual(
            get_powerlaw_series_group_for_series(POWERLAW_SERIES_HASHRATE), "Bitcoin Network"
        )
        self.assertEqual(
            get_powerlaw_series_group_for_series(POWERLAW_SERIES_LIQUID_TRANSACTIONS),
            "Liquid Network",
        )
        self.assertEqual(
            get_powerlaw_series_group_for_series(POWERLAW_SERIES_FILECOIN_BTC), "Shitcoins"
        )
        self.assertEqual(
            get_powerlaw_series_group_for_series(POWERLAW_SERIES_MONERO_BTC), "Shitcoins"
        )
        self.assertEqual(
            get_powerlaw_series_group_for_series(POWERLAW_SERIES_LITECOIN_BTC), "Shitcoins"
        )
        self.assertEqual(
            get_powerlaw_series_group_for_series(POWERLAW_SERIES_DOGECOIN_BTC), "Shitcoins"
        )
        self.assertEqual(get_powerlaw_series_group_for_series(POWERLAW_SERIES_US_M2), "Fiat Money")
        self.assertEqual(
            get_powerlaw_series_group_for_series(POWERLAW_SERIES_RUSSIAN_M2), "Fiat Money"
        )
        self.assertIn(POWERLAW_SERIES_HASHRATE, group_map["Bitcoin Network"])
        self.assertIn(POWERLAW_SERIES_BITCOIN_NETWORK_SIMULATION, group_map["Bitcoin Network"])
        self.assertIn(POWERLAW_SERIES_US_M2, group_map["Fiat Money"])
        self.assertIn(POWERLAW_SERIES_RUSSIAN_M2, group_map["Fiat Money"])

    def test_filecoin_btc_config_uses_btc_units(self):
        filecoin_config = get_active_model_config(
            MODE_POWERLAW,
            POWERLAW_SERIES_FILECOIN_BTC,
            POWERLAW_SERIES_PRICE,
            CURRENCY_DOLLAR,
        )

        self.assertEqual(filecoin_config.currency_unit, "BTC")
        self.assertFalse(filecoin_config.supports_currency_selector)

    def test_shitcoins_use_own_model_origins(self):
        expected_origin_by_series = {
            POWERLAW_SERIES_FILECOIN_BTC: FILECOIN_BTC_MODEL_ORIGIN_ABS_DAYS,
            POWERLAW_SERIES_MONERO_BTC: MONERO_BTC_MODEL_ORIGIN_ABS_DAYS,
            POWERLAW_SERIES_LITECOIN_BTC: LITECOIN_BTC_MODEL_ORIGIN_ABS_DAYS,
            POWERLAW_SERIES_DOGECOIN_BTC: DOGECOIN_BTC_MODEL_ORIGIN_ABS_DAYS,
        }

        for series_name, expected_origin in expected_origin_by_series.items():
            with self.subTest(series_name=series_name):
                series_config = get_active_model_config(
                    MODE_POWERLAW,
                    series_name,
                    POWERLAW_SERIES_PRICE,
                    CURRENCY_DOLLAR,
                )

                self.assertEqual(series_config.model_origin_abs_day, expected_origin)
                self.assert_default_params_are_within_powerlaw_bounds(series_config)

    def test_price_series_config_tracks_selected_currency(self):
        euro_config = get_active_model_config(
            MODE_POWERLAW,
            POWERLAW_SERIES_PRICE,
            POWERLAW_SERIES_PRICE,
            CURRENCY_EURO,
        )

        self.assertEqual(euro_config.currency_prefix, "€")
        self.assertEqual(euro_config.currency_decimals, 2)
        self.assertEqual(euro_config.target_series_unit, CURRENCY_EURO)
        self.assertTrue(euro_config.supports_currency_selector)

    def test_bitcoin_network_simulation_uses_bitcoin_powerlaw_defaults(self):
        sim_config = get_active_model_config(
            MODE_POWERLAW,
            POWERLAW_SERIES_BITCOIN_NETWORK_SIMULATION,
            POWERLAW_SERIES_PRICE,
            CURRENCY_DOLLAR,
        )
        price_config = get_active_model_config(
            MODE_POWERLAW,
            POWERLAW_SERIES_PRICE,
            POWERLAW_SERIES_PRICE,
            CURRENCY_DOLLAR,
        )

        self.assertEqual(sim_config.default_a, price_config.default_a)
        self.assertEqual(sim_config.default_b, price_config.default_b)
        self.assertFalse(sim_config.supports_currency_selector)
        self.assertTrue(sim_config.lock_price_scale_to_log)
        self.assert_default_params_are_within_powerlaw_bounds(sim_config)

    def test_logperiodic_difficulty_uses_filtered_analysis_start(self):
        difficulty_config = get_active_model_config(
            MODE_LOGPERIODIC,
            POWERLAW_SERIES_PRICE,
            POWERLAW_SERIES_DIFFICULTY,
            CURRENCY_DOLLAR,
        )

        self.assertEqual(difficulty_config.series_name, POWERLAW_SERIES_DIFFICULTY)
        self.assertEqual(
            difficulty_config.analysis_min_abs_day,
            DIFFICULTY_HASHRATE_ANALYSIS_START_ABS_DAYS,
        )
        self.assertEqual(difficulty_config.oscillator_parameter_bounds["lambda_val"], (1.5, 8.0))
        self.assertTrue(difficulty_config.lock_price_scale_to_log)

    def test_portfolio_always_uses_bitcoin_price_series(self):
        self.assertEqual(
            get_selected_series_name(
                MODE_PORTFOLIO,
                POWERLAW_SERIES_LIQUID_TRANSACTIONS,
                POWERLAW_SERIES_DIFFICULTY,
            ),
            POWERLAW_SERIES_PRICE,
        )

    def test_non_price_series_disable_currency_selector(self):
        self.assertFalse(
            series_supports_currency_selector(
                MODE_POWERLAW,
                POWERLAW_SERIES_LIQUID_TRANSACTIONS,
                POWERLAW_SERIES_PRICE,
            )
        )
        self.assertTrue(
            series_supports_currency_selector(
                MODE_POWERLAW,
                POWERLAW_SERIES_PRICE,
                POWERLAW_SERIES_PRICE,
            )
        )

    def test_liquid_transactions_uses_own_model_origin(self):
        liquid_config = get_active_model_config(
            MODE_POWERLAW,
            POWERLAW_SERIES_LIQUID_TRANSACTIONS,
            POWERLAW_SERIES_PRICE,
            CURRENCY_DOLLAR,
        )

        self.assertEqual(
            liquid_config.model_origin_abs_day,
            LIQUID_TRANSACTIONS_MODEL_ORIGIN_ABS_DAYS,
        )
        self.assert_default_params_are_within_powerlaw_bounds(liquid_config)

    def test_lightning_and_liquid_btc_use_own_model_origins(self):
        lightning_nodes_config = get_active_model_config(
            MODE_POWERLAW,
            POWERLAW_SERIES_LIGHTNING_NODES,
            POWERLAW_SERIES_PRICE,
            CURRENCY_DOLLAR,
        )
        lightning_btc_config = get_active_model_config(
            MODE_POWERLAW,
            POWERLAW_SERIES_LIGHTNING_CAPACITY,
            POWERLAW_SERIES_PRICE,
            CURRENCY_DOLLAR,
        )
        liquid_btc_config = get_active_model_config(
            MODE_POWERLAW,
            POWERLAW_SERIES_LIQUID_BTC,
            POWERLAW_SERIES_PRICE,
            CURRENCY_DOLLAR,
        )

        self.assertEqual(
            lightning_nodes_config.model_origin_abs_day,
            LIGHTNING_MODEL_ORIGIN_ABS_DAYS,
        )
        self.assertEqual(
            lightning_btc_config.model_origin_abs_day,
            LIGHTNING_MODEL_ORIGIN_ABS_DAYS,
        )
        self.assertEqual(
            liquid_btc_config.model_origin_abs_day,
            LIQUID_BTC_MODEL_ORIGIN_ABS_DAYS,
        )
        self.assert_default_params_are_within_powerlaw_bounds(lightning_nodes_config)
        self.assert_default_params_are_within_powerlaw_bounds(lightning_btc_config)
        self.assert_default_params_are_within_powerlaw_bounds(liquid_btc_config)

    def test_us_m2_config_uses_fred_billions_units(self):
        m2_config = get_active_model_config(
            MODE_POWERLAW,
            POWERLAW_SERIES_US_M2,
            POWERLAW_SERIES_PRICE,
            CURRENCY_DOLLAR,
        )

        self.assertEqual(m2_config.target_series_name, "U.S. M2 money supply")
        self.assertEqual(m2_config.target_series_unit, "Billions USD")
        self.assertEqual(m2_config.currency_suffix, "B")
        self.assertFalse(m2_config.supports_currency_selector)
        self.assertTrue(m2_config.lock_price_scale_to_log)
        self.assertEqual(m2_config.model_origin_abs_day, US_M2_MODEL_ORIGIN_ABS_DAYS)
        self.assert_default_params_are_within_powerlaw_bounds(m2_config)

    def test_russian_m2_config_uses_cbr_trillion_rub_units(self):
        m2_config = get_active_model_config(
            MODE_POWERLAW,
            POWERLAW_SERIES_RUSSIAN_M2,
            POWERLAW_SERIES_PRICE,
            CURRENCY_DOLLAR,
        )

        self.assertEqual(m2_config.target_series_name, "Russian M2 money supply")
        self.assertEqual(m2_config.target_series_unit, "Trillion RUB")
        self.assertEqual(m2_config.currency_prefix, "₽")
        self.assertEqual(m2_config.currency_suffix, "T")
        self.assertFalse(m2_config.supports_currency_selector)
        self.assertTrue(m2_config.lock_price_scale_to_log)
        self.assertEqual(m2_config.model_origin_abs_day, RUSSIAN_M2_MODEL_ORIGIN_ABS_DAYS)
        self.assert_default_params_are_within_powerlaw_bounds(m2_config)

    def test_session_defaults_include_price_and_series_specific_models(self):
        defaults = dict(iter_session_model_defaults())

        self.assertIn("A_price", defaults)
        self.assertIn("B_price", defaults)
        self.assertIn("A_euro", defaults)
        self.assertIn("B_euro", defaults)
        self.assertIn("A_liquid_transactions", defaults)
        self.assertIn("B_liquid_transactions", defaults)
        self.assertIn("A_us_m2", defaults)
        self.assertIn("B_us_m2", defaults)
        self.assertIn("A_russian_m2", defaults)
        self.assertIn("B_russian_m2", defaults)
        self.assertIn("A_bitcoin_network_simulation", defaults)
        self.assertIn("B_bitcoin_network_simulation", defaults)


if __name__ == "__main__":
    unittest.main()
