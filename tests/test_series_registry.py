import unittest

from core.constants import (
    CURRENCY_DOLLAR,
    CURRENCY_EURO,
    DIFFICULTY_HASHRATE_ANALYSIS_START_ABS_DAYS,
    MODE_LOGPERIODIC,
    MODE_PORTFOLIO,
    MODE_POWERLAW,
    POWERLAW_SERIES_DIFFICULTY,
    POWERLAW_SERIES_HASHRATE,
    POWERLAW_SERIES_LIQUID_TRANSACTIONS,
    POWERLAW_SERIES_PRICE,
    POWERLAW_SERIES_REVENUE,
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
            get_powerlaw_series_group_for_series(POWERLAW_SERIES_HASHRATE), "Bitcoin network"
        )
        self.assertEqual(
            get_powerlaw_series_group_for_series(POWERLAW_SERIES_LIQUID_TRANSACTIONS),
            "Liquid",
        )
        self.assertIn(POWERLAW_SERIES_HASHRATE, group_map["Bitcoin network"])

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

    def test_session_defaults_include_price_and_series_specific_models(self):
        defaults = dict(iter_session_model_defaults())

        self.assertIn("A_price", defaults)
        self.assertIn("B_price", defaults)
        self.assertIn("A_euro", defaults)
        self.assertIn("B_euro", defaults)
        self.assertIn("A_liquid_transactions", defaults)
        self.assertIn("B_liquid_transactions", defaults)


if __name__ == "__main__":
    unittest.main()
