import unittest

from core.constants import (
    CURRENCY_ALUMINUM,
    CURRENCY_CHF,
    CURRENCY_COPPER,
    CURRENCY_EU_ELECTRICITY,
    CURRENCY_US_ELECTRICITY,
    CURRENCY_DOLLAR,
    CURRENCY_EURO,
    CURRENCY_GOLD,
    CURRENCY_IRON,
    CURRENCY_NDAQ,
    CURRENCY_NVDA,
    CURRENCY_AAPL,
    CURRENCY_MSFT,
    CURRENCY_AMZN,
    CURRENCY_GOOGL,
    CURRENCY_GROUPS,
    CURRENCY_KO,
    CURRENCY_OIL,
    CURRENCY_OPTIONS,
    CURRENCY_SILVER,
    CURRENCY_SP500,
    CURRENCY_UAH,
    CURRENCY_US_HOUSING,
    POWERLAW_SERIES_BITCOIN_MARKET_CAP,
    POWERLAW_SERIES_BITCOIN_NETWORK_SIMULATION,
    POWERLAW_SERIES_BITCOIN_VOLATILITY,
    POWERLAW_SERIES_LIGHTNING_CAPACITY,
    POWERLAW_SERIES_LIGHTNING_NODES,
    POWERLAW_SERIES_LIQUID_BTC,
    POWERLAW_SERIES_LIQUID_TRANSACTIONS,
    POWERLAW_SERIES_OPTIONS,
    POWERLAW_SERIES_USDT_SUPPLY,
)


class TestConstantsSeries(unittest.TestCase):
    def test_currency_options_follow_requested_order(self):
        self.assertEqual(
            CURRENCY_OPTIONS,
            [
                CURRENCY_EURO,
                CURRENCY_DOLLAR,
                CURRENCY_CHF,
                CURRENCY_UAH,
                CURRENCY_GOLD,
                CURRENCY_SILVER,
                CURRENCY_COPPER,
                CURRENCY_ALUMINUM,
                CURRENCY_IRON,
                CURRENCY_OIL,
                CURRENCY_US_ELECTRICITY,
                CURRENCY_EU_ELECTRICITY,
                CURRENCY_US_HOUSING,
                CURRENCY_SP500,
                CURRENCY_NDAQ,
                CURRENCY_NVDA,
                CURRENCY_AAPL,
                CURRENCY_MSFT,
                CURRENCY_AMZN,
                CURRENCY_GOOGL,
                CURRENCY_KO,
            ],
        )

    def test_currency_groups_cover_each_quote_unit_once(self):
        self.assertEqual(
            list(CURRENCY_GROUPS),
            ["Fiat", "Commodities", "Energy & housing", "Indices", "Stocks"],
        )
        grouped = [currency for group in CURRENCY_GROUPS.values() for currency in group]
        self.assertEqual(len(grouped), len(set(grouped)))
        self.assertIn(CURRENCY_KO, CURRENCY_GROUPS["Stocks"])

    def test_powerlaw_series_options_include_lightning_and_liquid_series(self):
        self.assertIn(POWERLAW_SERIES_LIGHTNING_NODES, POWERLAW_SERIES_OPTIONS)
        self.assertIn(POWERLAW_SERIES_LIGHTNING_CAPACITY, POWERLAW_SERIES_OPTIONS)
        self.assertIn(POWERLAW_SERIES_LIQUID_BTC, POWERLAW_SERIES_OPTIONS)
        self.assertIn(POWERLAW_SERIES_LIQUID_TRANSACTIONS, POWERLAW_SERIES_OPTIONS)
        self.assertIn(POWERLAW_SERIES_BITCOIN_NETWORK_SIMULATION, POWERLAW_SERIES_OPTIONS)
        self.assertIn(POWERLAW_SERIES_BITCOIN_MARKET_CAP, POWERLAW_SERIES_OPTIONS)
        self.assertIn(POWERLAW_SERIES_BITCOIN_VOLATILITY, POWERLAW_SERIES_OPTIONS)
        self.assertIn(POWERLAW_SERIES_USDT_SUPPLY, POWERLAW_SERIES_OPTIONS)


if __name__ == "__main__":
    unittest.main()
