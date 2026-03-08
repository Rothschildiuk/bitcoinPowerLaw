import unittest

from core.constants import (
    POWERLAW_SERIES_FILECOIN_BTC,
    POWERLAW_SERIES_LIGHTNING_CAPACITY,
    POWERLAW_SERIES_LIGHTNING_NODES,
    POWERLAW_SERIES_LIQUID_BTC,
    POWERLAW_SERIES_LIQUID_TRANSACTIONS,
    POWERLAW_SERIES_OPTIONS,
)


class TestConstantsSeries(unittest.TestCase):
    def test_powerlaw_series_options_include_lightning_and_liquid_series(self):
        self.assertIn(POWERLAW_SERIES_LIGHTNING_NODES, POWERLAW_SERIES_OPTIONS)
        self.assertIn(POWERLAW_SERIES_LIGHTNING_CAPACITY, POWERLAW_SERIES_OPTIONS)
        self.assertIn(POWERLAW_SERIES_LIQUID_BTC, POWERLAW_SERIES_OPTIONS)
        self.assertIn(POWERLAW_SERIES_LIQUID_TRANSACTIONS, POWERLAW_SERIES_OPTIONS)
        self.assertIn(POWERLAW_SERIES_FILECOIN_BTC, POWERLAW_SERIES_OPTIONS)


if __name__ == "__main__":
    unittest.main()
