import unittest

import numpy as np
import pandas as pd

from ui.kpi import calculate_powerlaw_band_shares


class TestUIKpi(unittest.TestCase):
    def test_calculate_powerlaw_band_shares_counts_four_sigma_zones(self):
        df_display = pd.DataFrame(
            {
                "Res": [
                    0.95,
                    0.75,
                    0.50,
                    0.10,
                    -0.10,
                    -0.50,
                    -0.75,
                    -0.95,
                    1.20,
                    -1.20,
                    np.nan,
                ]
            }
        )

        shares = calculate_powerlaw_band_shares(
            df_display,
            p2_5=-1.0,
            p16_5=-0.7,
            p83_5=0.7,
            p97_5=1.0,
        )

        self.assertEqual(shares["plus_two_to_plus_one"], 20.0)
        self.assertEqual(shares["plus_one_to_powerlaw"], 20.0)
        self.assertEqual(shares["powerlaw_to_minus_one"], 20.0)
        self.assertEqual(shares["minus_one_to_minus_two"], 20.0)

    def test_calculate_powerlaw_band_shares_returns_zero_for_no_valid_residuals(self):
        df_display = pd.DataFrame({"Res": [np.nan, np.inf]})

        shares = calculate_powerlaw_band_shares(
            df_display,
            p2_5=-1.0,
            p16_5=-0.7,
            p83_5=0.7,
            p97_5=1.0,
        )

        self.assertEqual(
            shares,
            {
                "plus_two_to_plus_one": 0.0,
                "plus_one_to_powerlaw": 0.0,
                "powerlaw_to_minus_one": 0.0,
                "minus_one_to_minus_two": 0.0,
            },
        )


if __name__ == "__main__":
    unittest.main()
