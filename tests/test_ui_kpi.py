import unittest

import numpy as np
import pandas as pd

from ui.kpi import calculate_powerlaw_band_shares


class TestUIKpi(unittest.TestCase):
    def test_calculate_powerlaw_band_shares_counts_half_sigma_zones(self):
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

        self.assertEqual(
            [band["label"] for band in shares],
            [
                "-2σ to -1.5σ",
                "-1.5σ to -1σ",
                "-1σ to -0.5σ",
                "-0.5σ to Power Law",
                "Power Law to +0.5σ",
                "+0.5σ to +1σ",
                "+1σ to +1.5σ",
                "+1.5σ to +2σ",
            ],
        )
        self.assertEqual([band["share"] for band in shares], [10.0] * 8)

    def test_calculate_powerlaw_band_shares_returns_zero_for_no_valid_residuals(self):
        df_display = pd.DataFrame({"Res": [np.nan, np.inf]})

        shares = calculate_powerlaw_band_shares(
            df_display,
            p2_5=-1.0,
            p16_5=-0.7,
            p83_5=0.7,
            p97_5=1.0,
        )

        self.assertEqual([band["share"] for band in shares], [0.0] * 8)


if __name__ == "__main__":
    unittest.main()
