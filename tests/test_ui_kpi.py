import unittest

import numpy as np
import pandas as pd

from ui.kpi import calculate_powerlaw_band_shares


class TestUIKpi(unittest.TestCase):
    def test_calculate_powerlaw_band_shares_counts_quarter_sigma_zones(self):
        df_display = pd.DataFrame(
            {
                "Res": [
                    -1.875,
                    -1.625,
                    -1.375,
                    -1.125,
                    -0.875,
                    -0.625,
                    -0.375,
                    -0.125,
                    0.125,
                    0.375,
                    0.625,
                    0.875,
                    1.125,
                    1.375,
                    1.625,
                    1.875,
                    -2.20,
                    2.20,
                    np.nan,
                ]
            }
        )

        shares = calculate_powerlaw_band_shares(
            df_display,
            p2_5=-2.0,
            p16_5=-1.0,
            p83_5=1.0,
            p97_5=2.0,
        )

        self.assertEqual(
            [band["label"] for band in shares],
            [
                "-2σ to -1.75σ",
                "-1.75σ to -1.5σ",
                "-1.5σ to -1.25σ",
                "-1.25σ to -1σ",
                "-1σ to -0.75σ",
                "-0.75σ to -0.5σ",
                "-0.5σ to -0.25σ",
                "-0.25σ to Power Law",
                "Power Law to +0.25σ",
                "+0.25σ to +0.5σ",
                "+0.5σ to +0.75σ",
                "+0.75σ to +1σ",
                "+1σ to +1.25σ",
                "+1.25σ to +1.5σ",
                "+1.5σ to +1.75σ",
                "+1.75σ to +2σ",
            ],
        )
        expected_share = 100.0 / 18.0
        for band in shares:
            self.assertAlmostEqual(band["share"], expected_share)

    def test_calculate_powerlaw_band_shares_returns_zero_for_no_valid_residuals(self):
        df_display = pd.DataFrame({"Res": [np.nan, np.inf]})

        shares = calculate_powerlaw_band_shares(
            df_display,
            p2_5=-1.0,
            p16_5=-0.7,
            p83_5=0.7,
            p97_5=1.0,
        )

        self.assertEqual([band["share"] for band in shares], [0.0] * 16)


if __name__ == "__main__":
    unittest.main()
