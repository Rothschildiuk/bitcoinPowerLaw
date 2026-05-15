import unittest

import numpy as np
import pandas as pd

from ui.kpi import calculate_powerlaw_band_shares


class TestUIKpi(unittest.TestCase):
    def test_calculate_powerlaw_band_shares_counts_eighth_sigma_zones_and_tails(self):
        df_display = pd.DataFrame(
            {
                "Res": [
                    -1.9375,
                    -1.8125,
                    -1.6875,
                    -1.5625,
                    -1.4375,
                    -1.3125,
                    -1.1875,
                    -1.0625,
                    -0.9375,
                    -0.8125,
                    -0.6875,
                    -0.5625,
                    -0.4375,
                    -0.3125,
                    -0.1875,
                    -0.0625,
                    0.0625,
                    0.1875,
                    0.3125,
                    0.4375,
                    0.5625,
                    0.6875,
                    0.8125,
                    0.9375,
                    1.0625,
                    1.1875,
                    1.3125,
                    1.4375,
                    1.5625,
                    1.6875,
                    1.8125,
                    1.9375,
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
                "< -2σ",
                "-2σ to -1.875σ",
                "-1.875σ to -1.75σ",
                "-1.75σ to -1.625σ",
                "-1.625σ to -1.5σ",
                "-1.5σ to -1.375σ",
                "-1.375σ to -1.25σ",
                "-1.25σ to -1.125σ",
                "-1.125σ to -1σ",
                "-1σ to -0.875σ",
                "-0.875σ to -0.75σ",
                "-0.75σ to -0.625σ",
                "-0.625σ to -0.5σ",
                "-0.5σ to -0.375σ",
                "-0.375σ to -0.25σ",
                "-0.25σ to -0.125σ",
                "-0.125σ to Power Law",
                "Power Law to +0.125σ",
                "+0.125σ to +0.25σ",
                "+0.25σ to +0.375σ",
                "+0.375σ to +0.5σ",
                "+0.5σ to +0.625σ",
                "+0.625σ to +0.75σ",
                "+0.75σ to +0.875σ",
                "+0.875σ to +1σ",
                "+1σ to +1.125σ",
                "+1.125σ to +1.25σ",
                "+1.25σ to +1.375σ",
                "+1.375σ to +1.5σ",
                "+1.5σ to +1.625σ",
                "+1.625σ to +1.75σ",
                "+1.75σ to +1.875σ",
                "+1.875σ to +2σ",
                "> +2σ",
            ],
        )
        expected_share = 100.0 / 34.0
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

        self.assertEqual([band["share"] for band in shares], [0.0] * 34)


if __name__ == "__main__":
    unittest.main()
