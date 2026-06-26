import unittest

import numpy as np
import pandas as pd

from ui.kpi import (
    calculate_current_powerlaw_sigma_level,
    calculate_negative_two_sigma_monthly_growth,
    calculate_powerlaw_band_shares,
    filter_sigma_band_history,
    filter_sigma_band_history_percent_range,
    format_sigma_band_history_percent_range,
    resolve_sigma_band_history_percent_range,
    _render_sigma_band_chart,
    _resolve_display_conversion_rate,
)


class TestUIKpi(unittest.TestCase):
    def test_calculate_powerlaw_band_shares_counts_quarter_sigma_centered_zones(self):
        df_display = pd.DataFrame(
            {
                "Res": [
                    -2.25,
                    -2.0,
                    -1.75,
                    -1.5,
                    -1.25,
                    -1.0,
                    -0.75,
                    -0.5,
                    -0.25,
                    0.0,
                    0.25,
                    0.5,
                    0.75,
                    1.0,
                    1.25,
                    1.5,
                    1.75,
                    2.0,
                    2.25,
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
                "< -2.125σ",
                "-2.125σ to -1.875σ",
                "-1.875σ to -1.625σ",
                "-1.625σ to -1.375σ",
                "-1.375σ to -1.125σ",
                "-1.125σ to -0.875σ",
                "-0.875σ to -0.625σ",
                "-0.625σ to -0.375σ",
                "-0.375σ to -0.125σ",
                "-0.125σ to +0.125σ",
                "+0.125σ to +0.375σ",
                "+0.375σ to +0.625σ",
                "+0.625σ to +0.875σ",
                "+0.875σ to +1.125σ",
                "+1.125σ to +1.375σ",
                "+1.375σ to +1.625σ",
                "+1.625σ to +1.875σ",
                "+1.875σ to +2.125σ",
                "> +2.125σ",
            ],
        )
        self.assertEqual(
            [band["compact_label"] for band in shares],
            [
                "-2.25",
                "-2",
                "-1.75",
                "-1.5",
                "-1.25",
                "-1",
                "-0.75",
                "-0.5",
                "-0.25",
                "0",
                "+0.25",
                "+0.5",
                "+0.75",
                "+1",
                "+1.25",
                "+1.5",
                "+1.75",
                "+2",
                "+2.25",
            ],
        )
        expected_share = 100.0 / 19.0
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

        self.assertEqual([band["share"] for band in shares], [0.0] * 19)

    def test_filter_sigma_band_history_keeps_only_recent_years(self):
        df_display = pd.DataFrame(
            {"Res": [-1.0, 0.0, 1.0]},
            index=pd.to_datetime(["2020-01-01", "2023-01-01", "2024-01-01"]),
        )

        filtered = filter_sigma_band_history(df_display, history_years=1)

        self.assertEqual(filtered["Res"].tolist(), [0.0, 1.0])

    def test_filter_sigma_band_history_zero_keeps_all_history(self):
        df_display = pd.DataFrame(
            {"Res": [-1.0, 0.0, 1.0]},
            index=pd.to_datetime(["2020-01-01", "2023-01-01", "2024-01-01"]),
        )

        filtered = filter_sigma_band_history(df_display, history_years=0)

        pd.testing.assert_frame_equal(filtered, df_display)

    def test_resolve_sigma_band_history_percent_range_clamps_and_sorts(self):
        self.assertEqual(
            resolve_sigma_band_history_percent_range((120.2, -5.7)),
            (0, 100),
        )
        self.assertEqual(
            resolve_sigma_band_history_percent_range((70, 50)),
            (50, 70),
        )
        self.assertEqual(
            resolve_sigma_band_history_percent_range(3),
            (0, 100),
        )

    def test_filter_sigma_band_history_percent_range_filters_by_time_position(self):
        df_display = pd.DataFrame(
            {"Res": [-1.0, 0.0, 1.0]},
            index=pd.to_datetime(["2020-01-01", "2022-01-01", "2024-01-01"]),
        )

        filtered = filter_sigma_band_history_percent_range(df_display, (50, 100))

        self.assertEqual(filtered["Res"].tolist(), [0.0, 1.0])

    def test_filter_sigma_band_history_percent_range_supports_initial_slice(self):
        df_display = pd.DataFrame(
            {"Res": [-1.0, 0.0, 1.0]},
            index=pd.to_datetime(["2020-01-01", "2022-01-01", "2024-01-01"]),
        )

        filtered = filter_sigma_band_history_percent_range(df_display, (0, 70))

        self.assertEqual(filtered["Res"].tolist(), [-1.0, 0.0])

    def test_filter_sigma_band_history_percent_range_keeps_nearest_for_empty_range(self):
        df_display = pd.DataFrame(
            {"Res": [-1.0, 0.0, 1.0]},
            index=pd.to_datetime(["2020-01-01", "2022-01-01", "2024-01-01"]),
        )

        filtered = filter_sigma_band_history_percent_range(df_display, (25, 25))

        self.assertEqual(filtered["Res"].tolist(), [-1.0])

    def test_format_sigma_band_history_percent_range(self):
        self.assertEqual(
            format_sigma_band_history_percent_range((0, 70)),
            "0-70% history",
        )

    def test_calculate_current_powerlaw_sigma_level_uses_latest_valid_residual(self):
        df_display = pd.DataFrame({"Res": [-0.5, np.nan, 0.25]})

        sigma_level = calculate_current_powerlaw_sigma_level(
            df_display,
            p2_5=-2.0,
            p16_5=-1.0,
            p83_5=1.0,
            p97_5=2.0,
        )

        self.assertAlmostEqual(sigma_level, 0.25)

    def test_calculate_negative_two_sigma_monthly_growth_uses_centered_today_window(self):
        df_display = pd.DataFrame(
            {"AbsDays": [1000.0]},
            index=pd.to_datetime(["2024-01-16"]),
        )

        growth = calculate_negative_two_sigma_monthly_growth(
            df_display,
            a_active=2.0,
            b_active=0.5,
            p2_5=-0.4,
            today="2024-01-16",
        )

        expected_growth = (10**1.6) * ((1015.0**0.5) - (985.0**0.5))
        self.assertTrue(np.isclose(growth, expected_growth))

    def test_resolve_display_conversion_rate_uses_fair_display_ratio(self):
        df_display = pd.DataFrame({"Fair": [100.0], "FairDisplay": [90.0]})

        self.assertAlmostEqual(_resolve_display_conversion_rate(df_display), 0.9)

    def test_render_sigma_band_chart_marks_current_band(self):
        shares = calculate_powerlaw_band_shares(
            pd.DataFrame({"Res": [-0.25, 0.25]}),
            p2_5=-2.0,
            p16_5=-1.0,
            p83_5=1.0,
            p97_5=2.0,
        )
        rendered = {}

        def capture_markdown(html, unsafe_allow_html=False):
            rendered["html"] = html
            rendered["unsafe_allow_html"] = unsafe_allow_html

        import ui.kpi

        original_markdown = ui.kpi.st.markdown
        ui.kpi.st.markdown = capture_markdown
        try:
            _render_sigma_band_chart(shares, current_sigma_level=0.25)
        finally:
            ui.kpi.st.markdown = original_markdown

        self.assertTrue(rendered["unsafe_allow_html"])
        self.assertEqual(rendered["html"].count("class='sigma-bar-item"), 19)
        self.assertEqual(rendered["html"].count("sigma-bar-item-current"), 1)
        self.assertIn(
            "<div class='sigma-bar-item' title='-0.125σ to +0.125σ'>",
            rendered["html"],
        )


if __name__ == "__main__":
    unittest.main()
