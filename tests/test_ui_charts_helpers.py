import unittest

import numpy as np
import pandas as pd

from ui.charts import (
    _main_chart_plotly_config,
    _resolve_log_time_axis,
    _resolve_model_view_max,
    _resolve_powerlaw_y_range,
    _resolve_time_axis_start_date,
)


class TestUIChartsHelpers(unittest.TestCase):
    def test_main_chart_config_adds_spike_lines_toggle(self):
        config = _main_chart_plotly_config()

        self.assertTrue(config["displayModeBar"])
        modebar_buttons = [
            button for button_group in config["modeBarButtons"] for button in button_group
        ]
        self.assertIn("toggleSpikelines", modebar_buttons)
        self.assertLess(
            modebar_buttons.index("pan2d"),
            modebar_buttons.index("toggleSpikelines"),
        )
        self.assertLess(
            modebar_buttons.index("toggleSpikelines"),
            modebar_buttons.index("zoomIn2d"),
        )

    def test_resolve_powerlaw_y_range_log_includes_padding_and_positive_bounds(self):
        df_display = pd.DataFrame(
            {
                "CloseDisplay": [10.0, 100.0, 1000.0],
                "Days": [1.0, 10.0, 100.0],
            },
            index=pd.to_datetime(["2010-01-01", "2011-01-01", "2012-01-01"]),
        )
        fair_values = np.array([12.0, 110.0, 900.0], dtype=float)

        y_range = _resolve_powerlaw_y_range(
            df_display=df_display,
            m_fair_display=fair_values,
            p2_5=-0.2,
            p97_5=0.2,
            use_log_scale=True,
        )

        self.assertIsNotNone(y_range)
        self.assertEqual(len(y_range), 2)
        self.assertLess(y_range[0], y_range[1])
        self.assertLess(y_range[0], np.log10(df_display["CloseDisplay"].min()))

    def test_resolve_powerlaw_y_range_linear_has_non_negative_floor(self):
        df_display = pd.DataFrame(
            {
                "CloseDisplay": [2.0, 3.0, 4.0],
                "Days": [1.0, 2.0, 3.0],
            },
            index=pd.to_datetime(["2010-01-01", "2010-01-02", "2010-01-03"]),
        )
        fair_values = np.array([2.5, 2.8, 3.2], dtype=float)

        y_range = _resolve_powerlaw_y_range(
            df_display=df_display,
            m_fair_display=fair_values,
            p2_5=-0.1,
            p97_5=0.1,
            use_log_scale=False,
        )

        self.assertIsNotNone(y_range)
        self.assertGreaterEqual(y_range[0], 0.0)
        self.assertLess(y_range[0], y_range[1])

    def test_resolve_log_time_axis_uses_data_span_for_ticks(self):
        index = pd.to_datetime(["2013-06-01", "2014-06-01", "2015-06-01"])
        df_display = pd.DataFrame({"Days": [10.0, 500.0, 900.0]}, index=index)
        current_gen_date = pd.Timestamp("2009-01-03")
        m_dates = [pd.Timestamp("2016-01-01"), pd.Timestamp("2017-01-01")]

        x_range, tick_days, tick_labels = _resolve_log_time_axis(
            df_display=df_display,
            current_gen_date=current_gen_date,
            view_max=3000,
            m_dates=m_dates,
        )

        self.assertEqual(len(x_range), 2)
        self.assertLess(x_range[0], x_range[1])
        self.assertEqual(len(tick_days), len(tick_labels))
        self.assertTrue(all(day > 0 for day in tick_days))
        self.assertIn("2014", tick_labels)
        expected_start_day = max(
            1.0,
            float((_resolve_time_axis_start_date(df_display) - current_gen_date).days),
        )
        self.assertEqual(x_range[0], np.log10(expected_start_day))

    def test_resolve_time_axis_start_date_adds_three_month_padding(self):
        df_display = pd.DataFrame(
            {"Days": [100.0, 200.0]},
            index=pd.to_datetime(["2015-06-15", "2016-06-15"]),
        )

        start_date = _resolve_time_axis_start_date(df_display)

        self.assertEqual(start_date, pd.Timestamp("2015-03-17"))

    def test_resolve_model_view_max_limits_horizon_to_five_years_after_latest_data(self):
        current_gen_date = pd.Timestamp("2009-01-03")
        df_display = pd.DataFrame(
            {"Days": [100.0, 200.0]},
            index=pd.to_datetime(["2025-03-05", "2026-03-05"]),
        )

        view_max = _resolve_model_view_max(df_display, current_gen_date)

        self.assertEqual(
            view_max,
            float(((pd.Timestamp("2031-03-05") - current_gen_date).days)),
        )


if __name__ == "__main__":
    unittest.main()
