import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from core.constants import MODE_LOGPERIODIC, TIME_LOG
from ui.charts import (
    _main_chart_plotly_config,
    _resolve_optional_sigma_offsets,
    _resolve_log_time_axis,
    _resolve_model_view_max,
    _resolve_powerlaw_y_range,
    _resolve_time_axis_start_date,
    render_main_model_chart,
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

    def test_resolve_model_view_max_extends_ten_years_after_today(self):
        current_gen_date = pd.Timestamp("2009-01-03")
        df_display = pd.DataFrame(
            {"Days": [100.0, 200.0]},
            index=pd.to_datetime(["2025-03-05", "2026-03-05"]),
        )

        view_max = _resolve_model_view_max(
            df_display,
            current_gen_date,
            today=pd.Timestamp("2026-05-13"),
        )

        self.assertEqual(
            view_max,
            float(((pd.Timestamp("2036-05-13") - current_gen_date).days)),
        )

    def test_resolve_model_view_max_uses_latest_data_when_newer_than_today(self):
        current_gen_date = pd.Timestamp("2009-01-03")
        df_display = pd.DataFrame(
            {"Days": [100.0, 200.0]},
            index=pd.to_datetime(["2026-03-05", "2027-03-05"]),
        )

        view_max = _resolve_model_view_max(
            df_display,
            current_gen_date,
            today=pd.Timestamp("2026-05-13"),
        )

        self.assertEqual(
            view_max,
            float(((pd.Timestamp("2037-03-05") - current_gen_date).days)),
        )

    def test_resolve_optional_sigma_offsets_interpolates_half_sigma_lines(self):
        offsets = _resolve_optional_sigma_offsets(
            p2_5=-1.0,
            p16_5=-0.4,
            p83_5=0.6,
            p97_5=1.2,
        )

        self.assertEqual([level for level, _ in offsets], [-1.5, -0.5, 0.5, 1.5])
        self.assertTrue(np.allclose([offset for _, offset in offsets], [-0.7, -0.2, 0.3, 0.9]))

    def test_logperiodic_powerlaw_slope_trace_is_hidden_by_default(self):
        dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
        days = np.array([4000.0, 4001.0, 4002.0])
        df_display = pd.DataFrame(
            {
                "AbsDays": days,
                "Days": days,
                "CloseDisplay": [100.0, 101.0, 102.0],
                "Res": [0.1, 0.2, 0.3],
            },
            index=dates,
        )
        captured = {}

        def capture_plotly_chart(fig, **kwargs):
            captured["fig"] = fig

        with patch("ui.charts.st.plotly_chart", side_effect=capture_plotly_chart):
            render_main_model_chart(
                mode=MODE_LOGPERIODIC,
                time_scale=TIME_LOG,
                price_scale=TIME_LOG,
                df_display=df_display,
                current_gen_date=pd.Timestamp("2009-01-03"),
                view_max=5000.0,
                plot_x_model=days,
                plot_x_main=days,
                plot_x_osc=days,
                m_log_d=np.log10(days),
                m_dates=dates,
                m_dates_str=np.array(["01.01.2020", "02.01.2020", "03.01.2020"]),
                m_fair_display=np.array([100.0, 101.0, 102.0]),
                historical_powerlaw_slopes=np.array([5.5, 5.6, 5.7]),
                show_historical_powerlaw_slope=True,
                m_osc_y=np.array([0.1, 0.2, 0.3]),
                m_osc_y_by_harmonic={1: np.array([0.1, 0.2, 0.3])},
                perrenod_curve=None,
                residual_sigma_log=1.0,
                p2_5=-0.2,
                p16_5=-0.1,
                p83_5=0.1,
                p97_5=0.2,
                peak_powerlaw_overlay=None,
                osc_t1_age=1.0,
                osc_lambda=2.0,
                selected_harmonic_count=1,
                pl_template="plotly_dark",
                pl_bg_color="#000",
                pl_grid_color="#333",
                pl_btc_color="#fff",
                pl_legend_color="#fff",
                pl_text_color="#fff",
                c_hover_bg="#111",
                c_hover_text="#fff",
                c_border="#333",
                currency_prefix="$",
                currency_suffix="",
                currency_decimals=0,
                target_series_name="Bitcoin",
                target_series_unit="USD",
                show_halving_lines=False,
                chart_key="test-logperiodic-slope",
            )

        slope_traces = [
            trace for trace in captured["fig"].data if str(trace.name).startswith("PowerLaw B")
        ]
        self.assertEqual(len(slope_traces), 1)
        self.assertEqual(slope_traces[0].visible, "legendonly")

    def test_logperiodic_perrenod_curve_is_rendered_when_available(self):
        dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
        days = np.array([4000.0, 4001.0, 4002.0])
        df_display = pd.DataFrame(
            {
                "AbsDays": days,
                "Days": days,
                "CloseDisplay": [100.0, 101.0, 102.0],
                "Res": [0.1, 0.2, 0.3],
            },
            index=dates,
        )
        captured = {}

        def capture_plotly_chart(fig, **kwargs):
            captured["fig"] = fig

        with patch("ui.charts.st.plotly_chart", side_effect=capture_plotly_chart):
            render_main_model_chart(
                mode=MODE_LOGPERIODIC,
                time_scale=TIME_LOG,
                price_scale=TIME_LOG,
                df_display=df_display,
                current_gen_date=pd.Timestamp("2009-01-03"),
                view_max=5000.0,
                plot_x_model=days,
                plot_x_main=days,
                plot_x_osc=days,
                m_log_d=np.log10(days),
                m_dates=dates,
                m_dates_str=np.array(["01.01.2020", "02.01.2020", "03.01.2020"]),
                m_fair_display=np.array([100.0, 101.0, 102.0]),
                historical_powerlaw_slopes=np.array([5.5, 5.6, 5.7]),
                show_historical_powerlaw_slope=False,
                m_osc_y=np.array([0.1, 0.2, 0.3]),
                m_osc_y_by_harmonic={
                    1: np.array([0.1, 0.2, 0.3]),
                    2: np.array([0.2, 0.3, 0.4]),
                    3: np.array([0.3, 0.4, 0.5]),
                },
                perrenod_curve={
                    "label": "DSI ω,2ω,4ω decayed",
                    "r2": 53.56,
                    "values": np.array([0.3, 0.2, 0.1]),
                },
                residual_sigma_log=1.0,
                p2_5=-0.2,
                p16_5=-0.1,
                p83_5=0.1,
                p97_5=0.2,
                peak_powerlaw_overlay=None,
                osc_t1_age=1.0,
                osc_lambda=2.0,
                selected_harmonic_count=3,
                pl_template="plotly_dark",
                pl_bg_color="#000",
                pl_grid_color="#333",
                pl_btc_color="#fff",
                pl_legend_color="#fff",
                pl_text_color="#fff",
                c_hover_bg="#111",
                c_hover_text="#fff",
                c_border="#333",
                currency_prefix="$",
                currency_suffix="",
                currency_decimals=0,
                target_series_name="Bitcoin",
                target_series_unit="USD",
                show_halving_lines=False,
                chart_key="test-logperiodic-perrenod",
            )

        perrenod_traces = [
            trace
            for trace in captured["fig"].data
            if str(trace.name).startswith("DSI ω,2ω,4ω decayed")
        ]
        locked_dsi_traces = [
            trace
            for trace in captured["fig"].data
            if str(trace.name) in {"DSI ω", "DSI ω,2ω", "DSI ω,2ω,4ω"}
        ]
        self.assertEqual(len(perrenod_traces), 1)
        self.assertEqual(perrenod_traces[0].visible, None)
        self.assertEqual(len(locked_dsi_traces), 3)
        self.assertTrue(all(trace.visible == "legendonly" for trace in locked_dsi_traces))


if __name__ == "__main__":
    unittest.main()
