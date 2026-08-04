import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from core.constants import (
    MODE_POWERLAW,
    POWERLAW_SIGMA_MODE_HISTORICAL,
    POWERLAW_SIGMA_MODE_SEGMENTED,
    TIME_LOG,
)
from ui.charts import (
    _convert_log_offsets_to_sigma_levels,
    _iter_moving_average_series,
    _main_chart_plotly_config,
    _resolve_optional_sigma_offsets,
    _resolve_log_time_axis,
    _resolve_model_view_max,
    _resolve_powerlaw_y_range,
    _resolve_trace_sample_indices,
    _sample_trace_values,
    _resolve_time_axis_start_date,
    render_main_model_chart,
    render_powerlaw_oscillator_chart,
)


class TestUIChartsHelpers(unittest.TestCase):
    def test_powerlaw_oscillator_uses_straight_unit_baseline_for_classic_and_historical(self):
        dates = pd.date_range("2020-01-01", periods=4, freq="D")
        days = np.array([4000.0, 4001.0, 4002.0, 4003.0])
        df_display = pd.DataFrame(
            {
                "Days": days[:3],
                "CloseDisplay": [110.0, 90.0, 120.0],
                "FairDisplay": [100.0, 100.0, 100.0],
            },
            index=dates[:3],
        )

        for sigma_mode, expected_model_name in (
            ("Classic", "Power regression"),
            (POWERLAW_SIGMA_MODE_HISTORICAL, "Historical PowerLaw"),
        ):
            captured = {}

            def capture_plotly_chart(fig, **kwargs):
                captured["fig"] = fig

            with patch("ui.charts.st.plotly_chart", side_effect=capture_plotly_chart):
                render_powerlaw_oscillator_chart(
                    mode=MODE_POWERLAW,
                    time_scale=TIME_LOG,
                    df_display=df_display,
                    current_gen_date=pd.Timestamp("2009-01-03"),
                    view_max=float(days[-1]),
                    plot_x_model=days,
                    plot_x_main=days[:3],
                    m_dates=dates,
                    m_dates_str=dates.strftime("%d.%m.%Y").to_numpy(),
                    historical_powerlaw_fair=np.log10([100.0, 100.0, 100.0]),
                    historical_powerlaw_sigma_offsets=np.tile(
                        np.array([[-0.2], [-0.1], [0.1], [0.2]]), (1, 3)
                    ),
                    powerlaw_sigma_display_mode=sigma_mode,
                    p2_5=-0.2,
                    p16_5=-0.1,
                    p83_5=0.1,
                    p97_5=0.2,
                    pl_template="plotly_dark",
                    pl_bg_color="#000",
                    pl_grid_color="#333",
                    pl_btc_color="#fff",
                    pl_legend_color="#fff",
                    pl_text_color="#fff",
                    c_hover_bg="#111",
                    c_hover_text="#fff",
                    c_border="#333",
                    target_series_name="Bitcoin",
                    target_series_unit="USD",
                    show_halving_lines=False,
                    chart_key=f"test-oscillator-{sigma_mode}",
                )

            model_trace = next(
                trace for trace in captured["fig"].data if trace.name == expected_model_name
            )
            np.testing.assert_allclose(model_trace.y, 1.0)
            self.assertEqual(model_trace.x[-1], days[-1])
            oscillator_trace = next(
                trace for trace in captured["fig"].data if trace.name == "Bitcoin oscillator"
            )
            np.testing.assert_allclose(oscillator_trace.y, [1.1, 0.9, 1.2])
            self.assertEqual(captured["fig"].layout.yaxis.type, "log")

    def test_historical_powerlaw_continues_from_latest_fit_to_model_horizon(self):
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        display_days = np.array([4000.0, 4001.0, 4002.0])
        model_days = np.array([4000.0, 4001.0, 4002.0, 4003.0, 4004.0])
        historical_intercepts = np.array([1.0, 1.0, 1.0])
        historical_slopes = np.array([0.5, 0.5, 0.5])
        historical_fair = historical_intercepts + historical_slopes * np.log10(display_days)
        captured = {}

        def capture_plotly_chart(fig, **kwargs):
            captured["fig"] = fig

        with patch("ui.charts.st.plotly_chart", side_effect=capture_plotly_chart):
            render_main_model_chart(
                mode=MODE_POWERLAW,
                time_scale=TIME_LOG,
                price_scale=TIME_LOG,
                df_display=pd.DataFrame(
                    {"Days": display_days, "CloseDisplay": [100.0, 101.0, 102.0]},
                    index=dates[:3],
                ),
                current_gen_date=pd.Timestamp("2009-01-03"),
                view_max=float(model_days[-1]),
                plot_x_model=model_days,
                plot_x_main=display_days,
                m_log_d=np.log10(model_days),
                m_dates=dates,
                m_dates_str=dates.strftime("%d.%m.%Y").to_numpy(),
                m_fair_display=np.linspace(100.0, 104.0, len(model_days)),
                historical_powerlaw_intercepts=historical_intercepts,
                historical_powerlaw_slopes=historical_slopes,
                historical_powerlaw_fair=historical_fair,
                historical_powerlaw_sigma_offsets=np.tile(
                    np.array([[-0.2], [-0.1], [0.1], [0.2]]), (1, 3)
                ),
                powerlaw_sigma_display_mode=POWERLAW_SIGMA_MODE_HISTORICAL,
                residual_sigma_log=1.0,
                p2_5=-0.2,
                p16_5=-0.1,
                p83_5=0.1,
                p97_5=0.2,
                peak_powerlaw_overlay=None,
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
                chart_key="test-historical-powerlaw-extension",
            )

        fair_traces = [
            trace
            for trace in captured["fig"].data
            if trace.name == "Historical PowerLaw" and not trace.showlegend
        ]
        self.assertEqual(len(fair_traces), 2)
        self.assertEqual(fair_traces[1].x[0], model_days[2])
        self.assertEqual(fair_traces[1].x[-1], model_days[-1])
        for sigma_name in ("Historical +2σ", "Historical +1σ", "Historical -1σ", "Historical -2σ"):
            sigma_traces = [trace for trace in captured["fig"].data if trace.name == sigma_name]
            self.assertEqual(len(sigma_traces), 2)
            self.assertEqual(sigma_traces[1].x[-1], model_days[-1])

    def test_convert_log_offsets_to_sigma_levels_uses_powerlaw_percentile_scale(self):
        sigma_levels = _convert_log_offsets_to_sigma_levels(
            np.array([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]),
            (-0.2, -0.1, 0.1, 0.2),
        )

        np.testing.assert_allclose(sigma_levels, [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])

    def test_iter_moving_average_series_uses_requested_windows(self):
        df_display = pd.DataFrame(
            {"CloseDisplay": [10.0, 20.0, 30.0, 40.0]},
            index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]),
        )

        lines = _iter_moving_average_series(df_display, (2, 3))

        self.assertEqual([window for window, _ in lines], [2, 3])
        np.testing.assert_allclose(lines[0][1].to_numpy(), [np.nan, 15.0, 25.0, 35.0])
        np.testing.assert_allclose(lines[1][1].to_numpy(), [np.nan, np.nan, 20.0, 30.0])

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

    def test_resolve_trace_sample_indices_keeps_short_traces_full_resolution(self):
        sample_indices = _resolve_trace_sample_indices(5, max_points=5)

        self.assertIsInstance(sample_indices, slice)

    def test_resolve_trace_sample_indices_downsamples_and_keeps_endpoints(self):
        sample_indices = _resolve_trace_sample_indices(10, max_points=4)

        self.assertEqual(sample_indices.tolist(), [0, 3, 6, 9])
        np.testing.assert_array_equal(
            _sample_trace_values(np.arange(10), sample_indices),
            np.array([0, 3, 6, 9]),
        )
        self.assertEqual(
            _sample_trace_values(list("abcdefghij"), sample_indices),
            ["a", "d", "g", "j"],
        )

    def test_powerlaw_chart_defaults_hide_fit_points_and_show_major_sigma_lines(self):
        dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
        days = np.array([4000.0, 4001.0, 4002.0])
        df_display = pd.DataFrame(
            {
                "Days": days,
                "CloseDisplay": [100.0, 101.0, 102.0],
            },
            index=dates,
        )
        overlay = {
            "peak": {
                "model_values": np.array([140.0, 141.0, 142.0]),
                "peak_days": np.array([4000.0, 4002.0]),
                "peak_values": np.array([120.0, 122.0]),
            },
            "trough": {
                "model_values": np.array([80.0, 81.0, 82.0]),
                "trough_days": np.array([4000.0, 4002.0]),
                "trough_values": np.array([90.0, 92.0]),
            },
        }
        captured = {}

        def capture_plotly_chart(fig, **kwargs):
            captured["fig"] = fig

        with patch("ui.charts.st.plotly_chart", side_effect=capture_plotly_chart):
            render_main_model_chart(
                mode=MODE_POWERLAW,
                time_scale=TIME_LOG,
                price_scale=TIME_LOG,
                df_display=df_display,
                current_gen_date=pd.Timestamp("2009-01-03"),
                view_max=5000.0,
                plot_x_model=days,
                plot_x_main=days,
                m_log_d=np.log10(days),
                m_dates=dates,
                m_dates_str=np.array(["01.01.2020", "02.01.2020", "03.01.2020"]),
                m_fair_display=np.array([100.0, 101.0, 102.0]),
                historical_powerlaw_slopes=np.array([5.5, 5.6, 5.7]),
                show_historical_powerlaw_slope=False,
                residual_sigma_log=1.0,
                p2_5=-0.2,
                p16_5=-0.1,
                p83_5=0.1,
                p97_5=0.2,
                peak_powerlaw_overlay=overlay,
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
                chart_key="test-powerlaw-default-visibility",
            )

        traces_by_name = {str(trace.name): trace for trace in captured["fig"].data}
        for name in {
            "+2σ (97.725th percentile)",
            "+1σ (84.135th percentile)",
            "-1σ (15.865th percentile)",
            "-2σ (2.275th percentile)",
        }:
            self.assertIn(traces_by_name[name].visible, (None, True))

        self.assertNotIn("Peak PowerLaw", traces_by_name)
        self.assertNotIn("Trough PowerLaw", traces_by_name)
        self.assertNotIn("Peak fit points", traces_by_name)
        self.assertNotIn("Trough fit points", traces_by_name)
        self.assertNotIn("Peak/Trough PowerLaw", traces_by_name)
        self.assertLess(captured["fig"].layout.legend.y, 0.0)
        self.assertEqual(captured["fig"].layout.legend.yanchor, "top")
        self.assertEqual(captured["fig"].layout.legend.font.size, 13)
        self.assertGreaterEqual(captured["fig"].layout.margin.b, 70)
        self.assertEqual(captured["fig"].layout.legend.groupclick, "togglegroup")
        self.assertEqual(
            traces_by_name["-2σ (2.275th percentile)"].legendgroup,
            traces_by_name["+2σ (97.725th percentile)"].legendgroup,
        )
        self.assertEqual(
            traces_by_name["-1σ (15.865th percentile)"].legendgroup,
            traces_by_name["+1σ (84.135th percentile)"].legendgroup,
        )
        self.assertEqual(
            traces_by_name["-1.5σ"].legendgroup,
            traces_by_name["+1.5σ"].legendgroup,
        )
        self.assertEqual(
            traces_by_name["-0.5σ"].legendgroup,
            traces_by_name["+0.5σ"].legendgroup,
        )
        legend_sigma_names = [
            str(trace.name)
            for trace in captured["fig"].data
            if trace.showlegend and str(trace.legendgroup).startswith("sigma_abs_")
        ]
        self.assertEqual(
            legend_sigma_names,
            [
                "±2σ (2.275th/97.725th percentile)",
                "±1.5σ",
                "±1σ (15.865th/84.135th percentile)",
                "±0.5σ",
            ],
        )

    def test_powerlaw_chart_segmented_sigma_replaces_classic_sigma_lines(self):
        dates = pd.date_range("2020-01-01", periods=20, freq="D")
        days = np.arange(4000.0, 4020.0)
        log_days = np.log10(days)
        model_log = 1.0 + 0.5 * log_days
        residuals = np.array(
            [
                0.03,
                0.04,
                0.06,
                0.07,
                0.12,
                0.13,
                0.16,
                0.17,
                0.23,
                0.24,
                -0.03,
                -0.04,
                -0.06,
                -0.07,
                -0.12,
                -0.13,
                -0.16,
                -0.17,
                -0.23,
                -0.24,
            ]
        )
        log_close = model_log + residuals
        df_display = pd.DataFrame(
            {
                "Days": days,
                "LogClose": log_close,
                "ModelLog": model_log,
                "Res": residuals,
                "CloseDisplay": np.power(10.0, log_close),
            },
            index=dates,
        )
        captured = {}

        def capture_plotly_chart(fig, **kwargs):
            captured["fig"] = fig

        with patch("ui.charts.st.plotly_chart", side_effect=capture_plotly_chart):
            render_main_model_chart(
                mode=MODE_POWERLAW,
                time_scale=TIME_LOG,
                price_scale=TIME_LOG,
                df_display=df_display,
                current_gen_date=pd.Timestamp("2009-01-03"),
                view_max=5000.0,
                plot_x_model=days,
                plot_x_main=days,
                m_log_d=log_days,
                m_dates=dates,
                m_dates_str=dates.strftime("%d.%m.%Y").to_numpy(),
                m_fair_display=np.power(10.0, model_log),
                historical_powerlaw_slopes=np.array([], dtype=float),
                show_historical_powerlaw_slope=False,
                residual_sigma_log=1.0,
                p2_5=-0.2,
                p16_5=-0.1,
                p83_5=0.1,
                p97_5=0.2,
                peak_powerlaw_overlay=None,
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
                chart_key="test-powerlaw-segmented-sigma",
                powerlaw_sigma_display_mode=POWERLAW_SIGMA_MODE_SEGMENTED,
            )

        trace_names = {str(trace.name) for trace in captured["fig"].data}
        self.assertNotIn("+1σ (84.135th percentile)", trace_names)
        self.assertNotIn("-1σ (15.865th percentile)", trace_names)
        self.assertIn("Power regression", trace_names)
        self.assertIn("Segmented sigma 0σ to ±0.5σ", trace_names)
        self.assertIn("Segmented sigma -0.5σ to 0σ", trace_names)
        self.assertIn("Segmented sigma ±0.5σ to ±1σ", trace_names)
        self.assertIn("Segmented sigma -1σ to -0.5σ", trace_names)
        segmented_legend_names = [
            str(trace.name)
            for trace in captured["fig"].data
            if trace.showlegend and str(trace.legendgroup).startswith("segmented_sigma_")
        ]
        self.assertIn("Segmented sigma 0σ to ±0.5σ", segmented_legend_names)
        self.assertIn("Segmented sigma ±0.5σ to ±1σ", segmented_legend_names)
        self.assertGreaterEqual(len(segmented_legend_names), 2)
        segmented_legend_groups = [
            str(trace.legendgroup)
            for trace in captured["fig"].data
            if str(trace.legendgroup).startswith("segmented_sigma_")
        ]
        self.assertLess(len(set(segmented_legend_groups)), len(segmented_legend_groups))
        self.assertEqual(segmented_legend_names.count("Segmented sigma 0σ to ±0.5σ"), 1)
        self.assertEqual(segmented_legend_names.count("Segmented sigma ±0.5σ to ±1σ"), 1)
        segmented_traces = [
            trace
            for trace in captured["fig"].data
            if str(trace.legendgroup).startswith("segmented_sigma_")
        ]
        hidden_by_default = {
            str(trace.name)
            for trace in segmented_traces
            if trace.showlegend and trace.visible == "legendonly"
        }
        visible_by_default = {
            str(trace.name)
            for trace in segmented_traces
            if trace.showlegend and trace.visible is True
        }
        self.assertIn("Segmented sigma 0σ to ±0.5σ", hidden_by_default)
        self.assertIn("Segmented sigma ±1σ to ±1.5σ", hidden_by_default)
        self.assertIn("Segmented sigma > ±2σ", hidden_by_default)
        self.assertNotIn("Segmented sigma ±0.5σ to ±1σ", hidden_by_default)
        self.assertIn("Segmented sigma ±0.5σ to ±1σ", visible_by_default)
        self.assertIn("Segmented sigma ±1.5σ to ±2σ", visible_by_default)
        segmented_colors = {trace.line.color for trace in segmented_traces}
        self.assertGreaterEqual(len(segmented_colors), 3)
        self.assertIn("#06b6d4", segmented_colors)
        self.assertIn("#22c55e", segmented_colors)

    def test_powerlaw_chart_downsamples_rendered_model_lines_but_keeps_date_hover_trace(self):
        dates = pd.date_range("2020-01-01", periods=3, freq="D")
        display_days = np.array([4000.0, 4001.0, 4002.0])
        model_days = np.arange(1.0, 3002.0)
        model_dates = pd.date_range("2010-01-04", periods=len(model_days), freq="D")
        captured = {}

        def capture_plotly_chart(fig, **kwargs):
            captured["fig"] = fig

        with patch("ui.charts.st.plotly_chart", side_effect=capture_plotly_chart):
            render_main_model_chart(
                mode=MODE_POWERLAW,
                time_scale=TIME_LOG,
                price_scale=TIME_LOG,
                df_display=pd.DataFrame(
                    {
                        "Days": display_days,
                        "CloseDisplay": [100.0, 101.0, 102.0],
                    },
                    index=dates,
                ),
                current_gen_date=pd.Timestamp("2009-01-03"),
                view_max=float(model_days[-1]),
                plot_x_model=model_days,
                plot_x_main=display_days,
                m_log_d=np.log10(model_days),
                m_dates=model_dates,
                m_dates_str=model_dates.strftime("%d.%m.%Y").to_numpy(),
                m_fair_display=np.linspace(100.0, 200.0, len(model_days)),
                historical_powerlaw_slopes=np.array([5.5, 5.6, 5.7]),
                show_historical_powerlaw_slope=False,
                residual_sigma_log=1.0,
                p2_5=-0.2,
                p16_5=-0.1,
                p83_5=0.1,
                p97_5=0.2,
                peak_powerlaw_overlay=None,
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
                chart_key="test-powerlaw-model-line-sampling",
            )

        power_regression_trace = next(
            trace
            for trace in captured["fig"].data
            if trace.name == "Power regression" and not trace.showlegend
        )
        self.assertLess(len(power_regression_trace.x), len(model_days))
        self.assertEqual(power_regression_trace.x[0], model_days[0])
        self.assertEqual(power_regression_trace.x[-1], model_days[-1])

        date_hover_trace = next(trace for trace in captured["fig"].data if str(trace.name) == "")
        self.assertEqual(len(date_hover_trace.x), len(model_days))

    def test_powerlaw_halving_lines_are_toggleable_from_legend(self):
        dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
        days = np.array([4000.0, 4001.0, 4002.0])
        df_display = pd.DataFrame(
            {
                "Days": days,
                "CloseDisplay": [100.0, 101.0, 102.0],
            },
            index=dates,
        )
        captured = {}

        def capture_plotly_chart(fig, **kwargs):
            captured["fig"] = fig

        with patch("ui.charts.st.plotly_chart", side_effect=capture_plotly_chart):
            render_main_model_chart(
                mode=MODE_POWERLAW,
                time_scale=TIME_LOG,
                price_scale=TIME_LOG,
                df_display=df_display,
                current_gen_date=pd.Timestamp("2009-01-03"),
                view_max=5000.0,
                plot_x_model=days,
                plot_x_main=days,
                m_log_d=np.log10(days),
                m_dates=dates,
                m_dates_str=np.array(["01.01.2020", "02.01.2020", "03.01.2020"]),
                m_fair_display=np.array([100.0, 101.0, 102.0]),
                historical_powerlaw_slopes=np.array([5.5, 5.6, 5.7]),
                show_historical_powerlaw_slope=False,
                residual_sigma_log=1.0,
                p2_5=-0.2,
                p16_5=-0.1,
                p83_5=0.1,
                p97_5=0.2,
                peak_powerlaw_overlay=None,
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
                show_halving_lines=True,
                chart_key="test-powerlaw-halvings",
            )

        halving_traces = [trace for trace in captured["fig"].data if trace.name == "Halvings"]
        self.assertEqual(len(halving_traces), 1)
        self.assertEqual(halving_traces[0].legendgroup, "halvings")
        self.assertEqual(halving_traces[0].mode, "lines")
        self.assertEqual(len(halving_traces[0].x), 12)


if __name__ == "__main__":
    unittest.main()
