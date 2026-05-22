import unittest
from unittest.mock import patch

import numpy as np

from core.constants import KEY_LOGPERIODIC_HARMONICS, KEY_LOGPERIODIC_SHOW_DECAYED_DSI
from core import oscillator


class TestOscillator(unittest.TestCase):
    def test_format_cycle_anchor_date_uses_years_from_genesis(self):
        self.assertEqual(oscillator.format_cycle_anchor_date(1.0), "2010-01-03")
        self.assertEqual(
            oscillator.format_cycle_anchor_date(1.0, origin_offset_days=20),
            "2010-01-23",
        )

    def test_fit_oscillator_component_returns_none_when_lambda_is_invalid(self):
        log_days = np.linspace(1.0, 2.0, 128)
        residuals = np.sin(log_days)

        result = oscillator.fit_oscillator_component(
            log_days,
            residuals,
            t1_cycle_age_years=1.0,
            cycle_lambda=1.0,
        )
        self.assertIsNone(result)

    def test_compute_oscillator_overlay_returns_full_length_series_from_masked_fit(
        self,
    ):
        log_days = np.linspace(0.5, 3.0, 250)
        settings = oscillator.OscillatorSettings(
            t1_age=1.0,
            lambda_val=2.01,
        )
        fit_result = oscillator.fit_oscillator_component(
            log_days,
            np.cos(log_days),
            settings.t1_age,
            settings.lambda_val,
        )
        self.assertIsNotNone(fit_result)
        _, _, _, predicted_residuals = fit_result
        mask = np.zeros(log_days.shape, dtype=bool)
        mask[-150:] = True
        full_residuals = np.zeros_like(log_days)
        full_residuals[mask] = predicted_residuals[mask]
        model_log_values = np.linspace(10.0, 20.0, log_days.size)
        actual_log_values = model_log_values + full_residuals

        result = oscillator.compute_oscillator_overlay(
            log_days,
            full_residuals,
            model_log_values,
            actual_log_values,
            mask,
            settings,
            current_r2=12.5,
        )

        self.assertEqual(result.model_values.shape, log_days.shape)
        self.assertTrue(np.allclose(result.model_values[~mask], 0.0))
        self.assertGreater(np.max(np.abs(result.model_values[mask])), 0.0)
        self.assertTrue(np.isclose(result.reference_log_day, log_days[mask][0]))
        self.assertGreater(result.combined_r2, 99.0)
        self.assertEqual(result.harmonic_coefficients.shape, (1,))

    def test_fit_oscillator_component_uses_requested_harmonics(self):
        log_days = np.linspace(0.5, 3.0, 300)
        t1_age = 1.0
        lambda_val = 2.4
        log_lambda = np.log10(lambda_val)
        angular_frequency = 2 * np.pi / log_lambda
        phase_shift = -angular_frequency * np.log10(t1_age * 365.25)
        phase_values = angular_frequency * log_days + phase_shift
        residuals = 0.25 * np.cos(phase_values) + 0.65 * np.cos(phase_values * 2)

        one_harmonic_r2 = oscillator.compute_oscillator_fit_r2(
            log_days,
            residuals,
            t1_age,
            lambda_val,
            harmonic_count=1,
        )
        two_harmonic_r2 = oscillator.compute_oscillator_fit_r2(
            log_days,
            residuals,
            t1_age,
            lambda_val,
            harmonic_count=2,
        )
        fit_result = oscillator.fit_oscillator_component(
            log_days,
            residuals,
            t1_age,
            lambda_val,
            harmonic_count=2,
        )

        self.assertGreater(two_harmonic_r2, one_harmonic_r2)
        self.assertGreater(two_harmonic_r2, 99.0)
        self.assertEqual(fit_result[0].shape, (2,))

    def test_three_mode_dsi_uses_fourth_harmonic_not_third(self):
        log_days = np.linspace(0.5, 3.0, 300)
        t1_age = 1.0
        lambda_val = 2.4
        log_lambda = np.log10(lambda_val)
        angular_frequency = 2 * np.pi / log_lambda
        phase_shift = -angular_frequency * np.log10(t1_age * 365.25)
        phase_values = angular_frequency * log_days + phase_shift
        residuals = (
            0.25 * np.cos(phase_values)
            + 0.45 * np.cos(phase_values * 2)
            + 0.65 * np.cos(phase_values * 4)
        )

        three_mode_r2 = oscillator.compute_oscillator_fit_r2(
            log_days,
            residuals,
            t1_age,
            lambda_val,
            harmonic_count=3,
        )
        self.assertEqual(oscillator.resolve_harmonic_multipliers(3), (1, 2, 4))
        self.assertGreater(three_mode_r2, 99.0)

    def test_dsi_mode_option_combines_harmonics_and_decay(self):
        self.assertEqual(
            oscillator.resolve_dsi_mode_option(3, True),
            oscillator.DSI_MODE_DECAYED_3,
        )
        self.assertEqual(
            oscillator.format_dsi_mode_option(oscillator.DSI_MODE_DECAYED_3),
            "ω,2ω,4ω Decayed",
        )
        session_state = {}

        with patch.object(oscillator.st, "session_state", session_state):
            oscillator.apply_dsi_mode_option(oscillator.DSI_MODE_DECAYED_3)

        self.assertEqual(session_state[KEY_LOGPERIODIC_HARMONICS], 3)
        self.assertTrue(session_state[KEY_LOGPERIODIC_SHOW_DECAYED_DSI])
        self.assertEqual(
            session_state[oscillator.KEY_LOGPERIODIC_DSI_MODE],
            oscillator.DSI_MODE_DECAYED_3,
        )

    def test_build_oscillator_curve_sums_harmonic_coefficients(self):
        log_days = np.linspace(0.5, 3.0, 300)
        curve = oscillator.build_oscillator_curve(
            log_days=log_days,
            amplitude=0.0,
            angular_frequency=2.0,
            phase_shift=0.3,
            reference_log_day=float(log_days.min()),
            harmonic_coefficients=np.array([0.25, 0.65]),
        )
        expected = 0.25 * np.cos(2.0 * log_days + 0.3) + 0.65 * np.cos(2.0 * (2.0 * log_days + 0.3))

        self.assertTrue(np.allclose(curve, expected))

    def test_build_oscillator_curve_uses_dsi_mode_multipliers(self):
        log_days = np.linspace(0.5, 3.0, 300)
        curve = oscillator.build_oscillator_curve(
            log_days=log_days,
            amplitude=0.0,
            angular_frequency=2.0,
            phase_shift=0.3,
            reference_log_day=float(log_days.min()),
            harmonic_coefficients=np.array([0.25, 0.45, 0.65]),
        )
        phase_values = 2.0 * log_days + 0.3
        expected = (
            0.25 * np.cos(phase_values)
            + 0.45 * np.cos(phase_values * 2)
            + 0.65 * np.cos(phase_values * 4)
        )

        self.assertTrue(np.allclose(curve, expected))

    def test_compute_oscillator_model_stats_reports_information_criteria(self):
        log_days = np.linspace(0.5, 3.0, 300)
        t1_age = 1.0
        lambda_val = 2.4
        log_lambda = np.log10(lambda_val)
        angular_frequency = 2 * np.pi / log_lambda
        phase_shift = -angular_frequency * np.log10(t1_age * 365.25)
        phase_values = angular_frequency * log_days + phase_shift
        residuals = 0.25 * np.cos(phase_values) + 0.45 * np.cos(phase_values * 2)

        stats = oscillator.compute_oscillator_model_stats(
            log_days,
            residuals,
            t1_age,
            lambda_val,
            harmonic_count=2,
        )

        self.assertEqual(stats.mode_multipliers, (1, 2))
        self.assertEqual(stats.parameter_count, 4)
        self.assertGreater(stats.r2, 99.0)
        self.assertTrue(np.isfinite(stats.aic))
        self.assertTrue(np.isfinite(stats.bic))
        self.assertLess(stats.rmse, 1e-10)

    def test_dsi_regression_stats_fit_cos_sin_modes_without_fixed_phase(self):
        log_days = np.linspace(0.5, 3.0, 300)
        lambda_val = 2.4
        omega = 2 * np.pi / np.log10(lambda_val)
        phase_values = omega * log_days
        residuals = (
            0.10
            + 0.25 * np.cos(phase_values)
            - 0.15 * np.sin(phase_values)
            + 0.45 * np.cos(phase_values * 2)
            + 0.30 * np.sin(phase_values * 2)
        )

        stats = oscillator.compute_dsi_regression_stats(
            log_days,
            residuals,
            lambda_val,
            harmonic_count=2,
        )

        self.assertEqual(stats.label, "DSI ω,2ω")
        self.assertEqual(stats.parameter_count, 6)
        self.assertGreater(stats.r2, 99.0)
        self.assertLess(stats.rmse, 1e-10)

    def test_build_dsi_regression_curve_predicts_decayed_modes(self):
        fit_days = np.linspace(365.25, 365.25 * 10.0, 300)
        predict_days = np.linspace(365.25, 365.25 * 11.0, 330)
        fit_log_days = np.log10(fit_days)
        predict_log_days = np.log10(predict_days)
        lambda_val = 2.4
        omega = 2 * np.pi / np.log10(lambda_val)
        fit_decay = 1.0 / (fit_days / 365.25 + 2.0)
        predict_decay = 1.0 / (predict_days / 365.25 + 2.0)
        fit_residuals = 0.2 + fit_decay * (
            0.7 * np.cos(omega * fit_log_days)
            - 0.3 * np.sin(2.0 * omega * fit_log_days)
            + 0.4 * np.cos(4.0 * omega * fit_log_days)
        )
        expected_prediction = 0.2 + predict_decay * (
            0.7 * np.cos(omega * predict_log_days)
            - 0.3 * np.sin(2.0 * omega * predict_log_days)
            + 0.4 * np.cos(4.0 * omega * predict_log_days)
        )

        prediction = oscillator.build_dsi_regression_curve(
            fit_log_days,
            fit_residuals,
            predict_log_days,
            lambda_val,
            harmonic_count=3,
            fit_days_since_genesis=fit_days,
            predict_days_since_genesis=predict_days,
            decay_model="reciprocal_age",
        )

        self.assertTrue(np.allclose(prediction, expected_prediction))

    def test_sidebar_logperiodic_r2_toggles_current_dsi_decay_model(self):
        days = np.linspace(365.25, 365.25 * 10.0, 300)
        log_days = np.log10(days)
        lambda_val = 2.4
        omega = 2 * np.pi / np.log10(lambda_val)
        decay = 1.0 / (days / 365.25 + 2.0)
        residuals = 0.2 + decay * (
            0.7 * np.cos(omega * log_days)
            - 0.3 * np.sin(2.0 * omega * log_days)
            + 0.4 * np.cos(4.0 * omega * log_days)
        )

        sidebar_r2 = oscillator.compute_sidebar_logperiodic_r2(
            log_days,
            residuals,
            days,
            {
                "t1_age": 1.0,
                "lambda_val": 2.0,
            },
            harmonic_count=3,
            lambda_bounds=(2.0, 2.8),
            show_decayed_regression=False,
        )
        decayed_sidebar_r2 = oscillator.compute_sidebar_logperiodic_r2(
            log_days,
            residuals,
            days,
            {
                "t1_age": 1.0,
                "lambda_val": 2.0,
            },
            harmonic_count=3,
            lambda_bounds=(2.0, 2.8),
            show_decayed_regression=True,
        )
        current_dsi_stats = oscillator.compute_dsi_regression_stats(
            log_days,
            residuals,
            2.0,
            harmonic_count=3,
        )
        current_decayed_stats = oscillator.compute_dsi_regression_stats(
            log_days,
            residuals,
            2.0,
            harmonic_count=3,
            days_since_genesis=days,
            decay_model="reciprocal_age",
        )

        self.assertAlmostEqual(sidebar_r2, current_dsi_stats.r2)
        self.assertAlmostEqual(decayed_sidebar_r2, current_decayed_stats.r2)
        self.assertNotAlmostEqual(sidebar_r2, decayed_sidebar_r2)

    def test_sidebar_logperiodic_r2_changes_with_selected_dsi_modes(self):
        days = np.linspace(365.25, 365.25 * 12.0, 360)
        log_days = np.log10(days)
        lambda_val = 2.6
        omega = 2 * np.pi / np.log10(lambda_val)
        residuals = (
            0.3
            + 0.4 * np.cos(omega * log_days)
            - 0.2 * np.sin(2.0 * omega * log_days)
            + 0.8 * np.cos(4.0 * omega * log_days)
        )
        params = {
            "t1_age": 1.0,
            "lambda_val": lambda_val,
        }

        two_mode_r2 = oscillator.compute_sidebar_logperiodic_r2(
            log_days,
            residuals,
            days,
            params,
            harmonic_count=2,
            lambda_bounds=(2.0, 2.8),
            show_decayed_regression=True,
        )
        three_mode_r2 = oscillator.compute_sidebar_logperiodic_r2(
            log_days,
            residuals,
            days,
            params,
            harmonic_count=3,
            lambda_bounds=(2.0, 2.8),
            show_decayed_regression=True,
        )

        self.assertGreater(three_mode_r2, two_mode_r2)

    def test_optimized_dsi_regression_lambda_prefers_signal_lambda(self):
        log_days = np.linspace(0.5, 3.0, 300)
        lambda_val = 2.4
        omega = 2 * np.pi / np.log10(lambda_val)
        residuals = np.cos(omega * log_days) + 0.5 * np.sin(omega * log_days * 2)

        stats = oscillator.optimize_dsi_regression_lambda(
            log_days,
            residuals,
            harmonic_count=2,
            min_lambda=2.0,
            max_lambda=2.8,
        )

        self.assertAlmostEqual(float(stats.parameter_label.split()[1]), lambda_val, places=2)
        self.assertGreater(stats.r2, 99.0)

    def test_linear_cycle_regression_stats_fit_fixed_year_cycle(self):
        days_since_genesis = np.linspace(365.25, 365.25 * 10.0, 600)
        age_years = days_since_genesis / 365.25
        residuals = 0.2 + 0.7 * np.cos(2.0 * np.pi * age_years / 4.0)

        stats = oscillator.compute_linear_cycle_regression_stats(
            days_since_genesis,
            residuals,
            cycle_years=4.0,
        )

        self.assertEqual(stats.label, "Linear 4y")
        self.assertEqual(stats.parameter_count, 3)
        self.assertGreater(stats.r2, 99.0)

    def test_regression_stats_table_uses_decayed_r2_column_without_linear_rows(self):
        current_rows = [
            oscillator.OscillatorModelStats(
                harmonic_count=1,
                mode_multipliers=(1,),
                parameter_count=3,
                r2=34.11,
                aic=0.0,
                bic=0.0,
                rmse=0.0,
            )
        ]
        rows = [
            oscillator.RegressionComparisonStats(
                label="Linear 4y",
                parameter_label="4y",
                parameter_count=3,
                r2=21.83,
                aic=0.0,
                bic=0.0,
                rmse=0.0,
            ),
            oscillator.RegressionComparisonStats(
                label="DSI ω,2ω",
                parameter_label="λ 4.26",
                parameter_count=6,
                r2=38.20,
                aic=0.0,
                bic=0.0,
                rmse=0.0,
            ),
            oscillator.RegressionComparisonStats(
                label="DSI ω,2ω decayed",
                parameter_label="λ 4.17",
                parameter_count=6,
                r2=49.30,
                aic=0.0,
                bic=0.0,
                rmse=0.0,
            ),
        ]
        captured = {}

        with patch.object(
            oscillator.st,
            "markdown",
            side_effect=lambda html, **_: captured.setdefault("html", html),
        ):
            oscillator.render_logperiodic_regression_stats_table(current_rows, rows)

        self.assertIn("<th>Decayed R²%</th>", captured["html"])
        self.assertIn("<td>38.20</td><td>49.30</td>", captured["html"])
        self.assertNotIn("Current λ DSI ω", captured["html"])
        self.assertNotIn("34.11", captured["html"])
        self.assertNotIn("Linear 4y", captured["html"])
        self.assertNotIn("DSI ω,2ω decayed</td>", captured["html"])

    def test_compute_oscillator_overlay_keeps_baseline_r2_when_fit_is_not_possible(
        self,
    ):
        log_days = np.array([1.0, 2.0, 3.0])
        residuals = np.array([0.1, 0.2, 0.3])
        model_log_values = np.array([10.0, 10.1, 10.2])
        actual_log_values = model_log_values + residuals
        settings = oscillator.OscillatorSettings(
            t1_age=1.0,
            lambda_val=1.0,
        )

        result = oscillator.compute_oscillator_overlay(
            log_days,
            residuals,
            model_log_values,
            actual_log_values,
            np.array([False, True, False]),
            settings,
            current_r2=42.0,
        )

        self.assertTrue(np.allclose(result.model_values, 0.0))
        self.assertEqual(result.combined_r2, 42.0)
        self.assertEqual(result.amplitude, 0.0)
        self.assertEqual(result.angular_frequency, 0.0)
        self.assertEqual(result.phase_shift, 0.0)

    def test_compute_oscillator_overlay_recomputes_masked_trend_r2_when_fit_fails_with_multiple_points(
        self,
    ):
        log_days = np.array([1.0, 1.5, 2.0, 2.5], dtype=float)
        residuals = np.array([0.0, 0.1, -0.2, 0.0], dtype=float)
        model_log_values = np.array([10.0, 10.5, 11.0, 11.5], dtype=float)
        actual_log_values = model_log_values + residuals
        fit_mask = np.array([False, True, True, False])
        settings = oscillator.OscillatorSettings(
            t1_age=1.0,
            lambda_val=1.0,
        )

        result = oscillator.compute_oscillator_overlay(
            log_days,
            residuals,
            model_log_values,
            actual_log_values,
            fit_mask,
            settings,
            current_r2=42.0,
        )

        expected_r2 = (
            oscillator.calculate_r2_score(
                actual_log_values[fit_mask],
                model_log_values[fit_mask],
            )
            * 100.0
        )

        self.assertTrue(np.allclose(result.model_values, 0.0))
        self.assertTrue(np.isclose(result.combined_r2, expected_r2))
        self.assertTrue(np.isclose(result.reference_log_day, log_days[fit_mask][0]))

    def test_optimize_oscillator_parameters_returns_values_within_bounds(self):
        rng = np.random.default_rng(42)
        log_days = np.linspace(0.5, 3.0, 250)
        residuals = np.cos(2.5 * log_days) * np.exp(-0.3 * (log_days - log_days.min()))
        residuals = residuals + rng.normal(0.0, 0.01, size=log_days.shape)

        initial_params = {
            "t1_age": 2.49,
            "lambda_val": 2.01,
        }

        optimized = oscillator.optimize_oscillator_parameters(log_days, residuals, initial_params)

        self.assertTrue(0.5 <= optimized["t1_age"] <= 3.0)
        self.assertTrue(1.5 <= optimized["lambda_val"] <= 3.0)

    def test_optimize_single_oscillator_parameter_returns_value_in_bounds(self):
        log_days = np.linspace(0.5, 3.0, 250)
        residuals = np.cos(2.5 * log_days) * np.exp(-0.3 * (log_days - log_days.min()))
        current_params = {
            "t1_age": 2.49,
            "lambda_val": 2.01,
        }

        best_value, best_r2 = oscillator.optimize_single_oscillator_parameter(
            log_days,
            residuals,
            current_params,
            parameter_key="lambda_val",
            min_value=1.5,
            max_value=3.0,
            grid_points=25,
        )

        self.assertTrue(1.5 <= best_value <= 3.0)
        self.assertTrue(np.isfinite(best_r2))

    def test_optimize_oscillator_parameters_respects_lambda_override_bounds(self):
        log_days = np.linspace(0.5, 3.0, 250)
        residuals = np.cos(2.5 * log_days) * np.exp(-0.3 * (log_days - log_days.min()))
        initial_params = {
            "t1_age": 2.49,
            "lambda_val": 4.72,
        }

        optimized = oscillator.optimize_oscillator_parameters(
            log_days,
            residuals,
            initial_params,
            bounds_override={"lambda_val": (1.5, 8.0)},
            parameter_order=["t1_age", "lambda_val"],
        )

        self.assertTrue(0.5 <= optimized["t1_age"] <= 3.0)
        self.assertTrue(1.5 <= optimized["lambda_val"] <= 8.0)

    def test_optimize_visible_oscillator_parameters_matches_sequential_af_strategy(
        self,
    ):
        log_days = np.linspace(0.5, 3.0, 250)
        residuals = np.cos(2.5 * log_days) * np.exp(-0.3 * (log_days - log_days.min()))
        initial_params = {
            "t1_age": 1.61,
            "lambda_val": 5.00,
        }

        optimized = oscillator.optimize_visible_oscillator_parameters(
            log_days,
            residuals,
            initial_params,
            bounds_override={"lambda_val": (1.5, 8.0)},
            parameter_order=["t1_age", "lambda_val"],
            step_map={"t1_age": 0.01, "lambda_val": 0.01},
        )

        optimized_r2 = oscillator.compute_oscillator_fit_r2(
            log_days,
            residuals,
            optimized["t1_age"],
            optimized["lambda_val"],
        )
        baseline_r2 = oscillator.compute_oscillator_fit_r2(
            log_days,
            residuals,
            initial_params["t1_age"],
            initial_params["lambda_val"],
        )

        self.assertGreaterEqual(optimized_r2, baseline_r2)


if __name__ == "__main__":
    unittest.main()
