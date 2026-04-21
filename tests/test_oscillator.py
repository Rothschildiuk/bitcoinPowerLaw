import unittest

import numpy as np

from core import oscillator


class TestOscillator(unittest.TestCase):
    def test_fit_oscillator_component_returns_none_when_lambda_is_invalid(self):
        log_days = np.linspace(1.0, 2.0, 128)
        residuals = np.sin(log_days)

        result = oscillator.fit_oscillator_component(
            log_days,
            residuals,
            t1_cycle_age_years=1.0,
            cycle_lambda=1.0,
            top_amplitude_factor=1.0,
            bottom_amplitude_factor=1.0,
            impulse_damping=0.0,
        )
        self.assertIsNone(result)

    def test_build_oscillator_curve_applies_asymmetric_amplitude_scaling(self):
        log_days = np.array([0.0, 1.0, 2.0, 3.0])
        curve = oscillator.build_oscillator_curve(
            log_days=log_days,
            amplitude=2.0,
            angular_frequency=3.0,
            phase_shift=0.0,
            top_amplitude_factor=1.7,
            bottom_amplitude_factor=0.5,
            damping_factor=0.0,
            reference_log_day=0.0,
        )

        positives = curve[curve > 0]
        negatives = np.abs(curve[curve < 0])
        self.assertGreater(positives.size, 0)
        self.assertGreater(negatives.size, 0)
        self.assertGreater(np.max(positives), np.max(negatives))

    def test_compute_impulse_decay_is_one_at_reference_and_decreases_after(self):
        log_days = np.array([1.0, 1.5, 2.0])
        decay = oscillator.compute_impulse_decay(
            log_days,
            damping_factor=0.8,
            reference_log_day=1.0,
        )

        self.assertTrue(np.isclose(decay[0], 1.0))
        self.assertLess(decay[1], decay[0])
        self.assertLess(decay[2], decay[1])

    def test_compute_oscillator_overlay_returns_full_length_series_from_masked_fit(self):
        log_days = np.linspace(0.5, 3.0, 250)
        settings = oscillator.OscillatorSettings(
            t1_age=1.0,
            lambda_val=2.01,
            amp_factor_top=1.0,
            amp_factor_bottom=1.0,
            impulse_damping=0.0,
        )
        fit_result = oscillator.fit_oscillator_component(
            log_days,
            np.cos(log_days),
            settings.t1_age,
            settings.lambda_val,
            settings.amp_factor_top,
            settings.amp_factor_bottom,
            settings.impulse_damping,
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

    def test_compute_oscillator_overlay_keeps_baseline_r2_when_fit_is_not_possible(self):
        log_days = np.array([1.0, 2.0, 3.0])
        residuals = np.array([0.1, 0.2, 0.3])
        model_log_values = np.array([10.0, 10.1, 10.2])
        actual_log_values = model_log_values + residuals
        settings = oscillator.OscillatorSettings(
            t1_age=1.0,
            lambda_val=1.0,
            amp_factor_top=1.0,
            amp_factor_bottom=1.0,
            impulse_damping=0.0,
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

    def test_compute_oscillator_overlay_recomputes_masked_trend_r2_when_fit_fails_with_multiple_points(self):
        log_days = np.array([1.0, 1.5, 2.0, 2.5], dtype=float)
        residuals = np.array([0.0, 0.1, -0.2, 0.0], dtype=float)
        model_log_values = np.array([10.0, 10.5, 11.0, 11.5], dtype=float)
        actual_log_values = model_log_values + residuals
        fit_mask = np.array([False, True, True, False])
        settings = oscillator.OscillatorSettings(
            t1_age=1.0,
            lambda_val=1.0,
            amp_factor_top=1.0,
            amp_factor_bottom=1.0,
            impulse_damping=0.0,
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

        expected_r2 = oscillator.calculate_r2_score(
            actual_log_values[fit_mask],
            model_log_values[fit_mask],
        ) * 100.0

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
            "amp_factor_top": 1.13,
            "amp_factor_bottom": 0.88,
            "impulse_damping": 1.71,
        }

        optimized = oscillator.optimize_oscillator_parameters(log_days, residuals, initial_params)

        self.assertTrue(0.5 <= optimized["t1_age"] <= 3.0)
        self.assertTrue(1.5 <= optimized["lambda_val"] <= 3.0)
        self.assertTrue(0.1 <= optimized["amp_factor_top"] <= 10.0)
        self.assertTrue(0.1 <= optimized["amp_factor_bottom"] <= 10.0)
        self.assertTrue(0.0 <= optimized["impulse_damping"] <= 2.0)

    def test_optimize_single_oscillator_parameter_returns_value_in_bounds(self):
        log_days = np.linspace(0.5, 3.0, 250)
        residuals = np.cos(2.5 * log_days) * np.exp(-0.3 * (log_days - log_days.min()))
        current_params = {
            "t1_age": 2.49,
            "lambda_val": 2.01,
            "amp_factor_top": 1.13,
            "amp_factor_bottom": 0.88,
            "impulse_damping": 1.71,
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
            "amp_factor_top": 0.69,
            "amp_factor_bottom": 0.78,
            "impulse_damping": 0.25,
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

    def test_optimize_visible_oscillator_parameters_matches_sequential_af_strategy(self):
        log_days = np.linspace(0.5, 3.0, 250)
        residuals = np.cos(2.5 * log_days) * np.exp(-0.3 * (log_days - log_days.min()))
        initial_params = {
            "t1_age": 1.61,
            "lambda_val": 5.00,
            "amp_factor_top": 0.68,
            "amp_factor_bottom": 0.74,
            "impulse_damping": 0.11,
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
            optimized["amp_factor_top"],
            optimized["amp_factor_bottom"],
            optimized["impulse_damping"],
        )
        baseline_r2 = oscillator.compute_oscillator_fit_r2(
            log_days,
            residuals,
            initial_params["t1_age"],
            initial_params["lambda_val"],
            initial_params["amp_factor_top"],
            initial_params["amp_factor_bottom"],
            initial_params["impulse_damping"],
        )

        self.assertGreaterEqual(optimized_r2, baseline_r2)


if __name__ == "__main__":
    unittest.main()
