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

        self.assertTrue(0.1 <= optimized["t1_age"] <= 5.0)
        self.assertTrue(1.5 <= optimized["lambda_val"] <= 3.0)
        self.assertTrue(0.1 <= optimized["amp_factor_top"] <= 10.0)
        self.assertTrue(0.1 <= optimized["amp_factor_bottom"] <= 10.0)
        self.assertTrue(0.0 <= optimized["impulse_damping"] <= 2.0)


if __name__ == "__main__":
    unittest.main()
