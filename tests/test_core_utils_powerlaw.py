import unittest

import numpy as np

from core.power_law import calculate_r2_for_manual_params, fit_powerlaw_regression
from core.utils import calculate_r2_score, get_stable_trend_fit


class TestCoreUtilsAndPowerLaw(unittest.TestCase):
    def test_calculate_r2_score_perfect_fit_returns_one(self):
        actual = np.array([1.0, 2.0, 3.0, 4.0])
        predicted = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(calculate_r2_score(actual, predicted), 1.0)

    def test_calculate_r2_score_constant_series_returns_zero(self):
        actual = np.array([5.0, 5.0, 5.0, 5.0])
        predicted = np.array([4.0, 5.0, 6.0, 7.0])
        self.assertEqual(calculate_r2_score(actual, predicted), 0.0)

    def test_get_stable_trend_fit_refits_on_unstable_residuals(self):
        log_days = np.linspace(1.0, 3.0, 150)
        true_intercept = 0.5
        true_slope = 2.0
        log_prices = true_intercept + true_slope * log_days

        intercept, slope, trend, residuals = get_stable_trend_fit(
            log_days,
            log_prices,
            intercept_a=-100.0,
            slope_b=100.0,
            residual_threshold=0.1,
        )

        self.assertTrue(np.isclose(intercept, true_intercept, atol=1e-10))
        self.assertTrue(np.isclose(slope, true_slope, atol=1e-10))
        self.assertTrue(np.allclose(trend, log_prices))
        self.assertTrue(np.allclose(residuals, 0.0))

    def test_fit_powerlaw_regression_returns_expected_parameters(self):
        days = np.arange(1, 301, dtype=float)
        slope_true = 5.8
        intercept_true = -17.0
        log_days = np.log10(days)
        log_prices = slope_true * log_days + intercept_true

        slope, intercept, r2 = fit_powerlaw_regression(days, log_prices, genesis_offset_days=0)

        self.assertTrue(np.isclose(slope, slope_true, atol=1e-12))
        self.assertTrue(np.isclose(intercept, intercept_true, atol=1e-12))
        self.assertTrue(np.isclose(r2, 1.0, atol=1e-12))

    def test_fit_powerlaw_regression_returns_zeros_when_too_few_points(self):
        days = np.arange(1, 60, dtype=float)
        log_prices = np.log10(days)

        slope, intercept, r2 = fit_powerlaw_regression(days, log_prices, genesis_offset_days=0)
        self.assertEqual((slope, intercept, r2), (0.0, 0.0, 0.0))

    def test_calculate_r2_for_manual_params_matches_perfect_fit(self):
        days = np.arange(1, 300, dtype=float)
        slope = 4.2
        intercept = -10.0
        log_prices = intercept + slope * np.log10(days)

        r2 = calculate_r2_for_manual_params(days, log_prices, 0, intercept, slope)
        self.assertTrue(np.isclose(r2, 1.0, atol=1e-12))


if __name__ == "__main__":
    unittest.main()
