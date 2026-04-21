import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from core.power_law import (
    calculate_r2_for_manual_params,
    find_best_fit_params_for_offset,
    fit_powerlaw_regression,
)
from core.utils import (
    TrendComputationResult,
    calculate_r2_score,
    calculate_monthly_buy_portfolio_values,
    evaluate_powerlaw_values,
    get_stable_trend_fit,
    normalize_periodic_growth_rate,
    powerlaw_parameters_are_unstable,
    resolve_trend_parameters,
)


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

    def test_resolve_trend_parameters_preserves_supplied_values_in_powerlaw_mode(self):
        log_days = np.linspace(1.0, 3.0, 10)
        intercept = 0.5
        slope = 2.0
        log_prices = intercept + slope * log_days

        result = resolve_trend_parameters(
            log_days,
            log_prices,
            intercept_a=intercept,
            slope_b=slope,
            active_mode="PowerLaw",
        )

        self.assertIsInstance(result, TrendComputationResult)
        self.assertTrue(np.isclose(result.intercept_a, intercept))
        self.assertTrue(np.isclose(result.slope_b, slope))
        self.assertTrue(np.allclose(result.trend_log_prices, log_prices))
        self.assertTrue(np.allclose(result.residual_series, 0.0))

    def test_resolve_trend_parameters_refits_unstable_values_in_logperiodic_mode(self):
        log_days = np.linspace(1.0, 3.0, 150)
        true_intercept = 0.5
        true_slope = 2.0
        log_prices = true_intercept + true_slope * log_days

        result = resolve_trend_parameters(
            log_days,
            log_prices,
            intercept_a=-100.0,
            slope_b=100.0,
            active_mode="LogPeriodic",
        )

        self.assertTrue(np.isclose(result.intercept_a, true_intercept, atol=1e-10))
        self.assertTrue(np.isclose(result.slope_b, true_slope, atol=1e-10))
        self.assertTrue(np.allclose(result.trend_log_prices, log_prices))
        self.assertTrue(np.allclose(result.residual_series, 0.0))

    def test_resolve_trend_parameters_clips_powerlaw_mode_exponents_before_residuals(self):
        log_days = np.array([1.0, 2.0, 3.0], dtype=float)
        log_prices = np.array([5.0, 6.0, 7.0], dtype=float)

        result = resolve_trend_parameters(
            log_days,
            log_prices,
            intercept_a=250.0,
            slope_b=40.0,
            active_mode="PowerLaw",
        )

        self.assertTrue(np.all(result.trend_log_prices <= 300.0))
        self.assertTrue(np.allclose(result.residual_series, log_prices - result.trend_log_prices))

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

    @patch("core.power_law.fit_powerlaw_regression")
    def test_find_best_fit_params_for_offset_preserves_requested_offset(self, mock_fit):
        mock_fit.return_value = (5.5, -16.1, 0.98)

        offset, intercept, slope, r2 = find_best_fit_params_for_offset(
            np.array([10.0, 20.0, 30.0]),
            np.array([1.0, 2.0, 3.0]),
            genesis_offset_days=42,
        )

        self.assertEqual(offset, 42)
        self.assertEqual(intercept, -16.1)
        self.assertEqual(slope, 5.5)
        self.assertEqual(r2, 0.98)
        self.assertEqual(mock_fit.call_args.args[2], 42)

    def test_evaluate_powerlaw_values_clips_extreme_exponents(self):
        values, exponents, was_clipped = evaluate_powerlaw_values(
            np.array([1.0, 2.0, 3.0]),
            intercept_a=250.0,
            slope_b=40.0,
        )

        self.assertTrue(was_clipped)
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertTrue(np.all(exponents <= 300.0))

    def test_powerlaw_parameters_are_unstable_for_negative_r2_or_clipping(self):
        self.assertTrue(powerlaw_parameters_are_unstable(-0.01))
        self.assertTrue(powerlaw_parameters_are_unstable(0.5, was_clipped=True))
        self.assertFalse(powerlaw_parameters_are_unstable(0.5, was_clipped=False))

    def test_normalize_periodic_growth_rate_removes_month_length_variance(self):
        previous_values = np.array([100.0, 100.0])
        target_monthly_return = 0.03
        current_values = np.array(
            [
                100.0 * ((1.0 + target_monthly_return) ** (28.0 / 30.44)),
                100.0 * ((1.0 + target_monthly_return) ** (31.0 / 30.44)),
            ]
        )
        normalized = normalize_periodic_growth_rate(
            current_values,
            previous_values,
            np.array([28.0, 31.0]),
            30.44,
        )

        self.assertTrue(np.allclose(normalized, np.array([3.0, 3.0]), atol=1e-9))

    def test_calculate_monthly_buy_portfolio_values_keeps_hold_only_when_disabled(self):
        date_index = pd.to_datetime(["2026-01-15", "2026-02-15", "2026-03-15"])
        fair_prices = np.array([100.0, 110.0, 120.0], dtype=float)

        btc_holdings, portfolio_values, invested_capital = calculate_monthly_buy_portfolio_values(
            date_index=date_index,
            current_gen_date=pd.Timestamp("2009-01-03"),
            fair_prices=fair_prices,
            intercept_a=2.0,
            slope_b=0.0,
            initial_btc_amount=2.0,
            monthly_buy_amount=0.0,
            purchase_anchor_day=pd.Timestamp("2026-01-15"),
        )

        self.assertTrue(np.allclose(btc_holdings, np.array([2.0, 2.0, 2.0])))
        self.assertTrue(np.allclose(portfolio_values, fair_prices * 2.0))
        self.assertTrue(np.allclose(invested_capital, np.array([0.0, 0.0, 0.0])))

    def test_calculate_monthly_buy_portfolio_values_adds_btc_from_next_month(self):
        date_index = pd.to_datetime(["2026-01-15", "2026-02-15", "2026-03-15", "2026-04-15"])
        fair_prices = np.full(4, 100.0, dtype=float)

        btc_holdings, portfolio_values, invested_capital = calculate_monthly_buy_portfolio_values(
            date_index=date_index,
            current_gen_date=pd.Timestamp("2009-01-03"),
            fair_prices=fair_prices,
            intercept_a=2.0,
            slope_b=0.0,
            initial_btc_amount=2.0,
            monthly_buy_amount=100.0,
            purchase_anchor_day=pd.Timestamp("2026-01-15"),
        )

        self.assertTrue(np.allclose(btc_holdings, np.array([2.0, 3.0, 4.0, 5.0])))
        self.assertTrue(np.allclose(portfolio_values, np.array([200.0, 300.0, 400.0, 500.0])))
        self.assertTrue(np.allclose(invested_capital, np.array([0.0, 100.0, 200.0, 300.0])))

    def test_calculate_monthly_buy_portfolio_values_supports_zero_initial_btc(self):
        date_index = pd.to_datetime(["2026-01-15", "2026-02-15", "2026-03-15"])
        fair_prices = np.full(3, 100.0, dtype=float)

        btc_holdings, portfolio_values, invested_capital = calculate_monthly_buy_portfolio_values(
            date_index=date_index,
            current_gen_date=pd.Timestamp("2009-01-03"),
            fair_prices=fair_prices,
            intercept_a=2.0,
            slope_b=0.0,
            initial_btc_amount=0.0,
            monthly_buy_amount=100.0,
            purchase_anchor_day=pd.Timestamp("2026-01-15"),
        )

        self.assertTrue(np.allclose(btc_holdings, np.array([0.0, 1.0, 2.0])))
        self.assertTrue(np.allclose(portfolio_values, np.array([0.0, 100.0, 200.0])))
        self.assertTrue(np.allclose(invested_capital, np.array([0.0, 100.0, 200.0])))


if __name__ == "__main__":
    unittest.main()
