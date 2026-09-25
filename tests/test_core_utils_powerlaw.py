import unittest

import numpy as np
import pandas as pd

from core.power_law import (
    calculate_r2_for_manual_params,
    calculate_r2_for_manual_params_on_rolling_mean,
    fit_peak_powerlaw_envelope,
    fit_powerlaw_regression,
    fit_trough_powerlaw_envelope,
)
from core.utils import (
    TrendComputationResult,
    calculate_expanding_powerlaw_parameters,
    calculate_expanding_powerlaw_r2,
    calculate_r2_score,
    calculate_monthly_buy_portfolio_values,
    evaluate_powerlaw_values,
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

    def test_fit_peak_powerlaw_envelope_uses_local_highs(self):
        absolute_days = np.array([10.0, 20.0, 30.0, 100.0, 110.0, 120.0, 200.0, 210.0])
        log_prices = np.array([1.0, 1.4, 1.1, 2.0, 2.4, 2.1, 3.0, 3.4])
        model_days = np.array([10.0, 100.0, 200.0])

        result = fit_peak_powerlaw_envelope(
            absolute_days,
            log_prices,
            genesis_offset_days=0.0,
            model_days=model_days,
            window_days=40.0,
            min_peak_count=3,
        )

        self.assertIsNotNone(result)
        self.assertTrue(np.allclose(result["peak_days"], np.array([20.0, 110.0, 210.0])))
        self.assertEqual(len(result["model_values"]), len(model_days))

    def test_fit_trough_powerlaw_envelope_uses_local_lows(self):
        absolute_days = np.array([10.0, 20.0, 30.0, 100.0, 110.0, 120.0, 200.0, 210.0])
        log_prices = np.array([1.4, 1.0, 1.3, 2.4, 2.0, 2.3, 3.4, 3.0])
        model_days = np.array([10.0, 100.0, 200.0])

        result = fit_trough_powerlaw_envelope(
            absolute_days,
            log_prices,
            genesis_offset_days=0.0,
            model_days=model_days,
            window_days=40.0,
            min_trough_count=3,
        )

        self.assertIsNotNone(result)
        self.assertTrue(np.allclose(result["trough_days"], np.array([20.0, 110.0, 210.0])))
        self.assertEqual(len(result["model_values"]), len(model_days))

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
        )

        self.assertIsInstance(result, TrendComputationResult)
        self.assertTrue(np.isclose(result.intercept_a, intercept))
        self.assertTrue(np.isclose(result.slope_b, slope))
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

    def test_calculate_r2_for_manual_params_on_rolling_mean_uses_smoothed_values(self):
        days = np.arange(1, 220, dtype=float)
        slope = -0.1
        intercept = 2.0
        window_days = 3
        rolling_days = days[window_days - 1 :]
        target_rolling_values = 10 ** (intercept + slope * np.log10(rolling_days))
        values = np.empty_like(days)
        values[0] = target_rolling_values[0]
        values[1] = target_rolling_values[0]
        for idx, target in enumerate(target_rolling_values, start=window_days - 1):
            values[idx] = (target * window_days) - values[idx - 1] - values[idx - 2]

        r2 = calculate_r2_for_manual_params_on_rolling_mean(
            days,
            values,
            0,
            intercept,
            slope,
            window_days=window_days,
        )

        self.assertTrue(np.isclose(r2, 1.0, atol=1e-12))

    def test_evaluate_powerlaw_values_clips_extreme_exponents(self):
        values, exponents, was_clipped = evaluate_powerlaw_values(
            np.array([1.0, 2.0, 3.0]),
            intercept_a=250.0,
            slope_b=40.0,
        )

        self.assertTrue(was_clipped)
        self.assertTrue(np.all(np.isfinite(values)))
        self.assertTrue(np.all(exponents <= 300.0))

    def test_calculate_expanding_powerlaw_parameters_uses_prefix_regressions(self):
        log_days = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
        log_prices = np.array([3.0, 5.0, 7.0, 20.0], dtype=float)

        intercepts, slopes, fitted = calculate_expanding_powerlaw_parameters(
            log_days,
            log_prices,
            min_points=3,
        )
        expected_last_slope, expected_last_intercept = np.polyfit(log_days, log_prices, 1)
        expected_last = expected_last_intercept + expected_last_slope * log_days[-1]

        self.assertTrue(np.isnan(fitted[0]))
        self.assertTrue(np.isnan(fitted[1]))
        self.assertTrue(np.isclose(fitted[2], log_prices[2], atol=1e-12))
        self.assertTrue(np.isclose(fitted[3], expected_last, atol=1e-12))
        self.assertTrue(np.isclose(slopes[2], 2.0, atol=1e-12))
        self.assertTrue(np.isclose(intercepts[2], 1.0, atol=1e-12))
        self.assertTrue(np.isclose(slopes[3], expected_last_slope, atol=1e-12))

    def test_calculate_expanding_powerlaw_r2_matches_each_prefix_fit(self):
        log_days = np.log10(np.arange(1.0, 301.0))
        log_prices = 2.0 + 5.0 * log_days + 0.3 * np.sin(np.arange(300.0))

        r2_values = calculate_expanding_powerlaw_r2(log_days, log_prices, min_points=100)

        self.assertTrue(np.isnan(r2_values[:99]).all())
        for count in (100, 300):
            slope, intercept = np.polyfit(log_days[:count], log_prices[:count], 1)
            expected = calculate_r2_score(log_prices[:count], intercept + slope * log_days[:count])
            self.assertTrue(np.isclose(r2_values[count - 1], expected, atol=1e-9))

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

    def test_calculate_monthly_buy_portfolio_values_supports_monthly_sells(self):
        date_index = pd.to_datetime(["2026-01-15", "2026-02-15", "2026-03-15", "2026-04-15"])
        fair_prices = np.full(4, 100.0, dtype=float)

        btc_holdings, portfolio_values, invested_capital = calculate_monthly_buy_portfolio_values(
            date_index=date_index,
            current_gen_date=pd.Timestamp("2009-01-03"),
            fair_prices=fair_prices,
            intercept_a=2.0,
            slope_b=0.0,
            initial_btc_amount=2.0,
            monthly_buy_amount=-100.0,
            purchase_anchor_day=pd.Timestamp("2026-01-15"),
        )

        self.assertTrue(np.allclose(btc_holdings, np.array([2.0, 1.0, 0.0, 0.0])))
        self.assertTrue(np.allclose(portfolio_values, np.array([200.0, 100.0, 0.0, 0.0])))
        self.assertTrue(np.allclose(invested_capital, np.array([0.0, -100.0, -200.0, -200.0])))

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
