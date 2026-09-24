import unittest

import numpy as np
import pandas as pd

from core.constants import FLOOR_MODEL_TROUGH_ENVELOPE
from core.power_law import build_causal_powerlaw_floor_prices
from core.utils import (
    PortfolioProjectionResult,
    PortfolioSettings,
    build_portfolio_real_data_backtest,
    build_portfolio_projection,
    build_portfolio_view_model,
    estimate_current_monthly_pension,
    get_growth_change_labels,
    interpolate_sigma_level_from_log_offset,
    normalize_periodic_growth_rate,
    rate_withdrawal_attractiveness,
    resolve_backtest_monthly_prices,
    resolve_projection_anchor_day,
    resolve_portfolio_scenario_log_offset,
)


def build_daily_powerlaw_prices(
    start,
    end,
    gen_date,
    *,
    intercept=0.0,
    slope=1.0,
    cycle_days=None,
    cycle_amplitude=0.3,
    tail_start=None,
    tail_multiplier=1.0,
):
    """Daily prices that follow a PowerLaw, optionally cyclical and with a regime break."""
    dates = pd.date_range(start, end, freq="D")
    days = np.maximum((dates - pd.Timestamp(gen_date)).days.to_numpy(dtype=float), 1.0)
    log_prices = intercept + slope * np.log10(days)
    if cycle_days is not None:
        log_prices = log_prices + cycle_amplitude * np.sin(2.0 * np.pi * days / float(cycle_days))
    prices = np.power(10.0, log_prices)
    if tail_start is not None:
        prices = np.where(
            dates >= pd.Timestamp(tail_start), prices * float(tail_multiplier), prices
        )
    return pd.DataFrame({"CloseDisplay": prices}, index=dates)


def average_month_growth(current_price, previous_price, elapsed_days):
    return previous_price * ((current_price / previous_price) ** (30.44 / elapsed_days) - 1.0)


class TestPortfolioHelpers(unittest.TestCase):
    def test_interpolate_sigma_level_from_log_offset_returns_exact_fractional_sigma(self):
        sigma_level = interpolate_sigma_level_from_log_offset(
            log_offset=0.3,
            percentile_offsets=(-1.0, -0.4, 0.6, 1.2),
        )

        self.assertTrue(np.isclose(sigma_level, 0.5))

    def test_estimate_current_monthly_pension_uses_current_exact_sigma_line(self):
        estimate = estimate_current_monthly_pension(
            current_price=3000.0,
            current_model_log=np.log10(3000.0),
            current_date=pd.Timestamp("2026-01-31"),
            current_gen_date=pd.Timestamp("2026-01-01"),
            intercept_a=2.0,
            slope_b=1.0,
            btc_amount=0.8,
            sell_mom_change_pct=50.0,
            percentile_offsets=(-0.5, -0.25, 0.25, 0.5),
        )

        # 30 days since the origin plus one average month: 100 * (30 + 30.44).
        next_price = 100.0 * (30.0 + 30.44)

        self.assertTrue(np.isclose(estimate.current_sigma_level, 0.0))
        self.assertTrue(np.isclose(estimate.current_floor_price, 3000.0 * 10**-0.5))
        self.assertTrue(np.isclose(estimate.next_month_floor_price, next_price * 10**-0.5))
        self.assertTrue(
            np.isclose(
                estimate.floor_monthly_growth_per_btc,
                (next_price - 3000.0) * 10**-0.5,
            )
        )
        self.assertTrue(
            np.isclose(
                estimate.minimum_monthly_withdrawal,
                (next_price - 3000.0) * 10**-0.5 * 0.8,
            )
        )
        self.assertTrue(np.isclose(estimate.next_month_price, next_price))
        self.assertTrue(np.isclose(estimate.monthly_growth_per_btc, next_price - 3000.0))
        self.assertTrue(np.isclose(estimate.max_monthly_withdrawal, (next_price - 3000.0) * 0.8))
        self.assertTrue(
            np.isclose(
                estimate.minimum_btc_to_sell,
                estimate.minimum_monthly_withdrawal / estimate.next_month_floor_price,
            )
        )
        self.assertTrue(
            np.isclose(
                estimate.minimum_btc_to_sell_today,
                estimate.minimum_monthly_withdrawal / estimate.current_price,
            )
        )
        self.assertTrue(
            np.isclose(
                estimate.minimum_btc_sell_reduction_pct,
                (1.0 - (estimate.minimum_btc_to_sell_today / estimate.minimum_btc_to_sell)) * 100.0,
            )
        )
        self.assertTrue(
            np.isclose(estimate.model_btc_to_sell, (next_price - 3000.0) * 0.8 / next_price)
        )
        self.assertTrue(
            np.isclose(estimate.selected_monthly_withdrawal, (next_price - 3000.0) * 0.4)
        )
        self.assertEqual(estimate.withdrawal_rating, "Attractive")

    def test_estimate_current_monthly_pension_ignores_calendar_month_length(self):
        gen_date = pd.Timestamp("2009-01-03")
        estimates = [
            estimate_current_monthly_pension(
                current_price=float((pd.Timestamp(current_date) - gen_date).days),
                current_model_log=np.log10((pd.Timestamp(current_date) - gen_date).days),
                current_date=pd.Timestamp(current_date),
                current_gen_date=gen_date,
                intercept_a=0.0,
                slope_b=1.0,
                btc_amount=1.0,
                sell_mom_change_pct=100.0,
                percentile_offsets=(-0.5, -0.25, 0.25, 0.5),
            )
            for current_date in ("2026-01-31", "2026-02-01")
        ]

        # A calendar month from 31 January reaches only 28 February, but the price grows by
        # 1 per day, so both dates should see the same average-month growth.
        self.assertTrue(np.isclose(estimates[0].monthly_growth_per_btc, 30.44))
        self.assertTrue(np.isclose(estimates[1].monthly_growth_per_btc, 30.44))

    def test_rate_withdrawal_attractiveness_uses_sigma_bands(self):
        self.assertEqual(rate_withdrawal_attractiveness(-1.2)[0], "Not attractive")
        self.assertEqual(rate_withdrawal_attractiveness(-0.4)[0], "Cautious")
        self.assertEqual(rate_withdrawal_attractiveness(0.4)[0], "Attractive")
        self.assertEqual(rate_withdrawal_attractiveness(1.2)[0], "Very attractive")

    def test_resolve_projection_anchor_day_uses_newer_of_data_and_today(self):
        df_index = pd.to_datetime(["2026-01-15", "2026-02-15"])

        anchor_day = resolve_projection_anchor_day(df_index, today=pd.Timestamp("2026-03-01"))

        self.assertEqual(anchor_day, pd.Timestamp("2026-03-01"))

    def test_build_portfolio_projection_uses_expected_frequency_and_labels(self):
        settings = PortfolioSettings(
            btc_amount=2.0,
            monthly_buy_amount=0.0,
            forecast_unit="Day",
            forecast_horizon=3,
        )

        result = build_portfolio_projection(
            df_index=pd.to_datetime(["2026-01-15"]),
            current_gen_date=pd.Timestamp("2009-01-03"),
            intercept_a=2.0,
            slope_b=0.0,
            settings=settings,
            anchor_day=pd.Timestamp("2026-01-15"),
        )

        self.assertEqual(result.table_title, "Daily growth table")
        self.assertEqual(result.change_usd_col, "DoD_USD")
        self.assertEqual(result.change_pct_col, "DoD_pct")
        self.assertEqual(result.forecast_unit, "Day")
        self.assertEqual(result.portfolio_df["Date"].iloc[0], pd.Timestamp("2026-01-14"))
        self.assertTrue((result.portfolio_df["Date"].diff().dt.days.iloc[1:] == 1).all())

    def test_build_portfolio_projection_prepends_the_requested_history_periods(self):
        """The history setting decides how many realised periods precede the anchor."""
        for forecast_unit, history_periods, expected_first_date in (
            # Default: nothing ahead of the anchor but the row the view model drops.
            ("Day", 0, pd.Timestamp("2026-03-14")),
            ("Month", 0, pd.Timestamp("2026-02-01")),
            ("Year", 0, pd.Timestamp("2025-01-01")),
            # A non-zero setting pushes the frame that many periods further back.
            ("Day", 5, pd.Timestamp("2026-03-09")),
            ("Month", 6, pd.Timestamp("2025-08-01")),
            ("Year", 3, pd.Timestamp("2022-01-01")),
        ):
            with self.subTest(forecast_unit=forecast_unit, history_periods=history_periods):
                settings = PortfolioSettings(
                    btc_amount=1.0,
                    monthly_buy_amount=0.0,
                    forecast_unit=forecast_unit,
                    forecast_horizon=3,
                    history_periods=history_periods,
                )

                result = build_portfolio_projection(
                    df_index=pd.to_datetime(["2026-03-15"]),
                    current_gen_date=pd.Timestamp("2009-01-03"),
                    intercept_a=2.0,
                    slope_b=0.0,
                    settings=settings,
                    anchor_day=pd.Timestamp("2026-03-15"),
                )

                self.assertEqual(result.history_periods, history_periods)
                self.assertEqual(result.portfolio_df["Date"].iloc[0], expected_first_date)
                self.assertEqual(len(result.portfolio_df), 3 + history_periods + 1)

    def test_build_portfolio_projection_clamps_the_history_periods_setting(self):
        for requested, expected in ((-5, 0), (500, 100)):
            with self.subTest(requested=requested):
                settings = PortfolioSettings(
                    btc_amount=1.0,
                    monthly_buy_amount=0.0,
                    forecast_unit="Month",
                    forecast_horizon=3,
                    history_periods=requested,
                )

                result = build_portfolio_projection(
                    df_index=pd.to_datetime(["2026-03-15"]),
                    current_gen_date=pd.Timestamp("2009-01-03"),
                    intercept_a=2.0,
                    slope_b=0.0,
                    settings=settings,
                    anchor_day=pd.Timestamp("2026-03-15"),
                )

                self.assertEqual(result.history_periods, expected)
                self.assertEqual(len(result.portfolio_df), 3 + expected + 1)

    def test_build_portfolio_projection_clips_period_days_to_one(self):
        settings = PortfolioSettings(
            btc_amount=1.0,
            monthly_buy_amount=0.0,
            forecast_unit="Year",
            forecast_horizon=2,
        )

        result = build_portfolio_projection(
            df_index=pd.to_datetime(["2026-01-15"]),
            current_gen_date=pd.Timestamp("2030-01-01"),
            intercept_a=2.0,
            slope_b=0.0,
            settings=settings,
            anchor_day=pd.Timestamp("2026-01-15"),
        )

        self.assertTrue(np.allclose(result.portfolio_df["FairPriceUSD"], 100.0))

    def test_build_portfolio_projection_month_uses_normalized_growth_rate(self):
        settings = PortfolioSettings(
            btc_amount=1.0,
            monthly_buy_amount=0.0,
            forecast_unit="Month",
            forecast_horizon=3,
        )

        result = build_portfolio_projection(
            df_index=pd.to_datetime(["2026-01-15"]),
            current_gen_date=pd.Timestamp("2009-01-03"),
            intercept_a=0.0,
            slope_b=1.0,
            settings=settings,
            anchor_day=pd.Timestamp("2026-03-15"),
        )

        expected_pct = normalize_periodic_growth_rate(
            result.portfolio_df["PortfolioUSD"].to_numpy(dtype=float),
            result.portfolio_df["PortfolioUSD"].shift(1).to_numpy(dtype=float),
            result.portfolio_df["Date"].diff().dt.days.to_numpy(dtype=float),
            30.44,
        )

        self.assertTrue(
            np.allclose(
                result.portfolio_df["MoM_pct"].to_numpy(dtype=float)[1:],
                expected_pct[1:],
                atol=1e-12,
                equal_nan=True,
            )
        )

    def test_build_portfolio_view_model_month_change_follows_remaining_btc_when_selling(self):
        settings = PortfolioSettings(
            btc_amount=2.0,
            monthly_buy_amount=0.0,
            monthly_mom_change_pct=25.0,
            forecast_unit="Month",
            forecast_horizon=8,
        )
        projection_result = build_portfolio_projection(
            df_index=pd.to_datetime(["2026-09-24"]),
            current_gen_date=pd.Timestamp("2009-01-03"),
            intercept_a=-17.0,
            slope_b=5.8,
            settings=settings,
            anchor_day=pd.Timestamp("2026-09-24"),
        )

        view_model = build_portfolio_view_model(
            projection_result,
            monthly_buy_amount=0.0,
            monthly_mom_change_pct=25.0,
            currency_unit="USD",
        )

        table = view_model.table_df.iloc[-3:]
        change = table["MoM Change (USD)"].to_numpy(dtype=float)
        not_selling_change = view_model.portfolio_display_df["MoM_USD"].iloc[-3:]
        self.assertTrue(np.all(change < not_selling_change.to_numpy(dtype=float)))
        self.assertTrue(np.allclose(table["Monthly withdrawal (USD)"], change * 0.25))

    def test_build_portfolio_projection_month_change_ignores_calendar_month_length(self):
        settings = PortfolioSettings(
            btc_amount=1.0,
            monthly_buy_amount=0.0,
            forecast_unit="Month",
            forecast_horizon=6,
        )

        # With B = 1 the price grows by exactly 1 per day, so a raw diff would follow
        # the 28-31 day month lengths while the normalised change stays flat.
        result = build_portfolio_projection(
            df_index=pd.to_datetime(["2026-01-15"]),
            current_gen_date=pd.Timestamp("2009-01-03"),
            intercept_a=0.0,
            slope_b=1.0,
            settings=settings,
            anchor_day=pd.Timestamp("2026-01-15"),
        )

        elapsed_days = result.portfolio_df["Date"].diff().dt.days.to_numpy(dtype=float)[1:]
        monthly_change = result.portfolio_df["MoM_USD"].to_numpy(dtype=float)[1:]
        self.assertIn(28.0, elapsed_days)
        self.assertIn(31.0, elapsed_days)
        self.assertTrue(np.allclose(monthly_change, 30.44, rtol=1e-3))

    def test_build_portfolio_projection_applies_sigma_scenario_multiplier(self):
        settings = PortfolioSettings(
            btc_amount=1.5,
            monthly_buy_amount=0.0,
            forecast_unit="Year",
            forecast_horizon=2,
            sigma_level=1,
            residual_sigma_log=np.log10(2.0),
        )

        result = build_portfolio_projection(
            df_index=pd.to_datetime(["2026-01-15"]),
            current_gen_date=pd.Timestamp("2009-01-03"),
            intercept_a=2.0,
            slope_b=0.0,
            settings=settings,
            anchor_day=pd.Timestamp("2026-01-15"),
        )

        self.assertTrue(np.allclose(result.portfolio_df["FairPriceUSD"], 200.0))
        self.assertTrue(np.allclose(result.portfolio_df["PortfolioUSD"], 300.0))

    def test_build_portfolio_projection_prefers_percentile_scenario_offsets(self):
        settings = PortfolioSettings(
            btc_amount=1.5,
            monthly_buy_amount=0.0,
            forecast_unit="Year",
            forecast_horizon=2,
            sigma_level=-2,
            residual_sigma_log=np.log10(0.25),
            residual_percentile_offsets_log=(
                np.log10(0.5),
                np.log10(0.75),
                np.log10(1.5),
                np.log10(2.0),
            ),
        )

        result = build_portfolio_projection(
            df_index=pd.to_datetime(["2026-01-15"]),
            current_gen_date=pd.Timestamp("2009-01-03"),
            intercept_a=2.0,
            slope_b=0.0,
            settings=settings,
            anchor_day=pd.Timestamp("2026-01-15"),
        )

        self.assertTrue(np.allclose(result.portfolio_df["FairPriceUSD"], 50.0))
        self.assertTrue(np.allclose(result.portfolio_df["PortfolioUSD"], 75.0))
        self.assertTrue(np.isclose(resolve_portfolio_scenario_log_offset(settings), np.log10(0.5)))

    def test_resolve_portfolio_scenario_log_offset_interpolates_half_sigma_levels(self):
        settings = PortfolioSettings(
            btc_amount=1.0,
            monthly_buy_amount=0.0,
            forecast_unit="Year",
            forecast_horizon=1,
            sigma_level=1.5,
            residual_percentile_offsets_log=(-0.4, -0.2, 0.2, 0.6),
        )

        self.assertTrue(np.isclose(resolve_portfolio_scenario_log_offset(settings), 0.4))

    def test_build_portfolio_projection_uses_sigma_scenario_for_monthly_buys(self):
        settings = PortfolioSettings(
            btc_amount=0.0,
            monthly_buy_amount=100.0,
            forecast_unit="Month",
            forecast_horizon=2,
            sigma_level=1,
            residual_sigma_log=np.log10(2.0),
        )

        result = build_portfolio_projection(
            df_index=pd.to_datetime(["2026-01-15"]),
            current_gen_date=pd.Timestamp("2009-01-03"),
            intercept_a=2.0,
            slope_b=0.0,
            settings=settings,
            anchor_day=pd.Timestamp("2026-03-15"),
        )

        dca_btc = result.portfolio_df["DcaBTC"].to_numpy(dtype=float)
        dca_value = result.portfolio_df["DcaPortfolioUSD"].to_numpy(dtype=float)

        # Buying starts in the month after the anchor, so only the final row holds BTC.
        self.assertTrue(np.allclose(dca_btc[:-1], 0.0))
        self.assertTrue(np.isclose(dca_btc[-1], 0.5))
        self.assertTrue(np.allclose(dca_value[:-1], 0.0))
        self.assertTrue(np.isclose(dca_value[-1], 100.0))

    def test_build_portfolio_projection_sells_percentage_of_positive_monthly_change(self):
        settings = PortfolioSettings(
            btc_amount=1.0,
            monthly_buy_amount=0.0,
            monthly_mom_change_pct=50.0,
            forecast_unit="Month",
            forecast_horizon=2,
        )

        result = build_portfolio_projection(
            df_index=pd.to_datetime(["2026-03-15"]),
            current_gen_date=pd.Timestamp("2026-01-01"),
            intercept_a=0.0,
            slope_b=1.0,
            settings=settings,
            anchor_day=pd.Timestamp("2026-03-15"),
        )

        expected_cash_flow = -(average_month_growth(90.0, 59.0, 31.0) * 1.0 * 0.5)
        expected_april_btc = 1.0 + (expected_cash_flow / 90.0)

        dca_btc = result.portfolio_df["DcaBTC"].to_numpy(dtype=float)
        invested_capital = result.portfolio_df["DcaInvestedCapitalUSD"].to_numpy(dtype=float)
        dca_value = result.portfolio_df["DcaPortfolioUSD"].to_numpy(dtype=float)

        self.assertTrue(np.allclose(dca_btc[:-1], 1.0))
        self.assertTrue(np.isclose(dca_btc[-1], expected_april_btc))
        self.assertTrue(np.allclose(invested_capital[:-1], 0.0))
        self.assertTrue(np.isclose(invested_capital[-1], expected_cash_flow))
        self.assertTrue(
            np.allclose(dca_value[-3:], np.array([31.0, 59.0, 90.0 + expected_cash_flow]))
        )

    def test_build_portfolio_projection_does_not_buy_on_negative_monthly_change(self):
        settings = PortfolioSettings(
            btc_amount=1.0,
            monthly_buy_amount=0.0,
            monthly_mom_change_pct=50.0,
            forecast_unit="Month",
            forecast_horizon=2,
        )

        result = build_portfolio_projection(
            df_index=pd.to_datetime(["2026-03-15"]),
            current_gen_date=pd.Timestamp("2025-11-01"),
            intercept_a=3.0,
            slope_b=-1.0,
            settings=settings,
            anchor_day=pd.Timestamp("2026-03-15"),
        )

        self.assertTrue(np.allclose(result.portfolio_df["DcaBTC"], 1.0))
        self.assertTrue(np.allclose(result.portfolio_df["DcaInvestedCapitalUSD"], 0.0))

    def test_build_portfolio_projection_clamps_monthly_change_sell_percentage(self):
        settings = PortfolioSettings(
            btc_amount=1.0,
            monthly_buy_amount=0.0,
            monthly_mom_change_pct=150.0,
            forecast_unit="Month",
            forecast_horizon=2,
        )

        result = build_portfolio_projection(
            df_index=pd.to_datetime(["2026-03-15"]),
            current_gen_date=pd.Timestamp("2026-01-01"),
            intercept_a=0.0,
            slope_b=1.0,
            settings=settings,
            anchor_day=pd.Timestamp("2026-03-15"),
        )

        expected_cash_flow = -average_month_growth(90.0, 59.0, 31.0)
        expected_april_btc = 1.0 + (expected_cash_flow / 90.0)

        dca_btc = result.portfolio_df["DcaBTC"].to_numpy(dtype=float)
        invested_capital = result.portfolio_df["DcaInvestedCapitalUSD"].to_numpy(dtype=float)

        self.assertTrue(np.allclose(dca_btc[:-1], 1.0))
        self.assertTrue(np.isclose(dca_btc[-1], expected_april_btc))
        self.assertTrue(np.allclose(invested_capital[:-1], 0.0))
        self.assertTrue(np.isclose(invested_capital[-1], expected_cash_flow))

    def test_build_portfolio_projection_sells_even_amounts_across_month_lengths(self):
        settings = PortfolioSettings(
            btc_amount=1.0,
            monthly_buy_amount=0.0,
            monthly_mom_change_pct=100.0,
            forecast_unit="Month",
            forecast_horizon=6,
        )

        # With B = 1 the price grows by exactly 1 per day, so calendar withdrawals would
        # follow the 28-31 day month lengths.
        result = build_portfolio_projection(
            df_index=pd.to_datetime(["2026-01-15"]),
            current_gen_date=pd.Timestamp("2009-01-03"),
            intercept_a=0.0,
            slope_b=1.0,
            settings=settings,
            anchor_day=pd.Timestamp("2026-01-15"),
        )

        withdrawals = -np.diff(result.portfolio_df["DcaInvestedCapitalUSD"].to_numpy(dtype=float))
        btc_before_sale = result.portfolio_df["DcaBTC"].to_numpy(dtype=float)[:-1]
        selling = withdrawals > 0.0
        self.assertGreaterEqual(np.count_nonzero(selling), 5)
        self.assertTrue(
            np.allclose(withdrawals[selling] / btc_before_sale[selling], 30.44, rtol=1e-3)
        )

        # Selling the whole growth still leaves the remaining BTC value close to flat.
        dca_value = result.portfolio_df["DcaPortfolioUSD"].to_numpy(dtype=float)[1:][selling]
        self.assertTrue(np.allclose(dca_value, dca_value[0], rtol=1e-3))

    def test_build_portfolio_view_model_excludes_baseline_and_adds_dca_columns(self):
        projection_result = PortfolioProjectionResult(
            portfolio_df=pd.DataFrame(
                {
                    "Date": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"]),
                    "FairPriceUSD": [100.0, 110.0, 120.0],
                    "PortfolioUSD": [200.0, 220.0, 240.0],
                    "DcaBTC": [2.0, 3.0, 4.0],
                    "DcaPortfolioUSD": [200.0, 330.0, 480.0],
                    "DcaInvestedCapitalUSD": [0.0, 100.0, 200.0],
                    "MoM_USD": [np.nan, 20.0, 20.0],
                    "MoM_pct": [np.nan, 10.0, 9.09],
                }
            ),
            table_title="Monthly growth table",
            forecast_unit="Month",
            change_usd_col="MoM_USD",
            change_pct_col="MoM_pct",
            history_periods=0,
        )

        view_model = build_portfolio_view_model(
            projection_result,
            monthly_buy_amount=100.0,
            currency_unit="USD",
        )

        self.assertEqual(len(view_model.portfolio_display_df), 2)
        self.assertTrue(view_model.dca_enabled)
        self.assertEqual(view_model.baseline_value, 220.0)
        self.assertEqual(view_model.last_value, 240.0)
        self.assertEqual(view_model.last_dca_value, 480.0)
        self.assertEqual(view_model.last_dca_invested_capital, 200.0)
        self.assertAlmostEqual(view_model.total_growth_pct, (240.0 / 220.0 - 1.0) * 100.0)
        self.assertIn("Remaining BTC value (USD)", view_model.table_df.columns)
        self.assertIn("Monthly withdrawal (USD)", view_model.table_df.columns)
        self.assertIn("Net cash flow (USD)", view_model.table_df.columns)
        self.assertIn("BTC after monthly cash flow", view_model.table_df.columns)
        self.assertTrue(
            np.allclose(view_model.table_df["Monthly withdrawal (USD)"], np.array([0.0, 0.0]))
        )

    def test_build_portfolio_view_model_measures_growth_from_the_anchor_period(self):
        """Total growth covers the forecast horizon, not the prepended history rows."""
        settings = PortfolioSettings(
            btc_amount=1.0,
            monthly_buy_amount=0.0,
            forecast_unit="Month",
            forecast_horizon=3,
        )

        projection_result = build_portfolio_projection(
            df_index=pd.to_datetime(["2026-03-15"]),
            current_gen_date=pd.Timestamp("2026-01-01"),
            intercept_a=0.0,
            slope_b=1.0,
            settings=settings,
            anchor_day=pd.Timestamp("2026-03-15"),
        )
        view_model = build_portfolio_view_model(
            projection_result,
            monthly_buy_amount=0.0,
            currency_unit="USD",
        )

        portfolio_df = projection_result.portfolio_df
        anchor_rows = portfolio_df.loc[portfolio_df["Date"] == pd.Timestamp("2026-03-01")]
        self.assertEqual(len(anchor_rows), 1)
        anchor_value = float(anchor_rows["PortfolioUSD"].iloc[0])
        oldest_history_value = float(portfolio_df["PortfolioUSD"].iloc[0])

        self.assertGreater(anchor_value, oldest_history_value)
        self.assertEqual(view_model.baseline_value, anchor_value)
        self.assertAlmostEqual(
            view_model.total_growth_pct,
            ((view_model.last_value / anchor_value) - 1.0) * 100.0,
        )

        # The table highlights exactly the rows that precede the anchor period.
        display_dates = list(view_model.portfolio_display_df["Date"])
        self.assertEqual(
            display_dates.index(pd.Timestamp("2026-03-01")),
            projection_result.history_periods,
        )

    def test_build_portfolio_view_model_enables_dca_for_monthly_change_percentage(self):
        projection_result = PortfolioProjectionResult(
            portfolio_df=pd.DataFrame(
                {
                    "Date": pd.to_datetime(["2026-01-01", "2026-02-01"]),
                    "FairPriceUSD": [100.0, 120.0],
                    "PortfolioUSD": [100.0, 120.0],
                    "DcaBTC": [1.0, 1.1],
                    "DcaPortfolioUSD": [100.0, 132.0],
                    "DcaInvestedCapitalUSD": [0.0, 12.0],
                    "MoM_USD": [np.nan, 20.0],
                    "MoM_pct": [np.nan, 20.0],
                }
            ),
            table_title="Monthly growth table",
            forecast_unit="Month",
            change_usd_col="MoM_USD",
            change_pct_col="MoM_pct",
            history_periods=0,
        )

        view_model = build_portfolio_view_model(
            projection_result,
            monthly_buy_amount=0.0,
            monthly_mom_change_pct=60.0,
            currency_unit="USD",
        )

        self.assertTrue(view_model.dca_enabled)
        self.assertIn("Remaining BTC value (USD)", view_model.table_df.columns)

    def test_build_portfolio_view_model_shows_monthly_withdrawal_as_positive_amount(self):
        projection_result = PortfolioProjectionResult(
            portfolio_df=pd.DataFrame(
                {
                    "Date": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"]),
                    "FairPriceUSD": [100.0, 110.0, 120.0],
                    "PortfolioUSD": [200.0, 220.0, 240.0],
                    "DcaBTC": [2.0, 1.9, 1.75],
                    "DcaPortfolioUSD": [200.0, 209.0, 210.0],
                    "DcaInvestedCapitalUSD": [0.0, -100.0, -250.0],
                    "MoM_USD": [np.nan, 20.0, 20.0],
                    "MoM_pct": [np.nan, 10.0, 9.09],
                }
            ),
            table_title="Monthly growth table",
            forecast_unit="Month",
            change_usd_col="MoM_USD",
            change_pct_col="MoM_pct",
            history_periods=0,
        )

        view_model = build_portfolio_view_model(
            projection_result,
            monthly_buy_amount=0.0,
            monthly_mom_change_pct=100.0,
            currency_unit="USD",
        )

        self.assertTrue(
            np.allclose(view_model.table_df["Monthly withdrawal (USD)"], np.array([100.0, 150.0]))
        )

    def test_build_portfolio_view_model_avoids_divide_by_zero_growth(self):
        projection_result = PortfolioProjectionResult(
            portfolio_df=pd.DataFrame(
                {
                    "Date": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"]),
                    "FairPriceUSD": [100.0, 100.0, 120.0],
                    "PortfolioUSD": [0.0, 0.0, 120.0],
                    "DcaBTC": [0.0, 0.0, 1.0],
                    "DcaPortfolioUSD": [0.0, 0.0, 120.0],
                    "DcaInvestedCapitalUSD": [0.0, 0.0, 100.0],
                    "MoM_USD": [np.nan, 0.0, 120.0],
                    "MoM_pct": [np.nan, np.nan, np.nan],
                }
            ),
            table_title="Monthly growth table",
            forecast_unit="Month",
            change_usd_col="MoM_USD",
            change_pct_col="MoM_pct",
            history_periods=0,
        )

        view_model = build_portfolio_view_model(
            projection_result,
            monthly_buy_amount=0.0,
            currency_unit="USD",
        )

        self.assertFalse(view_model.dca_enabled)
        self.assertEqual(view_model.total_growth_pct, 0.0)
        self.assertNotIn("Remaining BTC value (USD)", view_model.table_df.columns)

    def test_get_growth_change_labels_maps_supported_units(self):
        self.assertEqual(
            get_growth_change_labels("Year", "USD"),
            ("YoY Change (USD)", "YoY Change (%)"),
        )
        self.assertEqual(
            get_growth_change_labels("Day", "BTC"),
            ("DoD Change (BTC)", "DoD Change (%)"),
        )
        self.assertEqual(
            get_growth_change_labels("Month", "EUR"),
            ("MoM Change (EUR)", "MoM Change (%)"),
        )

    def test_build_portfolio_real_data_backtest_applies_historical_strategy(self):
        dates = pd.date_range("2021-01-01", periods=72, freq="MS")
        price_df = pd.DataFrame(
            {"CloseDisplay": np.linspace(100.0, 200.0, len(dates))},
            index=dates,
        )
        settings = PortfolioSettings(
            btc_amount=2.0,
            monthly_buy_amount=10.0,
            monthly_mom_change_pct=50.0,
            forecast_unit="Month",
            forecast_horizon=12,
        )

        result = build_portfolio_real_data_backtest(price_df, settings, "USD", years=5)

        self.assertIsNotNone(result)
        self.assertGreaterEqual(len(result.backtest_df), 60)
        self.assertIn("Strategy value (USD)", result.table_df.columns)
        self.assertIn("Historical monthly withdrawal (USD)", result.table_df.columns)
        self.assertIn("BTC after strategy", result.table_df.columns)
        self.assertGreaterEqual(result.strategy_btc, 0.0)
        self.assertTrue((result.backtest_df["MonthlyWithdrawal"] >= 0.0).all())

    def test_resolve_backtest_monthly_prices_dates_rows_on_the_traded_day(self):
        """Each row is dated on the day it trades, not on the month-start label."""
        gen_date = pd.Timestamp("2016-01-01")
        # A month that ends mid-month, as the newest month of a live snapshot does.
        price_df = build_daily_powerlaw_prices("2020-01-01", "2026-09-13", gen_date)

        monthly_prices = resolve_backtest_monthly_prices(price_df, 5)

        self.assertEqual(monthly_prices.index[-1], pd.Timestamp("2026-09-13"))
        self.assertTrue((monthly_prices.index.day > 1).any())
        # One row per month, each carrying that month's closing price.
        periods = monthly_prices.index.to_period("M")
        self.assertEqual(len(set(periods)), len(monthly_prices))
        for date, price in monthly_prices.items():
            self.assertEqual(price, price_df.loc[date, "CloseDisplay"])

    def test_build_causal_floor_prices_ignore_prices_from_after_each_month(self):
        """The defining property: a month's floor must not move when later prices arrive."""
        gen_date = pd.Timestamp("2016-01-01")
        regime_break = pd.Timestamp("2021-01-01")
        # Identical history up to the break, then a tenfold regime change afterwards.
        with_future = build_daily_powerlaw_prices(
            "2016-01-02",
            "2026-01-01",
            gen_date,
            tail_start=regime_break,
            tail_multiplier=10.0,
        )
        past_only = with_future.loc[: regime_break - pd.Timedelta(days=1)]
        monthly_index = pd.date_range("2019-01-01", "2020-12-01", freq="MS")

        floor_with_future = build_causal_powerlaw_floor_prices(
            with_future["CloseDisplay"], monthly_index, gen_date
        )
        floor_without_future = build_causal_powerlaw_floor_prices(
            past_only["CloseDisplay"], monthly_index, gen_date
        )

        self.assertTrue(np.all(np.isfinite(floor_with_future.to_numpy(dtype=float))))
        self.assertTrue(np.allclose(floor_with_future, floor_without_future))

        # A full-sample fit would move these same months, which is the bias removed here.
        full_days = np.maximum(
            (with_future.index - gen_date).days.to_numpy(dtype=float),
            1.0,
        )
        full_slope, full_intercept = np.polyfit(
            np.log10(full_days),
            np.log10(with_future["CloseDisplay"].to_numpy(dtype=float)),
            1,
        )
        monthly_days = np.maximum(
            (monthly_index - gen_date).days.to_numpy(dtype=float),
            1.0,
        )
        full_sample_floor = np.power(10.0, full_intercept + full_slope * np.log10(monthly_days))
        self.assertFalse(
            np.allclose(full_sample_floor, floor_with_future.to_numpy(dtype=float), rtol=0.05)
        )

    def test_build_causal_floor_prices_ignore_the_future_for_the_trough_envelope(self):
        gen_date = pd.Timestamp("2016-01-01")
        regime_break = pd.Timestamp("2021-01-01")
        with_future = build_daily_powerlaw_prices(
            "2016-01-02",
            "2026-01-01",
            gen_date,
            cycle_days=500,
            tail_start=regime_break,
            tail_multiplier=10.0,
        )
        past_only = with_future.loc[: regime_break - pd.Timedelta(days=1)]
        monthly_index = pd.date_range("2019-01-01", "2020-12-01", freq="MS")

        floor_with_future = build_causal_powerlaw_floor_prices(
            with_future["CloseDisplay"],
            monthly_index,
            gen_date,
            floor_model=FLOOR_MODEL_TROUGH_ENVELOPE,
        )
        floor_without_future = build_causal_powerlaw_floor_prices(
            past_only["CloseDisplay"],
            monthly_index,
            gen_date,
            floor_model=FLOOR_MODEL_TROUGH_ENVELOPE,
        )

        self.assertTrue(np.all(np.isfinite(floor_with_future.to_numpy(dtype=float))))
        self.assertTrue(np.allclose(floor_with_future, floor_without_future))

    def test_build_causal_floor_prices_leave_months_without_history_unfitted(self):
        gen_date = pd.Timestamp("2016-01-01")
        daily_prices = build_daily_powerlaw_prices("2016-01-02", "2016-12-31", gen_date)
        monthly_index = pd.date_range("2016-02-01", "2016-12-01", freq="MS")

        floor_prices = build_causal_powerlaw_floor_prices(
            daily_prices["CloseDisplay"], monthly_index, gen_date
        )

        # 100 daily rows are required before a fit is attempted, so early months stay NaN.
        self.assertTrue(np.isnan(floor_prices.iloc[0]))
        self.assertTrue(np.isfinite(floor_prices.iloc[-1]))

    def test_build_portfolio_real_data_backtest_can_sell_floor_growth(self):
        gen_date = pd.Timestamp("2016-01-01")
        price_df = build_daily_powerlaw_prices("2016-01-02", "2026-01-01", gen_date)
        settings = PortfolioSettings(
            btc_amount=1.0,
            monthly_buy_amount=0.0,
            monthly_mom_change_pct=0.0,
            forecast_unit="Month",
            forecast_horizon=12,
        )
        monthly_prices = resolve_backtest_monthly_prices(price_df, 5)
        floor_prices = build_causal_powerlaw_floor_prices(
            price_df["CloseDisplay"], monthly_prices.index, gen_date
        )

        full_sell = build_portfolio_real_data_backtest(
            price_df,
            settings,
            "USD",
            years=5,
            floor_prices=floor_prices,
            sell_mom_change_pct=100.0,
            strategy_name="-2σ floor: sell 100% growth",
        )
        half_sell = build_portfolio_real_data_backtest(
            price_df,
            settings,
            "USD",
            years=5,
            floor_prices=floor_prices,
            sell_mom_change_pct=50.0,
            strategy_name="-2σ floor: sell 50% growth",
        )

        self.assertIsNotNone(full_sell)
        self.assertIsNotNone(half_sell)
        self.assertEqual(full_sell.monthly_withdrawal_label, "-2σ monthly withdrawal (USD)")
        self.assertEqual(full_sell.months_without_floor, 0)
        self.assertGreater(full_sell.backtest_df["MonthlyWithdrawal"].sum(), 0.0)
        self.assertGreater(
            full_sell.backtest_df["MonthlyWithdrawal"].sum(),
            half_sell.backtest_df["MonthlyWithdrawal"].sum(),
        )
        self.assertLess(full_sell.strategy_btc, half_sell.strategy_btc)

    def test_build_portfolio_real_data_backtest_uses_starting_capital(self):
        dates = pd.date_range("2021-01-01", periods=72, freq="MS")
        price_df = pd.DataFrame({"CloseDisplay": np.full(len(dates), 100.0)}, index=dates)
        settings = PortfolioSettings(
            btc_amount=999.0,
            monthly_buy_amount=0.0,
            monthly_mom_change_pct=0.0,
            forecast_unit="Month",
            forecast_horizon=12,
        )

        result = build_portfolio_real_data_backtest(
            price_df,
            settings,
            "USD",
            years=5,
            sell_mom_change_pct=150.0,
            initial_capital=250.0,
        )

        self.assertIsNotNone(result)
        self.assertTrue(np.isclose(result.start_value, 250.0))
        self.assertTrue(np.isclose(result.strategy_btc, 2.5))
        self.assertEqual(result.sell_mom_change_pct, 150.0)


if __name__ == "__main__":
    unittest.main()
