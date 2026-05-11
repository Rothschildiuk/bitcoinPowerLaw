import unittest

import numpy as np
import pandas as pd

from core.utils import (
    PortfolioProjectionResult,
    PortfolioSettings,
    build_portfolio_projection,
    build_portfolio_view_model,
    estimate_current_monthly_pension,
    get_growth_change_labels,
    interpolate_sigma_level_from_log_offset,
    normalize_periodic_growth_rate,
    rate_withdrawal_attractiveness,
    resolve_projection_anchor_day,
    resolve_portfolio_scenario_log_offset,
)


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

        self.assertTrue(np.isclose(estimate.current_sigma_level, 0.0))
        self.assertTrue(np.isclose(estimate.current_floor_price, 3000.0 * 10**-0.5))
        self.assertTrue(np.isclose(estimate.next_month_floor_price, 5800.0 * 10**-0.5))
        self.assertTrue(
            np.isclose(
                estimate.floor_monthly_growth_per_btc,
                (5800.0 - 3000.0) * 10**-0.5,
            )
        )
        self.assertTrue(
            np.isclose(
                estimate.minimum_monthly_withdrawal,
                (5800.0 - 3000.0) * 10**-0.5 * 0.8,
            )
        )
        self.assertTrue(np.isclose(estimate.next_month_price, 5800.0))
        self.assertTrue(np.isclose(estimate.monthly_growth_per_btc, 2800.0))
        self.assertTrue(np.isclose(estimate.max_monthly_withdrawal, 2240.0))
        self.assertTrue(np.isclose(estimate.selected_monthly_withdrawal, 1120.0))
        self.assertEqual(estimate.withdrawal_rating, "Attractive")

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
        self.assertEqual(list(result.portfolio_df["Date"].diff().dt.days.iloc[1:]), [1, 1, 1])

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

        self.assertTrue(
            np.allclose(result.portfolio_df["FairPriceUSD"], np.array([100.0, 100.0, 100.0]))
        )

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

        self.assertTrue(
            np.allclose(result.portfolio_df["FairPriceUSD"], np.array([200.0, 200.0, 200.0]))
        )
        self.assertTrue(
            np.allclose(result.portfolio_df["PortfolioUSD"], np.array([300.0, 300.0, 300.0]))
        )

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

        self.assertTrue(
            np.allclose(result.portfolio_df["FairPriceUSD"], np.array([50.0, 50.0, 50.0]))
        )
        self.assertTrue(
            np.allclose(result.portfolio_df["PortfolioUSD"], np.array([75.0, 75.0, 75.0]))
        )
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

        self.assertTrue(np.allclose(result.portfolio_df["DcaBTC"], np.array([0.0, 0.0, 0.5])))
        self.assertTrue(
            np.allclose(result.portfolio_df["DcaPortfolioUSD"], np.array([0.0, 0.0, 100.0]))
        )

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

        expected_cash_flow = -(((90.0 - 59.0) * 1.0) * 0.5)
        expected_april_btc = 1.0 + (expected_cash_flow / 90.0)

        self.assertTrue(
            np.allclose(result.portfolio_df["DcaBTC"], np.array([1.0, 1.0, expected_april_btc]))
        )
        self.assertTrue(
            np.allclose(
                result.portfolio_df["DcaInvestedCapitalUSD"],
                np.array([0.0, 0.0, expected_cash_flow]),
            )
        )
        self.assertTrue(
            np.allclose(
                result.portfolio_df["DcaPortfolioUSD"],
                np.array([31.0, 59.0, 90.0 + expected_cash_flow]),
            )
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

        self.assertTrue(np.allclose(result.portfolio_df["DcaBTC"], np.array([1.0, 1.0, 1.0])))
        self.assertTrue(
            np.allclose(result.portfolio_df["DcaInvestedCapitalUSD"], np.array([0.0, 0.0, 0.0]))
        )

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

        expected_cash_flow = -(90.0 - 59.0)
        expected_april_btc = 1.0 + (expected_cash_flow / 90.0)

        self.assertTrue(
            np.allclose(result.portfolio_df["DcaBTC"], np.array([1.0, 1.0, expected_april_btc]))
        )
        self.assertTrue(
            np.allclose(
                result.portfolio_df["DcaInvestedCapitalUSD"],
                np.array([0.0, 0.0, expected_cash_flow]),
            )
        )

    def test_build_portfolio_projection_keeps_capital_flat_when_selling_full_monthly_growth(self):
        settings = PortfolioSettings(
            btc_amount=1.0,
            monthly_buy_amount=0.0,
            monthly_mom_change_pct=100.0,
            forecast_unit="Month",
            forecast_horizon=4,
        )

        result = build_portfolio_projection(
            df_index=pd.to_datetime(["2026-03-15"]),
            current_gen_date=pd.Timestamp("2026-01-01"),
            intercept_a=0.0,
            slope_b=1.0,
            settings=settings,
            anchor_day=pd.Timestamp("2026-03-15"),
        )

        self.assertTrue(
            np.allclose(
                result.portfolio_df["DcaPortfolioUSD"].to_numpy(dtype=float)[2:],
                np.array([59.0, 59.0, 59.0]),
                atol=1e-9,
            )
        )

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
        )

        view_model = build_portfolio_view_model(
            projection_result,
            monthly_buy_amount=100.0,
            currency_unit="USD",
        )

        self.assertEqual(len(view_model.portfolio_display_df), 2)
        self.assertTrue(view_model.dca_enabled)
        self.assertEqual(view_model.baseline_value, 200.0)
        self.assertEqual(view_model.last_value, 240.0)
        self.assertEqual(view_model.last_dca_value, 480.0)
        self.assertEqual(view_model.last_dca_invested_capital, 200.0)
        self.assertEqual(view_model.total_growth_pct, 20.0)
        self.assertIn("Portfolio + monthly cash flow (USD)", view_model.table_df.columns)
        self.assertIn("Monthly withdrawal (USD)", view_model.table_df.columns)
        self.assertIn("Net cash flow (USD)", view_model.table_df.columns)
        self.assertIn("BTC after monthly cash flow", view_model.table_df.columns)
        self.assertTrue(
            np.allclose(view_model.table_df["Monthly withdrawal (USD)"], np.array([0.0, 0.0]))
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
        )

        view_model = build_portfolio_view_model(
            projection_result,
            monthly_buy_amount=0.0,
            monthly_mom_change_pct=60.0,
            currency_unit="USD",
        )

        self.assertTrue(view_model.dca_enabled)
        self.assertIn("Portfolio + monthly cash flow (USD)", view_model.table_df.columns)

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
                    "Date": pd.to_datetime(["2026-01-01", "2026-02-01"]),
                    "FairPriceUSD": [100.0, 120.0],
                    "PortfolioUSD": [0.0, 120.0],
                    "DcaBTC": [0.0, 1.0],
                    "DcaPortfolioUSD": [0.0, 120.0],
                    "DcaInvestedCapitalUSD": [0.0, 100.0],
                    "MoM_USD": [np.nan, 120.0],
                    "MoM_pct": [np.nan, np.nan],
                }
            ),
            table_title="Monthly growth table",
            forecast_unit="Month",
            change_usd_col="MoM_USD",
            change_pct_col="MoM_pct",
        )

        view_model = build_portfolio_view_model(
            projection_result,
            monthly_buy_amount=0.0,
            currency_unit="USD",
        )

        self.assertFalse(view_model.dca_enabled)
        self.assertEqual(view_model.total_growth_pct, 0.0)
        self.assertNotIn("Portfolio + monthly cash flow (USD)", view_model.table_df.columns)

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


if __name__ == "__main__":
    unittest.main()
