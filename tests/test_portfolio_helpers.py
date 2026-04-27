import unittest

import numpy as np
import pandas as pd

from core.utils import (
    PortfolioProjectionResult,
    PortfolioSettings,
    build_portfolio_projection,
    build_portfolio_view_model,
    get_growth_change_labels,
    normalize_periodic_growth_rate,
    resolve_projection_anchor_day,
)


class TestPortfolioHelpers(unittest.TestCase):
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
        self.assertIn("Portfolio + monthly buys (USD)", view_model.table_df.columns)
        self.assertIn("Invested cash (USD)", view_model.table_df.columns)
        self.assertIn("BTC after monthly buys", view_model.table_df.columns)

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
        self.assertNotIn("Portfolio + monthly buys (USD)", view_model.table_df.columns)

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
