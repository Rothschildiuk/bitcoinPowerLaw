import unittest
from unittest.mock import patch

import pandas as pd

from services import price_service


class TestPriceService(unittest.TestCase):
    def test_extract_close_series_handles_empty_input(self):
        series = price_service._extract_close_series(pd.DataFrame())
        self.assertTrue(series.empty)

    def test_extract_close_series_uses_first_close_column_for_multiindex_like_shape(self):
        index = pd.to_datetime(["2024-01-01", "2024-01-02"])
        download_df = pd.DataFrame(
            {
                ("Close", "BTC-USD"): [10.0, 20.0],
                ("Open", "BTC-USD"): [9.0, 19.0],
            },
            index=index,
        )
        series = price_service._extract_close_series(download_df)
        self.assertListEqual(series.tolist(), [10.0, 20.0])

    @patch("services.price_service.load_reference_series")
    def test_build_currency_close_series_for_eur(self, mock_load_reference_series):
        idx = pd.to_datetime(["2024-01-01", "2024-01-02"])
        raw_df = pd.DataFrame({"Close": [100.0, 120.0]}, index=idx)
        eur_usd = pd.Series([2.0, 2.0], index=idx)
        xau_usd = pd.Series(dtype=float)
        mock_load_reference_series.return_value = (eur_usd, xau_usd)

        result = price_service.build_currency_close_series(raw_df, "EUR")
        self.assertListEqual(result.round(6).tolist(), [50.0, 60.0])

    @patch("services.price_service._safe_download_close_series")
    @patch("services.price_service.pd.read_csv")
    def test_load_prepared_price_data_appends_tail_when_history_is_stale(
        self, mock_read_csv, mock_safe_download
    ):
        price_service.load_prepared_price_data.clear()
        csv_df = pd.DataFrame(
            {
                "Date": ["2020-01-01", "2020-01-02"],
                "Price": [7000.0, 7100.0],
            }
        )
        mock_read_csv.return_value = csv_df

        tail_index = pd.to_datetime(["2020-01-03"])
        mock_safe_download.return_value = pd.Series([7200.0], index=tail_index)

        result = price_service.load_prepared_price_data(
            price_history_url="unused.csv",
            stale_after_days=0,
        )

        self.assertIn(pd.Timestamp("2020-01-03"), result.index)
        self.assertEqual(float(result.loc[pd.Timestamp("2020-01-03"), "Close"]), 7200.0)
        self.assertIn("AbsDays", result.columns)
        self.assertIn("LogClose", result.columns)


if __name__ == "__main__":
    unittest.main()
