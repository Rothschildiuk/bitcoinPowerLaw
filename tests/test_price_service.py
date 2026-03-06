import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from services import price_service


class _FakeHttpResponse:
    def __init__(self, payload_bytes):
        self._payload_bytes = payload_bytes

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self._payload_bytes


class TestPriceService(unittest.TestCase):
    def setUp(self):
        self._temp_dir = TemporaryDirectory()
        self._cache_dir = Path(self._temp_dir.name)
        self._cache_patch = patch.object(price_service, "LOCAL_DATA_CACHE_DIR", self._cache_dir)
        self._cache_patch.start()

    def tearDown(self):
        self._cache_patch.stop()
        self._temp_dir.cleanup()

    def test_load_or_refresh_dataframe_cache_uses_fresh_local_snapshot(self):
        cached_df = pd.DataFrame(
            {"Close": [100.0]},
            index=pd.to_datetime(["2024-01-01"]),
        )

        price_service._write_cached_dataframe("sample", cached_df)

        result = price_service._load_or_refresh_dataframe_cache(
            "sample",
            lambda: self.fail("fetch_fn should not run when cache is fresh"),
            min_check_interval_seconds=3600,
        )

        self.assertListEqual(result["Close"].tolist(), [100.0])

    def test_load_or_refresh_dataframe_cache_returns_stale_snapshot_on_refresh_error(self):
        cached_df = pd.DataFrame(
            {"Close": [200.0]},
            index=pd.to_datetime(["2024-01-01"]),
        )

        price_service._write_cached_dataframe("sample", cached_df)
        stale_meta = self._cache_dir / "sample.meta.json"
        stale_meta.write_text(
            f'{{"version":{price_service.LOCAL_CACHE_SCHEMA_VERSION},"checked_at":"2020-01-01T00:00:00"}}',
            encoding="utf-8",
        )

        result = price_service._load_or_refresh_dataframe_cache(
            "sample",
            lambda: (_ for _ in ()).throw(RuntimeError("network down")),
            min_check_interval_seconds=1,
        )

        self.assertListEqual(result["Close"].tolist(), [200.0])

    def test_load_or_refresh_dataframe_cache_refreshes_outdated_schema_cache(self):
        stale_df = pd.DataFrame(
            {"Close": [1.0, 2.0]},
            index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
        )
        fresh_df = pd.DataFrame(
            {"Close": [10.0, 20.0, 30.0]},
            index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        )

        price_service._write_cached_dataframe("sample", stale_df)
        stale_meta = self._cache_dir / "sample.meta.json"
        stale_meta.write_text(
            '{"version":1,"checked_at":"2026-01-01T00:00:00"}',
            encoding="utf-8",
        )

        result = price_service._load_or_refresh_dataframe_cache(
            "sample",
            lambda: fresh_df,
            min_check_interval_seconds=3600,
            validator_fn=lambda df: len(df) >= 3,
        )

        self.assertListEqual(result["Close"].tolist(), [10.0, 20.0, 30.0])

    def test_write_and_read_cached_dataframe_preserves_range_index_frames(self):
        raw_df = pd.DataFrame(
            {
                "day": ["2024-01-01", "2024-01-02"],
                "nodes_with_channels": [1000, 1010],
            }
        )

        price_service._write_cached_dataframe("raw_sample", raw_df)
        restored_df = price_service._read_cached_dataframe("raw_sample")

        self.assertListEqual(restored_df.columns.tolist(), raw_df.columns.tolist())
        self.assertIsInstance(restored_df.index, pd.RangeIndex)
        self.assertListEqual(restored_df["day"].tolist(), ["2024-01-01", "2024-01-02"])

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

    @patch("services.price_service.time.sleep")
    @patch("services.price_service.urllib.request.urlopen")
    def test_fetch_json_with_retry_retries_before_success(self, mock_urlopen, mock_sleep):
        success_response = _FakeHttpResponse(b'{"ok": true}')
        mock_urlopen.side_effect = [Exception("boom-1"), Exception("boom-2"), success_response]

        payload = price_service._fetch_json_with_retry(
            "https://example.test/data",
            retries=3,
            initial_backoff_seconds=0.1,
        )

        self.assertEqual(payload, {"ok": True})
        self.assertEqual(mock_urlopen.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)

    @patch("services.price_service._safe_download_close_series")
    @patch("services.price_service._safe_download_btc_tail_from_coingecko")
    @patch("services.price_service.pd.read_csv")
    def test_load_prepared_price_data_appends_tail_when_history_is_stale(
        self, mock_read_csv, mock_cg_tail, mock_safe_download
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
        mock_cg_tail.return_value = pd.Series(dtype=float)

        result = price_service.load_prepared_price_data(
            price_history_url="unused.csv",
            stale_after_days=0,
        )

        self.assertIn(pd.Timestamp("2020-01-03"), result.index)
        self.assertEqual(float(result.loc[pd.Timestamp("2020-01-03"), "Close"]), 7200.0)
        self.assertIn("AbsDays", result.columns)
        self.assertIn("LogClose", result.columns)

    @patch("services.price_service._safe_download_close_series")
    @patch("services.price_service._safe_download_btc_tail_from_coingecko")
    @patch("services.price_service._safe_download_btc_tail_from_coincap")
    @patch("services.price_service.pd.read_csv")
    def test_load_prepared_price_data_uses_coingecko_fallback_when_yfinance_empty(
        self, mock_read_csv, mock_cc_tail, mock_cg_tail, mock_safe_download
    ):
        price_service.load_prepared_price_data.clear()
        csv_df = pd.DataFrame(
            {
                "Date": ["2020-01-01", "2020-01-02"],
                "Price": [7000.0, 7100.0],
            }
        )
        mock_read_csv.return_value = csv_df
        mock_safe_download.return_value = pd.Series(dtype=float)
        mock_cg_tail.return_value = pd.Series(
            [7300.0],
            index=pd.to_datetime(["2020-01-03"]),
        )
        mock_cc_tail.return_value = pd.Series(dtype=float)

        result = price_service.load_prepared_price_data(
            price_history_url="unused.csv",
            stale_after_days=0,
        )

        self.assertIn(pd.Timestamp("2020-01-03"), result.index)
        self.assertEqual(float(result.loc[pd.Timestamp("2020-01-03"), "Close"]), 7300.0)

    @patch("services.price_service._safe_download_close_series")
    @patch("services.price_service._safe_download_btc_tail_from_coingecko")
    @patch("services.price_service._safe_download_btc_tail_from_coincap")
    @patch("services.price_service.pd.read_csv")
    def test_load_prepared_price_data_uses_coincap_fallback_when_others_empty(
        self, mock_read_csv, mock_cc_tail, mock_cg_tail, mock_safe_download
    ):
        price_service.load_prepared_price_data.clear()
        csv_df = pd.DataFrame(
            {
                "Date": ["2020-01-01", "2020-01-02"],
                "Price": [7000.0, 7100.0],
            }
        )
        mock_read_csv.return_value = csv_df
        mock_safe_download.return_value = pd.Series(dtype=float)
        mock_cg_tail.return_value = pd.Series(dtype=float)
        mock_cc_tail.return_value = pd.Series(
            [7350.0],
            index=pd.to_datetime(["2020-01-03"]),
        )

        result = price_service.load_prepared_price_data(
            price_history_url="unused.csv",
            stale_after_days=0,
        )

        self.assertIn(pd.Timestamp("2020-01-03"), result.index)
        self.assertEqual(float(result.loc[pd.Timestamp("2020-01-03"), "Close"]), 7350.0)

    @patch("services.price_service.pd.read_csv")
    def test_load_prepared_miner_revenue_data_parses_timestamp_value(self, mock_read_csv):
        price_service.load_prepared_miner_revenue_data.clear()
        mock_read_csv.return_value = pd.DataFrame(
            {
                "Timestamp": ["2020-01-01", "2020-01-02"],
                "Value": [11_000_000.0, 12_500_000.0],
            }
        )

        result = price_service.load_prepared_miner_revenue_data("unused.csv")

        self.assertListEqual(
            result.index.strftime("%Y-%m-%d").tolist(),
            ["2020-01-01", "2020-01-02"],
        )
        self.assertListEqual(result["Close"].tolist(), [11_000_000.0, 12_500_000.0])
        self.assertIn("AbsDays", result.columns)
        self.assertIn("LogClose", result.columns)

    @patch("services.price_service.pd.read_csv")
    def test_load_prepared_difficulty_data_parses_timestamp_value(self, mock_read_csv):
        price_service.load_prepared_difficulty_data.clear()
        mock_read_csv.return_value = pd.DataFrame(
            {
                "Timestamp": ["2009-12-31", "2010-01-01", "2020-01-02"],
                "Value": [9.9e12, 1.0e13, 1.3e13],
            }
        )

        result = price_service.load_prepared_difficulty_data("unused.csv")
        self.assertListEqual(
            result.index.strftime("%Y-%m-%d").tolist(),
            ["2010-01-01", "2020-01-02"],
        )
        self.assertListEqual(result["Close"].tolist(), [1.0e13, 1.3e13])

    @patch("services.price_service.pd.read_csv")
    def test_load_prepared_hashrate_data_parses_timestamp_value(self, mock_read_csv):
        price_service.load_prepared_hashrate_data.clear()
        mock_read_csv.return_value = pd.DataFrame(
            {
                "Timestamp": ["2009-12-31", "2010-01-01", "2020-01-02"],
                "Value": [90_000_000.0, 100_000_000.0, 120_000_000.0],
            }
        )

        result = price_service.load_prepared_hashrate_data("unused.csv")
        self.assertListEqual(
            result.index.strftime("%Y-%m-%d").tolist(),
            ["2010-01-01", "2020-01-02"],
        )
        self.assertListEqual(result["Close"].tolist(), [100_000_000.0, 120_000_000.0])

    @patch("services.price_service.load_bitcoin_visuals_daily_data")
    def test_load_prepared_lightning_nodes_data_parses_day_and_nodes(self, mock_load_daily):
        price_service.load_prepared_lightning_nodes_data.clear()
        mock_load_daily.return_value = pd.DataFrame(
            {
                "day": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "nodes_with_channels": [11_000, "", 12_500],
            }
        )

        result = price_service.load_prepared_lightning_nodes_data("unused.csv")
        self.assertListEqual(
            result.index.strftime("%Y-%m-%d").tolist(),
            ["2024-01-01", "2024-01-03"],
        )
        self.assertListEqual(result["Close"].tolist(), [11_000.0, 12_500.0])

    @patch("services.price_service.load_bitcoin_visuals_daily_data")
    def test_load_prepared_lightning_capacity_data_parses_day_and_capacity(self, mock_load_daily):
        price_service.load_prepared_lightning_capacity_data.clear()
        mock_load_daily.return_value = pd.DataFrame(
            {
                "day": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "capacity_total": [4_500.5, 0.0, 4_700.25],
            }
        )

        result = price_service.load_prepared_lightning_capacity_data("unused.csv")
        self.assertListEqual(
            result.index.strftime("%Y-%m-%d").tolist(),
            ["2024-01-01", "2024-01-03"],
        )
        self.assertListEqual(result["Close"].tolist(), [4_500.5, 4_700.25])

    @patch("services.price_service._fetch_json_with_retry")
    def test_load_prepared_liquid_btc_data_parses_official_reserves_api(self, mock_fetch_json):
        price_service.load_prepared_liquid_btc_data.clear()
        mock_fetch_json.side_effect = [
            [
                {"date": "2024-01-01", "amount": "100000000"},
                {"date": "2024-02-01", "amount": "200000000"},
            ],
            {"amount": "250000000", "lastBlockUpdate": 1},
        ]

        result = price_service.load_prepared_liquid_btc_data(
            liquid_reserves_month_url="unused-month",
            liquid_reserves_url="unused-current",
        )

        self.assertFalse(result.empty)
        self.assertAlmostEqual(float(result.iloc[0]["Close"]), 1.0)
        self.assertAlmostEqual(float(result.iloc[-1]["Close"]), 2.5)
        self.assertIn("AbsDays", result.columns)
        self.assertIn("LogClose", result.columns)

    @patch("services.price_service._fetch_json_with_retry")
    def test_load_prepared_liquid_btc_data_raises_for_empty_month_payload(self, mock_fetch_json):
        price_service.load_prepared_liquid_btc_data.clear()
        mock_fetch_json.return_value = []

        with self.assertRaises(ValueError):
            price_service.load_prepared_liquid_btc_data(
                liquid_reserves_month_url="unused-month",
                liquid_reserves_url="unused-current",
            )

    @patch("services.price_service._fetch_json_with_retry")
    def test_load_prepared_liquid_btc_data_raises_for_missing_required_columns(
        self, mock_fetch_json
    ):
        price_service.load_prepared_liquid_btc_data.clear()
        mock_fetch_json.return_value = [{"date": "2024-01-01", "value": "100000000"}]

        with self.assertRaises(ValueError):
            price_service.load_prepared_liquid_btc_data(
                liquid_reserves_month_url="unused-month",
                liquid_reserves_url="unused-current",
            )

    @patch("services.price_service._fetch_json_with_retry")
    def test_load_prepared_liquid_btc_data_prefers_current_amount_for_today_duplicate(
        self, mock_fetch_json
    ):
        price_service.load_prepared_liquid_btc_data.clear()
        today = pd.Timestamp.utcnow().tz_localize(None).normalize().strftime("%Y-%m-%d")
        mock_fetch_json.side_effect = [
            [
                {"date": "2024-01-01", "amount": "100000000"},
                {"date": today, "amount": "200000000"},
            ],
            {"amount": "333000000", "lastBlockUpdate": 1},
        ]

        result = price_service.load_prepared_liquid_btc_data(
            liquid_reserves_month_url="unused-month",
            liquid_reserves_url="unused-current",
        )
        self.assertAlmostEqual(float(result.iloc[-1]["Close"]), 3.33)

    @patch("services.price_service._fetch_json_with_retry")
    def test_load_prepared_liquid_btc_data_returns_monotonic_daily_index_without_nans(
        self, mock_fetch_json
    ):
        price_service.load_prepared_liquid_btc_data.clear()
        mock_fetch_json.side_effect = [
            [
                {"date": "2024-01-01", "amount": "100000000"},
                {"date": "2024-03-01", "amount": "300000000"},
                {"date": "bad-date", "amount": "not-a-number"},
            ],
            {"amount": "350000000", "lastBlockUpdate": 1},
        ]

        result = price_service.load_prepared_liquid_btc_data(
            liquid_reserves_month_url="unused-month",
            liquid_reserves_url="unused-current",
        )

        self.assertTrue(result.index.is_monotonic_increasing)
        self.assertFalse(result[["Close", "AbsDays", "LogClose"]].isna().any().any())
        self.assertTrue((result["Close"] > 0).all())

    @patch("services.price_service._fetch_json_with_retry")
    def test_load_prepared_liquid_transactions_data_parses_weekly_transactions(
        self, mock_fetch_json
    ):
        price_service.load_prepared_liquid_transactions_data.clear()
        mock_fetch_json.return_value = {
            "data": {
                "Weekly Transactions": [
                    {
                        "Start Time": "2024-01-01 00:00:00",
                        "End Time": "2024-01-07 23:59:59",
                        "TX Total": 1200,
                    },
                    {
                        "Start Time": "2024-01-08 00:00:00",
                        "End Time": "2024-01-14 23:59:59",
                        "TX Total": 1350,
                    },
                ]
            }
        }

        result = price_service.load_prepared_liquid_transactions_data("unused-url")

        self.assertListEqual(
            result.index.strftime("%Y-%m-%d").tolist(),
            ["2024-01-01", "2024-01-08"],
        )
        self.assertListEqual(result["Close"].tolist(), [1200.0, 1350.0])
        self.assertIn("AbsDays", result.columns)
        self.assertIn("LogClose", result.columns)

    @patch("services.price_service._fetch_json_with_retry")
    def test_load_prepared_liquid_transactions_data_raises_for_missing_data(self, mock_fetch_json):
        price_service.load_prepared_liquid_transactions_data.clear()
        mock_fetch_json.return_value = {"data": {"Weekly Transactions": []}}

        with self.assertRaises(ValueError):
            price_service.load_prepared_liquid_transactions_data("unused-url")


if __name__ == "__main__":
    unittest.main()
