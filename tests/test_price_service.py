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
        self._snapshot_dir = self._cache_dir / "snapshots"
        self._cache_patch = patch.object(price_service, "LOCAL_DATA_CACHE_DIR", self._cache_dir)
        self._snapshot_patch = patch.object(price_service, "SNAPSHOT_DATA_DIR", self._snapshot_dir)
        self._cache_patch.start()
        self._snapshot_patch.start()

    def tearDown(self):
        self._cache_patch.stop()
        self._snapshot_patch.stop()
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

    def test_write_and_read_snapshot_dataframe_preserves_date_index(self):
        snapshot_df = pd.DataFrame(
            {"Close": [100.0, 101.0]},
            index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
        )

        price_service.write_snapshot_dataframe("prepared_price_data", snapshot_df)
        restored_df = price_service._read_snapshot_dataframe("prepared_price_data")

        self.assertListEqual(restored_df["Close"].tolist(), [100.0, 101.0])
        self.assertListEqual(
            restored_df.index.strftime("%Y-%m-%d").tolist(),
            ["2024-01-01", "2024-01-02"],
        )

    def test_extract_close_series_handles_empty_input(self):
        series = price_service._extract_close_series(pd.DataFrame())
        self.assertTrue(series.empty)

    def test_load_snapshot_or_live_prefers_snapshot_over_live_fetch(self):
        snapshot_df = pd.DataFrame(
            {"Close": [42.0], "AbsDays": [1], "LogClose": [1.623249]},
            index=pd.to_datetime(["2009-01-04"]),
        )
        price_service.write_snapshot_dataframe("prepared_price_data", snapshot_df)

        result = price_service._load_snapshot_or_live(
            "prepared_price_data",
            lambda data_df: True,
            lambda: self.fail("live loader should not run when snapshot is valid"),
        )

        self.assertEqual(float(result.iloc[0]["Close"]), 42.0)

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

    @patch("services.price_service.pd.read_csv")
    def test_build_normalized_csv_adapter_applies_postprocess_and_cache_contract(
        self, mock_read_csv
    ):
        mock_read_csv.return_value = pd.DataFrame(
            {
                "Timestamp": ["2010-01-01", "2010-01-02", "2010-01-03"],
                "Value": [1.0, 2.0, 3.0],
            }
        )

        adapter = price_service._build_normalized_csv_adapter(
            "sample_chart",
            "unused.csv",
            refresh_seconds=3600,
            validator_fn=price_service._validate_prepared_chart_data,
            postprocess_fn=lambda data_df: data_df.iloc[1:],
        )
        result = price_service._load_source_adapter(adapter)

        self.assertEqual(adapter.cache_key, "sample_chart")
        self.assertEqual(adapter.refresh_seconds, 3600)
        self.assertListEqual(result["Close"].tolist(), [2.0, 3.0])

    @patch("services.price_service.pd.read_csv")
    def test_build_blockchain_chart_adapter_reads_headerless_csv(self, mock_read_csv):
        mock_read_csv.return_value = pd.DataFrame(
            {
                "Timestamp": ["2009-01-03", "2009-01-04"],
                "Value": [1.0, 2.0],
            }
        )

        adapter = price_service._build_blockchain_chart_adapter(
            "difficulty_sample",
            "unused.csv",
        )
        result = adapter.fetch_fn()

        _, kwargs = mock_read_csv.call_args
        self.assertIsNone(kwargs["header"])
        self.assertEqual(kwargs["names"], ["Timestamp", "Value"])
        self.assertListEqual(
            result.index.strftime("%Y-%m-%d").tolist(),
            ["2009-01-03", "2009-01-04"],
        )

    @patch("services.price_service._fetch_json_with_retry")
    def test_safe_download_cryptocompare_histoday_parses_daily_close_series(self, mock_fetch_json):
        mock_fetch_json.return_value = {
            "Response": "Success",
            "Data": {
                "Data": [
                    {"time": 1704067200, "close": 0.5},
                    {"time": 1704153600, "close": 0.6},
                ]
            },
        }

        result = price_service._safe_download_cryptocompare_histoday("XMR", "USD")

        self.assertListEqual(result.index.strftime("%Y-%m-%d").tolist(), ["2024-01-01", "2024-01-02"])
        self.assertListEqual(result.tolist(), [0.5, 0.6])

    @patch("services.price_service._fetch_text_with_retry")
    def test_safe_download_coinlore_monero_usd_parses_daily_close_series(self, mock_fetch_text):
        mock_fetch_text.return_value = """
        <table id="ohlc"><tbody>
        <tr class="txtr"><td class="nwt txtl font-bold"> May 22, 2014</td><td class="nwt">$1.59</td><td class="nwt">$2.19</td><td class="nwt">$1.36</td><td class="nwt">$2.10</td></tr>
        <tr class="txtr"><td class="nwt txtl font-bold"> May 21, 2014</td><td class="nwt">$2.47</td><td class="nwt">$2.65</td><td class="nwt">$1.23</td><td class="nwt">$1.60</td></tr>
        </tbody></table>
        """

        result = price_service._safe_download_coinlore_monero_usd()

        self.assertListEqual(result.index.strftime("%Y-%m-%d").tolist(), ["2014-05-21", "2014-05-22"])
        self.assertListEqual(result.tolist(), [1.6, 2.1])

    @patch("services.price_service.load_reference_series")
    def test_build_currency_close_series_for_eur(self, mock_load_reference_series):
        idx = pd.to_datetime(["2024-01-01", "2024-01-02"])
        raw_df = pd.DataFrame({"Close": [100.0, 120.0]}, index=idx)
        eur_usd = pd.Series([2.0, 2.0], index=idx)
        xau_usd = pd.Series(dtype=float)
        mock_load_reference_series.return_value = (eur_usd, xau_usd)

        result = price_service.build_currency_close_series(raw_df, "EUR")
        self.assertListEqual(result.round(6).tolist(), [50.0, 60.0])

    @patch("services.price_service._safe_download_cryptocompare_histoday")
    @patch("services.price_service._safe_download_crypto_btc_via_usd")
    def test_load_prepared_filecoin_btc_data_normalizes_fil_btc_history(
        self, mock_cross_download, mock_direct_download
    ):
        price_service.load_prepared_filecoin_btc_data.clear()
        idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
        mock_direct_download.return_value = pd.Series([0.001, 0.0009, 0.0008], index=idx)

        result = price_service.load_prepared_filecoin_btc_data("2020-01-01")

        self.assertEqual(mock_direct_download.call_args[0], ("FIL", "BTC"))
        mock_cross_download.assert_not_called()
        self.assertListEqual(result["Close"].round(7).tolist(), [0.001, 0.0009, 0.0008])
        self.assertTrue((result["AbsDays"] > 0).all())
        self.assertTrue((result["LogClose"] < 0).all())

    @patch("services.price_service._safe_download_cryptocompare_histoday")
    @patch("services.price_service._safe_download_crypto_btc_via_usd")
    @patch("services.price_service._safe_download_monero_btc_via_coinlore")
    def test_load_prepared_monero_btc_data_falls_back_to_coinlore(
        self,
        mock_coinlore_download,
        mock_cross_download,
        mock_direct_download,
    ):
        price_service.load_prepared_monero_btc_data.clear()
        mock_direct_download.return_value = pd.Series(dtype=float)
        mock_cross_download.return_value = pd.Series(dtype=float)
        idx = pd.to_datetime(["2014-05-21", "2014-05-22", "2014-05-23"])
        mock_coinlore_download.return_value = pd.Series([0.0065, 0.0071, 0.0082], index=idx)

        result = price_service.load_prepared_monero_btc_data("2014-01-01", source="live")

        self.assertEqual(mock_direct_download.call_args[0], ("XMR", "BTC"))
        self.assertEqual(mock_cross_download.call_args[0], ("XMR", "2014-01-01"))
        self.assertEqual(mock_coinlore_download.call_args[0], ("2014-01-01",))
        self.assertListEqual(
            result.index.strftime("%Y-%m-%d").tolist(),
            ["2014-05-21", "2014-05-22", "2014-05-23"],
        )
        self.assertListEqual(result["Close"].round(4).tolist(), [0.0065, 0.0071, 0.0082])

    @patch("services.price_service._safe_download_cryptocompare_histoday")
    def test_safe_download_crypto_btc_via_usd_derives_ratio_from_usd_pairs(self, mock_download):
        def fake_download(fsym, tsym):
            if (fsym, tsym) == ("DOGE", "USD"):
                return pd.Series(
                    [0.10, 0.20],
                    index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
                )
            if (fsym, tsym) == ("BTC", "USD"):
                return pd.Series(
                    [50_000.0, 40_000.0],
                    index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
                )
            self.fail(f"Unexpected pair: {(fsym, tsym)}")

        mock_download.side_effect = fake_download

        result = price_service._safe_download_crypto_btc_via_usd("DOGE", "2024-01-01")

        self.assertListEqual(result.index.strftime("%Y-%m-%d").tolist(), ["2024-01-01", "2024-01-02"])
        self.assertListEqual(result.round(8).tolist(), [0.000002, 0.000005])

    @patch("services.price_service._safe_download_cryptocompare_histoday")
    @patch("services.price_service._safe_download_crypto_btc_via_usd")
    def test_load_prepared_dogecoin_btc_data_uses_usd_cross_source(
        self, mock_cross_download, mock_direct_download
    ):
        price_service.load_prepared_dogecoin_btc_data.clear()
        mock_direct_download.return_value = pd.Series(dtype=float)
        idx = pd.to_datetime(["2014-02-01", "2014-02-02", "2014-02-03"])
        mock_cross_download.return_value = pd.Series(
            [0.00000018, 0.00000019, 0.00000021],
            index=idx,
        )

        result = price_service.load_prepared_dogecoin_btc_data("2014-01-01")

        self.assertEqual(mock_direct_download.call_args[0], ("DOGE", "BTC"))
        self.assertEqual(mock_cross_download.call_args[0], ("DOGE", "2014-01-01"))
        self.assertListEqual(result.index.strftime("%Y-%m-%d").tolist(), ["2014-02-01", "2014-02-02", "2014-02-03"])
        self.assertListEqual(result["Close"].round(8).tolist(), [0.00000018, 0.00000019, 0.00000021])
        self.assertTrue((result["LogClose"] < 0).all())

    @patch("services.price_service._safe_download_cryptocompare_histoday")
    @patch("services.price_service._safe_download_crypto_btc_via_usd")
    def test_load_prepared_dogecoin_btc_data_falls_back_to_usd_cross_when_direct_btc_empty(
        self, mock_cross_download, mock_direct_download
    ):
        price_service.load_prepared_dogecoin_btc_data.clear()
        mock_direct_download.return_value = pd.Series(dtype=float)
        idx = pd.to_datetime(["2014-02-01", "2014-02-02"])
        mock_cross_download.return_value = pd.Series([0.00000018, 0.00000019], index=idx)

        result = price_service.load_prepared_dogecoin_btc_data("2014-01-01")

        self.assertEqual(mock_direct_download.call_args[0], ("DOGE", "BTC"))
        self.assertEqual(mock_cross_download.call_args[0], ("DOGE", "2014-01-01"))
        self.assertListEqual(result.index.strftime("%Y-%m-%d").tolist(), ["2014-02-01", "2014-02-02"])

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
                "Timestamp": ["2009-01-03", "2020-01-01", "2020-01-02"],
                "Value": [10_000_000.0, 11_000_000.0, 12_500_000.0],
            }
        )

        result = price_service.load_prepared_miner_revenue_data("unused.csv")

        self.assertListEqual(
            result.index.strftime("%Y-%m-%d").tolist(),
            ["2009-01-03", "2020-01-01", "2020-01-02"],
        )
        self.assertListEqual(result["Close"].tolist(), [10_000_000.0, 11_000_000.0, 12_500_000.0])
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
            ["2009-12-31", "2010-01-01", "2020-01-02"],
        )
        self.assertListEqual(result["Close"].tolist(), [9.9e12, 1.0e13, 1.3e13])

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
            ["2009-12-31", "2010-01-01", "2020-01-02"],
        )
        self.assertListEqual(result["Close"].tolist(), [90_000_000.0, 100_000_000.0, 120_000_000.0])

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
