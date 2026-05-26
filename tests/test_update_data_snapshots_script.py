import unittest
from unittest.mock import patch

import pandas as pd

from scripts import update_data_snapshots


class TestUpdateDataSnapshotsScript(unittest.TestCase):
    @patch("scripts.update_data_snapshots.load_prepared_price_data")
    def test_prepared_price_snapshot_job_uses_auto_source(self, mock_load_prepared_price_data):
        mock_load_prepared_price_data.return_value = pd.DataFrame(
            {"Close": [1.0]},
            index=pd.to_datetime(["2024-01-01"]),
        )

        result = update_data_snapshots._build_snapshot_jobs()["prepared_price_data"]()

        mock_load_prepared_price_data.assert_called_once_with(source="auto")
        self.assertEqual(float(result.iloc[0]["Close"]), 1.0)

    @patch("scripts.update_data_snapshots.load_prepared_bitcoin_supply_data")
    def test_bitcoin_supply_snapshot_job_uses_live_source(self, mock_load_prepared_supply):
        mock_load_prepared_supply.return_value = pd.DataFrame(
            {"Close": [19_000_000.0]},
            index=pd.to_datetime(["2024-01-01"]),
        )

        result = update_data_snapshots._build_snapshot_jobs()["prepared_bitcoin_supply_data"]()

        mock_load_prepared_supply.assert_called_once_with(source="live")
        self.assertEqual(float(result.iloc[0]["Close"]), 19_000_000.0)

    @patch("scripts.update_data_snapshots.build_incremental_reference_series_snapshot")
    def test_reference_snapshot_job_uses_incremental_builder(
        self,
        mock_build_incremental_reference_series_snapshot,
    ):
        mock_build_incremental_reference_series_snapshot.return_value = pd.DataFrame(
            {"EURUSD": [1.0]},
            index=pd.to_datetime(["2024-01-01"]),
        )

        result = update_data_snapshots._build_snapshot_jobs()["reference_series"]()

        mock_build_incremental_reference_series_snapshot.assert_called_once_with()
        self.assertEqual(float(result.iloc[0]["EURUSD"]), 1.0)


if __name__ == "__main__":
    unittest.main()
