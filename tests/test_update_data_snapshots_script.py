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

    @patch("scripts.update_data_snapshots.load_bitcoin_visuals_daily_data")
    def test_bitcoin_visuals_snapshot_job_uses_auto_source(self, mock_load_daily_data):
        mock_load_daily_data.return_value = pd.DataFrame(
            {"day": ["2024-01-01"], "nodes_with_channels": [1000]},
        )

        result = update_data_snapshots._build_snapshot_jobs()["bitcoin_visuals_daily_data"]()

        mock_load_daily_data.assert_called_once_with(source="auto")
        self.assertEqual(str(result.iloc[0]["day"]), "2024-01-01")

    @patch("scripts.update_data_snapshots.load_prepared_lightning_nodes_data")
    def test_lightning_nodes_snapshot_job_uses_auto_source(self, mock_load_lightning_nodes):
        mock_load_lightning_nodes.return_value = pd.DataFrame(
            {"Close": [1000.0]},
            index=pd.to_datetime(["2024-01-01"]),
        )

        result = update_data_snapshots._build_snapshot_jobs()["prepared_lightning_nodes_data"]()

        mock_load_lightning_nodes.assert_called_once_with(source="auto")
        self.assertEqual(float(result.iloc[0]["Close"]), 1000.0)

    @patch("scripts.update_data_snapshots.load_prepared_lightning_capacity_data")
    def test_lightning_capacity_snapshot_job_uses_auto_source(self, mock_load_lightning_capacity):
        mock_load_lightning_capacity.return_value = pd.DataFrame(
            {"Close": [5.0]},
            index=pd.to_datetime(["2024-01-01"]),
        )

        result = update_data_snapshots._build_snapshot_jobs()["prepared_lightning_capacity_data"]()

        mock_load_lightning_capacity.assert_called_once_with(source="auto")
        self.assertEqual(float(result.iloc[0]["Close"]), 5.0)

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

    @patch("scripts.update_data_snapshots.write_snapshot_refresh_metadata")
    @patch("scripts.update_data_snapshots.write_snapshot_dataframe")
    def test_update_snapshots_writes_refresh_metadata(
        self,
        mock_write_snapshot_dataframe,
        mock_write_snapshot_refresh_metadata,
    ):
        with patch.object(
            update_data_snapshots,
            "_build_snapshot_jobs",
            return_value={
                "prepared_price_data": lambda: pd.DataFrame(
                    {"Close": [1.0]},
                    index=pd.to_datetime(["2024-01-01"]),
                )
            },
        ):
            update_data_snapshots.update_snapshots(["prepared_price_data"])

        mock_write_snapshot_dataframe.assert_called_once()
        mock_write_snapshot_refresh_metadata.assert_called_once_with()

    @patch("scripts.update_data_snapshots.write_snapshot_refresh_metadata")
    @patch("scripts.update_data_snapshots.write_snapshot_dataframe")
    def test_update_snapshots_dry_run_skips_refresh_metadata(
        self,
        mock_write_snapshot_dataframe,
        mock_write_snapshot_refresh_metadata,
    ):
        with patch.object(
            update_data_snapshots,
            "_build_snapshot_jobs",
            return_value={
                "prepared_price_data": lambda: pd.DataFrame(
                    {"Close": [1.0]},
                    index=pd.to_datetime(["2024-01-01"]),
                )
            },
        ):
            update_data_snapshots.update_snapshots(["prepared_price_data"], dry_run=True)

        mock_write_snapshot_dataframe.assert_not_called()
        mock_write_snapshot_refresh_metadata.assert_not_called()


if __name__ == "__main__":
    unittest.main()
