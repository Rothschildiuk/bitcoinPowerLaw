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

    @patch("scripts.update_data_snapshots.read_snapshot_dataframe", return_value=None)
    @patch("scripts.update_data_snapshots.write_snapshot_refresh_metadata")
    @patch("scripts.update_data_snapshots.write_snapshot_dataframe", return_value=True)
    def test_update_snapshots_writes_refresh_metadata(
        self,
        mock_write_snapshot_dataframe,
        mock_write_snapshot_refresh_metadata,
        _mock_read_snapshot_dataframe,
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

    @patch("scripts.update_data_snapshots.read_snapshot_dataframe", return_value=None)
    @patch("scripts.update_data_snapshots.write_snapshot_refresh_metadata")
    @patch("scripts.update_data_snapshots.write_snapshot_dataframe")
    def test_update_snapshots_dry_run_skips_refresh_metadata(
        self,
        mock_write_snapshot_dataframe,
        mock_write_snapshot_refresh_metadata,
        _mock_read_snapshot_dataframe,
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

    def _run_update(self, jobs, stored=None, written=True):
        """Run update_snapshots over fake jobs; returns (kept, write mock, metadata mock)."""
        with (
            patch.object(update_data_snapshots, "_build_snapshot_jobs", return_value=jobs),
            patch.object(
                update_data_snapshots,
                "read_snapshot_dataframe",
                side_effect=lambda key: (stored or {}).get(key),
            ),
            patch.object(
                update_data_snapshots, "write_snapshot_dataframe", return_value=written
            ) as mock_write,
            patch.object(update_data_snapshots, "write_snapshot_refresh_metadata") as mock_metadata,
        ):
            kept = update_data_snapshots.update_snapshots()
        return kept, mock_write, mock_metadata

    def test_update_snapshots_keeps_going_when_one_source_fails(self):
        def failing_job():
            raise RuntimeError("source down")

        kept, mock_write, mock_metadata = self._run_update(
            {
                "prepared_usdt_supply_data": failing_job,
                "prepared_price_data": lambda: price_frame(["2024-01-01", "2024-01-02"]),
            }
        )

        self.assertEqual(kept, ["prepared_usdt_supply_data"])
        mock_write.assert_called_once()
        self.assertEqual(mock_write.call_args.args[0], "prepared_price_data")
        mock_metadata.assert_called_once_with()

    def test_update_snapshots_keeps_stored_snapshot_with_fewer_rows(self):
        kept, mock_write, mock_metadata = self._run_update(
            {"prepared_price_data": lambda: price_frame(["2024-01-01"])},
            stored={"prepared_price_data": price_frame(["2024-01-01", "2024-01-02"])},
        )

        self.assertEqual(kept, ["prepared_price_data"])
        mock_write.assert_not_called()
        mock_metadata.assert_not_called()

    def test_update_snapshots_keeps_stored_snapshot_that_ends_later(self):
        kept, mock_write, _ = self._run_update(
            {"prepared_price_data": lambda: price_frame(["2023-12-30", "2023-12-31"])},
            stored={"prepared_price_data": price_frame(["2024-01-01", "2024-01-02"])},
        )

        self.assertEqual(kept, ["prepared_price_data"])
        mock_write.assert_not_called()

    def test_update_snapshots_compares_day_column_snapshots(self):
        kept, mock_write, _ = self._run_update(
            {
                "bitcoin_visuals_daily_data": lambda: pd.DataFrame(
                    {"day": ["2023-12-31", "2024-01-01"], "blocks_count": [1, 2]}
                )
            },
            stored={
                "bitcoin_visuals_daily_data": pd.DataFrame(
                    {"day": ["2024-01-01", "2024-01-02"], "blocks_count": [1, 2]}
                )
            },
        )

        self.assertEqual(kept, ["bitcoin_visuals_daily_data"])
        mock_write.assert_not_called()

    def test_update_snapshots_skips_refresh_metadata_when_nothing_changed(self):
        kept, mock_write, mock_metadata = self._run_update(
            {"prepared_price_data": lambda: price_frame(["2024-01-01", "2024-01-02"])},
            stored={"prepared_price_data": price_frame(["2024-01-01", "2024-01-02"])},
            written=False,
        )

        self.assertEqual(kept, [])
        mock_write.assert_called_once()
        mock_metadata.assert_not_called()


def price_frame(dates):
    return pd.DataFrame({"Close": [1.0] * len(dates)}, index=pd.to_datetime(dates))


if __name__ == "__main__":
    unittest.main()
