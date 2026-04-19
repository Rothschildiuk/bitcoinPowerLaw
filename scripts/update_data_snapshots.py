from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.price_service import (  # noqa: E402
    load_bitcoin_visuals_daily_data,
    load_prepared_difficulty_data,
    load_prepared_dogecoin_btc_data,
    load_prepared_filecoin_btc_data,
    load_prepared_hashrate_data,
    load_prepared_lightning_capacity_data,
    load_prepared_lightning_nodes_data,
    load_prepared_liquid_btc_data,
    load_prepared_liquid_transactions_data,
    load_prepared_litecoin_btc_data,
    load_prepared_miner_revenue_data,
    load_prepared_monero_btc_data,
    load_prepared_price_data,
    load_prepared_russian_m2_data,
    load_prepared_us_m2_data,
    load_reference_series,
    write_snapshot_dataframe,
)


def _build_snapshot_jobs():
    return {
        "reference_series": lambda: _build_reference_snapshot(),
        "prepared_price_data": lambda: load_prepared_price_data(source="live"),
        "prepared_miner_revenue_data": lambda: load_prepared_miner_revenue_data(source="live"),
        "prepared_difficulty_data": lambda: load_prepared_difficulty_data(source="live"),
        "prepared_hashrate_data": lambda: load_prepared_hashrate_data(source="live"),
        "bitcoin_visuals_daily_data": lambda: load_bitcoin_visuals_daily_data(source="live"),
        "prepared_lightning_nodes_data": lambda: load_prepared_lightning_nodes_data(source="live"),
        "prepared_lightning_capacity_data": lambda: load_prepared_lightning_capacity_data(
            source="live"
        ),
        "prepared_liquid_btc_data": lambda: load_prepared_liquid_btc_data(source="live"),
        "prepared_liquid_transactions_data": lambda: load_prepared_liquid_transactions_data(
            source="live"
        ),
        "prepared_filecoin_btc_data": lambda: load_prepared_filecoin_btc_data(source="live"),
        "prepared_monero_btc_data": lambda: load_prepared_monero_btc_data(source="live"),
        "prepared_litecoin_btc_data": lambda: load_prepared_litecoin_btc_data(source="live"),
        "prepared_dogecoin_btc_data": lambda: load_prepared_dogecoin_btc_data(source="live"),
        "prepared_us_m2_data": lambda: load_prepared_us_m2_data(source="live"),
        "prepared_russian_m2_data": lambda: load_prepared_russian_m2_data(source="live"),
    }


def _build_reference_snapshot():
    eur_usd, xau_usd = load_reference_series("2010-01-01", source="live")
    return (
        eur_usd.rename("EURUSD").to_frame().join(xau_usd.rename("XAUUSD"), how="outer").sort_index()
    )


def update_snapshots(selected_jobs: list[str] | None = None, *, dry_run: bool = False):
    jobs = _build_snapshot_jobs()
    target_names = selected_jobs or list(jobs.keys())

    for snapshot_key in target_names:
        if snapshot_key not in jobs:
            raise ValueError(f"Unknown snapshot key: {snapshot_key}")

        frame = jobs[snapshot_key]()
        if frame is None or frame.empty:
            raise ValueError(f"Snapshot job returned empty frame: {snapshot_key}")

        if not dry_run:
            write_snapshot_dataframe(snapshot_key, frame)
        print(f"{snapshot_key}: rows={len(frame)}")


def main():
    parser = argparse.ArgumentParser(
        description="Refresh checked-in daily snapshot CSV files used by the app at runtime."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and validate snapshots without writing data/snapshots/*.csv",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        help="Specific snapshot keys to refresh.",
    )
    args = parser.parse_args()

    update_snapshots(args.only, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
