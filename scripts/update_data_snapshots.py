from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.price_service import (  # noqa: E402
    build_incremental_reference_series_snapshot,
    load_bitcoin_visuals_daily_data,
    load_prepared_difficulty_data,
    load_prepared_bitcoin_supply_data,
    load_prepared_hashrate_data,
    load_prepared_lightning_capacity_data,
    load_prepared_lightning_nodes_data,
    load_prepared_liquid_btc_data,
    load_prepared_liquid_transactions_data,
    load_prepared_miner_revenue_data,
    load_prepared_price_data,
    load_prepared_usdt_supply_data,
    read_snapshot_dataframe,
    write_snapshot_refresh_metadata,
    write_snapshot_dataframe,
)


def _build_snapshot_jobs():
    return {
        "reference_series": lambda: _build_reference_snapshot(),
        "prepared_price_data": lambda: load_prepared_price_data(source="auto"),
        "prepared_miner_revenue_data": lambda: load_prepared_miner_revenue_data(source="live"),
        "prepared_bitcoin_supply_data": lambda: load_prepared_bitcoin_supply_data(source="live"),
        "prepared_difficulty_data": lambda: load_prepared_difficulty_data(source="live"),
        "prepared_hashrate_data": lambda: load_prepared_hashrate_data(source="live"),
        "bitcoin_visuals_daily_data": lambda: load_bitcoin_visuals_daily_data(source="auto"),
        "prepared_lightning_nodes_data": lambda: load_prepared_lightning_nodes_data(source="auto"),
        "prepared_lightning_capacity_data": lambda: load_prepared_lightning_capacity_data(
            source="auto"
        ),
        "prepared_liquid_btc_data": lambda: load_prepared_liquid_btc_data(source="live"),
        "prepared_liquid_transactions_data": lambda: load_prepared_liquid_transactions_data(
            source="live"
        ),
        "prepared_usdt_supply_data": lambda: load_prepared_usdt_supply_data(source="live"),
    }


def _build_reference_snapshot():
    return build_incremental_reference_series_snapshot()


def _snapshot_last_date(frame):
    if isinstance(frame.index, pd.DatetimeIndex):
        dates = frame.index
    else:
        date_column = next((name for name in ("Date", "day") if name in frame.columns), None)
        if date_column is None:
            return None
        dates = pd.to_datetime(frame[date_column], errors="coerce")
    last_date = dates.max()
    return None if pd.isna(last_date) else pd.Timestamp(last_date)


def _find_snapshot_regression(snapshot_key, frame):
    """Why ``frame`` is worse than the checked-in snapshot, or None when it is not."""
    existing = read_snapshot_dataframe(snapshot_key)
    if existing is None or existing.empty:
        return None
    if len(frame) < len(existing):
        return f"has {len(frame)} rows, fewer than the {len(existing)} already stored"
    new_last_date = _snapshot_last_date(frame)
    existing_last_date = _snapshot_last_date(existing)
    if (
        new_last_date is not None
        and existing_last_date is not None
        and new_last_date < existing_last_date
    ):
        return (
            f"ends on {new_last_date.date()}, before the stored "
            f"{existing_last_date.date()}"
        )
    return None


def _warn(message):
    # GitHub Actions turns this into an annotation on the run summary.
    print(f"::warning::{message}")


def update_snapshots(selected_jobs: list[str] | None = None, *, dry_run: bool = False):
    """Refresh each snapshot on its own, keeping the stored one when a refresh looks bad.

    One failing source used to abort the whole refresh, so no series updated at all.
    A source that errors, comes back empty, or would move a snapshot backwards now only
    keeps its previous file. Refresh metadata is written only when some data changed,
    so its timestamp says when the data last moved rather than when the job ran.
    """
    jobs = _build_snapshot_jobs()
    target_names = selected_jobs or list(jobs.keys())
    for snapshot_key in target_names:
        if snapshot_key not in jobs:
            raise ValueError(f"Unknown snapshot key: {snapshot_key}")

    kept_snapshots = []
    any_changed = False
    for snapshot_key in target_names:
        try:
            frame = jobs[snapshot_key]()
        except Exception as exc:
            _warn(f"{snapshot_key}: refresh failed ({exc}); keeping the stored snapshot.")
            kept_snapshots.append(snapshot_key)
            continue

        if frame is None or frame.empty:
            _warn(f"{snapshot_key}: refresh returned no rows; keeping the stored snapshot.")
            kept_snapshots.append(snapshot_key)
            continue

        regression = _find_snapshot_regression(snapshot_key, frame)
        if regression is not None:
            _warn(f"{snapshot_key}: refreshed data {regression}; keeping the stored snapshot.")
            kept_snapshots.append(snapshot_key)
            continue

        changed = False
        if not dry_run:
            changed = bool(write_snapshot_dataframe(snapshot_key, frame))
            any_changed = any_changed or changed
        print(f"{snapshot_key}: rows={len(frame)}{'' if changed or dry_run else ' (unchanged)'}")

    if not dry_run and any_changed:
        write_snapshot_refresh_metadata()
    if kept_snapshots:
        print(f"Kept stored snapshots for: {', '.join(kept_snapshots)}")
    return kept_snapshots


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
