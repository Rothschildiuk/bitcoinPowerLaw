from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.constants import (  # noqa: E402
    CURRENCY_DOLLAR,
    CURRENCY_EURO,
    CURRENCY_GOLD,
    MODE_POWERLAW,
    POWERLAW_SERIES_DIFFICULTY,
    POWERLAW_SERIES_HASHRATE,
    POWERLAW_SERIES_LIGHTNING_CAPACITY,
    POWERLAW_SERIES_LIGHTNING_NODES,
    POWERLAW_SERIES_LIQUID_BTC,
    POWERLAW_SERIES_LIQUID_TRANSACTIONS,
    POWERLAW_SERIES_PRICE,
    POWERLAW_SERIES_REVENUE,
)
from core.power_law import fit_powerlaw_regression  # noqa: E402
from core.series_registry import get_active_model_config  # noqa: E402
from services.price_service import (  # noqa: E402
    build_currency_close_series,
    load_prepared_difficulty_data,
    load_prepared_hashrate_data,
    load_prepared_lightning_capacity_data,
    load_prepared_lightning_nodes_data,
    load_prepared_liquid_btc_data,
    load_prepared_liquid_transactions_data,
    load_prepared_miner_revenue_data,
    load_prepared_price_data,
)

CONSTANTS_PATH = REPO_ROOT / "core" / "constants.py"

DEFAULT_CASES = [
    (POWERLAW_SERIES_PRICE, CURRENCY_DOLLAR, "DEFAULT_A", "DEFAULT_B"),
    (POWERLAW_SERIES_PRICE, CURRENCY_EURO, "DEFAULT_EURO_A", "DEFAULT_EURO_B"),
    (POWERLAW_SERIES_PRICE, CURRENCY_GOLD, "DEFAULT_GOLD_A", "DEFAULT_GOLD_B"),
    (POWERLAW_SERIES_REVENUE, CURRENCY_DOLLAR, "DEFAULT_REVENUE_A", "DEFAULT_REVENUE_B"),
    (
        POWERLAW_SERIES_DIFFICULTY,
        CURRENCY_DOLLAR,
        "DEFAULT_DIFFICULTY_A",
        "DEFAULT_DIFFICULTY_B",
    ),
    (POWERLAW_SERIES_HASHRATE, CURRENCY_DOLLAR, "DEFAULT_HASHRATE_A", "DEFAULT_HASHRATE_B"),
    (
        POWERLAW_SERIES_LIGHTNING_NODES,
        CURRENCY_DOLLAR,
        "DEFAULT_LIGHTNING_NODES_A",
        "DEFAULT_LIGHTNING_NODES_B",
    ),
    (
        POWERLAW_SERIES_LIGHTNING_CAPACITY,
        CURRENCY_DOLLAR,
        "DEFAULT_LIGHTNING_CAPACITY_A",
        "DEFAULT_LIGHTNING_CAPACITY_B",
    ),
    (POWERLAW_SERIES_LIQUID_BTC, CURRENCY_DOLLAR, "DEFAULT_LIQUID_BTC_A", "DEFAULT_LIQUID_BTC_B"),
    (
        POWERLAW_SERIES_LIQUID_TRANSACTIONS,
        CURRENCY_DOLLAR,
        "DEFAULT_LIQUID_TRANSACTIONS_A",
        "DEFAULT_LIQUID_TRANSACTIONS_B",
    ),
]


def _load_series_frames():
    return {
        POWERLAW_SERIES_PRICE: load_prepared_price_data(),
        POWERLAW_SERIES_REVENUE: load_prepared_miner_revenue_data(),
        POWERLAW_SERIES_DIFFICULTY: load_prepared_difficulty_data(),
        POWERLAW_SERIES_HASHRATE: load_prepared_hashrate_data(),
        POWERLAW_SERIES_LIGHTNING_NODES: load_prepared_lightning_nodes_data(),
        POWERLAW_SERIES_LIGHTNING_CAPACITY: load_prepared_lightning_capacity_data(),
        POWERLAW_SERIES_LIQUID_BTC: load_prepared_liquid_btc_data(),
        POWERLAW_SERIES_LIQUID_TRANSACTIONS: load_prepared_liquid_transactions_data(),
    }


def _prepare_fit_frame(series_name, currency, series_frames):
    raw_df_usd = series_frames[POWERLAW_SERIES_PRICE]
    config = get_active_model_config(MODE_POWERLAW, series_name, series_name, currency)

    if config.supports_currency_selector:
        fit_df = raw_df_usd.copy()
        fit_df["Close"] = build_currency_close_series(raw_df_usd, currency)
        fit_df = fit_df[fit_df["Close"] > 0].copy()
        fit_df["LogClose"] = np.log10(fit_df["Close"])
    else:
        fit_df = series_frames[series_name].copy()

    if config.analysis_min_abs_day is not None:
        fit_df = fit_df[fit_df["AbsDays"] >= int(config.analysis_min_abs_day)].copy()

    return fit_df


def compute_default_updates():
    series_frames = _load_series_frames()
    replacements = {}
    summary_rows = []

    for series_name, currency, a_name, b_name in DEFAULT_CASES:
        fit_df = _prepare_fit_frame(series_name, currency, series_frames)
        slope_b, intercept_a, r2_value = fit_powerlaw_regression(
            fit_df["AbsDays"].values,
            fit_df["LogClose"].values,
            0,
        )
        replacements[a_name] = f"{intercept_a:.3f}"
        replacements[b_name] = f"{slope_b:.3f}"
        summary_rows.append(
            {
                "series": series_name,
                "currency": currency,
                "a_name": a_name,
                "a_value": f"{intercept_a:.3f}",
                "b_name": b_name,
                "b_value": f"{slope_b:.3f}",
                "r2": f"{r2_value:.6f}",
                "rows": str(len(fit_df)),
            }
        )

    return replacements, summary_rows


def update_constants_file(constants_path, replacements):
    content = constants_path.read_text(encoding="utf-8")
    updated_content = content

    for constant_name, constant_value in replacements.items():
        pattern = rf"^{constant_name} = -?\d+\.\d+$"
        replacement = f"{constant_name} = {constant_value}"
        updated_content, count = re.subn(
            pattern,
            replacement,
            updated_content,
            count=1,
            flags=re.MULTILINE,
        )
        if count != 1:
            raise ValueError(f"Unable to update constant: {constant_name}")

    if updated_content != content:
        constants_path.write_text(updated_content, encoding="utf-8")


def print_summary(summary_rows):
    print("Updated PowerLaw defaults:")
    for row in summary_rows:
        print(
            f"- {row['series']} [{row['currency']}]: "
            f"{row['a_name']}={row['a_value']}, "
            f"{row['b_name']}={row['b_value']}, "
            f"R2={row['r2']}, rows={row['rows']}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Recompute and update hardcoded PowerLaw defaults."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the recomputed defaults without writing core/constants.py",
    )
    args = parser.parse_args()

    replacements, summary_rows = compute_default_updates()
    if not args.dry_run:
        update_constants_file(CONSTANTS_PATH, replacements)
    print_summary(summary_rows)


if __name__ == "__main__":
    main()
