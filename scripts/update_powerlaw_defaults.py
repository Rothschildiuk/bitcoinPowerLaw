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
    CURRENCY_ALUMINUM,
    CURRENCY_CHF,
    CURRENCY_COPPER,
    CURRENCY_DOLLAR,
    CURRENCY_EURO,
    CURRENCY_GOLD,
    CURRENCY_IRON,
    CURRENCY_NDAQ,
    CURRENCY_OIL,
    CURRENCY_RUB,
    CURRENCY_SILVER,
    CURRENCY_SP500,
    CURRENCY_UAH,
    CURRENCY_US_HOUSING,
    MODE_POWERLAW,
    POWERLAW_SERIES_DOGECOIN_BTC,
    POWERLAW_SERIES_BITCOIN_MARKET_CAP,
    POWERLAW_SERIES_BITCOIN_VOLATILITY,
    POWERLAW_SERIES_DIFFICULTY,
    POWERLAW_SERIES_FILECOIN_BTC,
    POWERLAW_SERIES_HASHRATE,
    POWERLAW_SERIES_LITECOIN_BTC,
    POWERLAW_SERIES_LIGHTNING_CAPACITY,
    POWERLAW_SERIES_LIGHTNING_NODES,
    POWERLAW_SERIES_LIQUID_BTC,
    POWERLAW_SERIES_LIQUID_TRANSACTIONS,
    POWERLAW_SERIES_MONERO_BTC,
    POWERLAW_SERIES_PRICE,
    POWERLAW_SERIES_REVENUE,
    POWERLAW_SERIES_USDT_SUPPLY,
    POWERLAW_SERIES_US_M2,
)
from core.power_law import fit_powerlaw_regression  # noqa: E402
from core.series_registry import get_active_model_config  # noqa: E402
from services.price_service import (  # noqa: E402
    build_currency_close_series,
    build_prepared_bitcoin_market_cap_data,
    build_prepared_bitcoin_volatility_data,
    load_prepared_dogecoin_btc_data,
    load_prepared_difficulty_data,
    load_prepared_filecoin_btc_data,
    load_prepared_hashrate_data,
    load_prepared_litecoin_btc_data,
    load_prepared_lightning_capacity_data,
    load_prepared_lightning_nodes_data,
    load_prepared_liquid_btc_data,
    load_prepared_liquid_transactions_data,
    load_prepared_miner_revenue_data,
    load_prepared_bitcoin_supply_data,
    load_prepared_monero_btc_data,
    load_prepared_price_data,
    load_prepared_usdt_supply_data,
    load_prepared_us_m2_data,
)

CONSTANTS_PATH = REPO_ROOT / "core" / "constants.py"

DEFAULT_CASES = [
    (POWERLAW_SERIES_PRICE, CURRENCY_DOLLAR, "DEFAULT_A", "DEFAULT_B"),
    (POWERLAW_SERIES_PRICE, CURRENCY_EURO, "DEFAULT_EURO_A", "DEFAULT_EURO_B"),
    (POWERLAW_SERIES_PRICE, CURRENCY_CHF, "DEFAULT_CHF_A", "DEFAULT_CHF_B"),
    (POWERLAW_SERIES_PRICE, CURRENCY_UAH, "DEFAULT_UAH_A", "DEFAULT_UAH_B"),
    (POWERLAW_SERIES_PRICE, CURRENCY_RUB, "DEFAULT_RUB_A", "DEFAULT_RUB_B"),
    (POWERLAW_SERIES_PRICE, CURRENCY_GOLD, "DEFAULT_GOLD_A", "DEFAULT_GOLD_B"),
    (POWERLAW_SERIES_PRICE, CURRENCY_SILVER, "DEFAULT_SILVER_A", "DEFAULT_SILVER_B"),
    (POWERLAW_SERIES_PRICE, CURRENCY_COPPER, "DEFAULT_COPPER_A", "DEFAULT_COPPER_B"),
    (
        POWERLAW_SERIES_PRICE,
        CURRENCY_ALUMINUM,
        "DEFAULT_ALUMINUM_A",
        "DEFAULT_ALUMINUM_B",
    ),
    (POWERLAW_SERIES_PRICE, CURRENCY_IRON, "DEFAULT_IRON_A", "DEFAULT_IRON_B"),
    (POWERLAW_SERIES_PRICE, CURRENCY_OIL, "DEFAULT_OIL_A", "DEFAULT_OIL_B"),
    (
        POWERLAW_SERIES_PRICE,
        CURRENCY_US_HOUSING,
        "DEFAULT_US_HOUSING_A",
        "DEFAULT_US_HOUSING_B",
    ),
    (POWERLAW_SERIES_PRICE, CURRENCY_SP500, "DEFAULT_SP500_A", "DEFAULT_SP500_B"),
    (POWERLAW_SERIES_PRICE, CURRENCY_NDAQ, "DEFAULT_NDAQ_A", "DEFAULT_NDAQ_B"),
    (
        POWERLAW_SERIES_REVENUE,
        CURRENCY_DOLLAR,
        "DEFAULT_REVENUE_A",
        "DEFAULT_REVENUE_B",
    ),
    (
        POWERLAW_SERIES_BITCOIN_MARKET_CAP,
        CURRENCY_DOLLAR,
        "DEFAULT_BITCOIN_MARKET_CAP_A",
        "DEFAULT_BITCOIN_MARKET_CAP_B",
    ),
    (
        POWERLAW_SERIES_BITCOIN_VOLATILITY,
        CURRENCY_DOLLAR,
        "DEFAULT_BITCOIN_VOLATILITY_A",
        "DEFAULT_BITCOIN_VOLATILITY_B",
    ),
    (
        POWERLAW_SERIES_DIFFICULTY,
        CURRENCY_DOLLAR,
        "DEFAULT_DIFFICULTY_A",
        "DEFAULT_DIFFICULTY_B",
    ),
    (
        POWERLAW_SERIES_HASHRATE,
        CURRENCY_DOLLAR,
        "DEFAULT_HASHRATE_A",
        "DEFAULT_HASHRATE_B",
    ),
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
    (
        POWERLAW_SERIES_LIQUID_BTC,
        CURRENCY_DOLLAR,
        "DEFAULT_LIQUID_BTC_A",
        "DEFAULT_LIQUID_BTC_B",
    ),
    (
        POWERLAW_SERIES_LIQUID_TRANSACTIONS,
        CURRENCY_DOLLAR,
        "DEFAULT_LIQUID_TRANSACTIONS_A",
        "DEFAULT_LIQUID_TRANSACTIONS_B",
    ),
    (
        POWERLAW_SERIES_FILECOIN_BTC,
        CURRENCY_DOLLAR,
        "DEFAULT_FILECOIN_BTC_A",
        "DEFAULT_FILECOIN_BTC_B",
    ),
    (
        POWERLAW_SERIES_MONERO_BTC,
        CURRENCY_DOLLAR,
        "DEFAULT_MONERO_BTC_A",
        "DEFAULT_MONERO_BTC_B",
    ),
    (
        POWERLAW_SERIES_LITECOIN_BTC,
        CURRENCY_DOLLAR,
        "DEFAULT_LITECOIN_BTC_A",
        "DEFAULT_LITECOIN_BTC_B",
    ),
    (
        POWERLAW_SERIES_DOGECOIN_BTC,
        CURRENCY_DOLLAR,
        "DEFAULT_DOGECOIN_BTC_A",
        "DEFAULT_DOGECOIN_BTC_B",
    ),
    (POWERLAW_SERIES_US_M2, CURRENCY_DOLLAR, "DEFAULT_US_M2_A", "DEFAULT_US_M2_B"),
    (
        POWERLAW_SERIES_USDT_SUPPLY,
        CURRENCY_DOLLAR,
        "DEFAULT_USDT_SUPPLY_A",
        "DEFAULT_USDT_SUPPLY_B",
    ),
]


def _load_series_frames(source="snapshot"):
    prepared_price_data = load_prepared_price_data(source=source)
    prepared_bitcoin_supply_data = load_prepared_bitcoin_supply_data(source=source)
    return {
        POWERLAW_SERIES_PRICE: prepared_price_data,
        POWERLAW_SERIES_REVENUE: load_prepared_miner_revenue_data(source=source),
        POWERLAW_SERIES_BITCOIN_MARKET_CAP: build_prepared_bitcoin_market_cap_data(
            prepared_price_data,
            prepared_bitcoin_supply_data,
        ),
        POWERLAW_SERIES_BITCOIN_VOLATILITY: build_prepared_bitcoin_volatility_data(
            prepared_price_data
        ),
        POWERLAW_SERIES_DIFFICULTY: load_prepared_difficulty_data(source=source),
        POWERLAW_SERIES_HASHRATE: load_prepared_hashrate_data(source=source),
        POWERLAW_SERIES_LIGHTNING_NODES: load_prepared_lightning_nodes_data(source=source),
        POWERLAW_SERIES_LIGHTNING_CAPACITY: load_prepared_lightning_capacity_data(source=source),
        POWERLAW_SERIES_LIQUID_BTC: load_prepared_liquid_btc_data(source=source),
        POWERLAW_SERIES_LIQUID_TRANSACTIONS: load_prepared_liquid_transactions_data(
            source=source
        ),
        POWERLAW_SERIES_FILECOIN_BTC: load_prepared_filecoin_btc_data(source=source),
        POWERLAW_SERIES_MONERO_BTC: load_prepared_monero_btc_data(source=source),
        POWERLAW_SERIES_LITECOIN_BTC: load_prepared_litecoin_btc_data(source=source),
        POWERLAW_SERIES_DOGECOIN_BTC: load_prepared_dogecoin_btc_data(source=source),
        POWERLAW_SERIES_US_M2: load_prepared_us_m2_data(source=source),
        POWERLAW_SERIES_USDT_SUPPLY: load_prepared_usdt_supply_data(source=source),
    }


def _prepare_fit_frame(series_name, currency, series_frames, source="snapshot"):
    raw_df_usd = series_frames[POWERLAW_SERIES_PRICE]
    config = get_active_model_config(MODE_POWERLAW, series_name, currency)

    if config.supports_currency_selector:
        fit_df = raw_df_usd.copy()
        fit_df["Close"] = build_currency_close_series(
            raw_df_usd,
            currency,
            source=source,
        )
        fit_df = fit_df[fit_df["Close"] > 0].copy()
        fit_df["LogClose"] = np.log10(fit_df["Close"])
    else:
        fit_df = series_frames[series_name].copy()

    if config.analysis_min_abs_day is not None:
        fit_df = fit_df[fit_df["AbsDays"] >= int(config.analysis_min_abs_day)].copy()

    return fit_df


def compute_default_updates(source="snapshot"):
    series_frames = _load_series_frames(source=source)
    replacements = {}
    powerlaw_summary_rows = []

    for series_name, currency, a_name, b_name in DEFAULT_CASES:
        fit_df = _prepare_fit_frame(
            series_name,
            currency,
            series_frames,
            source=source,
        )
        config = get_active_model_config(MODE_POWERLAW, series_name, currency)
        genesis_offset = (
            int(config.model_origin_abs_day)
            if config.model_origin_abs_day is not None
            else 0
        )
        slope_b, intercept_a, r2_value = fit_powerlaw_regression(
            fit_df["AbsDays"].values,
            fit_df["LogClose"].values,
            genesis_offset,
        )
        replacements[a_name] = f"{intercept_a:.3f}"
        replacements[b_name] = f"{slope_b:.3f}"
        powerlaw_summary_rows.append(
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

    return replacements, powerlaw_summary_rows


def _replace_scalar_constant(content, constant_name, constant_value):
    pattern = rf"^{constant_name} = -?\d+\.\d+$"
    replacement = f"{constant_name} = {constant_value}"
    updated_content, count = re.subn(
        pattern,
        replacement,
        content,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise ValueError(f"Unable to update constant: {constant_name}")
    return updated_content


def update_constants_content(content, replacements):
    updated_content = content

    for constant_name, constant_value in replacements.items():
        updated_content = _replace_scalar_constant(
            updated_content, constant_name, constant_value
        )

    return updated_content


def update_constants_file(constants_path, replacements):
    content = constants_path.read_text(encoding="utf-8")
    updated_content = update_constants_content(content, replacements)

    if updated_content != content:
        constants_path.write_text(updated_content, encoding="utf-8")


def print_summary(powerlaw_summary_rows):
    print("Updated PowerLaw defaults:")
    for row in powerlaw_summary_rows:
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

    replacements, powerlaw_summary_rows = compute_default_updates()
    if not args.dry_run:
        update_constants_file(CONSTANTS_PATH, replacements)
    print_summary(powerlaw_summary_rows)


if __name__ == "__main__":
    main()
