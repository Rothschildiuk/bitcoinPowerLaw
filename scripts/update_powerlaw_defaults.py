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
    MODE_LOGPERIODIC,
    MODE_POWERLAW,
    OSC_DEFAULTS,
    OSC_DEFAULTS_DIFFICULTY,
    OSC_DEFAULTS_HASHRATE,
    POWERLAW_SERIES_DOGECOIN_BTC,
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
)
from core import oscillator  # noqa: E402
from core.power_law import fit_powerlaw_regression  # noqa: E402
from core.series_registry import get_active_model_config  # noqa: E402
from core.utils import get_stable_trend_fit  # noqa: E402
from services.price_service import (  # noqa: E402
    build_currency_close_series,
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
    load_prepared_monero_btc_data,
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
    (POWERLAW_SERIES_FILECOIN_BTC, CURRENCY_DOLLAR, "DEFAULT_FILECOIN_BTC_A", "DEFAULT_FILECOIN_BTC_B"),
    (POWERLAW_SERIES_MONERO_BTC, CURRENCY_DOLLAR, "DEFAULT_MONERO_BTC_A", "DEFAULT_MONERO_BTC_B"),
    (POWERLAW_SERIES_LITECOIN_BTC, CURRENCY_DOLLAR, "DEFAULT_LITECOIN_BTC_A", "DEFAULT_LITECOIN_BTC_B"),
    (
        POWERLAW_SERIES_DOGECOIN_BTC,
        CURRENCY_DOLLAR,
        "DEFAULT_DOGECOIN_BTC_A",
        "DEFAULT_DOGECOIN_BTC_B",
    ),
]

OSCILLATOR_DEFAULT_CASES = [
    (
        POWERLAW_SERIES_PRICE,
        CURRENCY_DOLLAR,
        "OSC_DEFAULTS",
        OSC_DEFAULTS,
    ),
    (
        POWERLAW_SERIES_DIFFICULTY,
        CURRENCY_DOLLAR,
        "OSC_DEFAULTS_DIFFICULTY",
        OSC_DEFAULTS_DIFFICULTY,
    ),
    (
        POWERLAW_SERIES_HASHRATE,
        CURRENCY_DOLLAR,
        "OSC_DEFAULTS_HASHRATE",
        OSC_DEFAULTS_HASHRATE,
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
        POWERLAW_SERIES_FILECOIN_BTC: load_prepared_filecoin_btc_data(),
        POWERLAW_SERIES_MONERO_BTC: load_prepared_monero_btc_data(),
        POWERLAW_SERIES_LITECOIN_BTC: load_prepared_litecoin_btc_data(),
        POWERLAW_SERIES_DOGECOIN_BTC: load_prepared_dogecoin_btc_data(),
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


def _prepare_logperiodic_fit_arrays(
    series_name,
    currency,
    series_frames,
    *,
    intercept_a,
    slope_b,
):
    fit_df = _prepare_fit_frame(series_name, currency, series_frames)
    config = get_active_model_config(MODE_LOGPERIODIC, POWERLAW_SERIES_PRICE, series_name, currency)

    abs_days = fit_df["AbsDays"].to_numpy(dtype=float)
    log_close = fit_df["LogClose"].to_numpy(dtype=float)
    days_since_genesis = abs_days
    valid_days_mask = days_since_genesis > 0
    if config.oscillator_min_abs_day is not None:
        valid_days_mask = valid_days_mask & (abs_days >= float(config.oscillator_min_abs_day))

    log_days = np.log10(days_since_genesis[valid_days_mask])
    _, _, _, residual_series = get_stable_trend_fit(
        log_days,
        log_close[valid_days_mask],
        float(intercept_a),
        float(slope_b),
    )
    return config, log_days, residual_series


def compute_default_updates():
    series_frames = _load_series_frames()
    replacements = {}
    powerlaw_summary_rows = []
    oscillator_summary_rows = []
    powerlaw_results = {}

    for series_name, currency, a_name, b_name in DEFAULT_CASES:
        fit_df = _prepare_fit_frame(series_name, currency, series_frames)
        slope_b, intercept_a, r2_value = fit_powerlaw_regression(
            fit_df["AbsDays"].values,
            fit_df["LogClose"].values,
            0,
        )
        replacements[a_name] = f"{intercept_a:.3f}"
        replacements[b_name] = f"{slope_b:.3f}"
        powerlaw_results[(series_name, currency)] = {
            "intercept_a": float(intercept_a),
            "slope_b": float(slope_b),
        }
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

    for series_name, currency, dict_name, oscillator_defaults in OSCILLATOR_DEFAULT_CASES:
        fit_params = powerlaw_results[(series_name, currency)]
        config, log_days, residual_series = _prepare_logperiodic_fit_arrays(
            series_name,
            currency,
            series_frames,
            intercept_a=fit_params["intercept_a"],
            slope_b=fit_params["slope_b"],
        )
        optimized = oscillator.optimize_visible_oscillator_parameters(
            log_days,
            residual_series,
            dict(oscillator_defaults),
            bounds_override=config.oscillator_parameter_bounds,
            parameter_order=["t1_age", "lambda_val"],
            step_map={"t1_age": 0.01, "lambda_val": 0.01},
            passes=24,
        )
        replacements[f"{dict_name}.t1_age"] = f"{optimized['t1_age']:.2f}"
        replacements[f"{dict_name}.lambda_val"] = f"{optimized['lambda_val']:.2f}"
        oscillator_r2 = oscillator.compute_oscillator_fit_r2(
            log_days,
            residual_series,
            optimized["t1_age"],
            optimized["lambda_val"],
            optimized["amp_factor_top"],
            optimized["amp_factor_bottom"],
            optimized["impulse_damping"],
        )
        oscillator_summary_rows.append(
            {
                "series": series_name,
                "currency": currency,
                "dict_name": dict_name,
                "t1_age": f"{optimized['t1_age']:.2f}",
                "lambda_val": f"{optimized['lambda_val']:.2f}",
                "r2": f"{oscillator_r2:.4f}",
                "rows": str(len(log_days)),
            }
        )

    return replacements, powerlaw_summary_rows, oscillator_summary_rows


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


def _replace_mapping_entry(content, mapping_name, entry_name, entry_value):
    block_pattern = rf"(?ms)^({mapping_name} = \{{\n.*?^\}})"
    block_match = re.search(block_pattern, content)
    if block_match is None:
        raise ValueError(f"Unable to find mapping: {mapping_name}")

    block_content = block_match.group(1)
    entry_pattern = rf'^(\s*"{re.escape(entry_name)}":\s*)-?\d+(?:\.\d+)?(,)$'
    updated_block, count = re.subn(
        entry_pattern,
        rf"\g<1>{entry_value}\g<2>",
        block_content,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise ValueError(f"Unable to update mapping entry: {mapping_name}.{entry_name}")
    return content.replace(block_content, updated_block, 1)


def update_constants_content(content, replacements):
    updated_content = content

    for constant_name, constant_value in replacements.items():
        if "." in constant_name:
            mapping_name, entry_name = constant_name.split(".", 1)
            updated_content = _replace_mapping_entry(
                updated_content,
                mapping_name,
                entry_name,
                constant_value,
            )
            continue
        updated_content = _replace_scalar_constant(updated_content, constant_name, constant_value)

    return updated_content


def update_constants_file(constants_path, replacements):
    content = constants_path.read_text(encoding="utf-8")
    updated_content = update_constants_content(content, replacements)

    if updated_content != content:
        constants_path.write_text(updated_content, encoding="utf-8")


def print_summary(powerlaw_summary_rows, oscillator_summary_rows):
    print("Updated PowerLaw defaults:")
    for row in powerlaw_summary_rows:
        print(
            f"- {row['series']} [{row['currency']}]: "
            f"{row['a_name']}={row['a_value']}, "
            f"{row['b_name']}={row['b_value']}, "
            f"R2={row['r2']}, rows={row['rows']}"
        )
    print("Updated LogPeriodic defaults:")
    for row in oscillator_summary_rows:
        print(
            f"- {row['series']} [{row['currency']}]: "
            f"{row['dict_name']}.t1_age={row['t1_age']}, "
            f"{row['dict_name']}.lambda_val={row['lambda_val']}, "
            f"R2={row['r2']}, rows={row['rows']}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Recompute and update hardcoded PowerLaw and LogPeriodic defaults."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the recomputed defaults without writing core/constants.py",
    )
    args = parser.parse_args()

    replacements, powerlaw_summary_rows, oscillator_summary_rows = compute_default_updates()
    if not args.dry_run:
        update_constants_file(CONSTANTS_PATH, replacements)
    print_summary(powerlaw_summary_rows, oscillator_summary_rows)


if __name__ == "__main__":
    main()
