from dataclasses import dataclass, replace

from core.constants import (
    CURRENCY_DOLLAR,
    CURRENCY_EURO,
    CURRENCY_GOLD,
    DEFAULT_A,
    DEFAULT_B,
    DEFAULT_DIFFICULTY_A,
    DEFAULT_DIFFICULTY_B,
    DEFAULT_EURO_A,
    DEFAULT_EURO_B,
    DEFAULT_GOLD_A,
    DEFAULT_GOLD_B,
    DEFAULT_HASHRATE_A,
    DEFAULT_HASHRATE_B,
    DEFAULT_LIGHTNING_CAPACITY_A,
    DEFAULT_LIGHTNING_CAPACITY_B,
    DEFAULT_LIGHTNING_NODES_A,
    DEFAULT_LIGHTNING_NODES_B,
    DEFAULT_LIQUID_BTC_A,
    DEFAULT_LIQUID_BTC_B,
    DEFAULT_LIQUID_TRANSACTIONS_A,
    DEFAULT_LIQUID_TRANSACTIONS_B,
    DEFAULT_REVENUE_A,
    DEFAULT_REVENUE_B,
    KEY_A_DIFFICULTY,
    KEY_A_EURO,
    KEY_A_GOLD,
    KEY_A_HASHRATE,
    KEY_A_LIGHTNING_CAPACITY,
    KEY_A_LIGHTNING_NODES,
    KEY_A_LIQUID_BTC,
    KEY_A_LIQUID_TRANSACTIONS,
    KEY_A_PRICE,
    KEY_A_REVENUE,
    KEY_B_DIFFICULTY,
    KEY_B_EURO,
    KEY_B_GOLD,
    KEY_B_HASHRATE,
    KEY_B_LIGHTNING_CAPACITY,
    KEY_B_LIGHTNING_NODES,
    KEY_B_LIQUID_BTC,
    KEY_B_LIQUID_TRANSACTIONS,
    KEY_B_PRICE,
    KEY_B_REVENUE,
    DIFFICULTY_HASHRATE_ANALYSIS_START_ABS_DAYS,
    MODE_LOGPERIODIC,
    MODE_PORTFOLIO,
    OSC_DEFAULTS,
    OSC_DEFAULTS_DIFFICULTY,
    OSC_DEFAULTS_HASHRATE,
    POWERLAW_SERIES_DIFFICULTY,
    POWERLAW_SERIES_HASHRATE,
    POWERLAW_SERIES_LIGHTNING_CAPACITY,
    POWERLAW_SERIES_LIGHTNING_NODES,
    POWERLAW_SERIES_LIQUID_BTC,
    POWERLAW_SERIES_LIQUID_TRANSACTIONS,
    POWERLAW_SERIES_PRICE,
    POWERLAW_SERIES_REVENUE,
)


@dataclass(frozen=True)
class SeriesModelConfig:
    series_name: str
    a_key: str
    b_key: str
    default_a: float
    default_b: float
    target_series_name: str
    target_series_unit: str
    currency_prefix: str
    currency_suffix: str
    currency_decimals: int
    currency_unit: str
    supports_currency_selector: bool = False
    powerlaw_enabled: bool = True
    logperiodic_enabled: bool = False
    lock_price_scale_to_log: bool = False
    show_halving_lines: bool = False
    analysis_min_abs_day: int | None = None
    oscillator_defaults: dict | None = None
    oscillator_min_abs_day: int | None = None
    oscillator_parameter_bounds: dict[str, tuple[float, float]] | None = None


_BASE_SERIES_CONFIGS = {
    POWERLAW_SERIES_PRICE: SeriesModelConfig(
        series_name=POWERLAW_SERIES_PRICE,
        a_key=KEY_A_PRICE,
        b_key=KEY_B_PRICE,
        default_a=DEFAULT_A,
        default_b=DEFAULT_B,
        target_series_name="Bitcoin price",
        target_series_unit=CURRENCY_DOLLAR,
        currency_prefix="$",
        currency_suffix="",
        currency_decimals=0,
        currency_unit=CURRENCY_DOLLAR,
        supports_currency_selector=True,
        logperiodic_enabled=True,
        oscillator_defaults=OSC_DEFAULTS,
    ),
    POWERLAW_SERIES_REVENUE: SeriesModelConfig(
        series_name=POWERLAW_SERIES_REVENUE,
        a_key=KEY_A_REVENUE,
        b_key=KEY_B_REVENUE,
        default_a=DEFAULT_REVENUE_A,
        default_b=DEFAULT_REVENUE_B,
        target_series_name="Miner revenue",
        target_series_unit="USD/day",
        currency_prefix="$",
        currency_suffix="",
        currency_decimals=0,
        currency_unit=CURRENCY_DOLLAR,
        show_halving_lines=True,
    ),
    POWERLAW_SERIES_DIFFICULTY: SeriesModelConfig(
        series_name=POWERLAW_SERIES_DIFFICULTY,
        a_key=KEY_A_DIFFICULTY,
        b_key=KEY_B_DIFFICULTY,
        default_a=DEFAULT_DIFFICULTY_A,
        default_b=DEFAULT_DIFFICULTY_B,
        target_series_name="Mining difficulty",
        target_series_unit="Difficulty",
        currency_prefix="",
        currency_suffix="",
        currency_decimals=0,
        currency_unit="RAW",
        logperiodic_enabled=True,
        lock_price_scale_to_log=True,
        show_halving_lines=True,
        analysis_min_abs_day=DIFFICULTY_HASHRATE_ANALYSIS_START_ABS_DAYS,
        oscillator_defaults=OSC_DEFAULTS_DIFFICULTY,
        oscillator_parameter_bounds={"lambda_val": (1.5, 8.0)},
    ),
    POWERLAW_SERIES_HASHRATE: SeriesModelConfig(
        series_name=POWERLAW_SERIES_HASHRATE,
        a_key=KEY_A_HASHRATE,
        b_key=KEY_B_HASHRATE,
        default_a=DEFAULT_HASHRATE_A,
        default_b=DEFAULT_HASHRATE_B,
        target_series_name="Network hashrate",
        target_series_unit="Hashrate",
        currency_prefix="",
        currency_suffix="",
        currency_decimals=0,
        currency_unit="RAW",
        logperiodic_enabled=True,
        lock_price_scale_to_log=True,
        show_halving_lines=True,
        analysis_min_abs_day=DIFFICULTY_HASHRATE_ANALYSIS_START_ABS_DAYS,
        oscillator_defaults=OSC_DEFAULTS_HASHRATE,
        oscillator_parameter_bounds={"lambda_val": (1.5, 8.0)},
    ),
    POWERLAW_SERIES_LIGHTNING_NODES: SeriesModelConfig(
        series_name=POWERLAW_SERIES_LIGHTNING_NODES,
        a_key=KEY_A_LIGHTNING_NODES,
        b_key=KEY_B_LIGHTNING_NODES,
        default_a=DEFAULT_LIGHTNING_NODES_A,
        default_b=DEFAULT_LIGHTNING_NODES_B,
        target_series_name="Lightning nodes",
        target_series_unit="Nodes",
        currency_prefix="",
        currency_suffix="",
        currency_decimals=0,
        currency_unit="RAW",
        show_halving_lines=True,
    ),
    POWERLAW_SERIES_LIGHTNING_CAPACITY: SeriesModelConfig(
        series_name=POWERLAW_SERIES_LIGHTNING_CAPACITY,
        a_key=KEY_A_LIGHTNING_CAPACITY,
        b_key=KEY_B_LIGHTNING_CAPACITY,
        default_a=DEFAULT_LIGHTNING_CAPACITY_A,
        default_b=DEFAULT_LIGHTNING_CAPACITY_B,
        target_series_name="Lightning capacity",
        target_series_unit="BTC",
        currency_prefix="",
        currency_suffix=" BTC",
        currency_decimals=3,
        currency_unit="BTC",
        show_halving_lines=True,
    ),
    POWERLAW_SERIES_LIQUID_BTC: SeriesModelConfig(
        series_name=POWERLAW_SERIES_LIQUID_BTC,
        a_key=KEY_A_LIQUID_BTC,
        b_key=KEY_B_LIQUID_BTC,
        default_a=DEFAULT_LIQUID_BTC_A,
        default_b=DEFAULT_LIQUID_BTC_B,
        target_series_name="Liquid BTC balance",
        target_series_unit="BTC",
        currency_prefix="",
        currency_suffix=" BTC",
        currency_decimals=3,
        currency_unit="BTC",
        show_halving_lines=True,
    ),
    POWERLAW_SERIES_LIQUID_TRANSACTIONS: SeriesModelConfig(
        series_name=POWERLAW_SERIES_LIQUID_TRANSACTIONS,
        a_key=KEY_A_LIQUID_TRANSACTIONS,
        b_key=KEY_B_LIQUID_TRANSACTIONS,
        default_a=DEFAULT_LIQUID_TRANSACTIONS_A,
        default_b=DEFAULT_LIQUID_TRANSACTIONS_B,
        target_series_name="Liquid transactions",
        target_series_unit="Transactions/week",
        currency_prefix="",
        currency_suffix="",
        currency_decimals=0,
        currency_unit="RAW",
        show_halving_lines=True,
    ),
}

_PRICE_CURRENCY_OVERRIDES = {
    CURRENCY_DOLLAR: {
        "a_key": KEY_A_PRICE,
        "b_key": KEY_B_PRICE,
        "default_a": DEFAULT_A,
        "default_b": DEFAULT_B,
        "target_series_unit": CURRENCY_DOLLAR,
        "currency_prefix": "$",
        "currency_suffix": "",
        "currency_decimals": 0,
        "currency_unit": CURRENCY_DOLLAR,
    },
    CURRENCY_EURO: {
        "a_key": KEY_A_EURO,
        "b_key": KEY_B_EURO,
        "default_a": DEFAULT_EURO_A,
        "default_b": DEFAULT_EURO_B,
        "target_series_unit": CURRENCY_EURO,
        "currency_prefix": "€",
        "currency_suffix": "",
        "currency_decimals": 2,
        "currency_unit": CURRENCY_EURO,
    },
    CURRENCY_GOLD: {
        "a_key": KEY_A_GOLD,
        "b_key": KEY_B_GOLD,
        "default_a": DEFAULT_GOLD_A,
        "default_b": DEFAULT_GOLD_B,
        "target_series_unit": CURRENCY_GOLD,
        "currency_prefix": "",
        "currency_suffix": " oz",
        "currency_decimals": 2,
        "currency_unit": "XAU",
    },
}

_POWERLAW_SERIES_GROUPS = [
    (
        "Bitcoin network",
        [
            POWERLAW_SERIES_PRICE,
            POWERLAW_SERIES_REVENUE,
            POWERLAW_SERIES_DIFFICULTY,
            POWERLAW_SERIES_HASHRATE,
        ],
    ),
    (
        "Lightning Network",
        [
            POWERLAW_SERIES_LIGHTNING_NODES,
            POWERLAW_SERIES_LIGHTNING_CAPACITY,
        ],
    ),
    (
        "Liquid",
        [
            POWERLAW_SERIES_LIQUID_BTC,
            POWERLAW_SERIES_LIQUID_TRANSACTIONS,
        ],
    ),
]


def get_series_config(series_name, selected_currency=CURRENCY_DOLLAR):
    base_config = _BASE_SERIES_CONFIGS.get(series_name, _BASE_SERIES_CONFIGS[POWERLAW_SERIES_PRICE])
    if not base_config.supports_currency_selector:
        return base_config

    override = _PRICE_CURRENCY_OVERRIDES.get(
        selected_currency, _PRICE_CURRENCY_OVERRIDES[CURRENCY_DOLLAR]
    )
    return replace(base_config, **override)


def get_powerlaw_series_options():
    return [
        config.series_name for config in _BASE_SERIES_CONFIGS.values() if config.powerlaw_enabled
    ]


def get_powerlaw_series_groups():
    return list(_POWERLAW_SERIES_GROUPS)


def get_powerlaw_series_group_map():
    return {
        group_name: list(series_options) for group_name, series_options in _POWERLAW_SERIES_GROUPS
    }


def get_powerlaw_series_group_for_series(series_name):
    for group_name, series_options in _POWERLAW_SERIES_GROUPS:
        if series_name in series_options:
            return group_name
    return _POWERLAW_SERIES_GROUPS[0][0]


def get_logperiodic_series_options():
    return [
        config.series_name for config in _BASE_SERIES_CONFIGS.values() if config.logperiodic_enabled
    ]


def get_selected_series_name(mode, powerlaw_series, logperiodic_series):
    if mode == MODE_LOGPERIODIC:
        return logperiodic_series
    if mode == MODE_PORTFOLIO:
        return POWERLAW_SERIES_PRICE
    return powerlaw_series


def get_active_model_config(mode, powerlaw_series, logperiodic_series, selected_currency):
    selected_series = get_selected_series_name(mode, powerlaw_series, logperiodic_series)
    return get_series_config(selected_series, selected_currency=selected_currency)


def series_supports_currency_selector(mode, powerlaw_series, logperiodic_series):
    return get_active_model_config(
        mode, powerlaw_series, logperiodic_series, selected_currency=CURRENCY_DOLLAR
    ).supports_currency_selector


def iter_session_model_defaults():
    seen_keys = set()
    for currency in (CURRENCY_DOLLAR, CURRENCY_EURO, CURRENCY_GOLD):
        price_config = get_series_config(POWERLAW_SERIES_PRICE, selected_currency=currency)
        for key, value in (
            (price_config.a_key, price_config.default_a),
            (price_config.b_key, price_config.default_b),
        ):
            if key not in seen_keys:
                seen_keys.add(key)
                yield key, value

    for series_name, config in _BASE_SERIES_CONFIGS.items():
        if series_name == POWERLAW_SERIES_PRICE:
            continue
        for key, value in (
            (config.a_key, config.default_a),
            (config.b_key, config.default_b),
        ):
            if key not in seen_keys:
                seen_keys.add(key)
                yield key, value
