import pandas as pd

# App/meta
APP_VERSION = "1.0"
GENESIS_DATE = pd.to_datetime("2009-01-03")

# Model defaults
DEFAULT_A = -16.520
DEFAULT_B = 5.693
DEFAULT_EURO_A = -16.687
DEFAULT_EURO_B = 5.721
DEFAULT_GOLD_A = -19.141
DEFAULT_GOLD_B = 5.523
DEFAULT_REVENUE_A = -7.652
DEFAULT_REVENUE_B = 4.096
OSC_DEFAULTS = {
    "lambda_val": 2.01,
    "t1_age": 2.489,
    "amp_factor_top": 1.13,
    "amp_factor_bottom": 0.88,
    "impulse_damping": 1.72,
}

# Forecast limits
DEFAULT_FORECAST_HORIZON = 6
FORECAST_HORIZON_MIN = 1
FORECAST_HORIZON_MAX = 100

# Session-state keys
KEY_THEME_MODE = "theme_mode"
KEY_LAST_MODE = "last_mode"
KEY_CHART_REVISION = "chart_revision"
KEY_MODE_SELECTOR = "mode_selector"
KEY_POWERLAW_SERIES = "powerlaw_series"
KEY_CURRENCY_SELECTOR = "currency_selector"
KEY_TIME_SCALE = "time_scale"
KEY_GENESIS_OFFSET = "genesis_offset"
KEY_A = "A"
KEY_B = "B"
KEY_A_PRICE = "A_price"
KEY_B_PRICE = "B_price"
KEY_A_EURO = "A_euro"
KEY_B_EURO = "B_euro"
KEY_A_GOLD = "A_gold"
KEY_B_GOLD = "B_gold"
KEY_A_REVENUE = "A_revenue"
KEY_B_REVENUE = "B_revenue"
KEY_POWERLAW_AUTO_FIT = "powerlaw_auto_fit"
KEY_LOGPERIODIC_AUTO_FIT = "logperiodic_auto_fit"
KEY_OSC_AUTOFIT_SIGNATURE = "osc_autofit_signature"
KEY_OSC_AUTOFIT_BEST_PARAMS = "osc_autofit_best_params"
KEY_PORTFOLIO_BTC_AMOUNT = "portfolio_btc_amount"
KEY_PORTFOLIO_FORECAST_UNIT = "portfolio_forecast_unit"
KEY_PORTFOLIO_FORECAST_HORIZON = "portfolio_forecast_horizon"
KEY_PORTFOLIO_FORECAST_MONTHS_LEGACY = "portfolio_forecast_months"

# UI options
MODE_POWERLAW = "PowerLaw"
MODE_LOGPERIODIC = "LogPeriodic"
MODE_PORTFOLIO = "Portfolio"
POWERLAW_SERIES_PRICE = "Bitcoin"
POWERLAW_SERIES_REVENUE = "Miner revenue"
POWERLAW_SERIES_OPTIONS = [POWERLAW_SERIES_PRICE, POWERLAW_SERIES_REVENUE]
CURRENCY_DOLLAR = "USD"
CURRENCY_EURO = "EUR"
CURRENCY_GOLD = "GOLD"
CURRENCY_OPTIONS = [CURRENCY_DOLLAR, CURRENCY_EURO, CURRENCY_GOLD]
TIME_LOG = "Log"
TIME_LIN = "Lin"
THEME_OPTIONS = ["Dark 🌑"]
DEFAULT_THEME = THEME_OPTIONS[0]
