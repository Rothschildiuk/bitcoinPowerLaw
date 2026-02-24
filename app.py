import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import oscillator, power_law
from core.constants import (
    APP_VERSION,
    CURRENCY_DOLLAR,
    CURRENCY_EURO,
    CURRENCY_GOLD,
    DEFAULT_FORECAST_HORIZON,
    DEFAULT_THEME,
    DEFAULT_A,
    DEFAULT_B,
    DEFAULT_EURO_A,
    DEFAULT_EURO_B,
    DEFAULT_GOLD_A,
    DEFAULT_GOLD_B,
    DEFAULT_HASHRATE_A,
    DEFAULT_HASHRATE_B,
    DEFAULT_DIFFICULTY_A,
    DEFAULT_DIFFICULTY_B,
    DEFAULT_REVENUE_A,
    DEFAULT_REVENUE_B,
    FORECAST_HORIZON_MAX,
    FORECAST_HORIZON_MIN,
    GENESIS_DATE,
    KEY_A_PRICE,
    KEY_B_PRICE,
    KEY_A_EURO,
    KEY_B_EURO,
    KEY_A_GOLD,
    KEY_B_GOLD,
    KEY_A_HASHRATE,
    KEY_B_HASHRATE,
    KEY_A_DIFFICULTY,
    KEY_B_DIFFICULTY,
    KEY_A_REVENUE,
    KEY_B_REVENUE,
    KEY_A,
    KEY_B,
    KEY_CHART_REVISION,
    KEY_CURRENCY_SELECTOR,
    KEY_GENESIS_OFFSET,
    KEY_LAST_MODE,
    KEY_POWERLAW_SERIES,
    KEY_PORTFOLIO_BTC_AMOUNT,
    KEY_PORTFOLIO_FORECAST_HORIZON,
    KEY_PORTFOLIO_FORECAST_UNIT,
    KEY_THEME_MODE,
    MODE_LOGPERIODIC,
    MODE_POWERLAW,
    OSC_DEFAULTS,
    POWERLAW_SERIES_PRICE,
    POWERLAW_SERIES_REVENUE,
    POWERLAW_SERIES_DIFFICULTY,
    POWERLAW_SERIES_HASHRATE,
    TIME_LOG,
)
from core.utils import calculate_r2_score, get_stable_trend_fit
from services.price_service import (
    build_currency_close_series,
    load_prepared_difficulty_data,
    load_prepared_hashrate_data,
    load_prepared_miner_revenue_data,
    load_prepared_price_data,
)
from ui.charts import render_main_model_chart
from ui.kpi import render_model_kpis
from ui.sidebar import render_sidebar_panel
from ui.theme import apply_theme_css, get_theme


def initialize_app_session_state(absolute_days=None, log_prices=None):
    defaults = {
        KEY_THEME_MODE: DEFAULT_THEME,
        KEY_LAST_MODE: MODE_POWERLAW,
        KEY_CURRENCY_SELECTOR: CURRENCY_DOLLAR,
        KEY_CHART_REVISION: 0,
        KEY_POWERLAW_SERIES: POWERLAW_SERIES_PRICE,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    # Light theme is disabled by product decision; always force dark theme.
    st.session_state[KEY_THEME_MODE] = DEFAULT_THEME

    if KEY_A_PRICE not in st.session_state:
        st.session_state[KEY_A_PRICE] = DEFAULT_A
    if KEY_B_PRICE not in st.session_state:
        st.session_state[KEY_B_PRICE] = DEFAULT_B
    if KEY_A_EURO not in st.session_state:
        st.session_state[KEY_A_EURO] = DEFAULT_EURO_A
    if KEY_B_EURO not in st.session_state:
        st.session_state[KEY_B_EURO] = DEFAULT_EURO_B
    if KEY_A_GOLD not in st.session_state:
        st.session_state[KEY_A_GOLD] = DEFAULT_GOLD_A
    if KEY_B_GOLD not in st.session_state:
        st.session_state[KEY_B_GOLD] = DEFAULT_GOLD_B
    if KEY_A_REVENUE not in st.session_state:
        st.session_state[KEY_A_REVENUE] = DEFAULT_REVENUE_A
    if KEY_B_REVENUE not in st.session_state:
        st.session_state[KEY_B_REVENUE] = DEFAULT_REVENUE_B
    if KEY_A_DIFFICULTY not in st.session_state:
        st.session_state[KEY_A_DIFFICULTY] = DEFAULT_DIFFICULTY_A
    if KEY_B_DIFFICULTY not in st.session_state:
        st.session_state[KEY_B_DIFFICULTY] = DEFAULT_DIFFICULTY_B
    if KEY_A_HASHRATE not in st.session_state:
        st.session_state[KEY_A_HASHRATE] = DEFAULT_HASHRATE_A
    if KEY_B_HASHRATE not in st.session_state:
        st.session_state[KEY_B_HASHRATE] = DEFAULT_HASHRATE_B
    if KEY_A not in st.session_state:
        st.session_state[KEY_A] = st.session_state[KEY_A_PRICE]
    if KEY_B not in st.session_state:
        st.session_state[KEY_B] = st.session_state[KEY_B_PRICE]


def resolve_currency_format(selected_currency):
    if selected_currency == CURRENCY_EURO:
        return {"prefix": "€", "suffix": "", "decimals": 2, "unit": "EUR"}
    if selected_currency == CURRENCY_GOLD:
        return {"prefix": "", "suffix": " oz", "decimals": 2, "unit": "XAU"}
    return {"prefix": "$", "suffix": "", "decimals": 0, "unit": "USD"}


def format_currency_value(value, prefix, suffix, decimals):
    return f"{prefix}{value:,.{decimals}f}{suffix}"


def resolve_trend_parameters(display_df, active_mode, a_key, b_key, default_a, default_b):
    intercept_a = float(st.session_state.get(a_key, default_a))
    slope_b = float(st.session_state.get(b_key, default_b))
    trend_log_prices = intercept_a + slope_b * display_df["LogD"].values
    residual_series = display_df["LogClose"].values - trend_log_prices

    # In oscillator mode, fallback to best-fit trend when session A/B is clearly invalid.
    # This prevents "price-like" residuals after mode toggles or stale widget state.
    if active_mode == MODE_LOGPERIODIC:
        intercept_a, slope_b, trend_log_prices, residual_series = get_stable_trend_fit(
            display_df["LogD"].values,
            display_df["LogClose"].values,
            intercept_a,
            slope_b,
        )

    return intercept_a, slope_b, trend_log_prices, residual_series


def calculate_percentile_offsets(display_df, genesis_offset_days):
    """
    Compute stable percentile offsets from a baseline best-fit trend for the current offset.
    This keeps percentile bands moving together with manual A/B adjustments.
    """
    fitted_b, fitted_a, _ = power_law.fit_powerlaw_regression(
        display_df["AbsDays"].values,
        display_df["LogClose"].values,
        genesis_offset_days,
    )
    if fitted_a == 0.0 and fitted_b == 0.0:
        baseline_residuals = display_df["Res"].values
    else:
        base_days = np.maximum(display_df["AbsDays"].values - genesis_offset_days, 1.0)
        baseline_log = fitted_a + fitted_b * np.log10(base_days)
        baseline_residuals = display_df["LogClose"].values - baseline_log

    return np.percentile(baseline_residuals, [2.5, 16.5, 83.5, 97.5])


@st.cache_data(ttl=3600)
def prepare_model_grid(current_gen_date, a_active, b_active, view_max):
    m_x = np.arange(1.0, float(np.ceil(view_max)) + 1.0)
    m_dates = [current_gen_date + pd.Timedelta(days=float(d)) for d in m_x]
    m_log_d = np.log10(m_x)
    m_fair_usd = 10 ** (a_active + b_active * m_log_d)
    m_dates_str = [d.strftime("%d.%m.%Y") for d in m_dates]
    return m_x, m_dates, m_log_d, m_fair_usd, m_dates_str


@st.cache_data(ttl=3600)
def build_portfolio_projection(
    _df_index, current_gen_date, a_active, b_active, btc_amount, forecast_unit, forecast_horizon
):
    # Anchor projections to a stable "current day" across environments.
    # If market data on a host is stale, use UTC today instead of lagging by the last data row.
    latest_data_day = pd.Timestamp(_df_index.max()).normalize()
    utc_today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    anchor_day = max(latest_data_day, utc_today)

    if forecast_unit == "Year":
        latest_year = int(anchor_day.year)
        start_period = pd.Timestamp(f"{latest_year - 1}-01-01")
        date_index = pd.date_range(start=start_period, periods=forecast_horizon + 1, freq="YS")
        change_usd_col, change_pct_col = "YoY_USD", "YoY_pct"
        table_title = "Yearly growth table"
    elif forecast_unit == "Day":
        latest_day = anchor_day
        start_period = latest_day - pd.Timedelta(days=1)
        date_index = pd.date_range(start=start_period, periods=forecast_horizon + 1, freq="D")
        change_usd_col, change_pct_col = "DoD_USD", "DoD_pct"
        table_title = "Daily growth table"
    else:
        latest_month_start = anchor_day.to_period("M").to_timestamp()
        start_period = latest_month_start - pd.offsets.MonthBegin(1)
        date_index = pd.date_range(start=start_period, periods=forecast_horizon + 1, freq="MS")
        change_usd_col, change_pct_col = "MoM_USD", "MoM_pct"
        table_title = "Monthly growth table"

    period_days = np.maximum((date_index - current_gen_date).days.astype(float), 1.0)
    period_fair_price = 10 ** (a_active + b_active * np.log10(period_days))
    period_portfolio_value = period_fair_price * btc_amount

    portfolio_df = pd.DataFrame(
        {
            "Date": date_index,
            "FairPriceUSD": period_fair_price,
            "PortfolioUSD": period_portfolio_value,
        }
    )
    portfolio_df[change_usd_col] = portfolio_df["PortfolioUSD"].diff()
    portfolio_df[change_pct_col] = portfolio_df["PortfolioUSD"].pct_change() * 100

    return portfolio_df, table_title, forecast_unit, change_usd_col, change_pct_col


def get_growth_change_labels(forecast_unit, currency_unit):
    prefix = "YoY" if forecast_unit == "Year" else ("DoD" if forecast_unit == "Day" else "MoM")
    return f"{prefix} Change ({currency_unit})", f"{prefix} Change (%)"


def render_portfolio_view(
    df_display,
    current_gen_date,
    a_active,
    b_active,
    currency_prefix,
    currency_suffix,
    currency_decimals,
    currency_unit,
    pl_template,
    pl_text_color,
    pl_bg_color,
    pl_grid_color,
    c_hover_bg,
    c_border,
    c_hover_text,
):
    display_currency_decimals = 0

    def format_portfolio_money(value):
        return f"{currency_prefix}{value:,.{display_currency_decimals}f}{currency_suffix}"

    st.markdown("### Portfolio Growth (Fair Price / Power Law)")
    btc_amount = float(st.session_state.get(KEY_PORTFOLIO_BTC_AMOUNT, 2.0))
    forecast_unit = st.session_state.get(KEY_PORTFOLIO_FORECAST_UNIT, "Month")
    forecast_horizon = int(
        st.session_state.get(KEY_PORTFOLIO_FORECAST_HORIZON, DEFAULT_FORECAST_HORIZON)
    )
    portfolio_df, table_title, forecast_unit, change_usd_col, change_pct_col = (
        build_portfolio_projection(
            df_display.index,
            current_gen_date,
            a_active,
            b_active,
            btc_amount,
            forecast_unit,
            forecast_horizon,
        )
    )
    portfolio_display_df = portfolio_df.iloc[1:].copy()
    portfolio_display_df["FairPriceDisplay"] = portfolio_display_df["FairPriceUSD"]
    portfolio_display_df["PortfolioDisplay"] = portfolio_display_df["PortfolioUSD"]
    portfolio_display_df["ChangeDisplay"] = portfolio_display_df[change_usd_col]

    baseline_value = portfolio_df["PortfolioUSD"].iloc[0]
    last_value = portfolio_display_df["PortfolioUSD"].iloc[-1]
    total_growth_pct = (
        ((last_value - baseline_value) / baseline_value) * 100 if baseline_value > 0 else 0.0
    )

    g1, g2, g3 = st.columns(3)
    g1.metric(
        "Current Fair Price",
        format_portfolio_money(df_display["FairDisplay"].iloc[-1]),
    )
    g2.metric(
        "Portfolio (end of horizon)",
        format_portfolio_money(portfolio_display_df["PortfolioDisplay"].iloc[-1]),
    )
    g3.metric("Total Growth", f"{total_growth_pct:+.1f}%")

    portfolio_fig = go.Figure()
    portfolio_fig.add_trace(
        go.Scatter(
            x=portfolio_display_df["Date"],
            y=portfolio_display_df["PortfolioDisplay"],
            mode="lines+markers",
            name="Portfolio by fair price",
            line=dict(color="#f0b90b", width=2),
            hovertemplate=(
                "<b>%{x|%d.%m.%Y}</b><br>Portfolio: "
                f"{currency_prefix}%{{y:,.{display_currency_decimals}f}}{currency_suffix}<extra></extra>"
            ),
        )
    )
    portfolio_fig.update_layout(
        height=320,
        margin=dict(t=10, b=0, l=50, r=20),
        template=pl_template,
        font=dict(color=pl_text_color),
        paper_bgcolor=pl_bg_color,
        plot_bgcolor=pl_bg_color,
        xaxis=dict(gridcolor=pl_grid_color, tickfont=dict(color=pl_text_color)),
        yaxis=dict(gridcolor=pl_grid_color, tickfont=dict(color=pl_text_color)),
        hoverlabel=dict(
            bgcolor=c_hover_bg, bordercolor=c_border, font=dict(color=c_hover_text, size=13)
        ),
    )
    st.plotly_chart(
        portfolio_fig,
        width="stretch",
        theme=None,
        config={"displayModeBar": False},
        key=f"portfolio_{st.session_state[KEY_THEME_MODE]}_{st.session_state[KEY_CHART_REVISION]}",
    )

    st.markdown(f"#### {table_title}")
    table_df = portfolio_display_df.copy()
    if forecast_unit == "Year":
        table_df["Date"] = table_df["Date"].dt.strftime("%Y")
    elif forecast_unit == "Day":
        table_df["Date"] = table_df["Date"].dt.strftime("%Y-%m-%d")
    else:
        table_df["Date"] = table_df["Date"].dt.strftime("%Y-%m")
    period_change_usd_label, period_change_pct_label = get_growth_change_labels(
        forecast_unit, currency_unit
    )
    table_df = table_df.rename(
        columns={
            "FairPriceDisplay": f"Fair Price ({currency_unit})",
            "PortfolioDisplay": f"Portfolio ({currency_unit})",
            "ChangeDisplay": period_change_usd_label,
            change_pct_col: period_change_pct_label,
        }
    )
    display_columns = [
        "Date",
        f"Fair Price ({currency_unit})",
        f"Portfolio ({currency_unit})",
        period_change_usd_label,
        period_change_pct_label,
    ]
    table_df = table_df[display_columns]
    money_fmt = f"{currency_prefix}{{:,.{display_currency_decimals}f}}{currency_suffix}"
    st.dataframe(
        table_df.style.format(
            {
                f"Fair Price ({currency_unit})": money_fmt,
                f"Portfolio ({currency_unit})": money_fmt,
                period_change_usd_label: money_fmt,
                period_change_pct_label: "{:+.2f}%",
            }
        ),
        width="stretch",
        hide_index=True,
    )


# --- Page Configuration ---
st.set_page_config(
    layout="wide", page_icon="🚀", page_title="BTC Power Law Pro", initial_sidebar_state="expanded"
)


try:
    raw_df_usd = load_prepared_price_data()
except Exception as e:
    st.error(f"Error loading BTC price data: {e}")
    st.stop()

try:
    raw_revenue_df = load_prepared_miner_revenue_data()
except Exception as e:
    st.error(f"Error loading miner revenue data: {e}")
    st.stop()

try:
    raw_difficulty_df = load_prepared_difficulty_data()
except Exception as e:
    st.error(f"Error loading difficulty data: {e}")
    st.stop()

try:
    raw_hashrate_df = load_prepared_hashrate_data()
except Exception as e:
    st.error(f"Error loading hashrate data: {e}")
    st.stop()

if KEY_CURRENCY_SELECTOR not in st.session_state:
    st.session_state[KEY_CURRENCY_SELECTOR] = CURRENCY_DOLLAR

raw_df_usd = raw_df_usd[raw_df_usd["Close"] > 0].copy()
raw_df_usd["LogClose"] = np.log10(raw_df_usd["Close"])
raw_revenue_df = raw_revenue_df[raw_revenue_df["Close"] > 0].copy()
raw_revenue_df["LogClose"] = np.log10(raw_revenue_df["Close"])
raw_difficulty_df = raw_difficulty_df[raw_difficulty_df["Close"] > 0].copy()
raw_difficulty_df["LogClose"] = np.log10(raw_difficulty_df["Close"])
raw_hashrate_df = raw_hashrate_df[raw_hashrate_df["Close"] > 0].copy()
raw_hashrate_df["LogClose"] = np.log10(raw_hashrate_df["Close"])

# Use current session currency for sidebar AF/R2 calculations in PowerLaw Bitcoin mode.
sidebar_currency = st.session_state.get(KEY_CURRENCY_SELECTOR, CURRENCY_DOLLAR)
if sidebar_currency not in [CURRENCY_DOLLAR, CURRENCY_EURO, CURRENCY_GOLD]:
    sidebar_currency = CURRENCY_DOLLAR
sidebar_price_close = build_currency_close_series(raw_df_usd, sidebar_currency)
sidebar_price_close = sidebar_price_close[sidebar_price_close > 0]
sidebar_price_log_close = np.log10(sidebar_price_close.values)

price_absolute_days = raw_df_usd["AbsDays"].values
price_log_close = sidebar_price_log_close
revenue_absolute_days = raw_revenue_df["AbsDays"].values
revenue_log_close = raw_revenue_df["LogClose"].values
difficulty_absolute_days = raw_difficulty_df["AbsDays"].values
difficulty_log_close = raw_difficulty_df["LogClose"].values
hashrate_absolute_days = raw_hashrate_df["AbsDays"].values
hashrate_log_close = raw_hashrate_df["LogClose"].values

# --- THEME + STATE ---
initialize_app_session_state(price_absolute_days, price_log_close)

theme = get_theme(True)
apply_theme_css(theme)

c_text_main = theme["c_text_main"]
pl_template = theme["pl_template"]
pl_bg_color = theme["pl_bg_color"]
pl_grid_color = theme["pl_grid_color"]
pl_btc_color = theme["pl_btc_color"]
pl_legend_color = theme["pl_legend_color"]
pl_text_color = theme["pl_text_color"]
c_hover_bg = theme["c_hover_bg"]
c_hover_text = theme["c_hover_text"]
c_border = theme["c_border"]

# --- SIDEBAR ASSEMBLY ---
mode, currency, time_scale, price_scale, current_r2, powerlaw_series = render_sidebar_panel(
    price_absolute_days,
    price_log_close,
    revenue_absolute_days,
    revenue_log_close,
    difficulty_absolute_days,
    difficulty_log_close,
    hashrate_absolute_days,
    hashrate_log_close,
    c_text_main,
    APP_VERSION,
    FORECAST_HORIZON_MIN,
    FORECAST_HORIZON_MAX,
)

# Keep backward-compatible keys in sync for oscillator mode code paths.
if sidebar_currency == CURRENCY_GOLD:
    st.session_state[KEY_A] = float(st.session_state.get(KEY_A_GOLD, DEFAULT_GOLD_A))
    st.session_state[KEY_B] = float(st.session_state.get(KEY_B_GOLD, DEFAULT_GOLD_B))
elif sidebar_currency == CURRENCY_EURO:
    st.session_state[KEY_A] = float(st.session_state.get(KEY_A_EURO, DEFAULT_EURO_A))
    st.session_state[KEY_B] = float(st.session_state.get(KEY_B_EURO, DEFAULT_EURO_B))
else:
    st.session_state[KEY_A] = float(st.session_state.get(KEY_A_PRICE, DEFAULT_A))
    st.session_state[KEY_B] = float(st.session_state.get(KEY_B_PRICE, DEFAULT_B))

if (
    mode == MODE_POWERLAW
    and powerlaw_series
    in [POWERLAW_SERIES_REVENUE, POWERLAW_SERIES_DIFFICULTY, POWERLAW_SERIES_HASHRATE]
    and currency != CURRENCY_DOLLAR
):
    st.session_state[KEY_CURRENCY_SELECTOR] = CURRENCY_DOLLAR
    st.rerun()
if (
    not (
        mode == MODE_POWERLAW
        and powerlaw_series
        in [POWERLAW_SERIES_REVENUE, POWERLAW_SERIES_DIFFICULTY, POWERLAW_SERIES_HASHRATE]
    )
    and currency != st.session_state.get(KEY_CURRENCY_SELECTOR, CURRENCY_DOLLAR)
):
    st.rerun()

# --- MAIN CALCULATIONS ---
genesis_offset = int(st.session_state.get(KEY_GENESIS_OFFSET, 0))
current_gen_date = GENESIS_DATE + pd.Timedelta(days=genesis_offset)

if mode == MODE_POWERLAW and powerlaw_series == POWERLAW_SERIES_REVENUE:
    raw_df = raw_revenue_df.copy()
    active_abs_days = revenue_absolute_days
    active_a_key = KEY_A_REVENUE
    active_b_key = KEY_B_REVENUE
    active_default_a = DEFAULT_REVENUE_A
    active_default_b = DEFAULT_REVENUE_B
    target_series_name = "Miner revenue"
    target_series_unit = "USD/day"
    currency = CURRENCY_DOLLAR
elif mode == MODE_POWERLAW and powerlaw_series == POWERLAW_SERIES_DIFFICULTY:
    raw_df = raw_difficulty_df.copy()
    active_abs_days = difficulty_absolute_days
    active_a_key = KEY_A_DIFFICULTY
    active_b_key = KEY_B_DIFFICULTY
    active_default_a = DEFAULT_DIFFICULTY_A
    active_default_b = DEFAULT_DIFFICULTY_B
    target_series_name = "Mining difficulty"
    target_series_unit = "Difficulty"
    currency = CURRENCY_DOLLAR
elif mode == MODE_POWERLAW and powerlaw_series == POWERLAW_SERIES_HASHRATE:
    raw_df = raw_hashrate_df.copy()
    active_abs_days = hashrate_absolute_days
    active_a_key = KEY_A_HASHRATE
    active_b_key = KEY_B_HASHRATE
    active_default_a = DEFAULT_HASHRATE_A
    active_default_b = DEFAULT_HASHRATE_B
    target_series_name = "Network hashrate"
    target_series_unit = "Hashrate"
    currency = CURRENCY_DOLLAR
else:
    raw_df = raw_df_usd.copy()
    raw_df["Close"] = build_currency_close_series(raw_df_usd, currency)
    raw_df = raw_df[raw_df["Close"] > 0].copy()
    raw_df["LogClose"] = np.log10(raw_df["Close"])
    active_abs_days = raw_df["AbsDays"].values
    if currency == CURRENCY_GOLD:
        active_a_key = KEY_A_GOLD
        active_b_key = KEY_B_GOLD
        active_default_a = DEFAULT_GOLD_A
        active_default_b = DEFAULT_GOLD_B
    elif currency == CURRENCY_EURO:
        active_a_key = KEY_A_EURO
        active_b_key = KEY_B_EURO
        active_default_a = DEFAULT_EURO_A
        active_default_b = DEFAULT_EURO_B
    else:
        active_a_key = KEY_A_PRICE
        active_b_key = KEY_B_PRICE
        active_default_a = DEFAULT_A
        active_default_b = DEFAULT_B
    target_series_name = "Bitcoin price"
    target_series_unit = currency

valid_idx = active_abs_days > genesis_offset
df_display = raw_df.iloc[valid_idx].copy()
if df_display.empty:
    st.error("No data available for the selected parameters.")
    st.stop()

df_display["Days"] = df_display["AbsDays"] - genesis_offset
df_display["LogD"] = np.log10(df_display["Days"])
a_active, b_active, model_log_vals, residual_vals = resolve_trend_parameters(
    df_display,
    mode,
    active_a_key,
    active_b_key,
    active_default_a,
    active_default_b,
)
df_display["ModelLog"] = model_log_vals
df_display["Res"] = residual_vals
df_display["Fair"] = 10 ** df_display["ModelLog"]

currency_context = resolve_currency_format(currency)
if mode == MODE_POWERLAW and powerlaw_series in [POWERLAW_SERIES_DIFFICULTY, POWERLAW_SERIES_HASHRATE]:
    currency_context = {"prefix": "", "suffix": "", "decimals": 0, "unit": "RAW"}
currency_prefix = currency_context["prefix"]
currency_suffix = currency_context["suffix"]
currency_decimals = int(currency_context["decimals"])
currency_unit = currency_context["unit"]
df_display["CloseDisplay"] = df_display["Close"]
df_display["FairDisplay"] = df_display["Fair"]

# Calculate R2 for Trend if not returned by sidebar (LogPeriodic mode)
if mode == MODE_LOGPERIODIC:
    current_r2 = calculate_r2_score(df_display["LogClose"].values, df_display["ModelLog"].values)

p2_5, p16_5, p83_5, p97_5 = calculate_percentile_offsets(df_display, genesis_offset)

# --- OSCILLATOR CALC ---
osc_lambda = float(st.session_state.get("lambda_val", OSC_DEFAULTS["lambda_val"]))
osc_t1_age = float(st.session_state.get("t1_age", OSC_DEFAULTS["t1_age"]))
osc_amp_top = float(st.session_state.get("amp_factor_top", OSC_DEFAULTS["amp_factor_top"]))
osc_amp_bottom = float(st.session_state.get("amp_factor_bottom", OSC_DEFAULTS["amp_factor_bottom"]))
osc_damping = float(st.session_state.get("impulse_damping", OSC_DEFAULTS["impulse_damping"]))
osc_amp, osc_omega, osc_phi = 0.0, 0.0, 0.0
r2_combined = current_r2

if mode == MODE_LOGPERIODIC:
    try:
        fit_result = oscillator.fit_oscillator_component(
            df_display["LogD"].values,
            df_display["Res"].values,
            osc_t1_age,
            osc_lambda,
            osc_amp_top,
            osc_amp_bottom,
            osc_damping,
        )
        if fit_result is None:
            osc_amp, osc_omega, osc_phi = 0.0, 0.0, 0.0
            osc_model_vals = np.zeros_like(df_display["Res"].values, dtype=float)
        else:
            osc_amp, osc_omega, osc_phi, osc_model_vals = fit_result

        total_model_log = df_display["ModelLog"] + osc_model_vals
        r2_combined = calculate_r2_score(df_display["LogClose"].values, total_model_log) * 100
    except Exception as e:
        st.error(f"LogPeriodic Error: {e}")
        osc_t1_age = OSC_DEFAULTS["t1_age"]
        osc_lambda = OSC_DEFAULTS["lambda_val"]
        osc_amp_top = OSC_DEFAULTS["amp_factor_top"]
        osc_amp_bottom = OSC_DEFAULTS["amp_factor_bottom"]
        osc_damping = OSC_DEFAULTS["impulse_damping"]
        osc_amp, osc_omega, osc_phi, r2_combined = 0, 0, 0, current_r2

# --- VIZ SETUP ---
view_max = df_display["Days"].max() + 365 * 10

# Use daily grid so unified hover has matching x-values across all traces.
m_x, m_dates, m_log_d, m_fair_usd, m_dates_str = prepare_model_grid(
    current_gen_date, a_active, b_active, view_max
)
m_fair_display = m_fair_usd

m_osc_y = oscillator.build_oscillator_curve(
    m_log_d,
    osc_amp,
    osc_omega,
    osc_phi,
    osc_amp_top,
    osc_amp_bottom,
    osc_damping,
    float(df_display["LogD"].min()),
)

is_log_time = time_scale == TIME_LOG
plot_x_model = m_x if is_log_time else m_dates
plot_x_main = df_display["Days"] if is_log_time else df_display.index
plot_x_osc = df_display["Days"] if is_log_time else df_display.index

if mode in [MODE_POWERLAW, MODE_LOGPERIODIC]:
    render_main_model_chart(
        mode=mode,
        time_scale=time_scale,
        price_scale=price_scale,
        df_display=df_display,
        current_gen_date=current_gen_date,
        view_max=view_max,
        plot_x_model=plot_x_model,
        plot_x_main=plot_x_main,
        plot_x_osc=plot_x_osc,
        m_log_d=m_log_d,
        m_dates=m_dates,
        m_dates_str=m_dates_str,
        m_fair_display=m_fair_display,
        m_osc_y=m_osc_y,
        p2_5=p2_5,
        p16_5=p16_5,
        p83_5=p83_5,
        p97_5=p97_5,
        osc_t1_age=osc_t1_age,
        osc_lambda=osc_lambda,
        pl_template=pl_template,
        pl_bg_color=pl_bg_color,
        pl_grid_color=pl_grid_color,
        pl_btc_color=pl_btc_color,
        pl_legend_color=pl_legend_color,
        pl_text_color=pl_text_color,
        c_hover_bg=c_hover_bg,
        c_hover_text=c_hover_text,
        c_border=c_border,
        currency_prefix=currency_prefix,
        currency_suffix=currency_suffix,
        currency_decimals=currency_decimals,
        target_series_name=target_series_name,
        target_series_unit=target_series_unit,
        show_halving_lines=(
            mode == MODE_POWERLAW
            and powerlaw_series
            in [POWERLAW_SERIES_REVENUE, POWERLAW_SERIES_DIFFICULTY, POWERLAW_SERIES_HASHRATE]
        ),
        chart_key=(
            f"chart_{mode}_{powerlaw_series}_{currency}_{time_scale}_{price_scale}_"
            f"{st.session_state[KEY_THEME_MODE]}_{st.session_state[KEY_CHART_REVISION]}"
        ),
    )
else:
    render_portfolio_view(
        df_display,
        current_gen_date,
        a_active,
        b_active,
        currency_prefix,
        currency_suffix,
        currency_decimals,
        currency_unit,
        pl_template,
        pl_text_color,
        pl_bg_color,
        pl_grid_color,
        c_hover_bg,
        c_border,
        c_hover_text,
    )

# --- KPI ---
render_model_kpis(
    df_display,
    a_active,
    b_active,
    p97_5,
    currency_prefix,
    currency_suffix,
    currency_decimals,
    target_series_name,
    target_series_unit,
)
