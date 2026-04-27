import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import oscillator, power_law
from core.constants import (
    APP_VERSION,
    BAND_METHOD_GAUSSIAN,
    BAND_METHOD_QUANTILE,
    CURRENCY_DOLLAR,
    CURRENCY_EURO,
    CURRENCY_GOLD,
    DEFAULT_FORECAST_HORIZON,
    DEFAULT_THEME,
    FORECAST_HORIZON_MAX,
    FORECAST_HORIZON_MIN,
    GENESIS_DATE,
    KEY_A,
    KEY_A_PRICE,
    KEY_B,
    KEY_B_PRICE,
    KEY_BAND_METHOD,
    KEY_BITCOIN_NETWORK_SIMULATION_RESOLUTION,
    KEY_BITCOIN_NETWORK_SIMULATION_SEED,
    KEY_CHART_REVISION,
    KEY_CURRENCY_SELECTOR,
    KEY_GENESIS_OFFSET,
    KEY_LAST_MODE,
    KEY_LOGPERIODIC_SERIES,
    KEY_POWERLAW_SERIES,
    KEY_PORTFOLIO_BTC_AMOUNT,
    KEY_PORTFOLIO_FORECAST_HORIZON,
    KEY_PORTFOLIO_FORECAST_UNIT,
    KEY_PORTFOLIO_MONTHLY_BUY_AMOUNT,
    KEY_THEME_MODE,
    MODE_LOGPERIODIC,
    MODE_PORTFOLIO,
    MODE_POWERLAW,
    OSC_DEFAULTS,
    POWERLAW_SERIES_DOGECOIN_BTC,
    POWERLAW_SERIES_DIFFICULTY,
    POWERLAW_SERIES_FILECOIN_BTC,
    POWERLAW_SERIES_HASHRATE,
    POWERLAW_SERIES_BITCOIN_NETWORK_SIMULATION,
    POWERLAW_SERIES_LITECOIN_BTC,
    POWERLAW_SERIES_LIGHTNING_CAPACITY,
    POWERLAW_SERIES_LIGHTNING_NODES,
    POWERLAW_SERIES_LIQUID_BTC,
    POWERLAW_SERIES_LIQUID_TRANSACTIONS,
    POWERLAW_SERIES_MONERO_BTC,
    POWERLAW_SERIES_PRICE,
    POWERLAW_SERIES_REVENUE,
    POWERLAW_SERIES_RUSSIAN_M2,
    POWERLAW_SERIES_US_M2,
    TIME_LOG,
)
from core.series_registry import (
    get_active_model_config,
    get_selected_series_name,
    iter_session_model_defaults,
    series_supports_currency_selector,
)
from core.simulation import build_bitcoin_network_simulation
from core.utils import (
    PortfolioSettings,
    build_portfolio_projection,
    build_portfolio_view_model,
    calculate_r2_score,
    evaluate_powerlaw_values,
    powerlaw_parameters_are_unstable,
    resolve_trend_parameters,
)
from services.price_service import (
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
    load_prepared_russian_m2_data,
    load_prepared_us_m2_data,
)
from ui.charts import _resolve_model_view_max, render_main_model_chart
from ui.kpi import render_model_kpis
from ui.sidebar import render_sidebar_panel
from ui.theme import apply_theme_css, get_theme


def initialize_app_session_state():
    defaults = {
        KEY_THEME_MODE: DEFAULT_THEME,
        KEY_LAST_MODE: MODE_POWERLAW,
        KEY_CURRENCY_SELECTOR: CURRENCY_DOLLAR,
        KEY_CHART_REVISION: 0,
        KEY_POWERLAW_SERIES: POWERLAW_SERIES_PRICE,
        KEY_LOGPERIODIC_SERIES: POWERLAW_SERIES_PRICE,
        KEY_BAND_METHOD: BAND_METHOD_QUANTILE,
        KEY_BITCOIN_NETWORK_SIMULATION_SEED: 1,
        KEY_BITCOIN_NETWORK_SIMULATION_RESOLUTION: 0.00001,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    # Light theme is disabled by product decision; always force dark theme.
    st.session_state[KEY_THEME_MODE] = DEFAULT_THEME

    for key, value in iter_session_model_defaults():
        if key not in st.session_state:
            st.session_state[key] = value
    if KEY_A not in st.session_state:
        st.session_state[KEY_A] = st.session_state[KEY_A_PRICE]
    if KEY_B not in st.session_state:
        st.session_state[KEY_B] = st.session_state[KEY_B_PRICE]


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


def calculate_gaussian_offsets(display_df, genesis_offset_days):
    """
    Compute gaussian-like offsets (mean +/- 1σ and +/- 2σ) from baseline residuals.
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

    mu = float(np.mean(baseline_residuals))
    sigma = float(np.std(baseline_residuals))
    return (
        mu - 2.0 * sigma,
        mu - 1.0 * sigma,
        mu + 1.0 * sigma,
        mu + 2.0 * sigma,
    )


@st.cache_data(ttl=3600)
def prepare_model_grid(current_gen_date, a_active, b_active, view_max):
    m_x = np.arange(1.0, float(np.ceil(view_max)) + 1.0)
    m_dates = [current_gen_date + pd.Timedelta(days=float(d)) for d in m_x]
    m_log_d = np.log10(m_x)
    m_fair_usd, _, _ = evaluate_powerlaw_values(m_log_d, a_active, b_active)
    m_dates_str = [d.strftime("%d.%m.%Y") for d in m_dates]
    return m_x, m_dates, m_log_d, m_fair_usd, m_dates_str


@st.cache_data(ttl=3600)
def prepare_portfolio_projection(
    _df_index,
    current_gen_date,
    a_active,
    b_active,
    settings,
):
    return build_portfolio_projection(
        _df_index,
        current_gen_date,
        a_active,
        b_active,
        settings,
    )


def render_portfolio_view(
    df_display,
    current_gen_date,
    a_active,
    b_active,
    current_r2,
    model_was_clipped,
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
    display_currency_decimals = int(currency_decimals)

    def format_portfolio_money(value):
        return f"{currency_prefix}{value:,.{display_currency_decimals}f}{currency_suffix}"

    st.markdown("### Portfolio Growth (Fair Price / Power Law)")
    settings = PortfolioSettings(
        btc_amount=float(st.session_state.get(KEY_PORTFOLIO_BTC_AMOUNT, 2.0)),
        monthly_buy_amount=float(st.session_state.get(KEY_PORTFOLIO_MONTHLY_BUY_AMOUNT, 0.0)),
        forecast_unit=st.session_state.get(KEY_PORTFOLIO_FORECAST_UNIT, "Year"),
        forecast_horizon=int(
            st.session_state.get(KEY_PORTFOLIO_FORECAST_HORIZON, DEFAULT_FORECAST_HORIZON)
        ),
    )
    projection_result = prepare_portfolio_projection(
        df_display.index,
        current_gen_date,
        a_active,
        b_active,
        settings,
    )
    portfolio_view = build_portfolio_view_model(
        projection_result,
        settings.monthly_buy_amount,
        currency_unit,
    )

    unstable_portfolio = powerlaw_parameters_are_unstable(
        current_r2,
        was_clipped=model_was_clipped,
    )
    if unstable_portfolio:
        st.info(
            "Portfolio projection needs a stable model fit. Click Auto-fit model to calculate fair-value metrics."
        )
        return

    g1, g2, g3 = st.columns(3)
    g1.metric("Current Fair Price", format_portfolio_money(df_display["FairDisplay"].iloc[-1]))
    if portfolio_view.dca_enabled:
        g2.metric(
            "Hold-only portfolio",
            format_portfolio_money(portfolio_view.last_value),
            delta=f"{portfolio_view.total_growth_pct:+.1f}%",
        )
        g3.metric(
            "With monthly buys",
            format_portfolio_money(portfolio_view.last_dca_value),
            delta=format_portfolio_money(portfolio_view.last_dca_value - portfolio_view.last_value),
        )
        st.caption(
            f"Invested cash by horizon: {format_portfolio_money(portfolio_view.last_dca_invested_capital)}"
        )
        st.caption(
            "Experimental scenario: monthly buys start from the next calendar month and use the current selected currency."
        )
    else:
        g2.metric(
            "Portfolio (end of horizon)",
            format_portfolio_money(portfolio_view.last_value),
        )
        g3.metric("Total Growth", f"{portfolio_view.total_growth_pct:+.1f}%")

    portfolio_fig = go.Figure()
    portfolio_fig.add_trace(
        go.Scatter(
            x=portfolio_view.portfolio_display_df["Date"],
            y=portfolio_view.portfolio_display_df["PortfolioDisplay"],
            mode="lines+markers",
            name="Portfolio by fair price",
            line=dict(color="#f0b90b", width=2),
            hovertemplate=(
                "<b>%{x|%d.%m.%Y}</b><br>Portfolio: "
                f"{currency_prefix}%{{y:,.{display_currency_decimals}f}}{currency_suffix}<extra></extra>"
            ),
        )
    )
    if portfolio_view.dca_enabled:
        portfolio_fig.add_trace(
            go.Scatter(
                x=portfolio_view.portfolio_display_df["Date"],
                y=portfolio_view.portfolio_display_df["DcaPortfolioDisplay"],
                mode="lines+markers",
                name="Portfolio with monthly buys",
                line=dict(color="#14b8a6", width=2),
                hovertemplate=(
                    "<b>%{x|%d.%m.%Y}</b><br>Portfolio + monthly buys: "
                    f"{currency_prefix}%{{y:,.{display_currency_decimals}f}}{currency_suffix}<extra></extra>"
                ),
            )
        )
        portfolio_fig.add_trace(
            go.Scatter(
                x=portfolio_view.portfolio_display_df["Date"],
                y=portfolio_view.portfolio_display_df["DcaInvestedCapitalDisplay"],
                mode="lines+markers",
                name="Cumulative invested cash",
                line=dict(color="#8b5cf6", width=2, dash="dash"),
                hovertemplate=(
                    "<b>%{x|%d.%m.%Y}</b><br>Invested cash: "
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
        xaxis=dict(
            gridcolor=pl_grid_color,
            tickfont=dict(color=pl_text_color),
            range=[
                portfolio_view.portfolio_display_df["Date"].min() - pd.Timedelta(days=90),
                portfolio_view.portfolio_display_df["Date"].max(),
            ],
        ),
        yaxis=dict(gridcolor=pl_grid_color, tickfont=dict(color=pl_text_color)),
        hoverlabel=dict(
            bgcolor=c_hover_bg,
            bordercolor=c_border,
            font=dict(color=c_hover_text, size=13),
        ),
    )
    st.plotly_chart(
        portfolio_fig,
        width="stretch",
        theme=None,
        config={"displayModeBar": False},
        key=f"portfolio_{st.session_state[KEY_THEME_MODE]}_{st.session_state[KEY_CHART_REVISION]}",
    )

    st.markdown(f"#### {portfolio_view.table_title}")
    money_fmt = f"{currency_prefix}{{:,.{display_currency_decimals}f}}{currency_suffix}"
    style_format = {
        f"Fair Price ({currency_unit})": money_fmt,
        f"Portfolio ({currency_unit})": money_fmt,
        portfolio_view.period_change_usd_label: money_fmt,
        portfolio_view.period_change_pct_label: "{:+.2f}%",
    }
    if portfolio_view.dca_enabled:
        style_format[f"Portfolio + monthly buys ({currency_unit})"] = money_fmt
        style_format[f"Invested cash ({currency_unit})"] = money_fmt
        style_format["BTC after monthly buys"] = "{:,.6f}"
    st.dataframe(
        portfolio_view.table_df.style.format(style_format),
        width="stretch",
        hide_index=True,
    )


# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_icon="🚀",
    page_title="BTC Power Law Pro",
    initial_sidebar_state="expanded",
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

try:
    raw_lightning_nodes_df = load_prepared_lightning_nodes_data()
except Exception as e:
    st.error(f"Error loading Lightning node data: {e}")
    st.stop()

try:
    raw_lightning_capacity_df = load_prepared_lightning_capacity_data()
except Exception as e:
    st.error(f"Error loading Lightning capacity data: {e}")
    st.stop()

try:
    raw_liquid_btc_df = load_prepared_liquid_btc_data()
except Exception as e:
    st.error(f"Error loading Liquid BTC data: {e}")
    st.stop()

try:
    raw_liquid_transactions_df = load_prepared_liquid_transactions_data()
except Exception as e:
    st.error(f"Error loading Liquid transactions data: {e}")
    st.stop()

try:
    raw_filecoin_btc_df = load_prepared_filecoin_btc_data()
except Exception as e:
    st.error(f"Error loading Filecoin/BTC data: {e}")
    st.stop()

try:
    raw_monero_btc_df = load_prepared_monero_btc_data()
except Exception as e:
    st.error(f"Error loading Monero/BTC data: {e}")
    st.stop()

try:
    raw_litecoin_btc_df = load_prepared_litecoin_btc_data()
except Exception as e:
    st.error(f"Error loading Litecoin/BTC data: {e}")
    st.stop()

try:
    raw_dogecoin_btc_df = load_prepared_dogecoin_btc_data()
except Exception as e:
    st.error(f"Error loading Dogecoin/BTC data: {e}")
    st.stop()

try:
    raw_us_m2_df = load_prepared_us_m2_data()
except Exception as e:
    st.error(f"Error loading U.S. M2 data: {e}")
    st.stop()

try:
    raw_russian_m2_df = load_prepared_russian_m2_data()
except Exception as e:
    st.error(f"Error loading Russian M2 data: {e}")
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
raw_lightning_nodes_df = raw_lightning_nodes_df[raw_lightning_nodes_df["Close"] > 0].copy()
raw_lightning_nodes_df["LogClose"] = np.log10(raw_lightning_nodes_df["Close"])
raw_lightning_capacity_df = raw_lightning_capacity_df[raw_lightning_capacity_df["Close"] > 0].copy()
raw_lightning_capacity_df["LogClose"] = np.log10(raw_lightning_capacity_df["Close"])
raw_liquid_btc_df = raw_liquid_btc_df[raw_liquid_btc_df["Close"] > 0].copy()
raw_liquid_btc_df["LogClose"] = np.log10(raw_liquid_btc_df["Close"])
raw_liquid_transactions_df = raw_liquid_transactions_df[
    raw_liquid_transactions_df["Close"] > 0
].copy()
raw_liquid_transactions_df["LogClose"] = np.log10(raw_liquid_transactions_df["Close"])
raw_filecoin_btc_df = raw_filecoin_btc_df[raw_filecoin_btc_df["Close"] > 0].copy()
raw_filecoin_btc_df["LogClose"] = np.log10(raw_filecoin_btc_df["Close"])
raw_monero_btc_df = raw_monero_btc_df[raw_monero_btc_df["Close"] > 0].copy()
raw_monero_btc_df["LogClose"] = np.log10(raw_monero_btc_df["Close"])
raw_litecoin_btc_df = raw_litecoin_btc_df[raw_litecoin_btc_df["Close"] > 0].copy()
raw_litecoin_btc_df["LogClose"] = np.log10(raw_litecoin_btc_df["Close"])
raw_dogecoin_btc_df = raw_dogecoin_btc_df[raw_dogecoin_btc_df["Close"] > 0].copy()
raw_dogecoin_btc_df["LogClose"] = np.log10(raw_dogecoin_btc_df["Close"])
raw_us_m2_df = raw_us_m2_df[raw_us_m2_df["Close"] > 0].copy()
raw_us_m2_df["LogClose"] = np.log10(raw_us_m2_df["Close"])
raw_russian_m2_df = raw_russian_m2_df[raw_russian_m2_df["Close"] > 0].copy()
raw_russian_m2_df["LogClose"] = np.log10(raw_russian_m2_df["Close"])
raw_bitcoin_network_simulation_df = build_bitcoin_network_simulation(
    raw_df_usd,
    seed=int(st.session_state.get(KEY_BITCOIN_NETWORK_SIMULATION_SEED, 1)),
    resolution_days=float(st.session_state.get(KEY_BITCOIN_NETWORK_SIMULATION_RESOLUTION, 0.00001)),
)

# Use current session currency for sidebar AF/R2 calculations in PowerLaw Bitcoin mode.
sidebar_currency = st.session_state.get(KEY_CURRENCY_SELECTOR, CURRENCY_DOLLAR)
if sidebar_currency not in [CURRENCY_DOLLAR, CURRENCY_EURO, CURRENCY_GOLD]:
    sidebar_currency = CURRENCY_DOLLAR
sidebar_price_close = build_currency_close_series(raw_df_usd, sidebar_currency)
sidebar_price_close = sidebar_price_close[sidebar_price_close > 0]
sidebar_price_log_close = np.log10(sidebar_price_close.values)

raw_series_frames = {
    POWERLAW_SERIES_PRICE: raw_df_usd,
    POWERLAW_SERIES_REVENUE: raw_revenue_df,
    POWERLAW_SERIES_DIFFICULTY: raw_difficulty_df,
    POWERLAW_SERIES_HASHRATE: raw_hashrate_df,
    POWERLAW_SERIES_BITCOIN_NETWORK_SIMULATION: raw_bitcoin_network_simulation_df,
    POWERLAW_SERIES_LIGHTNING_NODES: raw_lightning_nodes_df,
    POWERLAW_SERIES_LIGHTNING_CAPACITY: raw_lightning_capacity_df,
    POWERLAW_SERIES_LIQUID_BTC: raw_liquid_btc_df,
    POWERLAW_SERIES_LIQUID_TRANSACTIONS: raw_liquid_transactions_df,
    POWERLAW_SERIES_FILECOIN_BTC: raw_filecoin_btc_df,
    POWERLAW_SERIES_MONERO_BTC: raw_monero_btc_df,
    POWERLAW_SERIES_LITECOIN_BTC: raw_litecoin_btc_df,
    POWERLAW_SERIES_DOGECOIN_BTC: raw_dogecoin_btc_df,
    POWERLAW_SERIES_US_M2: raw_us_m2_df,
    POWERLAW_SERIES_RUSSIAN_M2: raw_russian_m2_df,
}
sidebar_series_data = {
    POWERLAW_SERIES_PRICE: {
        "absolute_days": raw_df_usd["AbsDays"].values,
        "log_close": sidebar_price_log_close,
    },
    POWERLAW_SERIES_REVENUE: {
        "absolute_days": raw_revenue_df["AbsDays"].values,
        "log_close": raw_revenue_df["LogClose"].values,
    },
    POWERLAW_SERIES_DIFFICULTY: {
        "absolute_days": raw_difficulty_df["AbsDays"].values,
        "log_close": raw_difficulty_df["LogClose"].values,
    },
    POWERLAW_SERIES_HASHRATE: {
        "absolute_days": raw_hashrate_df["AbsDays"].values,
        "log_close": raw_hashrate_df["LogClose"].values,
    },
    POWERLAW_SERIES_BITCOIN_NETWORK_SIMULATION: {
        "absolute_days": raw_bitcoin_network_simulation_df["AbsDays"].values,
        "log_close": raw_bitcoin_network_simulation_df["LogClose"].values,
    },
    POWERLAW_SERIES_LIGHTNING_NODES: {
        "absolute_days": raw_lightning_nodes_df["AbsDays"].values,
        "log_close": raw_lightning_nodes_df["LogClose"].values,
    },
    POWERLAW_SERIES_LIGHTNING_CAPACITY: {
        "absolute_days": raw_lightning_capacity_df["AbsDays"].values,
        "log_close": raw_lightning_capacity_df["LogClose"].values,
    },
    POWERLAW_SERIES_LIQUID_BTC: {
        "absolute_days": raw_liquid_btc_df["AbsDays"].values,
        "log_close": raw_liquid_btc_df["LogClose"].values,
    },
    POWERLAW_SERIES_LIQUID_TRANSACTIONS: {
        "absolute_days": raw_liquid_transactions_df["AbsDays"].values,
        "log_close": raw_liquid_transactions_df["LogClose"].values,
    },
    POWERLAW_SERIES_FILECOIN_BTC: {
        "absolute_days": raw_filecoin_btc_df["AbsDays"].values,
        "log_close": raw_filecoin_btc_df["LogClose"].values,
    },
    POWERLAW_SERIES_MONERO_BTC: {
        "absolute_days": raw_monero_btc_df["AbsDays"].values,
        "log_close": raw_monero_btc_df["LogClose"].values,
    },
    POWERLAW_SERIES_LITECOIN_BTC: {
        "absolute_days": raw_litecoin_btc_df["AbsDays"].values,
        "log_close": raw_litecoin_btc_df["LogClose"].values,
    },
    POWERLAW_SERIES_DOGECOIN_BTC: {
        "absolute_days": raw_dogecoin_btc_df["AbsDays"].values,
        "log_close": raw_dogecoin_btc_df["LogClose"].values,
    },
    POWERLAW_SERIES_US_M2: {
        "absolute_days": raw_us_m2_df["AbsDays"].values,
        "log_close": raw_us_m2_df["LogClose"].values,
    },
    POWERLAW_SERIES_RUSSIAN_M2: {
        "absolute_days": raw_russian_m2_df["AbsDays"].values,
        "log_close": raw_russian_m2_df["LogClose"].values,
    },
}

# --- THEME + STATE ---
initialize_app_session_state()

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
(
    mode,
    currency,
    time_scale,
    price_scale,
    current_r2,
    powerlaw_series,
    logperiodic_series,
    band_method,
) = render_sidebar_panel(
    sidebar_series_data,
    c_text_main,
    APP_VERSION,
    FORECAST_HORIZON_MIN,
    FORECAST_HORIZON_MAX,
)
active_model = get_active_model_config(mode, powerlaw_series, logperiodic_series, currency)
st.session_state[KEY_A] = float(st.session_state.get(active_model.a_key, active_model.default_a))
st.session_state[KEY_B] = float(st.session_state.get(active_model.b_key, active_model.default_b))

selected_series_name = get_selected_series_name(mode, powerlaw_series, logperiodic_series)
active_series_supports_currency = series_supports_currency_selector(
    mode, powerlaw_series, logperiodic_series
)

if mode == MODE_POWERLAW and (not active_series_supports_currency) and currency != CURRENCY_DOLLAR:
    st.session_state[KEY_CURRENCY_SELECTOR] = CURRENCY_DOLLAR
    st.rerun()
if active_series_supports_currency and currency != st.session_state.get(
    KEY_CURRENCY_SELECTOR, CURRENCY_DOLLAR
):
    st.rerun()
if (
    mode == MODE_LOGPERIODIC
    and (not active_series_supports_currency)
    and currency != CURRENCY_DOLLAR
):
    st.session_state[KEY_CURRENCY_SELECTOR] = CURRENCY_DOLLAR
    st.rerun()

# --- MAIN CALCULATIONS ---
active_model = get_active_model_config(mode, powerlaw_series, logperiodic_series, currency)
session_genesis_offset = int(st.session_state.get(KEY_GENESIS_OFFSET, 0))
genesis_offset = (
    int(active_model.model_origin_abs_day)
    if active_model.model_origin_abs_day is not None
    else session_genesis_offset
)
current_gen_date = GENESIS_DATE + pd.Timedelta(days=genesis_offset)
if active_model.supports_currency_selector:
    raw_df = raw_df_usd.copy()
    raw_df["Close"] = build_currency_close_series(raw_df_usd, currency)
    raw_df = raw_df[raw_df["Close"] > 0].copy()
    raw_df["LogClose"] = np.log10(raw_df["Close"])
else:
    raw_df = raw_series_frames[selected_series_name].copy()

active_abs_days = raw_df["AbsDays"].values
active_a_key = active_model.a_key
active_b_key = active_model.b_key
active_default_a = active_model.default_a
active_default_b = active_model.default_b
target_series_name = active_model.target_series_name
target_series_unit = active_model.target_series_unit

if not active_model.supports_currency_selector:
    currency = CURRENCY_DOLLAR

valid_idx = active_abs_days > genesis_offset
if active_model.analysis_min_abs_day is not None:
    valid_idx = valid_idx & (active_abs_days >= int(active_model.analysis_min_abs_day))
df_display = raw_df.iloc[valid_idx].copy()
if df_display.empty:
    st.error("No data available for the selected parameters.")
    st.stop()

df_display["Days"] = df_display["AbsDays"] - genesis_offset
df_display["LogD"] = np.log10(df_display["Days"])
trend_result = resolve_trend_parameters(
    df_display["LogD"].values,
    df_display["LogClose"].values,
    intercept_a=float(st.session_state.get(active_a_key, active_default_a)),
    slope_b=float(st.session_state.get(active_b_key, active_default_b)),
    active_mode=mode,
)
a_active = trend_result.intercept_a
b_active = trend_result.slope_b
df_display["ModelLog"] = trend_result.trend_log_prices
df_display["Res"] = trend_result.residual_series
df_display["Fair"], _, fair_was_clipped = evaluate_powerlaw_values(
    df_display["ModelLog"].values,
    0.0,
    1.0,
)

currency_prefix = active_model.currency_prefix
currency_suffix = active_model.currency_suffix
currency_decimals = int(active_model.currency_decimals)
currency_unit = active_model.currency_unit
df_display["CloseDisplay"] = df_display["Close"]
df_display["FairDisplay"] = df_display["Fair"]

if mode in [MODE_POWERLAW, MODE_PORTFOLIO] and powerlaw_parameters_are_unstable(
    current_r2, was_clipped=fair_was_clipped
):
    st.warning(
        "Current PowerLaw parameters are unstable for the selected series. Use Auto-fit model or Reset parameters."
    )

# Use a shared LogPeriodic R² mask so scoring follows the same visible segment.
lp_r2_mask = np.ones(len(df_display), dtype=bool)
if mode == MODE_LOGPERIODIC and active_model.oscillator_min_abs_day is not None:
    lp_r2_mask = df_display["AbsDays"].values >= active_model.oscillator_min_abs_day

# Calculate R2 for Trend if not returned by sidebar (LogPeriodic mode)
if mode == MODE_LOGPERIODIC:
    if np.count_nonzero(lp_r2_mask) > 1:
        current_r2 = calculate_r2_score(
            df_display["LogClose"].values[lp_r2_mask],
            df_display["ModelLog"].values[lp_r2_mask],
        )
    else:
        current_r2 = 0.0

if mode == MODE_POWERLAW and band_method == BAND_METHOD_GAUSSIAN:
    p2_5, p16_5, p83_5, p97_5 = calculate_gaussian_offsets(df_display, genesis_offset)
else:
    p2_5, p16_5, p83_5, p97_5 = calculate_percentile_offsets(df_display, genesis_offset)

# --- OSCILLATOR CALC ---
osc_settings = oscillator.OscillatorSettings(
    t1_age=float(st.session_state.get("t1_age", OSC_DEFAULTS["t1_age"])),
    lambda_val=float(st.session_state.get("lambda_val", OSC_DEFAULTS["lambda_val"])),
    amp_factor_top=float(st.session_state.get("amp_factor_top", OSC_DEFAULTS["amp_factor_top"])),
    amp_factor_bottom=float(
        st.session_state.get("amp_factor_bottom", OSC_DEFAULTS["amp_factor_bottom"])
    ),
    impulse_damping=float(st.session_state.get("impulse_damping", OSC_DEFAULTS["impulse_damping"])),
)
osc_amp, osc_omega, osc_phi = 0.0, 0.0, 0.0
r2_combined = current_r2
osc_reference_log_day = float(df_display["LogD"].min())

if mode == MODE_LOGPERIODIC:
    try:
        osc_result = oscillator.compute_oscillator_overlay(
            df_display["LogD"].values,
            df_display["Res"].values,
            df_display["ModelLog"].values,
            df_display["LogClose"].values,
            lp_r2_mask,
            osc_settings,
            current_r2,
        )
        osc_settings = osc_result.settings
        osc_amp = osc_result.amplitude
        osc_omega = osc_result.angular_frequency
        osc_phi = osc_result.phase_shift
        r2_combined = osc_result.combined_r2
        osc_reference_log_day = osc_result.reference_log_day
    except Exception as e:
        st.error(f"LogPeriodic Error: {e}")
        osc_settings = oscillator.OscillatorSettings(
            t1_age=OSC_DEFAULTS["t1_age"],
            lambda_val=OSC_DEFAULTS["lambda_val"],
            amp_factor_top=OSC_DEFAULTS["amp_factor_top"],
            amp_factor_bottom=OSC_DEFAULTS["amp_factor_bottom"],
            impulse_damping=OSC_DEFAULTS["impulse_damping"],
        )
        osc_amp, osc_omega, osc_phi, r2_combined = 0, 0, 0, current_r2

# --- VIZ SETUP ---
view_max = _resolve_model_view_max(df_display, current_gen_date)

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
    osc_settings.amp_factor_top,
    osc_settings.amp_factor_bottom,
    osc_settings.impulse_damping,
    osc_reference_log_day,
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
        band_method=band_method,
        osc_t1_age=osc_settings.t1_age,
        osc_lambda=osc_settings.lambda_val,
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
        show_halving_lines=mode == MODE_POWERLAW and active_model.show_halving_lines,
        osc_visible_start_abs_day=(
            active_model.oscillator_min_abs_day if mode == MODE_LOGPERIODIC else None
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
        current_r2,
        fair_was_clipped,
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
