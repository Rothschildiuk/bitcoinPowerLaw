import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

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
    FORECAST_HORIZON_MAX,
    FORECAST_HORIZON_MIN,
    GENESIS_DATE,
    KEY_A,
    KEY_B,
    KEY_CHART_REVISION,
    KEY_CURRENCY_SELECTOR,
    KEY_GENESIS_OFFSET,
    KEY_LAST_MODE,
    KEY_PORTFOLIO_BTC_AMOUNT,
    KEY_PORTFOLIO_FORECAST_HORIZON,
    KEY_PORTFOLIO_FORECAST_UNIT,
    KEY_THEME_MODE,
    MODE_LOGPERIODIC,
    MODE_POWERLAW,
    OSC_DEFAULTS,
    TIME_LOG,
)
from core.utils import calculate_r2_score, get_stable_trend_fit
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
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if KEY_A not in st.session_state or KEY_B not in st.session_state:
        if absolute_days is not None and log_prices is not None:
            try:
                _, opt_a, opt_b, _ = power_law.find_best_fit_params(absolute_days, log_prices)
                if KEY_A not in st.session_state:
                    st.session_state[KEY_A] = float(round(opt_a, 3))
                if KEY_B not in st.session_state:
                    st.session_state[KEY_B] = float(round(opt_b, 3))
            except Exception:
                if KEY_A not in st.session_state:
                    st.session_state[KEY_A] = DEFAULT_A
                if KEY_B not in st.session_state:
                    st.session_state[KEY_B] = DEFAULT_B
        else:
            if KEY_A not in st.session_state:
                st.session_state[KEY_A] = DEFAULT_A
            if KEY_B not in st.session_state:
                st.session_state[KEY_B] = DEFAULT_B


def _extract_close_series(download_df):
    if download_df is None or download_df.empty:
        return pd.Series(dtype=float)
    close_series = download_df["Close"]
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.iloc[:, 0]
    close_series = pd.to_numeric(close_series, errors="coerce").dropna()
    if close_series.empty:
        return pd.Series(dtype=float)
    close_series.index = pd.to_datetime(close_series.index).tz_localize(None)
    return close_series.astype(float)


@st.cache_data(ttl=3600)
def load_reference_series(start_date):
    eur_usd = _extract_close_series(yf.download("EURUSD=X", start=start_date, progress=False))
    xau_usd = _extract_close_series(yf.download("XAUUSD=X", start=start_date, progress=False))
    if xau_usd.empty:
        xau_usd = _extract_close_series(yf.download("GC=F", start=start_date, progress=False))
    return eur_usd, xau_usd


def build_currency_close_series(raw_df, selected_currency):
    close_usd = raw_df["Close"].astype(float)
    if selected_currency == CURRENCY_DOLLAR:
        return close_usd

    start_date = str(raw_df.index.min().date())
    eur_usd, xau_usd = load_reference_series(start_date)

    if selected_currency == CURRENCY_EURO and not eur_usd.empty:
        eur_usd_aligned = eur_usd.reindex(close_usd.index).interpolate("time").ffill().bfill()
        return close_usd / eur_usd_aligned

    if selected_currency == CURRENCY_GOLD and not xau_usd.empty:
        xau_usd_aligned = xau_usd.reindex(close_usd.index).interpolate("time").ffill().bfill()
        return close_usd / xau_usd_aligned

    return close_usd


def resolve_currency_format(selected_currency):
    if selected_currency == CURRENCY_EURO:
        return {"prefix": "€", "suffix": "", "decimals": 2, "unit": "EUR"}
    if selected_currency == CURRENCY_GOLD:
        return {"prefix": "", "suffix": " oz", "decimals": 2, "unit": "XAU"}
    return {"prefix": "$", "suffix": "", "decimals": 2, "unit": "USD"}


def format_currency_value(value, prefix, suffix, decimals):
    return f"{prefix}{value:,.{decimals}f}{suffix}"


def resolve_trend_parameters(display_df, active_mode):
    intercept_a = float(st.session_state.get(KEY_A, DEFAULT_A))
    slope_b = float(st.session_state.get(KEY_B, DEFAULT_B))
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
    if forecast_unit == "Year":
        latest_year = int(_df_index.max().year)
        start_period = pd.Timestamp(f"{latest_year - 1}-01-01")
        date_index = pd.date_range(start=start_period, periods=forecast_horizon + 1, freq="YS")
        change_usd_col, change_pct_col = "YoY_USD", "YoY_pct"
        table_title = "Yearly growth table"
    elif forecast_unit == "Day":
        latest_day = _df_index.max().normalize()
        start_period = latest_day - pd.Timedelta(days=1)
        date_index = pd.date_range(start=start_period, periods=forecast_horizon + 1, freq="D")
        change_usd_col, change_pct_col = "DoD_USD", "DoD_pct"
        table_title = "Daily growth table"
    else:
        latest_month_start = _df_index.max().to_period("M").to_timestamp()
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
    st.markdown("### Portfolio Growth (Fair Price / Power Law)")
    btc_amount = float(st.session_state.get(KEY_PORTFOLIO_BTC_AMOUNT, 1.0))
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
        format_currency_value(
            portfolio_display_df["FairPriceDisplay"].iloc[0],
            currency_prefix,
            currency_suffix,
            currency_decimals,
        ),
    )
    g2.metric(
        "Portfolio (end of horizon)",
        format_currency_value(
            portfolio_display_df["PortfolioDisplay"].iloc[-1],
            currency_prefix,
            currency_suffix,
            currency_decimals,
        ),
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
                f"{currency_prefix}%{{y:,.{currency_decimals}f}}{currency_suffix}<extra></extra>"
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
    money_fmt = f"{currency_prefix}{{:,.{currency_decimals}f}}{currency_suffix}"
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


# --- DATA LOADING ---
@st.cache_data(ttl=3600)
def load_prepared_price_data():
    url = "https://raw.githubusercontent.com/Habrador/Bitcoin-price-visualization/main/Bitcoin-price-USD.csv"
    full_df = pd.read_csv(url)
    full_df["Date"] = pd.to_datetime(full_df["Date"])
    full_df.set_index("Date", inplace=True)
    full_df.rename(columns={"Price": "Close"}, inplace=True)
    full_df["Close"] = pd.to_numeric(full_df["Close"], errors="coerce")
    full_df = full_df.dropna(subset=["Close"]).sort_index()

    # FILTER INVALID PRICES TO PREVENT LOG ERRORS
    full_df = full_df[full_df["Close"] > 0]

    full_df["AbsDays"] = (full_df.index - GENESIS_DATE).days
    full_df["LogClose"] = np.log10(full_df["Close"])
    return full_df


try:
    raw_df_usd = load_prepared_price_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

if KEY_CURRENCY_SELECTOR not in st.session_state:
    st.session_state[KEY_CURRENCY_SELECTOR] = CURRENCY_DOLLAR

currency_for_fit = st.session_state.get(KEY_CURRENCY_SELECTOR, CURRENCY_DOLLAR)
raw_df = raw_df_usd.copy()
raw_df["Close"] = build_currency_close_series(raw_df_usd, currency_for_fit)
raw_df = raw_df[raw_df["Close"] > 0]
raw_df["LogClose"] = np.log10(raw_df["Close"])
all_absolute_days = raw_df["AbsDays"].values
all_log_close_prices = raw_df["LogClose"].values

# --- THEME + STATE ---
initialize_app_session_state(all_absolute_days, all_log_close_prices)

is_dark = "Dark" in st.session_state[KEY_THEME_MODE]

theme = get_theme(is_dark)
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
mode, currency, time_scale, price_scale, current_r2 = render_sidebar_panel(
    all_absolute_days,
    all_log_close_prices,
    c_text_main,
    APP_VERSION,
    FORECAST_HORIZON_MIN,
    FORECAST_HORIZON_MAX,
)
if currency != currency_for_fit:
    st.rerun()

# --- MAIN CALCULATIONS ---
genesis_offset = int(st.session_state.get(KEY_GENESIS_OFFSET, 0))
current_gen_date = GENESIS_DATE + pd.Timedelta(days=genesis_offset)

valid_idx = all_absolute_days > genesis_offset
df_display = raw_df.iloc[valid_idx].copy()
if df_display.empty:
    st.error("No data available for the selected parameters.")
    st.stop()

df_display["Days"] = df_display["AbsDays"] - genesis_offset
df_display["LogD"] = np.log10(df_display["Days"])
a_active, b_active, model_log_vals, residual_vals = resolve_trend_parameters(df_display, mode)
df_display["ModelLog"] = model_log_vals
df_display["Res"] = residual_vals
df_display["Fair"] = 10 ** df_display["ModelLog"]

currency_context = resolve_currency_format(currency)
currency_prefix = currency_context["prefix"]
currency_suffix = currency_context["suffix"]
currency_decimals = int(currency_context["decimals"])
currency_unit = currency_context["unit"]
df_display["CloseDisplay"] = df_display["Close"]
df_display["FairDisplay"] = df_display["Fair"]

# Calculate R2 for Trend if not returned by sidebar (LogPeriodic mode)
if mode == MODE_LOGPERIODIC:
    current_r2 = calculate_r2_score(df_display["LogClose"].values, df_display["ModelLog"].values)

p2_5, p16_5, p83_5, p97_5 = np.percentile(df_display["Res"], [2.5, 16.5, 83.5, 97.5])

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
        chart_key=f"chart_{mode}_{st.session_state[KEY_THEME_MODE]}_{st.session_state[KEY_CHART_REVISION]}",
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
)
