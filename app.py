import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

import oscillator

# --- MODULE IMPORTS ---
import powerLaw
from ui_theme import get_theme, apply_theme_css
from utils import calculate_r2_score, get_stable_trend_fit

GENESIS_DATE = pd.to_datetime("2009-01-03")
DEFAULT_A = -17.0
DEFAULT_B = 5.8
OSC_DEFAULTS = {
    "lambda_val": 1.94,
    "t1_age": 2.53,
    "amp_factor_top": 1.12,
    "amp_factor_bottom": 0.84,
    "impulse_damping": 0.0,
}


def initialize_app_session_state(absolute_days=None, log_prices=None):
    defaults = {
        "theme_mode": "Dark 🌑",
        "last_mode": "Power Law",
        "chart_revision": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if "A" not in st.session_state or "B" not in st.session_state:
        if absolute_days is not None and log_prices is not None:
            try:
                _, opt_a, opt_b, _ = powerLaw.find_best_fit_params(absolute_days, log_prices)
                if "A" not in st.session_state:
                    st.session_state["A"] = float(round(opt_a, 3))
                if "B" not in st.session_state:
                    st.session_state["B"] = float(round(opt_b, 3))
            except Exception:
                if "A" not in st.session_state:
                    st.session_state["A"] = DEFAULT_A
                if "B" not in st.session_state:
                    st.session_state["B"] = DEFAULT_B
        else:
            if "A" not in st.session_state:
                st.session_state["A"] = DEFAULT_A
            if "B" not in st.session_state:
                st.session_state["B"] = DEFAULT_B


def resolve_trend_parameters(absolute_days, log_prices, display_df, active_mode):
    intercept_a = float(st.session_state.get("A", DEFAULT_A))
    slope_b = float(st.session_state.get("B", DEFAULT_B))
    trend_log_prices = intercept_a + slope_b * display_df["LogD"].values
    residual_series = display_df["LogClose"].values - trend_log_prices

    # In oscillator mode, fallback to best-fit trend when session A/B is clearly invalid.
    # This prevents "price-like" residuals after mode toggles or stale widget state.
    if active_mode == "Oscillator":
        intercept_a, slope_b, trend_log_prices, residual_series = get_stable_trend_fit(
            display_df["LogD"].values,
            display_df["LogClose"].values,
            intercept_a,
            slope_b,
        )

    return intercept_a, slope_b, trend_log_prices, residual_series


# --- Page Configuration ---
st.set_page_config(
    layout="wide", page_icon="🚀", page_title="BTC Power Law Pro", initial_sidebar_state="expanded"
)


# --- DATA LOADING ---
@st.cache_data(ttl=3600)
def load_prepared_price_data():
    url = "https://raw.githubusercontent.com/Habrador/Bitcoin-price-visualization/main/Bitcoin-price-USD.csv"
    early_df = pd.read_csv(url)
    early_df["Date"] = pd.to_datetime(early_df["Date"])
    early_df.set_index("Date", inplace=True)
    early_df.rename(columns={"Price": "Close"}, inplace=True)

    recent_df = yf.download("BTC-USD", start="2014-09-17", progress=False)
    recent_df = (
        recent_df["Close"] if isinstance(recent_df.columns, pd.MultiIndex) else recent_df[["Close"]]
    )
    recent_df.columns = ["Close"]
    recent_df = recent_df.dropna()
    recent_df.index = pd.to_datetime(recent_df.index).tz_localize(None)

    early_df = early_df[early_df.index < "2014-09-17"]
    full_df = pd.concat([early_df, recent_df])

    # FILTER INVALID PRICES TO PREVENT LOG ERRORS
    full_df = full_df[full_df["Close"] > 0]

    full_df["AbsDays"] = (full_df.index - GENESIS_DATE).days
    full_df["LogClose"] = np.log10(full_df["Close"])
    return full_df


try:
    raw_df = load_prepared_price_data()
    all_absolute_days = raw_df["AbsDays"].values
    all_log_close_prices = raw_df["LogClose"].values
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# --- THEME + STATE ---
initialize_app_session_state(all_absolute_days, all_log_close_prices)

is_dark = "Dark" in st.session_state.theme_mode

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
with st.sidebar:
    st.markdown(f"<div class='sidebar-title'>BTC MODEL</div>", unsafe_allow_html=True)

    # NAVIGATION (Mode Switcher)
    mode = st.radio(
        "Mode", ["Power Law", "Oscillator"], horizontal=True, label_visibility="collapsed"
    )
    if mode != st.session_state.last_mode:
        st.session_state.chart_revision += 1
        st.session_state.last_mode = mode

    # GLOBAL TIME SCALE (Moved from powerLaw.py)
    time_scale = st.radio("Time", ["Log", "Lin"], index=0, horizontal=True)

    current_r2 = 0.0
    price_scale = "Log"  # Default

    if mode == "Power Law":
        # Updated unpacking: removed time_scale
        price_scale, current_r2 = powerLaw.render_sidebar(
            all_absolute_days, all_log_close_prices, c_text_main
        )
    else:
        # Oscillator Mode
        oscillator.render_sidebar(all_absolute_days, all_log_close_prices, c_text_main)

    st.markdown("<hr style='margin: 10px 0 5px 0; opacity:0.1;'>", unsafe_allow_html=True)

    # 3. Theme
    new_theme = st.radio(
        "Theme",
        ["Dark 🌑", "Light ☀️"],
        index=0 if "Dark" in st.session_state.theme_mode else 1,
        horizontal=True,
    )
    if new_theme != st.session_state.theme_mode:
        st.session_state.theme_mode = new_theme
        st.rerun()

# --- MAIN CALCULATIONS ---
genesis_offset = int(st.session_state.get("genesis_offset", 0))
current_gen_date = GENESIS_DATE + pd.Timedelta(days=genesis_offset)

valid_idx = all_absolute_days > genesis_offset
df_display = raw_df.iloc[valid_idx].copy()
if df_display.empty:
    st.error("No data available for the selected parameters.")
    st.stop()

df_display["Days"] = df_display["AbsDays"] - genesis_offset
df_display["LogD"] = np.log10(df_display["Days"])
a_active, b_active, model_log_vals, residual_vals = resolve_trend_parameters(
    all_absolute_days, all_log_close_prices, df_display, mode
)
df_display["ModelLog"] = model_log_vals
df_display["Res"] = residual_vals
df_display["Fair"] = 10 ** df_display["ModelLog"]

# Calculate R2 for Trend if not returned by sidebar (Oscillator mode)
if mode == "Oscillator":
    current_r2 = calculate_r2_score(df_display["LogClose"].values, df_display["ModelLog"].values)

p2_5, p16_5, p83_5, p97_5 = np.percentile(df_display["Res"], [2.5, 16.5, 83.5, 97.5])

# --- OSCILLATOR CALC ---
try:
    osc_lambda = float(st.session_state.get("lambda_val", OSC_DEFAULTS["lambda_val"]))
    osc_t1_age = float(st.session_state.get("t1_age", OSC_DEFAULTS["t1_age"]))
    osc_amp_top = float(st.session_state.get("amp_factor_top", OSC_DEFAULTS["amp_factor_top"]))
    osc_amp_bottom = float(
        st.session_state.get("amp_factor_bottom", OSC_DEFAULTS["amp_factor_bottom"])
    )
    osc_damping = float(st.session_state.get("impulse_damping", OSC_DEFAULTS["impulse_damping"]))

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
    st.error(f"Oscillator Error: {e}")
    osc_t1_age = OSC_DEFAULTS["t1_age"]
    osc_lambda = OSC_DEFAULTS["lambda_val"]
    osc_amp_top = OSC_DEFAULTS["amp_factor_top"]
    osc_amp_bottom = OSC_DEFAULTS["amp_factor_bottom"]
    osc_damping = OSC_DEFAULTS["impulse_damping"]
    osc_amp, osc_omega, osc_phi, r2_combined = 0, 0, 0, current_r2

# --- VIZ SETUP ---
view_max = df_display["Days"].max() + 365 * 10

# Use daily grid so unified hover has matching x-values across all traces.
m_x = np.arange(1.0, float(np.ceil(view_max)) + 1.0)
m_dates = [current_gen_date + pd.Timedelta(days=float(d)) for d in m_x]
m_log_d = np.log10(m_x)
m_fair_usd = 10 ** (a_active + b_active * m_log_d)

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

is_log_time = time_scale == "Log"
plot_x_model = m_x if is_log_time else m_dates
plot_x_main = df_display["Days"] if is_log_time else df_display.index
plot_x_osc = df_display["Days"] if is_log_time else df_display.index

m_dates_str = [
    (
        d.strftime("%d.%m.%Y")
        if isinstance(d, pd.Timestamp)
        else (current_gen_date + pd.Timedelta(days=float(d))).strftime("%d.%m.%Y")
    )
    for d in m_x
]

# --- PLOTTING ---
fig = go.Figure()

if mode == "Power Law":
    # --- CHART 1: POWER LAW ---
    fig.add_trace(
        go.Scatter(
            x=plot_x_model,
            y=10 ** (a_active + b_active * m_log_d + p97_5),
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=plot_x_model,
            y=10 ** (a_active + b_active * m_log_d + p16_5),
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=plot_x_model,
            y=10 ** (a_active + b_active * m_log_d + p2_5),
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(14, 203, 129, 0.15)",
            name="Accumulation",
            customdata=m_dates_str,
            hovertemplate=f"<b>Accumulation</b>: $%{{y:,.0f}}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=plot_x_model,
            y=m_fair_usd,
            mode="lines",
            line=dict(color="#f0b90b", width=1.5, dash="dash"),
            name="Fair Value",
            customdata=m_dates_str,
            hovertemplate=f"<b>Fair Value</b>: $%{{y:,.0f}}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=plot_x_model,
            y=10 ** (a_active + b_active * m_log_d + p83_5),
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(234, 61, 47, 0.15)",
            name="Bubble",
            customdata=m_dates_str,
            hovertemplate=f"<b>Bubble</b>: $%{{y:,.0f}}<extra></extra>",
        )
    )

    btc_hover = (
        f"📅 %{{customdata}}<br><b>BTC Price</b>: $%{{y:,.0f}}<extra></extra>"
        if is_log_time
        else f"<b>BTC Price</b>: $%{{y:,.0f}}<extra></extra>"
    )
    fig.add_trace(
        go.Scatter(
            x=plot_x_main,
            y=df_display["Close"],
            mode="lines",
            name="BTC Price",
            line=dict(color=pl_btc_color, width=1.5),
            customdata=df_display.index.strftime("%d.%m.%Y"),
            hovertemplate=btc_hover,
        )
    )

    fig.update_yaxes(
        type="log" if price_scale == "Log" else "linear",
        range=(
            [np.log10(0.01), np.log10(df_display["Close"].max() * 8)]
            if price_scale == "Log"
            else None
        ),
        gridcolor=pl_grid_color,
        tickfont=dict(color=pl_text_color, size=14, family="Arial Black, sans-serif"),
    )

elif mode == "Oscillator":
    # --- CHART 2: OSCILLATOR ---
    fig.add_trace(
        go.Scatter(
            x=plot_x_osc,
            y=df_display["Res"],
            mode="lines",
            name="Oscillator",
            line=dict(color="#0ecb81", width=1.2),
            customdata=df_display.index.strftime("%d.%m.%Y"),
            hovertemplate=f"<b>Oscillator</b>: %{{y:.3f}}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=plot_x_model,
            y=m_osc_y,
            mode="lines",
            name="Osc Model",
            line=dict(color="#ea3d2f", width=2),
            hoverinfo="skip",
        )
    )
    fig.add_hline(y=0, line_width=1, line_color=pl_legend_color)
    fig.update_yaxes(
        type="linear",
        gridcolor=pl_grid_color,
        tickfont=dict(color=pl_text_color, size=14, family="Arial Black, sans-serif"),
    )

    # Halving lines (useful context)
    for i in range(6):
        halving_days_val = osc_t1_age * (osc_lambda**i) * 365.25
        if is_log_time:
            hv_x = halving_days_val
        else:
            hv_x = current_gen_date + pd.Timedelta(days=halving_days_val)

        fig.add_vline(x=hv_x, line_width=1.5, line_dash="dash", line_color="#ea3d2f", opacity=0.8)

# Common Layout
t_vals = [
    (pd.Timestamp(f"{y}-01-01") - current_gen_date).days
    for y in range(current_gen_date.year + 1, 2036)
    if (pd.Timestamp(f"{y}-01-01") - current_gen_date).days > 0
]
t_text = [
    str(y)
    for y in range(current_gen_date.year + 1, 2036)
    if (pd.Timestamp(f"{y}-01-01") - current_gen_date).days > 0
]

if is_log_time:
    x_range = [np.log10(max(1.0, view_max / 1000.0)), np.log10(view_max)]
    if t_vals:
        x_range = [np.log10(t_vals[0]), np.log10(view_max)]
    fig.update_xaxes(
        type="log",
        tickvals=t_vals,
        ticktext=t_text,
        range=x_range,
        gridcolor=pl_grid_color,
        tickfont=dict(color=pl_text_color, size=14, family="Arial Black, sans-serif"),
    )
else:
    fig.update_xaxes(
        type="date",
        gridcolor=pl_grid_color,
        tickfont=dict(color=pl_text_color, size=14, family="Arial Black, sans-serif"),
        range=[df_display.index.min(), m_dates[-1]],
        hoverformat="%d.%m.%Y",
    )

fig.update_layout(
    height=600,
    margin=dict(t=30, b=10, l=50, r=20),
    template=pl_template,
    font=dict(color=pl_text_color),
    legend=dict(
        orientation="h",
        y=1.02,
        x=0.5,
        xanchor="center",
        font=dict(size=14, color=pl_legend_color),
        bgcolor="rgba(0,0,0,0)",
    ),
    paper_bgcolor=pl_bg_color,
    plot_bgcolor=pl_bg_color,
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor=c_hover_bg, bordercolor=c_border, font=dict(color=c_hover_text, size=13)
    ),
)

st.plotly_chart(
    fig,
    width="stretch",
    theme=None,
    config={"displayModeBar": False},
    key=f"chart_{mode}_{st.session_state.theme_mode}_{st.session_state.chart_revision}",
)

# --- KPI ---
l_p, l_f = df_display["Close"].iloc[-1], df_display["Fair"].iloc[-1]
diff = ((l_p - l_f) / l_f) * 100
pot_target = 10 ** (a_active + b_active * np.log10(df_display["Days"].max()) + p97_5)
pot = ((pot_target - l_p) / l_p) * 100

k1, k2, k3 = st.columns(3)


def kpi_card(col, label, value, delta=None, d_color=None):
    delta_html = (
        f"<div class='metric-delta' style='color:{d_color}'>{delta}</div>"
        if delta
        else "<div class='metric-delta' style='visibility:hidden;'>-</div>"
    )
    col.markdown(
        f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div>{delta_html}</div>",
        unsafe_allow_html=True,
    )


kpi_card(k1, "BTC PRICE", f"${l_p:,.0f}")
kpi_card(
    k2,
    "FAIR VALUE",
    f"${l_f:,.0f}",
    f"{diff:+.1f}% from model",
    "#0ecb81" if diff < 0 else "#ea3d2f",
)
kpi_card(k3, "GROWTH POTENTIAL", f"+{pot:,.0f}%", "to top band", "#f0b90b")
