import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ssl

# --- SSL Fix for data downloading ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_icon="🚀",
    page_title="BTC Power Law Pro",
    initial_sidebar_state="expanded"
)

# --- THEME STATE MANAGEMENT ---
if 'theme_mode' not in st.session_state:
    st.session_state.theme_mode = "Dark 🌑"

is_dark = "Dark" in st.session_state.theme_mode

# --- DYNAMIC COLOR PALETTE ---
if is_dark:
    c_main_bg, c_sidebar_bg, c_card_bg = "#0e1117", "#161a25", "#1e222d"
    c_border, c_text_main, c_text_val = "#2d323e", "#d1d4dc", "#ffffff"
    c_btn_bg, c_btn_hover, c_btn_text = "#2d323e", "#3d4251", "#d1d4dc"
    pl_template, pl_bg_color, pl_grid_color = "plotly_dark", "rgba(0,0,0,0)", "#1e222d"
    pl_btc_color, pl_legend_color, pl_text_color = "#ffffff", "#848e9c", "#848e9c"
    c_hover_bg, c_hover_text = "#1e222d", "#ffffff"
else:
    c_main_bg, c_sidebar_bg, c_card_bg = "#ffffff", "#f4f4f4", "#f8f9fa"
    c_border, c_text_main, c_text_val = "#d0d0d0", "#000000", "#000000"
    c_btn_bg, c_btn_hover, c_btn_text = "#ffffff", "#e0e0e0", "#000000"
    pl_template, pl_bg_color, pl_grid_color = "plotly_white", "rgba(255,255,255,1)", "#e6e6e6"
    pl_btc_color, pl_legend_color, pl_text_color = "#000000", "#000000", "#000000"
    c_hover_bg, c_hover_text = "#ffffff", "#000000"

# --- CSS ---
st.markdown(f"""
    <style>
    .stApp, [data-testId="stAppViewContainer"] {{ background-color: {c_main_bg} !important; }}
    .block-container {{ padding-top: 1rem !important; padding-bottom: 0rem !important; padding-left: 1rem !important; padding-right: 1rem !important; }}
    footer {{visibility: hidden; display: none;}}
    [data-testId="stHeader"] {{ background-color: transparent; color: {c_text_main}; }}
    [data-testId="stHeader"] button {{ color: {c_text_main} !important; }}
    [data-testId="stAppDeployButton"] {{ display: none !important; }}
    #MainMenu {{ visibility: hidden !important; }}
    
    [data-testId="stSidebar"] {{ 
        width: 290px !important; 
        background-color: {c_sidebar_bg} !important; 
        border-right: 1px solid {c_border};
    }}
    [data-testId="stSidebarContent"] {{ overflow-x: hidden !important; }}
    [data-testId="stSidebar"] [data-testId="stVerticalBlock"] {{ gap: 0.4rem !important; padding-top: 0.5rem !important; }}
    [data-testId="stSidebar"] p, [data-testId="stSidebar"] span, [data-testId="stSidebar"] label {{ color: {c_text_main} !important; font-size: 13px !important; }}

    .metric-card {{
        background: {c_card_bg}; border: 1px solid {c_border}; border-radius: 8px; padding: 10px 16px; text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-top: 5px; min-height: 110px; display: flex; flex-direction: column; justify-content: center; align-items: center;
    }}
    .metric-label {{ color: {c_text_main}; font-size: 13px; font-weight: 600; margin-bottom: 4px; opacity: 0.8; }}
    .metric-value {{ color: {c_text_val}; font-size: 20px; font-weight: 800; }}
    .metric-delta {{ font-size: 12px; font-weight: 600; margin-top: 2px; }}

    div[data-testId="stSidebar"] .stRadio div[role="radiogroup"] {{ gap: 8px; }}
    div[data-testId="stSidebar"] .stRadio label p {{ font-size: 13px !important; }}
    
    .stButton > button {{
        width: 100% !important; background-color: {c_btn_bg} !important; color: {c_btn_text} !important;
        border: 1px solid {c_border} !important; border-radius: 4px !important; height: 28px !important;
        line-height: 26px !important; font-size: 13px !important; transition: 0.2s; font-weight: bold !important;
    }}
    .stSlider {{ margin-bottom: -10px !important; margin-top: 0px !important; }}
    
    .sidebar-title {{
        text-align: center; color: #f0b90b; margin-bottom: 5px !important; font-size: 1.5rem; font-weight: bold;
    }}
    </style>
    """, unsafe_allow_html=True)


# --- DATA LOADING (OPTIMIZED) ---
@st.cache_data
def load_and_prep_data():
    url = "https://raw.githubusercontent.com/Habrador/Bitcoin-price-visualization/main/Bitcoin-price-USD.csv"
    early_df = pd.read_csv(url)
    early_df['Date'] = pd.to_datetime(early_df['Date'])
    early_df.set_index('Date', inplace=True)
    early_df.rename(columns={'Price': 'Close'}, inplace=True)

    # Download latest data
    recent_df = yf.download('BTC-USD', start='2014-09-17', progress=False)
    recent_df = recent_df['Close'] if isinstance(recent_df.columns, pd.MultiIndex) else recent_df[['Close']]
    recent_df.columns = ['Close']
    recent_df = recent_df.dropna()
    recent_df.index = pd.to_datetime(recent_df.index).tz_localize(None)

    early_df = early_df[early_df.index < '2014-09-17']
    full_df = pd.concat([early_df, recent_df])

    # --- PERFORMANCE OPTIMIZATION: Pre-calculate static arrays ---
    genesis_static = pd.to_datetime('2009-01-03')
    full_df['AbsDays'] = (full_df.index - genesis_static).days
    full_df['LogClose'] = np.log10(full_df['Close'])

    return full_df


try:
    raw_df = load_and_prep_data()
    ALL_ABS_DAYS = raw_df['AbsDays'].values
    ALL_LOG_CLOSE = raw_df['LogClose'].values
except Exception as e:
    st.error(f"Error loading data: {e}");
    st.stop()


# --- MATH CORE ---
def calculate_regression_numpy(abs_days_array, log_price_array, offset_value):
    x_days = abs_days_array - offset_value
    mask = x_days > 0

    if np.sum(mask) < 100:
        return 0.0, 0.0, 0.0

    x_valid = x_days[mask]
    y_valid = log_price_array[mask]
    log_x = np.log10(x_valid)

    slope, intercept = np.polyfit(log_x, y_valid, 1)

    y_pred = slope * log_x + intercept
    ss_res = np.sum((y_valid - y_pred) ** 2)
    ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return slope, intercept, r2


@st.cache_data
def find_global_best_fit_optimized():
    b, a, r2 = calculate_regression_numpy(ALL_ABS_DAYS, ALL_LOG_CLOSE, 0)
    return 0, a, b, r2


opt_offset, opt_a_ideal, opt_b_ideal, opt_r2_ideal = find_global_best_fit_optimized()

if "genesis_offset" not in st.session_state:
    st.session_state["genesis_offset"] = 0
if "A" not in st.session_state:
    st.session_state["A"] = float(round(opt_a_ideal, 3))
if "B" not in st.session_state:
    st.session_state["B"] = float(round(opt_b_ideal, 3))

for k, v in {"t1_age": 2.53, "lambda_val": 1.94, "amp_factor_top": 1.12, "amp_factor_bottom": 0.84}.items():
    if k not in st.session_state: st.session_state[k] = v


def update_param(param_name, delta):
    st.session_state[param_name] = round(st.session_state[param_name] + delta, 3)


# --- CYCLOID MATH ---
def get_cycloid_wave(phase_array):
    """
    Calculates an Inverted Cycloid wave (Sharp peaks, smooth valleys).
    Input: phase_array (linear phase).
    Output: Normalized wave values centered around 0.
    """
    # Normalize phase to [-pi, pi] to handle periodicity
    # We want phase=0 to correspond to the Cusp (Peak).
    x_norm = (phase_array + np.pi) % (2 * np.pi) - np.pi
    x_abs = np.abs(x_norm)

    # Solve Kepler-like equation: theta - sin(theta) = x
    # Initial guess:
    # For small x (near cusp), theta ~ (6x)^(1/3)
    # For large x (near valley), theta ~ x
    theta = np.where(x_abs < 1.0, np.power(6 * x_abs, 1/3), x_abs)

    # Newton's method (5 iterations is enough for high precision)
    for _ in range(5):
        f = theta - np.sin(theta) - x_abs
        df = 1 - np.cos(theta)
        # Avoid division by zero at the cusp (theta=0)
        df = np.where(df < 1e-5, 1e-5, df)
        theta = theta - f / df

    # Inverted Cycloid shape:
    # Standard cycloid y = 1 - cos(theta) (Cusp at bottom)
    # Inverted: y = 1 + cos(theta) (Cusp at top, Peak value 2, Valley value 0)
    y_vals = 1 + np.cos(theta)

    # Remove DC component (Mean of cycloid over 2pi is 0.5 * Peak = 1.0?
    # Area = integral(1+cos t)(1-cos t)dt = pi. Mean = pi/2pi = 0.5.
    # Wait, integral y dx. y=1+cos(theta). dx = (1-cos(theta)) dtheta.
    # Int (1+cos)(1-cos) = Int (1 - cos^2) = Int sin^2 = pi.
    # Mean = Area / Length(2pi) = 0.5.
    # So we subtract 0.5 to center it.

    return y_vals - 0.5


# --- SIDEBAR UI ---
with st.sidebar:
    st.markdown(f"<div class='sidebar-title'>BTC MODEL</div>", unsafe_allow_html=True)

    c_v1, c_v2 = st.columns(2)
    price_scale = c_v1.radio("Price", ["Log", "Lin"], index=0, horizontal=True)
    time_scale = c_v2.radio("Time", ["Log", "Lin"], index=0, horizontal=True)

    auto_fit = st.checkbox("Auto-Fit A & B", value=False, help="Automatically calculate best Slope (B) and Intercept (A) when Offset changes.")

    if auto_fit:
        curr_off = st.session_state.get("genesis_offset", opt_offset)
        calc_b, calc_a, calc_r2 = calculate_regression_numpy(ALL_ABS_DAYS, ALL_LOG_CLOSE, curr_off)
        st.session_state["A"] = float(round(calc_a, 3))
        st.session_state["B"] = float(round(calc_b, 3))
        display_r2 = calc_r2
    else:
        _, _, display_r2 = calculate_regression_numpy(ALL_ABS_DAYS, ALL_LOG_CLOSE, st.session_state.genesis_offset)


    def fancy_control(label, key, step, min_v, max_v, disabled=False):
        st.markdown(f"**{label}**")
        c1, c2, c3 = st.columns([1, 2.5, 1])
        if c1.button("➖", key=f"{key}_m", disabled=disabled): update_param(key, -step)
        if c3.button("➕", key=f"{key}_p", disabled=disabled): update_param(key, step)
        return c2.slider(key, min_v, max_v, key=key, step=step, label_visibility="collapsed", disabled=disabled)


    a_slider = fancy_control("A (Intercept)", "A", 0.01, -25.0, 0.0, disabled=auto_fit)
    b_slider = fancy_control("B (Slope)", "B", 0.01, 1.0, 7.0, disabled=auto_fit)

    st.markdown(
        f"<p style='color:{c_text_main}; text-align:center; font-size: 0.75rem; margin-top: 2px; opacity: 0.7;'>"
        f"Current R²: {display_r2 * 100:.4f}%</p>",
        unsafe_allow_html=True)

    t1_age_slider = fancy_control("1st Cycle Age", "t1_age", 0.01, 0.1, 5.0)
    lambda_slider = fancy_control("Lambda", "lambda_val", 0.01, 1.5, 3.0)

    # --- NEW: Two Amplitude Sliders (Top & Bottom) ---
    lbl_amp_top = "Top Amplitude"
    lbl_amp_bot = "Bottom Amplitude"

    amp_top_slider = fancy_control(lbl_amp_top, "amp_factor_top", 0.01, 0.1, 10.0)
    amp_bot_slider = fancy_control(lbl_amp_bot, "amp_factor_bottom", 0.01, 0.1, 10.0)

    # --- NEW: Oscillator R2 Calculation & Display in Sidebar ---
    # We calculate the R2 of the Oscillator (Red line) vs Residuals (Price in Oscillator) locally here
    calc_days = ALL_ABS_DAYS - st.session_state.genesis_offset
    mask_calc = calc_days > 0

    osc_r2_display = 0.0
    if np.sum(mask_calc) > 100:
        c_log_d = np.log10(calc_days[mask_calc])
        # Re-calc Trend and Residuals for current A/B
        c_model_log = st.session_state.A + st.session_state.B * c_log_d
        c_res = ALL_LOG_CLOSE[mask_calc] - c_model_log

        # Calc Oscillator Params based on current Sliders
        l_log = np.log10(st.session_state.lambda_val)
        t1_log = np.log10(st.session_state.t1_age * 365.25)
        osc_omega = 2 * np.pi / l_log

        # Phi (Phase): Align CUSP (Peak) with t1.
        # Cycloid peak is at phase = 0.
        # So omega * t1_log + phi = 0  =>  phi = -omega * t1_log
        osc_phi = -osc_omega * t1_log

        # Generate Cycloid Wave
        # Phase for calculation
        calc_phase = osc_omega * c_log_d + osc_phi
        u_wave = get_cycloid_wave(calc_phase)

        # Fit Amplitude
        num = np.dot(c_res, u_wave)
        den = np.dot(u_wave, u_wave)
        amp = num / den if den != 0 else 0

        # --- APPLY DUAL FACTORS ---
        # Scale the wave shape itself based on polarity
        # u_wave > 0 is the "peak" part (cusp), u_wave < 0 is the "valley" part
        c_osc_pred = amp * u_wave
        c_osc_pred = np.where(u_wave > 0, c_osc_pred * st.session_state.amp_factor_top, c_osc_pred)
        c_osc_pred = np.where(u_wave < 0, c_osc_pred * st.session_state.amp_factor_bottom, c_osc_pred)

        # Calculate R2 of Red Line (pred) vs Green Line (c_res)
        ss_res_osc = np.sum((c_res - c_osc_pred) ** 2)
        ss_tot_osc = np.sum((c_res - np.mean(c_res)) ** 2)
        osc_r2_display = (1 - (ss_res_osc / ss_tot_osc)) * 100

    st.markdown(
        f"<p style='color:{c_text_main}; text-align:center; font-size: 0.75rem; margin-top: 2px; opacity: 0.7;'>"
        f"Oscillator R²: {osc_r2_display:.4f}%</p>",
        unsafe_allow_html=True)
    # -----------------------------------------------------------

    st.markdown("<hr style='margin: 10px 0 5px 0; opacity:0.1;'>", unsafe_allow_html=True)

    # --- UPDATED: Removed Language Switcher, Only Theme Switcher Remains ---
    new_theme = st.radio("Theme", ["Dark 🌑", "Light ☀️"],
                         index=0 if "Dark" in st.session_state.theme_mode else 1, horizontal=True)
    if new_theme != st.session_state.theme_mode:
        st.session_state.theme_mode = new_theme
        st.rerun()

# --- MAIN CALCULATIONS ---
gen_date_static = pd.to_datetime('2009-01-03')
current_gen_date = gen_date_static + pd.Timedelta(days=st.session_state.genesis_offset)

valid_idx = ALL_ABS_DAYS > st.session_state.genesis_offset
df_display = raw_df.iloc[valid_idx].copy()

df_display['Days'] = df_display['AbsDays'] - st.session_state.genesis_offset
df_display['LogD'] = np.log10(df_display['Days'])
df_display['ModelLog'] = st.session_state.A + st.session_state.B * df_display['LogD']
df_display['Res'] = df_display['LogClose'] - df_display['ModelLog']
df_display['Fair'] = 10 ** df_display['ModelLog']

p2_5, p16_5, p83_5, p97_5 = np.percentile(df_display['Res'], [2.5, 16.5, 83.5, 97.5])
current_r2 = (1 - (np.sum(df_display['Res'] ** 2) / np.sum(
    (df_display['LogClose'] - np.mean(df_display['LogClose'])) ** 2))) * 100

# --- NEW: Oscillator Controlled by Sidebar Parameters ---
# Replaces curve_fit with manual Cycloid model.

try:
    lambda_log = np.log10(st.session_state.lambda_val)
    t1_days_log = np.log10(st.session_state.t1_age * 365.25)

    # Omega (Frequency)
    osc_omega = 2 * np.pi / lambda_log

    # Phi (Phase): Peak at t1
    osc_phi = -osc_omega * t1_days_log

    # Generate Unit Wave (Inverted Cycloid)
    full_phase = osc_omega * df_display['LogD'] + osc_phi
    unit_wave = get_cycloid_wave(full_phase)

    # Fit Amplitude analytically
    numerator = np.dot(df_display['Res'], unit_wave)
    denominator = np.dot(unit_wave, unit_wave)
    osc_amp = numerator / denominator if denominator != 0 else 0

    # --- APPLY DUAL FACTORS ---
    # Not creating separate variables here just to calc R2_combined properly
    # We apply the same logic as in sidebar for the final combined fit metric
    osc_model_vals = osc_amp * unit_wave
    osc_model_vals = np.where(unit_wave > 0, osc_model_vals * st.session_state.amp_factor_top, osc_model_vals)
    osc_model_vals = np.where(unit_wave < 0, osc_model_vals * st.session_state.amp_factor_bottom, osc_model_vals)

    # 3. Calculate Combined R2
    total_model_log = df_display['ModelLog'] + osc_model_vals

    ss_res_total = np.sum((df_display['LogClose'] - total_model_log) ** 2)
    ss_tot = np.sum((df_display['LogClose'] - np.mean(df_display['LogClose'])) ** 2)
    r2_combined = (1 - (ss_res_total / ss_tot)) * 100

except Exception as e:
    osc_amp, osc_omega, osc_phi = 0, 0, 0
    r2_combined = current_r2

# Function for plotting later (wraps the cycloid gen with dual amp)
def cycloid_func_manual(x_log, amp, omega, phi, f_top, f_bot):
    phase = omega * x_log + phi
    base_wave = get_cycloid_wave(phase)
    y = amp * base_wave
    y = np.where(base_wave > 0, y * f_top, y)
    y = np.where(base_wave < 0, y * f_bot, y)
    return y

# --- VIZ ---
view_max = df_display['Days'].max() + 365 * 10
m_x = np.logspace(0, np.log10(view_max), 400) if time_scale == "Log" else np.linspace(1, view_max, 400)
m_dates = [current_gen_date + pd.Timedelta(days=float(d)) for d in m_x]
m_log_d = np.log10(m_x)
m_fair_usd = 10 ** (st.session_state.A + st.session_state.B * m_log_d)

# Calculate Oscillator Model Line (Red Line) using CYCLOID with DUAL AMPS
m_osc_y = cycloid_func_manual(
    m_log_d, osc_amp, osc_omega, osc_phi,
    st.session_state.amp_factor_top, st.session_state.amp_factor_bottom
)

is_log_time = (time_scale == "Log")
plot_x_model = m_x if is_log_time else m_dates
plot_x_main = df_display['Days'] if is_log_time else df_display.index
plot_x_osc = df_display['Days'] if is_log_time else df_display.index

m_dates_str = [
    d.strftime('%d.%m.%Y') if isinstance(d, pd.Timestamp) else (
            current_gen_date + pd.Timedelta(days=float(d))).strftime('%d.%m.%Y')
    for d in m_x
]

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.75, 0.25])

fig.add_trace(
    go.Scatter(x=plot_x_model, y=10 ** (st.session_state.A + st.session_state.B * m_log_d + p97_5), mode='lines',
               line=dict(width=0), showlegend=False, hoverinfo='skip'), 1, 1)


fig.add_trace(
    go.Scatter(x=plot_x_model, y=10 ** (st.session_state.A + st.session_state.B * m_log_d + p16_5), mode='lines',
               line=dict(width=0), showlegend=False, hoverinfo='skip'), 1, 1)

fig.add_trace(go.Scatter(
    x=plot_x_model, y=10 ** (st.session_state.A + st.session_state.B * m_log_d + p2_5),
    mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(14, 203, 129, 0.15)',
    name="Accumulation", customdata=m_dates_str,
    hovertemplate=f"<b>Accumulation</b>: $%{{y:,.0f}}<extra></extra>"
), 1, 1)

fig.add_trace(go.Scatter(
    x=plot_x_model, y=m_fair_usd,
    mode='lines', line=dict(color='#f0b90b', width=1.5, dash='dash'),
    name="Fair Value", customdata=m_dates_str,
    hovertemplate=f"<b>Fair Value</b>: $%{{y:,.0f}}<extra></extra>"
), 1, 1)

fig.add_trace(go.Scatter(
    x=plot_x_model, y=10 ** (st.session_state.A + st.session_state.B * m_log_d + p83_5),
    mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(234, 61, 47, 0.15)',
    name="Bubble", customdata=m_dates_str,
    hovertemplate=f"<b>Bubble</b>: $%{{y:,.0f}}<extra></extra>"
), 1, 1)

btc_hover = f"📅 %{{customdata}}<br><b>BTC Price</b>: $%{{y:,.0f}}<extra></extra>" if is_log_time else f"<b>BTC Price</b>: $%{{y:,.0f}}<extra></extra>"
fig.add_trace(go.Scatter(
    x=plot_x_main, y=df_display['Close'],
    mode='lines', name="BTC Price",
    line=dict(color=pl_btc_color, width=1.5),
    customdata=df_display.index.strftime('%d.%m.%Y'),
    hovertemplate=btc_hover
), 1, 1)

t_vals = [(pd.Timestamp(f'{y}-01-01') - current_gen_date).days for y in range(current_gen_date.year + 1, 2036) if
          (pd.Timestamp(f'{y}-01-01') - current_gen_date).days > 0]
t_text = [str(y) for y in range(current_gen_date.year + 1, 2036) if
          (pd.Timestamp(f'{y}-01-01') - current_gen_date).days > 0]

for r in [1, 2]:
    if is_log_time:
        fig.update_xaxes(
            type="log", tickvals=t_vals, ticktext=t_text,
            range=[np.log10(t_vals[0]), np.log10(view_max)],
            gridcolor=pl_grid_color, tickfont=dict(color=pl_text_color, size=14, family="Arial Black, sans-serif"),
            row=r, col=1
        )
    else:
        fig.update_xaxes(
            type="date", gridcolor=pl_grid_color,
            tickfont=dict(color=pl_text_color, size=14, family="Arial Black, sans-serif"),
            range=[df_display.index.min(), m_dates[-1]],
            hoverformat="%d.%m.%Y", row=r, col=1
        )

fig.update_yaxes(
    type="log" if price_scale == "Log" else "linear",
    range=[np.log10(0.01), np.log10(df_display['Close'].max() * 8)] if price_scale == "Log" else None,
    gridcolor=pl_grid_color, tickfont=dict(color=pl_text_color, size=14, family="Arial Black, sans-serif"), row=1, col=1
)

fig.add_trace(go.Scatter(
    x=plot_x_osc, y=df_display['Res'],
    mode='lines', name="Oscillator",
    line=dict(color='#0ecb81', width=1.2),
    customdata=df_display.index.strftime('%d.%m.%Y'),
    hovertemplate=f"<b>Oscillator</b>: %{{y:.3f}}<extra></extra>"
), 2, 1)

# --- NEW: Add the red oscillator model line to the bottom chart ---
fig.add_trace(go.Scatter(
    x=plot_x_model, y=m_osc_y,
    mode='lines', name='Osc Model',
    line=dict(color='#ea3d2f', width=2),
    hoverinfo='skip'
), 2, 1)

fig.add_hline(y=0, line_width=1, line_color=pl_legend_color, row=2, col=1)

for i in range(6):
    halving_days_val = st.session_state.t1_age * (st.session_state.lambda_val ** i) * 365.25
    halving_days_mid = st.session_state.t1_age * (st.session_state.lambda_val ** (i + 0.5)) * 365.25

    if is_log_time:
        hv_x1, hv_x2 = halving_days_val, halving_days_mid
    else:
        hv_x1 = current_gen_date + pd.Timedelta(days=halving_days_val)
        hv_x2 = current_gen_date + pd.Timedelta(days=halving_days_mid)

    fig.add_vline(x=hv_x1, line_width=1.5, line_dash="dash", line_color="#ea3d2f", opacity=0.8, row=2, col=1)
    fig.add_vline(x=hv_x2, line_width=1, line_dash="dot", line_color="#2b6aff", opacity=0.5, row=2, col=1)

fig.update_layout(
    height=720, margin=dict(t=30, b=10, l=50, r=20), template=pl_template,
    font=dict(color=pl_text_color),
    legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center", font=dict(size=14, color=pl_legend_color),
                bgcolor="rgba(0,0,0,0)"),
    paper_bgcolor=pl_bg_color, plot_bgcolor=pl_bg_color, hovermode='x unified',
    hoverlabel=dict(bgcolor=c_hover_bg, bordercolor=c_border, font=dict(color=c_hover_text, size=13))
)

st.plotly_chart(fig, width='stretch', theme=None, config={'displayModeBar': False})

# --- KPI ---
l_p, l_f = df_display['Close'].iloc[-1], df_display['Fair'].iloc[-1]
diff = ((l_p - l_f) / l_f) * 100
pot_target = 10 ** (st.session_state.A + st.session_state.B * np.log10(df_display['Days'].max()) + p97_5)
pot = ((pot_target - l_p) / l_p) * 100

k1, k2, k3, k4 = st.columns(4)


def kpi_card(col, label, value, delta=None, d_color=None):
    delta_html = f"<div class='metric-delta' style='color:{d_color}'>{delta}</div>" if delta else "<div class='metric-delta' style='visibility:hidden;'>-</div>"
    col.markdown(
        f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div>{delta_html}</div>",
        unsafe_allow_html=True)


kpi_card(k1, "BTC PRICE", f"${l_p:,.0f}")
kpi_card(k2, "FAIR VALUE", f"${l_f:,.0f}", f"{diff:+.1f}% from model", "#0ecb81" if diff < 0 else "#ea3d2f")

# --- UPDATED: Show Combined R2 (Trend + Oscillator) and Trend R2 separately ---
kpi_card(k3, "MODEL FIT (R²)", f"{r2_combined:.4f}%", f"Trend: {current_r2:.2f}%", "#f0b90b")

kpi_card(k4, "GROWTH POTENTIAL", f"+{pot:,.0f}%", "to top band", "#f0b90b")