import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import ssl

# --- MODULE IMPORTS ---
import powerLaw
import oscillator

# --- SSL Fix ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# --- Page Configuration ---
st.set_page_config(layout="wide", page_icon="🚀", page_title="BTC Power Law Pro", initial_sidebar_state="expanded")

# --- THEME STATE ---
if 'theme_mode' not in st.session_state:
    st.session_state.theme_mode = "Dark 🌑"

is_dark = "Dark" in st.session_state.theme_mode

# --- DYNAMIC COLORS ---
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
    
    [data-testId="stSidebar"] {{ width: 290px !important; background-color: {c_sidebar_bg} !important; border-right: 1px solid {c_border}; }}
    [data-testId="stSidebarContent"] {{ overflow-x: hidden !important; }}
    [data-testId="stSidebar"] [data-testId="stVerticalBlock"] {{ gap: 0.4rem !important; padding-top: 0.5rem !important; }}
    [data-testId="stSidebar"] p, [data-testId="stSidebar"] span, [data-testId="stSidebar"] label {{ color: {c_text_main} !important; font-size: 15px !important; }}

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

# --- DATA LOADING ---
@st.cache_data
def load_and_prep_data():
    url = "https://raw.githubusercontent.com/Habrador/Bitcoin-price-visualization/main/Bitcoin-price-USD.csv"
    early_df = pd.read_csv(url)
    early_df['Date'] = pd.to_datetime(early_df['Date'])
    early_df.set_index('Date', inplace=True)
    early_df.rename(columns={'Price': 'Close'}, inplace=True)

    recent_df = yf.download('BTC-USD', start='2014-09-17', progress=False)
    recent_df = recent_df['Close'] if isinstance(recent_df.columns, pd.MultiIndex) else recent_df[['Close']]
    recent_df.columns = ['Close']
    recent_df = recent_df.dropna()
    recent_df.index = pd.to_datetime(recent_df.index).tz_localize(None)

    early_df = early_df[early_df.index < '2014-09-17']
    full_df = pd.concat([early_df, recent_df])

    # FILTER INVALID PRICES TO PREVENT LOG ERRORS
    full_df = full_df[full_df['Close'] > 0]

    genesis_static = pd.to_datetime('2009-01-03')
    full_df['AbsDays'] = (full_df.index - genesis_static).days
    full_df['LogClose'] = np.log10(full_df['Close'])
    return full_df

try:
    raw_df = load_and_prep_data()
    ALL_ABS_DAYS = raw_df['AbsDays'].values
    ALL_LOG_CLOSE = raw_df['LogClose'].values
except Exception as e:
    st.error(f"Error loading data: {e}"); st.stop()

# --- INITIALIZATION ---
# Ensure A and B are initialized before Sidebar Assembly because Oscillator mode needs them
if "A" not in st.session_state or "B" not in st.session_state:
    try:
        _, opt_a, opt_b, _ = powerLaw.find_global_best_fit_optimized(ALL_ABS_DAYS, ALL_LOG_CLOSE)
        if "A" not in st.session_state: st.session_state["A"] = float(round(opt_a, 3))
        if "B" not in st.session_state: st.session_state["B"] = float(round(opt_b, 3))
    except Exception:
        # Fallback defaults if calculation fails
        if "A" not in st.session_state: st.session_state["A"] = -17.0
        if "B" not in st.session_state: st.session_state["B"] = 5.8

# --- SIDEBAR ASSEMBLY ---
with st.sidebar:
    st.markdown(f"<div class='sidebar-title'>BTC MODEL</div>", unsafe_allow_html=True)

    # NAVIGATION (Mode Switcher)
    mode = st.radio("Mode", ["Power Law", "Oscillator"], horizontal=True, label_visibility="collapsed")

    # GLOBAL TIME SCALE (Moved from powerLaw.py)
    time_scale = st.radio("Time", ["Log", "Lin"], index=0, horizontal=True)

    current_r2 = 0.0
    price_scale = "Log" # Default

    if mode == "Power Law":
        # Updated unpacking: removed time_scale
        price_scale, current_r2 = powerLaw.render_sidebar(ALL_ABS_DAYS, ALL_LOG_CLOSE, c_text_main)
    else:
        # Oscillator Mode
        oscillator.render_sidebar(ALL_ABS_DAYS, ALL_LOG_CLOSE, c_text_main)

    st.markdown("<hr style='margin: 10px 0 5px 0; opacity:0.1;'>", unsafe_allow_html=True)

    # 3. Theme
    new_theme = st.radio("Theme", ["Dark 🌑", "Light ☀️"], index=0 if "Dark" in st.session_state.theme_mode else 1, horizontal=True)
    if new_theme != st.session_state.theme_mode:
        st.session_state.theme_mode = new_theme
        st.rerun()

# --- MAIN CALCULATIONS ---
gen_date_static = pd.to_datetime('2009-01-03')
current_gen_date = gen_date_static + pd.Timedelta(days=st.session_state.get("genesis_offset", 0))

valid_idx = ALL_ABS_DAYS > st.session_state.get("genesis_offset", 0)
df_display = raw_df.iloc[valid_idx].copy()

df_display['Days'] = df_display['AbsDays'] - st.session_state.get("genesis_offset", 0)
df_display['LogD'] = np.log10(df_display['Days'])
df_display['ModelLog'] = st.session_state.A + st.session_state.B * df_display['LogD']
df_display['Res'] = df_display['LogClose'] - df_display['ModelLog']
df_display['Fair'] = 10 ** df_display['ModelLog']

# Calculate R2 for Trend if not returned by sidebar (Oscillator mode)
if mode == "Oscillator":
    ss_res = np.sum(df_display['Res'] ** 2)
    ss_tot = np.sum((df_display['LogClose'] - np.mean(df_display['LogClose'])) ** 2)
    current_r2 = 1 - (ss_res / ss_tot)

p2_5, p16_5, p97_5 = np.percentile(df_display['Res'], [2.5, 16.5, 97.5])
# Added p83_5 explicitly for Bubble calculation if needed,
# though originally it was unpacking 4 values. Checking original code...
# Original was: p2_5, p16_5, p83_5, p97_5 = np.percentile(...)
p2_5, p16_5, p83_5, p97_5 = np.percentile(df_display['Res'], [2.5, 16.5, 83.5, 97.5])

# --- OSCILLATOR CALC ---
try:
    lambda_log = np.log10(st.session_state.get("lambda_val", 1.94))
    t1_days_log = np.log10(st.session_state.get("t1_age", 2.53) * 365.25)
    osc_omega = 2 * np.pi / lambda_log
    osc_phi = -osc_omega * t1_days_log

    full_phase = osc_omega * df_display['LogD'] + osc_phi

    # NEW: USE OSCILLATOR (SINE/COSINE) WAVE
    unit_wave = oscillator.get_oscillator_wave(full_phase)

    numerator = np.dot(df_display['Res'], unit_wave)
    denominator = np.dot(unit_wave, unit_wave)

    # FORCE POSITIVE AMPLITUDE (Peaks always up)
    osc_amp = abs(numerator / denominator) if denominator > 1e-9 else 0

    osc_model_vals = osc_amp * unit_wave
    osc_model_vals = np.where(unit_wave > 0, osc_model_vals * st.session_state.get("amp_factor_top", 1.12), osc_model_vals)
    osc_model_vals = np.where(unit_wave < 0, osc_model_vals * st.session_state.get("amp_factor_bottom", 0.84), osc_model_vals)

    total_model_log = df_display['ModelLog'] + osc_model_vals
    ss_res_total = np.sum((df_display['LogClose'] - total_model_log) ** 2)
    ss_tot = np.sum((df_display['LogClose'] - np.mean(df_display['LogClose'])) ** 2)
    r2_combined = (1 - (ss_res_total / ss_tot)) * 100
except Exception as e:
    st.error(f"Oscillator Error: {e}")
    osc_amp, osc_omega, osc_phi, r2_combined = 0, 0, 0, current_r2

# --- VIZ SETUP ---
view_max = df_display['Days'].max() + 365 * 10
m_x = np.logspace(0, np.log10(view_max), 400) if time_scale == "Log" else np.linspace(1, view_max, 400)
m_dates = [current_gen_date + pd.Timedelta(days=float(d)) for d in m_x]
m_log_d = np.log10(m_x)
m_fair_usd = 10 ** (st.session_state.A + st.session_state.B * m_log_d)

m_osc_y = oscillator.oscillator_func_manual(
    m_log_d, osc_amp, osc_omega, osc_phi,
    st.session_state.get("amp_factor_top", 1.12), st.session_state.get("amp_factor_bottom", 0.84)
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

# --- PLOTTING ---
fig = go.Figure()

if mode == "Power Law":
    # --- CHART 1: POWER LAW ---
    fig.add_trace(go.Scatter(x=plot_x_model, y=10 ** (st.session_state.A + st.session_state.B * m_log_d + p97_5), mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=plot_x_model, y=10 ** (st.session_state.A + st.session_state.B * m_log_d + p16_5), mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))

    fig.add_trace(go.Scatter(
        x=plot_x_model, y=10 ** (st.session_state.A + st.session_state.B * m_log_d + p2_5),
        mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(14, 203, 129, 0.15)',
        name="Accumulation", customdata=m_dates_str, hovertemplate=f"<b>Accumulation</b>: $%{{y:,.0f}}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=plot_x_model, y=m_fair_usd,
        mode='lines', line=dict(color='#f0b90b', width=1.5, dash='dash'),
        name="Fair Value", customdata=m_dates_str, hovertemplate=f"<b>Fair Value</b>: $%{{y:,.0f}}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=plot_x_model, y=10 ** (st.session_state.A + st.session_state.B * m_log_d + p83_5),
        mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(234, 61, 47, 0.15)',
        name="Bubble", customdata=m_dates_str, hovertemplate=f"<b>Bubble</b>: $%{{y:,.0f}}<extra></extra>"
    ))

    btc_hover = f"📅 %{{customdata}}<br><b>BTC Price</b>: $%{{y:,.0f}}<extra></extra>" if is_log_time else f"<b>BTC Price</b>: $%{{y:,.0f}}<extra></extra>"
    fig.add_trace(go.Scatter(
        x=plot_x_main, y=df_display['Close'], mode='lines', name="BTC Price", line=dict(color=pl_btc_color, width=1.5),
        customdata=df_display.index.strftime('%d.%m.%Y'), hovertemplate=btc_hover
    ))

    fig.update_yaxes(
        type="log" if price_scale == "Log" else "linear",
        range=[np.log10(0.01), np.log10(df_display['Close'].max() * 8)] if price_scale == "Log" else None,
        gridcolor=pl_grid_color, tickfont=dict(color=pl_text_color, size=14, family="Arial Black, sans-serif")
    )

elif mode == "Oscillator":
    # --- CHART 2: OSCILLATOR ---
    fig.add_trace(go.Scatter(
        x=plot_x_osc, y=df_display['Res'], mode='lines', name="Oscillator", line=dict(color='#0ecb81', width=1.2),
        customdata=df_display.index.strftime('%d.%m.%Y'), hovertemplate=f"<b>Oscillator</b>: %{{y:.3f}}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(x=plot_x_model, y=m_osc_y, mode='lines', name='Osc Model', line=dict(color='#ea3d2f', width=2), hoverinfo='skip'))
    fig.add_hline(y=0, line_width=1, line_color=pl_legend_color)

    # Halving lines (useful context)
    for i in range(6):
        halving_days_val = st.session_state.get("t1_age", 2.53) * (st.session_state.get("lambda_val", 1.94) ** i) * 365.25
        if is_log_time:
            hv_x = halving_days_val
        else:
            hv_x = current_gen_date + pd.Timedelta(days=halving_days_val)

        fig.add_vline(x=hv_x, line_width=1.5, line_dash="dash", line_color="#ea3d2f", opacity=0.8)

# Common Layout
t_vals = [(pd.Timestamp(f'{y}-01-01') - current_gen_date).days for y in range(current_gen_date.year + 1, 2036) if (pd.Timestamp(f'{y}-01-01') - current_gen_date).days > 0]
t_text = [str(y) for y in range(current_gen_date.year + 1, 2036) if (pd.Timestamp(f'{y}-01-01') - current_gen_date).days > 0]

if is_log_time:
    fig.update_xaxes(type="log", tickvals=t_vals, ticktext=t_text, range=[np.log10(t_vals[0]), np.log10(view_max)], gridcolor=pl_grid_color, tickfont=dict(color=pl_text_color, size=14, family="Arial Black, sans-serif"))
else:
    fig.update_xaxes(type="date", gridcolor=pl_grid_color, tickfont=dict(color=pl_text_color, size=14, family="Arial Black, sans-serif"), range=[df_display.index.min(), m_dates[-1]], hoverformat="%d.%m.%Y")

fig.update_layout(
    height=600, margin=dict(t=30, b=10, l=50, r=20), template=pl_template, font=dict(color=pl_text_color),
    legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center", font=dict(size=14, color=pl_legend_color), bgcolor="rgba(0,0,0,0)"),
    paper_bgcolor=pl_bg_color, plot_bgcolor=pl_bg_color, hovermode='x unified',
    hoverlabel=dict(bgcolor=c_hover_bg, bordercolor=c_border, font=dict(color=c_hover_text, size=13))
)

st.plotly_chart(
    fig,
    width='stretch',
    theme=None,
    config={'displayModeBar': False},
    key=f"chart_{mode}_{st.session_state.theme_mode}"
)

# --- KPI ---
l_p, l_f = df_display['Close'].iloc[-1], df_display['Fair'].iloc[-1]
diff = ((l_p - l_f) / l_f) * 100
pot_target = 10 ** (st.session_state.A + st.session_state.B * np.log10(df_display['Days'].max()) + p97_5)
pot = ((pot_target - l_p) / l_p) * 100

k1, k2, k3 = st.columns(3)

def kpi_card(col, label, value, delta=None, d_color=None):
    delta_html = f"<div class='metric-delta' style='color:{d_color}'>{delta}</div>" if delta else "<div class='metric-delta' style='visibility:hidden;'>-</div>"
    col.markdown(f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div>{delta_html}</div>", unsafe_allow_html=True)

kpi_card(k1, "BTC PRICE", f"${l_p:,.0f}")
kpi_card(k2, "FAIR VALUE", f"${l_f:,.0f}", f"{diff:+.1f}% from model", "#0ecb81" if diff < 0 else "#ea3d2f")
kpi_card(k3, "GROWTH POTENTIAL", f"+{pot:,.0f}%", "to top band", "#f0b90b")