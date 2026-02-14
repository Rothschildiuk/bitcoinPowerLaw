import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ssl

# SSL Fix for data downloading
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Page Configuration
st.set_page_config(layout="wide", page_title="BTC Power Law Pro")

# --- EXCLUSIVE CSS FOR PREMIUM LOOK AND FULL HEIGHT ---
st.markdown("""
    <style>
    /* Main background and fonts */
    .main { background-color: #0e1117; }
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    
    /* Hide unnecessary Streamlit elements */
    header, footer, [data-testid="stHeader"] {visibility: hidden; display: none;}

    /* Sidebar Styling (Trading Style) */
    [data-testid="stSidebar"] {
        width: 320px !important;
        background-color: #161a25 !important;
        border-right: 1px solid #2d323e;
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.8rem !important;
        padding-top: 1.5rem !important;
    }

    /* Custom Metric Cards (KPI) - ALIGNMENT */
    .metric-card {
        background: #1e222d;
        border: 1px solid #2d323e;
        border-radius: 8px;
        padding: 10px 16px; /* Slightly less vertical padding because of flex */
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-top: 5px;
        
        /* Fix height and center */
        min-height: 110px; 
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .metric-label { color: #848e9c; font-size: 13px; font-weight: 500; margin-bottom: 4px; }
    .metric-value { color: #ffffff; font-size: 22px; font-weight: 700; }
    .metric-delta { font-size: 13px; font-weight: 600; margin-top: 2px; }

    /* Control buttons */
    .stButton > button {
        width: 100% !important;
        background-color: #2d323e !important;
        color: #d1d4dc !important;
        border: none !important;
        border-radius: 4px !important;
        height: 32px !important;
        line-height: 32px !important;
        font-size: 14px !important;
        transition: 0.2s;
        font-weight: bold !important;
    }
    .stButton > button:hover {
        background-color: #3d4251 !important;
        color: #ffffff !important;
    }

    /* Sliders */
    .stSlider { margin-bottom: -5px !important; margin-top: 5px !important; }
    
    [data-testid="stSidebar"] .stMarkdown p {
        font-size: 1rem !important;
        color: #d1d4dc;
        margin-bottom: 5px !important;
        font-weight: 500;
    }
    
    /* Radio buttons */
    .stRadio label {
        font-size: 0.9rem !important;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_data
def load_raw_data():
    # Early data
    url = "https://raw.githubusercontent.com/Habrador/Bitcoin-price-visualization/main/Bitcoin-price-USD.csv"
    early_df = pd.read_csv(url)
    early_df['Date'] = pd.to_datetime(early_df['Date'])
    early_df.set_index('Date', inplace=True)
    early_df.rename(columns={'Price': 'Close'}, inplace=True)

    # Yahoo Finance data
    recent_df = yf.download('BTC-USD', start='2014-09-17')
    if isinstance(recent_df.columns, pd.MultiIndex):
        recent_df = recent_df['Close']
    else:
        recent_df = recent_df[['Close']]
    recent_df.columns = ['Close']
    recent_df = recent_df.dropna()
    recent_df.index = pd.to_datetime(recent_df.index).tz_localize(None)

    # Merge
    early_df = early_df[early_df.index < '2014-09-17']
    return pd.concat([early_df, recent_df])


try:
    raw_df = load_raw_data()
except Exception as e:
    st.error(f"Loading error: {e}")
    st.stop()


# --- OFFSET AUTO-TUNING ALGORITHM ---
# This function finds the offset that yields the max R2 by iterating through days
@st.cache_data
def find_best_fit_params(df_in):
    best_r2_val = -1
    best_off_val = 150
    best_A_val = -17
    best_B_val = 5.8

    # Search in the range around the July date (July 26 ~ 204 days from genesis)
    # Range 180-230 days covers Summer-Autumn 2009
    for off in range(180, 230):
        gen_test = pd.to_datetime('2009-01-03') + pd.Timedelta(days=off)
        df_test = df_in[df_in.index > gen_test].copy()

        d_vals = (df_test.index - gen_test).days.values
        p_vals = df_test['Close'].values
        valid_mask = (d_vals > 0) & (p_vals > 0)

        if np.sum(valid_mask) < 100: continue

        log_d = np.log10(d_vals[valid_mask])
        log_p = np.log10(p_vals[valid_mask])

        slope, intercept = np.polyfit(log_d, log_p, 1)

        # R2 Calculation
        y_pred = slope * log_d + intercept
        ss_res = np.sum((log_p - y_pred) ** 2)
        ss_tot = np.sum((log_p - np.mean(log_p)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        if r2 > best_r2_val:
            best_r2_val = r2
            best_off_val = off
            best_A_val = intercept
            best_B_val = slope

    return best_off_val, best_A_val, best_B_val, best_r2_val


# Perform calculation once at startup
opt_offset, opt_A, opt_B, opt_R2 = find_best_fit_params(raw_df)

# === PARAMETER INITIALIZATION ===
# Set found optimal values as defaults
if 'genesis_offset' not in st.session_state: st.session_state.genesis_offset = int(opt_offset)
if 'A' not in st.session_state: st.session_state.A = float(round(opt_A, 3))
if 'B' not in st.session_state: st.session_state.B = float(round(opt_B, 3))

if 't1_age' not in st.session_state: st.session_state.t1_age = 1.88
if 'lambda_val' not in st.session_state: st.session_state.lambda_val = 2.12


def update_param(param_name, delta):
    st.session_state[param_name] = round(st.session_state[param_name] + delta, 3)


# === SIDEBAR ===
st.sidebar.markdown(
    "<h2 style='text-align: center; color: #f0b90b; margin-bottom: 20px; font-size: 1.8rem;'>BITCOIN MODEL</h2>",
    unsafe_allow_html=True)

c_v1, c_v2 = st.sidebar.columns(2)
with c_v1: price_scale = st.radio("Price", ["Log", "Lin"], index=0)
with c_v2: time_scale = st.radio("Time", ["Log", "Lin"], index=0)

# Now we display the R2 corresponding to CURRENT slider settings,
# but defaults are the max.
st.sidebar.markdown(
    f"<p style='color:gray; text-align:center; font-size: 0.85rem; margin-top: 10px; margin-bottom: 20px;'>Max R² (auto): {opt_R2 * 100:.3f}%<br>(offset {opt_offset}d)</p>",
    unsafe_allow_html=True)


def fancy_control(label, key, step, min_v, max_v):
    st.sidebar.markdown(f"**{label}**")
    c1, c2, c3 = st.sidebar.columns([1, 2, 1])
    with c1:
        if st.button("➖", key=f"{key}_m"): update_param(key, -step)
    with c3:
        if st.button("➕", key=f"{key}_p"): update_param(key, step)
    with c2:
        return st.slider(label, min_v, max_v, key=key, step=step, label_visibility="collapsed")


A = fancy_control("A (Intercept)", "A", 0.01, -25.0, 0.0)
B = fancy_control("B (Slope)", "B", 0.01, 1.0, 10.0)
genesis_offset = fancy_control("Genesis Offset", "genesis_offset", 1, -1000, 1000)  # 1 day step for precision
t1_age = fancy_control("1st Cycle Age", "t1_age", 0.01, 1.0, 5.0)
lambda_val = fancy_control("Lambda", "lambda_val", 0.01, 1.5, 3.0)

# === MODEL CALCULATIONS (LIVE) ===
df = raw_df.copy()
gen_date = pd.to_datetime('2009-01-03') + pd.Timedelta(days=genesis_offset)
df = df[df.index > gen_date].copy()
df['Days'] = (df.index - gen_date).days
df['LogP'], df['LogD'] = np.log10(df['Close']), np.log10(df['Days'])

df['ModelLog'] = A + B * df['LogD']
df['Res'] = df['LogP'] - df['ModelLog']
df['Fair'] = 10 ** df['ModelLog']

# Current R² for display
ss_res = np.sum(df['Res'] ** 2)
ss_tot = np.sum((df['LogP'] - np.mean(df['LogP'])) ** 2)
current_r2 = (1 - (ss_res / ss_tot)) * 100

# Percentiles for bands (recalculated dynamically based on current model)
res_vals = df['LogP'] - (A + B * df['LogD'])
p2_5, p16_5, p83_5, p97_5 = np.percentile(res_vals, [2.5, 16.5, 83.5, 97.5])

# Trend lines
view_max = df['Days'].max() + 365 * 1.5
m_x = np.logspace(0, np.log10(view_max), 400) if time_scale == "Log" else np.linspace(1, view_max, 400)
m_log_d = np.log10(m_x)
m_fair_log = A + B * m_log_d
m_fair_usd = 10 ** m_fair_log
m_bbl, m_acc = 10 ** (m_fair_log + p97_5), 10 ** (m_fair_log + p2_5)

# --- CHART (Optimized height) ---
# Increased vertical_spacing to 0.10 to free up space for legend
fig = make_subplots(rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.10, row_heights=[0.76, 0.24])

# Zones
fig.add_trace(go.Scatter(x=m_x, y=m_bbl, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1,
              col=1)
fig.add_trace(go.Scatter(x=m_x, y=10 ** (m_fair_log + p83_5), mode='lines', line=dict(width=0), fill='tonexty',
                         fillcolor='rgba(234, 61, 47, 0.15)', name='Bubble'), row=1, col=1)
fig.add_trace(go.Scatter(x=m_x, y=10 ** (m_fair_log + p16_5), mode='lines', line=dict(width=0), showlegend=False,
                         hoverinfo='skip'), row=1, col=1)
fig.add_trace(
    go.Scatter(x=m_x, y=m_acc, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(14, 203, 129, 0.15)',
               name='Accumulation'), row=1, col=1)

# Price lines
fig.add_trace(go.Scatter(x=m_x, y=m_fair_usd, mode='lines', line=dict(color='#f0b90b', width=1.5, dash='dash'),
                         name='Fair Value'), row=1, col=1)
fig.add_trace(
    go.Scatter(x=df['Days'], y=df['Close'], mode='lines', name='BTC Price', line=dict(color='#ffffff', width=1.3)),
    row=1, col=1)

t_vals = [(pd.Timestamp(f'{y}-01-01') - gen_date).days for y in range(gen_date.year + 1, 2028) if
          (pd.Timestamp(f'{y}-01-01') - gen_date).days > 0]
t_text = [str(y) for y in range(gen_date.year + 1, 2028) if (pd.Timestamp(f'{y}-01-01') - gen_date).days > 0]
xr = [np.log10(t_vals[0]), np.log10(view_max)] if time_scale == "Log" else [0, view_max]

fig.update_xaxes(type="log" if time_scale == "Log" else "linear", tickvals=t_vals, ticktext=t_text, range=xr,
                 gridcolor='#1e222d', row=1, col=1)
fig.update_yaxes(type="log" if price_scale == "Log" else "linear",
                 range=[np.log10(0.01), np.log10(df['Close'].max() * 8)] if price_scale == "Log" else None,
                 gridcolor='#1e222d', row=1, col=1)

# Oscillator
fig.add_trace(
    go.Scatter(x=df['Days'], y=df['Res'], mode='lines', name='Oscillator', line=dict(color='#0ecb81', width=1.2)), row=2,
    col=1)
fig.add_hline(y=0, line_width=1, line_color="#848e9c", row=2, col=1)
for i in range(6):
    fig.add_vline(x=t1_age * (lambda_val ** i) * 365.25, line_width=1, line_dash="dash", line_color="#ea3d2f",
                  opacity=0.3, row=2, col=1)
    fig.add_vline(x=t1_age * (lambda_val ** (i + 0.5)) * 365.25, line_width=0.8, line_dash="dot", line_color="#2b6aff",
                  opacity=0.2, row=2, col=1)

fig.update_xaxes(type="log" if time_scale == "Log" else "linear", tickvals=t_vals, ticktext=t_text, range=xr,
                 gridcolor='#1e222d', row=2, col=1)

# Moved legend to y=0.27 (between charts) and increased font size to 15
fig.update_layout(height=720, margin=dict(t=30, b=10, l=50, r=20), template="plotly_dark",
                  legend=dict(orientation="h", y=0.27, x=0.5, xanchor="center", font=dict(size=15, color="#848e9c")),
                  paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# === KPI DASHBOARD ===
l_p, l_f = df['Close'].iloc[-1], df['Fair'].iloc[-1]
diff = ((l_p - l_f) / l_f) * 100
pot = ((m_bbl[-1] - l_p) / l_p) * 100
diff_color = "#0ecb81" if diff < 0 else "#ea3d2f"

k1, k2, k3, k4 = st.columns(4)


def kpi_card(col, label, value, delta=None, d_color=None):
    # If no delta, insert invisible placeholder to keep height and alignment
    if delta:
        delta_html = f"<div class='metric-delta' style='color:{d_color}'>{delta}</div>"
    else:
        delta_html = f"<div class='metric-delta' style='visibility:hidden;'>-</div>"

    col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>
    """, unsafe_allow_html=True)


kpi_card(k1, "BTC PRICE", f"${l_p:,.0f}")
kpi_card(k2, "FAIR VALUE", f"${l_f:,.0f}", f"{diff:+.1f}% from model", diff_color)
kpi_card(k3, "MODEL FIT (R²)", f"{current_r2:.2f}%")
kpi_card(k4, "GROWTH POTENTIAL", f"+{pot:,.0f}%", "to top band", "#f0b90b")