import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ssl
# Fix for MAC OS
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="BTC Power Law & Percentiles")
st.markdown("<h3 style='margin-top: -40px; margin-bottom: 0px;'>Bitcoin: Power Law Trend, R² та DSI Осцилятор</h3>", unsafe_allow_html=True)

@st.cache_data
def load_raw_data():
    url = "https://raw.githubusercontent.com/Habrador/Bitcoin-price-visualization/main/Bitcoin-price-USD.csv"
    early_df = pd.read_csv(url)
    early_df['Date'] = pd.to_datetime(early_df['Date'])
    early_df.set_index('Date', inplace=True)
    early_df.rename(columns={'Price': 'Close'}, inplace=True)

    recent_df = yf.download('BTC-USD', start='2014-09-17')
    recent_df = recent_df[['Close']].dropna()
    recent_df.columns = ['Close']
    recent_df.index = pd.to_datetime(recent_df.index).tz_localize(None)

    early_df = early_df[early_df.index < '2014-09-17']
    full_df = pd.concat([early_df, recent_df])
    return full_df

try:
    raw_df = load_raw_data()
except Exception as e:
    st.error(f"Помилка завантаження даних: {e}")
    st.stop()

# --- AUTOFIT MATHEMATICS ---
df_fit = raw_df.copy()
genesis_date_fit = pd.to_datetime('2009-01-03')
df_fit = df_fit[df_fit.index > genesis_date_fit]

days = (df_fit.index - genesis_date_fit).days.values
price = df_fit['Close'].values

log_days = np.log10(days)
log_price = np.log10(price)

B_ideal, A_ideal = np.polyfit(log_days, log_price, 1)

p = np.poly1d([B_ideal, A_ideal])
yhat = p(log_days)
ybar = np.sum(log_price) / len(log_price)
r_squared = (np.sum((yhat - ybar)**2) / np.sum((log_price - ybar)**2)) * 100

perfect_residuals = log_price - yhat
p2_5 = np.percentile(perfect_residuals, 2.5)
p16_5 = np.percentile(perfect_residuals, 16.5)
p83_5 = np.percentile(perfect_residuals, 83.5)
p97_5 = np.percentile(perfect_residuals, 97.5)

# === SETTINGS PANEL (SIDEBAR) ===
st.sidebar.header("1. Базовий тренд (Power Law)")
st.sidebar.success(f"**Автопідгонка (Ідеал):**\n\nR² (Точність): **{r_squared:.2f}%**")

A = st.sidebar.slider("A (Axis Shift)", min_value=-25.0, max_value=0.0, value=float(A_ideal), step=0.01)
B = st.sidebar.slider("B (Slope Angle)", min_value=1.0, max_value=10.0, value=float(B_ideal), step=0.01)

st.sidebar.markdown("---")
st.sidebar.header("2. Resonances (DSI Cycles)")
genesis_offset = st.sidebar.slider("Genesis Offset (days from 03.01.2009)", min_value=-1000, max_value=1000, value=0, step=5)
t1_age = st.sidebar.slider("Age of 1st Peak (Phase, years)", min_value=1.0, max_value=5.0, value=2.45, step=0.01)
lambda_val = st.sidebar.slider("Lambda (λ) - Multiplier", min_value=1.5, max_value=3.0, value=2.07, step=0.01)

# === SCALE SWITCHES (ABOVE CHART) ===
col1, col2, _ = st.columns([1, 1, 4])
with col1:
    price_scale = st.radio("Price Scale", ["Log", "Lin"], horizontal=True)
with col2:
    time_scale = st.radio("Time Scale", ["Log", "Lin"], horizontal=True)

# === MAIN MODEL MATHEMATICS ===
df = raw_df.copy()
genesis_date = pd.to_datetime('2009-01-03') + pd.Timedelta(days=genesis_offset)
df = df[df.index > genesis_date].copy()

df['DaysFromGenesis'] = (df.index - genesis_date).days
df['YearsFromGenesis'] = df['DaysFromGenesis'] / 365.25

df['Log10_Price'] = np.log10(df['Close'])
df['Log10_Days'] = np.log10(df['DaysFromGenesis'])

# Calculate trend in logarithms
df['PowerLaw_Log10'] = A + B * df['Log10_Days']
df['Residuals'] = df['Log10_Price'] - df['PowerLaw_Log10']

# CONVERT TO REAL DOLLARS FOR THE UPPER CHART
df['PowerLaw_USD'] = 10 ** df['PowerLaw_Log10']
y_97_5_usd = 10 ** (df['PowerLaw_Log10'] + p97_5)
y_83_5_usd = 10 ** (df['PowerLaw_Log10'] + p83_5)
y_16_5_usd = 10 ** (df['PowerLaw_Log10'] + p16_5)
y_2_5_usd = 10 ** (df['PowerLaw_Log10'] + p2_5)

# DSI резонанси
# DSI resonances
cycles_to_draw = 6 # Number of cycles to draw
red_lines_years = [t1_age * (lambda_val ** i) for i in range(cycles_to_draw)]
blue_lines_years = [t1_age * (lambda_val ** (i + 0.5)) for i in range(cycles_to_draw)]

# --- VISUALIZATION ---
fig = make_subplots(rows=2, cols=1,
                    shared_xaxes=False,
                    vertical_spacing=0.08,
                    subplot_titles=(f"1. Trend and Channels (Time: {time_scale} | Price: {price_scale})",
                                    f"2. Residuals Oscillator (Genesis: {genesis_date.strftime('%d %b %Y')})"))

# Trend line (in USD)
fig.add_trace(go.Scatter(x=df['DaysFromGenesis'], y=df['PowerLaw_USD'], mode='lines', name='Power Law Trend', line=dict(color='cyan', dash='dash')), row=1, col=1)

# Deviation channels (in USD)
fig.add_trace(go.Scatter(x=df['DaysFromGenesis'], y=y_97_5_usd, mode='lines', line=dict(color='red', width=1, dash='dot'), name='97.5th Percentile (Bubble)'), row=1, col=1)
fig.add_trace(go.Scatter(x=df['DaysFromGenesis'], y=y_83_5_usd, mode='lines', line=dict(color='blue', width=1, dash='dot'), name='83.5th Percentile'), row=1, col=1)
fig.add_trace(go.Scatter(x=df['DaysFromGenesis'], y=y_16_5_usd, mode='lines', line=dict(color='blue', width=1, dash='dot'), name='16.5th Percentile'), row=1, col=1)
fig.add_trace(go.Scatter(x=df['DaysFromGenesis'], y=y_2_5_usd, mode='lines', line=dict(color='red', width=1, dash='dot'), name='2.5th Percentile (Bottom)'), row=1, col=1)

# Real BTC price (in USD)
fig.add_trace(go.Scatter(x=df['DaysFromGenesis'], y=df['Close'], mode='lines', name='Real Price (USD)', line=dict(color='orange')), row=1, col=1)

# DYNAMIC AXES (Depend on switches)
fig.update_xaxes(type="log" if time_scale == "Log" else "linear", title_text=f"Bitcoin Age (Days) - {time_scale}", row=1, col=1)
fig.update_yaxes(type="log" if price_scale == "Log" else "linear", title_text=f"Price (USD) - {price_scale}", row=1, col=1)

# Oscillator (The lower chart remains linear in time, as DSI waves work this way)
fig.add_trace(go.Scatter(x=df['YearsFromGenesis'], y=df['Residuals'], mode='lines', name='Oscillator (Residuals)', line=dict(color='white', width=1)), row=2, col=1)

for age in red_lines_years:
    fig.add_vline(x=age, line_width=1.5, line_dash="dash", line_color="red", row=2, col=1)
for age in blue_lines_years:
    fig.add_vline(x=age, line_width=1, line_dash="dot", line_color="blue", row=2, col=1)

fig.update_xaxes(title_text="Bitcoin Age (Linear Years from Genesis)", range=[0, 22], row=2, col=1)
fig.update_yaxes(title_text="Residual (Log10)", row=2, col=1)

fig.update_layout(height=750, margin=dict(t=40, b=20), template="plotly_dark", hovermode="x unified", showlegend=False)

st.plotly_chart(fig, use_container_width=True)