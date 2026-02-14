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
st.set_page_config(
    layout="wide",
    page_title="BTC Power Law Pro",
    initial_sidebar_state="expanded"
)

# --- THEME & LANGUAGE STATE MANAGEMENT ---
if 'lang' not in st.session_state:
    st.session_state.lang = "EN"
if 'theme_mode' not in st.session_state:
    st.session_state.theme_mode = "Dark 🌑"

# --- TRANSLATIONS ---
TRANS = {
    "EN": {
        "title": "BTC MODEL",
        "theme_label": "Theme",
        "lang_label": "Language",
        "price_scale": "Price",
        "time_scale": "Time",
        "max_r2": "Max R² (auto)",
        "offset_txt": "offset",
        "lbl_A": "A (Intercept)",
        "lbl_B": "B (Slope)",
        "lbl_gen": "Genesis Offset",
        "lbl_cycle": "1st Cycle Age",
        "lbl_lambda": "Lambda",
        "leg_bubble": "Bubble",
        "leg_accum": "Accumulation",
        "leg_fair": "Fair Value",
        "leg_price": "BTC Price",
        "leg_osc": "Oscillator",
        "kpi_price": "BTC PRICE",
        "kpi_fair": "FAIR VALUE",
        "kpi_fit": "MODEL FIT (R²)",
        "kpi_pot": "GROWTH POTENTIAL",
        "txt_from_model": "from model",
        "txt_to_top": "to top band",
        "hover_date": "Date",
        "hover_price": "Price",
        "hover_fair": "Fair",
        "hover_osc": "Osc"
    },
    "UA": {
        "title": "BTC МОДЕЛЬ",
        "theme_label": "Тема",
        "lang_label": "Мова",
        "price_scale": "Ціна",
        "time_scale": "Час",
        "max_r2": "Макс R² (авто)",
        "offset_txt": "зсув",
        "lbl_A": "A (Константа)",
        "lbl_B": "B (Нахил)",
        "lbl_gen": "Зсув Генезису",
        "lbl_cycle": "Вік 1-го циклу",
        "lbl_lambda": "Лямбда",
        "leg_bubble": "Бульбашка",
        "leg_accum": "Накопичення",
        "leg_fair": "Справедлива ціна",
        "leg_price": "Ціна BTC",
        "leg_osc": "Осцилятор",
        "kpi_price": "ЦІНА BTC",
        "kpi_fair": "СПРАВЕДЛИВА ЦІНА",
        "kpi_fit": "ТОЧНІСТЬ (R²)",
        "kpi_pot": "ПОТЕНЦІАЛ РОСТУ",
        "txt_from_model": "від моделі",
        "txt_to_top": "до верху",
        "hover_date": "Дата",
        "hover_price": "Ціна",
        "hover_fair": "Fair",
        "hover_osc": "Osc"
    }
}

T = TRANS[st.session_state.lang]
is_dark = "Dark" in st.session_state.theme_mode

# --- DYNAMIC COLOR PALETTE ---
if is_dark:
    c_main_bg, c_sidebar_bg, c_card_bg = "#0e1117", "#161a25", "#1e222d"
    c_border, c_text_main, c_text_val = "#2d323e", "#d1d4dc", "#ffffff"
    c_btn_bg, c_btn_hover, c_btn_text = "#2d323e", "#3d4251", "#d1d4dc"
    pl_template, pl_bg_color, pl_grid_color = "plotly_dark", "rgba(0,0,0,0)", "#1e222d"
    pl_btc_color, pl_legend_color, pl_text_color = "#ffffff", "#848e9c", "#848e9c"
else:
    c_main_bg, c_sidebar_bg, c_card_bg = "#ffffff", "#f4f4f4", "#f8f9fa"
    c_border, c_text_main, c_text_val = "#d0d0d0", "#000000", "#000000"
    c_btn_bg, c_btn_hover, c_btn_text = "#ffffff", "#e0e0e0", "#000000"
    pl_template, pl_bg_color, pl_grid_color = "plotly_white", "rgba(255,255,255,1)", "#e6e6e6"
    pl_btc_color, pl_legend_color, pl_text_color = "#111111", "#000000", "#000000"

# --- EXCLUSIVE CSS ---
st.markdown(f"""
    <style>
    .stApp, [data-testid="stAppViewContainer"] {{ background-color: {c_main_bg} !important; }}
    .block-container {{ padding-top: 1rem !important; padding-bottom: 0rem !important; padding-left: 1rem !important; padding-right: 1rem !important; }}
    
    footer {{visibility: hidden; display: none;}}
    
    [data-testid="stHeader"] {{ background-color: transparent; color: {c_text_main}; }}
    [data-testid="stHeader"] button {{ color: {c_text_main} !important; }}
    [data-testid="stAppDeployButton"] {{ display: none !important; }}
    #MainMenu {{ visibility: hidden !important; }}
    
    /* Sidebar Restore Spacing & No Scroll */
    [data-testid="stSidebar"] {{ 
        width: 320px !important; 
        background-color: {c_sidebar_bg} !important; 
        border-right: 1px solid {c_border};
        overflow: hidden !important;
    }}
    
    [data-testid="stSidebarContent"] {{
        overflow: hidden !important;
    }}

    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
        gap: 0.8rem !important; /* Повернуто комфортний відступ */
        padding-top: 0.5rem !important;
    }}
    
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {{ 
        color: {c_text_main} !important; 
    }}

    /* KPI Cards */
    .metric-card {{
        background: {c_card_bg}; border: 1px solid {c_border}; border-radius: 8px; padding: 10px 16px; text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-top: 5px; min-height: 110px; display: flex; flex-direction: column; justify-content: center; align-items: center;
    }}
    .metric-label {{ color: {c_text_main}; font-size: 13px; font-weight: 600; margin-bottom: 4px; opacity: 0.8; }}
    .metric-value {{ color: {c_text_val}; font-size: 22px; font-weight: 800; }}
    .metric-delta {{ font-size: 13px; font-weight: 600; margin-top: 2px; }}

    /* Sidebar Radio Tweaks */
    div[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {{ gap: 10px; }}
    div[data-testid="stSidebar"] .stRadio label p {{ font-size: 14px !important; }}
    
    .stButton > button {{
        width: 100% !important; background-color: {c_btn_bg} !important; color: {c_btn_text} !important;
        border: 1px solid {c_border} !important; border-radius: 4px !important; height: 32px !important;
        line-height: 30px !important; font-size: 14px !important; transition: 0.2s; font-weight: bold !important;
    }}
    .stSlider {{ margin-bottom: -5px !important; margin-top: 5px !important; }}
    
    .sidebar-title {{
        text-align: center; color: #f0b90b; margin-bottom: 10px !important; font-size: 1.8rem; font-weight: bold;
    }}
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_raw_data():
    url = "https://raw.githubusercontent.com/Habrador/Bitcoin-price-visualization/main/Bitcoin-price-USD.csv"
    early_df = pd.read_csv(url)
    early_df['Date'] = pd.to_datetime(early_df['Date'])
    early_df.set_index('Date', inplace=True)
    early_df.rename(columns={'Price': 'Close'}, inplace=True)
    recent_df = yf.download('BTC-USD', start='2014-09-17')
    recent_df = recent_df['Close'] if isinstance(recent_df.columns, pd.MultiIndex) else recent_df[['Close']]
    recent_df.columns = ['Close']
    recent_df = recent_df.dropna()
    recent_df.index = pd.to_datetime(recent_df.index).tz_localize(None)
    early_df = early_df[early_df.index < '2014-09-17']
    return pd.concat([early_df, recent_df])

try:
    raw_df = load_raw_data()
except Exception as e:
    st.error(f"Error: {e}"); st.stop()

@st.cache_data
def find_best_fit_params(df_in):
    best_r2_val, best_off_val, best_A_val, best_B_val = -1, 150, -17, 5.8
    for off in range(180, 230):
        gen_test = pd.to_datetime('2009-01-03') + pd.Timedelta(days=off)
        df_test = df_in[df_in.index > gen_test].copy()
        d_vals, p_vals = (df_test.index - gen_test).days.values, df_test['Close'].values
        valid = (d_vals > 0) & (p_vals > 0)
        if np.sum(valid) < 100: continue
        log_d, log_p = np.log10(d_vals[valid]), np.log10(p_vals[valid])
        slope, intercept = np.polyfit(log_d, log_p, 1)
        r2 = 1 - (np.sum((log_p - (slope * log_d + intercept))**2) / np.sum((log_p - np.mean(log_p))**2))
        if r2 > best_r2_val: best_r2_val, best_off_val, best_A_val, best_B_val = r2, off, intercept, slope
    return best_off_val, best_A_val, best_B_val, best_r2_val

opt_offset, opt_A, opt_B, opt_R2 = find_best_fit_params(raw_df)

# Initialize defaults
for k, v in {"genesis_offset": int(opt_offset), "A": float(round(opt_A, 3)), "B": float(round(opt_B, 3)), "t1_age": 1.88, "lambda_val": 2.12}.items():
    if k not in st.session_state: st.session_state[k] = v

def update_param(param_name, delta):
    st.session_state[param_name] = round(st.session_state[param_name] + delta, 3)

# --- SIDEBAR UI ---
with st.sidebar:
    st.markdown(f"<div class='sidebar-title'>{T['title']}</div>", unsafe_allow_html=True)

    c_v1, c_v2 = st.columns(2)
    price_scale = c_v1.radio(T['price_scale'], ["Log", "Lin"], index=0, horizontal=True)
    time_scale = c_v2.radio(T['time_scale'], ["Log", "Lin"], index=0, horizontal=True)

    st.markdown(f"<p style='color:{c_text_main}; text-align:center; font-size: 0.85rem;'>{T['max_r2']}: {opt_R2 * 100:.3f}% ({T['offset_txt']} {opt_offset}d)</p>", unsafe_allow_html=True)

    def fancy_control(label, key, step, min_v, max_v):
        st.markdown(f"**{label}**")
        c1, c2, c3 = st.columns([1, 2.5, 1])
        if c1.button("➖", key=f"{key}_m"): update_param(key, -step)
        if c3.button("➕", key=f"{key}_p"): update_param(key, step)
        return c2.slider(key, min_v, max_v, key=key, step=step, label_visibility="collapsed")

    A = fancy_control(T['lbl_A'], "A", 0.01, -25.0, 0.0)
    B = fancy_control(T['lbl_B'], "B", 0.01, 1.0, 10.0)
    genesis_offset = fancy_control(T['lbl_gen'], "genesis_offset", 1, -1000, 1000)
    t1_age = fancy_control(T['lbl_cycle'], "t1_age", 0.01, 1.0, 5.0)
    lambda_val = fancy_control(T['lbl_lambda'], "lambda_val", 0.01, 1.5, 3.0)

    # Bottom Settings
    st.markdown("<hr style='margin: 15px 0 10px 0; opacity:0.1;'>", unsafe_allow_html=True)
    cl_l, cl_t = st.columns(2)
    with cl_l:
        lang_choice = st.radio(T['lang_label'], ["EN 🇬🇧", "UA 🇺🇦"], index=0 if st.session_state.lang=="EN" else 1, horizontal=True)
        new_lang = "EN" if "EN" in lang_choice else "UA"
        if new_lang != st.session_state.lang:
            st.session_state.lang = new_lang
            st.rerun()
    with cl_t:
        new_theme = st.radio(T['theme_label'], ["Dark 🌑", "Light ☀️"], index=0 if "Dark" in st.session_state.theme_mode else 1, horizontal=True)
        if new_theme != st.session_state.theme_mode:
            st.session_state.theme_mode = new_theme
            st.rerun()

# --- CALCULATIONS ---
df = raw_df.copy()
gen_date = pd.to_datetime('2009-01-03') + pd.Timedelta(days=genesis_offset)
df = df[df.index > gen_date].copy()
df['Days'] = (df.index - gen_date).days
df['LogP'], df['LogD'] = np.log10(df['Close']), np.log10(df['Days'])
df['ModelLog'] = st.session_state.A + st.session_state.B * df['LogD']
df['Res'] = df['LogP'] - df['ModelLog']
df['Fair'] = 10 ** df['ModelLog']
current_r2 = (1 - (np.sum(df['Res']**2) / np.sum((df['LogP'] - np.mean(df['LogP']))**2))) * 100
p2_5, p16_5, p83_5, p97_5 = np.percentile(df['Res'], [2.5, 16.5, 83.5, 97.5])

# --- VIZ ---
view_max = df['Days'].max() + 365 * 1.5
m_x = np.logspace(0, np.log10(view_max), 400) if time_scale == "Log" else np.linspace(1, view_max, 400)
m_dates_str = [(gen_date + pd.Timedelta(days=d)).strftime('%d.%m.%Y') for d in m_x]
m_log_d = np.log10(m_x)
m_fair_usd = 10 ** (st.session_state.A + st.session_state.B * m_log_d)

fig = make_subplots(rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.10, row_heights=[0.76, 0.24])
fig.add_trace(go.Scatter(x=m_x, y=10**(st.session_state.A+st.session_state.B*m_log_d+p97_5), mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'), 1, 1)
fig.add_trace(go.Scatter(x=m_x, y=10**(st.session_state.A+st.session_state.B*m_log_d+p83_5), mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(234, 61, 47, 0.15)', name=T['leg_bubble'], customdata=m_dates_str, hovertemplate=f"{T['hover_date']}: %{{customdata}}<br>{T['hover_price']}: $%{{y:,.0f}}<extra></extra>"), 1, 1)
fig.add_trace(go.Scatter(x=m_x, y=10**(st.session_state.A+st.session_state.B*m_log_d+p16_5), mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'), 1, 1)
fig.add_trace(go.Scatter(x=m_x, y=10**(st.session_state.A+st.session_state.B*m_log_d+p2_5), mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(14, 203, 129, 0.15)', name=T['leg_accum'], customdata=m_dates_str, hovertemplate=f"{T['hover_date']}: %{{customdata}}<br>{T['hover_price']}: $%{{y:,.0f}}<extra></extra>"), 1, 1)
fig.add_trace(go.Scatter(x=m_x, y=m_fair_usd, mode='lines', line=dict(color='#f0b90b', width=1.5, dash='dash'), name=T['leg_fair'], customdata=m_dates_str, hovertemplate=f"{T['hover_date']}: %{{customdata}}<br>{T['hover_fair']}: $%{{y:,.0f}}<extra></extra>"), 1, 1)
fig.add_trace(go.Scatter(x=df['Days'], y=df['Close'], mode='lines', name=T['leg_price'], line=dict(color=pl_btc_color, width=1.3), customdata=df.index.strftime('%d.%m.%Y'), hovertemplate=f"{T['hover_date']}: %{{customdata}}<br>{T['hover_price']}: $%{{y:,.0f}}<extra></extra>"), 1, 1)

t_vals = [(pd.Timestamp(f'{y}-01-01') - gen_date).days for y in range(gen_date.year + 1, 2028) if (pd.Timestamp(f'{y}-01-01') - gen_date).days > 0]
t_text = [str(y) for y in range(gen_date.year + 1, 2028) if (pd.Timestamp(f'{y}-01-01') - gen_date).days > 0]
for r in [1, 2]:
    fig.update_xaxes(type="log" if time_scale == "Log" else "linear", tickvals=t_vals, ticktext=t_text, range=[np.log10(t_vals[0]), np.log10(view_max)] if time_scale == "Log" else [0, view_max], gridcolor=pl_grid_color, tickfont=dict(color=pl_text_color, size=13), row=r, col=1)
fig.update_yaxes(type="log" if price_scale == "Log" else "linear", range=[np.log10(0.01), np.log10(df['Close'].max() * 8)] if price_scale == "Log" else None, gridcolor=pl_grid_color, tickfont=dict(color=pl_text_color, size=13), row=1, col=1)

fig.add_trace(go.Scatter(x=df['Days'], y=df['Res'], mode='lines', name=T['leg_osc'], line=dict(color='#0ecb81', width=1.2), customdata=df.index.strftime('%d.%m.%Y'), hovertemplate=f"{T['hover_date']}: %{{customdata}}<br>{T['hover_osc']}: %{{y:.3f}}<extra></extra>"), 2, 1)
fig.add_hline(y=0, line_width=1, line_color=pl_legend_color, row=2, col=1)
for i in range(6):
    fig.add_vline(x=st.session_state.t1_age * (st.session_state.lambda_val ** i) * 365.25, line_width=1.5, line_dash="dash", line_color="#ea3d2f", opacity=0.8, row=2, col=1)
    fig.add_vline(x=st.session_state.t1_age * (st.session_state.lambda_val ** (i + 0.5)) * 365.25, line_width=1, line_dash="dot", line_color="#2b6aff", opacity=0.5, row=2, col=1)

fig.update_layout(height=720, margin=dict(t=30, b=10, l=50, r=20), template=pl_template, font=dict(color=pl_text_color), legend=dict(orientation="h", y=0.27, x=0.5, xanchor="center", font=dict(size=15, color=pl_legend_color)), paper_bgcolor=pl_bg_color, plot_bgcolor=pl_bg_color)
st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, theme=None)

# --- KPI ---
l_p, l_f = df['Close'].iloc[-1], df['Fair'].iloc[-1]
diff, pot = ((l_p - l_f) / l_f) * 100, ((10**(st.session_state.A+st.session_state.B*np.log10(df['Days'].max())+p97_5) - l_p) / l_p) * 100
k1, k2, k3, k4 = st.columns(4)

def kpi_card(col, label, value, delta=None, d_color=None):
    delta_html = f"<div class='metric-delta' style='color:{d_color}'>{delta}</div>" if delta else "<div class='metric-delta' style='visibility:hidden;'>-</div>"
    col.markdown(f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div>{delta_html}</div>", unsafe_allow_html=True)

kpi_card(k1, T['kpi_price'], f"${l_p:,.0f}")
kpi_card(k2, T['kpi_fair'], f"${l_f:,.0f}", f"{diff:+.1f}% {T['txt_from_model']}", "#0ecb81" if diff < 0 else "#ea3d2f")
kpi_card(k3, T['kpi_fit'], f"{current_r2:.2f}%")
kpi_card(k4, T['kpi_pot'], f"+{pot:,.0f}%", T['txt_to_top'], "#f0b90b")