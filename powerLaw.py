import streamlit as st
import numpy as np
from utils import fancy_control

# --- MATH CORE ---
def calculate_regression_numpy(abs_days_array, log_price_array, offset_value):
    """Calculates the optimal Slope(B) and Intercept(A) for a given offset."""
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

def calculate_manual_r2(abs_days_array, log_price_array, offset_value, A, B):
    """Calculates R2 for specific manual A and B values."""
    x_days = abs_days_array - offset_value
    mask = x_days > 0

    if np.sum(mask) < 100:
        return 0.0

    x_valid = x_days[mask]
    y_valid = log_price_array[mask]
    log_x = np.log10(x_valid)

    # Prediction based on manual A and B
    y_pred = A + B * log_x

    ss_res = np.sum((y_valid - y_pred) ** 2)
    ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)

    # R2 calculation
    r2 = 1 - (ss_res / ss_tot)
    return r2

def find_global_best_fit_optimized(all_abs_days, all_log_close):
    b, a, r2 = calculate_regression_numpy(all_abs_days, all_log_close, 0)
    return 0, a, b, r2

# --- SIDEBAR RENDERER ---
def render_sidebar(all_abs_days, all_log_close, text_color):

    # Initialize defaults if needed
    if "genesis_offset" not in st.session_state:
        st.session_state["genesis_offset"] = 0

    opt_offset, opt_a, opt_b, _ = find_global_best_fit_optimized(all_abs_days, all_log_close)

    if "A" not in st.session_state: st.session_state["A"] = float(round(opt_a, 3))
    if "B" not in st.session_state: st.session_state["B"] = float(round(opt_b, 3))

    # Controls
    c_v1, c_v2 = st.columns(2)
    price_scale = c_v1.radio("Price", ["Log", "Lin"], index=0, horizontal=True)
    time_scale = c_v2.radio("Time", ["Log", "Lin"], index=0, horizontal=True)

    auto_fit = st.checkbox("Auto-Fit A & B", value=False, help="Automatically calculate best Slope (B) and Intercept (A) when Offset changes.")

    display_r2 = 0.0

    # Get current values safely
    current_a = st.session_state.get("A", opt_a)
    current_b = st.session_state.get("B", opt_b)
    curr_off = st.session_state.get("genesis_offset", opt_offset)

    if auto_fit:
        # Calculate BEST fit parameters
        calc_b, calc_a, calc_r2 = calculate_regression_numpy(all_abs_days, all_log_close, curr_off)
        st.session_state["A"] = float(round(calc_a, 3))
        st.session_state["B"] = float(round(calc_b, 3))
        display_r2 = calc_r2
    else:
        # Calculate R2 based on MANUAL sliders (The Fix)
        display_r2 = calculate_manual_r2(all_abs_days, all_log_close, curr_off, current_a, current_b)

    st.markdown("**A (Intercept)**")
    fancy_control("A (Intercept)", "A", 0.01, -25.0, 0.0, disabled=auto_fit)

    st.markdown("**B (Slope)**")
    fancy_control("B (Slope)", "B", 0.01, 1.0, 7.0, disabled=auto_fit)

    st.markdown(
        f"<p style='color:{text_color}; margin-top: 2px;'>"
        f"PowerLaw R² = {display_r2 * 100:.4f}%</p>",
        unsafe_allow_html=True)

    return price_scale, time_scale, display_r2