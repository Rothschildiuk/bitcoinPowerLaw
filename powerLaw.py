import numpy as np
import streamlit as st

from utils import fancy_control, calculate_r2_score

# --- MATH CORE ---


def fit_powerlaw_regression(absolute_days, log_prices, genesis_offset_days):
    """Calculates the optimal Slope(B) and Intercept(A) for a given offset."""
    days_since_offset = absolute_days - genesis_offset_days
    positive_mask = days_since_offset > 0

    if np.sum(positive_mask) < 100:
        return 0.0, 0.0, 0.0

    valid_days = days_since_offset[positive_mask]
    valid_log_prices = log_prices[positive_mask]
    log_days = np.log10(valid_days)

    slope_b, intercept_a = np.polyfit(log_days, valid_log_prices, 1)
    predicted_log_prices = slope_b * log_days + intercept_a
    r2_score = calculate_r2_score(valid_log_prices, predicted_log_prices)

    return slope_b, intercept_a, r2_score


def calculate_r2_for_manual_params(
    absolute_days, log_prices, genesis_offset_days, intercept_a, slope_b
):
    """Calculates R2 for specific manual A and B values."""
    days_since_offset = absolute_days - genesis_offset_days
    positive_mask = days_since_offset > 0

    if np.sum(positive_mask) < 100:
        return 0.0

    valid_days = days_since_offset[positive_mask]
    valid_log_prices = log_prices[positive_mask]
    log_days = np.log10(valid_days)
    predicted_log_prices = intercept_a + slope_b * log_days

    return calculate_r2_score(valid_log_prices, predicted_log_prices)


def find_best_fit_params(absolute_days, log_prices):
    slope_b, intercept_a, r2_score = fit_powerlaw_regression(absolute_days, log_prices, 0)
    return 0, intercept_a, slope_b, r2_score


# Backward-compatible alias used by existing code.
def find_global_best_fit_optimized(all_abs_days, all_log_close):
    return find_best_fit_params(all_abs_days, all_log_close)


# --- SIDEBAR RENDERER ---
def render_sidebar(all_abs_days, all_log_close, text_color):
    # Initialize defaults if needed
    if "genesis_offset" not in st.session_state:
        st.session_state["genesis_offset"] = 0

    opt_offset, opt_a, opt_b, _ = find_best_fit_params(all_abs_days, all_log_close)

    if "A" not in st.session_state:
        st.session_state["A"] = float(round(opt_a, 3))
    if "B" not in st.session_state:
        st.session_state["B"] = float(round(opt_b, 3))

    def reset_powerlaw_params():
        st.session_state["genesis_offset"] = int(opt_offset)
        st.session_state["A"] = float(round(opt_a, 3))
        st.session_state["B"] = float(round(opt_b, 3))

    # Controls - Time scale removed, Price scale kept
    price_scale = st.radio("Price", ["Log", "Lin"], index=0, horizontal=True)

    auto_fit = st.checkbox(
        "Auto-Fit A & B",
        value=False,
        help="Automatically calculate best Slope (B) and Intercept (A) when Offset changes.",
    )

    display_r2 = 0.0

    # Get current values safely
    current_intercept_a = st.session_state.get("A", opt_a)
    current_slope_b = st.session_state.get("B", opt_b)
    current_offset_days = st.session_state.get("genesis_offset", opt_offset)

    if auto_fit:
        # Calculate BEST fit parameters
        fitted_slope_b, fitted_intercept_a, fitted_r2_score = fit_powerlaw_regression(
            all_abs_days, all_log_close, current_offset_days
        )
        st.session_state["A"] = float(round(fitted_intercept_a, 3))
        st.session_state["B"] = float(round(fitted_slope_b, 3))
        display_r2 = fitted_r2_score
    else:
        # Calculate R2 based on MANUAL sliders (The Fix)
        display_r2 = calculate_r2_for_manual_params(
            all_abs_days, all_log_close, current_offset_days, current_intercept_a, current_slope_b
        )

    st.markdown("**A (Intercept)**")
    fancy_control("A (Intercept)", "A", 0.01, -25.0, 0.0, disabled=auto_fit)

    st.markdown("**B (Slope)**")
    fancy_control("B (Slope)", "B", 0.01, 1.0, 7.0, disabled=auto_fit)

    st.markdown(
        f"<p style='color:{text_color}; margin-top: 2px;'>"
        f"PowerLaw R² = {display_r2 * 100:.4f}%</p>",
        unsafe_allow_html=True,
    )

    st.button("Reset parameters", use_container_width=True, on_click=reset_powerlaw_params)

    # Return only price_scale and r2
    return price_scale, display_r2
