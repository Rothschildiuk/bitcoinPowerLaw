import numpy as np
import streamlit as st

from core.constants import DEFAULT_A, DEFAULT_B, KEY_A, KEY_B, KEY_GENESIS_OFFSET
from core.utils import calculate_r2_score, fancy_control, inline_radio_control

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


def optimize_single_powerlaw_parameter(
    absolute_days,
    log_prices,
    genesis_offset_days,
    current_intercept_a,
    current_slope_b,
    parameter_key,
):
    if parameter_key == "A":
        coarse_candidates = np.linspace(-35.0, 0.0, 281)
        best_value = float(current_intercept_a)
        best_r2 = calculate_r2_for_manual_params(
            absolute_days, log_prices, genesis_offset_days, current_intercept_a, current_slope_b
        )
        for candidate in coarse_candidates:
            score = calculate_r2_for_manual_params(
                absolute_days, log_prices, genesis_offset_days, float(candidate), current_slope_b
            )
            if score > best_r2:
                best_r2 = score
                best_value = float(candidate)

        fine_lo = max(-35.0, best_value - 0.2)
        fine_hi = min(0.0, best_value + 0.2)
        fine_candidates = np.linspace(fine_lo, fine_hi, 401)
        for candidate in fine_candidates:
            score = calculate_r2_for_manual_params(
                absolute_days, log_prices, genesis_offset_days, float(candidate), current_slope_b
            )
            if score > best_r2:
                best_r2 = score
                best_value = float(candidate)

        return round(best_value, 3), best_r2

    if parameter_key == "B":
        coarse_candidates = np.linspace(1.0, 12.0, 221)
        best_value = float(current_slope_b)
        best_r2 = calculate_r2_for_manual_params(
            absolute_days, log_prices, genesis_offset_days, current_intercept_a, current_slope_b
        )
        for candidate in coarse_candidates:
            score = calculate_r2_for_manual_params(
                absolute_days, log_prices, genesis_offset_days, current_intercept_a, float(candidate)
            )
            if score > best_r2:
                best_r2 = score
                best_value = float(candidate)

        fine_lo = max(1.0, best_value - 0.15)
        fine_hi = min(12.0, best_value + 0.15)
        fine_candidates = np.linspace(fine_lo, fine_hi, 301)
        for candidate in fine_candidates:
            score = calculate_r2_for_manual_params(
                absolute_days, log_prices, genesis_offset_days, current_intercept_a, float(candidate)
            )
            if score > best_r2:
                best_r2 = score
                best_value = float(candidate)

        return round(best_value, 3), best_r2

    return round(float(current_intercept_a), 3), calculate_r2_for_manual_params(
        absolute_days, log_prices, genesis_offset_days, current_intercept_a, current_slope_b
    )


# Backward-compatible alias used by existing code.
def find_global_best_fit_optimized(all_abs_days, all_log_close):
    return find_best_fit_params(all_abs_days, all_log_close)


# --- SIDEBAR RENDERER ---
def render_sidebar(
    all_abs_days,
    all_log_close,
    text_color,
    show_price_scale=True,
    render_extra_controls=None,
    a_key=KEY_A,
    b_key=KEY_B,
    default_a=DEFAULT_A,
    default_b=DEFAULT_B,
):
    # Initialize defaults if needed
    if KEY_GENESIS_OFFSET not in st.session_state:
        st.session_state[KEY_GENESIS_OFFSET] = 0

    opt_offset = 0
    opt_a = default_a
    opt_b = default_b

    if a_key not in st.session_state:
        st.session_state[a_key] = float(default_a)
    if b_key not in st.session_state:
        st.session_state[b_key] = float(default_b)

    def reset_powerlaw_params():
        st.session_state[KEY_GENESIS_OFFSET] = int(opt_offset)
        st.session_state[a_key] = float(default_a)
        st.session_state[b_key] = float(default_b)

    # Controls - Time scale removed, Price scale optional
    if show_price_scale:
        price_scale = inline_radio_control("Price", ["Log", "Lin"])
    else:
        price_scale = "Log"

    def auto_fit_intercept():
        best_a, _ = optimize_single_powerlaw_parameter(
            all_abs_days,
            all_log_close,
            int(st.session_state.get(KEY_GENESIS_OFFSET, opt_offset)),
            float(st.session_state.get(a_key, opt_a)),
            float(st.session_state.get(b_key, opt_b)),
            "A",
        )
        st.session_state[a_key] = float(best_a)

    def auto_fit_slope():
        best_b, _ = optimize_single_powerlaw_parameter(
            all_abs_days,
            all_log_close,
            int(st.session_state.get(KEY_GENESIS_OFFSET, opt_offset)),
            float(st.session_state.get(a_key, opt_a)),
            float(st.session_state.get(b_key, opt_b)),
            "B",
        )
        st.session_state[b_key] = float(best_b)

    st.markdown("**A (Intercept)**")
    fancy_control(
        "A (Intercept)",
        a_key,
        0.001,
        -35.0,
        0.0,
        on_auto_fit=auto_fit_intercept,
        auto_fit_label="AF",
    )

    st.markdown("**B (Slope)**")
    fancy_control(
        "B (Slope)",
        b_key,
        0.001,
        1.0,
        12.0,
        on_auto_fit=auto_fit_slope,
        auto_fit_label="AF",
    )

    display_r2 = calculate_r2_for_manual_params(
        all_abs_days,
        all_log_close,
        int(st.session_state.get(KEY_GENESIS_OFFSET, opt_offset)),
        float(st.session_state.get(a_key, opt_a)),
        float(st.session_state.get(b_key, opt_b)),
    )

    if callable(render_extra_controls):
        render_extra_controls()

    st.markdown(
        f"<p style='color:{text_color}; margin-top: 2px;'>"
        f"PowerLaw R² = {display_r2 * 100:.4f}%</p>",
        unsafe_allow_html=True,
    )

    st.button("Reset parameters", use_container_width=True, on_click=reset_powerlaw_params)

    # Return only price_scale and r2
    return price_scale, display_r2
