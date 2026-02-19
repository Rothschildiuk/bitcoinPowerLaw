import numpy as np
import streamlit as st


def calculate_r2_score(actual_values, predicted_values):
    residual_sum_squares = np.sum((actual_values - predicted_values) ** 2)
    total_sum_squares = np.sum((actual_values - np.mean(actual_values)) ** 2)
    if total_sum_squares <= 1e-12:
        return 0.0
    return 1 - (residual_sum_squares / total_sum_squares)


def get_stable_trend_fit(log_days, log_prices, intercept_a, slope_b, residual_threshold=5.0):
    trend_log_prices = intercept_a + slope_b * log_days
    residual_series = log_prices - trend_log_prices
    median_abs_residual = float(np.median(np.abs(residual_series)))

    if (not np.isfinite(median_abs_residual)) or median_abs_residual > residual_threshold:
        fitted_slope_b, fitted_intercept_a = np.polyfit(log_days, log_prices, 1)
        intercept_a = float(fitted_intercept_a)
        slope_b = float(fitted_slope_b)
        trend_log_prices = intercept_a + slope_b * log_days
        residual_series = log_prices - trend_log_prices

    return intercept_a, slope_b, trend_log_prices, residual_series


def inline_radio_control(
    label, options, *, key=None, index=0, horizontal=True, columns_ratio=(1, 2.2)
):
    label_col, control_col = st.columns(list(columns_ratio))
    label_col.markdown(f"**{label}**")
    with control_col:
        return st.radio(
            label,
            options,
            index=index,
            key=key,
            horizontal=horizontal,
            label_visibility="collapsed",
        )


def fancy_control(label, key, step, min_v, max_v, disabled=False, on_manual_change=None):
    c1, c2, c3 = st.columns([1, 2.5, 1])
    current_value = st.session_state.get(key, min_v)
    try:
        current_value = float(current_value)
    except (TypeError, ValueError):
        current_value = min_v
    st.session_state[key] = round(min(max_v, max(min_v, current_value)), 3)

    def on_minus():
        new_val = st.session_state[key] - step
        st.session_state[key] = round(max(min_v, new_val), 3)
        if on_manual_change is not None:
            on_manual_change()

    def on_plus():
        new_val = st.session_state[key] + step
        st.session_state[key] = round(min(max_v, new_val), 3)
        if on_manual_change is not None:
            on_manual_change()

    def on_slider_change():
        if on_manual_change is not None:
            on_manual_change()

    if c1.button("➖", key=f"{key}_m", disabled=disabled, on_click=on_minus):
        pass
    if c3.button("➕", key=f"{key}_p", disabled=disabled, on_click=on_plus):
        pass

    return c2.slider(
        key,
        min_v,
        max_v,
        key=key,
        step=step,
        label_visibility="collapsed",
        disabled=disabled,
        on_change=on_slider_change,
    )
