import numpy as np
import streamlit as st

from core.constants import (
    DEFAULT_A,
    DEFAULT_B,
    KEY_A,
    KEY_B,
    KEY_GENESIS_OFFSET,
    POWERLAW_INTERCEPT_MAX,
    POWERLAW_INTERCEPT_MIN,
    POWERLAW_SLOPE_MAX,
    POWERLAW_SLOPE_MIN,
)
from core.optimization_utils import optimize_single_scalar_parameter
from core.utils import calculate_r2_score, evaluate_powerlaw_values, fancy_control

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


def _select_residual_episode_indices(
    valid_days,
    residual_values,
    threshold,
    direction,
    min_duration_days,
):
    if direction == "above":
        extreme_mask = residual_values >= float(threshold)
        choose_index = np.argmax
    else:
        extreme_mask = residual_values <= float(threshold)
        choose_index = np.argmin

    if float(min_duration_days) <= 1.0:
        return list(np.flatnonzero(extreme_mask))

    selected_indices = []
    episode_start = None
    previous_idx = None
    for idx, is_extreme in enumerate(extreme_mask):
        if is_extreme:
            if episode_start is None:
                episode_start = idx
            previous_idx = idx
            continue

        if episode_start is not None and previous_idx is not None:
            episode_days = valid_days[previous_idx] - valid_days[episode_start] + 1.0
            if episode_days >= float(min_duration_days):
                episode_slice = slice(episode_start, previous_idx + 1)
                local_values = residual_values[episode_slice]
                selected_indices.append(episode_start + int(choose_index(local_values)))
        episode_start = None
        previous_idx = None

    if episode_start is not None and previous_idx is not None:
        episode_days = valid_days[previous_idx] - valid_days[episode_start] + 1.0
        if episode_days >= float(min_duration_days):
            episode_slice = slice(episode_start, previous_idx + 1)
            local_values = residual_values[episode_slice]
            selected_indices.append(episode_start + int(choose_index(local_values)))

    return selected_indices


def fit_peak_powerlaw_envelope(
    absolute_days,
    log_prices,
    genesis_offset_days,
    model_days,
    min_peak_count=2,
    sigma_threshold=1.0,
    min_duration_days=1.0,
    residuals=None,
    threshold_offset=None,
):
    days_since_offset = np.asarray(absolute_days, dtype=float) - float(genesis_offset_days)
    log_prices = np.asarray(log_prices, dtype=float)
    valid_mask = (days_since_offset > 0.0) & np.isfinite(log_prices)
    if np.sum(valid_mask) < int(min_peak_count):
        return None

    valid_days = days_since_offset[valid_mask]
    valid_logs = log_prices[valid_mask]
    if residuals is None:
        baseline_slope, baseline_intercept = np.polyfit(np.log10(valid_days), valid_logs, 1)
        residual_values = valid_logs - (baseline_intercept + baseline_slope * np.log10(valid_days))
    else:
        residual_values = np.asarray(residuals, dtype=float)[valid_mask]
    if threshold_offset is None:
        residual_sigma = float(np.std(residual_values))
        if not np.isfinite(residual_sigma) or residual_sigma <= 0.0:
            return None
        threshold_offset = float(sigma_threshold) * residual_sigma
    peak_indices = _select_residual_episode_indices(
        valid_days,
        residual_values,
        threshold=threshold_offset,
        direction="above",
        min_duration_days=min_duration_days,
    )

    if len(peak_indices) < int(min_peak_count):
        return None

    peak_days = valid_days[peak_indices]
    peak_logs = valid_logs[peak_indices]
    slope_b, intercept_a = np.polyfit(np.log10(peak_days), peak_logs, 1)
    model_days = np.asarray(model_days, dtype=float)
    model_log = intercept_a + slope_b * np.log10(np.maximum(model_days, 1.0))
    model_values = np.power(10.0, model_log)
    return {
        "intercept": float(intercept_a),
        "slope": float(slope_b),
        "peak_days": peak_days,
        "peak_values": np.power(10.0, peak_logs),
        "model_values": model_values,
    }


def fit_trough_powerlaw_envelope(
    absolute_days,
    log_prices,
    genesis_offset_days,
    model_days,
    min_trough_count=2,
    sigma_threshold=-1.0,
    min_duration_days=1.0,
    residuals=None,
    threshold_offset=None,
):
    days_since_offset = np.asarray(absolute_days, dtype=float) - float(genesis_offset_days)
    log_prices = np.asarray(log_prices, dtype=float)
    valid_mask = (days_since_offset > 0.0) & np.isfinite(log_prices)
    if np.sum(valid_mask) < int(min_trough_count):
        return None

    valid_days = days_since_offset[valid_mask]
    valid_logs = log_prices[valid_mask]
    if residuals is None:
        baseline_slope, baseline_intercept = np.polyfit(np.log10(valid_days), valid_logs, 1)
        residual_values = valid_logs - (baseline_intercept + baseline_slope * np.log10(valid_days))
    else:
        residual_values = np.asarray(residuals, dtype=float)[valid_mask]
    if threshold_offset is None:
        residual_sigma = float(np.std(residual_values))
        if not np.isfinite(residual_sigma) or residual_sigma <= 0.0:
            return None
        threshold_offset = float(sigma_threshold) * residual_sigma
    trough_indices = _select_residual_episode_indices(
        valid_days,
        residual_values,
        threshold=threshold_offset,
        direction="below",
        min_duration_days=min_duration_days,
    )

    if len(trough_indices) < int(min_trough_count):
        return None

    trough_days = valid_days[trough_indices]
    trough_logs = valid_logs[trough_indices]
    slope_b, intercept_a = np.polyfit(np.log10(trough_days), trough_logs, 1)
    model_days = np.asarray(model_days, dtype=float)
    model_log = intercept_a + slope_b * np.log10(np.maximum(model_days, 1.0))
    model_values = np.power(10.0, model_log)
    return {
        "intercept": float(intercept_a),
        "slope": float(slope_b),
        "trough_days": trough_days,
        "trough_values": np.power(10.0, trough_logs),
        "model_values": model_values,
    }


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


def find_best_fit_params_for_offset(absolute_days, log_prices, genesis_offset_days):
    slope_b, intercept_a, r2_score = fit_powerlaw_regression(
        absolute_days,
        log_prices,
        genesis_offset_days,
    )
    return int(genesis_offset_days), intercept_a, slope_b, r2_score


def optimize_single_powerlaw_parameter(
    absolute_days,
    log_prices,
    genesis_offset_days,
    current_intercept_a,
    current_slope_b,
    parameter_key,
    a_min=POWERLAW_INTERCEPT_MIN,
    a_max=POWERLAW_INTERCEPT_MAX,
    b_min=POWERLAW_SLOPE_MIN,
    b_max=POWERLAW_SLOPE_MAX,
):
    if parameter_key == "A":
        best_value, best_r2 = optimize_single_scalar_parameter(
            float(current_intercept_a),
            lambda candidate: calculate_r2_for_manual_params(
                absolute_days, log_prices, genesis_offset_days, float(candidate), current_slope_b
            ),
            min_value=float(a_min),
            max_value=float(a_max),
            coarse_points=281,
            fine_window=0.2,
            fine_points=401,
        )
        return round(best_value, 3), best_r2

    if parameter_key == "B":
        best_value, best_r2 = optimize_single_scalar_parameter(
            float(current_slope_b),
            lambda candidate: calculate_r2_for_manual_params(
                absolute_days,
                log_prices,
                genesis_offset_days,
                current_intercept_a,
                float(candidate),
            ),
            min_value=float(b_min),
            max_value=float(b_max),
            coarse_points=221,
            fine_window=0.15,
            fine_points=301,
        )
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
    render_extra_controls=None,
    render_after_actions=None,
    a_key=KEY_A,
    b_key=KEY_B,
    default_a=DEFAULT_A,
    default_b=DEFAULT_B,
    reset_a=None,
    reset_b=None,
    a_min=POWERLAW_INTERCEPT_MIN,
    a_max=POWERLAW_INTERCEPT_MAX,
    b_min=POWERLAW_SLOPE_MIN,
    b_max=POWERLAW_SLOPE_MAX,
    genesis_offset_days=None,
):
    # Initialize defaults if needed
    if KEY_GENESIS_OFFSET not in st.session_state:
        st.session_state[KEY_GENESIS_OFFSET] = 0

    opt_offset = 0 if genesis_offset_days is None else int(genesis_offset_days)
    opt_a = default_a
    opt_b = default_b
    reset_a = default_a if reset_a is None else reset_a
    reset_b = default_b if reset_b is None else reset_b

    if a_key not in st.session_state:
        st.session_state[a_key] = float(default_a)
    if b_key not in st.session_state:
        st.session_state[b_key] = float(default_b)

    def current_offset():
        if genesis_offset_days is not None:
            return int(genesis_offset_days)
        return int(st.session_state.get(KEY_GENESIS_OFFSET, opt_offset))

    def reset_powerlaw_params():
        if genesis_offset_days is None:
            st.session_state[KEY_GENESIS_OFFSET] = int(opt_offset)
        st.session_state[a_key] = float(reset_a)
        st.session_state[b_key] = float(reset_b)

    def auto_fit_intercept():
        best_a, _ = optimize_single_powerlaw_parameter(
            all_abs_days,
            all_log_close,
            current_offset(),
            float(st.session_state.get(a_key, opt_a)),
            float(st.session_state.get(b_key, opt_b)),
            "A",
            a_min=a_min,
            a_max=a_max,
            b_min=b_min,
            b_max=b_max,
        )
        st.session_state[a_key] = float(best_a)

    def auto_fit_slope():
        best_b, _ = optimize_single_powerlaw_parameter(
            all_abs_days,
            all_log_close,
            current_offset(),
            float(st.session_state.get(a_key, opt_a)),
            float(st.session_state.get(b_key, opt_b)),
            "B",
            a_min=a_min,
            a_max=a_max,
            b_min=b_min,
            b_max=b_max,
        )
        st.session_state[b_key] = float(best_b)

    def auto_fit_model():
        _, best_a, best_b, _ = find_best_fit_params_for_offset(
            all_abs_days,
            all_log_close,
            current_offset(),
        )
        if best_a == 0.0 and best_b == 0.0:
            return
        st.session_state[a_key] = round(float(best_a), 3)
        st.session_state[b_key] = round(float(best_b), 3)

    st.markdown("**A (Intercept)**")
    fancy_control(
        "A (Intercept)",
        a_key,
        0.001,
        float(a_min),
        float(a_max),
        on_auto_fit=auto_fit_intercept,
        auto_fit_label="AF",
    )

    st.markdown("**B (Slope)**")
    fancy_control(
        "B (Slope)",
        b_key,
        0.001,
        float(b_min),
        float(b_max),
        on_auto_fit=auto_fit_slope,
        auto_fit_label="AF",
    )

    display_r2 = calculate_r2_for_manual_params(
        all_abs_days,
        all_log_close,
        current_offset(),
        float(st.session_state.get(a_key, opt_a)),
        float(st.session_state.get(b_key, opt_b)),
    )
    _, _, params_were_clipped = evaluate_powerlaw_values(
        np.log10(
            np.maximum(
                all_abs_days - current_offset(),
                1.0,
            )
        ),
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
    if display_r2 < 0 or params_were_clipped:
        st.caption(
            "Current A/B values are outside a stable operating range. Use Auto-fit model or Reset parameters."
        )

    st.button("Auto-fit model", use_container_width=True, on_click=auto_fit_model)
    st.button("Reset parameters", use_container_width=True, on_click=reset_powerlaw_params)
    if callable(render_after_actions):
        render_after_actions()

    return display_r2
