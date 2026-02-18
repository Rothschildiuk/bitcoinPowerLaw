import numpy as np
import streamlit as st

from utils import fancy_control, calculate_r2_score, get_stable_trend_fit

AUTO_FIT_MAX_PASSES = 2
AUTO_FIT_GRID_POINTS = 9


def fit_oscillator_component(
    log_days,
    residual_series,
    t1_cycle_age_years,
    cycle_lambda,
    top_amplitude_factor,
    bottom_amplitude_factor,
    impulse_damping,
):
    log_lambda = np.log10(cycle_lambda)
    if abs(log_lambda) <= 1e-9:
        return None

    t1_log_days = np.log10(t1_cycle_age_years * 365.25)
    angular_frequency = 2 * np.pi / log_lambda
    phase_shift = -angular_frequency * t1_log_days
    phase_values = angular_frequency * log_days + phase_shift

    base_wave = build_oscillator_wave(phase_values)
    decay_factor = compute_impulse_decay(log_days, impulse_damping, float(np.min(log_days)))
    decayed_wave = base_wave * decay_factor

    asymmetric_template = np.where(
        decayed_wave > 0, decayed_wave * top_amplitude_factor, decayed_wave
    )
    asymmetric_template = np.where(
        decayed_wave < 0, asymmetric_template * bottom_amplitude_factor, asymmetric_template
    )
    template_energy = np.dot(asymmetric_template, asymmetric_template)
    if template_energy <= 1e-12:
        return None

    fitted_amplitude = abs(np.dot(residual_series, asymmetric_template) / template_energy)
    predicted_residuals = fitted_amplitude * asymmetric_template
    return fitted_amplitude, angular_frequency, phase_shift, predicted_residuals


def compute_oscillator_fit_r2(
    log_days,
    residual_series,
    t1_cycle_age_years,
    cycle_lambda,
    top_amplitude_factor,
    bottom_amplitude_factor,
    impulse_damping,
):
    fit_result = fit_oscillator_component(
        log_days,
        residual_series,
        t1_cycle_age_years,
        cycle_lambda,
        top_amplitude_factor,
        bottom_amplitude_factor,
        impulse_damping,
    )
    if fit_result is None:
        return -1e9

    _, _, _, predicted_residuals = fit_result
    return calculate_r2_score(residual_series, predicted_residuals) * 100.0


def optimize_oscillator_parameters(log_days, residual_series, initial_params):
    optimized_params = dict(initial_params)
    bounds = {
        "t1_age": (0.1, 5.0),
        "lambda_val": (1.5, 3.0),
        "amp_factor_top": (0.1, 10.0),
        "amp_factor_bottom": (0.1, 10.0),
        "impulse_damping": (0.0, 2.0),
    }
    spans = {
        "t1_age": 1.20,
        "lambda_val": 0.60,
        "amp_factor_top": 1.60,
        "amp_factor_bottom": 1.60,
        "impulse_damping": 1.00,
    }
    order = ["t1_age", "lambda_val", "amp_factor_top", "amp_factor_bottom", "impulse_damping"]

    best_r2 = compute_oscillator_fit_r2(
        log_days,
        residual_series,
        optimized_params["t1_age"],
        optimized_params["lambda_val"],
        optimized_params["amp_factor_top"],
        optimized_params["amp_factor_bottom"],
        optimized_params["impulse_damping"],
    )

    for _ in range(AUTO_FIT_MAX_PASSES):
        improved = False
        for key in order:
            lo_b, hi_b = bounds[key]
            center = optimized_params[key]
            lo = max(lo_b, center - spans[key])
            hi = min(hi_b, center + spans[key])
            candidates = np.linspace(lo, hi, AUTO_FIT_GRID_POINTS)

            local_best_val = center
            local_best_r2 = best_r2
            for cand in candidates:
                trial = dict(optimized_params)
                trial[key] = float(cand)
                score = compute_oscillator_fit_r2(
                    log_days,
                    residual_series,
                    trial["t1_age"],
                    trial["lambda_val"],
                    trial["amp_factor_top"],
                    trial["amp_factor_bottom"],
                    trial["impulse_damping"],
                )
                if score > local_best_r2:
                    local_best_r2 = score
                    local_best_val = float(cand)

            if local_best_val != center:
                optimized_params[key] = local_best_val
                best_r2 = local_best_r2
                improved = True

        for k in spans:
            spans[k] *= 0.45
        if not improved:
            break

    return optimized_params


def build_autofit_signature(all_abs_days, all_log_close):
    if len(all_abs_days) == 0:
        return ("empty",)
    return (
        int(st.session_state.get("genesis_offset", 0)),
        round(float(st.session_state.get("A", 0.0)), 6),
        round(float(st.session_state.get("B", 0.0)), 6),
        len(all_abs_days),
        int(all_abs_days[0]),
        int(all_abs_days[-1]),
        round(float(all_log_close[0]), 6),
        round(float(all_log_close[-1]), 6),
    )


# --- OSCILLATOR MATH (SINUSOID) ---
def build_oscillator_wave(phase_values):
    """
    Calculates a Cosine wave (Sinusoid) instead of Inverted Cycloid.
    Using Cosine ensures the peak corresponds to phase=0 (consistent with t1_age).
    Range: [-1, 1]
    """
    return np.cos(phase_values)


def compute_impulse_decay(log_days, damping_factor, reference_log_day):
    relative_position = np.maximum(0.0, log_days - reference_log_day)
    return np.exp(-damping_factor * relative_position)


def build_oscillator_curve(
    log_days,
    amplitude,
    angular_frequency,
    phase_shift,
    top_amplitude_factor,
    bottom_amplitude_factor,
    damping_factor=0.0,
    reference_log_day=None,
):
    phase_values = angular_frequency * log_days + phase_shift
    base_wave = build_oscillator_wave(phase_values)
    if reference_log_day is None:
        reference_log_day = float(np.min(log_days))
    decay_factor = compute_impulse_decay(log_days, damping_factor, reference_log_day)
    base_wave = base_wave * decay_factor

    # Base wave amplitude scaling
    y_values = amplitude * base_wave
    positive_mask = base_wave > 0
    negative_mask = base_wave < 0
    y_values[positive_mask] *= top_amplitude_factor
    y_values[negative_mask] *= bottom_amplitude_factor
    return y_values


# Backward-compatible aliases for existing call sites.
safe_r2 = calculate_r2_score
_oscillator_r2 = compute_oscillator_fit_r2
_auto_fit_oscillator = optimize_oscillator_parameters
get_oscillator_wave = build_oscillator_wave
get_impulse_decay = compute_impulse_decay
oscillator_func_manual = build_oscillator_curve


# --- SIDEBAR RENDERER ---
def render_sidebar(all_abs_days, all_log_close, text_color):
    # Defaults
    defaults = {
        "t1_age": 2.49,
        "lambda_val": 2.01,
        "amp_factor_top": 1.16,
        "amp_factor_bottom": 0.88,
        "impulse_damping": 1.72,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    def reset_oscillator_params():
        for k, v in defaults.items():
            st.session_state[k] = v
        st.session_state.pop("osc_autofit_signature", None)
        st.session_state.pop("osc_autofit_best_params", None)

    auto_fit = st.checkbox(
        "Auto-Fit Oscillator",
        value=False,
        help="Automatically finds oscillator parameters with the highest Oscillator R².",
    )

    days_since_genesis = all_abs_days - st.session_state.get("genesis_offset", 0)
    valid_days_mask = days_since_genesis > 0
    oscillator_r2_display = 0.0

    if np.count_nonzero(valid_days_mask) > 100:
        log_days = np.log10(days_since_genesis[valid_days_mask])
        _, _, trend_log_prices, residual_series = get_stable_trend_fit(
            log_days,
            all_log_close[valid_days_mask],
            float(st.session_state.A),
            float(st.session_state.B),
        )

        if auto_fit:
            autofit_signature = build_autofit_signature(all_abs_days, all_log_close)
            cached_signature = st.session_state.get("osc_autofit_signature")
            cached_best_params = st.session_state.get("osc_autofit_best_params")

            if cached_signature == autofit_signature and isinstance(cached_best_params, dict):
                best_params = cached_best_params
            else:
                start_params = {
                    "t1_age": float(st.session_state.get("t1_age", defaults["t1_age"])),
                    "lambda_val": float(st.session_state.get("lambda_val", defaults["lambda_val"])),
                    "amp_factor_top": float(
                        st.session_state.get("amp_factor_top", defaults["amp_factor_top"])
                    ),
                    "amp_factor_bottom": float(
                        st.session_state.get("amp_factor_bottom", defaults["amp_factor_bottom"])
                    ),
                    "impulse_damping": float(
                        st.session_state.get("impulse_damping", defaults["impulse_damping"])
                    ),
                }
                best_params = optimize_oscillator_parameters(
                    log_days, residual_series, start_params
                )
                st.session_state["osc_autofit_signature"] = autofit_signature
                st.session_state["osc_autofit_best_params"] = best_params

            st.session_state["t1_age"] = round(best_params["t1_age"], 3)
            st.session_state["lambda_val"] = round(best_params["lambda_val"], 3)
            st.session_state["amp_factor_top"] = round(best_params["amp_factor_top"], 3)
            st.session_state["amp_factor_bottom"] = round(best_params["amp_factor_bottom"], 3)
            st.session_state["impulse_damping"] = round(best_params["impulse_damping"], 3)

    st.markdown("**1st Cycle Age**")
    fancy_control("1st Cycle Age", "t1_age", 0.01, 0.1, 5.0, disabled=auto_fit)

    st.markdown("**Lambda**")
    fancy_control("Lambda", "lambda_val", 0.01, 1.5, 3.0, disabled=auto_fit)

    st.markdown("**Top Amplitude**")
    fancy_control("Top Amplitude", "amp_factor_top", 0.01, 0.1, 10.0, disabled=auto_fit)

    st.markdown("**Bottom Amplitude**")
    fancy_control("Bottom Amplitude", "amp_factor_bottom", 0.01, 0.1, 10.0, disabled=auto_fit)

    st.markdown("**Impulse Damping**")
    fancy_control("Impulse Damping", "impulse_damping", 0.01, 0.0, 2.0, disabled=auto_fit)

    # --- R2 Calculation for Sidebar Display ---
    if np.count_nonzero(valid_days_mask) > 100:
        oscillator_r2_display = compute_oscillator_fit_r2(
            log_days,
            residual_series,
            st.session_state.get("t1_age", defaults["t1_age"]),
            st.session_state.get("lambda_val", defaults["lambda_val"]),
            st.session_state.get("amp_factor_top", defaults["amp_factor_top"]),
            st.session_state.get("amp_factor_bottom", defaults["amp_factor_bottom"]),
            st.session_state.get("impulse_damping", defaults["impulse_damping"]),
        )

    st.markdown(
        f"<p style='color:{text_color}; margin-top: 2px;'>"
        f"Oscillator R² = {oscillator_r2_display:.4f}%</p>",
        unsafe_allow_html=True,
    )

    st.button("Reset parameters", use_container_width=True, on_click=reset_oscillator_params)
