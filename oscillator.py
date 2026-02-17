import streamlit as st
import numpy as np
from utils import fancy_control

def safe_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot <= 1e-12:
        return 0.0
    return 1 - (ss_res / ss_tot)

def _oscillator_r2(c_log_d, c_res, t1_age, lambda_val, amp_top, amp_bottom, damping):
    l_log = np.log10(lambda_val)
    if abs(l_log) <= 1e-9:
        return -1e9

    t1_log = np.log10(t1_age * 365.25)
    osc_omega = 2 * np.pi / l_log
    osc_phi = -osc_omega * t1_log
    phase = osc_omega * c_log_d + osc_phi

    u_wave = get_oscillator_wave(phase)
    decay = get_impulse_decay(c_log_d, damping, float(np.min(c_log_d)))
    base = u_wave * decay

    template = np.where(base > 0, base * amp_top, base)
    template = np.where(base < 0, template * amp_bottom, template)
    den = np.dot(template, template)
    if den <= 1e-12:
        return -1e9

    amp = abs(np.dot(c_res, template) / den)
    pred = amp * template
    return safe_r2(c_res, pred) * 100.0

def _auto_fit_oscillator(c_log_d, c_res, initial_params):
    params = dict(initial_params)
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

    best_r2 = _oscillator_r2(
        c_log_d, c_res,
        params["t1_age"], params["lambda_val"],
        params["amp_factor_top"], params["amp_factor_bottom"], params["impulse_damping"]
    )

    for _ in range(3):
        improved = False
        for key in order:
            lo_b, hi_b = bounds[key]
            center = params[key]
            lo = max(lo_b, center - spans[key])
            hi = min(hi_b, center + spans[key])
            candidates = np.linspace(lo, hi, 11)

            local_best_val = center
            local_best_r2 = best_r2
            for cand in candidates:
                trial = dict(params)
                trial[key] = float(cand)
                score = _oscillator_r2(
                    c_log_d, c_res,
                    trial["t1_age"], trial["lambda_val"],
                    trial["amp_factor_top"], trial["amp_factor_bottom"], trial["impulse_damping"]
                )
                if score > local_best_r2:
                    local_best_r2 = score
                    local_best_val = float(cand)

            if local_best_val != center:
                params[key] = local_best_val
                best_r2 = local_best_r2
                improved = True

        for k in spans:
            spans[k] *= 0.45
        if not improved:
            break

    return params

# --- OSCILLATOR MATH (SINUSOID) ---
def get_oscillator_wave(phase_array):
    """
    Calculates a Cosine wave (Sinusoid) instead of Inverted Cycloid.
    Using Cosine ensures the peak corresponds to phase=0 (consistent with t1_age).
    Range: [-1, 1]
    """
    return np.cos(phase_array)

def get_impulse_decay(log_x, damping, ref_log):
    rel = np.maximum(0.0, log_x - ref_log)
    return np.exp(-damping * rel)

def oscillator_func_manual(x_log, amp, omega, phi, f_top, f_bot, damping=0.0, ref_log=None):
    phase = omega * x_log + phi
    base_wave = get_oscillator_wave(phase)
    if ref_log is None:
        ref_log = float(np.min(x_log))
    decay = get_impulse_decay(x_log, damping, ref_log)
    base_wave = base_wave * decay

    # Base wave amplitude scaling
    y = amp * base_wave

    # Asymmetric amplitude (Top vs Bottom)
    # Since cosine goes [-1, 1], positive values are "tops", negative are "bottoms"
    y = np.where(base_wave > 0, y * f_top, y)
    y = np.where(base_wave < 0, y * f_bot, y)
    return y

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
        if k not in st.session_state: st.session_state[k] = v

    def reset_oscillator_params():
        for k, v in defaults.items():
            st.session_state[k] = v

    auto_fit = st.checkbox(
        "Auto-Fit Oscillator",
        value=False,
        help="Automatically finds oscillator parameters with the highest Oscillator R²."
    )

    calc_days = all_abs_days - st.session_state.get("genesis_offset", 0)
    mask_calc = calc_days > 0
    osc_r2_display = 0.0

    if np.sum(mask_calc) > 100:
        c_log_d = np.log10(calc_days[mask_calc])
        c_model_log = st.session_state.A + st.session_state.B * c_log_d
        c_res = all_log_close[mask_calc] - c_model_log

        # Fallback if session trend is clearly invalid.
        median_abs_res = float(np.median(np.abs(c_res)))
        if (not np.isfinite(median_abs_res)) or median_abs_res > 5.0:
            fit_b, fit_a = np.polyfit(c_log_d, all_log_close[mask_calc], 1)
            c_model_log = fit_a + fit_b * c_log_d
            c_res = all_log_close[mask_calc] - c_model_log

        if auto_fit:
            start_params = {
                "t1_age": float(st.session_state.get("t1_age", defaults["t1_age"])),
                "lambda_val": float(st.session_state.get("lambda_val", defaults["lambda_val"])),
                "amp_factor_top": float(st.session_state.get("amp_factor_top", defaults["amp_factor_top"])),
                "amp_factor_bottom": float(st.session_state.get("amp_factor_bottom", defaults["amp_factor_bottom"])),
                "impulse_damping": float(st.session_state.get("impulse_damping", defaults["impulse_damping"])),
            }
            best_params = _auto_fit_oscillator(c_log_d, c_res, start_params)
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
    if np.sum(mask_calc) > 100:
        osc_r2_display = _oscillator_r2(
            c_log_d, c_res,
            st.session_state.get("t1_age", defaults["t1_age"]),
            st.session_state.get("lambda_val", defaults["lambda_val"]),
            st.session_state.get("amp_factor_top", defaults["amp_factor_top"]),
            st.session_state.get("amp_factor_bottom", defaults["amp_factor_bottom"]),
            st.session_state.get("impulse_damping", defaults["impulse_damping"]),
        )

    st.markdown(
        f"<p style='color:{text_color}; margin-top: 2px;'>"
        f"Oscillator R² = {osc_r2_display:.4f}%</p>",
        unsafe_allow_html=True)

    st.button("Reset parameters", use_container_width=True, on_click=reset_oscillator_params)
