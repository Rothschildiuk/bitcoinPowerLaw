import streamlit as st
import numpy as np
from utils import fancy_control

def safe_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot <= 1e-12:
        return 0.0
    return 1 - (ss_res / ss_tot)

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

    st.markdown("**1st Cycle Age**")
    fancy_control("1st Cycle Age", "t1_age", 0.01, 0.1, 5.0)

    st.markdown("**Lambda**")
    fancy_control("Lambda", "lambda_val", 0.01, 1.5, 3.0)

    st.markdown("**Top Amplitude**")
    fancy_control("Top Amplitude", "amp_factor_top", 0.01, 0.1, 10.0)

    st.markdown("**Bottom Amplitude**")
    fancy_control("Bottom Amplitude", "amp_factor_bottom", 0.01, 0.1, 10.0)

    st.markdown("**Impulse Damping**")
    fancy_control("Impulse Damping", "impulse_damping", 0.01, 0.0, 2.0)

    # --- R2 Calculation for Sidebar Display ---
    calc_days = all_abs_days - st.session_state.get("genesis_offset", 0)
    mask_calc = calc_days > 0
    osc_r2_display = 0.0

    if np.sum(mask_calc) > 100:
        c_log_d = np.log10(calc_days[mask_calc])
        # Current Trend
        c_model_log = st.session_state.A + st.session_state.B * c_log_d
        c_res = all_log_close[mask_calc] - c_model_log

        # Fallback if session trend is clearly invalid.
        median_abs_res = float(np.median(np.abs(c_res)))
        if (not np.isfinite(median_abs_res)) or median_abs_res > 5.0:
            fit_b, fit_a = np.polyfit(c_log_d, all_log_close[mask_calc], 1)
            c_model_log = fit_a + fit_b * c_log_d
            c_res = all_log_close[mask_calc] - c_model_log

        l_log = np.log10(st.session_state.lambda_val)
        t1_log = np.log10(st.session_state.t1_age * 365.25)
        osc_omega = 2 * np.pi / l_log
        osc_phi = -osc_omega * t1_log

        calc_phase = osc_omega * c_log_d + osc_phi

        # New Sinusoid Logic
        u_wave = get_oscillator_wave(calc_phase)
        decay = get_impulse_decay(c_log_d, st.session_state.get("impulse_damping", 0.0), float(np.min(c_log_d)))
        u_wave = u_wave * decay
        num = np.dot(c_res, u_wave)
        den = np.dot(u_wave, u_wave)

        # FORCE POSITIVE AMPLITUDE
        amp = abs(num / den) if den > 1e-9 else 0

        c_osc_pred = amp * u_wave
        c_osc_pred = np.where(u_wave > 0, c_osc_pred * st.session_state.amp_factor_top, c_osc_pred)
        c_osc_pred = np.where(u_wave < 0, c_osc_pred * st.session_state.amp_factor_bottom, c_osc_pred)

        osc_r2_display = safe_r2(c_res, c_osc_pred) * 100

    st.markdown(
        f"<p style='color:{text_color}; margin-top: 2px;'>"
        f"Oscillator R² = {osc_r2_display:.4f}%</p>",
        unsafe_allow_html=True)

    st.button("Reset parameters", use_container_width=True, on_click=reset_oscillator_params)
