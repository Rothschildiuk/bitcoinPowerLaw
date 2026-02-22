import streamlit as st

from core import oscillator, power_law
from core.constants import (
    CURRENCY_DOLLAR,
    CURRENCY_OPTIONS,
    DEFAULT_FORECAST_HORIZON,
    KEY_CHART_REVISION,
    KEY_CURRENCY_SELECTOR,
    KEY_LAST_MODE,
    KEY_MODE_SELECTOR,
    KEY_PORTFOLIO_BTC_AMOUNT,
    KEY_PORTFOLIO_FORECAST_HORIZON,
    KEY_PORTFOLIO_FORECAST_MONTHS_LEGACY,
    KEY_PORTFOLIO_FORECAST_UNIT,
    KEY_TIME_SCALE,
    MODE_LOGPERIODIC,
    MODE_PORTFOLIO,
    MODE_POWERLAW,
    TIME_LOG,
    TIME_LIN,
)
from core.utils import inline_radio_control


def _render_portfolio_sidebar_controls(forecast_horizon_min, forecast_horizon_max):
    st.markdown("**BTC quantity**")
    st.number_input(
        "BTC quantity",
        min_value=0.0,
        value=float(st.session_state.get(KEY_PORTFOLIO_BTC_AMOUNT, 1.0)),
        step=0.01,
        format="%.2f",
        key=KEY_PORTFOLIO_BTC_AMOUNT,
        label_visibility="collapsed",
    )
    st.markdown("**Forecast unit**")
    forecast_unit = st.radio(
        "Forecast unit",
        ["Day", "Month", "Year"],
        horizontal=True,
        key=KEY_PORTFOLIO_FORECAST_UNIT,
        label_visibility="collapsed",
    )
    default_horizon = int(
        st.session_state.get(
            KEY_PORTFOLIO_FORECAST_HORIZON,
            st.session_state.get(KEY_PORTFOLIO_FORECAST_MONTHS_LEGACY, DEFAULT_FORECAST_HORIZON),
        )
    )
    st.session_state[KEY_PORTFOLIO_FORECAST_HORIZON] = max(
        forecast_horizon_min, min(forecast_horizon_max, default_horizon)
    )

    horizon_label_map = {"Day": "days", "Month": "months", "Year": "years"}
    horizon_label = horizon_label_map.get(forecast_unit, "months")
    st.markdown(f"**Forecast horizon ({horizon_label})**")

    def on_horizon_minus():
        st.session_state[KEY_PORTFOLIO_FORECAST_HORIZON] = max(
            forecast_horizon_min, int(st.session_state[KEY_PORTFOLIO_FORECAST_HORIZON]) - 1
        )

    def on_horizon_plus():
        st.session_state[KEY_PORTFOLIO_FORECAST_HORIZON] = min(
            forecast_horizon_max, int(st.session_state[KEY_PORTFOLIO_FORECAST_HORIZON]) + 1
        )

    h1, h2, h3 = st.columns([1, 2.5, 1])
    h1.button("➖", key="portfolio_horizon_m", on_click=on_horizon_minus)
    h3.button("➕", key="portfolio_horizon_p", on_click=on_horizon_plus)
    h2.slider(
        f"Forecast horizon ({horizon_label})",
        min_value=forecast_horizon_min,
        max_value=forecast_horizon_max,
        step=1,
        key=KEY_PORTFOLIO_FORECAST_HORIZON,
        label_visibility="collapsed",
    )


def render_sidebar_panel(
    all_absolute_days,
    all_log_close_prices,
    c_text_main,
    app_version,
    forecast_horizon_min,
    forecast_horizon_max,
):
    with st.sidebar:
        st.markdown("<div class='app-title'>Bitcoin PowerLaw</div>", unsafe_allow_html=True)
        st.caption(f"Version {app_version}")

        mode_options = [MODE_POWERLAW, MODE_LOGPERIODIC, MODE_PORTFOLIO]
        if st.session_state.get(KEY_MODE_SELECTOR) not in mode_options:
            st.session_state[KEY_MODE_SELECTOR] = st.session_state.get(KEY_LAST_MODE, MODE_POWERLAW)
        mode = st.segmented_control(
            "Mode",
            mode_options,
            selection_mode="single",
            key=KEY_MODE_SELECTOR,
            width="stretch",
        )
        if mode is None:
            fallback_mode = st.session_state.get(KEY_LAST_MODE, MODE_POWERLAW)
            if fallback_mode not in mode_options:
                fallback_mode = MODE_POWERLAW
            st.session_state[KEY_MODE_SELECTOR] = fallback_mode
            st.rerun()
        if mode not in mode_options:
            mode = st.session_state.get(KEY_LAST_MODE, MODE_POWERLAW)
            if mode not in mode_options:
                mode = MODE_POWERLAW
            st.session_state[KEY_MODE_SELECTOR] = mode
            st.rerun()
        if mode != st.session_state[KEY_LAST_MODE]:
            st.session_state[KEY_CHART_REVISION] += 1
            st.session_state[KEY_LAST_MODE] = mode

        selected_currency = st.session_state.get(KEY_CURRENCY_SELECTOR, CURRENCY_DOLLAR)
        if selected_currency not in CURRENCY_OPTIONS:
            selected_currency = CURRENCY_DOLLAR
            st.session_state[KEY_CURRENCY_SELECTOR] = selected_currency

        currency = st.radio(
            "Currency",
            CURRENCY_OPTIONS,
            horizontal=True,
            index=CURRENCY_OPTIONS.index(selected_currency),
            key=KEY_CURRENCY_SELECTOR,
        )

        if mode != MODE_PORTFOLIO:
            time_scale = inline_radio_control("Time", [TIME_LOG, TIME_LIN], key=KEY_TIME_SCALE)
        else:
            time_scale = st.session_state.get(KEY_TIME_SCALE, TIME_LOG)

        current_r2 = 0.0
        price_scale = "Log"

        if mode in [MODE_POWERLAW, MODE_PORTFOLIO]:
            price_scale, current_r2 = power_law.render_sidebar(
                all_absolute_days,
                all_log_close_prices,
                c_text_main,
                show_price_scale=(mode != MODE_PORTFOLIO),
                render_extra_controls=(
                    lambda: (
                        _render_portfolio_sidebar_controls(
                            forecast_horizon_min, forecast_horizon_max
                        )
                        if mode == MODE_PORTFOLIO
                        else None
                    )
                ),
            )
        else:
            oscillator.render_sidebar(all_absolute_days, all_log_close_prices, c_text_main)

    return mode, currency, time_scale, price_scale, current_r2
