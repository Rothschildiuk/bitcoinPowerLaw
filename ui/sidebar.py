import streamlit as st

from core import oscillator, power_law
from core.constants import (
    BAND_METHOD_OPTIONS,
    BAND_METHOD_QUANTILE,
    CURRENCY_DOLLAR,
    CURRENCY_OPTIONS,
    DEFAULT_FORECAST_HORIZON,
    KEY_A,
    KEY_B,
    KEY_BAND_METHOD,
    KEY_CHART_REVISION,
    KEY_CURRENCY_SELECTOR,
    KEY_LAST_MODE,
    KEY_LOGPERIODIC_LAST_SERIES,
    KEY_MODE_SELECTOR,
    KEY_LOGPERIODIC_SERIES,
    KEY_POWERLAW_SERIES,
    KEY_PORTFOLIO_BTC_AMOUNT,
    KEY_PORTFOLIO_FORECAST_HORIZON,
    KEY_PORTFOLIO_FORECAST_MONTHS_LEGACY,
    KEY_PORTFOLIO_FORECAST_UNIT,
    KEY_TIME_SCALE,
    MODE_LOGPERIODIC,
    MODE_PORTFOLIO,
    MODE_POWERLAW,
    POWERLAW_SERIES_PRICE,
    TIME_LOG,
    TIME_LIN,
)
from core.series_registry import (
    get_active_model_config,
    get_logperiodic_series_options,
    get_powerlaw_series_options,
    series_supports_currency_selector,
)


def _render_portfolio_sidebar_controls(forecast_horizon_min, forecast_horizon_max):
    st.markdown("**BTC quantity**")
    st.number_input(
        "BTC quantity",
        min_value=0.0,
        value=float(st.session_state.get(KEY_PORTFOLIO_BTC_AMOUNT, 2.0)),
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
            forecast_horizon_min,
            int(st.session_state[KEY_PORTFOLIO_FORECAST_HORIZON]) - 1,
        )

    def on_horizon_plus():
        st.session_state[KEY_PORTFOLIO_FORECAST_HORIZON] = min(
            forecast_horizon_max,
            int(st.session_state[KEY_PORTFOLIO_FORECAST_HORIZON]) + 1,
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
    sidebar_series_data,
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

        powerlaw_series_options = get_powerlaw_series_options()
        logperiodic_series_options = get_logperiodic_series_options()

        powerlaw_series = st.session_state.get(KEY_POWERLAW_SERIES, POWERLAW_SERIES_PRICE)
        if powerlaw_series not in powerlaw_series_options:
            powerlaw_series = POWERLAW_SERIES_PRICE
            st.session_state[KEY_POWERLAW_SERIES] = powerlaw_series
        if mode == MODE_POWERLAW:
            powerlaw_series = st.radio(
                "PowerLaw series",
                powerlaw_series_options,
                horizontal=True,
                key=KEY_POWERLAW_SERIES,
            )
            if powerlaw_series is None:
                powerlaw_series = st.session_state.get(KEY_POWERLAW_SERIES, POWERLAW_SERIES_PRICE)
                st.session_state[KEY_POWERLAW_SERIES] = powerlaw_series
                st.rerun()
        logperiodic_series = st.session_state.get(KEY_LOGPERIODIC_SERIES, POWERLAW_SERIES_PRICE)
        if logperiodic_series not in logperiodic_series_options:
            logperiodic_series = POWERLAW_SERIES_PRICE
            st.session_state[KEY_LOGPERIODIC_SERIES] = logperiodic_series
        if mode == MODE_LOGPERIODIC:
            logperiodic_series = st.radio(
                "LogPeriodic series",
                logperiodic_series_options,
                horizontal=True,
                key=KEY_LOGPERIODIC_SERIES,
                width="stretch",
            )
            if logperiodic_series is None:
                logperiodic_series = st.session_state.get(
                    KEY_LOGPERIODIC_SERIES, POWERLAW_SERIES_PRICE
                )
                st.session_state[KEY_LOGPERIODIC_SERIES] = logperiodic_series
                st.rerun()

        time_scale = st.session_state.get(KEY_TIME_SCALE, TIME_LOG)
        price_scale = st.session_state.get("price_scale_selector", "Log")
        powerlaw_model = get_active_model_config(
            MODE_POWERLAW,
            powerlaw_series,
            logperiodic_series,
            selected_currency=selected_currency,
        )
        hide_price_scale = mode == MODE_POWERLAW and powerlaw_model.lock_price_scale_to_log

        if mode == MODE_POWERLAW:
            time_col, price_col = st.columns(2)
            with time_col:
                st.markdown("**Time**")
                time_scale = st.radio(
                    "Time",
                    [TIME_LOG, TIME_LIN],
                    horizontal=True,
                    key=KEY_TIME_SCALE,
                    label_visibility="collapsed",
                    width="stretch",
                )
            with price_col:
                st.markdown("**Price**")
                if hide_price_scale:
                    price_scale = "Log"
                    st.session_state["price_scale_selector"] = "Log"
                    st.radio(
                        "Price",
                        ["Log"],
                        horizontal=True,
                        key="price_scale_series_fixed",
                        label_visibility="collapsed",
                        disabled=True,
                        width="stretch",
                    )
                else:
                    price_scale = st.radio(
                        "Price",
                        ["Log", "Lin"],
                        horizontal=True,
                        key="price_scale_selector",
                        label_visibility="collapsed",
                        width="stretch",
                    )

        is_non_price_series = not series_supports_currency_selector(
            mode, powerlaw_series, logperiodic_series
        )

        if is_non_price_series:
            currency = CURRENCY_DOLLAR
            st.session_state[KEY_CURRENCY_SELECTOR] = CURRENCY_DOLLAR
            st.markdown("**Currency**")
            st.caption("This series does not use currency conversion.")
        else:
            currency = st.radio(
                "Currency",
                CURRENCY_OPTIONS,
                horizontal=True,
                index=CURRENCY_OPTIONS.index(selected_currency),
                key=KEY_CURRENCY_SELECTOR,
                width="stretch",
            )

        band_method = st.session_state.get(KEY_BAND_METHOD, BAND_METHOD_QUANTILE)
        if band_method not in BAND_METHOD_OPTIONS:
            band_method = BAND_METHOD_QUANTILE
            st.session_state[KEY_BAND_METHOD] = band_method
        if mode == MODE_POWERLAW:
            band_method = st.radio(
                "Band model",
                BAND_METHOD_OPTIONS,
                horizontal=True,
                key=KEY_BAND_METHOD,
                width="stretch",
            )
        else:
            st.session_state[KEY_BAND_METHOD] = BAND_METHOD_QUANTILE
            band_method = BAND_METHOD_QUANTILE

        current_r2 = 0.0
        a_min = -35.0
        a_max = 0.0
        b_min = 1.0
        b_max = 12.0
        active_model = get_active_model_config(
            mode,
            powerlaw_series,
            logperiodic_series,
            selected_currency=currency,
        )
        active_series_data = sidebar_series_data[active_model.series_name]
        model_abs_days = active_series_data["absolute_days"]
        model_log_close = active_series_data["log_close"]
        a_key = active_model.a_key
        b_key = active_model.b_key
        default_a = active_model.default_a
        default_b = active_model.default_b

        if mode in [MODE_POWERLAW, MODE_PORTFOLIO]:
            current_r2 = power_law.render_sidebar(
                model_abs_days,
                model_log_close,
                c_text_main,
                render_extra_controls=(
                    lambda: (
                        _render_portfolio_sidebar_controls(
                            forecast_horizon_min, forecast_horizon_max
                        )
                        if mode == MODE_PORTFOLIO
                        else None
                    )
                ),
                a_key=a_key,
                b_key=b_key,
                default_a=default_a,
                default_b=default_b,
                a_min=a_min,
                a_max=a_max,
                b_min=b_min,
                b_max=b_max,
            )
        else:
            # Keep legacy A/B keys aligned before rendering LogPeriodic controls,
            # so sidebar R² is computed for the newly selected series immediately.
            st.session_state[KEY_A] = float(st.session_state.get(a_key, default_a))
            st.session_state[KEY_B] = float(st.session_state.get(b_key, default_b))
            active_osc_defaults = active_model.oscillator_defaults
            last_lp_series = st.session_state.get(KEY_LOGPERIODIC_LAST_SERIES)
            if last_lp_series != logperiodic_series:
                for k, v in active_osc_defaults.items():
                    st.session_state[k] = v
                st.session_state[KEY_LOGPERIODIC_LAST_SERIES] = logperiodic_series

            oscillator.render_sidebar(
                model_abs_days,
                model_log_close,
                c_text_main,
                defaults_override=active_osc_defaults,
                min_abs_day_for_fit=active_model.oscillator_min_abs_day,
            )

    return (
        mode,
        currency,
        time_scale,
        price_scale,
        current_r2,
        powerlaw_series,
        logperiodic_series,
        band_method,
    )
