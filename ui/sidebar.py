import streamlit as st

from core import oscillator, power_law
from core.constants import (
    BAND_METHOD_OPTIONS,
    BAND_METHOD_QUANTILE,
    CURRENCY_DOLLAR,
    CURRENCY_EURO,
    CURRENCY_GOLD,
    CURRENCY_OPTIONS,
    DEFAULT_A,
    DEFAULT_B,
    DEFAULT_EURO_A,
    DEFAULT_EURO_B,
    DEFAULT_GOLD_A,
    DEFAULT_GOLD_B,
    DEFAULT_HASHRATE_A,
    DEFAULT_HASHRATE_B,
    DEFAULT_DIFFICULTY_A,
    DEFAULT_DIFFICULTY_B,
    OSCILLATOR_DIFF_HASH_START_ABS_DAYS,
    DEFAULT_REVENUE_A,
    DEFAULT_REVENUE_B,
    DEFAULT_FORECAST_HORIZON,
    KEY_A,
    KEY_A_PRICE,
    KEY_A_EURO,
    KEY_A_GOLD,
    KEY_A_HASHRATE,
    KEY_A_DIFFICULTY,
    KEY_A_REVENUE,
    KEY_B,
    KEY_BAND_METHOD,
    KEY_B_PRICE,
    KEY_B_EURO,
    KEY_B_GOLD,
    KEY_B_HASHRATE,
    KEY_B_DIFFICULTY,
    KEY_B_REVENUE,
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
    LOGPERIODIC_SERIES_OPTIONS,
    MODE_LOGPERIODIC,
    MODE_PORTFOLIO,
    MODE_POWERLAW,
    OSC_DEFAULTS,
    OSC_DEFAULTS_DIFFICULTY,
    OSC_DEFAULTS_HASHRATE,
    POWERLAW_SERIES_OPTIONS,
    POWERLAW_SERIES_DIFFICULTY,
    POWERLAW_SERIES_HASHRATE,
    POWERLAW_SERIES_PRICE,
    POWERLAW_SERIES_REVENUE,
    TIME_LOG,
    TIME_LIN,
)
from core.utils import inline_radio_control


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
    price_absolute_days,
    price_log_close,
    revenue_absolute_days,
    revenue_log_close,
    difficulty_absolute_days,
    difficulty_log_close,
    hashrate_absolute_days,
    hashrate_log_close,
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

        powerlaw_series = st.session_state.get(KEY_POWERLAW_SERIES, POWERLAW_SERIES_PRICE)
        if powerlaw_series not in POWERLAW_SERIES_OPTIONS:
            powerlaw_series = POWERLAW_SERIES_PRICE
            st.session_state[KEY_POWERLAW_SERIES] = powerlaw_series
        if mode == MODE_POWERLAW:
            powerlaw_series = st.radio(
                "PowerLaw series",
                POWERLAW_SERIES_OPTIONS,
                horizontal=True,
                key=KEY_POWERLAW_SERIES,
            )
            if powerlaw_series is None:
                powerlaw_series = st.session_state.get(KEY_POWERLAW_SERIES, POWERLAW_SERIES_PRICE)
                st.session_state[KEY_POWERLAW_SERIES] = powerlaw_series
                st.rerun()
        logperiodic_series = st.session_state.get(KEY_LOGPERIODIC_SERIES, POWERLAW_SERIES_PRICE)
        if logperiodic_series not in LOGPERIODIC_SERIES_OPTIONS:
            logperiodic_series = POWERLAW_SERIES_PRICE
            st.session_state[KEY_LOGPERIODIC_SERIES] = logperiodic_series
        if mode == MODE_LOGPERIODIC:
            logperiodic_series = st.radio(
                "LogPeriodic series",
                LOGPERIODIC_SERIES_OPTIONS,
                horizontal=True,
                key=KEY_LOGPERIODIC_SERIES,
            )
            if logperiodic_series is None:
                logperiodic_series = st.session_state.get(KEY_LOGPERIODIC_SERIES, POWERLAW_SERIES_PRICE)
                st.session_state[KEY_LOGPERIODIC_SERIES] = logperiodic_series
                st.rerun()

        is_non_price_series = (
            (mode == MODE_POWERLAW and powerlaw_series != POWERLAW_SERIES_PRICE)
            or (mode == MODE_LOGPERIODIC and logperiodic_series != POWERLAW_SERIES_PRICE)
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
            )

        if mode != MODE_PORTFOLIO:
            time_scale = inline_radio_control("Time", [TIME_LOG, TIME_LIN], key=KEY_TIME_SCALE)
        else:
            time_scale = st.session_state.get(KEY_TIME_SCALE, TIME_LOG)

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
            )
        else:
            st.session_state[KEY_BAND_METHOD] = BAND_METHOD_QUANTILE
            band_method = BAND_METHOD_QUANTILE

        current_r2 = 0.0
        price_scale = "Log"
        model_abs_days = price_absolute_days
        model_log_close = price_log_close
        a_key = KEY_A_PRICE
        b_key = KEY_B_PRICE
        default_a = DEFAULT_A
        default_b = DEFAULT_B

        if (
            mode in [MODE_POWERLAW, MODE_PORTFOLIO]
            and (mode != MODE_POWERLAW or powerlaw_series == POWERLAW_SERIES_PRICE)
            and currency == CURRENCY_GOLD
        ):
            a_key = KEY_A_GOLD
            b_key = KEY_B_GOLD
            default_a = DEFAULT_GOLD_A
            default_b = DEFAULT_GOLD_B
        if (
            mode in [MODE_POWERLAW, MODE_PORTFOLIO]
            and (mode != MODE_POWERLAW or powerlaw_series == POWERLAW_SERIES_PRICE)
            and currency == CURRENCY_EURO
        ):
            a_key = KEY_A_EURO
            b_key = KEY_B_EURO
            default_a = DEFAULT_EURO_A
            default_b = DEFAULT_EURO_B

        if mode == MODE_POWERLAW and powerlaw_series == POWERLAW_SERIES_REVENUE:
            model_abs_days = revenue_absolute_days
            model_log_close = revenue_log_close
            a_key = KEY_A_REVENUE
            b_key = KEY_B_REVENUE
            default_a = DEFAULT_REVENUE_A
            default_b = DEFAULT_REVENUE_B
        if mode == MODE_POWERLAW and powerlaw_series == POWERLAW_SERIES_DIFFICULTY:
            model_abs_days = difficulty_absolute_days
            model_log_close = difficulty_log_close
            a_key = KEY_A_DIFFICULTY
            b_key = KEY_B_DIFFICULTY
            default_a = DEFAULT_DIFFICULTY_A
            default_b = DEFAULT_DIFFICULTY_B
        if mode == MODE_POWERLAW and powerlaw_series == POWERLAW_SERIES_HASHRATE:
            model_abs_days = hashrate_absolute_days
            model_log_close = hashrate_log_close
            a_key = KEY_A_HASHRATE
            b_key = KEY_B_HASHRATE
            default_a = DEFAULT_HASHRATE_A
            default_b = DEFAULT_HASHRATE_B
        if mode == MODE_LOGPERIODIC and logperiodic_series == POWERLAW_SERIES_DIFFICULTY:
            model_abs_days = difficulty_absolute_days
            model_log_close = difficulty_log_close
            a_key = KEY_A_DIFFICULTY
            b_key = KEY_B_DIFFICULTY
            default_a = DEFAULT_DIFFICULTY_A
            default_b = DEFAULT_DIFFICULTY_B
        if mode == MODE_LOGPERIODIC and logperiodic_series == POWERLAW_SERIES_HASHRATE:
            model_abs_days = hashrate_absolute_days
            model_log_close = hashrate_log_close
            a_key = KEY_A_HASHRATE
            b_key = KEY_B_HASHRATE
            default_a = DEFAULT_HASHRATE_A
            default_b = DEFAULT_HASHRATE_B

        if mode in [MODE_POWERLAW, MODE_PORTFOLIO]:
            hide_price_scale = (
                mode == MODE_POWERLAW
                and powerlaw_series in [POWERLAW_SERIES_DIFFICULTY, POWERLAW_SERIES_HASHRATE]
            )
            price_scale, current_r2 = power_law.render_sidebar(
                model_abs_days,
                model_log_close,
                c_text_main,
                show_price_scale=(mode != MODE_PORTFOLIO and not hide_price_scale),
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
            )
        else:
            # Keep legacy A/B keys aligned before rendering LogPeriodic controls,
            # so sidebar R² is computed for the newly selected series immediately.
            st.session_state[KEY_A] = float(st.session_state.get(a_key, default_a))
            st.session_state[KEY_B] = float(st.session_state.get(b_key, default_b))
            active_osc_defaults = (
                OSC_DEFAULTS_HASHRATE
                if logperiodic_series == POWERLAW_SERIES_HASHRATE
                else (
                    OSC_DEFAULTS_DIFFICULTY
                    if logperiodic_series == POWERLAW_SERIES_DIFFICULTY
                    else OSC_DEFAULTS
                )
            )
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
                min_abs_day_for_fit=(
                    OSCILLATOR_DIFF_HASH_START_ABS_DAYS
                    if logperiodic_series in [POWERLAW_SERIES_DIFFICULTY, POWERLAW_SERIES_HASHRATE]
                    else None
                ),
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
