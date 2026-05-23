import streamlit as st

from core import oscillator, power_law
from core.constants import (
    CURRENCY_DOLLAR,
    CURRENCY_OPTIONS,
    DEFAULT_FORECAST_HORIZON,
    KEY_A,
    KEY_B,
    KEY_BITCOIN_NETWORK_SIMULATION_RESOLUTION,
    KEY_BITCOIN_NETWORK_SIMULATION_SEED,
    KEY_CHART_REVISION,
    KEY_CURRENCY_SELECTOR,
    KEY_LAST_MODE,
    KEY_LOGPERIODIC_HARMONICS,
    KEY_LOGPERIODIC_LAST_SERIES,
    KEY_LOGPERIODIC_SHOW_DECAYED_DSI,
    KEY_MODE_SELECTOR,
    KEY_LOGPERIODIC_SERIES,
    KEY_POWERLAW_ENVELOPE_SIGMA,
    KEY_POWERLAW_SERIES,
    KEY_PORTFOLIO_BTC_AMOUNT,
    KEY_PORTFOLIO_FORECAST_HORIZON,
    KEY_PORTFOLIO_FORECAST_MONTHS_LEGACY,
    KEY_PORTFOLIO_FORECAST_UNIT,
    KEY_PORTFOLIO_MONTHLY_BUY_AMOUNT,
    KEY_PORTFOLIO_MONTHLY_MOM_CHANGE_PCT,
    KEY_PORTFOLIO_SIGMA_LEVEL,
    KEY_PORTFOLIO_STRATEGY_VIEW,
    KEY_TIME_SCALE,
    MODE_LOGPERIODIC,
    MODE_PORTFOLIO,
    PORTFOLIO_RESET_A,
    PORTFOLIO_RESET_B,
    MODE_POWERLAW,
    POWERLAW_INTERCEPT_MAX,
    POWERLAW_INTERCEPT_MIN,
    POWERLAW_SERIES_BITCOIN_NETWORK_SIMULATION,
    POWERLAW_SERIES_PRICE,
    POWERLAW_SLOPE_MAX,
    POWERLAW_SLOPE_MIN,
    PORTFOLIO_SIGMA_CURRENT,
    PORTFOLIO_SIGMA_PEAK_POWERLAW,
    PORTFOLIO_SIGMA_TROUGH_POWERLAW,
    PORTFOLIO_VIEW_ACCUMULATION,
    PORTFOLIO_VIEW_PENSION,
    PORTFOLIO_VIEW_STRATEGY_TESTER,
    TIME_LOG,
    TIME_LIN,
)
from core.series_registry import (
    get_active_model_config,
    get_logperiodic_series_options,
    get_powerlaw_series_group_for_series,
    get_powerlaw_series_group_map,
    get_powerlaw_series_groups,
    get_powerlaw_series_options,
    series_supports_currency_selector,
)

KEY_PORTFOLIO_BTC_AMOUNT_INPUT = f"{KEY_PORTFOLIO_BTC_AMOUNT}_input"
KEY_LAST_PORTFOLIO_VIEW = "last_portfolio_strategy_view"


def _sync_portfolio_forecast_unit_default(selected_portfolio_view):
    last_portfolio_view = st.session_state.get(KEY_LAST_PORTFOLIO_VIEW)
    entering_accumulation = (
        selected_portfolio_view == PORTFOLIO_VIEW_ACCUMULATION
        and last_portfolio_view != PORTFOLIO_VIEW_ACCUMULATION
    )
    if entering_accumulation or KEY_PORTFOLIO_FORECAST_UNIT not in st.session_state:
        st.session_state[KEY_PORTFOLIO_FORECAST_UNIT] = "Month"
    st.session_state[KEY_LAST_PORTFOLIO_VIEW] = selected_portfolio_view


def _render_portfolio_sidebar_controls(forecast_horizon_min, forecast_horizon_max):
    portfolio_view_options = [
        PORTFOLIO_VIEW_ACCUMULATION,
        PORTFOLIO_VIEW_PENSION,
        PORTFOLIO_VIEW_STRATEGY_TESTER,
    ]
    selected_portfolio_view = st.session_state.get(
        KEY_PORTFOLIO_STRATEGY_VIEW, PORTFOLIO_VIEW_ACCUMULATION
    )
    if selected_portfolio_view not in portfolio_view_options:
        selected_portfolio_view = PORTFOLIO_VIEW_ACCUMULATION
        st.session_state[KEY_PORTFOLIO_STRATEGY_VIEW] = selected_portfolio_view

    st.markdown("**Portfolio strategy**")
    selected_portfolio_view = st.radio(
        "Portfolio strategy",
        portfolio_view_options,
        index=portfolio_view_options.index(selected_portfolio_view),
        horizontal=False,
        key=KEY_PORTFOLIO_STRATEGY_VIEW,
        label_visibility="collapsed",
    )
    _sync_portfolio_forecast_unit_default(selected_portfolio_view)

    if float(st.session_state.get(KEY_PORTFOLIO_BTC_AMOUNT, 2.0)) == 0.0:
        st.session_state[KEY_PORTFOLIO_BTC_AMOUNT] = 2.0
    if KEY_PORTFOLIO_BTC_AMOUNT_INPUT not in st.session_state:
        st.session_state[KEY_PORTFOLIO_BTC_AMOUNT_INPUT] = float(
            st.session_state.get(KEY_PORTFOLIO_BTC_AMOUNT, 2.0)
        )
    if float(st.session_state.get(KEY_PORTFOLIO_BTC_AMOUNT_INPUT, 2.0)) == 0.0:
        st.session_state[KEY_PORTFOLIO_BTC_AMOUNT_INPUT] = 2.0
    st.markdown("**BTC quantity**")
    btc_amount = st.number_input(
        "BTC quantity",
        min_value=0.0,
        step=0.0001,
        format="%.4f",
        key=KEY_PORTFOLIO_BTC_AMOUNT_INPUT,
        label_visibility="collapsed",
    )
    st.session_state[KEY_PORTFOLIO_BTC_AMOUNT] = float(btc_amount)

    if selected_portfolio_view == PORTFOLIO_VIEW_ACCUMULATION:
        st.markdown("**Monthly buy/sell amount**")
        st.number_input(
            "Monthly buy/sell amount",
            step=10,
            format="%d",
            key=KEY_PORTFOLIO_MONTHLY_BUY_AMOUNT,
            label_visibility="collapsed",
        )
        st.caption("Adds a second capital-growth line using fixed monthly cash flow.")

        st.markdown("**Sell % of MoM Change**")
        sell_mom_pct = float(st.session_state.get(KEY_PORTFOLIO_MONTHLY_MOM_CHANGE_PCT, 0.0))
        sell_mom_pct = min(max(sell_mom_pct, 0.0), 100.0)
        st.session_state[KEY_PORTFOLIO_MONTHLY_MOM_CHANGE_PCT] = sell_mom_pct
        st.number_input(
            "Sell % of MoM Change",
            min_value=0.0,
            max_value=100.0,
            step=1.0,
            format="%.1f",
            key=KEY_PORTFOLIO_MONTHLY_MOM_CHANGE_PCT,
            label_visibility="collapsed",
        )
        st.caption("Adds a second capital-growth line using monthly model growth sells.")

    if selected_portfolio_view in [PORTFOLIO_VIEW_ACCUMULATION, PORTFOLIO_VIEW_PENSION]:
        st.markdown(
            "**Price scenario**"
            if selected_portfolio_view == PORTFOLIO_VIEW_ACCUMULATION
            else "**Pension sigma**"
        )
        sigma_options = [
            PORTFOLIO_SIGMA_CURRENT,
            -2.0,
            -1.5,
            -1.0,
            -0.5,
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
        ]
        if selected_portfolio_view in [
            PORTFOLIO_VIEW_ACCUMULATION,
            PORTFOLIO_VIEW_PENSION,
        ]:
            sigma_options.extend(
                [
                    PORTFOLIO_SIGMA_PEAK_POWERLAW,
                    PORTFOLIO_SIGMA_TROUGH_POWERLAW,
                ]
            )

        def format_sigma_option(value):
            if value == PORTFOLIO_SIGMA_CURRENT:
                return "Current sigma"
            if value == PORTFOLIO_SIGMA_PEAK_POWERLAW:
                return "Peak PowerLaw"
            if value == PORTFOLIO_SIGMA_TROUGH_POWERLAW:
                return "Trough PowerLaw"
            if value == 0.0:
                return "0 sigma"
            return f"{value:+g} sigma"

        selected_sigma = st.session_state.get(KEY_PORTFOLIO_SIGMA_LEVEL, 0.0)
        if selected_sigma not in sigma_options:
            selected_sigma = -2.0 if selected_portfolio_view == PORTFOLIO_VIEW_PENSION else 0.0
            st.session_state[KEY_PORTFOLIO_SIGMA_LEVEL] = selected_sigma

        st.radio(
            (
                "Price scenario"
                if selected_portfolio_view == PORTFOLIO_VIEW_ACCUMULATION
                else "Pension sigma"
            ),
            sigma_options,
            index=sigma_options.index(selected_sigma),
            format_func=format_sigma_option,
            horizontal=True,
            key=KEY_PORTFOLIO_SIGMA_LEVEL,
            label_visibility="collapsed",
        )
    if selected_portfolio_view == PORTFOLIO_VIEW_ACCUMULATION:
        st.markdown("**Envelope sigma**")
        st.slider(
            "Envelope sigma",
            min_value=0.25,
            max_value=2.0,
            step=0.25,
            key=KEY_POWERLAW_ENVELOPE_SIGMA,
            label_visibility="collapsed",
        )
    if selected_portfolio_view == PORTFOLIO_VIEW_ACCUMULATION:
        st.markdown("**Forecast unit**")
        forecast_unit = st.radio(
            "Forecast unit",
            ["Year", "Month", "Day"],
            horizontal=True,
            key=KEY_PORTFOLIO_FORECAST_UNIT,
            label_visibility="collapsed",
        )
        default_horizon = int(
            st.session_state.get(
                KEY_PORTFOLIO_FORECAST_HORIZON,
                st.session_state.get(
                    KEY_PORTFOLIO_FORECAST_MONTHS_LEGACY, DEFAULT_FORECAST_HORIZON
                ),
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


def _render_powerlaw_series_selector(powerlaw_series):
    st.markdown("**PowerLaw series**")
    group_map = get_powerlaw_series_group_map()
    group_names = [group_name for group_name, _ in get_powerlaw_series_groups()]
    active_group = get_powerlaw_series_group_for_series(powerlaw_series)

    selected_group = st.radio(
        "Series family",
        group_names,
        index=group_names.index(active_group),
        horizontal=True,
        width="stretch",
        label_visibility="collapsed",
    )

    series_options = group_map[selected_group]
    if powerlaw_series not in series_options:
        powerlaw_series = series_options[0]
        st.session_state[KEY_POWERLAW_SERIES] = powerlaw_series

    selected_series = st.radio(
        "Series",
        series_options,
        index=series_options.index(powerlaw_series),
        horizontal=True,
        width="stretch",
        label_visibility="collapsed",
    )

    if selected_series != st.session_state.get(KEY_POWERLAW_SERIES):
        st.session_state[KEY_POWERLAW_SERIES] = selected_series
        st.rerun()

    return selected_series


def _render_bitcoin_network_simulation_controls():
    def create_new_simulation():
        st.session_state[KEY_BITCOIN_NETWORK_SIMULATION_SEED] = (
            int(st.session_state.get(KEY_BITCOIN_NETWORK_SIMULATION_SEED, 1)) + 1
        )
        st.session_state[KEY_CHART_REVISION] += 1

    st.markdown("**Bitcoin network simulation**")
    st.button(
        "Create new simulation",
        use_container_width=True,
        on_click=create_new_simulation,
    )
    st.markdown("**Resolution (days)**")
    st.slider(
        "Resolution (days)",
        min_value=0.00001,
        max_value=0.01,
        step=0.00001,
        format="%.5f",
        key=KEY_BITCOIN_NETWORK_SIMULATION_RESOLUTION,
        label_visibility="collapsed",
    )
    st.caption(f"Seed: {int(st.session_state.get(KEY_BITCOIN_NETWORK_SIMULATION_SEED, 1))}")


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
            powerlaw_series = _render_powerlaw_series_selector(powerlaw_series)
            if powerlaw_series is None:
                powerlaw_series = st.session_state.get(KEY_POWERLAW_SERIES, POWERLAW_SERIES_PRICE)
                st.session_state[KEY_POWERLAW_SERIES] = powerlaw_series
                st.rerun()
        logperiodic_series = st.session_state.get(KEY_LOGPERIODIC_SERIES, POWERLAW_SERIES_PRICE)
        if logperiodic_series not in logperiodic_series_options:
            logperiodic_series = POWERLAW_SERIES_PRICE
            st.session_state[KEY_LOGPERIODIC_SERIES] = logperiodic_series
        if mode == MODE_LOGPERIODIC:
            st.markdown("**LogPeriodic series**")
            logperiodic_series = st.radio(
                "LogPeriodic series",
                logperiodic_series_options,
                horizontal=True,
                key=KEY_LOGPERIODIC_SERIES,
                width="stretch",
                label_visibility="collapsed",
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
            st.markdown("**Envelope sigma**")
            st.slider(
                "Envelope sigma",
                min_value=0.25,
                max_value=2.0,
                step=0.25,
                key=KEY_POWERLAW_ENVELOPE_SIGMA,
                label_visibility="collapsed",
            )
        is_non_price_series = not series_supports_currency_selector(
            mode, powerlaw_series, logperiodic_series
        )

        if is_non_price_series:
            currency = CURRENCY_DOLLAR
            st.markdown("**Currency**")
            st.caption("This series does not use currency conversion.")
        else:
            currency = st.radio(
                "Currency",
                CURRENCY_OPTIONS,
                horizontal=True,
                key=KEY_CURRENCY_SELECTOR,
                width="stretch",
            )

        current_r2 = 0.0
        active_model = get_active_model_config(
            mode,
            powerlaw_series,
            logperiodic_series,
            selected_currency=currency,
        )
        a_min, a_max = active_model.powerlaw_intercept_bounds or (
            POWERLAW_INTERCEPT_MIN,
            POWERLAW_INTERCEPT_MAX,
        )
        b_min, b_max = active_model.powerlaw_slope_bounds or (
            POWERLAW_SLOPE_MIN,
            POWERLAW_SLOPE_MAX,
        )
        active_series_data = sidebar_series_data[active_model.series_name]
        model_abs_days = active_series_data["absolute_days"]
        model_log_close = active_series_data["log_close"]
        if active_model.analysis_min_abs_day is not None:
            analysis_mask = model_abs_days >= float(active_model.analysis_min_abs_day)
            model_abs_days = model_abs_days[analysis_mask]
            model_log_close = model_log_close[analysis_mask]
        a_key = active_model.a_key
        b_key = active_model.b_key
        default_a = active_model.default_a
        default_b = active_model.default_b
        model_origin_abs_day = active_model.model_origin_abs_day

        if mode in [MODE_POWERLAW, MODE_PORTFOLIO]:
            reset_a = default_a if mode == MODE_POWERLAW else PORTFOLIO_RESET_A
            reset_b = default_b if mode == MODE_POWERLAW else PORTFOLIO_RESET_B
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
                        else (
                            _render_bitcoin_network_simulation_controls()
                            if active_model.series_name
                            == POWERLAW_SERIES_BITCOIN_NETWORK_SIMULATION
                            else None
                        )
                    )
                ),
                a_key=a_key,
                b_key=b_key,
                default_a=default_a,
                default_b=default_b,
                reset_a=reset_a,
                reset_b=reset_b,
                a_min=a_min,
                a_max=a_max,
                b_min=b_min,
                b_max=b_max,
                genesis_offset_days=model_origin_abs_day,
            )
        else:
            # Keep legacy A/B keys aligned before rendering LogPeriodic controls,
            # so sidebar R² is computed for the newly selected series immediately.
            st.session_state[KEY_A] = float(st.session_state.get(a_key, default_a))
            st.session_state[KEY_B] = float(st.session_state.get(b_key, default_b))
            active_osc_defaults = active_model.oscillator_defaults
            logperiodic_defaults_signature = f"{logperiodic_series}:{currency}"
            last_lp_series = st.session_state.get(KEY_LOGPERIODIC_LAST_SERIES)
            if last_lp_series != logperiodic_defaults_signature:
                for k, v in active_osc_defaults.items():
                    if k == "harmonic_count":
                        st.session_state[KEY_LOGPERIODIC_HARMONICS] = int(v)
                        oscillator.apply_dsi_mode_option(oscillator.DEFAULT_DSI_MODE)
                        continue
                    st.session_state[k] = v
                st.session_state[KEY_LOGPERIODIC_LAST_SERIES] = logperiodic_defaults_signature

            oscillator.render_sidebar(
                model_abs_days,
                model_log_close,
                c_text_main,
                defaults_override=active_osc_defaults,
                min_abs_day_for_fit=active_model.oscillator_min_abs_day,
                parameter_bounds_override=active_model.oscillator_parameter_bounds,
            )
    return (
        mode,
        currency,
        time_scale,
        price_scale,
        current_r2,
        powerlaw_series,
        logperiodic_series,
    )
