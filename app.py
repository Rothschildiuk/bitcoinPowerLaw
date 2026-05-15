import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import oscillator, power_law
from core.constants import (
    APP_VERSION,
    CURRENCY_DOLLAR,
    CURRENCY_EURO,
    CURRENCY_GOLD,
    CURRENCY_UAH,
    DEFAULT_FORECAST_HORIZON,
    DEFAULT_THEME,
    FORECAST_HORIZON_MAX,
    FORECAST_HORIZON_MIN,
    GENESIS_DATE,
    KEY_A,
    KEY_A_PRICE,
    KEY_B,
    KEY_B_PRICE,
    KEY_BITCOIN_NETWORK_SIMULATION_RESOLUTION,
    KEY_BITCOIN_NETWORK_SIMULATION_SEED,
    KEY_CHART_REVISION,
    KEY_CURRENCY_SELECTOR,
    KEY_GENESIS_OFFSET,
    KEY_LAST_MODE,
    KEY_LOGPERIODIC_HARMONICS,
    KEY_LOGPERIODIC_SERIES,
    KEY_LOGPERIODIC_SHOW_DECAYED_DSI,
    KEY_POWERLAW_ENVELOPE_SIGMA,
    KEY_POWERLAW_SERIES,
    KEY_PORTFOLIO_BACKTEST_HAS_RUN,
    KEY_PORTFOLIO_BACKTEST_FLOOR_MODEL,
    KEY_PORTFOLIO_BACKTEST_STRATEGY_PCT,
    KEY_PORTFOLIO_BACKTEST_YEARS,
    KEY_PORTFOLIO_BTC_AMOUNT,
    KEY_PORTFOLIO_FORECAST_HORIZON,
    KEY_PORTFOLIO_FORECAST_UNIT,
    KEY_PORTFOLIO_MONTHLY_BUY_AMOUNT,
    KEY_PORTFOLIO_MONTHLY_MOM_CHANGE_PCT,
    KEY_PORTFOLIO_PENSION_DIVISOR,
    KEY_PORTFOLIO_PENSION_PAYOUT_PCT,
    KEY_PORTFOLIO_SIGMA_LEVEL,
    KEY_PORTFOLIO_STRATEGY_VIEW,
    KEY_THEME_MODE,
    MODE_LOGPERIODIC,
    MODE_PORTFOLIO,
    MODE_POWERLAW,
    OSC_DEFAULTS,
    POWERLAW_SERIES_DOGECOIN_BTC,
    POWERLAW_SERIES_DIFFICULTY,
    POWERLAW_SERIES_FILECOIN_BTC,
    POWERLAW_SERIES_HASHRATE,
    POWERLAW_SERIES_BITCOIN_NETWORK_SIMULATION,
    POWERLAW_SERIES_BITCOIN_VOLATILITY,
    POWERLAW_SERIES_LITECOIN_BTC,
    POWERLAW_SERIES_LIGHTNING_CAPACITY,
    POWERLAW_SERIES_LIGHTNING_NODES,
    POWERLAW_SERIES_LIQUID_BTC,
    POWERLAW_SERIES_LIQUID_TRANSACTIONS,
    POWERLAW_SERIES_MONERO_BTC,
    POWERLAW_SERIES_PRICE,
    POWERLAW_SERIES_REVENUE,
    POWERLAW_SERIES_RUSSIAN_M2,
    POWERLAW_SERIES_US_M2,
    PORTFOLIO_SIGMA_CURRENT,
    PORTFOLIO_SIGMA_PEAK_POWERLAW,
    PORTFOLIO_SIGMA_TROUGH_POWERLAW,
    PORTFOLIO_VIEW_ACCUMULATION,
    PORTFOLIO_VIEW_PENSION,
    PORTFOLIO_VIEW_STRATEGY_TESTER,
    TIME_LOG,
)
from core.series_registry import (
    get_active_model_config,
    get_selected_series_name,
    iter_session_model_defaults,
    series_supports_currency_selector,
)
from core.simulation import build_bitcoin_network_simulation
from core.utils import (
    PortfolioSettings,
    build_portfolio_real_data_backtest,
    build_portfolio_projection,
    build_portfolio_view_model,
    calculate_expanding_powerlaw_parameters,
    calculate_r2_score,
    estimate_current_monthly_pension,
    evaluate_powerlaw_values,
    interpolate_sigma_level_from_log_offset,
    powerlaw_parameters_are_unstable,
    resolve_trend_parameters,
    resolve_portfolio_scenario_log_offset,
)
from services.price_service import (
    build_currency_close_series,
    build_prepared_bitcoin_volatility_data,
    load_prepared_dogecoin_btc_data,
    load_prepared_difficulty_data,
    load_prepared_filecoin_btc_data,
    load_prepared_hashrate_data,
    load_prepared_litecoin_btc_data,
    load_prepared_lightning_capacity_data,
    load_prepared_lightning_nodes_data,
    load_prepared_liquid_btc_data,
    load_prepared_liquid_transactions_data,
    load_prepared_miner_revenue_data,
    load_prepared_monero_btc_data,
    load_prepared_price_data,
    load_prepared_russian_m2_data,
    load_prepared_us_m2_data,
)
from ui.charts import (
    _resolve_model_view_max,
    render_main_model_chart,
)
from ui.kpi import render_model_kpis
from ui.sidebar import render_sidebar_panel
from ui.theme import apply_theme_css, get_theme


def initialize_app_session_state():
    defaults = {
        KEY_THEME_MODE: DEFAULT_THEME,
        KEY_LAST_MODE: MODE_POWERLAW,
        KEY_CURRENCY_SELECTOR: CURRENCY_EURO,
        KEY_CHART_REVISION: 0,
        KEY_POWERLAW_SERIES: POWERLAW_SERIES_PRICE,
        KEY_LOGPERIODIC_SERIES: POWERLAW_SERIES_PRICE,
        KEY_LOGPERIODIC_HARMONICS: int(OSC_DEFAULTS.get("harmonic_count", 1)),
        KEY_LOGPERIODIC_SHOW_DECAYED_DSI: False,
        KEY_POWERLAW_ENVELOPE_SIGMA: 1.0,
        KEY_BITCOIN_NETWORK_SIMULATION_SEED: 1,
        KEY_BITCOIN_NETWORK_SIMULATION_RESOLUTION: 0.00001,
        KEY_PORTFOLIO_SIGMA_LEVEL: 0,
        KEY_PORTFOLIO_MONTHLY_MOM_CHANGE_PCT: 0.0,
        KEY_PORTFOLIO_PENSION_PAYOUT_PCT: 100.0,
        KEY_PORTFOLIO_STRATEGY_VIEW: PORTFOLIO_VIEW_ACCUMULATION,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    # Light theme is disabled by product decision; always force dark theme.
    st.session_state[KEY_THEME_MODE] = DEFAULT_THEME

    for key, value in iter_session_model_defaults():
        if key not in st.session_state:
            st.session_state[key] = value
    if KEY_A not in st.session_state:
        st.session_state[KEY_A] = st.session_state[KEY_A_PRICE]
    if KEY_B not in st.session_state:
        st.session_state[KEY_B] = st.session_state[KEY_B_PRICE]


def calculate_percentile_offsets(display_df, genesis_offset_days):
    """
    Compute stable percentile offsets from a baseline best-fit trend for the current offset.
    This keeps percentile bands moving together with manual A/B adjustments.
    """
    fitted_b, fitted_a, _ = power_law.fit_powerlaw_regression(
        display_df["AbsDays"].values,
        display_df["LogClose"].values,
        genesis_offset_days,
    )
    if fitted_a == 0.0 and fitted_b == 0.0:
        baseline_residuals = display_df["Res"].values
    else:
        base_days = np.maximum(display_df["AbsDays"].values - genesis_offset_days, 1.0)
        baseline_log = fitted_a + fitted_b * np.log10(base_days)
        baseline_residuals = display_df["LogClose"].values - baseline_log

    return np.percentile(baseline_residuals, [2.5, 16.5, 83.5, 97.5])


def calculate_residual_sigma_log(display_df):
    residuals = pd.to_numeric(display_df["Res"], errors="coerce").to_numpy(dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    if residuals.size == 0:
        return 0.0
    sigma = float(np.std(residuals))
    return sigma if np.isfinite(sigma) else 0.0


def calculate_peak_powerlaw_overlay(
    display_df,
    genesis_offset_days,
    model_days,
    percentile_offsets,
    sigma_threshold,
):
    close_values = pd.to_numeric(display_df["CloseDisplay"], errors="coerce").to_numpy(dtype=float)
    log_prices = np.full(close_values.shape, np.nan, dtype=float)
    positive_mask = close_values > 0.0
    log_prices[positive_mask] = np.log10(close_values[positive_mask])
    sigma_threshold = abs(float(sigma_threshold))
    sigma_levels = [-2.0, -1.0, 0.0, 1.0, 2.0]
    sigma_offsets = [*percentile_offsets[:2], 0.0, *percentile_offsets[2:]]
    lower_offset = float(np.interp(-sigma_threshold, sigma_levels, sigma_offsets))
    upper_offset = float(np.interp(sigma_threshold, sigma_levels, sigma_offsets))
    residuals = pd.to_numeric(display_df["Res"], errors="coerce").to_numpy(dtype=float)
    peak_overlay = power_law.fit_peak_powerlaw_envelope(
        display_df["AbsDays"].to_numpy(dtype=float),
        log_prices,
        genesis_offset_days,
        model_days,
        residuals=residuals,
        threshold_offset=upper_offset,
    )
    trough_overlay = power_law.fit_trough_powerlaw_envelope(
        display_df["AbsDays"].to_numpy(dtype=float),
        log_prices,
        genesis_offset_days,
        model_days,
        residuals=residuals,
        threshold_offset=lower_offset,
    )
    return {
        "peak": peak_overlay,
        "trough": trough_overlay,
    }


def resolve_current_sigma_level(df_display, percentile_offsets):
    current_price = float(df_display["CloseDisplay"].iloc[-1])
    current_model_log = float(df_display["ModelLog"].iloc[-1])
    if current_price <= 0.0 or not np.isfinite(current_price) or not np.isfinite(current_model_log):
        return 0.0
    current_log_offset = float(np.log10(current_price) - current_model_log)
    return interpolate_sigma_level_from_log_offset(current_log_offset, percentile_offsets)


def style_portfolio_table(table_df, style_format, currency_unit, portfolio_view):
    column_colors = {
        f"Fair Price ({currency_unit})": "#f0b90b",
        f"Portfolio if not selling ({currency_unit})": "#f0b90b",
    }
    if portfolio_view.dca_enabled:
        column_colors.update(
            {
                f"Remaining BTC value ({currency_unit})": "#14b8a6",
                portfolio_view.period_cash_flow_label: "#38bdf8",
                f"Net cash flow ({currency_unit})": "#8b5cf6",
            }
        )

    def color_column_values(data):
        styled = pd.DataFrame("", index=data.index, columns=data.columns)
        for column_name, color in column_colors.items():
            if column_name in styled.columns:
                styled[column_name] = f"color: {color}; font-weight: 700;"
        return styled

    table_styles = [
        {
            "selector": f"th.col{table_df.columns.get_loc(column_name)}",
            "props": [
                ("color", color),
                ("border-bottom", f"2px solid {color}"),
                ("box-shadow", f"inset 0 -2px 0 {color}"),
            ],
        }
        for column_name, color in column_colors.items()
        if column_name in table_df.columns
    ]
    return (
        table_df.style.format(style_format)
        .apply(color_column_values, axis=None)
        .set_table_styles(
            table_styles,
            overwrite=False,
        )
    )


def style_backtest_table(table_df, style_format, currency_unit, monthly_withdrawal_label):
    column_colors = {
        f"Actual BTC price ({currency_unit})": "#f0b90b",
        f"Hold-only value ({currency_unit})": "#f0b90b",
        f"Strategy value ({currency_unit})": "#14b8a6",
        monthly_withdrawal_label: "#38bdf8",
        f"Net cash flow ({currency_unit})": "#8b5cf6",
    }

    def color_column_values(data):
        styled = pd.DataFrame("", index=data.index, columns=data.columns)
        for column_name, color in column_colors.items():
            if column_name in styled.columns:
                styled[column_name] = f"color: {color}; font-weight: 700;"
        return styled

    table_styles = [
        {
            "selector": f"th.col{table_df.columns.get_loc(column_name)}",
            "props": [
                ("color", color),
                ("border-bottom", f"2px solid {color}"),
                ("box-shadow", f"inset 0 -2px 0 {color}"),
            ],
        }
        for column_name, color in column_colors.items()
        if column_name in table_df.columns
    ]
    return (
        table_df.style.format(style_format)
        .apply(color_column_values, axis=None)
        .set_table_styles(table_styles, overwrite=False)
    )


@st.cache_data(ttl=3600)
def prepare_model_grid(current_gen_date, a_active, b_active, view_max):
    m_x = np.arange(1.0, float(np.ceil(view_max)) + 1.0)
    m_dates = [current_gen_date + pd.Timedelta(days=float(d)) for d in m_x]
    m_log_d = np.log10(m_x)
    m_fair_usd, _, _ = evaluate_powerlaw_values(m_log_d, a_active, b_active)
    m_dates_str = [d.strftime("%d.%m.%Y") for d in m_dates]
    return m_x, m_dates, m_log_d, m_fair_usd, m_dates_str


@st.cache_data(ttl=3600)
def prepare_portfolio_projection(
    _df_index,
    current_gen_date,
    a_active,
    b_active,
    settings,
):
    return build_portfolio_projection(
        _df_index,
        current_gen_date,
        a_active,
        b_active,
        settings,
    )


@st.cache_data(ttl=3600)
def prepare_bitcoin_network_simulation(base_df, seed, resolution_days):
    return build_bitcoin_network_simulation(
        base_df,
        seed=int(seed),
        resolution_days=float(resolution_days),
    )


def render_portfolio_view(
    df_display,
    current_gen_date,
    a_active,
    b_active,
    percentile_offsets,
    current_r2,
    model_was_clipped,
    currency_prefix,
    currency_suffix,
    currency_decimals,
    currency_unit,
    pl_template,
    pl_text_color,
    pl_bg_color,
    pl_grid_color,
    c_hover_bg,
    c_border,
    c_hover_text,
):
    display_currency_decimals = int(currency_decimals)

    def format_portfolio_money(value):
        return f"{currency_prefix}{value:,.{display_currency_decimals}f}{currency_suffix}"

    portfolio_strategy_view = st.session_state.get(
        KEY_PORTFOLIO_STRATEGY_VIEW, PORTFOLIO_VIEW_ACCUMULATION
    )
    if portfolio_strategy_view not in [
        PORTFOLIO_VIEW_ACCUMULATION,
        PORTFOLIO_VIEW_PENSION,
        PORTFOLIO_VIEW_STRATEGY_TESTER,
    ]:
        portfolio_strategy_view = PORTFOLIO_VIEW_ACCUMULATION
        st.session_state[KEY_PORTFOLIO_STRATEGY_VIEW] = portfolio_strategy_view

    title_by_view = {
        PORTFOLIO_VIEW_ACCUMULATION: "Portfolio Accumulation",
        PORTFOLIO_VIEW_PENSION: "Portfolio Pension",
        PORTFOLIO_VIEW_STRATEGY_TESTER: "Portfolio Strategy Tester",
    }
    st.markdown(f"### {title_by_view[portfolio_strategy_view]}")
    selected_sigma_level = (
        st.session_state.get(KEY_PORTFOLIO_SIGMA_LEVEL, 0.0)
        if portfolio_strategy_view in [PORTFOLIO_VIEW_ACCUMULATION, PORTFOLIO_VIEW_PENSION]
        else 0.0
    )
    use_current_sigma_scenario = selected_sigma_level == PORTFOLIO_SIGMA_CURRENT
    use_peak_powerlaw_scenario = selected_sigma_level == PORTFOLIO_SIGMA_PEAK_POWERLAW
    use_trough_powerlaw_scenario = selected_sigma_level == PORTFOLIO_SIGMA_TROUGH_POWERLAW
    projection_intercept_a = a_active
    projection_slope_b = b_active
    selected_envelope = None
    if use_peak_powerlaw_scenario or use_trough_powerlaw_scenario:
        envelope_overlay = calculate_peak_powerlaw_overlay(
            df_display,
            genesis_offset,
            df_display["Days"].to_numpy(dtype=float),
            percentile_offsets,
            st.session_state.get(KEY_POWERLAW_ENVELOPE_SIGMA, 1.0),
        )
        selected_envelope = envelope_overlay.get("peak" if use_peak_powerlaw_scenario else "trough")
        if selected_envelope is not None:
            projection_intercept_a = float(selected_envelope["intercept"])
            projection_slope_b = float(selected_envelope["slope"])
        else:
            st.warning(
                "Selected envelope scenario has too few fit points for the current sigma threshold. Falling back to base PowerLaw."
            )
    scenario_sigma_level = (
        resolve_current_sigma_level(df_display, percentile_offsets)
        if use_current_sigma_scenario
        else (
            0.0
            if use_peak_powerlaw_scenario or use_trough_powerlaw_scenario
            else float(selected_sigma_level)
        )
    )

    settings = PortfolioSettings(
        btc_amount=float(st.session_state.get(KEY_PORTFOLIO_BTC_AMOUNT, 2.0)),
        monthly_buy_amount=(
            float(st.session_state.get(KEY_PORTFOLIO_MONTHLY_BUY_AMOUNT, 0.0))
            if portfolio_strategy_view == PORTFOLIO_VIEW_ACCUMULATION
            else 0.0
        ),
        monthly_mom_change_pct=float(
            st.session_state.get(KEY_PORTFOLIO_MONTHLY_MOM_CHANGE_PCT, 0.0)
            if portfolio_strategy_view == PORTFOLIO_VIEW_ACCUMULATION
            else 0.0
        ),
        forecast_unit=st.session_state.get(KEY_PORTFOLIO_FORECAST_UNIT, "Year"),
        forecast_horizon=int(
            st.session_state.get(KEY_PORTFOLIO_FORECAST_HORIZON, DEFAULT_FORECAST_HORIZON)
        ),
        sigma_level=scenario_sigma_level,
        residual_sigma_log=calculate_residual_sigma_log(df_display),
        residual_percentile_offsets_log=(
            None
            if use_peak_powerlaw_scenario or use_trough_powerlaw_scenario
            else tuple(float(value) for value in percentile_offsets)
        ),
    )
    projection_result = prepare_portfolio_projection(
        df_display.index,
        current_gen_date,
        projection_intercept_a,
        projection_slope_b,
        settings,
    )
    portfolio_view = build_portfolio_view_model(
        projection_result,
        monthly_buy_amount=settings.monthly_buy_amount,
        currency_unit=currency_unit,
        monthly_mom_change_pct=settings.monthly_mom_change_pct,
    )

    unstable_portfolio = powerlaw_parameters_are_unstable(
        current_r2,
        was_clipped=model_was_clipped,
    )
    if unstable_portfolio:
        st.info(
            "Portfolio projection needs a stable model fit. Click Auto-fit model to calculate fair-value metrics."
        )
        return

    money_fmt = f"{currency_prefix}{{:,.{display_currency_decimals}f}}{currency_suffix}"
    scenario_multiplier = np.power(10.0, resolve_portfolio_scenario_log_offset(settings))
    current_projection_day = max(
        float((df_display.index[-1] - current_gen_date).days),
        1.0,
    )
    current_scenario_base, _, _ = evaluate_powerlaw_values(
        np.array([np.log10(current_projection_day)]),
        projection_intercept_a,
        projection_slope_b,
    )
    current_scenario_price = float(current_scenario_base[0]) * float(scenario_multiplier)
    if use_peak_powerlaw_scenario:
        current_price_label = "Current Peak PowerLaw Price"
    elif use_trough_powerlaw_scenario:
        current_price_label = "Current Trough PowerLaw Price"
    elif use_current_sigma_scenario:
        current_price_label = f"Current Sigma Price ({settings.sigma_level:+.2f}σ)"
    elif settings.sigma_level == 0:
        current_price_label = "Current Fair Price"
    else:
        current_price_label = f"Current {settings.sigma_level:+g} sigma Price"

    if portfolio_strategy_view == PORTFOLIO_VIEW_ACCUMULATION:
        g1, g2, g3 = st.columns(3)
        g1.metric(current_price_label, format_portfolio_money(current_scenario_price))
        if portfolio_view.dca_enabled:
            g2.metric(
                "Hold-only portfolio",
                format_portfolio_money(portfolio_view.last_value),
                delta=f"{portfolio_view.total_growth_pct:+.1f}%",
            )
            g3.metric(
                "Remaining BTC value",
                format_portfolio_money(portfolio_view.last_dca_value),
                delta=format_portfolio_money(
                    portfolio_view.last_dca_value - portfolio_view.last_value
                ),
            )
            st.caption(
                f"Net cash flow by horizon: {format_portfolio_money(portfolio_view.last_dca_invested_capital)}"
            )
            st.caption(
                "Monthly buy/sell starts from the next calendar month. MoM sell compares the current month start with the previous month start."
            )
        else:
            g2.metric(
                "Portfolio (end of horizon)",
                format_portfolio_money(portfolio_view.last_value),
            )
            g3.metric("Total Growth", f"{portfolio_view.total_growth_pct:+.1f}%")

        portfolio_fig = go.Figure()
        portfolio_fig.add_trace(
            go.Scatter(
                x=portfolio_view.portfolio_display_df["Date"],
                y=portfolio_view.portfolio_display_df["PortfolioDisplay"],
                mode="lines+markers",
                name="Portfolio if not selling",
                line=dict(color="#f0b90b", width=2),
                hovertemplate=(
                    "<b>%{x|%d.%m.%Y}</b><br>Portfolio if not selling: "
                    f"{currency_prefix}%{{y:,.{display_currency_decimals}f}}{currency_suffix}<extra></extra>"
                ),
            )
        )
        if use_current_sigma_scenario:
            st.caption(
                f"Portfolio scenario uses today's exact market position: {settings.sigma_level:+.2f} sigma."
            )
        elif use_peak_powerlaw_scenario:
            st.caption("Portfolio scenario uses the fitted Peak PowerLaw envelope.")
        elif use_trough_powerlaw_scenario:
            st.caption("Portfolio scenario uses the fitted Trough PowerLaw envelope.")
        elif settings.sigma_level != 0:
            st.caption(
                f"Portfolio scenario uses {settings.sigma_level:+g} sigma historical log-residual offset."
            )
        if portfolio_view.dca_enabled:
            cumulative_withdrawals_display = portfolio_view.portfolio_display_df[
                "DcaPeriodCashFlowDisplay"
            ].cumsum()
            portfolio_fig.add_trace(
                go.Scatter(
                    x=portfolio_view.portfolio_display_df["Date"],
                    y=(
                        portfolio_view.portfolio_display_df["DcaPortfolioDisplay"]
                        + cumulative_withdrawals_display
                    ),
                    mode="lines+markers",
                    name="Portfolio + cumulative withdrawals",
                    line=dict(color="#8b5cf6", width=2, dash="dash"),
                    hovertemplate=(
                        "<b>%{x|%d.%m.%Y}</b><br>Portfolio + withdrawals: "
                        f"{currency_prefix}%{{y:,.{display_currency_decimals}f}}{currency_suffix}<extra></extra>"
                    ),
                )
            )
            portfolio_fig.add_trace(
                go.Scatter(
                    x=portfolio_view.portfolio_display_df["Date"],
                    y=portfolio_view.portfolio_display_df["DcaPortfolioDisplay"],
                    mode="lines+markers",
                    name="Remaining BTC value",
                    line=dict(color="#14b8a6", width=2),
                    hovertemplate=(
                        "<b>%{x|%d.%m.%Y}</b><br>Remaining BTC value: "
                        f"{currency_prefix}%{{y:,.{display_currency_decimals}f}}{currency_suffix}<extra></extra>"
                    ),
                )
            )
            portfolio_fig.add_trace(
                go.Scatter(
                    x=portfolio_view.portfolio_display_df["Date"],
                    y=cumulative_withdrawals_display,
                    mode="lines+markers",
                    name="Cumulative withdrawals",
                    line=dict(color="#f97316", width=2, dash="dot"),
                    hovertemplate=(
                        "<b>%{x|%d.%m.%Y}</b><br>Cumulative withdrawals: "
                        f"{currency_prefix}%{{y:,.{display_currency_decimals}f}}{currency_suffix}<extra></extra>"
                    ),
                )
            )
            portfolio_fig.add_trace(
                go.Scatter(
                    x=portfolio_view.portfolio_display_df["Date"],
                    y=portfolio_view.portfolio_display_df["DcaPeriodCashFlowDisplay"],
                    mode="lines+markers",
                    name=portfolio_view.period_cash_flow_label,
                    line=dict(color="#38bdf8", width=2),
                    hovertemplate=(
                        f"<b>%{{x|%d.%m.%Y}}</b><br>{portfolio_view.period_cash_flow_label}: "
                        f"{currency_prefix}%{{y:,.{display_currency_decimals}f}}{currency_suffix}<extra></extra>"
                    ),
                )
            )
        portfolio_fig.update_layout(
            height=320,
            margin=dict(t=10, b=0, l=50, r=20),
            template=pl_template,
            font=dict(color=pl_text_color),
            paper_bgcolor=pl_bg_color,
            plot_bgcolor=pl_bg_color,
            xaxis=dict(
                gridcolor=pl_grid_color,
                tickfont=dict(color=pl_text_color),
                range=[
                    portfolio_view.portfolio_display_df["Date"].min() - pd.Timedelta(days=90),
                    portfolio_view.portfolio_display_df["Date"].max(),
                ],
            ),
            yaxis=dict(gridcolor=pl_grid_color, tickfont=dict(color=pl_text_color)),
            hoverlabel=dict(
                bgcolor=c_hover_bg,
                bordercolor=c_border,
                font=dict(color=c_hover_text, size=13),
            ),
        )
        st.plotly_chart(
            portfolio_fig,
            width="stretch",
            theme=None,
            config={"displayModeBar": False},
            key=f"portfolio_{st.session_state[KEY_THEME_MODE]}_{st.session_state[KEY_CHART_REVISION]}",
        )

        st.markdown(f"#### {portfolio_view.table_title}")
        style_format = {
            f"Fair Price ({currency_unit})": money_fmt,
            f"Portfolio if not selling ({currency_unit})": money_fmt,
            portfolio_view.period_change_usd_label: money_fmt,
            portfolio_view.period_change_pct_label: "{:+.2f}%",
        }
        if portfolio_view.dca_enabled:
            style_format[f"Remaining BTC value ({currency_unit})"] = money_fmt
            style_format[portfolio_view.period_cash_flow_label] = money_fmt
            style_format[f"Net cash flow ({currency_unit})"] = money_fmt
            style_format["BTC after monthly cash flow"] = "{:,.6f}"
        st.dataframe(
            style_portfolio_table(
                portfolio_view.table_df,
                style_format,
                currency_unit,
                portfolio_view,
            ),
            width="stretch",
            hide_index=True,
        )

    elif portfolio_strategy_view == PORTFOLIO_VIEW_PENSION:
        pension_floor_sigma = float(settings.sigma_level)
        if use_peak_powerlaw_scenario:
            pension_floor_label = "Peak PowerLaw"
        elif use_trough_powerlaw_scenario:
            pension_floor_label = "Trough PowerLaw"
        elif use_current_sigma_scenario:
            pension_floor_label = f"current {pension_floor_sigma:+.2f}σ"
        else:
            pension_floor_label = f"{pension_floor_sigma:g}σ"
        current_price_display = float(df_display["CloseDisplay"].iloc[-1])
        current_model_log = projection_intercept_a + projection_slope_b * np.log10(
            max(float(df_display["Days"].iloc[-1]), 1.0)
        )
        pension_estimate = estimate_current_monthly_pension(
            current_price=current_price_display,
            current_model_log=current_model_log,
            current_date=df_display.index[-1],
            current_gen_date=current_gen_date,
            intercept_a=projection_intercept_a,
            slope_b=projection_slope_b,
            btc_amount=settings.btc_amount,
            sell_mom_change_pct=settings.monthly_mom_change_pct,
            percentile_offsets=percentile_offsets,
            floor_sigma_level=pension_floor_sigma,
        )
        st.markdown("#### Monthly BTC pension estimate")
        if KEY_PORTFOLIO_PENSION_PAYOUT_PCT not in st.session_state:
            st.session_state[KEY_PORTFOLIO_PENSION_PAYOUT_PCT] = 100
        divisor_col, _ = st.columns([1, 3])
        with divisor_col:
            payout_options = [25, 50, 75, 100]
            current_payout_pct = int(
                min(
                    payout_options,
                    key=lambda option: abs(
                        option
                        - float(
                            st.session_state.get(
                                KEY_PORTFOLIO_PENSION_PAYOUT_PCT,
                                100.0,
                            )
                        )
                    ),
                )
            )
            with st.container(key="portfolio_pension_payout_radio"):
                st.radio(
                    "Conservative payout (%)",
                    options=payout_options,
                    index=payout_options.index(current_payout_pct),
                    horizontal=True,
                    key=KEY_PORTFOLIO_PENSION_PAYOUT_PCT,
                )
        conservative_payout_pct = min(
            max(float(st.session_state.get(KEY_PORTFOLIO_PENSION_PAYOUT_PCT, 100.0)), 1.0),
            100.0,
        )
        conservative_payout_ratio = conservative_payout_pct / 100.0
        conservative_monthly_withdrawal = (
            pension_estimate.minimum_monthly_withdrawal * conservative_payout_ratio
        )
        conservative_btc_to_sell = pension_estimate.minimum_btc_to_sell * conservative_payout_ratio
        conservative_btc_to_sell_today = (
            pension_estimate.minimum_btc_to_sell_today * conservative_payout_ratio
        )
        today_btc_sell_delta_pct = pension_estimate.minimum_btc_sell_today_delta_pct
        if abs(today_btc_sell_delta_pct) <= 1.0:
            today_btc_sell_delta_label = f"near {pension_floor_label}"
            today_btc_sell_delta_class = "pension-metric-note-neutral"
        elif today_btc_sell_delta_pct > 0.0:
            today_btc_sell_delta_label = f"more expensive than {pension_floor_label}"
            today_btc_sell_delta_class = "pension-metric-note-positive"
        else:
            today_btc_sell_delta_label = f"cheaper than {pension_floor_label}"
            today_btc_sell_delta_class = "pension-metric-note-negative"
        st.markdown(
            (
                "<div class='pension-metric-grid'>"
                "<div class='pension-metric-card'>"
                "<div class='pension-metric-label'>Market position today</div>"
                f"<div class='pension-metric-value'>{pension_estimate.current_sigma_level:+.3f}σ</div>"
                "<div class='pension-rating' "
                f"style='border-color:{pension_estimate.withdrawal_rating_color};"
                f"color:{pension_estimate.withdrawal_rating_color};'>"
                f"{pension_estimate.withdrawal_rating}</div>"
                "</div>"
                "<div class='pension-metric-card'>"
                "<div class='pension-metric-label'>Conservative monthly pension</div>"
                f"<div class='pension-metric-value'>{format_portfolio_money(conservative_monthly_withdrawal)}</div>"
                "<div class='pension-metric-delta'>"
                f"↑ sell {conservative_btc_to_sell:.4f} BTC at {pension_floor_label} ({conservative_payout_pct:g}%)"
                "</div>"
                f"<div class='pension-metric-note {today_btc_sell_delta_class}'>"
                f"Today: sell {conservative_btc_to_sell_today:.4f} BTC "
                f"({abs(today_btc_sell_delta_pct):.1f}% {today_btc_sell_delta_label})"
                "</div>"
                "</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        sigma_table_levels = [
            ("Current σ", None),
            ("-2σ", -2.0),
            ("-1σ", -1.0),
            ("0σ", 0.0),
            ("+1σ", 1.0),
            ("+2σ", 2.0),
        ]
        sigma_scenario_levels = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=float)
        sigma_scenario_offsets = np.array(
            [
                percentile_offsets[0],
                percentile_offsets[1],
                0.0,
                percentile_offsets[2],
                percentile_offsets[3],
            ],
            dtype=float,
        )
        pension_floor_offset = float(
            np.interp(pension_floor_sigma, sigma_scenario_levels, sigma_scenario_offsets)
        )
        sigma_table_rows = []
        for sigma_label, sigma_level in sigma_table_levels:
            if sigma_level is None:
                sigma_price = current_price_display
                sigma_next_price = pension_estimate.next_month_price
                sigma_diff_pct = 0.0
                sigma_label = f"Current σ ({pension_estimate.current_sigma_level:+.2f}σ)"
            else:
                sigma_offset = float(
                    np.interp(sigma_level, sigma_scenario_levels, sigma_scenario_offsets)
                )
                sigma_price = float(np.power(10.0, current_model_log + sigma_offset))
                sigma_next_price = float(
                    pension_estimate.next_month_floor_price
                    * np.power(10.0, sigma_offset - pension_floor_offset)
                )
                sigma_diff_pct = (
                    ((current_price_display / sigma_price) - 1.0) * 100.0
                    if sigma_price > 0.0
                    else 0.0
                )
            sigma_monthly_pension = (
                max(0.0, sigma_next_price - sigma_price)
                * max(float(settings.btc_amount), 0.0)
                * conservative_payout_ratio
            )
            if abs(sigma_diff_pct) <= 1.0:
                sigma_diff_class = "pension-sigma-diff-neutral"
            elif sigma_diff_pct > 0.0:
                sigma_diff_class = "pension-sigma-diff-positive"
            else:
                sigma_diff_class = "pension-sigma-diff-negative"
            sigma_table_rows.append(
                "<tr>"
                f"<td>{sigma_label}</td>"
                f"<td>{format_portfolio_money(sigma_price)}</td>"
                f"<td><span class='pension-sigma-diff {sigma_diff_class}'>{format_portfolio_money(sigma_monthly_pension)}</span></td>"
                f"<td><span class='pension-sigma-diff {sigma_diff_class}'>{sigma_diff_pct:+.1f}%</span></td>"
                "</tr>"
            )
        st.markdown(
            (
                "<div class='pension-sigma-table-wrap'>"
                "<table class='pension-sigma-table'>"
                "<thead><tr><th>Level</th><th>Today price</th><th>Monthly pension</th><th>Current price diff</th></tr></thead>"
                f"<tbody>{''.join(sigma_table_rows)}</tbody>"
                "</table>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        st.caption(
            f"Conservative pension uses {conservative_payout_pct:g}% of one month of model growth on {pension_floor_label}."
        )
        st.caption(pension_estimate.withdrawal_rating_note)

    if portfolio_strategy_view != PORTFOLIO_VIEW_STRATEGY_TESTER:
        return

    st.markdown("#### Strategy tester")
    if KEY_PORTFOLIO_BACKTEST_STRATEGY_PCT not in st.session_state:
        st.session_state[KEY_PORTFOLIO_BACKTEST_STRATEGY_PCT] = 100.0
    if KEY_PORTFOLIO_BACKTEST_FLOOR_MODEL not in st.session_state:
        st.session_state[KEY_PORTFOLIO_BACKTEST_FLOOR_MODEL] = "-2σ"
    if KEY_PORTFOLIO_BACKTEST_YEARS not in st.session_state:
        st.session_state[KEY_PORTFOLIO_BACKTEST_YEARS] = 6
    if KEY_PORTFOLIO_BACKTEST_HAS_RUN not in st.session_state:
        st.session_state[KEY_PORTFOLIO_BACKTEST_HAS_RUN] = False

    strategy_labels = {
        10.0: "Sell 10% growth",
        25.0: "Sell 25% growth",
        50.0: "Sell 50% growth",
        75.0: "Sell 75% growth",
        100.0: "Sell 100% growth",
        150.0: "Sell 150% growth",
    }
    strategy_colors = {
        10.0: "#818cf8",
        25.0: "#60a5fa",
        50.0: "#38bdf8",
        75.0: "#22d3ee",
        100.0: "#14b8a6",
        150.0: "#f97316",
    }
    floor_model_options = {
        "-2σ": "-2σ",
        "trough_envelope_sigma_1": "Trough PowerLaw Envelope σ1",
    }

    with st.form("portfolio_strategy_tester"):
        s1, s2, s3, s4 = st.columns([1.35, 1.45, 0.9, 0.65])
        with s1:
            st.selectbox(
                "Strategy",
                list(strategy_labels.keys()),
                format_func=lambda value: strategy_labels[float(value)],
                key=KEY_PORTFOLIO_BACKTEST_STRATEGY_PCT,
            )
        with s2:
            st.selectbox(
                "Withdrawal floor",
                list(floor_model_options.keys()),
                format_func=lambda value: floor_model_options[value],
                key=KEY_PORTFOLIO_BACKTEST_FLOOR_MODEL,
            )
        with s3:
            st.slider(
                "Years to test",
                min_value=1,
                max_value=10,
                step=1,
                key=KEY_PORTFOLIO_BACKTEST_YEARS,
            )
        with s4:
            st.markdown("**Currency**")
            st.markdown(f"`{currency_unit}`")
        submitted = st.form_submit_button("Test strategy", type="primary", width="stretch")
        if submitted:
            st.session_state[KEY_PORTFOLIO_BACKTEST_HAS_RUN] = True

    if st.session_state.get(KEY_PORTFOLIO_BACKTEST_HAS_RUN, False):
        selected_sell_pct = float(st.session_state[KEY_PORTFOLIO_BACKTEST_STRATEGY_PCT])
        backtest_years = int(st.session_state[KEY_PORTFOLIO_BACKTEST_YEARS])
        selected_floor_model = st.session_state.get(KEY_PORTFOLIO_BACKTEST_FLOOR_MODEL, "-2σ")
        floor_intercept_a = None
        floor_slope_b = None
        floor_model_label = floor_model_options.get(selected_floor_model, "-2σ")
        if selected_floor_model == "trough_envelope_sigma_1":
            trough_overlay = calculate_peak_powerlaw_overlay(
                df_display,
                genesis_offset,
                df_display["Days"].to_numpy(dtype=float),
                percentile_offsets,
                1.0,
            ).get("trough")
            if trough_overlay is None:
                st.warning(
                    "Trough PowerLaw Envelope σ1 has too few fit points. Falling back to -2σ floor."
                )
                floor_model_label = "-2σ"
            else:
                floor_intercept_a = float(trough_overlay["intercept"])
                floor_slope_b = float(trough_overlay["slope"])
        result = build_portfolio_real_data_backtest(
            df_display,
            settings,
            currency_unit,
            years=backtest_years,
            current_gen_date=current_gen_date,
            intercept_a=a_active,
            slope_b=b_active,
            percentile_offsets=percentile_offsets,
            sell_mom_change_pct=selected_sell_pct,
            strategy_name=f"{floor_model_label}: sell {selected_sell_pct:.0f}% growth",
            floor_intercept_a=floor_intercept_a,
            floor_slope_b=floor_slope_b,
            floor_model_label=floor_model_label,
        )
    else:
        st.caption("Choose a strategy and period, then click Test strategy.")

    if st.session_state.get(KEY_PORTFOLIO_BACKTEST_HAS_RUN, False) and result is not None:
        st.markdown(f"##### Last {backtest_years} years: {result.strategy_name}")
        total_withdrawal = float(result.backtest_df["MonthlyWithdrawal"].sum())
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Starting BTC",
            f"{max(float(settings.btc_amount), 0.0):,.4f} BTC",
            delta=format_portfolio_money(result.start_value),
        )
        m2.metric(
            "Hold-only today",
            format_portfolio_money(result.hold_last_value),
            delta=f"{result.total_return_pct:+.1f}%",
        )
        m3.metric(
            "Strategy today",
            format_portfolio_money(result.strategy_last_value),
            delta=f"{result.strategy_btc:.4f} BTC left",
        )
        m4.metric("Total withdrawn", format_portfolio_money(total_withdrawal))

        backtest_fig = go.Figure()
        backtest_fig.add_trace(
            go.Scatter(
                x=result.backtest_df["Date"],
                y=result.backtest_df["HoldValue"],
                mode="lines+markers",
                name="Hold-only value",
                line=dict(color="#f0b90b", width=2),
                hovertemplate=(
                    "<b>%{x|%Y-%m}</b><br>Hold-only: "
                    f"{currency_prefix}%{{y:,.{display_currency_decimals}f}}{currency_suffix}<extra></extra>"
                ),
            )
        )
        selected_color = strategy_colors[selected_sell_pct]
        backtest_fig.add_trace(
            go.Scatter(
                x=result.backtest_df["Date"],
                y=result.backtest_df["StrategyValue"],
                mode="lines+markers",
                name=result.strategy_name,
                line=dict(color=selected_color, width=2),
                hovertemplate=(
                    f"<b>%{{x|%Y-%m}}</b><br>{result.strategy_name}: "
                    f"{currency_prefix}%{{y:,.{display_currency_decimals}f}}{currency_suffix}<extra></extra>"
                ),
            )
        )
        backtest_fig.add_trace(
            go.Scatter(
                x=result.backtest_df["Date"],
                y=result.backtest_df["MonthlyWithdrawal"],
                mode="lines",
                name=f"Monthly withdrawal {result.sell_mom_change_pct:.0f}%",
                line=dict(color="#38bdf8", width=1.8, dash="dot"),
                hovertemplate=(
                    f"<b>%{{x|%Y-%m}}</b><br>Withdrawal {result.sell_mom_change_pct:.0f}%: "
                    f"{currency_prefix}%{{y:,.{display_currency_decimals}f}}{currency_suffix}<extra></extra>"
                ),
            )
        )
        backtest_fig.add_trace(
            go.Scatter(
                x=result.backtest_df["Date"],
                y=-result.backtest_df["NetCashFlow"],
                mode="lines",
                name="Cumulative withdrawals",
                line=dict(color="#8b5cf6", width=1.8, dash="dash"),
                hovertemplate=(
                    "<b>%{x|%Y-%m}</b><br>Cumulative withdrawals: "
                    f"{currency_prefix}%{{y:,.{display_currency_decimals}f}}{currency_suffix}<extra></extra>"
                ),
            )
        )
        total_strategy_value = (
            result.backtest_df["StrategyValue"] - result.backtest_df["NetCashFlow"]
        )
        backtest_fig.add_trace(
            go.Scatter(
                x=result.backtest_df["Date"],
                y=total_strategy_value,
                mode="lines+markers",
                name="Strategy value + cumulative withdrawals",
                line=dict(color="#22c55e", width=2.2),
                visible="legendonly",
                hovertemplate=(
                    "<b>%{x|%Y-%m}</b><br>Total strategy value: "
                    f"{currency_prefix}%{{y:,.{display_currency_decimals}f}}{currency_suffix}<extra></extra>"
                ),
            )
        )
        backtest_fig.update_layout(
            height=320,
            margin=dict(t=10, b=0, l=50, r=20),
            template=pl_template,
            font=dict(color=pl_text_color),
            paper_bgcolor=pl_bg_color,
            plot_bgcolor=pl_bg_color,
            xaxis=dict(gridcolor=pl_grid_color, tickfont=dict(color=pl_text_color)),
            yaxis=dict(gridcolor=pl_grid_color, tickfont=dict(color=pl_text_color)),
            hoverlabel=dict(
                bgcolor=c_hover_bg,
                bordercolor=c_border,
                font=dict(color=c_hover_text, size=13),
            ),
        )
        st.plotly_chart(
            backtest_fig,
            width="stretch",
            theme=None,
            config={"displayModeBar": False},
            key=f"portfolio_backtest_{st.session_state[KEY_THEME_MODE]}_{st.session_state[KEY_CHART_REVISION]}",
        )

        backtest_format = {
            f"Actual BTC price ({currency_unit})": money_fmt,
            f"Hold-only value ({currency_unit})": money_fmt,
            f"Strategy value ({currency_unit})": money_fmt,
            f"Monthly buy ({currency_unit})": money_fmt,
            result.monthly_withdrawal_label: money_fmt,
            f"Net cash flow ({currency_unit})": money_fmt,
            "BTC after strategy": "{:,.6f}",
        }
        st.dataframe(
            style_backtest_table(
                result.table_df,
                backtest_format,
                currency_unit,
                result.monthly_withdrawal_label,
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            f"Backtest starts with the sidebar BTC quantity. Withdrawals are calculated from one month of model growth on the selected {result.monthly_withdrawal_label.replace(f' ({currency_unit})', '')}, then sold at the real historical monthly BTC price."
        )


# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_icon="🚀",
    page_title="BTC Power Law Pro",
    initial_sidebar_state="expanded",
)


try:
    raw_df_usd = load_prepared_price_data()
except Exception as e:
    st.error(f"Error loading BTC price data: {e}")
    st.stop()

try:
    raw_revenue_df = load_prepared_miner_revenue_data()
except Exception as e:
    st.error(f"Error loading miner revenue data: {e}")
    st.stop()

try:
    raw_difficulty_df = load_prepared_difficulty_data()
except Exception as e:
    st.error(f"Error loading difficulty data: {e}")
    st.stop()

try:
    raw_hashrate_df = load_prepared_hashrate_data()
except Exception as e:
    st.error(f"Error loading hashrate data: {e}")
    st.stop()

try:
    raw_lightning_nodes_df = load_prepared_lightning_nodes_data()
except Exception as e:
    st.error(f"Error loading Lightning node data: {e}")
    st.stop()

try:
    raw_lightning_capacity_df = load_prepared_lightning_capacity_data()
except Exception as e:
    st.error(f"Error loading Lightning capacity data: {e}")
    st.stop()

try:
    raw_liquid_btc_df = load_prepared_liquid_btc_data()
except Exception as e:
    st.error(f"Error loading Liquid BTC data: {e}")
    st.stop()

try:
    raw_liquid_transactions_df = load_prepared_liquid_transactions_data()
except Exception as e:
    st.error(f"Error loading Liquid transactions data: {e}")
    st.stop()

try:
    raw_filecoin_btc_df = load_prepared_filecoin_btc_data()
except Exception as e:
    st.error(f"Error loading Filecoin/BTC data: {e}")
    st.stop()

try:
    raw_monero_btc_df = load_prepared_monero_btc_data()
except Exception as e:
    st.error(f"Error loading Monero/BTC data: {e}")
    st.stop()

try:
    raw_litecoin_btc_df = load_prepared_litecoin_btc_data()
except Exception as e:
    st.error(f"Error loading Litecoin/BTC data: {e}")
    st.stop()

try:
    raw_dogecoin_btc_df = load_prepared_dogecoin_btc_data()
except Exception as e:
    st.error(f"Error loading Dogecoin/BTC data: {e}")
    st.stop()

try:
    raw_us_m2_df = load_prepared_us_m2_data()
except Exception as e:
    st.error(f"Error loading U.S. M2 data: {e}")
    st.stop()

try:
    raw_russian_m2_df = load_prepared_russian_m2_data()
except Exception as e:
    st.error(f"Error loading Russian M2 data: {e}")
    st.stop()

if KEY_CURRENCY_SELECTOR not in st.session_state:
    st.session_state[KEY_CURRENCY_SELECTOR] = CURRENCY_EURO

raw_df_usd = raw_df_usd[raw_df_usd["Close"] > 0].copy()
raw_df_usd["LogClose"] = np.log10(raw_df_usd["Close"])
raw_revenue_df = raw_revenue_df[raw_revenue_df["Close"] > 0].copy()
raw_revenue_df["LogClose"] = np.log10(raw_revenue_df["Close"])
raw_difficulty_df = raw_difficulty_df[raw_difficulty_df["Close"] > 0].copy()
raw_difficulty_df["LogClose"] = np.log10(raw_difficulty_df["Close"])
raw_hashrate_df = raw_hashrate_df[raw_hashrate_df["Close"] > 0].copy()
raw_hashrate_df["LogClose"] = np.log10(raw_hashrate_df["Close"])
raw_lightning_nodes_df = raw_lightning_nodes_df[raw_lightning_nodes_df["Close"] > 0].copy()
raw_lightning_nodes_df["LogClose"] = np.log10(raw_lightning_nodes_df["Close"])
raw_lightning_capacity_df = raw_lightning_capacity_df[raw_lightning_capacity_df["Close"] > 0].copy()
raw_lightning_capacity_df["LogClose"] = np.log10(raw_lightning_capacity_df["Close"])
raw_liquid_btc_df = raw_liquid_btc_df[raw_liquid_btc_df["Close"] > 0].copy()
raw_liquid_btc_df["LogClose"] = np.log10(raw_liquid_btc_df["Close"])
raw_liquid_transactions_df = raw_liquid_transactions_df[
    raw_liquid_transactions_df["Close"] > 0
].copy()
raw_liquid_transactions_df["LogClose"] = np.log10(raw_liquid_transactions_df["Close"])
raw_filecoin_btc_df = raw_filecoin_btc_df[raw_filecoin_btc_df["Close"] > 0].copy()
raw_filecoin_btc_df["LogClose"] = np.log10(raw_filecoin_btc_df["Close"])
raw_monero_btc_df = raw_monero_btc_df[raw_monero_btc_df["Close"] > 0].copy()
raw_monero_btc_df["LogClose"] = np.log10(raw_monero_btc_df["Close"])
raw_litecoin_btc_df = raw_litecoin_btc_df[raw_litecoin_btc_df["Close"] > 0].copy()
raw_litecoin_btc_df["LogClose"] = np.log10(raw_litecoin_btc_df["Close"])
raw_dogecoin_btc_df = raw_dogecoin_btc_df[raw_dogecoin_btc_df["Close"] > 0].copy()
raw_dogecoin_btc_df["LogClose"] = np.log10(raw_dogecoin_btc_df["Close"])
raw_us_m2_df = raw_us_m2_df[raw_us_m2_df["Close"] > 0].copy()
raw_us_m2_df["LogClose"] = np.log10(raw_us_m2_df["Close"])
raw_russian_m2_df = raw_russian_m2_df[raw_russian_m2_df["Close"] > 0].copy()
raw_russian_m2_df["LogClose"] = np.log10(raw_russian_m2_df["Close"])
raw_bitcoin_volatility_df = build_prepared_bitcoin_volatility_data(raw_df_usd)
raw_bitcoin_network_simulation_df = prepare_bitcoin_network_simulation(
    raw_df_usd,
    seed=int(st.session_state.get(KEY_BITCOIN_NETWORK_SIMULATION_SEED, 1)),
    resolution_days=float(st.session_state.get(KEY_BITCOIN_NETWORK_SIMULATION_RESOLUTION, 0.00001)),
)

# Use current session currency for sidebar AF/R2 calculations in PowerLaw Bitcoin mode.
sidebar_currency = st.session_state.get(KEY_CURRENCY_SELECTOR, CURRENCY_DOLLAR)
if sidebar_currency not in [CURRENCY_EURO, CURRENCY_DOLLAR, CURRENCY_UAH, CURRENCY_GOLD]:
    sidebar_currency = CURRENCY_DOLLAR
sidebar_price_close = build_currency_close_series(raw_df_usd, sidebar_currency)
sidebar_price_close = sidebar_price_close[sidebar_price_close > 0]
sidebar_price_log_close = np.log10(sidebar_price_close.values)

raw_series_frames = {
    POWERLAW_SERIES_PRICE: raw_df_usd,
    POWERLAW_SERIES_REVENUE: raw_revenue_df,
    POWERLAW_SERIES_BITCOIN_VOLATILITY: raw_bitcoin_volatility_df,
    POWERLAW_SERIES_DIFFICULTY: raw_difficulty_df,
    POWERLAW_SERIES_HASHRATE: raw_hashrate_df,
    POWERLAW_SERIES_BITCOIN_NETWORK_SIMULATION: raw_bitcoin_network_simulation_df,
    POWERLAW_SERIES_LIGHTNING_NODES: raw_lightning_nodes_df,
    POWERLAW_SERIES_LIGHTNING_CAPACITY: raw_lightning_capacity_df,
    POWERLAW_SERIES_LIQUID_BTC: raw_liquid_btc_df,
    POWERLAW_SERIES_LIQUID_TRANSACTIONS: raw_liquid_transactions_df,
    POWERLAW_SERIES_FILECOIN_BTC: raw_filecoin_btc_df,
    POWERLAW_SERIES_MONERO_BTC: raw_monero_btc_df,
    POWERLAW_SERIES_LITECOIN_BTC: raw_litecoin_btc_df,
    POWERLAW_SERIES_DOGECOIN_BTC: raw_dogecoin_btc_df,
    POWERLAW_SERIES_US_M2: raw_us_m2_df,
    POWERLAW_SERIES_RUSSIAN_M2: raw_russian_m2_df,
}
sidebar_series_data = {
    POWERLAW_SERIES_PRICE: {
        "absolute_days": raw_df_usd["AbsDays"].values,
        "log_close": sidebar_price_log_close,
    },
    POWERLAW_SERIES_REVENUE: {
        "absolute_days": raw_revenue_df["AbsDays"].values,
        "log_close": raw_revenue_df["LogClose"].values,
    },
    POWERLAW_SERIES_BITCOIN_VOLATILITY: {
        "absolute_days": raw_bitcoin_volatility_df["AbsDays"].values,
        "log_close": raw_bitcoin_volatility_df["LogClose"].values,
    },
    POWERLAW_SERIES_DIFFICULTY: {
        "absolute_days": raw_difficulty_df["AbsDays"].values,
        "log_close": raw_difficulty_df["LogClose"].values,
    },
    POWERLAW_SERIES_HASHRATE: {
        "absolute_days": raw_hashrate_df["AbsDays"].values,
        "log_close": raw_hashrate_df["LogClose"].values,
    },
    POWERLAW_SERIES_BITCOIN_NETWORK_SIMULATION: {
        "absolute_days": raw_bitcoin_network_simulation_df["AbsDays"].values,
        "log_close": raw_bitcoin_network_simulation_df["LogClose"].values,
    },
    POWERLAW_SERIES_LIGHTNING_NODES: {
        "absolute_days": raw_lightning_nodes_df["AbsDays"].values,
        "log_close": raw_lightning_nodes_df["LogClose"].values,
    },
    POWERLAW_SERIES_LIGHTNING_CAPACITY: {
        "absolute_days": raw_lightning_capacity_df["AbsDays"].values,
        "log_close": raw_lightning_capacity_df["LogClose"].values,
    },
    POWERLAW_SERIES_LIQUID_BTC: {
        "absolute_days": raw_liquid_btc_df["AbsDays"].values,
        "log_close": raw_liquid_btc_df["LogClose"].values,
    },
    POWERLAW_SERIES_LIQUID_TRANSACTIONS: {
        "absolute_days": raw_liquid_transactions_df["AbsDays"].values,
        "log_close": raw_liquid_transactions_df["LogClose"].values,
    },
    POWERLAW_SERIES_FILECOIN_BTC: {
        "absolute_days": raw_filecoin_btc_df["AbsDays"].values,
        "log_close": raw_filecoin_btc_df["LogClose"].values,
    },
    POWERLAW_SERIES_MONERO_BTC: {
        "absolute_days": raw_monero_btc_df["AbsDays"].values,
        "log_close": raw_monero_btc_df["LogClose"].values,
    },
    POWERLAW_SERIES_LITECOIN_BTC: {
        "absolute_days": raw_litecoin_btc_df["AbsDays"].values,
        "log_close": raw_litecoin_btc_df["LogClose"].values,
    },
    POWERLAW_SERIES_DOGECOIN_BTC: {
        "absolute_days": raw_dogecoin_btc_df["AbsDays"].values,
        "log_close": raw_dogecoin_btc_df["LogClose"].values,
    },
    POWERLAW_SERIES_US_M2: {
        "absolute_days": raw_us_m2_df["AbsDays"].values,
        "log_close": raw_us_m2_df["LogClose"].values,
    },
    POWERLAW_SERIES_RUSSIAN_M2: {
        "absolute_days": raw_russian_m2_df["AbsDays"].values,
        "log_close": raw_russian_m2_df["LogClose"].values,
    },
}

# --- THEME + STATE ---
initialize_app_session_state()

theme = get_theme(True)
apply_theme_css(theme)

c_text_main = theme["c_text_main"]
pl_template = theme["pl_template"]
pl_bg_color = theme["pl_bg_color"]
pl_grid_color = theme["pl_grid_color"]
pl_btc_color = theme["pl_btc_color"]
pl_legend_color = theme["pl_legend_color"]
pl_text_color = theme["pl_text_color"]
c_hover_bg = theme["c_hover_bg"]
c_hover_text = theme["c_hover_text"]
c_border = theme["c_border"]

# --- SIDEBAR ASSEMBLY ---
(
    mode,
    currency,
    time_scale,
    price_scale,
    current_r2,
    powerlaw_series,
    logperiodic_series,
) = render_sidebar_panel(
    sidebar_series_data,
    c_text_main,
    APP_VERSION,
    FORECAST_HORIZON_MIN,
    FORECAST_HORIZON_MAX,
)
active_model = get_active_model_config(mode, powerlaw_series, logperiodic_series, currency)
st.session_state[KEY_A] = float(st.session_state.get(active_model.a_key, active_model.default_a))
st.session_state[KEY_B] = float(st.session_state.get(active_model.b_key, active_model.default_b))

selected_series_name = get_selected_series_name(mode, powerlaw_series, logperiodic_series)
active_series_supports_currency = series_supports_currency_selector(
    mode, powerlaw_series, logperiodic_series
)

if mode == MODE_POWERLAW and (not active_series_supports_currency) and currency != CURRENCY_DOLLAR:
    st.session_state[KEY_CURRENCY_SELECTOR] = CURRENCY_DOLLAR
    st.rerun()
if active_series_supports_currency and currency != st.session_state.get(
    KEY_CURRENCY_SELECTOR, CURRENCY_DOLLAR
):
    st.rerun()
if (
    mode == MODE_LOGPERIODIC
    and (not active_series_supports_currency)
    and currency != CURRENCY_DOLLAR
):
    st.session_state[KEY_CURRENCY_SELECTOR] = CURRENCY_DOLLAR
    st.rerun()

# --- MAIN CALCULATIONS ---
active_model = get_active_model_config(mode, powerlaw_series, logperiodic_series, currency)
session_genesis_offset = int(st.session_state.get(KEY_GENESIS_OFFSET, 0))
genesis_offset = (
    int(active_model.model_origin_abs_day)
    if active_model.model_origin_abs_day is not None
    else session_genesis_offset
)
current_gen_date = GENESIS_DATE + pd.Timedelta(days=genesis_offset)
if active_model.supports_currency_selector:
    raw_df = raw_df_usd.copy()
    raw_df["Close"] = build_currency_close_series(raw_df_usd, currency)
    raw_df = raw_df[raw_df["Close"] > 0].copy()
    raw_df["LogClose"] = np.log10(raw_df["Close"])
else:
    raw_df = raw_series_frames[selected_series_name].copy()

active_abs_days = raw_df["AbsDays"].values
active_a_key = active_model.a_key
active_b_key = active_model.b_key
active_default_a = active_model.default_a
active_default_b = active_model.default_b
target_series_name = active_model.target_series_name
target_series_unit = active_model.target_series_unit

if not active_model.supports_currency_selector:
    currency = CURRENCY_DOLLAR

valid_idx = active_abs_days > genesis_offset
if active_model.analysis_min_abs_day is not None:
    valid_idx = valid_idx & (active_abs_days >= int(active_model.analysis_min_abs_day))
df_display = raw_df.iloc[valid_idx].copy()
if df_display.empty:
    st.error("No data available for the selected parameters.")
    st.stop()

df_display["Days"] = df_display["AbsDays"] - genesis_offset
df_display["LogD"] = np.log10(df_display["Days"])
trend_result = resolve_trend_parameters(
    df_display["LogD"].values,
    df_display["LogClose"].values,
    intercept_a=float(st.session_state.get(active_a_key, active_default_a)),
    slope_b=float(st.session_state.get(active_b_key, active_default_b)),
    active_mode=mode,
)
a_active = trend_result.intercept_a
b_active = trend_result.slope_b
df_display["ModelLog"] = trend_result.trend_log_prices
df_display["Res"] = trend_result.residual_series
df_display["Fair"], _, fair_was_clipped = evaluate_powerlaw_values(
    df_display["ModelLog"].values,
    0.0,
    1.0,
)
historical_powerlaw_slopes = np.array([], dtype=float)
if mode == MODE_LOGPERIODIC:
    _, historical_powerlaw_slopes, _ = calculate_expanding_powerlaw_parameters(
        df_display["LogD"].values,
        df_display["LogClose"].values,
    )

currency_prefix = active_model.currency_prefix
currency_suffix = active_model.currency_suffix
currency_decimals = int(active_model.currency_decimals)
currency_unit = active_model.currency_unit
df_display["CloseDisplay"] = df_display["Close"]
df_display["FairDisplay"] = df_display["Fair"]

if mode in [MODE_POWERLAW, MODE_PORTFOLIO] and powerlaw_parameters_are_unstable(
    current_r2, was_clipped=fair_was_clipped
):
    st.warning(
        "Current PowerLaw parameters are unstable for the selected series. Use Auto-fit model or Reset parameters."
    )

# Use a shared LogPeriodic R² mask so scoring follows the same visible segment.
lp_r2_mask = np.ones(len(df_display), dtype=bool)
if mode == MODE_LOGPERIODIC and active_model.oscillator_min_abs_day is not None:
    lp_r2_mask = df_display["AbsDays"].values >= active_model.oscillator_min_abs_day

# Calculate R2 for Trend if not returned by sidebar (LogPeriodic mode)
if mode == MODE_LOGPERIODIC:
    if np.count_nonzero(lp_r2_mask) > 1:
        current_r2 = calculate_r2_score(
            df_display["LogClose"].values[lp_r2_mask],
            df_display["ModelLog"].values[lp_r2_mask],
        )
    else:
        current_r2 = 0.0

p2_5, p16_5, p83_5, p97_5 = calculate_percentile_offsets(df_display, genesis_offset)
residual_sigma_log = calculate_residual_sigma_log(df_display)

# --- OSCILLATOR CALC ---
osc_settings = oscillator.OscillatorSettings(
    t1_age=float(st.session_state.get("t1_age", OSC_DEFAULTS["t1_age"])),
    lambda_val=float(st.session_state.get("lambda_val", OSC_DEFAULTS["lambda_val"])),
    amp_factor_top=float(st.session_state.get("amp_factor_top", OSC_DEFAULTS["amp_factor_top"])),
    amp_factor_bottom=float(
        st.session_state.get("amp_factor_bottom", OSC_DEFAULTS["amp_factor_bottom"])
    ),
    impulse_damping=float(st.session_state.get("impulse_damping", OSC_DEFAULTS["impulse_damping"])),
    harmonic_count=int(st.session_state.get(KEY_LOGPERIODIC_HARMONICS, 1)),
)
osc_amp, osc_omega, osc_phi = 0.0, 0.0, 0.0
r2_combined = current_r2
osc_reference_log_day = float(df_display["LogD"].min())
osc_harmonic_coefficients = np.array([], dtype=float)
selected_harmonic_count = max(1, min(3, int(st.session_state.get(KEY_LOGPERIODIC_HARMONICS, 1))))
logperiodic_stats_rows = None
perrenod_stats_rows = None
perrenod_curve = None

if mode == MODE_LOGPERIODIC:
    try:
        osc_result = oscillator.compute_oscillator_overlay(
            df_display["LogD"].values,
            df_display["Res"].values,
            df_display["ModelLog"].values,
            df_display["LogClose"].values,
            lp_r2_mask,
            osc_settings,
            current_r2,
        )
        osc_settings = osc_result.settings
        osc_amp = osc_result.amplitude
        osc_omega = osc_result.angular_frequency
        osc_phi = osc_result.phase_shift
        r2_combined = osc_result.combined_r2
        osc_reference_log_day = osc_result.reference_log_day
        osc_harmonic_coefficients = osc_result.harmonic_coefficients
        stats_params = {
            "t1_age": osc_settings.t1_age,
            "lambda_val": osc_settings.lambda_val,
            "amp_factor_top": osc_settings.amp_factor_top,
            "amp_factor_bottom": osc_settings.amp_factor_bottom,
            "impulse_damping": osc_settings.impulse_damping,
        }
        fit_log_days = df_display["LogD"].values[lp_r2_mask]
        fit_residuals = df_display["Res"].values[lp_r2_mask]
        fit_days = df_display["Days"].values[lp_r2_mask]
        logperiodic_stats_rows = oscillator.compute_oscillator_model_stats_table(
            fit_log_days,
            fit_residuals,
            stats_params,
        )
        perrenod_stats_rows = oscillator.compute_perrenod_comparison_stats_table(
            fit_log_days,
            fit_residuals,
            fit_days,
            (
                active_model.oscillator_parameter_bounds.get("lambda_val", (1.5, 5.0))
                if active_model.oscillator_parameter_bounds
                else (1.5, 5.0)
            ),
        )
    except Exception as e:
        st.error(f"LogPeriodic Error: {e}")
        osc_settings = oscillator.OscillatorSettings(
            t1_age=OSC_DEFAULTS["t1_age"],
            lambda_val=OSC_DEFAULTS["lambda_val"],
            amp_factor_top=OSC_DEFAULTS["amp_factor_top"],
            amp_factor_bottom=OSC_DEFAULTS["amp_factor_bottom"],
            impulse_damping=OSC_DEFAULTS["impulse_damping"],
            harmonic_count=1,
        )
        osc_amp, osc_omega, osc_phi, r2_combined = 0, 0, 0, current_r2

# --- VIZ SETUP ---
view_max = _resolve_model_view_max(df_display, current_gen_date)

# Use daily grid so unified hover has matching x-values across all traces.
m_x, m_dates, m_log_d, m_fair_usd, m_dates_str = prepare_model_grid(
    current_gen_date, a_active, b_active, view_max
)
m_fair_display = m_fair_usd

m_osc_y = np.array([], dtype=float)
if mode == MODE_LOGPERIODIC:
    m_osc_y = oscillator.build_oscillator_curve(
        m_log_d,
        osc_amp,
        osc_omega,
        osc_phi,
        osc_settings.amp_factor_top,
        osc_settings.amp_factor_bottom,
        osc_settings.impulse_damping,
        osc_reference_log_day,
        osc_harmonic_coefficients,
    )
m_osc_y_by_harmonic = {selected_harmonic_count: m_osc_y}
if mode == MODE_LOGPERIODIC:
    m_osc_y_by_harmonic = {}
    for harmonic_count in range(1, selected_harmonic_count + 1):
        harmonic_settings = oscillator.OscillatorSettings(
            t1_age=osc_settings.t1_age,
            lambda_val=osc_settings.lambda_val,
            amp_factor_top=osc_settings.amp_factor_top,
            amp_factor_bottom=osc_settings.amp_factor_bottom,
            impulse_damping=osc_settings.impulse_damping,
            harmonic_count=harmonic_count,
        )
        harmonic_result = oscillator.compute_oscillator_overlay(
            df_display["LogD"].values,
            df_display["Res"].values,
            df_display["ModelLog"].values,
            df_display["LogClose"].values,
            lp_r2_mask,
            harmonic_settings,
            current_r2,
        )
        m_osc_y_by_harmonic[harmonic_count] = oscillator.build_oscillator_curve(
            m_log_d,
            harmonic_result.amplitude,
            harmonic_result.angular_frequency,
            harmonic_result.phase_shift,
            harmonic_result.settings.amp_factor_top,
            harmonic_result.settings.amp_factor_bottom,
            harmonic_result.settings.impulse_damping,
            harmonic_result.reference_log_day,
            harmonic_result.harmonic_coefficients,
        )
    if perrenod_stats_rows and bool(st.session_state.get(KEY_LOGPERIODIC_SHOW_DECAYED_DSI, True)):
        target_perrenod_row = next(
            (
                row
                for row in perrenod_stats_rows
                if row is not None and row.label == "DSI ω,2ω,4ω decayed"
            ),
            None,
        )
        if target_perrenod_row is not None:
            try:
                target_lambda = float(target_perrenod_row.parameter_label.split()[1])
                perrenod_curve_values = oscillator.build_dsi_regression_curve(
                    df_display["LogD"].values[lp_r2_mask],
                    df_display["Res"].values[lp_r2_mask],
                    m_log_d,
                    target_lambda,
                    harmonic_count=3,
                    fit_days_since_genesis=df_display["Days"].values[lp_r2_mask],
                    predict_days_since_genesis=m_x,
                    decay_model="reciprocal_age",
                )
                if perrenod_curve_values is not None:
                    perrenod_curve = {
                        "label": target_perrenod_row.label,
                        "r2": target_perrenod_row.r2,
                        "values": perrenod_curve_values,
                    }
            except (IndexError, TypeError, ValueError):
                perrenod_curve = None

is_log_time = time_scale == TIME_LOG
plot_x_model = m_x if is_log_time else m_dates
plot_x_main = df_display["Days"] if is_log_time else df_display.index
plot_x_osc = df_display["Days"] if is_log_time else df_display.index
peak_powerlaw_overlay = None
if mode == MODE_POWERLAW:
    peak_powerlaw_overlay = calculate_peak_powerlaw_overlay(
        df_display,
        genesis_offset,
        m_x,
        (p2_5, p16_5, p83_5, p97_5),
        st.session_state.get(KEY_POWERLAW_ENVELOPE_SIGMA, 1.0),
    )

if mode in [MODE_POWERLAW, MODE_LOGPERIODIC]:
    render_main_model_chart(
        mode=mode,
        time_scale=time_scale,
        price_scale=price_scale,
        df_display=df_display,
        current_gen_date=current_gen_date,
        view_max=view_max,
        plot_x_model=plot_x_model,
        plot_x_main=plot_x_main,
        plot_x_osc=plot_x_osc,
        m_log_d=m_log_d,
        m_dates=m_dates,
        m_dates_str=m_dates_str,
        m_fair_display=m_fair_display,
        historical_powerlaw_slopes=historical_powerlaw_slopes,
        show_historical_powerlaw_slope=mode == MODE_LOGPERIODIC,
        m_osc_y=m_osc_y,
        m_osc_y_by_harmonic=m_osc_y_by_harmonic,
        perrenod_curve=perrenod_curve,
        residual_sigma_log=residual_sigma_log,
        p2_5=p2_5,
        p16_5=p16_5,
        p83_5=p83_5,
        p97_5=p97_5,
        peak_powerlaw_overlay=peak_powerlaw_overlay,
        osc_t1_age=osc_settings.t1_age,
        osc_lambda=osc_settings.lambda_val,
        selected_harmonic_count=selected_harmonic_count,
        pl_template=pl_template,
        pl_bg_color=pl_bg_color,
        pl_grid_color=pl_grid_color,
        pl_btc_color=pl_btc_color,
        pl_legend_color=pl_legend_color,
        pl_text_color=pl_text_color,
        c_hover_bg=c_hover_bg,
        c_hover_text=c_hover_text,
        c_border=c_border,
        currency_prefix=currency_prefix,
        currency_suffix=currency_suffix,
        currency_decimals=currency_decimals,
        target_series_name=target_series_name,
        target_series_unit=target_series_unit,
        show_halving_lines=mode == MODE_POWERLAW and active_model.show_halving_lines,
        osc_visible_start_abs_day=(
            active_model.oscillator_min_abs_day if mode == MODE_LOGPERIODIC else None
        ),
        chart_key=(
            f"chart_{mode}_{powerlaw_series}_{currency}_{time_scale}_{price_scale}_"
            f"{selected_harmonic_count}_"
            f"{st.session_state[KEY_THEME_MODE]}_{st.session_state[KEY_CHART_REVISION]}"
        ),
    )
else:
    render_portfolio_view(
        df_display,
        current_gen_date,
        a_active,
        b_active,
        (p2_5, p16_5, p83_5, p97_5),
        current_r2,
        fair_was_clipped,
        currency_prefix,
        currency_suffix,
        currency_decimals,
        currency_unit,
        pl_template,
        pl_text_color,
        pl_bg_color,
        pl_grid_color,
        c_hover_bg,
        c_border,
        c_hover_text,
    )

# --- KPI ---
render_model_kpis(
    df_display,
    a_active,
    b_active,
    p2_5,
    p16_5,
    p83_5,
    p97_5,
    currency_prefix,
    currency_suffix,
    currency_decimals,
    target_series_name,
    target_series_unit,
    logperiodic_stats_rows=logperiodic_stats_rows,
    perrenod_stats_rows=perrenod_stats_rows,
)
