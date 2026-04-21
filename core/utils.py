from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

from core.constants import POWERLAW_EXPONENT_MAX, POWERLAW_EXPONENT_MIN


@dataclass(frozen=True)
class PortfolioSettings:
    btc_amount: float
    monthly_buy_amount: float
    forecast_unit: str
    forecast_horizon: int


@dataclass(frozen=True)
class PortfolioProjectionResult:
    portfolio_df: pd.DataFrame
    table_title: str
    forecast_unit: str
    change_usd_col: str
    change_pct_col: str


@dataclass(frozen=True)
class PortfolioViewModel:
    portfolio_display_df: pd.DataFrame
    table_df: pd.DataFrame
    table_title: str
    dca_enabled: bool
    baseline_value: float
    last_value: float
    last_dca_value: float
    last_dca_invested_capital: float
    total_growth_pct: float
    period_change_usd_label: str
    period_change_pct_label: str


@dataclass(frozen=True)
class TrendComputationResult:
    intercept_a: float
    slope_b: float
    trend_log_prices: np.ndarray
    residual_series: np.ndarray


def resolve_projection_anchor_day(df_index, today=None):
    latest_data_day = pd.Timestamp(df_index.max()).normalize()
    if today is None:
        today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    else:
        today = pd.Timestamp(today)
        if today.tzinfo is not None:
            today = today.tz_localize(None)
        today = today.normalize()
    return max(latest_data_day, today)


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


def resolve_trend_parameters(log_days, log_prices, *, intercept_a, slope_b, active_mode):
    _, clipped_exponents, _ = evaluate_powerlaw_values(
        log_days,
        intercept_a,
        slope_b,
    )
    trend_log_prices = clipped_exponents
    residual_series = log_prices - trend_log_prices

    if active_mode == "LogPeriodic":
        intercept_a, slope_b, trend_log_prices, residual_series = get_stable_trend_fit(
            log_days,
            log_prices,
            intercept_a,
            slope_b,
        )

    return TrendComputationResult(
        intercept_a=float(intercept_a),
        slope_b=float(slope_b),
        trend_log_prices=np.asarray(trend_log_prices, dtype=float),
        residual_series=np.asarray(residual_series, dtype=float),
    )


def evaluate_powerlaw_values(
    log_days,
    intercept_a,
    slope_b,
    exponent_min=POWERLAW_EXPONENT_MIN,
    exponent_max=POWERLAW_EXPONENT_MAX,
):
    exponents = intercept_a + slope_b * np.asarray(log_days, dtype=float)
    clipped_exponents = np.clip(exponents, float(exponent_min), float(exponent_max))
    values = np.power(10.0, clipped_exponents)
    was_clipped = bool(np.any(~np.isclose(exponents, clipped_exponents)))
    return values, clipped_exponents, was_clipped


def powerlaw_parameters_are_unstable(
    r2_score,
    *,
    was_clipped=False,
    min_r2=0.0,
):
    if was_clipped:
        return True
    if not np.isfinite(r2_score):
        return True
    return float(r2_score) < float(min_r2)


def normalize_periodic_growth_rate(current_values, previous_values, elapsed_days, target_days):
    current_arr = np.asarray(current_values, dtype=float)
    previous_arr = np.asarray(previous_values, dtype=float)
    elapsed_arr = np.asarray(elapsed_days, dtype=float)
    normalized = np.full(current_arr.shape, np.nan, dtype=float)

    valid_mask = (
        np.isfinite(current_arr)
        & np.isfinite(previous_arr)
        & np.isfinite(elapsed_arr)
        & (current_arr > 0.0)
        & (previous_arr > 0.0)
        & (elapsed_arr > 0.0)
    )
    if not np.any(valid_mask):
        return normalized

    gross_return = current_arr[valid_mask] / previous_arr[valid_mask]
    normalized[valid_mask] = (
        np.power(gross_return, float(target_days) / elapsed_arr[valid_mask]) - 1.0
    ) * 100.0
    return normalized


def calculate_monthly_buy_portfolio_values(
    date_index,
    current_gen_date,
    fair_prices,
    intercept_a,
    slope_b,
    initial_btc_amount,
    monthly_buy_amount,
    purchase_anchor_day,
):
    projection_dates = pd.to_datetime(date_index)
    fair_price_arr = np.asarray(fair_prices, dtype=float)
    total_btc = np.full(fair_price_arr.shape, float(initial_btc_amount), dtype=float)
    invested_capital = np.zeros(fair_price_arr.shape, dtype=float)

    if (
        projection_dates.empty
        or float(monthly_buy_amount) <= 0.0
        or fair_price_arr.size == 0
        or not np.any(np.isfinite(fair_price_arr) & (fair_price_arr > 0.0))
    ):
        return total_btc, fair_price_arr * total_btc, invested_capital

    purchase_start = pd.Timestamp(purchase_anchor_day).to_period("M").to_timestamp()
    if purchase_start < pd.Timestamp(purchase_anchor_day).normalize():
        purchase_start += pd.offsets.MonthBegin(1)

    purchase_end = pd.Timestamp(projection_dates.max()).normalize()
    purchase_dates = pd.date_range(start=purchase_start, end=purchase_end, freq="MS")
    if purchase_dates.empty:
        return total_btc, fair_price_arr * total_btc, invested_capital

    purchase_days = np.maximum((purchase_dates - current_gen_date).days.astype(float), 1.0)
    purchase_prices, _, _ = evaluate_powerlaw_values(
        np.log10(purchase_days),
        intercept_a,
        slope_b,
    )
    valid_purchase_mask = np.isfinite(purchase_prices) & (purchase_prices > 0.0)
    if not np.any(valid_purchase_mask):
        return total_btc, fair_price_arr * total_btc, invested_capital

    valid_purchase_dates = purchase_dates[valid_purchase_mask]
    purchased_btc = float(monthly_buy_amount) / purchase_prices[valid_purchase_mask]
    cumulative_btc = np.cumsum(purchased_btc)
    cumulative_invested_capital = np.cumsum(
        np.full(valid_purchase_dates.shape, float(monthly_buy_amount), dtype=float)
    )
    purchase_positions = (
        np.searchsorted(
            valid_purchase_dates.to_numpy(dtype="datetime64[ns]"),
            projection_dates.to_numpy(dtype="datetime64[ns]"),
            side="right",
        )
        - 1
    )

    additional_btc = np.zeros_like(total_btc)
    applicable_mask = purchase_positions >= 0
    additional_btc[applicable_mask] = cumulative_btc[purchase_positions[applicable_mask]]
    invested_capital[applicable_mask] = cumulative_invested_capital[
        purchase_positions[applicable_mask]
    ]
    total_btc = total_btc + additional_btc

    return total_btc, fair_price_arr * total_btc, invested_capital


def build_portfolio_projection(
    df_index,
    current_gen_date,
    intercept_a,
    slope_b,
    settings,
    anchor_day=None,
):
    average_month_days = 30.44
    anchor_day = resolve_projection_anchor_day(df_index, today=anchor_day)

    if settings.forecast_unit == "Year":
        latest_year = int(anchor_day.year)
        start_period = pd.Timestamp(f"{latest_year - 1}-01-01")
        date_index = pd.date_range(
            start=start_period,
            periods=settings.forecast_horizon + 1,
            freq="YS",
        )
        change_usd_col, change_pct_col = "YoY_USD", "YoY_pct"
        table_title = "Yearly growth table"
    elif settings.forecast_unit == "Day":
        latest_day = anchor_day
        start_period = latest_day - pd.Timedelta(days=1)
        date_index = pd.date_range(
            start=start_period,
            periods=settings.forecast_horizon + 1,
            freq="D",
        )
        change_usd_col, change_pct_col = "DoD_USD", "DoD_pct"
        table_title = "Daily growth table"
    else:
        latest_month_start = anchor_day.to_period("M").to_timestamp()
        start_period = latest_month_start - pd.offsets.MonthBegin(1)
        date_index = pd.date_range(
            start=start_period,
            periods=settings.forecast_horizon + 1,
            freq="MS",
        )
        change_usd_col, change_pct_col = "MoM_USD", "MoM_pct"
        table_title = "Monthly growth table"

    period_days = np.maximum((date_index - current_gen_date).days.astype(float), 1.0)
    period_fair_price, _, _ = evaluate_powerlaw_values(
        np.log10(period_days),
        intercept_a,
        slope_b,
    )
    period_portfolio_value = period_fair_price * settings.btc_amount
    dca_btc_holdings, dca_portfolio_value, dca_invested_capital = (
        calculate_monthly_buy_portfolio_values(
            date_index=date_index,
            current_gen_date=current_gen_date,
            fair_prices=period_fair_price,
            intercept_a=intercept_a,
            slope_b=slope_b,
            initial_btc_amount=settings.btc_amount,
            monthly_buy_amount=settings.monthly_buy_amount,
            purchase_anchor_day=anchor_day,
        )
    )

    portfolio_df = pd.DataFrame(
        {
            "Date": date_index,
            "FairPriceUSD": period_fair_price,
            "PortfolioUSD": period_portfolio_value,
            "DcaBTC": dca_btc_holdings,
            "DcaPortfolioUSD": dca_portfolio_value,
            "DcaInvestedCapitalUSD": dca_invested_capital,
        }
    )
    portfolio_df[change_usd_col] = portfolio_df["PortfolioUSD"].diff()
    if settings.forecast_unit == "Month":
        elapsed_days = portfolio_df["Date"].diff().dt.days.to_numpy(dtype=float)
        previous_values = portfolio_df["PortfolioUSD"].shift(1).to_numpy(dtype=float)
        current_values = portfolio_df["PortfolioUSD"].to_numpy(dtype=float)
        portfolio_df[change_pct_col] = normalize_periodic_growth_rate(
            current_values,
            previous_values,
            elapsed_days,
            average_month_days,
        )
    else:
        portfolio_df[change_pct_col] = portfolio_df["PortfolioUSD"].pct_change() * 100

    return PortfolioProjectionResult(
        portfolio_df=portfolio_df,
        table_title=table_title,
        forecast_unit=settings.forecast_unit,
        change_usd_col=change_usd_col,
        change_pct_col=change_pct_col,
    )


def get_growth_change_labels(forecast_unit, currency_unit):
    prefix = "YoY" if forecast_unit == "Year" else ("DoD" if forecast_unit == "Day" else "MoM")
    return f"{prefix} Change ({currency_unit})", f"{prefix} Change (%)"


def build_portfolio_view_model(projection_result, monthly_buy_amount, currency_unit):
    portfolio_display_df = projection_result.portfolio_df.iloc[1:].copy()
    portfolio_display_df["FairPriceDisplay"] = portfolio_display_df["FairPriceUSD"]
    portfolio_display_df["PortfolioDisplay"] = portfolio_display_df["PortfolioUSD"]
    portfolio_display_df["DcaPortfolioDisplay"] = portfolio_display_df["DcaPortfolioUSD"]
    portfolio_display_df["DcaBTCDisplay"] = portfolio_display_df["DcaBTC"]
    portfolio_display_df["DcaInvestedCapitalDisplay"] = portfolio_display_df[
        "DcaInvestedCapitalUSD"
    ]
    portfolio_display_df["ChangeDisplay"] = portfolio_display_df[projection_result.change_usd_col]
    dca_enabled = monthly_buy_amount > 0.0

    baseline_value = projection_result.portfolio_df["PortfolioUSD"].iloc[0]
    last_value = portfolio_display_df["PortfolioUSD"].iloc[-1]
    last_dca_value = portfolio_display_df["DcaPortfolioDisplay"].iloc[-1]
    last_dca_invested_capital = portfolio_display_df["DcaInvestedCapitalDisplay"].iloc[-1]
    total_growth_pct = (
        ((last_value - baseline_value) / baseline_value) * 100 if baseline_value > 0 else 0.0
    )

    period_change_usd_label, period_change_pct_label = get_growth_change_labels(
        projection_result.forecast_unit,
        currency_unit,
    )
    table_df = portfolio_display_df.copy()
    if projection_result.forecast_unit == "Year":
        table_df["Date"] = table_df["Date"].dt.strftime("%Y")
    elif projection_result.forecast_unit == "Day":
        table_df["Date"] = table_df["Date"].dt.strftime("%Y-%m-%d")
    else:
        table_df["Date"] = table_df["Date"].dt.strftime("%Y-%m")
    table_df = table_df.rename(
        columns={
            "FairPriceDisplay": f"Fair Price ({currency_unit})",
            "PortfolioDisplay": f"Portfolio ({currency_unit})",
            "DcaPortfolioDisplay": f"Portfolio + monthly buys ({currency_unit})",
            "DcaInvestedCapitalDisplay": f"Invested cash ({currency_unit})",
            "DcaBTCDisplay": "BTC after monthly buys",
            "ChangeDisplay": period_change_usd_label,
            projection_result.change_pct_col: period_change_pct_label,
        }
    )
    display_columns = [
        "Date",
        f"Fair Price ({currency_unit})",
        f"Portfolio ({currency_unit})",
    ]
    if dca_enabled:
        display_columns.extend(
            [
                f"Portfolio + monthly buys ({currency_unit})",
                f"Invested cash ({currency_unit})",
                "BTC after monthly buys",
            ]
        )
    display_columns.extend([period_change_usd_label, period_change_pct_label])

    return PortfolioViewModel(
        portfolio_display_df=portfolio_display_df,
        table_df=table_df[display_columns],
        table_title=projection_result.table_title,
        dca_enabled=dca_enabled,
        baseline_value=baseline_value,
        last_value=last_value,
        last_dca_value=last_dca_value,
        last_dca_invested_capital=last_dca_invested_capital,
        total_growth_pct=total_growth_pct,
        period_change_usd_label=period_change_usd_label,
        period_change_pct_label=period_change_pct_label,
    )


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


def fancy_control(
    label,
    key,
    step,
    min_v,
    max_v,
    disabled=False,
    on_manual_change=None,
    on_auto_fit=None,
    auto_fit_label="AF",
    show_buttons=True,
):
    has_auto_fit = on_auto_fit is not None and show_buttons
    if not show_buttons:
        step_text = f"{step:.10f}".rstrip("0")
        precision = len(step_text.split(".")[1]) if "." in step_text else 0
        display_format = f"%.{precision}f"
        current_value = st.session_state.get(key, min_v)
        try:
            current_value = float(current_value)
        except (TypeError, ValueError):
            current_value = min_v
        st.session_state[key] = round(min(max_v, max(min_v, current_value)), precision)

        def on_slider_change():
            if on_manual_change is not None:
                on_manual_change()

        return st.slider(
            key,
            min_v,
            max_v,
            key=key,
            step=step,
            format=display_format,
            label_visibility="collapsed",
            disabled=disabled,
            on_change=on_slider_change,
        )

    if has_auto_fit:
        c1, c2, c3, c4 = st.columns([1, 2.5, 1, 1])
    else:
        c1, c2, c3 = st.columns([1, 2.5, 1])
    step_text = f"{step:.10f}".rstrip("0")
    precision = len(step_text.split(".")[1]) if "." in step_text else 0
    display_format = f"%.{precision}f"
    current_value = st.session_state.get(key, min_v)
    try:
        current_value = float(current_value)
    except (TypeError, ValueError):
        current_value = min_v
    st.session_state[key] = round(min(max_v, max(min_v, current_value)), precision)

    def on_minus():
        new_val = st.session_state[key] - step
        st.session_state[key] = round(max(min_v, new_val), precision)
        if on_manual_change is not None:
            on_manual_change()

    def on_plus():
        new_val = st.session_state[key] + step
        st.session_state[key] = round(min(max_v, new_val), precision)
        if on_manual_change is not None:
            on_manual_change()

    def on_slider_change():
        if on_manual_change is not None:
            on_manual_change()

    minus_clicked = c1.button("➖", key=f"{key}_m", disabled=disabled)
    plus_clicked = c3.button("➕", key=f"{key}_p", disabled=disabled)
    auto_fit_clicked = (
        c4.button(auto_fit_label, key=f"{key}_af", disabled=disabled) if has_auto_fit else False
    )

    if minus_clicked:
        on_minus()
    if plus_clicked:
        on_plus()
    if auto_fit_clicked and on_auto_fit is not None:
        on_auto_fit()

    return c2.slider(
        key,
        min_v,
        max_v,
        key=key,
        step=step,
        format=display_format,
        label_visibility="collapsed",
        disabled=disabled,
        on_change=on_slider_change,
    )
