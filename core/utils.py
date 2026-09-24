from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.constants import (
    DEFAULT_PORTFOLIO_HISTORY_PERIODS,
    GAUSSIAN_SIGMA_PERCENTILES,
    PORTFOLIO_HISTORY_PERIODS_MAX,
    PORTFOLIO_HISTORY_PERIODS_MIN,
    POWERLAW_EXPONENT_MAX,
    POWERLAW_EXPONENT_MIN,
)

# Average Gregorian month; monthly figures are reported over this span so that 28-31 day
# calendar months do not make them zigzag.
AVERAGE_MONTH_DAYS = 30.44


@dataclass(frozen=True)
class PortfolioSettings:
    btc_amount: float
    monthly_buy_amount: float
    forecast_unit: str
    forecast_horizon: int
    history_periods: int = DEFAULT_PORTFOLIO_HISTORY_PERIODS
    monthly_mom_change_pct: float = 0.0
    sigma_level: float = 0.0
    residual_sigma_log: float = 0.0
    residual_percentile_offsets_log: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class PortfolioProjectionResult:
    portfolio_df: pd.DataFrame
    table_title: str
    forecast_unit: str
    change_usd_col: str
    change_pct_col: str
    history_periods: int


@dataclass(frozen=True)
class PortfolioViewModel:
    portfolio_display_df: pd.DataFrame
    table_df: pd.DataFrame
    table_title: str
    dca_enabled: bool
    dca_is_buying: bool
    baseline_value: float
    last_value: float
    last_dca_value: float
    last_dca_invested_capital: float
    total_growth_pct: float
    period_change_usd_label: str
    period_change_pct_label: str
    period_cash_flow_label: str


@dataclass(frozen=True)
class PortfolioBacktestResult:
    strategy_name: str
    sell_mom_change_pct: float
    backtest_df: pd.DataFrame
    table_df: pd.DataFrame
    start_value: float
    hold_last_value: float
    strategy_last_value: float
    net_cash_flow: float
    strategy_btc: float
    total_return_pct: float
    strategy_return_pct: float
    monthly_withdrawal_label: str
    months_without_floor: int = 0


@dataclass(frozen=True)
class CurrentMonthlyPensionEstimate:
    current_sigma_level: float
    withdrawal_rating: str
    withdrawal_rating_color: str
    withdrawal_rating_note: str
    current_price: float
    current_floor_price: float
    next_month_floor_price: float
    next_month_price: float
    floor_monthly_growth_per_btc: float
    monthly_growth_per_btc: float
    minimum_monthly_withdrawal: float
    max_monthly_withdrawal: float
    minimum_btc_to_sell: float
    minimum_btc_to_sell_today: float
    minimum_btc_sell_reduction_pct: float
    minimum_btc_sell_today_delta_pct: float
    model_btc_to_sell: float
    selected_monthly_withdrawal: float
    next_month_date: pd.Timestamp


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


def resolve_trend_parameters(log_days, log_prices, *, intercept_a, slope_b):
    _, clipped_exponents, _ = evaluate_powerlaw_values(
        log_days,
        intercept_a,
        slope_b,
    )
    trend_log_prices = clipped_exponents
    residual_series = log_prices - trend_log_prices

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


def calculate_expanding_powerlaw_parameters(log_days, log_prices, min_points=100):
    log_days_arr = np.asarray(log_days, dtype=float)
    log_prices_arr = np.asarray(log_prices, dtype=float)
    valid_mask = np.isfinite(log_days_arr) & np.isfinite(log_prices_arr)

    x = np.where(valid_mask, log_days_arr, 0.0)
    y = np.where(valid_mask, log_prices_arr, 0.0)
    n = np.cumsum(valid_mask.astype(float))
    sum_x = np.cumsum(x)
    sum_y = np.cumsum(y)
    sum_xx = np.cumsum(x * x)
    sum_xy = np.cumsum(x * y)

    denominator = (n * sum_xx) - (sum_x * sum_x)
    slopes = np.full(log_days_arr.shape, np.nan, dtype=float)
    intercepts = np.full(log_days_arr.shape, np.nan, dtype=float)
    fitted_log_prices = np.full(log_days_arr.shape, np.nan, dtype=float)
    fit_mask = valid_mask & (n >= float(min_points)) & (np.abs(denominator) > 1e-12)
    if not np.any(fit_mask):
        return intercepts, slopes, fitted_log_prices

    slopes[fit_mask] = ((n[fit_mask] * sum_xy[fit_mask]) - (sum_x[fit_mask] * sum_y[fit_mask])) / (
        denominator[fit_mask]
    )
    intercepts[fit_mask] = (sum_y[fit_mask] - (slopes[fit_mask] * sum_x[fit_mask])) / n[fit_mask]
    fitted_log_prices[fit_mask] = intercepts[fit_mask] + (slopes[fit_mask] * log_days_arr[fit_mask])
    fitted_log_prices = np.clip(
        fitted_log_prices,
        float(POWERLAW_EXPONENT_MIN),
        float(POWERLAW_EXPONENT_MAX),
    )

    return intercepts, slopes, fitted_log_prices


def calculate_historical_sigma_offsets(
    log_days,
    log_prices,
    intercepts,
    slopes,
    min_points=100,
    recalculation_step=7,
):
    """Return causal percentile sigma offsets for successive historical fits."""
    log_days_arr = np.asarray(log_days, dtype=float)
    log_prices_arr = np.asarray(log_prices, dtype=float)
    intercepts_arr = np.asarray(intercepts, dtype=float)
    slopes_arr = np.asarray(slopes, dtype=float)
    offsets = np.full((4, len(log_days_arr)), np.nan, dtype=float)
    latest_offsets = None

    for index in range(len(log_days_arr)):
        if (
            index + 1 >= int(min_points)
            and np.isfinite(intercepts_arr[index])
            and np.isfinite(slopes_arr[index])
            and (latest_offsets is None or index % int(recalculation_step) == 0)
        ):
            historical_model = intercepts_arr[index] + slopes_arr[index] * log_days_arr[: index + 1]
            residuals = log_prices_arr[: index + 1] - historical_model
            finite_residuals = residuals[np.isfinite(residuals)]
            if len(finite_residuals) >= int(min_points):
                latest_offsets = np.percentile(
                    finite_residuals,
                    GAUSSIAN_SIGMA_PERCENTILES,
                )

        if latest_offsets is not None:
            offsets[:, index] = latest_offsets

    return offsets


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


def resolve_portfolio_scenario_log_offset(settings):
    percentile_offsets = settings.residual_percentile_offsets_log
    if percentile_offsets is not None:
        scenario_levels = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=float)
        scenario_offsets = np.array(
            [
                percentile_offsets[0],
                percentile_offsets[1],
                0.0,
                percentile_offsets[2],
                percentile_offsets[3],
            ],
            dtype=float,
        )
        log_offset = float(
            np.interp(
                float(settings.sigma_level),
                scenario_levels,
                scenario_offsets,
            )
        )
    else:
        log_offset = float(settings.sigma_level) * float(settings.residual_sigma_log)

    return log_offset if np.isfinite(log_offset) else 0.0


def interpolate_sigma_offset_from_level(sigma_level, percentile_offsets):
    """Log-offset of a sigma level, interpolated between the stored percentiles."""
    return float(
        np.interp(
            float(sigma_level),
            (-2.0, -1.0, 0.0, 1.0, 2.0),
            (
                percentile_offsets[0],
                percentile_offsets[1],
                0.0,
                percentile_offsets[2],
                percentile_offsets[3],
            ),
        )
    )


def interpolate_sigma_level_from_log_offset(log_offset, percentile_offsets):
    offsets = np.array(
        [
            percentile_offsets[0],
            percentile_offsets[1],
            0.0,
            percentile_offsets[2],
            percentile_offsets[3],
        ],
        dtype=float,
    )
    levels = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=float)
    if not np.isfinite(log_offset) or not np.all(np.isfinite(offsets)):
        return 0.0

    sort_order = np.argsort(offsets)
    offsets = offsets[sort_order]
    levels = levels[sort_order]
    unique_offsets, unique_indices = np.unique(offsets, return_index=True)
    offsets = unique_offsets
    levels = levels[unique_indices]
    if offsets.size < 2:
        return 0.0

    log_offset = float(log_offset)
    if log_offset <= offsets[0]:
        x0, x1 = offsets[0], offsets[1]
        y0, y1 = levels[0], levels[1]
    elif log_offset >= offsets[-1]:
        x0, x1 = offsets[-2], offsets[-1]
        y0, y1 = levels[-2], levels[-1]
    else:
        return float(np.interp(log_offset, offsets, levels))

    if np.isclose(x0, x1):
        return float(y0)
    return float(y0 + ((log_offset - x0) / (x1 - x0)) * (y1 - y0))


def rate_withdrawal_attractiveness(sigma_level):
    sigma_level = float(sigma_level)
    if not np.isfinite(sigma_level):
        sigma_level = 0.0
    if sigma_level < -1.0:
        return (
            "Not attractive",
            "#ef4444",
            "Below -1σ; prefer cash buffer over selling BTC if possible.",
        )
    if sigma_level < 0.0:
        return (
            "Cautious",
            "#f97316",
            "Below fair value; withdraw only if needed.",
        )
    if sigma_level < 1.0:
        return (
            "Attractive",
            "#f0b90b",
            "Above fair value; growth withdrawals are more reasonable.",
        )
    return (
        "Very attractive",
        "#0ecb81",
        "Above +1σ; historically expensive zone for taking profits.",
    )


def estimate_current_monthly_pension(
    *,
    current_price,
    current_model_log,
    current_date,
    current_gen_date,
    intercept_a,
    slope_b,
    btc_amount,
    sell_mom_change_pct,
    percentile_offsets,
    floor_sigma_level=-2.0,
):
    current_price = float(current_price)
    current_model_log = float(current_model_log)
    if not np.isfinite(current_price) or current_price <= 0.0 or not np.isfinite(current_model_log):
        current_price = 0.0
        current_log_offset = 0.0
    else:
        current_log_offset = float(np.log10(current_price) - current_model_log)

    current_sigma_level = interpolate_sigma_level_from_log_offset(
        current_log_offset,
        percentile_offsets,
    )
    rating, rating_color, rating_note = rate_withdrawal_attractiveness(current_sigma_level)
    current_date = pd.Timestamp(current_date)
    if current_date.tzinfo is not None:
        current_date = current_date.tz_localize(None)
    # One average month ahead rather than one calendar month, so the estimate does not
    # dip whenever the next month is February.
    next_month_date = current_date + pd.Timedelta(days=AVERAGE_MONTH_DAYS)
    current_days = float((current_date - pd.Timestamp(current_gen_date)).days)
    next_month_days = max(1.0, current_days + AVERAGE_MONTH_DAYS)
    next_month_fair_price, _, _ = evaluate_powerlaw_values(
        np.array([np.log10(next_month_days)]),
        intercept_a,
        slope_b,
    )
    scenario_levels = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=float)
    scenario_offsets = np.array(
        [
            percentile_offsets[0],
            percentile_offsets[1],
            0.0,
            percentile_offsets[2],
            percentile_offsets[3],
        ],
        dtype=float,
    )
    current_floor_offset = float(
        np.interp(float(floor_sigma_level), scenario_levels, scenario_offsets)
    )
    current_floor_price = float(np.power(10.0, current_model_log + current_floor_offset))
    next_month_floor_price = float(next_month_fair_price[0] * np.power(10.0, current_floor_offset))
    next_month_price = float(next_month_fair_price[0] * np.power(10.0, current_log_offset))
    floor_monthly_growth_per_btc = max(0.0, next_month_floor_price - current_floor_price)
    monthly_growth_per_btc = max(0.0, next_month_price - current_price)
    minimum_monthly_withdrawal = floor_monthly_growth_per_btc * max(float(btc_amount), 0.0)
    max_monthly_withdrawal = monthly_growth_per_btc * max(float(btc_amount), 0.0)
    minimum_btc_to_sell = (
        minimum_monthly_withdrawal / next_month_floor_price if next_month_floor_price > 0.0 else 0.0
    )
    minimum_btc_to_sell_today = (
        minimum_monthly_withdrawal / current_price if current_price > 0.0 else 0.0
    )
    minimum_btc_sell_reduction_pct = (
        max(0.0, (1.0 - (minimum_btc_to_sell_today / minimum_btc_to_sell)) * 100.0)
        if minimum_btc_to_sell > 0.0
        else 0.0
    )
    minimum_btc_sell_today_delta_pct = (
        ((current_price / current_floor_price) - 1.0) * 100.0 if current_floor_price > 0.0 else 0.0
    )
    model_btc_to_sell = max_monthly_withdrawal / next_month_price if next_month_price > 0.0 else 0.0
    sell_ratio = min(max(float(sell_mom_change_pct) / 100.0, 0.0), 1.0)

    return CurrentMonthlyPensionEstimate(
        current_sigma_level=current_sigma_level,
        withdrawal_rating=rating,
        withdrawal_rating_color=rating_color,
        withdrawal_rating_note=rating_note,
        current_price=current_price,
        current_floor_price=current_floor_price,
        next_month_floor_price=next_month_floor_price,
        next_month_price=next_month_price,
        floor_monthly_growth_per_btc=floor_monthly_growth_per_btc,
        monthly_growth_per_btc=monthly_growth_per_btc,
        minimum_monthly_withdrawal=minimum_monthly_withdrawal,
        max_monthly_withdrawal=max_monthly_withdrawal,
        minimum_btc_to_sell=minimum_btc_to_sell,
        minimum_btc_to_sell_today=minimum_btc_to_sell_today,
        minimum_btc_sell_reduction_pct=minimum_btc_sell_reduction_pct,
        minimum_btc_sell_today_delta_pct=minimum_btc_sell_today_delta_pct,
        model_btc_to_sell=model_btc_to_sell,
        selected_monthly_withdrawal=max_monthly_withdrawal * sell_ratio,
        next_month_date=next_month_date,
    )


def calculate_monthly_buy_portfolio_values(
    date_index,
    current_gen_date,
    fair_prices,
    intercept_a,
    slope_b,
    initial_btc_amount,
    monthly_buy_amount,
    purchase_anchor_day,
    log_price_offset=0.0,
    monthly_mom_change_pct=0.0,
):
    projection_dates = pd.to_datetime(date_index)
    fair_price_arr = np.asarray(fair_prices, dtype=float)
    total_btc = np.full(fair_price_arr.shape, float(initial_btc_amount), dtype=float)
    invested_capital = np.zeros(fair_price_arr.shape, dtype=float)
    monthly_cash_flow = float(monthly_buy_amount)
    if not np.isfinite(monthly_cash_flow):
        monthly_cash_flow = 0.0
    monthly_mom_change_ratio = float(monthly_mom_change_pct) / 100.0
    if not np.isfinite(monthly_mom_change_ratio):
        monthly_mom_change_ratio = 0.0
    monthly_mom_change_ratio = min(max(monthly_mom_change_ratio, 0.0), 1.0)

    if (
        projection_dates.empty
        or (monthly_cash_flow == 0.0 and monthly_mom_change_ratio == 0.0)
        or fair_price_arr.size == 0
        or not np.any(np.isfinite(fair_price_arr) & (fair_price_arr > 0.0))
    ):
        return total_btc, fair_price_arr * total_btc, invested_capital

    anchor_date = pd.Timestamp(purchase_anchor_day).normalize()
    current_month_start = anchor_date.to_period("M").to_timestamp()
    cash_flow_start = current_month_start
    if cash_flow_start < anchor_date:
        cash_flow_start += pd.offsets.MonthBegin(1)
    purchase_start = cash_flow_start

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
    price_multiplier = np.power(10.0, float(log_price_offset))
    purchase_prices = purchase_prices * price_multiplier
    previous_purchase_dates = purchase_dates - pd.offsets.MonthBegin(1)
    previous_purchase_days = np.maximum(
        (previous_purchase_dates - current_gen_date).days.astype(float),
        1.0,
    )
    purchase_elapsed_days = (purchase_dates - previous_purchase_dates).days.astype(float)
    previous_purchase_prices, _, _ = evaluate_powerlaw_values(
        np.log10(previous_purchase_days),
        intercept_a,
        slope_b,
    )
    previous_purchase_prices = previous_purchase_prices * price_multiplier
    valid_purchase_mask = np.isfinite(purchase_prices) & (purchase_prices > 0.0)
    if not np.any(valid_purchase_mask):
        return total_btc, fair_price_arr * total_btc, invested_capital

    valid_purchase_dates = purchase_dates[valid_purchase_mask]
    valid_purchase_prices = purchase_prices[valid_purchase_mask]
    valid_previous_purchase_prices = previous_purchase_prices[valid_purchase_mask]
    # Growth per average month, so the withdrawal does not shrink in February and swell
    # in 31-day months. Over a year it still sums to about the calendar growth.
    valid_average_month_growth_pct = normalize_periodic_growth_rate(
        valid_purchase_prices,
        valid_previous_purchase_prices,
        purchase_elapsed_days[valid_purchase_mask],
        AVERAGE_MONTH_DAYS,
    )
    purchased_btc = np.zeros(valid_purchase_dates.shape, dtype=float)
    realized_cash_flow = np.zeros(valid_purchase_dates.shape, dtype=float)
    running_btc = float(initial_btc_amount)
    for index, purchase_price in enumerate(valid_purchase_prices):
        mom_change_cash_flow = 0.0
        scheduled_cash_flow = (
            monthly_cash_flow
            if pd.Timestamp(valid_purchase_dates[index]) >= cash_flow_start
            else 0.0
        )
        previous_purchase_price = float(valid_previous_purchase_prices[index])
        average_month_growth_pct = float(valid_average_month_growth_pct[index])
        if np.isfinite(average_month_growth_pct):
            current_position_mom_change = (
                previous_purchase_price * average_month_growth_pct / 100.0 * float(running_btc)
            )
            mom_change_cash_flow = -(
                max(current_position_mom_change, 0.0) * monthly_mom_change_ratio
            )

        cash_flow = scheduled_cash_flow + mom_change_cash_flow
        if cash_flow < 0.0:
            sell_btc = min(running_btc, abs(cash_flow) / float(purchase_price))
            purchased_btc[index] = -sell_btc
            realized_cash_flow[index] = -(sell_btc * float(purchase_price))
        else:
            purchased_btc[index] = cash_flow / float(purchase_price)
            realized_cash_flow[index] = cash_flow
        running_btc += purchased_btc[index]

    cumulative_btc = np.cumsum(purchased_btc)
    cumulative_invested_capital = np.cumsum(realized_cash_flow)
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
    anchor_day = resolve_projection_anchor_day(df_index, today=anchor_day)
    history_periods = min(
        max(int(settings.history_periods), PORTFOLIO_HISTORY_PERIODS_MIN),
        PORTFOLIO_HISTORY_PERIODS_MAX,
    )
    # Row 0 is the pre-display row the view model drops; the anchor follows the history.
    projection_lookback = history_periods + 1

    if settings.forecast_unit == "Year":
        latest_year = int(anchor_day.year)
        start_period = pd.Timestamp(f"{latest_year - projection_lookback}-01-01")
        date_index = pd.date_range(
            start=start_period,
            periods=settings.forecast_horizon + projection_lookback,
            freq="YS",
        )
        change_usd_col, change_pct_col = "YoY_USD", "YoY_pct"
        table_title = "Yearly growth table"
    elif settings.forecast_unit == "Day":
        latest_day = anchor_day
        start_period = latest_day - pd.Timedelta(days=projection_lookback)
        date_index = pd.date_range(
            start=start_period,
            periods=settings.forecast_horizon + projection_lookback,
            freq="D",
        )
        change_usd_col, change_pct_col = "DoD_USD", "DoD_pct"
        table_title = "Daily growth table"
    else:
        latest_month_start = anchor_day.to_period("M").to_timestamp()
        start_period = latest_month_start - pd.offsets.MonthBegin(projection_lookback)
        date_index = pd.date_range(
            start=start_period,
            periods=settings.forecast_horizon + projection_lookback,
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
    log_price_offset = resolve_portfolio_scenario_log_offset(settings)
    price_multiplier = np.power(10.0, log_price_offset)
    period_fair_price = period_fair_price * price_multiplier
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
            log_price_offset=log_price_offset,
            monthly_mom_change_pct=settings.monthly_mom_change_pct,
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
            AVERAGE_MONTH_DAYS,
        )
        # Calendar months run 28-31 days, so a raw diff zigzags month to month. Report the
        # money change over the same average month the percentage is normalised to.
        normalized_pct = portfolio_df[change_pct_col].to_numpy(dtype=float)
        portfolio_df[change_usd_col] = np.where(
            np.isfinite(normalized_pct),
            previous_values * normalized_pct / 100.0,
            portfolio_df[change_usd_col].to_numpy(dtype=float),
        )
    else:
        portfolio_df[change_pct_col] = portfolio_df["PortfolioUSD"].pct_change() * 100

    # Market growth of the BTC still held, so buys and sells shrink or grow the change
    # instead of it tracking the untouched initial holding.
    previous_dca_values = portfolio_df["DcaPortfolioUSD"].shift(1).to_numpy(dtype=float)
    if settings.forecast_unit == "Month":
        price_pct = normalize_periodic_growth_rate(
            portfolio_df["FairPriceUSD"].to_numpy(dtype=float),
            portfolio_df["FairPriceUSD"].shift(1).to_numpy(dtype=float),
            portfolio_df["Date"].diff().dt.days.to_numpy(dtype=float),
            AVERAGE_MONTH_DAYS,
        )
        dca_change = previous_dca_values * price_pct / 100.0
    else:
        dca_change = (
            portfolio_df["DcaPortfolioUSD"].diff() - portfolio_df["DcaInvestedCapitalUSD"].diff()
        ).to_numpy(dtype=float)
    portfolio_df["DcaChangeUSD"] = dca_change
    with np.errstate(divide="ignore", invalid="ignore"):
        portfolio_df["DcaChangePct"] = np.where(
            previous_dca_values > 0.0,
            dca_change / previous_dca_values * 100.0,
            np.nan,
        )

    return PortfolioProjectionResult(
        portfolio_df=portfolio_df,
        table_title=table_title,
        forecast_unit=settings.forecast_unit,
        change_usd_col=change_usd_col,
        change_pct_col=change_pct_col,
        history_periods=history_periods,
    )


def get_growth_change_labels(forecast_unit, currency_unit):
    prefix = "YoY" if forecast_unit == "Year" else ("DoD" if forecast_unit == "Day" else "MoM")
    return f"{prefix} Change ({currency_unit})", f"{prefix} Change (%)"


def get_period_cash_flow_label(forecast_unit, currency_unit, is_buying=False):
    prefix = (
        "Yearly" if forecast_unit == "Year" else ("Daily" if forecast_unit == "Day" else "Monthly")
    )
    return f"{prefix} {'buy' if is_buying else 'withdrawal'} ({currency_unit})"


def build_portfolio_view_model(
    projection_result,
    monthly_buy_amount,
    currency_unit,
    monthly_mom_change_pct=0.0,
):
    projection_df = projection_result.portfolio_df.copy()
    projection_df["DcaPeriodCashFlowUSD"] = (
        projection_df["DcaInvestedCapitalUSD"].diff().fillna(projection_df["DcaInvestedCapitalUSD"])
    )
    portfolio_display_df = projection_df.iloc[1:].copy()
    portfolio_display_df["FairPriceDisplay"] = portfolio_display_df["FairPriceUSD"]
    portfolio_display_df["PortfolioDisplay"] = portfolio_display_df["PortfolioUSD"]
    portfolio_display_df["DcaPortfolioDisplay"] = portfolio_display_df["DcaPortfolioUSD"]
    portfolio_display_df["DcaBTCDisplay"] = portfolio_display_df["DcaBTC"]
    portfolio_display_df["DcaInvestedCapitalDisplay"] = portfolio_display_df[
        "DcaInvestedCapitalUSD"
    ]
    # Buying and selling are exclusive, so the period flow column shows one direction.
    dca_is_buying = monthly_buy_amount > 0.0 and monthly_mom_change_pct == 0.0
    period_flow_sign = 1.0 if dca_is_buying else -1.0
    portfolio_display_df["DcaPeriodCashFlowDisplay"] = np.maximum(
        period_flow_sign * portfolio_display_df["DcaPeriodCashFlowUSD"].to_numpy(dtype=float),
        0.0,
    )
    portfolio_display_df["ChangeDisplay"] = portfolio_display_df[projection_result.change_usd_col]
    dca_enabled = monthly_buy_amount != 0.0 or monthly_mom_change_pct != 0.0
    change_pct_source_col = projection_result.change_pct_col
    if dca_enabled and "DcaChangeUSD" in portfolio_display_df.columns:
        portfolio_display_df["ChangeDisplay"] = portfolio_display_df["DcaChangeUSD"]
        change_pct_source_col = "DcaChangePct"

    # Growth is reported from the anchor period (today), not from the historical
    # context rows that build_portfolio_projection prepends ahead of it. Row 0 is the
    # pre-display row, so the anchor sits one row past the history block.
    anchor_position = min(
        max(int(projection_result.history_periods), 0) + 1,
        len(projection_df) - 1,
    )
    baseline_value = projection_df["PortfolioUSD"].iloc[anchor_position]
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
    period_cash_flow_label = get_period_cash_flow_label(
        projection_result.forecast_unit,
        currency_unit,
        is_buying=dca_is_buying,
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
            "PortfolioDisplay": f"Portfolio if not selling ({currency_unit})",
            "DcaPortfolioDisplay": f"Remaining BTC value ({currency_unit})",
            "DcaPeriodCashFlowDisplay": period_cash_flow_label,
            "DcaInvestedCapitalDisplay": f"Net cash flow ({currency_unit})",
            "DcaBTCDisplay": "BTC after monthly cash flow",
            "ChangeDisplay": period_change_usd_label,
            change_pct_source_col: period_change_pct_label,
        }
    )
    display_columns = [
        "Date",
        f"Fair Price ({currency_unit})",
        f"Portfolio if not selling ({currency_unit})",
    ]
    if dca_enabled:
        display_columns.extend(
            [
                f"Remaining BTC value ({currency_unit})",
                period_cash_flow_label,
                f"Net cash flow ({currency_unit})",
                "BTC after monthly cash flow",
            ]
        )
    display_columns.extend([period_change_usd_label, period_change_pct_label])

    return PortfolioViewModel(
        portfolio_display_df=portfolio_display_df,
        table_df=table_df[display_columns],
        table_title=projection_result.table_title,
        dca_enabled=dca_enabled,
        dca_is_buying=dca_is_buying,
        baseline_value=baseline_value,
        last_value=last_value,
        last_dca_value=last_dca_value,
        last_dca_invested_capital=last_dca_invested_capital,
        total_growth_pct=total_growth_pct,
        period_change_usd_label=period_change_usd_label,
        period_change_pct_label=period_change_pct_label,
        period_cash_flow_label=period_cash_flow_label,
    )


def resolve_backtest_monthly_prices(price_display_df, years):
    """Backtest window as one row per month, dated on that month's last observation.

    Each row is the day the strategy actually trades, so callers can fit and evaluate the
    model on the same date as the price they transact at. Labelling these rows with the
    month start instead would date the model a month before the trade it sizes.
    """
    if price_display_df is None or price_display_df.empty:
        return None

    price_series = pd.to_numeric(price_display_df["CloseDisplay"], errors="coerce").dropna()
    price_series = price_series[price_series > 0.0].sort_index()
    if price_series.empty:
        return None

    latest_date = pd.Timestamp(price_series.index.max())
    start_date = latest_date - pd.DateOffset(years=int(years))
    windowed_series = price_series[price_series.index >= start_date]
    if windowed_series.empty:
        return None

    month_periods = pd.DatetimeIndex(windowed_series.index).to_period("M")
    is_last_of_month = np.append(month_periods[:-1] != month_periods[1:], True)
    monthly_prices = windowed_series[is_last_of_month]
    if monthly_prices.size < 2:
        return None
    return monthly_prices


def build_portfolio_real_data_backtest(
    price_display_df,
    settings,
    currency_unit,
    years=5,
    *,
    floor_prices=None,
    sell_mom_change_pct=None,
    strategy_name=None,
    initial_capital=None,
    floor_model_label=None,
):
    """Replay a withdrawal strategy over real prices.

    ``floor_prices`` sizes the withdrawals and must already be walk-forward; build it
    with ``build_causal_powerlaw_floor_prices``. Without it the strategy falls back to
    withdrawing from realised month-on-month price growth, which is causal by nature.
    """
    monthly_prices = resolve_backtest_monthly_prices(price_display_df, years)
    if monthly_prices is None:
        return None

    if initial_capital is None:
        initial_btc = max(float(settings.btc_amount), 0.0)
    else:
        initial_capital = max(float(initial_capital), 0.0)
        first_price = float(monthly_prices.iloc[0])
        initial_btc = initial_capital / first_price if first_price > 0.0 else 0.0
    strategy_btc = initial_btc
    net_cash_flow = 0.0
    sell_pct = (
        float(settings.monthly_mom_change_pct)
        if sell_mom_change_pct is None
        else float(sell_mom_change_pct)
    )
    sell_ratio = min(max(sell_pct / 100.0, 0.0), 10.0)
    monthly_cash_flow = float(settings.monthly_buy_amount)
    if not np.isfinite(monthly_cash_flow):
        monthly_cash_flow = 0.0

    if floor_prices is not None:
        floor_prices = pd.Series(floor_prices, dtype=float).reindex(monthly_prices.index)
        if not np.any(np.isfinite(floor_prices.to_numpy(dtype=float))):
            floor_prices = None

    rows = []
    previous_actual_price = None
    previous_floor_price = None
    months_without_floor = 0
    for date, price in monthly_prices.items():
        price = float(price)
        hold_value = initial_btc * price
        if floor_prices is None:
            positive_price_growth = (
                max(price - previous_actual_price, 0.0)
                if previous_actual_price is not None
                else 0.0
            )
        else:
            floor_price = float(floor_prices.loc[date])
            if np.isfinite(floor_price):
                positive_price_growth = (
                    max(floor_price - previous_floor_price, 0.0)
                    if previous_floor_price is not None
                    else 0.0
                )
                previous_floor_price = floor_price
            else:
                # Too little history behind this month to fit a floor on it yet.
                months_without_floor += 1
                positive_price_growth = 0.0
        withdrawal = positive_price_growth * strategy_btc * sell_ratio
        period_cash_flow = monthly_cash_flow - withdrawal
        if period_cash_flow >= 0.0:
            strategy_btc += period_cash_flow / price
        else:
            sell_btc = min(strategy_btc, abs(period_cash_flow) / price)
            period_cash_flow = -(sell_btc * price)
            strategy_btc -= sell_btc
        net_cash_flow += period_cash_flow
        strategy_value = strategy_btc * price

        rows.append(
            {
                "Date": pd.Timestamp(date),
                "ActualPrice": price,
                "HoldValue": hold_value,
                "StrategyValue": strategy_value,
                "MonthlyBuy": max(period_cash_flow, 0.0),
                "MonthlyWithdrawal": max(-period_cash_flow, 0.0),
                "NetCashFlow": net_cash_flow,
                "StrategyBTC": strategy_btc,
            }
        )
        previous_actual_price = price

    backtest_df = pd.DataFrame(rows)
    start_value = float(backtest_df["HoldValue"].iloc[0])
    hold_last_value = float(backtest_df["HoldValue"].iloc[-1])
    strategy_last_value = float(backtest_df["StrategyValue"].iloc[-1])
    total_return_pct = (
        ((hold_last_value - start_value) / start_value) * 100.0 if start_value else 0.0
    )
    strategy_basis = start_value + max(net_cash_flow, 0.0)
    strategy_return_pct = (
        ((strategy_last_value - strategy_basis) / strategy_basis) * 100.0
        if strategy_basis > 0.0
        else 0.0
    )

    monthly_withdrawal_label = (
        f"{floor_model_label} monthly withdrawal ({currency_unit})"
        if floor_model_label
        else f"-2σ monthly withdrawal ({currency_unit})"
    )
    if floor_prices is None:
        monthly_withdrawal_label = f"Historical monthly withdrawal ({currency_unit})"
    table_df = backtest_df.copy()
    table_df["Date"] = table_df["Date"].dt.strftime("%Y-%m")
    table_df = table_df.rename(
        columns={
            "ActualPrice": f"Actual BTC price ({currency_unit})",
            "HoldValue": f"Hold-only value ({currency_unit})",
            "StrategyValue": f"Strategy value ({currency_unit})",
            "MonthlyBuy": f"Monthly buy ({currency_unit})",
            "MonthlyWithdrawal": monthly_withdrawal_label,
            "NetCashFlow": f"Net cash flow ({currency_unit})",
            "StrategyBTC": "BTC after strategy",
        }
    )

    return PortfolioBacktestResult(
        strategy_name=strategy_name or f"Sell {sell_pct:.0f}% of monthly growth",
        sell_mom_change_pct=float(sell_pct),
        backtest_df=backtest_df,
        table_df=table_df[
            [
                "Date",
                f"Actual BTC price ({currency_unit})",
                f"Hold-only value ({currency_unit})",
                f"Strategy value ({currency_unit})",
                f"Monthly buy ({currency_unit})",
                monthly_withdrawal_label,
                f"Net cash flow ({currency_unit})",
                "BTC after strategy",
            ]
        ],
        start_value=start_value,
        hold_last_value=hold_last_value,
        strategy_last_value=strategy_last_value,
        net_cash_flow=float(net_cash_flow),
        strategy_btc=float(strategy_btc),
        total_return_pct=float(total_return_pct),
        strategy_return_pct=float(strategy_return_pct),
        monthly_withdrawal_label=monthly_withdrawal_label,
        months_without_floor=int(months_without_floor),
    )
