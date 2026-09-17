import numpy as np
import pandas as pd
import streamlit as st

from core.constants import (
    DEFAULT_A,
    DEFAULT_B,
    FLOOR_MODEL_SIGMA,
    FLOOR_MODEL_TROUGH_ENVELOPE,
    KEY_A,
    KEY_B,
    KEY_GENESIS_OFFSET,
    POWERLAW_EXPONENT_MAX,
    POWERLAW_EXPONENT_MIN,
)
from core.utils import (
    calculate_expanding_powerlaw_parameters,
    calculate_historical_sigma_offsets,
    calculate_r2_score,
    interpolate_sigma_offset_from_level,
)

# --- MATH CORE ---


def fit_powerlaw_regression(absolute_days, log_prices, genesis_offset_days):
    """Calculates the optimal Slope(B) and Intercept(A) for a given offset."""
    days_since_offset = absolute_days - genesis_offset_days
    positive_mask = days_since_offset > 0

    if np.sum(positive_mask) < 100:
        return 0.0, 0.0, 0.0

    valid_days = days_since_offset[positive_mask]
    valid_log_prices = log_prices[positive_mask]
    log_days = np.log10(valid_days)

    slope_b, intercept_a = np.polyfit(log_days, valid_log_prices, 1)
    predicted_log_prices = slope_b * log_days + intercept_a
    r2_score = calculate_r2_score(valid_log_prices, predicted_log_prices)

    return slope_b, intercept_a, r2_score


def _select_residual_episode_indices(
    valid_days,
    residual_values,
    threshold,
    direction,
    min_duration_days,
):
    if direction == "above":
        extreme_mask = residual_values >= float(threshold)
        choose_index = np.argmax
    else:
        extreme_mask = residual_values <= float(threshold)
        choose_index = np.argmin

    if float(min_duration_days) <= 1.0:
        return list(np.flatnonzero(extreme_mask))

    selected_indices = []
    episode_start = None
    previous_idx = None
    for idx, is_extreme in enumerate(extreme_mask):
        if is_extreme:
            if episode_start is None:
                episode_start = idx
            previous_idx = idx
            continue

        if episode_start is not None and previous_idx is not None:
            episode_days = valid_days[previous_idx] - valid_days[episode_start] + 1.0
            if episode_days >= float(min_duration_days):
                episode_slice = slice(episode_start, previous_idx + 1)
                local_values = residual_values[episode_slice]
                selected_indices.append(episode_start + int(choose_index(local_values)))
        episode_start = None
        previous_idx = None

    if episode_start is not None and previous_idx is not None:
        episode_days = valid_days[previous_idx] - valid_days[episode_start] + 1.0
        if episode_days >= float(min_duration_days):
            episode_slice = slice(episode_start, previous_idx + 1)
            local_values = residual_values[episode_slice]
            selected_indices.append(episode_start + int(choose_index(local_values)))

    return selected_indices


def _select_local_extreme_indices(valid_days, values, window_days, direction):
    """Select one strict local extreme inside each centered time window."""
    valid_days = np.asarray(valid_days, dtype=float)
    values = np.asarray(values, dtype=float)
    half_window_days = max(float(window_days) / 2.0, 0.0)
    if half_window_days <= 0.0:
        return []

    selected_indices = []
    days_are_sorted = bool(np.all(valid_days[:-1] <= valid_days[1:]))
    for index, day in enumerate(valid_days):
        if days_are_sorted:
            neighborhood = np.arange(
                np.searchsorted(valid_days, day - half_window_days, side="left"),
                np.searchsorted(valid_days, day + half_window_days, side="right"),
            )
        else:
            neighborhood = np.flatnonzero(np.abs(valid_days - day) <= half_window_days)
        if neighborhood.size == 0:
            continue
        neighborhood_values = values[neighborhood]
        local_position = (
            int(np.argmax(neighborhood_values))
            if direction == "above"
            else int(np.argmin(neighborhood_values))
        )
        if int(neighborhood[local_position]) == index:
            selected_indices.append(index)
    return selected_indices


def fit_peak_powerlaw_envelope(
    absolute_days,
    log_prices,
    genesis_offset_days,
    model_days,
    min_peak_count=2,
    sigma_threshold=1.0,
    min_duration_days=1.0,
    residuals=None,
    threshold_offset=None,
    window_days=None,
):
    days_since_offset = np.asarray(absolute_days, dtype=float) - float(genesis_offset_days)
    log_prices = np.asarray(log_prices, dtype=float)
    valid_mask = (days_since_offset > 0.0) & np.isfinite(log_prices)
    if np.sum(valid_mask) < int(min_peak_count):
        return None

    valid_days = days_since_offset[valid_mask]
    valid_logs = log_prices[valid_mask]
    if residuals is None:
        baseline_slope, baseline_intercept = np.polyfit(np.log10(valid_days), valid_logs, 1)
        residual_values = valid_logs - (baseline_intercept + baseline_slope * np.log10(valid_days))
    else:
        residual_values = np.asarray(residuals, dtype=float)[valid_mask]
    explicit_threshold = threshold_offset is not None
    if threshold_offset is None and window_days is None:
        residual_sigma = float(np.std(residual_values))
        if not np.isfinite(residual_sigma) or residual_sigma <= 0.0:
            return None
        threshold_offset = float(sigma_threshold) * residual_sigma
    if window_days is not None:
        peak_indices = _select_local_extreme_indices(
            valid_days,
            valid_logs,
            window_days,
            direction="above",
        )
        if explicit_threshold:
            peak_indices = [
                index for index in peak_indices if residual_values[index] >= float(threshold_offset)
            ]
    else:
        peak_indices = _select_residual_episode_indices(
            valid_days,
            residual_values,
            threshold=threshold_offset,
            direction="above",
            min_duration_days=min_duration_days,
        )

    if len(peak_indices) < int(min_peak_count):
        return None

    peak_days = valid_days[peak_indices]
    peak_logs = valid_logs[peak_indices]
    slope_b, intercept_a = np.polyfit(np.log10(peak_days), peak_logs, 1)
    model_days = np.asarray(model_days, dtype=float)
    model_log = intercept_a + slope_b * np.log10(np.maximum(model_days, 1.0))
    model_values = np.power(10.0, model_log)
    return {
        "intercept": float(intercept_a),
        "slope": float(slope_b),
        "peak_days": peak_days,
        "peak_values": np.power(10.0, peak_logs),
        "model_values": model_values,
    }


def fit_trough_powerlaw_envelope(
    absolute_days,
    log_prices,
    genesis_offset_days,
    model_days,
    min_trough_count=2,
    sigma_threshold=-1.0,
    min_duration_days=1.0,
    residuals=None,
    threshold_offset=None,
    window_days=None,
):
    days_since_offset = np.asarray(absolute_days, dtype=float) - float(genesis_offset_days)
    log_prices = np.asarray(log_prices, dtype=float)
    valid_mask = (days_since_offset > 0.0) & np.isfinite(log_prices)
    if np.sum(valid_mask) < int(min_trough_count):
        return None

    valid_days = days_since_offset[valid_mask]
    valid_logs = log_prices[valid_mask]
    if residuals is None:
        baseline_slope, baseline_intercept = np.polyfit(np.log10(valid_days), valid_logs, 1)
        residual_values = valid_logs - (baseline_intercept + baseline_slope * np.log10(valid_days))
    else:
        residual_values = np.asarray(residuals, dtype=float)[valid_mask]
    explicit_threshold = threshold_offset is not None
    if threshold_offset is None and window_days is None:
        residual_sigma = float(np.std(residual_values))
        if not np.isfinite(residual_sigma) or residual_sigma <= 0.0:
            return None
        threshold_offset = float(sigma_threshold) * residual_sigma
    if window_days is not None:
        trough_indices = _select_local_extreme_indices(
            valid_days,
            valid_logs,
            window_days,
            direction="below",
        )
        if explicit_threshold:
            trough_indices = [
                index
                for index in trough_indices
                if residual_values[index] <= float(threshold_offset)
            ]
    else:
        trough_indices = _select_residual_episode_indices(
            valid_days,
            residual_values,
            threshold=threshold_offset,
            direction="below",
            min_duration_days=min_duration_days,
        )

    if len(trough_indices) < int(min_trough_count):
        return None

    trough_days = valid_days[trough_indices]
    trough_logs = valid_logs[trough_indices]
    slope_b, intercept_a = np.polyfit(np.log10(trough_days), trough_logs, 1)
    model_days = np.asarray(model_days, dtype=float)
    model_log = intercept_a + slope_b * np.log10(np.maximum(model_days, 1.0))
    model_values = np.power(10.0, model_log)
    return {
        "intercept": float(intercept_a),
        "slope": float(slope_b),
        "trough_days": trough_days,
        "trough_values": np.power(10.0, trough_logs),
        "model_values": model_values,
    }


def calculate_r2_for_manual_params(
    absolute_days, log_prices, genesis_offset_days, intercept_a, slope_b
):
    """Calculates R2 for specific manual A and B values."""
    days_since_offset = absolute_days - genesis_offset_days
    positive_mask = days_since_offset > 0

    if np.sum(positive_mask) < 100:
        return 0.0

    valid_days = days_since_offset[positive_mask]
    valid_log_prices = log_prices[positive_mask]
    log_days = np.log10(valid_days)
    predicted_log_prices = intercept_a + slope_b * log_days

    return calculate_r2_score(valid_log_prices, predicted_log_prices)


def calculate_r2_for_manual_params_on_rolling_mean(
    absolute_days,
    values,
    genesis_offset_days,
    intercept_a,
    slope_b,
    window_days=90,
):
    """Calculates R2 for manual A/B against a rolling mean of raw positive values."""
    absolute_days = np.asarray(absolute_days, dtype=float)
    values = np.asarray(values, dtype=float)
    days_since_offset = absolute_days - genesis_offset_days
    positive_mask = (days_since_offset > 0) & np.isfinite(values) & (values > 0)

    if np.sum(positive_mask) < int(window_days):
        return 0.0

    valid_days = days_since_offset[positive_mask]
    valid_values = values[positive_mask]
    rolling_kernel = np.ones(int(window_days), dtype=float) / float(window_days)
    rolling_values = np.convolve(valid_values, rolling_kernel, mode="valid")
    rolling_days = valid_days[int(window_days) - 1 :]
    rolling_mask = np.isfinite(rolling_values) & (rolling_values > 0)

    if np.sum(rolling_mask) < 100:
        return 0.0

    valid_log_values = np.log10(rolling_values[rolling_mask])
    log_days = np.log10(rolling_days[rolling_mask])
    predicted_log_values = intercept_a + slope_b * log_days

    return calculate_r2_score(valid_log_values, predicted_log_values)


def build_causal_powerlaw_floor_prices(
    daily_price_series,
    monthly_index,
    current_gen_date,
    *,
    floor_model=FLOOR_MODEL_SIGMA,
    sigma_level=-2.0,
    envelope_sigma_threshold=1.0,
    envelope_window_days=365.25,
    min_points=100,
    sigma_recalculation_step=7,
):
    """Refit the withdrawal floor month by month from data known at that month.

    Sizing withdrawals from a model fitted on the whole history lets every month of a
    backtest see its own future. Here each month's floor is fitted on the prefix of the
    daily history up to that month instead, so stepping through ``monthly_index`` is
    walk-forward. Months without enough history behind them come back as NaN.
    """
    monthly_index = pd.DatetimeIndex(monthly_index)
    daily_prices = pd.to_numeric(pd.Series(daily_price_series), errors="coerce").dropna()
    daily_prices = daily_prices[daily_prices > 0.0].sort_index()
    if daily_prices.empty or monthly_index.empty:
        return pd.Series(np.nan, index=monthly_index, dtype=float)

    current_gen_date = pd.Timestamp(current_gen_date)
    daily_dates = pd.DatetimeIndex(daily_prices.index)
    daily_days = np.maximum((daily_dates - current_gen_date).days.to_numpy(dtype=float), 1.0)
    daily_log_days = np.log10(daily_days)
    daily_log_prices = np.log10(daily_prices.to_numpy(dtype=float))

    intercepts, slopes, _ = calculate_expanding_powerlaw_parameters(
        daily_log_days,
        daily_log_prices,
        min_points=min_points,
    )
    sigma_offsets = calculate_historical_sigma_offsets(
        daily_log_days,
        daily_log_prices,
        intercepts,
        slopes,
        min_points=min_points,
        recalculation_step=sigma_recalculation_step,
    )

    # Newest daily observation available on each entry's own date. Rows are dated on
    # the day the strategy trades, so the fit sees that day's close and nothing later.
    cutoff_positions = (
        np.searchsorted(
            daily_dates.to_numpy(dtype="datetime64[ns]"),
            monthly_index.to_numpy(dtype="datetime64[ns]"),
            side="right",
        )
        - 1
    )
    monthly_days = np.maximum(
        (monthly_index - current_gen_date).days.to_numpy(dtype=float),
        1.0,
    )
    monthly_log_days = np.log10(monthly_days)

    floor_values = np.full(len(monthly_index), np.nan, dtype=float)
    for position, cutoff in enumerate(cutoff_positions):
        cutoff = int(cutoff)
        if cutoff < 0 or (cutoff + 1) < int(min_points):
            continue

        intercept = intercepts[cutoff]
        slope = slopes[cutoff]
        offsets = sigma_offsets[:, cutoff]
        if not (np.isfinite(intercept) and np.isfinite(slope)) or not np.all(np.isfinite(offsets)):
            continue

        if floor_model == FLOOR_MODEL_TROUGH_ENVELOPE:
            prefix_log_days = daily_log_days[: cutoff + 1]
            prefix_log_prices = daily_log_prices[: cutoff + 1]
            envelope = fit_trough_powerlaw_envelope(
                daily_days[: cutoff + 1],
                prefix_log_prices,
                0.0,
                np.array([monthly_days[position]], dtype=float),
                residuals=prefix_log_prices - (intercept + slope * prefix_log_days),
                threshold_offset=interpolate_sigma_offset_from_level(
                    -abs(float(envelope_sigma_threshold)),
                    offsets,
                ),
                window_days=envelope_window_days,
            )
            if envelope is None:
                continue
            floor_values[position] = float(envelope["model_values"][0])
            continue

        floor_log = (
            intercept
            + slope * monthly_log_days[position]
            + interpolate_sigma_offset_from_level(sigma_level, offsets)
        )
        floor_values[position] = float(
            np.power(10.0, np.clip(floor_log, POWERLAW_EXPONENT_MIN, POWERLAW_EXPONENT_MAX))
        )

    return pd.Series(floor_values, index=monthly_index, dtype=float)


# --- SIDEBAR RENDERER ---
def render_sidebar(
    all_abs_days,
    all_log_close,
    text_color,
    r2_values=None,
    r2_rolling_window_days=None,
    r2_label="PowerLaw R²",
    render_extra_controls=None,
    render_after_actions=None,
    a_key=KEY_A,
    b_key=KEY_B,
    default_a=DEFAULT_A,
    default_b=DEFAULT_B,
    genesis_offset_days=None,
):
    # Initialize defaults if needed
    if KEY_GENESIS_OFFSET not in st.session_state:
        st.session_state[KEY_GENESIS_OFFSET] = 0

    opt_offset = 0 if genesis_offset_days is None else int(genesis_offset_days)
    opt_a = default_a
    opt_b = default_b
    if a_key not in st.session_state:
        st.session_state[a_key] = float(default_a)
    if b_key not in st.session_state:
        st.session_state[b_key] = float(default_b)
    current_offset = (
        int(genesis_offset_days)
        if genesis_offset_days is not None
        else int(st.session_state.get(KEY_GENESIS_OFFSET, opt_offset))
    )
    current_a = float(st.session_state.get(a_key, opt_a))
    current_b = float(st.session_state.get(b_key, opt_b))

    if r2_values is not None and r2_rolling_window_days is not None:
        display_r2 = calculate_r2_for_manual_params_on_rolling_mean(
            all_abs_days,
            r2_values,
            current_offset,
            current_a,
            current_b,
            window_days=int(r2_rolling_window_days),
        )
    else:
        display_r2 = calculate_r2_for_manual_params(
            all_abs_days,
            all_log_close,
            current_offset,
            current_a,
            current_b,
        )

    if callable(render_extra_controls):
        render_extra_controls()

    if callable(render_after_actions):
        render_after_actions()

    return display_r2
