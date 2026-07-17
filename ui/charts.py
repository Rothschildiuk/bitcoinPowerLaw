import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.constants import (
    MODE_POWERLAW,
    POWERLAW_SIGMA_MODE_CLASSIC,
    POWERLAW_SIGMA_MODE_HISTORICAL,
    POWERLAW_SIGMA_MODE_SEGMENTED,
    TIME_LOG,
)
from core.utils import evaluate_powerlaw_values

HALVING_DATES = [
    pd.Timestamp("2012-11-28"),
    pd.Timestamp("2016-07-09"),
    pd.Timestamp("2020-05-11"),
    pd.Timestamp("2024-04-20"),
]
TIME_AXIS_LEADING_PADDING_DAYS = 90
MODEL_FORWARD_YEARS = 10
MAX_RENDERED_MODEL_POINTS = 2500
OPTIONAL_SIGMA_LEVELS = (-1.5, -0.5, 0.5, 1.5)
SEGMENTED_SIGMA_STEP = 0.5
SEGMENTED_SIGMA_HALF_STEP = SEGMENTED_SIGMA_STEP / 2.0
SEGMENTED_SIGMA_LEVELS = tuple(
    float(value)
    for value in np.arange(
        -2.0 - SEGMENTED_SIGMA_HALF_STEP,
        2.0 + SEGMENTED_SIGMA_HALF_STEP + SEGMENTED_SIGMA_STEP,
        SEGMENTED_SIGMA_STEP,
    )
)
SEGMENTED_SIGMA_PAIR_STYLES = (
    (0.5, "#06b6d4", "solid"),
    (1.0, "#22c55e", "solid"),
    (1.5, "#f59e0b", "dash"),
    (2.0, "#f97316", "dash"),
    (np.inf, "#ef4444", "dash"),
)
SEGMENTED_SIGMA_DEFAULT_HIDDEN_LEGENDS = {
    "Segmented sigma 0σ to ±0.5σ",
    "Segmented sigma ±1σ to ±1.5σ",
    "Segmented sigma > ±2σ",
}
MOVING_AVERAGE_LINE_STYLES = (
    "#22c55e",
    "#f97316",
    "#a78bfa",
)


def _main_chart_plotly_config():
    return {
        "displayModeBar": True,
        "modeBarButtons": [
            ["toImage"],
            ["zoom2d", "pan2d", "toggleSpikelines"],
            ["zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
        ],
    }


def _resolve_time_axis_start_date(df_display, padding_days=TIME_AXIS_LEADING_PADDING_DAYS):
    first_data_date = pd.Timestamp(df_display.index.min())
    return first_data_date - pd.Timedelta(days=int(padding_days))


def _resolve_model_view_max(
    df_display,
    current_gen_date,
    forward_years=MODEL_FORWARD_YEARS,
    today=None,
):
    latest_data_date = pd.Timestamp(df_display.index.max()).normalize()
    today_date = (
        pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today).normalize()
    )
    anchor_date = max(latest_data_date, today_date)
    horizon_end_date = anchor_date + pd.DateOffset(years=int(forward_years))
    return max(1.0, float((horizon_end_date - current_gen_date).days))


def _resolve_powerlaw_y_range(
    df_display,
    m_fair_display,
    p2_5,
    p97_5,
    use_log_scale,
    model_x=None,
    visible_start_day=None,
    include_bands=True,
):
    btc_vals = pd.to_numeric(df_display["CloseDisplay"], errors="coerce").to_numpy(dtype=float)
    fair_vals = np.asarray(m_fair_display, dtype=float)
    if model_x is not None and visible_start_day is not None:
        model_x_arr = np.asarray(model_x)
        if np.issubdtype(model_x_arr.dtype, np.number):
            visible_mask = model_x_arr.astype(float) >= float(visible_start_day)
        else:
            model_x_ts = pd.to_datetime(model_x_arr)
            visible_start_ts = pd.Timestamp(visible_start_day)
            visible_mask = model_x_ts >= visible_start_ts
        if np.any(visible_mask):
            fair_vals = fair_vals[visible_mask]

    candidate_parts = [btc_vals]
    if include_bands:
        lower_band, _, _ = evaluate_powerlaw_values(np.log10(fair_vals), p2_5, 1.0)
        upper_band, _, _ = evaluate_powerlaw_values(np.log10(fair_vals), p97_5, 1.0)
        candidate_parts.extend([lower_band, upper_band])
    else:
        candidate_parts.append(fair_vals)

    candidate = np.concatenate(candidate_parts)
    candidate = candidate[np.isfinite(candidate)]
    if use_log_scale:
        candidate = candidate[candidate > 0]
    if candidate.size == 0:
        return None

    y_min = float(np.min(candidate))
    y_max = float(np.max(candidate))
    if y_max <= y_min:
        y_max = y_min * (10.0 if use_log_scale else 1.1)

    if use_log_scale:
        return [np.log10(max(y_min * 0.95, 1e-12)), np.log10(y_max * 1.05)]

    pad = (y_max - y_min) * 0.02
    return [max(0.0, y_min - pad), y_max + pad]


def _resolve_log_time_axis(df_display, current_gen_date, view_max, m_dates):
    padded_start_date = _resolve_time_axis_start_date(df_display)
    range_start_day = max(1.0, float((padded_start_date - current_gen_date).days))
    range_end_day = max(float(view_max), range_start_day + 1.0)

    start_year = int(padded_start_date.year)
    end_year = int(max(df_display.index.max().year, m_dates[-1].year))
    tick_days = []
    tick_labels = []
    for year in range(start_year, end_year + 1):
        delta_days = (pd.Timestamp(f"{year}-01-01") - current_gen_date).days
        if range_start_day <= delta_days <= range_end_day:
            tick_days.append(delta_days)
            tick_labels.append(str(year))

    return [np.log10(range_start_day), np.log10(range_end_day)], tick_days, tick_labels


def _resolve_optional_sigma_offsets(p2_5, p16_5, p83_5, p97_5):
    scenario_levels = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=float)
    scenario_offsets = np.array([p2_5, p16_5, 0.0, p83_5, p97_5], dtype=float)
    if not np.all(np.isfinite(scenario_offsets)):
        return []

    return [
        (level, float(np.interp(level, scenario_levels, scenario_offsets)))
        for level in OPTIONAL_SIGMA_LEVELS
    ]


def _resolve_trace_sample_indices(values_length, max_points=MAX_RENDERED_MODEL_POINTS):
    values_length = int(values_length)
    max_points = max(2, int(max_points))
    if values_length <= max_points:
        return slice(None)

    stride = int(np.ceil((values_length - 1) / (max_points - 1)))
    sample_indices = np.arange(0, values_length, stride, dtype=int)
    if sample_indices[-1] != values_length - 1:
        sample_indices = np.append(sample_indices, values_length - 1)
    return sample_indices


def _sample_trace_values(values, sample_indices):
    if isinstance(sample_indices, slice):
        return values
    if isinstance(values, pd.Series):
        return values.iloc[sample_indices]

    values_array = np.asarray(values)
    sampled_values = values_array[sample_indices]
    if isinstance(values, list):
        return sampled_values.tolist()
    return sampled_values


def _convert_log_offsets_to_sigma_levels(values, percentile_offsets):
    values_arr = np.asarray(values, dtype=float)
    sigma_values = np.full(values_arr.shape, np.nan, dtype=float)
    valid_mask = np.isfinite(values_arr)
    if not np.any(valid_mask):
        return sigma_values

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
    if not np.all(np.isfinite(offsets)):
        return values_arr

    sort_order = np.argsort(offsets)
    offsets = offsets[sort_order]
    levels = levels[sort_order]
    offsets, unique_indices = np.unique(offsets, return_index=True)
    levels = levels[unique_indices]
    if offsets.size < 2:
        sigma_values[valid_mask] = 0.0
        return sigma_values

    valid_values = values_arr[valid_mask]
    converted_values = np.interp(valid_values, offsets, levels)

    below_mask = valid_values <= offsets[0]
    if np.any(below_mask):
        converted_values[below_mask] = levels[0] + (
            (valid_values[below_mask] - offsets[0])
            / (offsets[1] - offsets[0])
            * (levels[1] - levels[0])
        )

    above_mask = valid_values >= offsets[-1]
    if np.any(above_mask):
        converted_values[above_mask] = levels[-2] + (
            (valid_values[above_mask] - offsets[-2])
            / (offsets[-1] - offsets[-2])
            * (levels[-1] - levels[-2])
        )

    sigma_values[valid_mask] = converted_values
    return sigma_values


def _iter_moving_average_series(df_display, windows):
    if not windows:
        return []

    close_values = pd.to_numeric(df_display["CloseDisplay"], errors="coerce")
    lines = []
    for window in windows:
        window_days = int(window)
        if window_days <= 1:
            continue

        rolling_values = close_values.rolling(window_days, min_periods=window_days).mean()
        if rolling_values.notna().any():
            lines.append((window_days, rolling_values))

    return lines


def _format_sigma_line_name(level):
    return f"{level:+g}σ"


def _optional_sigma_line_style(level):
    return dict(color="rgba(168, 85, 247, 0.76)", width=1.0, dash="dash")


def _format_segmented_sigma_name(label):
    return f"Segmented sigma {label}"


def _format_segmented_sigma_level(value):
    if np.isclose(float(value), 0.0):
        return "0σ"
    return f"{float(value):+g}σ"


def _format_segmented_sigma_label(lower_level, upper_level):
    if lower_level is None:
        return f"< {_format_segmented_sigma_level(upper_level)}"
    if upper_level is None:
        return f"> {_format_segmented_sigma_level(lower_level)}"
    return (
        f"{_format_segmented_sigma_level(lower_level)} "
        f"to {_format_segmented_sigma_level(upper_level)}"
    )


def _format_segmented_sigma_abs_label(lower_level, upper_level):
    if lower_level is None or upper_level is None:
        threshold = upper_level if lower_level is None else lower_level
        return f"> ±{abs(float(threshold)):g}σ"

    low_abs = min(abs(float(lower_level)), abs(float(upper_level)))
    high_abs = max(abs(float(lower_level)), abs(float(upper_level)))
    if np.isclose(low_abs, 0.0):
        return f"0σ to ±{high_abs:g}σ"
    return f"±{low_abs:g}σ to ±{high_abs:g}σ"


def _resolve_segmented_sigma_style(lower_level, upper_level):
    if lower_level is None or upper_level is None:
        high_abs = np.inf
    else:
        high_abs = max(abs(float(lower_level)), abs(float(upper_level)))

    for max_abs_level, color, dash in SEGMENTED_SIGMA_PAIR_STYLES:
        if high_abs <= float(max_abs_level):
            return color, dash
    return SEGMENTED_SIGMA_PAIR_STYLES[-1][1], SEGMENTED_SIGMA_PAIR_STYLES[-1][2]


def _resolve_segmented_sigma_legend(lower_level, upper_level):
    label = _format_segmented_sigma_abs_label(lower_level, upper_level)
    group_key = (
        label.replace(" ", "_")
        .replace("±", "abs_")
        .replace(">", "gt_")
        .replace(".", "_")
        .replace("σ", "sigma")
    )
    return f"Segmented sigma {label}", f"segmented_sigma_{group_key}"


def _resolve_segmented_sigma_band_edges(center_level):
    lower_level = float(center_level) - SEGMENTED_SIGMA_HALF_STEP
    upper_level = float(center_level) + SEGMENTED_SIGMA_HALF_STEP
    if np.isclose(center_level, SEGMENTED_SIGMA_LEVELS[0]):
        lower_level = None
    if np.isclose(center_level, SEGMENTED_SIGMA_LEVELS[-1]):
        upper_level = None
    return lower_level, upper_level


def _iter_segmented_sigma_bands():
    for center_level in reversed(SEGMENTED_SIGMA_LEVELS):
        lower_level, upper_level = _resolve_segmented_sigma_band_edges(center_level)
        label = _format_segmented_sigma_label(lower_level, upper_level)
        line_color, line_dash = _resolve_segmented_sigma_style(lower_level, upper_level)
        yield lower_level, upper_level, label, line_color, line_dash


def _resolve_segment_mask(sigma_levels, lower_level, upper_level):
    if lower_level is None:
        return sigma_levels < float(upper_level)
    if upper_level is None:
        return sigma_levels >= float(lower_level)
    return (sigma_levels >= float(lower_level)) & (sigma_levels < float(upper_level))


def _iter_segmented_powerlaw_sigma_lines(df_display, m_log_d, p2_5, p16_5, p83_5, p97_5):
    if "Days" not in df_display:
        return []
    if "LogClose" in df_display:
        log_prices = pd.to_numeric(df_display["LogClose"], errors="coerce").to_numpy(dtype=float)
    elif "CloseDisplay" in df_display:
        close_values = pd.to_numeric(df_display["CloseDisplay"], errors="coerce").to_numpy(
            dtype=float
        )
        log_prices = np.full(close_values.shape, np.nan, dtype=float)
        positive_mask = close_values > 0.0
        log_prices[positive_mask] = np.log10(close_values[positive_mask])
    else:
        return []

    if "Res" in df_display:
        residuals = pd.to_numeric(df_display["Res"], errors="coerce").to_numpy(dtype=float)
    elif "ModelLog" in df_display:
        model_log = pd.to_numeric(df_display["ModelLog"], errors="coerce").to_numpy(dtype=float)
        residuals = log_prices - model_log
    else:
        return []

    days = pd.to_numeric(df_display["Days"], errors="coerce").to_numpy(dtype=float)
    log_days = np.full(days.shape, np.nan, dtype=float)
    positive_day_mask = days > 0.0
    log_days[positive_day_mask] = np.log10(days[positive_day_mask])
    sigma_levels = _convert_log_offsets_to_sigma_levels(residuals, (p2_5, p16_5, p83_5, p97_5))
    valid_mask = np.isfinite(log_days) & np.isfinite(log_prices) & np.isfinite(sigma_levels)
    if np.count_nonzero(valid_mask) < 2:
        return []

    model_log_days = np.asarray(m_log_d, dtype=float)
    lines = []
    for lower_level, upper_level, label, color, dash in _iter_segmented_sigma_bands():
        segment_mask = valid_mask & _resolve_segment_mask(sigma_levels, lower_level, upper_level)
        if np.count_nonzero(segment_mask) < 2:
            continue
        segment_log_days = log_days[segment_mask]
        if np.unique(segment_log_days).size < 2:
            continue
        slope_b, intercept_a = np.polyfit(segment_log_days, log_prices[segment_mask], 1)
        if not np.isfinite(intercept_a) or not np.isfinite(slope_b):
            continue
        model_values, _, was_clipped = evaluate_powerlaw_values(
            model_log_days,
            float(intercept_a),
            float(slope_b),
        )
        if was_clipped:
            continue
        legend_name, legendgroup = _resolve_segmented_sigma_legend(lower_level, upper_level)
        lines.append(
            {
                "name": _format_segmented_sigma_name(label),
                "legend_name": legend_name,
                "legendgroup": legendgroup,
                "values": model_values,
                "line": dict(color=color, width=1.25, dash=dash),
                "point_count": int(np.count_nonzero(segment_mask)),
            }
        )

    return lines


def _add_halving_trace(fig, current_gen_date, is_log_time, y_range, *, legendrank=35):
    if y_range is None or len(y_range) != 2:
        return
    y_min, y_max = float(y_range[0]), float(y_range[1])
    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
        return

    halving_x_values = []
    halving_y_values = []
    halving_hover_values = []
    for halving_date in HALVING_DATES:
        halving_x = (
            max(1.0, float((halving_date - current_gen_date).days)) if is_log_time else halving_date
        )
        halving_x_values.extend([halving_x, halving_x, None])
        halving_y_values.extend([y_min, y_max, None])
        halving_hover_values.extend(
            [halving_date.strftime("%d.%m.%Y"), halving_date.strftime("%d.%m.%Y"), None]
        )

    fig.add_trace(
        go.Scatter(
            x=halving_x_values,
            y=halving_y_values,
            mode="lines",
            name="Halvings",
            legendgroup="halvings",
            legendrank=legendrank,
            line=dict(color="#f0b90b", width=1.1, dash="dash"),
            opacity=0.75,
            customdata=halving_hover_values,
            hovertemplate="<b>Halving</b><br>%{customdata}<extra></extra>",
        )
    )


def render_main_model_chart(
    *,
    mode,
    time_scale,
    price_scale,
    df_display,
    current_gen_date,
    view_max,
    plot_x_model,
    plot_x_main,
    m_log_d,
    m_dates,
    m_dates_str,
    m_fair_display,
    residual_sigma_log,
    p2_5,
    p16_5,
    p83_5,
    p97_5,
    peak_powerlaw_overlay,
    pl_template,
    pl_bg_color,
    pl_grid_color,
    pl_btc_color,
    pl_legend_color,
    pl_text_color,
    c_hover_bg,
    c_hover_text,
    c_border,
    currency_prefix,
    currency_suffix,
    currency_decimals,
    target_series_name,
    target_series_unit,
    show_halving_lines,
    chart_key,
    historical_powerlaw_fair=None,
    historical_powerlaw_sigma_offsets=None,
    moving_average_windows=None,
    powerlaw_sigma_display_mode=POWERLAW_SIGMA_MODE_CLASSIC,
    **_removed_options,
):
    fig = go.Figure()
    tick_font = dict(color=pl_text_color, size=14, family="Arial Black, sans-serif")
    hover_label = dict(
        bgcolor=c_hover_bg, bordercolor=c_border, font=dict(color=c_hover_text, size=13)
    )
    is_log_time = time_scale == TIME_LOG
    model_sample_indices = _resolve_trace_sample_indices(len(plot_x_model))
    plot_x_model_render = _sample_trace_values(plot_x_model, model_sample_indices)

    if mode == MODE_POWERLAW:
        p97_5_name = "+2σ (97.5th percentile)"
        p83_5_name = "+1σ (83.5th percentile)"
        p16_5_name = "-1σ (16.5th percentile)"
        p2_5_name = "-2σ (2.5th percentile)"

        p2_5_series, _, _ = evaluate_powerlaw_values(np.log10(m_fair_display), p2_5, 1.0)
        p16_5_series, _, _ = evaluate_powerlaw_values(np.log10(m_fair_display), p16_5, 1.0)
        p83_5_series, _, _ = evaluate_powerlaw_values(np.log10(m_fair_display), p83_5, 1.0)
        p97_5_series, _, _ = evaluate_powerlaw_values(np.log10(m_fair_display), p97_5, 1.0)

        if is_log_time:
            fig.add_trace(
                go.Scatter(
                    x=plot_x_model,
                    y=p97_5_series,
                    mode="lines",
                    line=dict(color="rgba(0,0,0,0)", width=0.1),
                    name="",
                    legendrank=0,
                    showlegend=False,
                    customdata=m_dates_str,
                    hovertemplate="<b>📅 %{customdata}</b><extra></extra>",
                )
            )

        main_series_label = f"{target_series_name} ({target_series_unit})"
        btc_hover = (
            (
                f"<b>{main_series_label}</b>: "
                f"{currency_prefix}%{{y:,.{currency_decimals}f}}{currency_suffix}<extra></extra>"
            )
            if is_log_time
            else (
                f"<b>{main_series_label}</b>: "
                f"{currency_prefix}%{{y:,.{currency_decimals}f}}{currency_suffix}<extra></extra>"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=plot_x_main,
                y=df_display["CloseDisplay"],
                mode="lines",
                name=main_series_label,
                line=dict(color=pl_btc_color, width=1.5),
                legendrank=10,
                hovertemplate=btc_hover,
            )
        )
        for ma_index, (window_days, moving_average) in enumerate(
            _iter_moving_average_series(df_display, moving_average_windows)
        ):
            ma_name = f"{window_days}D MA"
            ma_color = MOVING_AVERAGE_LINE_STYLES[ma_index % len(MOVING_AVERAGE_LINE_STYLES)]
            fig.add_trace(
                go.Scatter(
                    x=plot_x_main,
                    y=moving_average,
                    mode="lines",
                    name=ma_name,
                    line=dict(color=ma_color, width=1.4),
                    legendrank=11 + ma_index,
                    customdata=df_display.index.strftime("%d.%m.%Y"),
                    hovertemplate=(
                        f"<b>{ma_name}</b>: "
                        f"{currency_prefix}%{{y:,.{currency_decimals}f}}{currency_suffix}"
                        "<br>%{customdata}<extra></extra>"
                    ),
                )
            )

        def add_model_line(y_values, name, line, legendgroup, visible=True):
            fig.add_trace(
                go.Scatter(
                    x=plot_x_model_render,
                    y=_sample_trace_values(y_values, model_sample_indices),
                    mode="lines",
                    line=line,
                    name=name,
                    legendgroup=legendgroup,
                    showlegend=False,
                    visible=visible,
                    hovertemplate=(
                        f"<b>{name}</b>: "
                        f"{currency_prefix}%{{y:,.{currency_decimals}f}}{currency_suffix}<extra></extra>"
                    ),
                )
            )

        def add_legend_item(name, line, legendgroup, visible=True, legendrank=None):
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    line=line,
                    name=name,
                    legendgroup=legendgroup,
                    legendrank=legendrank,
                    visible=visible,
                    hoverinfo="skip",
                    showlegend=True,
                )
            )

        def add_segmented_model_line(
            y_values,
            name,
            legend_name,
            legendgroup,
            line,
            showlegend,
            visible=True,
        ):
            fig.add_trace(
                go.Scatter(
                    x=plot_x_model_render,
                    y=_sample_trace_values(y_values, model_sample_indices),
                    mode="lines",
                    line=line,
                    name=legend_name if showlegend else name,
                    legendgroup=legendgroup,
                    showlegend=showlegend,
                    visible=visible,
                    hovertemplate=(
                        f"<b>{name}</b>: "
                        f"{currency_prefix}%{{y:,.{currency_decimals}f}}{currency_suffix}<extra></extra>"
                    ),
                )
            )

        use_segmented_sigma = powerlaw_sigma_display_mode == POWERLAW_SIGMA_MODE_SEGMENTED
        use_historical_powerlaw = powerlaw_sigma_display_mode == POWERLAW_SIGMA_MODE_HISTORICAL
        use_classic_sigma = not use_segmented_sigma and not use_historical_powerlaw
        historical_powerlaw_available = False
        optional_sigma_series = {}
        if use_classic_sigma:
            for sigma_level, sigma_offset in _resolve_optional_sigma_offsets(
                p2_5,
                p16_5,
                p83_5,
                p97_5,
            ):
                sigma_series, _, _ = evaluate_powerlaw_values(
                    np.log10(m_fair_display),
                    sigma_offset,
                    1.0,
                )
                optional_sigma_series[sigma_level] = sigma_series

            add_model_line(
                p97_5_series,
                p97_5_name,
                dict(color="#ea3d2f", width=1.2, dash="dot"),
                "sigma_abs_2",
            )
            add_model_line(
                optional_sigma_series[1.5],
                _format_sigma_line_name(1.5),
                _optional_sigma_line_style(1.5),
                "sigma_abs_1_5",
                visible="legendonly",
            )
            add_model_line(
                p83_5_series,
                p83_5_name,
                dict(color="#1199d6", width=1.2, dash="dot"),
                "sigma_abs_1",
            )
            add_model_line(
                optional_sigma_series[0.5],
                _format_sigma_line_name(0.5),
                _optional_sigma_line_style(0.5),
                "sigma_abs_0_5",
                visible="legendonly",
            )
        if use_historical_powerlaw:
            historical_fair_values = np.asarray(
                historical_powerlaw_fair if historical_powerlaw_fair is not None else [],
                dtype=float,
            )
            if historical_fair_values.shape == df_display["CloseDisplay"].shape and np.any(
                np.isfinite(historical_fair_values)
            ):
                historical_fair_display, _, _ = evaluate_powerlaw_values(
                    historical_fair_values,
                    0.0,
                    1.0,
                )
                historical_powerlaw_available = True
                fig.add_trace(
                    go.Scatter(
                        x=plot_x_main,
                        y=historical_fair_display,
                        mode="lines",
                        line=dict(color="#f0b90b", width=2.2),
                        name="Historical PowerLaw",
                        legendgroup="historical_power_regression",
                        showlegend=False,
                        customdata=df_display.index.strftime("%d.%m.%Y"),
                        hovertemplate=(
                            "<b>Historical PowerLaw</b>: "
                            f"{currency_prefix}%{{y:,.{currency_decimals}f}}{currency_suffix}"
                            "<br>%{customdata}<extra></extra>"
                        ),
                    )
                )
                historical_sigma_offsets = np.asarray(
                    (
                        historical_powerlaw_sigma_offsets
                        if historical_powerlaw_sigma_offsets is not None
                        else []
                    ),
                    dtype=float,
                )
                if historical_sigma_offsets.shape == (4, len(historical_fair_display)):
                    historical_sigma_styles = (
                        (3, "+2σ", "#ea3d2f", "historical_sigma_2"),
                        (2, "+1σ", "#1199d6", "historical_sigma_1"),
                        (1, "-1σ", "#1199d6", "historical_sigma_1"),
                        (0, "-2σ", "#ea3d2f", "historical_sigma_2"),
                    )
                    for offset_index, sigma_label, color, legendgroup in historical_sigma_styles:
                        sigma_values, _, _ = evaluate_powerlaw_values(
                            historical_fair_values,
                            historical_sigma_offsets[offset_index],
                            1.0,
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=plot_x_main,
                                y=sigma_values,
                                mode="lines",
                                line=dict(color=color, width=1.2, dash="dot"),
                                name=f"Historical {sigma_label}",
                                legendgroup=legendgroup,
                                showlegend=False,
                                hoverinfo="skip",
                            )
                        )
            else:
                add_model_line(
                    m_fair_display,
                    "Power regression",
                    dict(color="#f0b90b", width=1.8),
                    "power_regression",
                )
        else:
            add_model_line(
                m_fair_display,
                "Power regression",
                dict(color="#f0b90b", width=1.8),
                "power_regression",
            )
        if use_segmented_sigma:
            segmented_sigma_lines = _iter_segmented_powerlaw_sigma_lines(
                df_display,
                m_log_d,
                p2_5,
                p16_5,
                p83_5,
                p97_5,
            )
            shown_segmented_legendgroups = set()
            for sigma_line in segmented_sigma_lines:
                show_segmented_legend = (
                    sigma_line["legendgroup"] not in shown_segmented_legendgroups
                )
                shown_segmented_legendgroups.add(sigma_line["legendgroup"])
                segmented_visible = (
                    "legendonly"
                    if sigma_line["legend_name"] in SEGMENTED_SIGMA_DEFAULT_HIDDEN_LEGENDS
                    else True
                )
                add_segmented_model_line(
                    sigma_line["values"],
                    sigma_line["name"],
                    sigma_line["legend_name"],
                    sigma_line["legendgroup"],
                    sigma_line["line"],
                    show_segmented_legend,
                    segmented_visible,
                )
        if use_classic_sigma:
            add_model_line(
                optional_sigma_series[-0.5],
                _format_sigma_line_name(-0.5),
                _optional_sigma_line_style(-0.5),
                "sigma_abs_0_5",
                visible="legendonly",
            )
            add_model_line(
                p16_5_series,
                p16_5_name,
                dict(color="#1199d6", width=1.2, dash="dot"),
                "sigma_abs_1",
            )
            add_model_line(
                optional_sigma_series[-1.5],
                _format_sigma_line_name(-1.5),
                _optional_sigma_line_style(-1.5),
                "sigma_abs_1_5",
                visible="legendonly",
            )
            add_model_line(
                p2_5_series,
                p2_5_name,
                dict(color="#ea3d2f", width=1.2, dash="dot"),
                "sigma_abs_2",
            )
            add_legend_item(
                "±2σ (2.5th/97.5th percentile)",
                dict(color="#ea3d2f", width=1.2, dash="dot"),
                "sigma_abs_2",
                legendrank=100,
            )
            add_legend_item(
                "±1.5σ",
                _optional_sigma_line_style(-1.5),
                "sigma_abs_1_5",
                visible="legendonly",
                legendrank=110,
            )
            add_legend_item(
                "±1σ (16.5th/83.5th percentile)",
                dict(color="#1199d6", width=1.2, dash="dot"),
                "sigma_abs_1",
                legendrank=120,
            )
            add_legend_item(
                "±0.5σ",
                _optional_sigma_line_style(-0.5),
                "sigma_abs_0_5",
                visible="legendonly",
                legendrank=130,
            )
        add_legend_item(
            "Historical PowerLaw" if historical_powerlaw_available else "Power regression",
            dict(color="#f0b90b", width=2.2 if historical_powerlaw_available else 1.8),
            "historical_power_regression" if historical_powerlaw_available else "power_regression",
            legendrank=20,
        )
        if historical_powerlaw_available:
            add_legend_item(
                "Historical ±2σ",
                dict(color="#ea3d2f", width=1.2, dash="dot"),
                "historical_sigma_2",
                legendrank=100,
            )
            add_legend_item(
                "Historical ±1σ",
                dict(color="#1199d6", width=1.2, dash="dot"),
                "historical_sigma_1",
                legendrank=110,
            )
        y_range_model_x = plot_x_model if is_log_time else m_dates
        y_range_visible_start = (
            max(1.0, float(df_display["Days"].min())) if is_log_time else df_display.index.min()
        )
        powerlaw_y_range = _resolve_powerlaw_y_range(
            df_display,
            m_fair_display,
            p2_5,
            p97_5,
            use_log_scale=(price_scale == TIME_LOG),
            model_x=y_range_model_x,
            visible_start_day=y_range_visible_start,
        )
        if show_halving_lines and powerlaw_y_range is not None:
            halving_y_range = (
                [10.0 ** powerlaw_y_range[0], 10.0 ** powerlaw_y_range[1]]
                if price_scale == TIME_LOG
                else powerlaw_y_range
            )
            _add_halving_trace(fig, current_gen_date, is_log_time, halving_y_range)
        fig.update_yaxes(
            type="log" if price_scale == TIME_LOG else "linear",
            range=powerlaw_y_range,
            gridcolor=pl_grid_color,
            tickfont=tick_font,
        )

    if is_log_time:
        x_range, t_vals, t_text = _resolve_log_time_axis(
            df_display, current_gen_date, view_max, m_dates
        )
        fig.update_xaxes(
            type="log",
            tickvals=t_vals,
            ticktext=t_text,
            range=x_range,
            gridcolor=pl_grid_color,
            tickfont=tick_font,
            unifiedhovertitle=dict(text=" "),
        )
    else:
        fig.update_xaxes(
            type="date",
            gridcolor=pl_grid_color,
            tickfont=tick_font,
            range=[_resolve_time_axis_start_date(df_display), m_dates[-1]],
            hoverformat="%d.%m.%Y",
        )

    spike_axis_style = dict(
        showspikes=False,
        spikecolor=pl_legend_color,
        spikedash="dot",
        spikemode="across",
        spikesnap="cursor",
        spikethickness=1,
    )
    fig.update_xaxes(**spike_axis_style)
    fig.update_yaxes(**spike_axis_style)

    fig.update_layout(
        height=600,
        margin=dict(t=40, b=72, l=50, r=20),
        template=pl_template,
        font=dict(color=pl_text_color),
        legend=dict(
            orientation="h",
            y=-0.12,
            yanchor="top",
            x=0,
            xanchor="left",
            font=dict(size=13, color=pl_legend_color),
            bgcolor="rgba(0,0,0,0)",
            groupclick="togglegroup",
            traceorder="normal",
        ),
        paper_bgcolor=pl_bg_color,
        plot_bgcolor=pl_bg_color,
        hovermode="x unified",
        hoverlabel=hover_label,
    )
    st.plotly_chart(
        fig,
        width="stretch",
        theme=None,
        config=_main_chart_plotly_config(),
        key=chart_key,
    )
