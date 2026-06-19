import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core.constants import MODE_LOGPERIODIC, MODE_POWERLAW, TIME_LOG
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
LOGPERIODIC_SIGMA_DISPLAY_RANGE = (-3.0, 3.0)
LOGPERIODIC_EXTREMA_HARMONICS = (
    (1, "ω", "solid", 1.5, 0.82),
    (2, "ω,2ω", "dash", 1.15, 0.62),
    (3, "ω,2ω,4ω", "dot", 0.95, 0.48),
)
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


def _add_logperiodic_extrema_traces(fig, extrema_lines, current_gen_date, is_log_time, y_range):
    if y_range is None or len(y_range) != 2:
        return
    y_min, y_max = float(y_range[0]), float(y_range[1])
    if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
        return

    for kind, name, legendrank in (
        ("high", "Cycle highs", 36),
        ("low", "Cycle lows", 37),
    ):
        kind_lines = [line for line in extrema_lines if line["kind"] == kind]
        if not kind_lines:
            continue

        x_values = []
        y_values = []
        hover_values = []
        for line in kind_lines:
            x_value = line["x"]
            if is_log_time:
                hover_date = current_gen_date + pd.Timedelta(days=float(x_value))
            else:
                hover_date = pd.Timestamp(x_value)
            x_values.extend([x_value, x_value, None])
            y_values.extend([y_min, y_max, None])
            hover_values.extend(
                [hover_date.strftime("%d.%m.%Y"), hover_date.strftime("%d.%m.%Y"), None]
            )

        line_style = kind_lines[0]
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                name=name,
                legendgroup=f"cycle_{kind}s",
                legendrank=legendrank,
                line=dict(
                    color=line_style["color"],
                    width=line_style["width"],
                    dash=line_style["dash"],
                ),
                opacity=line_style["opacity"],
                customdata=hover_values,
                hovertemplate=f"<b>{name}</b><br>%{{customdata}}<extra></extra>",
            )
        )


def _resolve_linear_y_span(*series_parts):
    finite_values = []
    for values in series_parts:
        if values is None:
            continue
        values_arr = np.asarray(values, dtype=float)
        finite_values.extend(values_arr[np.isfinite(values_arr)].tolist())
    if not finite_values:
        return None

    y_min = float(np.min(finite_values))
    y_max = float(np.max(finite_values))
    if y_max <= y_min:
        y_max = y_min + 1.0
    padding = max((y_max - y_min) * 0.02, 0.1)
    return [y_min - padding, y_max + padding]


def _iter_logperiodic_extrema_lines(plot_x_model, harmonic_curves, selected_harmonic_count):
    if not harmonic_curves:
        return []

    x_values = np.asarray(plot_x_model)
    extrema_lines = []
    style_by_count = {
        harmonic_count: (label, dash, width, opacity)
        for harmonic_count, label, dash, width, opacity in LOGPERIODIC_EXTREMA_HARMONICS
    }
    harmonic_count = int(selected_harmonic_count)
    if harmonic_count not in harmonic_curves or harmonic_count not in style_by_count:
        return []

    y_values = np.asarray(harmonic_curves[harmonic_count], dtype=float)
    if x_values.size != y_values.size or y_values.size < 3:
        return []

    finite_mask = np.isfinite(y_values)
    if not np.all(finite_mask):
        x_curve = x_values[finite_mask]
        y_curve = y_values[finite_mask]
    else:
        x_curve = x_values
        y_curve = y_values
    if y_curve.size < 3:
        return []

    label, dash, width, opacity = style_by_count[harmonic_count]
    previous_values = y_curve[:-2]
    current_values = y_curve[1:-1]
    next_values = y_curve[2:]
    local_high_mask = (current_values > previous_values) & (current_values >= next_values)
    local_low_mask = (current_values < previous_values) & (current_values <= next_values)

    for x_value in x_curve[1:-1][local_high_mask]:
        extrema_lines.append(
            {
                "x": x_value,
                "kind": "high",
                "label": label,
                "color": "#ea3d2f",
                "dash": dash,
                "width": width,
                "opacity": opacity,
            }
        )
    for x_value in x_curve[1:-1][local_low_mask]:
        extrema_lines.append(
            {
                "x": x_value,
                "kind": "low",
                "label": label,
                "color": "#1199d6",
                "dash": dash,
                "width": width,
                "opacity": opacity,
            }
        )

    return extrema_lines


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
    plot_x_osc,
    m_log_d,
    m_dates,
    m_dates_str,
    m_fair_display,
    historical_powerlaw_slopes,
    show_historical_powerlaw_slope,
    m_osc_y,
    m_osc_y_by_harmonic,
    perrenod_curve,
    residual_sigma_log,
    p2_5,
    p16_5,
    p83_5,
    p97_5,
    peak_powerlaw_overlay,
    osc_t1_age,
    osc_lambda,
    selected_harmonic_count,
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
    bitcoin_residual_overlay_df=None,
    osc_visible_start_abs_day=None,
    moving_average_windows=None,
):
    fig = (
        make_subplots(specs=[[{"secondary_y": True}]]) if mode == MODE_LOGPERIODIC else go.Figure()
    )
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

        optional_sigma_series = {}
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
        add_model_line(
            m_fair_display,
            "Power regression",
            dict(color="#f0b90b", width=1.8),
            "power_regression",
        )
        if peak_powerlaw_overlay is not None and peak_powerlaw_overlay.get("peak") is not None:
            peak_overlay = peak_powerlaw_overlay["peak"]
            peak_values = peak_overlay["model_values"]
            add_model_line(
                peak_values,
                "Peak PowerLaw",
                dict(color="#22c55e", width=1.6, dash="longdash"),
                "powerlaw_envelope",
                visible="legendonly",
            )
            peak_days = np.asarray(peak_overlay["peak_days"], dtype=float)
            peak_x = (
                peak_days
                if is_log_time
                else [current_gen_date + pd.Timedelta(days=float(day)) for day in peak_days]
            )
            fig.add_trace(
                go.Scatter(
                    x=peak_x,
                    y=peak_overlay["peak_values"],
                    mode="markers",
                    marker=dict(
                        color="#f97316",
                        size=9,
                        symbol="circle",
                        line=dict(color="#fff7ed", width=1.4),
                    ),
                    name="Peak fit points",
                    legendgroup="peak_fit_points",
                    legendrank=31,
                    visible="legendonly",
                    showlegend=True,
                    customdata=[
                        (current_gen_date + pd.Timedelta(days=float(day))).strftime("%d.%m.%Y")
                        for day in peak_days
                    ],
                    hovertemplate=(
                        f"<b>Peak fit point</b>: "
                        f"{currency_prefix}%{{y:,.{currency_decimals}f}}{currency_suffix}"
                        "<br>%{customdata}<extra></extra>"
                    ),
                )
            )
        if peak_powerlaw_overlay is not None and peak_powerlaw_overlay.get("trough") is not None:
            trough_overlay = peak_powerlaw_overlay["trough"]
            trough_values = trough_overlay["model_values"]
            add_model_line(
                trough_values,
                "Trough PowerLaw",
                dict(color="#22c55e", width=1.6, dash="longdash"),
                "powerlaw_envelope",
                visible="legendonly",
            )
            trough_days = np.asarray(trough_overlay["trough_days"], dtype=float)
            trough_x = (
                trough_days
                if is_log_time
                else [current_gen_date + pd.Timedelta(days=float(day)) for day in trough_days]
            )
            fig.add_trace(
                go.Scatter(
                    x=trough_x,
                    y=trough_overlay["trough_values"],
                    mode="markers",
                    marker=dict(
                        color="#f97316",
                        size=9,
                        symbol="circle",
                        line=dict(color="#fff7ed", width=1.4),
                    ),
                    name="Trough fit points",
                    legendgroup="trough_fit_points",
                    legendrank=41,
                    visible="legendonly",
                    showlegend=True,
                    customdata=[
                        (current_gen_date + pd.Timedelta(days=float(day))).strftime("%d.%m.%Y")
                        for day in trough_days
                    ],
                    hovertemplate=(
                        f"<b>Trough fit point</b>: "
                        f"{currency_prefix}%{{y:,.{currency_decimals}f}}{currency_suffix}"
                        "<br>%{customdata}<extra></extra>"
                    ),
                )
            )
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
            "Power regression",
            dict(color="#f0b90b", width=1.8),
            "power_regression",
            legendrank=20,
        )
        if (
            peak_powerlaw_overlay is not None
            and peak_powerlaw_overlay.get("peak") is not None
            and peak_powerlaw_overlay.get("trough") is not None
        ):
            add_legend_item(
                "Peak/Trough PowerLaw",
                dict(color="#22c55e", width=1.6, dash="longdash"),
                "powerlaw_envelope",
                visible="legendonly",
                legendrank=30,
            )
        elif peak_powerlaw_overlay is not None and peak_powerlaw_overlay.get("peak") is not None:
            add_legend_item(
                "Peak PowerLaw",
                dict(color="#22c55e", width=1.6, dash="longdash"),
                "powerlaw_envelope",
                visible="legendonly",
                legendrank=30,
            )
        elif peak_powerlaw_overlay is not None and peak_powerlaw_overlay.get("trough") is not None:
            add_legend_item(
                "Trough PowerLaw",
                dict(color="#22c55e", width=1.6, dash="longdash"),
                "powerlaw_envelope",
                visible="legendonly",
                legendrank=40,
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
    else:
        osc_mask = np.ones(len(df_display), dtype=bool)
        if osc_visible_start_abs_day is not None:
            osc_mask = df_display["AbsDays"].to_numpy(dtype=float) >= float(
                osc_visible_start_abs_day
            )

        osc_x_vals = np.asarray(plot_x_osc)[osc_mask]
        logperiodic_sigma_offsets = (p2_5, p16_5, p83_5, p97_5)
        osc_y_vals = _convert_log_offsets_to_sigma_levels(
            df_display["Res"].to_numpy(dtype=float)[osc_mask],
            logperiodic_sigma_offsets,
        )
        osc_dates = df_display.index.strftime("%d.%m.%Y").to_numpy()[osc_mask]
        osc_prices = df_display["CloseDisplay"].to_numpy(dtype=float)[osc_mask]
        osc_hover_data = np.column_stack([osc_dates, osc_prices])

        if bitcoin_residual_overlay_df is not None and not bitcoin_residual_overlay_df.empty:
            btc_residual_x = (
                bitcoin_residual_overlay_df["Days"]
                if is_log_time
                else bitcoin_residual_overlay_df.index
            )
            btc_residual_values = pd.to_numeric(
                bitcoin_residual_overlay_df["ResidualSigma"], errors="coerce"
            )
            btc_residual_mask = np.isfinite(btc_residual_values.to_numpy(dtype=float))
            if np.any(btc_residual_mask):
                fig.add_trace(
                    go.Scatter(
                        x=np.asarray(btc_residual_x)[btc_residual_mask],
                        y=btc_residual_values.to_numpy(dtype=float)[btc_residual_mask],
                        mode="lines",
                        name="Bitcoin price residual σ",
                        line=dict(color=pl_btc_color, width=1.4),
                        customdata=bitcoin_residual_overlay_df.index.strftime(
                            "%d.%m.%Y"
                        ).to_numpy()[btc_residual_mask],
                        hovertemplate="<b>%{customdata}</b><br>Bitcoin price residual: %{y:.2f}σ<extra></extra>",
                        visible="legendonly",
                    )
                )

        fig.add_trace(
            go.Scatter(
                x=osc_x_vals,
                y=osc_y_vals,
                mode="lines",
                name="power-law residual σ",
                line=dict(color="rgba(180, 185, 192, 0.42)", width=1.1),
                customdata=osc_hover_data,
                hovertemplate=(
                    "<b>%{customdata[0]}</b>"
                    f"<br>{target_series_name}: "
                    f"{currency_prefix}%{{customdata[1]:,.{currency_decimals}f}}{currency_suffix}"
                    "<br><b>power-law residual σ</b>: %{y:.2f}σ<extra></extra>"
                ),
            )
        )
        if show_historical_powerlaw_slope:
            slope_vals = np.asarray(historical_powerlaw_slopes, dtype=float)[osc_mask]
            finite_slope_vals = slope_vals[np.isfinite(slope_vals)]
            final_slope_label = f" {finite_slope_vals[-1]:.3f}" if finite_slope_vals.size else ""
            fig.add_trace(
                go.Scatter(
                    x=osc_x_vals,
                    y=slope_vals,
                    mode="lines",
                    name=f"PowerLaw B{final_slope_label}",
                    line=dict(color="#f0b90b", width=1.9),
                    customdata=osc_dates,
                    hovertemplate="<b>%{customdata}</b><br>PowerLaw B: %{y:.3f}<extra></extra>",
                    visible="legendonly",
                ),
                secondary_y=True,
            )
        harmonic_curves = m_osc_y_by_harmonic or {selected_harmonic_count: m_osc_y}
        harmonic_colors = {1: "#2f80b7", 2: "#f28e2b", 3: "#2aa84a"}
        harmonic_labels = {1: "ω", 2: "ω,2ω", 3: "ω,2ω,4ω"}
        for harmonic_count in sorted(harmonic_curves):
            if harmonic_count > int(selected_harmonic_count):
                continue
            fig.add_trace(
                go.Scatter(
                    x=plot_x_model_render,
                    y=_convert_log_offsets_to_sigma_levels(
                        _sample_trace_values(
                            harmonic_curves[harmonic_count],
                            model_sample_indices,
                        ),
                        logperiodic_sigma_offsets,
                    ),
                    mode="lines",
                    name=f"DSI {harmonic_labels.get(harmonic_count, harmonic_count)}",
                    line=dict(
                        color=harmonic_colors.get(harmonic_count, "#ea3d2f"),
                        width=(2.6 if harmonic_count == int(selected_harmonic_count) else 1.9),
                    ),
                    hoverinfo="skip",
                    visible="legendonly",
                )
            )
        if perrenod_curve is not None:
            perrenod_name = str(perrenod_curve.get("label", "DSI ω,2ω,4ω Decayed")).replace(
                " decayed", " Decayed"
            )
            fig.add_trace(
                go.Scatter(
                    x=plot_x_model_render,
                    y=_convert_log_offsets_to_sigma_levels(
                        _sample_trace_values(perrenod_curve["values"], model_sample_indices),
                        logperiodic_sigma_offsets,
                    ),
                    mode="lines",
                    name=perrenod_name,
                    line=dict(color="#f0b90b", width=3.0, dash="solid"),
                    hoverinfo="skip",
                )
            )
        extrema_curves = harmonic_curves
        extrema_harmonic_count = selected_harmonic_count
        if perrenod_curve is not None and "values" in perrenod_curve:
            extrema_curves = {3: perrenod_curve["values"]}
            extrema_harmonic_count = 3
        extrema_values = (
            perrenod_curve["values"]
            if perrenod_curve is not None and "values" in perrenod_curve
            else harmonic_curves.get(selected_harmonic_count)
        )
        logperiodic_y_range = _resolve_linear_y_span(
            osc_y_vals,
            (
                _convert_log_offsets_to_sigma_levels(
                    extrema_values,
                    logperiodic_sigma_offsets,
                )
                if extrema_values is not None
                else None
            ),
        )
        logperiodic_y_range = [
            LOGPERIODIC_SIGMA_DISPLAY_RANGE[0],
            LOGPERIODIC_SIGMA_DISPLAY_RANGE[1],
        ]
        fig.add_hline(y=0, line_width=1, line_color=pl_legend_color)
        fig.update_yaxes(
            type="linear",
            title_text="Sigma residual",
            range=logperiodic_y_range,
            gridcolor=pl_grid_color,
            tickfont=tick_font,
            secondary_y=False,
        )
        if show_historical_powerlaw_slope:
            fig.update_yaxes(
                title_text="PowerLaw B",
                showgrid=False,
                tickfont=tick_font,
                secondary_y=True,
            )

        extrema_lines = _iter_logperiodic_extrema_lines(
            plot_x_model,
            extrema_curves,
            extrema_harmonic_count,
        )
        _add_logperiodic_extrema_traces(
            fig,
            extrema_lines,
            current_gen_date,
            is_log_time,
            logperiodic_y_range,
        )
        if show_halving_lines:
            _add_halving_trace(
                fig,
                current_gen_date,
                is_log_time,
                logperiodic_y_range,
                legendrank=35,
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
            font=dict(size=11, color=pl_legend_color),
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
