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
OPTIONAL_SIGMA_LEVELS = (-1.5, -0.5, 0.5, 1.5)


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


def _format_sigma_line_name(level):
    return f"{level:+g}σ"


def _optional_sigma_line_style(level):
    if level > 0.0:
        return dict(color="rgba(234, 61, 47, 0.72)", width=1.0, dash="dash")
    return dict(color="rgba(17, 153, 214, 0.72)", width=1.0, dash="dash")


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
    osc_visible_start_abs_day=None,
    chart_key,
):
    fig = (
        make_subplots(specs=[[{"secondary_y": True}]]) if mode == MODE_LOGPERIODIC else go.Figure()
    )
    tick_font = dict(color=pl_text_color, size=14, family="Arial Black, sans-serif")
    hover_label = dict(
        bgcolor=c_hover_bg, bordercolor=c_border, font=dict(color=c_hover_text, size=13)
    )
    is_log_time = time_scale == TIME_LOG

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
                customdata=df_display.index.strftime("%d.%m.%Y"),
                hovertemplate=btc_hover,
            )
        )

        def add_model_line(y_values, name, line, legendgroup, visible=True):
            fig.add_trace(
                go.Scatter(
                    x=plot_x_model,
                    y=y_values,
                    mode="lines",
                    line=line,
                    name=name,
                    legendgroup=legendgroup,
                    showlegend=False,
                    visible=visible,
                    customdata=m_dates_str,
                    hovertemplate=(
                        f"<b>{name}</b>: "
                        f"{currency_prefix}%{{y:,.{currency_decimals}f}}{currency_suffix}<extra></extra>"
                    ),
                )
            )

        def add_legend_item(name, line, legendgroup, visible=True):
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    line=line,
                    name=name,
                    legendgroup=legendgroup,
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
            "sigma_p2",
            visible="legendonly",
        )
        add_model_line(
            optional_sigma_series[1.5],
            _format_sigma_line_name(1.5),
            _optional_sigma_line_style(1.5),
            "sigma_p1_5",
            visible="legendonly",
        )
        add_model_line(
            p83_5_series,
            p83_5_name,
            dict(color="#1199d6", width=1.2, dash="dot"),
            "sigma_p1",
            visible="legendonly",
        )
        add_model_line(
            optional_sigma_series[0.5],
            _format_sigma_line_name(0.5),
            _optional_sigma_line_style(0.5),
            "sigma_p0_5",
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
                "peak_powerlaw",
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
                    legendgroup="peak_powerlaw",
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
                "trough_powerlaw",
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
                    legendgroup="trough_powerlaw",
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
            "sigma_m0_5",
            visible="legendonly",
        )
        add_model_line(
            p16_5_series,
            p16_5_name,
            dict(color="#1199d6", width=1.2, dash="dot"),
            "sigma_m1",
            visible="legendonly",
        )
        add_model_line(
            optional_sigma_series[-1.5],
            _format_sigma_line_name(-1.5),
            _optional_sigma_line_style(-1.5),
            "sigma_m1_5",
            visible="legendonly",
        )
        add_model_line(
            p2_5_series,
            p2_5_name,
            dict(color="#ea3d2f", width=1.2, dash="dot"),
            "sigma_m2",
            visible="legendonly",
        )
        add_legend_item(
            p2_5_name,
            dict(color="#ea3d2f", width=1.2, dash="dot"),
            "sigma_m2",
            visible="legendonly",
        )
        add_legend_item(
            _format_sigma_line_name(-1.5),
            _optional_sigma_line_style(-1.5),
            "sigma_m1_5",
            visible="legendonly",
        )
        add_legend_item(
            p16_5_name,
            dict(color="#1199d6", width=1.2, dash="dot"),
            "sigma_m1",
            visible="legendonly",
        )
        add_legend_item(
            _format_sigma_line_name(-0.5),
            _optional_sigma_line_style(-0.5),
            "sigma_m0_5",
            visible="legendonly",
        )
        add_legend_item(
            "Power regression",
            dict(color="#f0b90b", width=1.8),
            "power_regression",
        )
        if peak_powerlaw_overlay is not None and peak_powerlaw_overlay.get("peak") is not None:
            add_legend_item(
                "Peak PowerLaw",
                dict(color="#22c55e", width=1.6, dash="longdash"),
                "peak_powerlaw",
            )
        if peak_powerlaw_overlay is not None and peak_powerlaw_overlay.get("trough") is not None:
            add_legend_item(
                "Trough PowerLaw",
                dict(color="#22c55e", width=1.6, dash="longdash"),
                "trough_powerlaw",
            )
        add_legend_item(
            _format_sigma_line_name(0.5),
            _optional_sigma_line_style(0.5),
            "sigma_p0_5",
            visible="legendonly",
        )
        add_legend_item(
            p83_5_name,
            dict(color="#1199d6", width=1.2, dash="dot"),
            "sigma_p1",
            visible="legendonly",
        )
        add_legend_item(
            _format_sigma_line_name(1.5),
            _optional_sigma_line_style(1.5),
            "sigma_p1_5",
            visible="legendonly",
        )
        add_legend_item(
            p97_5_name,
            dict(color="#ea3d2f", width=1.2, dash="dot"),
            "sigma_p2",
            visible="legendonly",
        )
        if show_halving_lines:
            for halving_date in HALVING_DATES:
                halving_x = (
                    max(1.0, float((halving_date - current_gen_date).days))
                    if is_log_time
                    else halving_date
                )
                fig.add_vline(
                    x=halving_x,
                    line_width=1.1,
                    line_dash="dash",
                    line_color="#f0b90b",
                    opacity=0.75,
                )
                fig.add_annotation(
                    x=halving_x,
                    y=0.02,
                    yref="paper",
                    text=f"Halving {halving_date.year}",
                    showarrow=False,
                    yshift=0,
                    textangle=-90,
                    font=dict(size=9, color=pl_legend_color),
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
        sigma_scale = float(residual_sigma_log)
        if not np.isfinite(sigma_scale) or sigma_scale <= 0.0:
            sigma_scale = 1.0
        osc_y_vals = df_display["Res"].to_numpy(dtype=float)[osc_mask] / sigma_scale
        osc_dates = df_display.index.strftime("%d.%m.%Y").to_numpy()[osc_mask]

        fig.add_trace(
            go.Scatter(
                x=osc_x_vals,
                y=osc_y_vals,
                mode="lines",
                name="power-law residual σ",
                line=dict(color="rgba(180, 185, 192, 0.42)", width=1.1),
                customdata=osc_dates,
                hovertemplate="<b>power-law residual σ</b>: %{y:.2f}σ<extra></extra>",
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
                    x=plot_x_model,
                    y=np.asarray(harmonic_curves[harmonic_count], dtype=float) / sigma_scale,
                    mode="lines",
                    name=f"DSI {harmonic_labels.get(harmonic_count, harmonic_count)}",
                    line=dict(
                        color=harmonic_colors.get(harmonic_count, "#ea3d2f"),
                        width=2.6 if harmonic_count == int(selected_harmonic_count) else 1.9,
                    ),
                    hoverinfo="skip",
                    visible="legendonly",
                )
            )
        if perrenod_curve is not None:
            perrenod_label = str(perrenod_curve.get("label", "DSI decayed"))
            perrenod_r2 = perrenod_curve.get("r2")
            perrenod_name = (
                f"{perrenod_label} R² {perrenod_r2:.2f}%"
                if perrenod_r2 is not None and np.isfinite(perrenod_r2)
                else perrenod_label
            )
            fig.add_trace(
                go.Scatter(
                    x=plot_x_model,
                    y=np.asarray(perrenod_curve["values"], dtype=float) / sigma_scale,
                    mode="lines",
                    name=perrenod_name,
                    line=dict(color="#f0b90b", width=3.0, dash="solid"),
                    hoverinfo="skip",
                )
            )
        fig.add_hline(y=0, line_width=1, line_color=pl_legend_color)
        fig.update_yaxes(
            type="linear",
            title_text="Sigma residual",
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

        for i in range(6):
            halving_days_val = osc_t1_age * (osc_lambda**i) * 365.25
            # Skip far-future synthetic halvings outside the rendered model horizon.
            if halving_days_val > float(view_max):
                continue
            hv_x = (
                halving_days_val
                if is_log_time
                else current_gen_date + pd.Timedelta(days=halving_days_val)
            )
            fig.add_vline(
                x=hv_x, line_width=1.5, line_dash="dash", line_color="#ea3d2f", opacity=0.8
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
        margin=dict(t=30, b=10, l=50, r=20),
        template=pl_template,
        font=dict(color=pl_text_color),
        legend=dict(
            orientation="h",
            y=1.02,
            x=0.5,
            xanchor="center",
            font=dict(size=14, color=pl_legend_color),
            bgcolor="rgba(0,0,0,0)",
            groupclick="togglegroup",
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
