import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.constants import (
    BAND_METHOD_GAUSSIAN,
    MODE_LOGPERIODIC,
    MODE_POWERLAW,
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
MODEL_FORWARD_YEARS = 5


def _resolve_time_axis_start_date(df_display, padding_days=TIME_AXIS_LEADING_PADDING_DAYS):
    first_data_date = pd.Timestamp(df_display.index.min())
    return first_data_date - pd.Timedelta(days=int(padding_days))


def _resolve_model_view_max(df_display, current_gen_date, forward_years=MODEL_FORWARD_YEARS):
    latest_data_date = pd.Timestamp(df_display.index.max()).normalize()
    horizon_end_date = latest_data_date + pd.DateOffset(years=int(forward_years))
    return max(1.0, float((horizon_end_date - current_gen_date).days))


def _resolve_powerlaw_y_range(
    df_display,
    m_fair_display,
    p2_5,
    p97_5,
    use_log_scale,
    model_x=None,
    visible_start_day=None,
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

    lower_band, _, _ = evaluate_powerlaw_values(np.log10(fair_vals), p2_5, 1.0)
    upper_band, _, _ = evaluate_powerlaw_values(np.log10(fair_vals), p97_5, 1.0)

    candidate = np.concatenate([btc_vals, lower_band, upper_band])
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
    m_osc_y,
    p2_5,
    p16_5,
    p83_5,
    p97_5,
    band_method,
    osc_t1_age,
    osc_lambda,
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
    fig = go.Figure()
    tick_font = dict(color=pl_text_color, size=14, family="Arial Black, sans-serif")
    hover_label = dict(
        bgcolor=c_hover_bg, bordercolor=c_border, font=dict(color=c_hover_text, size=13)
    )
    is_log_time = time_scale == TIME_LOG

    if mode == MODE_POWERLAW:
        is_gaussian_band = band_method == BAND_METHOD_GAUSSIAN
        p97_5_name = "+2σ" if is_gaussian_band else "+2σ (97.5th percentile)"
        p83_5_name = "+1σ" if is_gaussian_band else "+1σ (83.5th percentile)"
        p16_5_name = "-1σ" if is_gaussian_band else "-1σ (16.5th percentile)"
        p2_5_name = "-2σ" if is_gaussian_band else "-2σ (2.5th percentile)"

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
                legendrank=10,
                line=dict(color=pl_btc_color, width=1.5),
                customdata=df_display.index.strftime("%d.%m.%Y"),
                hovertemplate=btc_hover,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=plot_x_model,
                y=p97_5_series,
                mode="lines",
                line=dict(color="#ea3d2f", width=1.2, dash="dot"),
                name=p97_5_name,
                legendrank=20,
                customdata=m_dates_str,
                hovertemplate=(
                    f"<b>{p97_5_name}</b>: "
                    f"{currency_prefix}%{{y:,.{currency_decimals}f}}{currency_suffix}<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=plot_x_model,
                y=p83_5_series,
                mode="lines",
                line=dict(color="#1199d6", width=1.2, dash="dot"),
                name=p83_5_name,
                legendrank=30,
                customdata=m_dates_str,
                hovertemplate=(
                    f"<b>{p83_5_name}</b>: "
                    f"{currency_prefix}%{{y:,.{currency_decimals}f}}{currency_suffix}<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=plot_x_model,
                y=m_fair_display,
                mode="lines",
                line=dict(color="#f0b90b", width=1.8),
                name="Power regression",
                legendrank=40,
                customdata=m_dates_str,
                hovertemplate=(
                    "<b>Power regression</b>: "
                    f"{currency_prefix}%{{y:,.{currency_decimals}f}}{currency_suffix}<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=plot_x_model,
                y=p16_5_series,
                mode="lines",
                line=dict(color="#1199d6", width=1.2, dash="dot"),
                name=p16_5_name,
                legendrank=50,
                customdata=m_dates_str,
                hovertemplate=(
                    f"<b>{p16_5_name}</b>: "
                    f"{currency_prefix}%{{y:,.{currency_decimals}f}}{currency_suffix}<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=plot_x_model,
                y=p2_5_series,
                mode="lines",
                line=dict(color="#ea3d2f", width=1.2, dash="dot"),
                name=p2_5_name,
                legendrank=60,
                customdata=m_dates_str,
                hovertemplate=(
                    f"<b>{p2_5_name}</b>: "
                    f"{currency_prefix}%{{y:,.{currency_decimals}f}}{currency_suffix}<extra></extra>"
                ),
            )
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
        osc_y_vals = df_display["Res"].to_numpy(dtype=float)[osc_mask]
        osc_dates = df_display.index.strftime("%d.%m.%Y").to_numpy()[osc_mask]

        fig.add_trace(
            go.Scatter(
                x=osc_x_vals,
                y=osc_y_vals,
                mode="lines",
                name="LogPeriodic",
                line=dict(color="#0ecb81", width=1.2),
                customdata=osc_dates,
                hovertemplate="<b>LogPeriodic</b>: %{y:.3f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=plot_x_model,
                y=m_osc_y,
                mode="lines",
                name="LogPeriodic Model",
                line=dict(color="#ea3d2f", width=2),
                hoverinfo="skip",
            )
        )
        fig.add_hline(y=0, line_width=1, line_color=pl_legend_color)
        fig.update_yaxes(type="linear", gridcolor=pl_grid_color, tickfont=tick_font)

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
        config={"displayModeBar": True},
        key=chart_key,
    )
