import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.constants import MODE_LOGPERIODIC, MODE_POWERLAW, TIME_LOG


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
    if use_log_scale and model_x is not None and visible_start_day is not None:
        model_x_arr = np.asarray(model_x, dtype=float)
        visible_mask = model_x_arr >= float(visible_start_day)
        if np.any(visible_mask):
            fair_vals = fair_vals[visible_mask]

    lower_band = 10 ** (np.log10(fair_vals) + p2_5)
    upper_band = 10 ** (np.log10(fair_vals) + p97_5)

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
    range_start_day = max(1.0, float(df_display["Days"].min()))
    range_end_day = max(float(view_max), range_start_day + 1.0)

    start_year = int(df_display.index.min().year)
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
    chart_key,
):
    fig = go.Figure()
    tick_font = dict(color=pl_text_color, size=14, family="Arial Black, sans-serif")
    hover_label = dict(
        bgcolor=c_hover_bg, bordercolor=c_border, font=dict(color=c_hover_text, size=13)
    )
    is_log_time = time_scale == TIME_LOG

    if mode == MODE_POWERLAW:
        p2_5_series = 10 ** (np.log10(m_fair_display) + p2_5)
        p16_5_series = 10 ** (np.log10(m_fair_display) + p16_5)
        p83_5_series = 10 ** (np.log10(m_fair_display) + p83_5)
        p97_5_series = 10 ** (np.log10(m_fair_display) + p97_5)

        fig.add_trace(
            go.Scatter(
                x=plot_x_model,
                y=p97_5_series,
                mode="lines",
                line=dict(color="#ea3d2f", width=1.2, dash="dot"),
                name="97.5th percentile",
                legendrank=20,
                customdata=m_dates_str,
                hovertemplate=(
                    "<b>97.5th percentile</b>: "
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
                name="83.5th percentile",
                legendrank=30,
                customdata=m_dates_str,
                hovertemplate=(
                    "<b>83.5th percentile</b>: "
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
                name="16.5th percentile",
                legendrank=50,
                customdata=m_dates_str,
                hovertemplate=(
                    "<b>16.5th percentile</b>: "
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
                name="2.5th percentile",
                legendrank=60,
                customdata=m_dates_str,
                hovertemplate=(
                    "<b>2.5th percentile</b>: "
                    f"{currency_prefix}%{{y:,.{currency_decimals}f}}{currency_suffix}<extra></extra>"
                ),
            )
        )

        btc_hover = (
            (
                "📅 %{customdata}<br><b>Bitcoin price</b>: "
                f"{currency_prefix}%{{y:,.{currency_decimals}f}}{currency_suffix}<extra></extra>"
            )
            if is_log_time
            else (
                "<b>Bitcoin price</b>: "
                f"{currency_prefix}%{{y:,.{currency_decimals}f}}{currency_suffix}<extra></extra>"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=plot_x_main,
                y=df_display["CloseDisplay"],
                mode="lines",
                name="Bitcoin price",
                legendrank=10,
                line=dict(color=pl_btc_color, width=1.5),
                customdata=df_display.index.strftime("%d.%m.%Y"),
                hovertemplate=btc_hover,
            )
        )
        powerlaw_y_range = _resolve_powerlaw_y_range(
            df_display,
            m_fair_display,
            p2_5,
            p97_5,
            use_log_scale=(price_scale == TIME_LOG),
            model_x=plot_x_model if is_log_time else None,
            visible_start_day=max(1.0, float(df_display["Days"].min())) if is_log_time else None,
        )
        fig.update_yaxes(
            type="log" if price_scale == TIME_LOG else "linear",
            range=powerlaw_y_range,
            gridcolor=pl_grid_color,
            tickfont=tick_font,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=plot_x_osc,
                y=df_display["Res"],
                mode="lines",
                name="LogPeriodic",
                line=dict(color="#0ecb81", width=1.2),
                customdata=df_display.index.strftime("%d.%m.%Y"),
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
        )
    else:
        fig.update_xaxes(
            type="date",
            gridcolor=pl_grid_color,
            tickfont=tick_font,
            range=[df_display.index.min(), m_dates[-1]],
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
        fig, width="stretch", theme=None, config={"displayModeBar": False}, key=chart_key
    )
