import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.constants import MODE_LOGPERIODIC, MODE_POWERLAW, TIME_LOG


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
    m_fair_usd,
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
    chart_key,
):
    fig = go.Figure()
    tick_font = dict(color=pl_text_color, size=14, family="Arial Black, sans-serif")
    hover_label = dict(
        bgcolor=c_hover_bg, bordercolor=c_border, font=dict(color=c_hover_text, size=13)
    )
    is_log_time = time_scale == TIME_LOG

    if mode == MODE_POWERLAW:
        fig.add_trace(
            go.Scatter(
                x=plot_x_model,
                y=10 ** (np.log10(m_fair_usd) + p97_5),
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=plot_x_model,
                y=10 ** (np.log10(m_fair_usd) + p16_5),
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=plot_x_model,
                y=10 ** (np.log10(m_fair_usd) + p2_5),
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(14, 203, 129, 0.15)",
                name="Accumulation",
                customdata=m_dates_str,
                hovertemplate="<b>Accumulation</b>: $%{y:,.0f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=plot_x_model,
                y=m_fair_usd,
                mode="lines",
                line=dict(color="#f0b90b", width=1.5, dash="dash"),
                name="Fair Value",
                customdata=m_dates_str,
                hovertemplate="<b>Fair Value</b>: $%{y:,.0f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=plot_x_model,
                y=10 ** (np.log10(m_fair_usd) + p83_5),
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(234, 61, 47, 0.15)",
                name="Bubble",
                customdata=m_dates_str,
                hovertemplate="<b>Bubble</b>: $%{y:,.0f}<extra></extra>",
            )
        )

        btc_hover = (
            "📅 %{customdata}<br><b>BTC Price</b>: $%{y:,.0f}<extra></extra>"
            if is_log_time
            else "<b>BTC Price</b>: $%{y:,.0f}<extra></extra>"
        )
        fig.add_trace(
            go.Scatter(
                x=plot_x_main,
                y=df_display["Close"],
                mode="lines",
                name="BTC Price",
                line=dict(color=pl_btc_color, width=1.5),
                customdata=df_display.index.strftime("%d.%m.%Y"),
                hovertemplate=btc_hover,
            )
        )
        fig.update_yaxes(
            type="log" if price_scale == TIME_LOG else "linear",
            range=(
                [np.log10(0.01), np.log10(df_display["Close"].max() * 8)]
                if price_scale == TIME_LOG
                else None
            ),
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

    t_vals = [
        (pd.Timestamp(f"{y}-01-01") - current_gen_date).days
        for y in range(current_gen_date.year + 1, 2036)
        if (pd.Timestamp(f"{y}-01-01") - current_gen_date).days > 0
    ]
    t_text = [
        str(y)
        for y in range(current_gen_date.year + 1, 2036)
        if (pd.Timestamp(f"{y}-01-01") - current_gen_date).days > 0
    ]
    if is_log_time:
        x_range = [np.log10(max(1.0, view_max / 1000.0)), np.log10(view_max)]
        if t_vals:
            x_range = [np.log10(t_vals[0]), np.log10(view_max)]
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
