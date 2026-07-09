"""COFER reserve-currency view."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.constants import COFER_CURRENCY_LABELS


def render_cofer_currency_share_view(
    cofer_df,
    selected_currencies,
    *,
    pl_template,
    pl_bg_color,
    pl_grid_color,
    pl_text_color,
    c_hover_bg,
    c_hover_text,
    c_border,
):
    selected_currencies = [
        currency for currency in selected_currencies if currency in cofer_df.columns
    ]
    if not selected_currencies:
        st.error("No COFER currencies selected.")
        return

    chart_df = cofer_df[selected_currencies].copy()
    latest_date = pd.Timestamp(chart_df.dropna(how="all").index.max())
    latest_values = chart_df.loc[latest_date].dropna().sort_values(ascending=False)

    st.markdown("### COFER reserve currency dominance")
    metric_columns = st.columns(min(len(latest_values), 4))
    for metric_col, (currency, value) in zip(metric_columns, latest_values.items()):
        metric_col.metric(COFER_CURRENCY_LABELS.get(currency, currency), f"{float(value):.2f}%")

    line_colors = {
        "BTC": "#f7931a",
        "USD": "#f0b90b",
        "EUR": "#3b82f6",
        "JPY": "#ef4444",
        "GBP": "#a855f7",
        "CNY": "#22c55e",
        "AUD": "#06b6d4",
        "CAD": "#f97316",
        "CHF": "#e5e7eb",
        "Other": "#94a3b8",
    }
    fig = go.Figure()
    for currency in selected_currencies:
        series = pd.to_numeric(chart_df[currency], errors="coerce")
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series,
                mode="lines",
                name=COFER_CURRENCY_LABELS.get(currency, currency),
                line=dict(color=line_colors.get(currency), width=2.3),
                connectgaps=False,
                hovertemplate="%{x|%Y-%m-%d}<br>%{fullData.name}: %{y:.2f}%<extra></extra>",
            )
        )

    y_values = chart_df.to_numpy(dtype=float)
    y_values = y_values[np.isfinite(y_values)]
    y_max = max(5.0, float(np.nanmax(y_values)) if y_values.size else 100.0)
    fig.update_layout(
        template=pl_template,
        paper_bgcolor=pl_bg_color,
        plot_bgcolor=pl_bg_color,
        font=dict(color=pl_text_color),
        height=620,
        margin=dict(l=20, r=20, t=32, b=20),
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"
        ),
        xaxis=dict(title="", showgrid=True, gridcolor=pl_grid_color, zeroline=False),
        yaxis=dict(
            title="Percent of world FX reserves",
            ticksuffix="%",
            range=[0, min(100.0, y_max * 1.08)],
            showgrid=True,
            gridcolor=pl_grid_color,
            zeroline=False,
        ),
        hoverlabel=dict(bgcolor=c_hover_bg, font_color=c_hover_text, bordercolor=c_border),
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": True,
            "modeBarButtons": [
                ["toImage"],
                ["zoom2d", "pan2d", "toggleSpikelines"],
                ["zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
            ],
        },
    )
    st.caption(f"Latest COFER quarter: {latest_date.year}-Q{latest_date.quarter}")
