import numpy as np
import streamlit as st


def _kpi_card(col, label, value, delta=None, d_color=None):
    delta_html = (
        f"<div class='metric-delta' style='color:{d_color}'>{delta}</div>"
        if delta
        else "<div class='metric-delta' style='visibility:hidden;'>-</div>"
    )
    col.markdown(
        f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div>{delta_html}</div>",
        unsafe_allow_html=True,
    )


def _format_money(value, currency_prefix, currency_suffix, currency_decimals):
    return f"{currency_prefix}{value:,.{currency_decimals}f}{currency_suffix}"


def render_model_kpis(
    df_display, a_active, b_active, p97_5, currency_prefix, currency_suffix, currency_decimals
):
    l_p, l_f = df_display["Close"].iloc[-1], df_display["Fair"].iloc[-1]
    l_p_display, l_f_display = df_display["CloseDisplay"].iloc[-1], df_display["FairDisplay"].iloc[-1]
    diff = ((l_p - l_f) / l_f) * 100
    pot_target = 10 ** (a_active + b_active * np.log10(df_display["Days"].max()) + p97_5)
    pot = ((pot_target - l_p) / l_p) * 100
    below_fair_pct = float((df_display["Close"] < df_display["Fair"]).mean() * 100.0)
    above_fair_pct = float((df_display["Close"] > df_display["Fair"]).mean() * 100.0)

    k1, k2, k3, k4, k5 = st.columns(5)
    _kpi_card(k1, "BTC PRICE", _format_money(l_p_display, currency_prefix, currency_suffix, currency_decimals))
    _kpi_card(
        k2,
        "FAIR VALUE",
        _format_money(l_f_display, currency_prefix, currency_suffix, currency_decimals),
        f"{diff:+.1f}% from model",
        "#0ecb81" if diff < 0 else "#ea3d2f",
    )
    _kpi_card(k3, "GROWTH POTENTIAL", f"+{pot:,.0f}%", "to top band", "#f0b90b")
    _kpi_card(k4, "TIME BELOW FAIR", f"{below_fair_pct:.1f}%", "history share", "#0ecb81")
    _kpi_card(k5, "TIME ABOVE FAIR", f"{above_fair_pct:.1f}%", "history share", "#ea3d2f")
