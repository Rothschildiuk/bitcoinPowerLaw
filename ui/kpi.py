import numpy as np
import streamlit as st

from core.utils import evaluate_powerlaw_values


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


def calculate_powerlaw_band_shares(df_display, p2_5, p16_5, p83_5, p97_5):
    residuals = np.asarray(df_display["Res"], dtype=float)
    valid_mask = np.isfinite(residuals)
    valid_residuals = residuals[valid_mask]
    if valid_residuals.size == 0:
        return {
            "plus_two_to_plus_one": 0.0,
            "plus_one_to_powerlaw": 0.0,
            "powerlaw_to_minus_one": 0.0,
            "minus_one_to_minus_two": 0.0,
        }

    p2_5, p16_5, p83_5, p97_5 = (float(p2_5), float(p16_5), float(p83_5), float(p97_5))
    thresholds = np.array([p2_5, p16_5, p83_5, p97_5], dtype=float)
    if not np.all(np.isfinite(thresholds)):
        return {
            "plus_two_to_plus_one": 0.0,
            "plus_one_to_powerlaw": 0.0,
            "powerlaw_to_minus_one": 0.0,
            "minus_one_to_minus_two": 0.0,
        }

    total = float(valid_residuals.size)
    return {
        "plus_two_to_plus_one": float(
            np.count_nonzero((valid_residuals >= p83_5) & (valid_residuals <= p97_5))
            / total
            * 100.0
        ),
        "plus_one_to_powerlaw": float(
            np.count_nonzero((valid_residuals >= 0.0) & (valid_residuals < p83_5)) / total * 100.0
        ),
        "powerlaw_to_minus_one": float(
            np.count_nonzero((valid_residuals >= p16_5) & (valid_residuals < 0.0)) / total * 100.0
        ),
        "minus_one_to_minus_two": float(
            np.count_nonzero((valid_residuals >= p2_5) & (valid_residuals < p16_5)) / total * 100.0
        ),
    }


def render_model_kpis(
    df_display,
    a_active,
    b_active,
    p2_5,
    p16_5,
    p83_5,
    p97_5,
    currency_prefix,
    currency_suffix,
    currency_decimals,
    target_series_name,
    target_series_unit,
):
    l_p, l_f = df_display["Close"].iloc[-1], df_display["Fair"].iloc[-1]
    l_p_display, l_f_display = (
        df_display["CloseDisplay"].iloc[-1],
        df_display["FairDisplay"].iloc[-1],
    )
    diff = ((l_p - l_f) / l_f) * 100
    pot_target, _, _ = evaluate_powerlaw_values(
        np.array([np.log10(df_display["Days"].max())]),
        a_active + p97_5,
        b_active,
    )
    pot_target = float(pot_target[0])
    pot = ((pot_target - l_p) / l_p) * 100
    band_shares = calculate_powerlaw_band_shares(df_display, p2_5, p16_5, p83_5, p97_5)

    k1, k2, k3, z1, z2, z3, z4 = st.columns([1.25, 1.25, 1.25, 1, 1, 1, 1])
    _kpi_card(
        k1,
        f"{target_series_name.upper()}",
        _format_money(l_p_display, currency_prefix, currency_suffix, currency_decimals),
        target_series_unit,
        "#9ba3af",
    )
    _kpi_card(
        k2,
        "FAIR VALUE",
        _format_money(l_f_display, currency_prefix, currency_suffix, currency_decimals),
        f"{diff:+.1f}% from model",
        "#0ecb81" if diff < 0 else "#ea3d2f",
    )
    _kpi_card(k3, "GROWTH POTENTIAL", f"+{pot:,.0f}%", "to top band", "#f0b90b")

    _kpi_card(
        z1,
        "TIME -2σ TO -1σ",
        f"{band_shares['minus_one_to_minus_two']:.1f}%",
        "history share",
        "#1199d6",
    )
    _kpi_card(
        z2,
        "TIME -1σ TO POWER LAW",
        f"{band_shares['powerlaw_to_minus_one']:.1f}%",
        "history share",
        "#0ecb81",
    )
    _kpi_card(
        z3,
        "TIME POWER LAW TO +1σ",
        f"{band_shares['plus_one_to_powerlaw']:.1f}%",
        "history share",
        "#f0b90b",
    )
    _kpi_card(
        z4,
        "TIME +1σ TO +2σ",
        f"{band_shares['plus_two_to_plus_one']:.1f}%",
        "history share",
        "#ea3d2f",
    )
