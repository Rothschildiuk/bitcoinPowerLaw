import numpy as np
import streamlit as st

from core.utils import evaluate_powerlaw_values

SIGMA_LEVELS = tuple(float(value) for value in np.arange(-2.0, 2.0 + 0.25, 0.25))


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


def _empty_band_shares():
    return [
        {
            "lower_level": lower_level,
            "upper_level": upper_level,
            "label": f"{_format_sigma_level(lower_level)} to {_format_sigma_level(upper_level)}",
            "share": 0.0,
        }
        for lower_level, upper_level in zip(SIGMA_LEVELS[:-1], SIGMA_LEVELS[1:])
    ]


def _format_sigma_level(value):
    if float(value) == 0.0:
        return "Power Law"
    return f"{value:+g}σ"


def _resolve_sigma_offsets(p2_5, p16_5, p83_5, p97_5):
    scenario_levels = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=float)
    scenario_offsets = np.array([p2_5, p16_5, 0.0, p83_5, p97_5], dtype=float)
    if not np.all(np.isfinite(scenario_offsets)):
        return None
    return {
        level: float(np.interp(level, scenario_levels, scenario_offsets)) for level in SIGMA_LEVELS
    }


def calculate_powerlaw_band_shares(df_display, p2_5, p16_5, p83_5, p97_5):
    residuals = np.asarray(df_display["Res"], dtype=float)
    valid_mask = np.isfinite(residuals)
    valid_residuals = residuals[valid_mask]
    if valid_residuals.size == 0:
        return _empty_band_shares()

    p2_5, p16_5, p83_5, p97_5 = (float(p2_5), float(p16_5), float(p83_5), float(p97_5))
    sigma_offsets = _resolve_sigma_offsets(p2_5, p16_5, p83_5, p97_5)
    if sigma_offsets is None:
        return _empty_band_shares()

    total = float(valid_residuals.size)
    band_shares = []
    for lower_level, upper_level in zip(SIGMA_LEVELS[:-1], SIGMA_LEVELS[1:]):
        lower_offset = sigma_offsets[lower_level]
        upper_offset = sigma_offsets[upper_level]
        if upper_level == SIGMA_LEVELS[-1]:
            count = np.count_nonzero(
                (valid_residuals >= lower_offset) & (valid_residuals <= upper_offset)
            )
        else:
            count = np.count_nonzero(
                (valid_residuals >= lower_offset) & (valid_residuals < upper_offset)
            )
        band_shares.append(
            {
                "lower_level": lower_level,
                "upper_level": upper_level,
                "label": f"{_format_sigma_level(lower_level)} to {_format_sigma_level(upper_level)}",
                "share": float(count / total * 100.0),
            }
        )
    return band_shares


def _render_sigma_band_chart(band_shares):
    max_share = max((float(band["share"]) for band in band_shares), default=0.0)
    scale_max = max(max_share, 1.0)
    bars = "".join(
        (
            "<div class='sigma-bar-item'>"
            f"<div class='sigma-bar-value'>{band['share']:.1f}%</div>"
            "<div class='sigma-bar-track'>"
            f"<div class='sigma-bar-fill' style='height:{(float(band['share']) / scale_max) * 100.0:.1f}%;'></div>"
            "</div>"
            f"<div class='sigma-bar-label'>{band['label']}</div>"
            "</div>"
        )
        for band in band_shares
    )
    st.markdown(
        (
            "<div class='sigma-chart-card'>"
            "<div class='sigma-chart-header'>"
            "<span>Sigma band</span>"
            "<span>History share</span>"
            "</div>"
            f"<div class='sigma-bars'>{bars}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


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

    k1, k2, k3 = st.columns(3)
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

    _render_sigma_band_chart(band_shares)
