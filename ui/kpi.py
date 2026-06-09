import numpy as np
import pandas as pd
import streamlit as st

from core.constants import KEY_SIGMA_BAND_HISTORY_YEARS
from core.utils import evaluate_powerlaw_values, interpolate_sigma_level_from_log_offset

SIGMA_STEP = 0.125
SIGMA_LEVELS = tuple(float(value) for value in np.arange(-2.0, 2.0 + SIGMA_STEP, SIGMA_STEP))
SIGMA_BAND_HISTORY_ALL = 0


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


def resolve_sigma_band_history_max_years(date_values):
    date_index = pd.to_datetime(date_values, errors="coerce")
    valid_dates = date_index[~pd.isna(date_index)]
    if len(valid_dates) == 0:
        return 15

    history_days = max((valid_dates.max() - valid_dates.min()).days, 0)
    return max(1, int(np.ceil(history_days / 365.25)))


def resolve_sigma_band_history_selection(history_years, max_history_years):
    selected_years = int(history_years or SIGMA_BAND_HISTORY_ALL)
    max_history_years = max(1, int(max_history_years))
    if selected_years < SIGMA_BAND_HISTORY_ALL or selected_years > max_history_years:
        return max_history_years
    return selected_years


def format_sigma_band_history_option(years):
    years = int(years)
    if years == SIGMA_BAND_HISTORY_ALL:
        return "All history"
    if years == 1:
        return "Last 1 year"
    return f"Last {years} years"


def _resolve_sigma_band_history_max_years(df_display):
    if df_display.empty:
        return 1

    return resolve_sigma_band_history_max_years(df_display.index)


def filter_sigma_band_history(df_display, history_years):
    history_years = int(history_years or SIGMA_BAND_HISTORY_ALL)
    if history_years <= 0 or df_display.empty:
        return df_display

    date_index = pd.to_datetime(df_display.index, errors="coerce")
    valid_mask = ~pd.isna(date_index)
    if not np.any(valid_mask):
        return df_display

    latest_date = date_index[valid_mask].max()
    start_date = latest_date - pd.DateOffset(years=history_years)
    return df_display.loc[np.asarray(date_index >= start_date)]


def _empty_band_shares():
    return [
        {
            "lower_level": None,
            "upper_level": SIGMA_LEVELS[0],
            "label": f"< {_format_sigma_level(SIGMA_LEVELS[0])}",
            "compact_label": _format_compact_sigma_band_label(None, SIGMA_LEVELS[0]),
            "share": 0.0,
        },
        *[
            {
                "lower_level": lower_level,
                "upper_level": upper_level,
                "label": f"{_format_sigma_level(lower_level)} to {_format_sigma_level(upper_level)}",
                "compact_label": _format_compact_sigma_band_label(lower_level, upper_level),
                "share": 0.0,
            }
            for lower_level, upper_level in zip(SIGMA_LEVELS[:-1], SIGMA_LEVELS[1:])
        ],
        {
            "lower_level": SIGMA_LEVELS[-1],
            "upper_level": None,
            "label": f"> {_format_sigma_level(SIGMA_LEVELS[-1])}",
            "compact_label": _format_compact_sigma_band_label(SIGMA_LEVELS[-1], None),
            "share": 0.0,
        },
    ]


def _format_sigma_level(value):
    if float(value) == 0.0:
        return "Power Law"
    return f"{value:+g}σ"


def _format_compact_sigma_level(value):
    if float(value) == 0.0:
        return "0"
    return f"{value:+g}"


def _format_compact_sigma_band_label(lower_level, upper_level):
    if lower_level is None:
        return f"< {_format_compact_sigma_level(upper_level)}"
    if upper_level is None:
        return f"> {_format_compact_sigma_level(lower_level)}"
    return (
        f"{_format_compact_sigma_level(lower_level)}<br>{_format_compact_sigma_level(upper_level)}"
    )


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
    lower_edge = sigma_offsets[SIGMA_LEVELS[0]]
    upper_edge = sigma_offsets[SIGMA_LEVELS[-1]]
    band_shares.append(
        {
            "lower_level": None,
            "upper_level": SIGMA_LEVELS[0],
            "label": f"< {_format_sigma_level(SIGMA_LEVELS[0])}",
            "compact_label": _format_compact_sigma_band_label(None, SIGMA_LEVELS[0]),
            "share": float(np.count_nonzero(valid_residuals < lower_edge) / total * 100.0),
        }
    )
    for lower_level, upper_level in zip(SIGMA_LEVELS[:-1], SIGMA_LEVELS[1:]):
        lower_offset = sigma_offsets[lower_level]
        upper_offset = sigma_offsets[upper_level]
        count = np.count_nonzero(
            (valid_residuals >= lower_offset) & (valid_residuals < upper_offset)
        )
        band_shares.append(
            {
                "lower_level": lower_level,
                "upper_level": upper_level,
                "label": f"{_format_sigma_level(lower_level)} to {_format_sigma_level(upper_level)}",
                "compact_label": _format_compact_sigma_band_label(lower_level, upper_level),
                "share": float(count / total * 100.0),
            }
        )
    band_shares.append(
        {
            "lower_level": SIGMA_LEVELS[-1],
            "upper_level": None,
            "label": f"> {_format_sigma_level(SIGMA_LEVELS[-1])}",
            "compact_label": _format_compact_sigma_band_label(SIGMA_LEVELS[-1], None),
            "share": float(np.count_nonzero(valid_residuals >= upper_edge) / total * 100.0),
        }
    )
    return band_shares


def calculate_current_powerlaw_sigma_level(df_display, p2_5, p16_5, p83_5, p97_5):
    residuals = np.asarray(df_display["Res"], dtype=float)
    valid_residuals = residuals[np.isfinite(residuals)]
    if valid_residuals.size == 0:
        return None

    percentile_offsets = (float(p2_5), float(p16_5), float(p83_5), float(p97_5))
    if not np.all(np.isfinite(percentile_offsets)):
        return None

    return interpolate_sigma_level_from_log_offset(valid_residuals[-1], percentile_offsets)


def _sigma_band_contains_level(band, sigma_level):
    if sigma_level is None or not np.isfinite(float(sigma_level)):
        return False

    sigma_level = float(sigma_level)
    lower_level = band["lower_level"]
    upper_level = band["upper_level"]
    if lower_level is None:
        return sigma_level < float(upper_level)
    if upper_level is None:
        return sigma_level >= float(lower_level)
    return float(lower_level) <= sigma_level < float(upper_level)


def _render_sigma_band_chart(
    band_shares,
    current_sigma_level=None,
    history_label="History share",
):
    max_share = max((float(band["share"]) for band in band_shares), default=0.0)
    scale_max = max(max_share, 1.0)

    def render_bar(band):
        item_class = "sigma-bar-item"
        aria_label = ""
        if _sigma_band_contains_level(band, current_sigma_level):
            item_class = "sigma-bar-item sigma-bar-item-current"
            aria_label = " aria-label='Current sigma band'"
        return (
            f"<div class='{item_class}'{aria_label}>"
            f"<div class='sigma-bar-value'>{band['share']:.1f}%</div>"
            "<div class='sigma-bar-track'>"
            f"<div class='sigma-bar-fill' style='height:{(float(band['share']) / scale_max) * 100.0:.1f}%;'></div>"
            "</div>"
            f"<div class='sigma-bar-label' title='{band['label']}'>{band.get('compact_label', band['label'])}</div>"
            "</div>"
        )

    bars = "".join(render_bar(band) for band in band_shares)
    st.markdown(
        (
            "<div class='sigma-chart-card'>"
            "<div class='sigma-chart-header'>"
            "<span>Sigma band</span>"
            f"<span>{history_label}</span>"
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
    logperiodic_stats_rows=None,
    perrenod_stats_rows=None,
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

    if logperiodic_stats_rows or perrenod_stats_rows:
        from core.oscillator import render_logperiodic_regression_stats_table

        render_logperiodic_regression_stats_table(logperiodic_stats_rows, perrenod_stats_rows)

    max_history_years = resolve_sigma_band_history_max_years(df_display.index)
    selected_history_years = resolve_sigma_band_history_selection(
        st.session_state.get(KEY_SIGMA_BAND_HISTORY_YEARS, SIGMA_BAND_HISTORY_ALL),
        max_history_years,
    )
    band_df = filter_sigma_band_history(df_display, selected_history_years)
    band_shares = calculate_powerlaw_band_shares(band_df, p2_5, p16_5, p83_5, p97_5)
    current_sigma_level = calculate_current_powerlaw_sigma_level(
        df_display, p2_5, p16_5, p83_5, p97_5
    )
    history_label = format_sigma_band_history_option(selected_history_years)
    _render_sigma_band_chart(band_shares, current_sigma_level, history_label)
