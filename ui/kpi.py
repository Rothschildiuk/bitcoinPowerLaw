from html import escape

import numpy as np
import pandas as pd
import streamlit as st

from core.constants import KEY_SIGMA_BAND_HISTORY_RANGE_PCT
from core.utils import evaluate_powerlaw_values, interpolate_sigma_level_from_log_offset

SIGMA_STEP = 0.25
SIGMA_HALF_STEP = SIGMA_STEP / 2.0
SIGMA_LEVELS = tuple(
    float(value) for value in np.arange(-2.25, 2.25 + SIGMA_STEP, SIGMA_STEP)
)
SIGMA_BAND_HISTORY_ALL = 0
SIGMA_BAND_HISTORY_PERCENT_MIN = 0
SIGMA_BAND_HISTORY_PERCENT_MAX = 100
SIGMA_BAND_HISTORY_PERCENT_RANGE_DEFAULT = (
    SIGMA_BAND_HISTORY_PERCENT_MIN,
    SIGMA_BAND_HISTORY_PERCENT_MAX,
)


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


def resolve_sigma_band_history_percent_range(history_percent_range):
    try:
        start_pct, end_pct = history_percent_range
    except (TypeError, ValueError):
        return SIGMA_BAND_HISTORY_PERCENT_RANGE_DEFAULT

    def coerce_percent(value, fallback):
        try:
            percent = float(value)
        except (TypeError, ValueError):
            return fallback
        if not np.isfinite(percent):
            return fallback
        return int(
            min(
                max(round(percent), SIGMA_BAND_HISTORY_PERCENT_MIN),
                SIGMA_BAND_HISTORY_PERCENT_MAX,
            )
        )

    start_pct = coerce_percent(start_pct, SIGMA_BAND_HISTORY_PERCENT_MIN)
    end_pct = coerce_percent(end_pct, SIGMA_BAND_HISTORY_PERCENT_MAX)
    if start_pct > end_pct:
        start_pct, end_pct = end_pct, start_pct
    return (start_pct, end_pct)


def format_sigma_band_history_percent_range(history_percent_range):
    start_pct, end_pct = resolve_sigma_band_history_percent_range(history_percent_range)
    return f"{start_pct}-{end_pct}% history"


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


def filter_sigma_band_history_percent_range(df_display, history_percent_range):
    if df_display.empty:
        return df_display

    start_pct, end_pct = resolve_sigma_band_history_percent_range(history_percent_range)
    if start_pct == SIGMA_BAND_HISTORY_PERCENT_MIN and end_pct == SIGMA_BAND_HISTORY_PERCENT_MAX:
        return df_display

    date_index = pd.to_datetime(df_display.index, errors="coerce")
    valid_mask = ~pd.isna(date_index)
    if not np.any(valid_mask):
        return df_display

    valid_dates = date_index[valid_mask]
    earliest_date = valid_dates.min()
    latest_date = valid_dates.max()
    history_span = latest_date - earliest_date
    if history_span <= pd.Timedelta(0):
        return df_display.loc[np.asarray(valid_mask)]

    start_date = earliest_date + history_span * (start_pct / 100.0)
    end_date = earliest_date + history_span * (end_pct / 100.0)
    selected_mask = np.asarray(valid_mask & (date_index >= start_date) & (date_index <= end_date))
    if np.any(selected_mask):
        return df_display.loc[selected_mask]

    midpoint_date = earliest_date + history_span * ((start_pct + end_pct) / 200.0)
    valid_positions = np.flatnonzero(np.asarray(valid_mask))
    nearest_valid_position = valid_positions[int(np.argmin(np.abs(valid_dates - midpoint_date)))]
    fallback_mask = np.zeros(len(df_display), dtype=bool)
    fallback_mask[nearest_valid_position] = True
    return df_display.loc[fallback_mask]


def _empty_band_shares():
    band_shares = []
    for center_level in SIGMA_LEVELS:
        lower_level, upper_level = _resolve_sigma_band_edges(center_level)
        band_shares.append(
            {
                "center_level": center_level,
                "lower_level": lower_level,
                "upper_level": upper_level,
                "label": _format_sigma_band_label(lower_level, upper_level),
                "compact_label": _format_compact_sigma_level(center_level),
                "share": 0.0,
            }
        )
    return band_shares


def _resolve_sigma_band_edges(center_level):
    lower_level = float(center_level) - SIGMA_HALF_STEP
    upper_level = float(center_level) + SIGMA_HALF_STEP
    if np.isclose(center_level, SIGMA_LEVELS[0]):
        lower_level = None
    if np.isclose(center_level, SIGMA_LEVELS[-1]):
        upper_level = None
    return lower_level, upper_level


def _format_sigma_band_label(lower_level, upper_level):
    if lower_level is None:
        return f"< {_format_sigma_level(upper_level)}"
    if upper_level is None:
        return f"> {_format_sigma_level(lower_level)}"
    return f"{_format_sigma_level(lower_level)} to {_format_sigma_level(upper_level)}"


def _interpolate_sigma_offsets(levels, p2_5, p16_5, p83_5, p97_5):
    scenario_levels = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=float)
    scenario_offsets = np.array([p2_5, p16_5, 0.0, p83_5, p97_5], dtype=float)
    if not np.all(np.isfinite(scenario_offsets)):
        return None

    def interpolate_level(level):
        level = float(level)
        if level <= scenario_levels[0]:
            x0, x1 = scenario_levels[0], scenario_levels[1]
            y0, y1 = scenario_offsets[0], scenario_offsets[1]
        elif level >= scenario_levels[-1]:
            x0, x1 = scenario_levels[-2], scenario_levels[-1]
            y0, y1 = scenario_offsets[-2], scenario_offsets[-1]
        else:
            return float(np.interp(level, scenario_levels, scenario_offsets))

        if np.isclose(x0, x1):
            return float(y0)
        return float(y0 + ((level - x0) / (x1 - x0)) * (y1 - y0))

    return {float(level): interpolate_level(level) for level in levels}


def _resolve_sigma_offsets(p2_5, p16_5, p83_5, p97_5):
    edge_levels = []
    for center_level in SIGMA_LEVELS:
        lower_level, upper_level = _resolve_sigma_band_edges(center_level)
        if lower_level is not None:
            edge_levels.append(lower_level)
        if upper_level is not None:
            edge_levels.append(upper_level)
    return _interpolate_sigma_offsets(
        edge_levels,
        p2_5,
        p16_5,
        p83_5,
        p97_5,
    )


def _format_sigma_level(value):
    if float(value) == 0.0:
        return "Power Law"
    return f"{value:+g}σ"


def _format_compact_sigma_level(value):
    if float(value) == 0.0:
        return "0"
    return f"{value:+g}"


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

    for center_level in SIGMA_LEVELS:
        lower_level, upper_level = _resolve_sigma_band_edges(center_level)
        if lower_level is None:
            upper_offset = sigma_offsets[upper_level]
            count = np.count_nonzero(valid_residuals < upper_offset)
        elif upper_level is None:
            lower_offset = sigma_offsets[lower_level]
            count = np.count_nonzero(valid_residuals >= lower_offset)
        else:
            lower_offset = sigma_offsets[lower_level]
            upper_offset = sigma_offsets[upper_level]
            count = np.count_nonzero(
                (valid_residuals >= lower_offset) & (valid_residuals < upper_offset)
            )
        band_shares.append(
            {
                "center_level": center_level,
                "lower_level": lower_level,
                "upper_level": upper_level,
                "label": _format_sigma_band_label(lower_level, upper_level),
                "compact_label": _format_compact_sigma_level(center_level),
                "share": float(count / total * 100.0),
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


def calculate_negative_two_sigma_monthly_growth(
    df_display,
    a_active,
    b_active,
    p2_5,
    today=None,
):
    if df_display.empty or "AbsDays" not in df_display:
        return 0.0

    date_index = pd.to_datetime(df_display.index, errors="coerce")
    valid_dates = date_index[~pd.isna(date_index)]
    valid_abs_days = np.asarray(df_display["AbsDays"], dtype=float)
    valid_abs_days = valid_abs_days[np.isfinite(valid_abs_days)]
    if len(valid_dates) == 0 or valid_abs_days.size == 0:
        return 0.0

    if today is None:
        today_date = pd.Timestamp.utcnow().tz_localize(None).normalize()
    else:
        today_date = pd.Timestamp(today)
        if today_date.tzinfo is not None:
            today_date = today_date.tz_localize(None)
        today_date = today_date.normalize()

    first_date = pd.Timestamp(valid_dates.min()).normalize()
    first_abs_day = float(valid_abs_days[0])
    start_date = today_date - pd.Timedelta(days=15)
    end_date = today_date + pd.Timedelta(days=15)
    start_days = first_abs_day + float((start_date - first_date).days)
    end_days = first_abs_day + float((end_date - first_date).days)
    if not np.isfinite(start_days) or not np.isfinite(end_days) or start_days <= 0.0:
        return 0.0
    end_days = max(end_days, start_days + 1.0)

    current_band_value, _, _ = evaluate_powerlaw_values(
        np.array([np.log10(start_days)]),
        float(a_active) + float(p2_5),
        float(b_active),
    )
    next_month_band_value, _, _ = evaluate_powerlaw_values(
        np.array([np.log10(end_days)]),
        float(a_active) + float(p2_5),
        float(b_active),
    )
    current_band_value = float(current_band_value[0])
    next_month_band_value = float(next_month_band_value[0])
    if not np.isfinite(current_band_value) or current_band_value <= 0.0:
        return 0.0

    monthly_growth = next_month_band_value - current_band_value
    if not np.isfinite(monthly_growth):
        return 0.0
    return max(0.0, float(monthly_growth))


def _resolve_display_conversion_rate(df_display):
    if "Fair" not in df_display or "FairDisplay" not in df_display:
        return 1.0

    fair = float(df_display["Fair"].iloc[-1])
    fair_display = float(df_display["FairDisplay"].iloc[-1])
    if not np.isfinite(fair) or fair <= 0.0 or not np.isfinite(fair_display) or fair_display <= 0.0:
        return 1.0
    return fair_display / fair


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
        tooltip_label = escape(str(band["label"]), quote=True)
        if _sigma_band_contains_level(band, current_sigma_level):
            item_class = "sigma-bar-item sigma-bar-item-current"
            aria_label = " aria-label='Current sigma band'"
        return (
            f"<div class='{item_class}' title='{tooltip_label}'{aria_label}>"
            f"<div class='sigma-bar-value'>{band['share']:.1f}%</div>"
            "<div class='sigma-bar-track'>"
            f"<div class='sigma-bar-fill' style='height:{(float(band['share']) / scale_max) * 100.0:.1f}%;'></div>"
            "</div>"
            f"<div class='sigma-bar-label'>{band.get('compact_label', band['label'])}</div>"
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
    monthly_growth = calculate_negative_two_sigma_monthly_growth(
        df_display,
        a_active,
        b_active,
        p2_5,
    )
    monthly_growth_display = monthly_growth * _resolve_display_conversion_rate(df_display)
    current_sigma_level = calculate_current_powerlaw_sigma_level(
        df_display, p2_5, p16_5, p83_5, p97_5
    )
    current_sigma_display = (
        f"{current_sigma_level:+.2f}σ" if current_sigma_level is not None else "N/A"
    )

    k1, k2, k3, k4 = st.columns(4)
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
    _kpi_card(
        k3,
        "-2σ MONTHLY GROWTH",
        _format_money(
            monthly_growth_display,
            currency_prefix,
            currency_suffix,
            currency_decimals,
        ),
        "per 1 BTC",
        "#f0b90b",
    )
    _kpi_card(
        k4,
        "CURRENT SIGMA",
        current_sigma_display,
        "from PowerLaw",
        "#9ba3af",
    )

    if logperiodic_stats_rows or perrenod_stats_rows:
        from core.oscillator import render_logperiodic_regression_stats_table

        render_logperiodic_regression_stats_table(logperiodic_stats_rows, perrenod_stats_rows)

    selected_history_range = resolve_sigma_band_history_percent_range(
        st.session_state.get(
            KEY_SIGMA_BAND_HISTORY_RANGE_PCT,
            SIGMA_BAND_HISTORY_PERCENT_RANGE_DEFAULT,
        )
    )
    band_df = filter_sigma_band_history_percent_range(df_display, selected_history_range)
    band_shares = calculate_powerlaw_band_shares(band_df, p2_5, p16_5, p83_5, p97_5)
    history_label = format_sigma_band_history_percent_range(selected_history_range)
    _render_sigma_band_chart(band_shares, current_sigma_level, history_label)
