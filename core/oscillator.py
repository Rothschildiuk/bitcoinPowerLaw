from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import streamlit as st

from core.constants import (
    DEFAULT_A,
    DEFAULT_B,
    GENESIS_DATE,
    KEY_A,
    KEY_B,
    KEY_GENESIS_OFFSET,
    KEY_LOGPERIODIC_HARMONICS,
    KEY_LOGPERIODIC_SHOW_DECAYED_DSI,
    OSC_DEFAULTS,
)
from core.optimization_utils import optimize_parameter_by_candidates
from core.utils import calculate_r2_score, fancy_control, get_stable_trend_fit

AUTO_FIT_MAX_PASSES = 2
AUTO_FIT_GRID_POINTS = 9
AUTO_FIT_SINGLE_GRID_POINTS = 41
AUTO_FIT_VISIBLE_PASSES = 12
DEFAULT_OSCILLATOR_PARAMETER_BOUNDS = {
    "t1_age": (0.5, 3.0),
    "lambda_val": (1.5, 5.0),
}


def format_cycle_anchor_date(age_years, origin_offset_days=0):
    anchor_days = int(origin_offset_days) + int(round(float(age_years) * 365.25))
    anchor_date = GENESIS_DATE + timedelta(days=anchor_days)
    return anchor_date.strftime("%Y-%m-%d")


@dataclass(frozen=True)
class OscillatorSettings:
    t1_age: float
    lambda_val: float
    harmonic_count: int = 1


@dataclass(frozen=True)
class OscillatorComputationResult:
    settings: OscillatorSettings
    amplitude: float
    angular_frequency: float
    phase_shift: float
    combined_r2: float
    reference_log_day: float
    model_values: np.ndarray
    harmonic_coefficients: np.ndarray


@dataclass(frozen=True)
class OscillatorModelStats:
    harmonic_count: int
    mode_multipliers: tuple[int, ...]
    parameter_count: int
    r2: float
    aic: float
    bic: float
    rmse: float


@dataclass(frozen=True)
class RegressionComparisonStats:
    label: str
    parameter_label: str
    parameter_count: int
    r2: float
    aic: float
    bic: float
    rmse: float


def _resolve_oscillator_bounds(bounds_override=None):
    bounds = dict(DEFAULT_OSCILLATOR_PARAMETER_BOUNDS)
    if bounds_override:
        bounds.update(bounds_override)
    return bounds


def _normalize_harmonic_count(harmonic_count):
    try:
        count = int(harmonic_count)
    except (TypeError, ValueError):
        count = 1
    return min(3, max(1, count))


def resolve_harmonic_multipliers(harmonic_count):
    count = _normalize_harmonic_count(harmonic_count)
    if count == 1:
        return (1,)
    if count == 2:
        return (1, 2)
    return (1, 2, 4)


def _build_harmonic_template(phase_values, harmonic_index):
    return build_oscillator_wave(phase_values * harmonic_index)


def fit_oscillator_component(
    log_days,
    residual_series,
    t1_cycle_age_years,
    cycle_lambda,
    harmonic_count=1,
):
    if (not np.isfinite(t1_cycle_age_years)) or t1_cycle_age_years <= 0:
        return None
    if (not np.isfinite(cycle_lambda)) or cycle_lambda <= 0:
        return None

    log_lambda = np.log10(cycle_lambda)
    if abs(log_lambda) <= 1e-9:
        return None

    t1_log_days = np.log10(t1_cycle_age_years * 365.25)
    angular_frequency = 2 * np.pi / log_lambda
    phase_shift = -angular_frequency * t1_log_days
    phase_values = angular_frequency * log_days + phase_shift
    mode_multipliers = resolve_harmonic_multipliers(harmonic_count)

    harmonic_templates = [
        _build_harmonic_template(phase_values, harmonic_multiplier)
        for harmonic_multiplier in mode_multipliers
    ]

    if len(mode_multipliers) == 1:
        asymmetric_template = harmonic_templates[0]
        template_energy = np.dot(asymmetric_template, asymmetric_template)
        if template_energy <= 1e-12:
            return None

        fitted_amplitude = abs(np.dot(residual_series, asymmetric_template) / template_energy)
        predicted_residuals = fitted_amplitude * asymmetric_template
        return fitted_amplitude, angular_frequency, phase_shift, predicted_residuals

    design_matrix = np.column_stack(harmonic_templates)
    if np.linalg.matrix_rank(design_matrix) == 0:
        return None
    harmonic_coefficients, *_ = np.linalg.lstsq(
        design_matrix, np.asarray(residual_series, dtype=float), rcond=None
    )
    if not np.all(np.isfinite(harmonic_coefficients)):
        return None

    predicted_residuals = design_matrix @ harmonic_coefficients
    return harmonic_coefficients, angular_frequency, phase_shift, predicted_residuals


def compute_oscillator_fit_r2(
    log_days,
    residual_series,
    t1_cycle_age_years,
    cycle_lambda,
    harmonic_count=1,
):
    fit_result = fit_oscillator_component(
        log_days,
        residual_series,
        t1_cycle_age_years,
        cycle_lambda,
        harmonic_count,
    )
    if fit_result is None:
        return -1e9

    _, _, _, predicted_residuals = fit_result
    return calculate_r2_score(residual_series, predicted_residuals) * 100.0


def compute_sidebar_logperiodic_r2(
    log_days,
    residual_series,
    days_since_genesis,
    current_params,
    *,
    harmonic_count,
    lambda_bounds,
    show_decayed_regression,
):
    decay_model = "reciprocal_age" if show_decayed_regression else "none"
    stats = compute_dsi_regression_stats(
        log_days,
        residual_series,
        current_params["lambda_val"],
        harmonic_count,
        days_since_genesis=days_since_genesis,
        decay_model=decay_model,
    )
    return float(stats.r2)


def compute_oscillator_model_stats(
    log_days,
    residual_series,
    t1_cycle_age_years,
    cycle_lambda,
    harmonic_count=1,
):
    residual_values = np.asarray(residual_series, dtype=float)
    fit_result = fit_oscillator_component(
        log_days,
        residual_values,
        t1_cycle_age_years,
        cycle_lambda,
        harmonic_count,
    )
    mode_multipliers = resolve_harmonic_multipliers(harmonic_count)
    coefficient_count = len(mode_multipliers)
    parameter_count = coefficient_count + 2
    if fit_result is None or residual_values.size == 0:
        return OscillatorModelStats(
            harmonic_count=_normalize_harmonic_count(harmonic_count),
            mode_multipliers=mode_multipliers,
            parameter_count=parameter_count,
            r2=-1e9,
            aic=np.inf,
            bic=np.inf,
            rmse=np.inf,
        )

    _, _, _, predicted_residuals = fit_result
    residual_errors = residual_values - np.asarray(predicted_residuals, dtype=float)
    valid_errors = residual_errors[np.isfinite(residual_errors)]
    n_obs = int(valid_errors.size)
    if n_obs == 0:
        sse = np.inf
        rmse = np.inf
    else:
        sse = float(np.sum(valid_errors**2))
        rmse = float(np.sqrt(sse / n_obs))
    safe_mse = max(sse / max(n_obs, 1), np.finfo(float).tiny)
    aic = float(n_obs * np.log(safe_mse) + 2 * parameter_count)
    bic = float(n_obs * np.log(safe_mse) + parameter_count * np.log(max(n_obs, 1)))
    return OscillatorModelStats(
        harmonic_count=_normalize_harmonic_count(harmonic_count),
        mode_multipliers=mode_multipliers,
        parameter_count=parameter_count,
        r2=calculate_r2_score(residual_values, predicted_residuals) * 100.0,
        aic=aic,
        bic=bic,
        rmse=rmse,
    )


def compute_oscillator_model_stats_table(log_days, residual_series, current_params):
    return [
        compute_oscillator_model_stats(
            log_days,
            residual_series,
            current_params["t1_age"],
            current_params["lambda_val"],
            harmonic_count=harmonic_count,
        )
        for harmonic_count in (1, 2, 3)
    ]


def _format_mode_label(mode_multipliers):
    return ",".join(f"{mode}ω" if mode != 1 else "ω" for mode in mode_multipliers)


def _format_metric_value(value, precision):
    if value is None:
        return "-"
    if not np.isfinite(value):
        return "-"
    return f"{value:.{precision}f}"


def _best_r2_row_index(stats_rows):
    finite_rows = [
        (index, float(row["r2"]))
        for index, row in enumerate(stats_rows)
        if row is not None and np.isfinite(row["r2"])
    ]
    if not finite_rows:
        return None
    return max(finite_rows, key=lambda item: item[1])[0]


def _compute_prediction_stats(observed_values, predicted_values, parameter_count):
    observed = np.asarray(observed_values, dtype=float)
    predicted = np.asarray(predicted_values, dtype=float)
    valid_mask = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[valid_mask]
    predicted = predicted[valid_mask]
    n_obs = int(observed.size)
    if n_obs == 0:
        return -1e9, np.inf, np.inf, np.inf

    errors = observed - predicted
    sse = float(np.sum(errors**2))
    rmse = float(np.sqrt(sse / n_obs))
    safe_mse = max(sse / n_obs, np.finfo(float).tiny)
    aic = float(n_obs * np.log(safe_mse) + 2 * parameter_count)
    bic = float(n_obs * np.log(safe_mse) + parameter_count * np.log(n_obs))
    return calculate_r2_score(observed, predicted) * 100.0, aic, bic, rmse


def _fit_design_matrix(design_matrix, observed_values):
    design = np.asarray(design_matrix, dtype=float)
    observed = np.asarray(observed_values, dtype=float)
    valid_mask = np.isfinite(observed) & np.all(np.isfinite(design), axis=1)
    design = design[valid_mask]
    observed = observed[valid_mask]
    if observed.size == 0 or design.size == 0 or np.linalg.matrix_rank(design) == 0:
        return None
    coefficients, *_ = np.linalg.lstsq(design, observed, rcond=None)
    if not np.all(np.isfinite(coefficients)):
        return None
    return valid_mask, design @ coefficients


def _build_dsi_regression_design(log_days, lambda_val, mode_multipliers, decay_values=None):
    log_lambda = np.log10(float(lambda_val))
    if abs(log_lambda) <= 1e-9:
        return None
    phase_values = (2.0 * np.pi / log_lambda) * np.asarray(log_days, dtype=float)
    if decay_values is None:
        decay = np.ones_like(phase_values)
    else:
        decay = np.asarray(decay_values, dtype=float)
    columns = [np.ones_like(phase_values)]
    for mode_multiplier in mode_multipliers:
        mode_phase = phase_values * mode_multiplier
        columns.append(np.cos(mode_phase) * decay)
        columns.append(np.sin(mode_phase) * decay)
    return np.column_stack(columns)


def _build_reciprocal_age_decay(days_since_genesis):
    age_years = np.maximum(np.asarray(days_since_genesis, dtype=float) / 365.25, 0.0)
    return 1.0 / (age_years + 2.0)


def compute_dsi_regression_stats(
    log_days,
    residual_series,
    lambda_val,
    harmonic_count,
    *,
    days_since_genesis=None,
    decay_model="none",
):
    mode_multipliers = resolve_harmonic_multipliers(harmonic_count)
    decay_values = None
    decay_suffix = ""
    if decay_model == "reciprocal_age":
        if days_since_genesis is None:
            return RegressionComparisonStats(
                label=f"DSI {_format_mode_label(mode_multipliers)} decayed",
                parameter_label=f"λ {lambda_val:.2f}",
                parameter_count=2 * len(mode_multipliers) + 2,
                r2=-1e9,
                aic=np.inf,
                bic=np.inf,
                rmse=np.inf,
            )
        decay_values = _build_reciprocal_age_decay(days_since_genesis)
        decay_suffix = " decayed"

    design_matrix = _build_dsi_regression_design(
        log_days,
        lambda_val,
        mode_multipliers,
        decay_values=decay_values,
    )
    parameter_count = 2 * len(mode_multipliers) + 2
    if design_matrix is None:
        return RegressionComparisonStats(
            label=f"DSI {_format_mode_label(mode_multipliers)}{decay_suffix}",
            parameter_label=f"λ {lambda_val:.2f}",
            parameter_count=parameter_count,
            r2=-1e9,
            aic=np.inf,
            bic=np.inf,
            rmse=np.inf,
        )

    fit_result = _fit_design_matrix(design_matrix, residual_series)
    if fit_result is None:
        r2, aic, bic, rmse = -1e9, np.inf, np.inf, np.inf
    else:
        valid_mask, predicted = fit_result
        r2, aic, bic, rmse = _compute_prediction_stats(
            np.asarray(residual_series, dtype=float)[valid_mask],
            predicted,
            parameter_count,
        )
    return RegressionComparisonStats(
        label=f"DSI {_format_mode_label(mode_multipliers)}{decay_suffix}",
        parameter_label=f"λ {lambda_val:.2f}",
        parameter_count=parameter_count,
        r2=r2,
        aic=aic,
        bic=bic,
        rmse=rmse,
    )


def build_dsi_regression_curve(
    fit_log_days,
    fit_residuals,
    predict_log_days,
    lambda_val,
    harmonic_count,
    *,
    fit_days_since_genesis=None,
    predict_days_since_genesis=None,
    decay_model="none",
):
    fit_decay_values = None
    predict_decay_values = None
    if decay_model == "reciprocal_age":
        if fit_days_since_genesis is None or predict_days_since_genesis is None:
            return None
        fit_decay_values = _build_reciprocal_age_decay(fit_days_since_genesis)
        predict_decay_values = _build_reciprocal_age_decay(predict_days_since_genesis)

    mode_multipliers = resolve_harmonic_multipliers(harmonic_count)
    fit_design = _build_dsi_regression_design(
        fit_log_days,
        lambda_val,
        mode_multipliers,
        decay_values=fit_decay_values,
    )
    predict_design = _build_dsi_regression_design(
        predict_log_days,
        lambda_val,
        mode_multipliers,
        decay_values=predict_decay_values,
    )
    if fit_design is None or predict_design is None:
        return None

    fit_residual_values = np.asarray(fit_residuals, dtype=float)
    valid_mask = np.isfinite(fit_residual_values) & np.all(np.isfinite(fit_design), axis=1)
    if np.count_nonzero(valid_mask) == 0:
        return None
    coefficients, *_ = np.linalg.lstsq(
        fit_design[valid_mask],
        fit_residual_values[valid_mask],
        rcond=None,
    )
    if not np.all(np.isfinite(coefficients)):
        return None
    return predict_design @ coefficients


def optimize_dsi_regression_lambda(
    log_days,
    residual_series,
    harmonic_count,
    *,
    min_lambda,
    max_lambda,
    days_since_genesis=None,
    decay_model="none",
):
    candidates = np.arange(float(min_lambda), float(max_lambda) + 0.005, 0.01)
    if candidates.size == 0:
        candidates = np.linspace(float(min_lambda), float(max_lambda), 41)
    best_stats = None
    for candidate in candidates:
        stats = compute_dsi_regression_stats(
            log_days,
            residual_series,
            float(candidate),
            harmonic_count,
            days_since_genesis=days_since_genesis,
            decay_model=decay_model,
        )
        if best_stats is None or stats.aic < best_stats.aic:
            best_stats = stats
    return best_stats


def compute_linear_cycle_regression_stats(
    days_since_genesis,
    residual_series,
    cycle_years,
):
    age_years = np.asarray(days_since_genesis, dtype=float) / 365.25
    phase_values = 2.0 * np.pi * age_years / float(cycle_years)
    design_matrix = np.column_stack(
        [np.ones_like(phase_values), np.cos(phase_values), np.sin(phase_values)]
    )
    parameter_count = 3
    fit_result = _fit_design_matrix(design_matrix, residual_series)
    if fit_result is None:
        r2, aic, bic, rmse = -1e9, np.inf, np.inf, np.inf
    else:
        valid_mask, predicted = fit_result
        r2, aic, bic, rmse = _compute_prediction_stats(
            np.asarray(residual_series, dtype=float)[valid_mask],
            predicted,
            parameter_count,
        )
    return RegressionComparisonStats(
        label=f"Linear {cycle_years:g}y",
        parameter_label=f"{cycle_years:g}y",
        parameter_count=parameter_count,
        r2=r2,
        aic=aic,
        bic=bic,
        rmse=rmse,
    )


def compute_perrenod_comparison_stats_table(
    log_days,
    residual_series,
    days_since_genesis,
    lambda_bounds,
):
    min_lambda, max_lambda = lambda_bounds
    rows = []
    for decay_model in ("none", "reciprocal_age"):
        for harmonic_count in (1, 2, 3):
            rows.append(
                optimize_dsi_regression_lambda(
                    log_days,
                    residual_series,
                    harmonic_count,
                    min_lambda=min_lambda,
                    max_lambda=max_lambda,
                    days_since_genesis=days_since_genesis,
                    decay_model=decay_model,
                )
            )
    return rows


def render_logperiodic_regression_stats_table(logperiodic_stats_rows, perrenod_stats_rows):
    combined_rows = []
    perrenod_rows = [row for row in perrenod_stats_rows or [] if row is not None]
    decayed_rows_by_model = {
        row.label.removesuffix(" decayed"): row
        for row in perrenod_rows
        if row.label.startswith("DSI ") and row.label.endswith(" decayed")
    }
    for row in perrenod_rows:
        if row is None:
            continue
        if not row.label.startswith("DSI "):
            continue
        if row.label.endswith(" decayed"):
            continue

        decayed_row = decayed_rows_by_model.get(row.label)
        combined_rows.append(
            {
                "model": row.label,
                "fit": row.parameter_label,
                "params": row.parameter_count,
                "r2": row.r2,
                "decayed_r2": decayed_row.r2 if decayed_row is not None else None,
            }
        )

    if not combined_rows:
        return

    best_index = _best_r2_row_index(combined_rows)
    rows_html = "".join(
        (
            f"<tr class='{'is-best' if index == best_index else ''}'>"
            f"<td>{row['model']}</td>"
            f"<td>{row['fit']}</td>"
            f"<td>{row['params']}</td>"
            f"<td>{_format_metric_value(row['r2'], 2)}</td>"
            f"<td>{_format_metric_value(row['decayed_r2'], 2)}</td>"
            "</tr>"
        )
        for index, row in enumerate(combined_rows)
    )
    st.markdown(
        (
            "<section class='lp-stats-panel'>"
            "<div class='lp-stats-panel-header'>"
            "<div>"
            "<div class='lp-stats-eyebrow'>LogPeriodic</div>"
            "<div class='lp-stats-title'>Regression comparison</div>"
            "</div>"
            "<div class='lp-stats-badge'>best R² highlighted</div>"
            "</div>"
            "<div class='lp-stats-table-wrap'>"
            "<table class='lp-stats-table lp-combined-table'>"
            "<thead><tr>"
            "<th>Model</th><th>Fit</th><th>Params</th><th>R²%</th><th>Decayed R²%</th>"
            "</tr></thead>"
            f"<tbody>{rows_html}</tbody>"
            "</table>"
            "</div>"
            "</section>"
        ),
        unsafe_allow_html=True,
    )


def render_regression_comparison_stats_table(stats_rows):
    render_logperiodic_regression_stats_table(None, stats_rows)


def render_oscillator_model_stats_table(stats_rows):
    render_logperiodic_regression_stats_table(stats_rows, None)


def compute_oscillator_overlay(
    log_days,
    residual_series,
    model_log_values,
    actual_log_values,
    fit_mask,
    settings,
    current_r2,
):
    osc_fit_mask = np.asarray(fit_mask, dtype=bool).copy()
    osc_model_vals = np.zeros_like(np.asarray(residual_series, dtype=float), dtype=float)
    osc_amp = 0.0
    osc_omega = 0.0
    osc_phi = 0.0
    r2_combined = float(current_r2)
    reference_log_day = float(np.min(log_days))
    harmonic_coefficients = np.array([], dtype=float)

    osc_fit_log_d = np.asarray(log_days, dtype=float)[osc_fit_mask]
    osc_fit_residuals = np.asarray(residual_series, dtype=float)[osc_fit_mask]
    if osc_fit_log_d.size > 0:
        reference_log_day = float(np.min(osc_fit_log_d))

    fit_result = None
    if osc_fit_log_d.size > 1:
        fit_result = fit_oscillator_component(
            osc_fit_log_d,
            osc_fit_residuals,
            settings.t1_age,
            settings.lambda_val,
            settings.harmonic_count,
        )

    if fit_result is not None:
        fitted_amplitude, osc_omega, osc_phi, osc_model_fit = fit_result
        harmonic_coefficients = np.atleast_1d(np.asarray(fitted_amplitude, dtype=float))
        osc_amp = float(harmonic_coefficients[0]) if harmonic_coefficients.size else 0.0
        osc_model_vals[osc_fit_mask] = osc_model_fit

    total_model_log = np.asarray(model_log_values, dtype=float) + osc_model_vals
    if np.count_nonzero(osc_fit_mask) > 1:
        r2_combined = (
            calculate_r2_score(
                np.asarray(actual_log_values, dtype=float)[osc_fit_mask],
                total_model_log[osc_fit_mask],
            )
            * 100.0
        )

    return OscillatorComputationResult(
        settings=settings,
        amplitude=float(osc_amp),
        angular_frequency=float(osc_omega),
        phase_shift=float(osc_phi),
        combined_r2=float(r2_combined),
        reference_log_day=float(reference_log_day),
        model_values=osc_model_vals,
        harmonic_coefficients=harmonic_coefficients,
    )


def optimize_oscillator_parameters(
    log_days,
    residual_series,
    initial_params,
    *,
    bounds_override=None,
    parameter_order=None,
):
    optimized_params = dict(initial_params)
    bounds = _resolve_oscillator_bounds(bounds_override)
    spans = {
        "t1_age": 1.20,
        "lambda_val": 0.60,
    }
    order = list(parameter_order or ["t1_age", "lambda_val"])

    best_r2 = compute_oscillator_fit_r2(
        log_days,
        residual_series,
        optimized_params["t1_age"],
        optimized_params["lambda_val"],
        optimized_params.get("harmonic_count", 1),
    )

    for _ in range(AUTO_FIT_MAX_PASSES):
        improved = False
        for key in order:
            lo_b, hi_b = bounds[key]
            center = optimized_params[key]
            lo = max(lo_b, center - spans[key])
            hi = min(hi_b, center + spans[key])
            candidates = np.linspace(lo, hi, AUTO_FIT_GRID_POINTS)

            local_best_val = center
            local_best_r2 = best_r2
            for cand in candidates:
                trial = dict(optimized_params)
                trial[key] = float(cand)
                score = compute_oscillator_fit_r2(
                    log_days,
                    residual_series,
                    trial["t1_age"],
                    trial["lambda_val"],
                    trial.get("harmonic_count", 1),
                )
                if score > local_best_r2:
                    local_best_r2 = score
                    local_best_val = float(cand)

            if local_best_val != center:
                optimized_params[key] = local_best_val
                best_r2 = local_best_r2
                improved = True

        for k in spans:
            spans[k] *= 0.45
        if not improved:
            break

    return optimized_params


def optimize_single_oscillator_parameter(
    log_days,
    residual_series,
    current_params,
    parameter_key,
    min_value,
    max_value,
    step_value=None,
    grid_points=AUTO_FIT_SINGLE_GRID_POINTS,
):
    params = dict(current_params)
    best_value, best_r2 = optimize_parameter_by_candidates(
        float(params[parameter_key]),
        lambda candidate: compute_oscillator_fit_r2(
            log_days,
            residual_series,
            params["t1_age"] if parameter_key != "t1_age" else float(candidate),
            params["lambda_val"] if parameter_key != "lambda_val" else float(candidate),
            params.get("harmonic_count", 1),
        ),
        min_value=min_value,
        max_value=max_value,
        step_value=step_value,
        grid_points=grid_points,
    )

    return best_value, best_r2


def optimize_visible_oscillator_parameters(
    log_days,
    residual_series,
    current_params,
    *,
    bounds_override=None,
    parameter_order=None,
    step_map=None,
    passes=AUTO_FIT_VISIBLE_PASSES,
):
    params = dict(current_params)
    bounds = _resolve_oscillator_bounds(bounds_override)
    order = list(parameter_order or ["t1_age", "lambda_val"])
    steps = dict(step_map or {})

    for _ in range(max(1, int(passes))):
        improved = False
        for sweep_order in (order, list(reversed(order))):
            for key in sweep_order:
                min_value, max_value = bounds[key]
                best_value, best_r2 = optimize_single_oscillator_parameter(
                    log_days,
                    residual_series,
                    params,
                    parameter_key=key,
                    min_value=min_value,
                    max_value=max_value,
                    step_value=steps.get(key),
                )
                previous_value = float(params[key])
                if not np.isfinite(best_r2):
                    continue
                params[key] = float(best_value)
                if not np.isclose(previous_value, float(best_value)):
                    improved = True
        if not improved:
            break

    return params


def build_autofit_signature(all_abs_days, all_log_close):
    if len(all_abs_days) == 0:
        return ("empty",)
    return (
        int(st.session_state.get(KEY_GENESIS_OFFSET, 0)),
        round(float(st.session_state.get(KEY_A, 0.0)), 6),
        round(float(st.session_state.get(KEY_B, 0.0)), 6),
        len(all_abs_days),
        int(all_abs_days[0]),
        int(all_abs_days[-1]),
        round(float(all_log_close[0]), 6),
        round(float(all_log_close[-1]), 6),
    )


# --- OSCILLATOR MATH (SINUSOID) ---
def build_oscillator_wave(phase_values):
    """
    Calculates a Cosine wave (Sinusoid) instead of Inverted Cycloid.
    Using Cosine ensures the peak corresponds to phase=0 (consistent with t1_age).
    Range: [-1, 1]
    """
    return np.cos(phase_values)


def build_oscillator_curve(
    log_days,
    amplitude,
    angular_frequency,
    phase_shift,
    reference_log_day=None,
    harmonic_coefficients=None,
):
    phase_values = angular_frequency * log_days + phase_shift
    coefficients = np.atleast_1d(
        np.asarray(
            harmonic_coefficients if harmonic_coefficients is not None else [amplitude],
            dtype=float,
        )
    )
    y_values = np.zeros_like(np.asarray(log_days, dtype=float), dtype=float)
    multipliers = resolve_harmonic_multipliers(len(coefficients))
    for harmonic_multiplier, coefficient in zip(multipliers, coefficients):
        template = _build_harmonic_template(phase_values, harmonic_multiplier)
        y_values += coefficient * template
    return y_values


# Backward-compatible aliases for existing call sites.
safe_r2 = calculate_r2_score
_oscillator_r2 = compute_oscillator_fit_r2
_auto_fit_oscillator = optimize_oscillator_parameters
get_oscillator_wave = build_oscillator_wave
oscillator_func_manual = build_oscillator_curve

KEY_LOGPERIODIC_DSI_MODE = "logperiodic_dsi_mode"
DSI_MODE_STANDARD_1 = "standard_1"
DSI_MODE_STANDARD_2 = "standard_2"
DSI_MODE_STANDARD_3 = "standard_3"
DSI_MODE_DECAYED_3 = "decayed_3"
DSI_MODE_OPTIONS = (
    DSI_MODE_STANDARD_1,
    DSI_MODE_STANDARD_2,
    DSI_MODE_STANDARD_3,
    DSI_MODE_DECAYED_3,
)
DSI_MODE_CONFIGS = {
    DSI_MODE_STANDARD_1: (1, False, "ω"),
    DSI_MODE_STANDARD_2: (2, False, "ω,2ω"),
    DSI_MODE_STANDARD_3: (3, False, "ω,2ω,4ω"),
    DSI_MODE_DECAYED_3: (3, True, "ω,2ω,4ω Decayed"),
}
DEFAULT_DSI_MODE = DSI_MODE_DECAYED_3


def resolve_dsi_mode_option(harmonic_count, show_decayed_regression=False):
    if show_decayed_regression:
        return DSI_MODE_DECAYED_3

    harmonic_count = _normalize_harmonic_count(harmonic_count)
    return {
        1: DSI_MODE_STANDARD_1,
        2: DSI_MODE_STANDARD_2,
        3: DSI_MODE_STANDARD_3,
    }.get(harmonic_count, DSI_MODE_STANDARD_1)


def apply_dsi_mode_option(option, *, sync_widget_key=True):
    harmonic_count, show_decayed_regression, _ = DSI_MODE_CONFIGS.get(
        option,
        DSI_MODE_CONFIGS[DSI_MODE_STANDARD_1],
    )
    st.session_state[KEY_LOGPERIODIC_HARMONICS] = harmonic_count
    st.session_state[KEY_LOGPERIODIC_SHOW_DECAYED_DSI] = show_decayed_regression
    if sync_widget_key:
        st.session_state[KEY_LOGPERIODIC_DSI_MODE] = resolve_dsi_mode_option(
            harmonic_count,
            show_decayed_regression,
        )


def format_dsi_mode_option(option):
    return DSI_MODE_CONFIGS.get(option, DSI_MODE_CONFIGS[DSI_MODE_STANDARD_1])[2]


# --- SIDEBAR RENDERER ---
def render_sidebar(
    all_abs_days,
    all_log_close,
    text_color,
    defaults_override=None,
    min_abs_day_for_fit=None,
    parameter_bounds_override=None,
    render_after_actions=None,
):
    defaults = dict(defaults_override or OSC_DEFAULTS)
    harmonic_options = [1, 2, 3]
    if KEY_LOGPERIODIC_HARMONICS not in st.session_state:
        st.session_state[KEY_LOGPERIODIC_HARMONICS] = int(
            defaults.get("harmonic_count", harmonic_options[0])
        )
    parameter_bounds = _resolve_oscillator_bounds(parameter_bounds_override)
    for k, v in defaults.items():
        if k == "harmonic_count":
            continue
        if k not in st.session_state:
            st.session_state[k] = v

    def reset_oscillator_params():
        for k, v in defaults.items():
            if k == "harmonic_count":
                st.session_state[KEY_LOGPERIODIC_HARMONICS] = int(v)
                continue
            st.session_state[k] = v
        apply_dsi_mode_option(DEFAULT_DSI_MODE)

    days_since_genesis = all_abs_days - st.session_state.get(KEY_GENESIS_OFFSET, 0)
    valid_days_mask = days_since_genesis > 0
    if min_abs_day_for_fit is not None:
        valid_days_mask = valid_days_mask & (all_abs_days >= float(min_abs_day_for_fit))
    oscillator_r2_display = 0.0
    log_days = None
    residual_series = None
    fit_days_since_genesis = None
    has_fit_data = np.count_nonzero(valid_days_mask) > 100

    if has_fit_data:
        fit_days_since_genesis = days_since_genesis[valid_days_mask]
        log_days = np.log10(days_since_genesis[valid_days_mask])
        _, _, trend_log_prices, residual_series = get_stable_trend_fit(
            log_days,
            all_log_close[valid_days_mask],
            float(st.session_state.get(KEY_A, DEFAULT_A)),
            float(st.session_state.get(KEY_B, DEFAULT_B)),
        )

    def auto_fit_single_parameter(parameter_key, min_value, max_value, step_value):
        if not has_fit_data:
            return

        current_params = {
            "t1_age": float(st.session_state.get("t1_age", defaults["t1_age"])),
            "lambda_val": float(st.session_state.get("lambda_val", defaults["lambda_val"])),
            "harmonic_count": _normalize_harmonic_count(
                st.session_state.get(KEY_LOGPERIODIC_HARMONICS, harmonic_options[0])
            ),
        }

        best_value, _ = optimize_single_oscillator_parameter(
            log_days,
            residual_series,
            current_params,
            parameter_key,
            min_value,
            max_value,
            step_value=step_value,
        )

        step_text = f"{step_value:.10f}".rstrip("0")
        precision = len(step_text.split(".")[1]) if "." in step_text else 0
        clipped_best = min(max_value, max(min_value, float(best_value)))
        st.session_state[parameter_key] = round(clipped_best, precision)

    def auto_fit_visible_parameters():
        if not has_fit_data:
            return

        current_params = {
            "t1_age": float(st.session_state.get("t1_age", defaults["t1_age"])),
            "lambda_val": float(st.session_state.get("lambda_val", defaults["lambda_val"])),
            "harmonic_count": _normalize_harmonic_count(
                st.session_state.get(KEY_LOGPERIODIC_HARMONICS, harmonic_options[0])
            ),
        }
        optimized = optimize_visible_oscillator_parameters(
            log_days,
            residual_series,
            current_params,
            bounds_override=parameter_bounds,
            parameter_order=["t1_age", "lambda_val"],
            step_map={"t1_age": 0.01, "lambda_val": 0.01},
        )
        st.session_state["t1_age"] = round(float(optimized["t1_age"]), 2)
        st.session_state["lambda_val"] = round(float(optimized["lambda_val"]), 2)

    def render_oscillator_control(title, key, step, min_v, max_v):
        st.markdown(f"**{title}**")
        fancy_control(
            title,
            key,
            step,
            min_v,
            max_v,
            on_auto_fit=lambda k=key, lo=min_v, hi=max_v, stp=step: auto_fit_single_parameter(
                k, lo, hi, stp
            ),
            auto_fit_label="AF",
        )

    t1_min, t1_max = parameter_bounds["t1_age"]
    lambda_min, lambda_max = parameter_bounds["lambda_val"]
    render_oscillator_control("1st Cycle Anchor", "t1_age", 0.01, t1_min, t1_max)
    st.caption(
        "Anchor date: "
        f"{format_cycle_anchor_date(st.session_state['t1_age'], st.session_state.get(KEY_GENESIS_OFFSET, 0))}"
    )
    render_oscillator_control("Lambda", "lambda_val", 0.01, lambda_min, lambda_max)
    if KEY_LOGPERIODIC_SHOW_DECAYED_DSI not in st.session_state:
        st.session_state[KEY_LOGPERIODIC_SHOW_DECAYED_DSI] = True
    if KEY_LOGPERIODIC_DSI_MODE not in st.session_state:
        st.session_state[KEY_LOGPERIODIC_DSI_MODE] = DEFAULT_DSI_MODE
    st.markdown("**DSI mode**")
    selected_dsi_mode = st.radio(
        "DSI mode",
        DSI_MODE_OPTIONS,
        format_func=format_dsi_mode_option,
        key=KEY_LOGPERIODIC_DSI_MODE,
        horizontal=True,
        label_visibility="collapsed",
    )
    apply_dsi_mode_option(selected_dsi_mode, sync_widget_key=False)
    # --- R2 Calculation for Sidebar Display ---
    if has_fit_data:
        harmonic_count = _normalize_harmonic_count(
            st.session_state.get(KEY_LOGPERIODIC_HARMONICS, harmonic_options[0])
        )
        current_params = {
            "t1_age": float(st.session_state.get("t1_age", defaults["t1_age"])),
            "lambda_val": float(st.session_state.get("lambda_val", defaults["lambda_val"])),
        }
        oscillator_r2_display = compute_sidebar_logperiodic_r2(
            log_days,
            residual_series,
            fit_days_since_genesis,
            current_params,
            harmonic_count=harmonic_count,
            lambda_bounds=parameter_bounds["lambda_val"],
            show_decayed_regression=bool(
                st.session_state.get(KEY_LOGPERIODIC_SHOW_DECAYED_DSI, True)
            ),
        )

    st.markdown(
        f"<p style='color:{text_color}; margin-top: 2px;'>"
        f"LogPeriodic R² = {oscillator_r2_display:.4f}%</p>",
        unsafe_allow_html=True,
    )
    st.button("Auto-fit model", use_container_width=True, on_click=auto_fit_visible_parameters)
    st.button("Reset parameters", use_container_width=True, on_click=reset_oscillator_params)
    if callable(render_after_actions):
        render_after_actions()
