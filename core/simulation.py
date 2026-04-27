import numpy as np
import pandas as pd

from core.constants import DEFAULT_A, DEFAULT_B
from core.utils import evaluate_powerlaw_values

HALVING_CYCLE_DAYS = 210_000 * 10 / 144


def _interpolate_anchor_process(rng, days, resolution_days, scale, mean_reversion):
    min_day = float(np.min(days))
    max_day = float(np.max(days))
    anchor_step = max(1.0, float(resolution_days))
    anchor_days = np.arange(min_day, max_day + anchor_step, anchor_step)
    anchor_values = np.zeros(len(anchor_days), dtype=float)

    for idx in range(1, len(anchor_values)):
        innovation = rng.normal(0.0, float(scale))
        anchor_values[idx] = float(mean_reversion) * anchor_values[idx - 1] + innovation

    return np.interp(days, anchor_days, anchor_values)


def _build_step_shock_process(rng, days, resolution_days):
    shock_values = np.zeros(len(days), dtype=float)
    positive_state = 0.0
    negative_state = 0.0
    texture_state = 0.0
    resolution_factor = np.sqrt(1.0 / max(0.00001, float(resolution_days)))

    for idx in range(1, len(days)):
        dt = max(0.0, float(days[idx] - days[idx - 1]))
        positive_state *= np.exp(-dt / 72.0)
        negative_state *= np.exp(-dt / 110.0)
        texture_state = (0.94**dt) * texture_state + rng.normal(
            0.0, 0.010 * np.sqrt(dt) * np.power(resolution_factor, 0.25)
        )

        positive_events = rng.poisson(0.0032 * dt * resolution_factor)
        negative_events = rng.poisson(0.0011 * dt * resolution_factor)
        if positive_events > 0:
            positive_state += np.sum(0.05 + rng.pareto(2.25, size=positive_events) * 0.12)
        if negative_events > 0:
            negative_state += np.sum(0.04 + rng.pareto(2.6, size=negative_events) * 0.09)

        shock_values[idx] = positive_state - negative_state + texture_state

    return shock_values


def _build_halving_cycle_residuals(rng, days):
    residuals = np.zeros(len(days), dtype=float)
    cycle_index = np.floor(days / HALVING_CYCLE_DAYS).astype(int)

    for cycle_id in np.unique(cycle_index):
        cycle_mask = cycle_index == cycle_id
        cycle_days = days[cycle_mask] - (cycle_id * HALVING_CYCLE_DAYS)
        cycle_phase = cycle_days / HALVING_CYCLE_DAYS
        maturity = 1.0 - np.exp(-float(cycle_id + 1) / 5.0)
        amplitude = rng.uniform(1.0, 1.45) * (1.36 - 0.36 * maturity)
        phase_shift = rng.normal(0.0, 0.035)

        boom_center = 0.38 + phase_shift
        bust_center = 0.68 + phase_shift
        accumulation_center = 0.92 + phase_shift

        boom = amplitude * np.exp(-0.5 * ((cycle_phase - boom_center) / 0.095) ** 2)
        blowoff = (
            0.36 * amplitude * np.exp(-0.5 * ((cycle_phase - (boom_center + 0.08)) / 0.035) ** 2)
        )
        bust = -0.88 * amplitude * np.exp(-0.5 * ((cycle_phase - bust_center) / 0.13) ** 2)
        accumulation = (
            -0.28 * amplitude * np.exp(-0.5 * ((cycle_phase - accumulation_center) / 0.16) ** 2)
        )

        residuals[cycle_mask] = boom + blowoff + bust + accumulation

    return residuals


def _remove_residual_powerlaw_drift(log_days, residuals):
    valid_mask = np.isfinite(log_days) & np.isfinite(residuals)
    if np.count_nonzero(valid_mask) < 2:
        return residuals

    drift_slope, drift_intercept = np.polyfit(log_days[valid_mask], residuals[valid_mask], 1)
    return residuals - (drift_intercept + drift_slope * log_days)


def build_bitcoin_network_simulation(
    base_df,
    seed=1,
    intercept_a=DEFAULT_A,
    slope_b=DEFAULT_B,
    resolution_days=0.00001,
):
    """Build a fully synthetic Bitcoin-like network trajectory around a PowerLaw baseline."""
    sim_df = base_df[["AbsDays"]].copy()
    sim_df.index = pd.to_datetime(base_df.index)

    days = np.maximum(sim_df["AbsDays"].to_numpy(dtype=float), 1.0)
    log_days = np.log10(days)
    baseline, baseline_log, _ = evaluate_powerlaw_values(
        log_days, float(intercept_a), float(slope_b)
    )

    rng = np.random.default_rng(int(seed))
    resolution_days = float(max(0.00001, min(0.01, resolution_days)))
    cycle_residuals = _build_halving_cycle_residuals(rng, days)
    step_shocks = _build_step_shock_process(rng, days, resolution_days)
    market_structure = _interpolate_anchor_process(
        rng,
        days,
        resolution_days=7.0,
        scale=0.052,
        mean_reversion=0.985,
    )
    microstructure = _interpolate_anchor_process(
        rng,
        days,
        resolution_days=resolution_days,
        scale=0.016 * np.power(1.0 / resolution_days, 0.08),
        mean_reversion=0.72,
    )
    jump_probability = 0.0015 * np.sqrt(1.0 / resolution_days)
    jump_values = rng.standard_t(df=4, size=len(days)) * 0.09
    jump_mask = rng.random(len(days)) < jump_probability
    jumps = np.where(jump_mask, jump_values, 0.0)

    log_residuals = cycle_residuals + step_shocks + market_structure + microstructure + jumps
    log_residuals = _remove_residual_powerlaw_drift(log_days, log_residuals)

    sim_df["Close"] = baseline * np.power(10.0, log_residuals)
    sim_df["Close"] = np.maximum(sim_df["Close"], 1e-12)
    sim_df["LogClose"] = baseline_log + log_residuals
    return sim_df
