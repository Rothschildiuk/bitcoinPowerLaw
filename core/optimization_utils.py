from __future__ import annotations

from typing import Callable

import numpy as np


def optimize_single_scalar_parameter(
    current_value: float,
    score_fn: Callable[[float], float],
    *,
    min_value: float,
    max_value: float,
    coarse_points: int,
    fine_window: float,
    fine_points: int,
):
    best_value = float(current_value)
    best_score = float(score_fn(best_value))

    coarse_candidates = np.linspace(min_value, max_value, coarse_points)
    for candidate in coarse_candidates:
        candidate = float(candidate)
        score = float(score_fn(candidate))
        if score > best_score:
            best_score = score
            best_value = candidate

    fine_lo = max(min_value, best_value - fine_window)
    fine_hi = min(max_value, best_value + fine_window)
    fine_candidates = np.linspace(fine_lo, fine_hi, fine_points)
    for candidate in fine_candidates:
        candidate = float(candidate)
        score = float(score_fn(candidate))
        if score > best_score:
            best_score = score
            best_value = candidate

    return best_value, best_score


def optimize_parameter_by_candidates(
    initial_value: float,
    score_fn: Callable[[float], float],
    *,
    min_value: float,
    max_value: float,
    step_value: float | None = None,
    grid_points: int = 41,
):
    if step_value is not None and step_value > 0:
        candidates = np.arange(min_value, max_value + (step_value * 0.5), step_value)
    else:
        candidates = np.linspace(min_value, max_value, grid_points)

    best_value = float(initial_value)
    best_score = float(score_fn(best_value))
    for candidate in candidates:
        candidate = float(candidate)
        score = float(score_fn(candidate))
        if score > best_score:
            best_score = score
            best_value = candidate

    return best_value, best_score
