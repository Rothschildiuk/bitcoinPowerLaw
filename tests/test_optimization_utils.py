import unittest

import numpy as np

from core.optimization_utils import (
    optimize_parameter_by_candidates,
    optimize_single_scalar_parameter,
)


class TestOptimizationUtils(unittest.TestCase):
    def test_optimize_single_scalar_parameter_finds_peak(self):
        # Peak at x=2.0 for this quadratic score.
        score_fn = lambda x: -((x - 2.0) ** 2) + 5.0
        best_value, best_score = optimize_single_scalar_parameter(
            current_value=0.0,
            score_fn=score_fn,
            min_value=-5.0,
            max_value=5.0,
            coarse_points=41,
            fine_window=0.5,
            fine_points=101,
        )

        self.assertTrue(np.isclose(best_value, 2.0, atol=0.02))
        self.assertTrue(np.isclose(best_score, 5.0, atol=0.02))

    def test_optimize_parameter_by_candidates_respects_step_grid(self):
        score_fn = lambda x: -((x - 3.0) ** 2)
        best_value, best_score = optimize_parameter_by_candidates(
            initial_value=0.0,
            score_fn=score_fn,
            min_value=1.0,
            max_value=5.0,
            step_value=0.5,
        )

        self.assertTrue(np.isclose(best_value, 3.0, atol=1e-12))
        self.assertTrue(np.isfinite(best_score))


if __name__ == "__main__":
    unittest.main()
