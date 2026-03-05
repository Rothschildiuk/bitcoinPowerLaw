import unittest
from unittest.mock import patch

import numpy as np

from core.power_law import optimize_single_powerlaw_parameter


class TestPowerLawBounds(unittest.TestCase):
    @patch("core.power_law.optimize_single_scalar_parameter")
    def test_optimize_single_powerlaw_parameter_passes_custom_bounds_for_intercept(
        self, mock_optimize_scalar
    ):
        mock_optimize_scalar.return_value = (-12.3456, 0.9)

        best_a, best_r2 = optimize_single_powerlaw_parameter(
            absolute_days=np.array([1.0, 2.0, 3.0]),
            log_prices=np.array([0.1, 0.2, 0.3]),
            genesis_offset_days=0,
            current_intercept_a=-10.0,
            current_slope_b=5.0,
            parameter_key="A",
            a_min=-80.0,
            a_max=10.0,
            b_min=0.1,
            b_max=30.0,
        )

        self.assertEqual(best_a, -12.346)
        self.assertEqual(best_r2, 0.9)
        self.assertEqual(mock_optimize_scalar.call_args.kwargs["min_value"], -80.0)
        self.assertEqual(mock_optimize_scalar.call_args.kwargs["max_value"], 10.0)

    @patch("core.power_law.optimize_single_scalar_parameter")
    def test_optimize_single_powerlaw_parameter_passes_custom_bounds_for_slope(
        self, mock_optimize_scalar
    ):
        mock_optimize_scalar.return_value = (8.7654, 0.8)

        best_b, best_r2 = optimize_single_powerlaw_parameter(
            absolute_days=np.array([1.0, 2.0, 3.0]),
            log_prices=np.array([0.1, 0.2, 0.3]),
            genesis_offset_days=0,
            current_intercept_a=-10.0,
            current_slope_b=5.0,
            parameter_key="B",
            a_min=-80.0,
            a_max=10.0,
            b_min=0.25,
            b_max=40.0,
        )

        self.assertEqual(best_b, 8.765)
        self.assertEqual(best_r2, 0.8)
        self.assertEqual(mock_optimize_scalar.call_args.kwargs["min_value"], 0.25)
        self.assertEqual(mock_optimize_scalar.call_args.kwargs["max_value"], 40.0)


if __name__ == "__main__":
    unittest.main()
