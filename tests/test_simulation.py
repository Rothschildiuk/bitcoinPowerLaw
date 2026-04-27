import unittest

import numpy as np
import pandas as pd

from core.constants import DEFAULT_B, GENESIS_DATE
from core.power_law import fit_powerlaw_regression
from core.simulation import build_bitcoin_network_simulation
from core.utils import evaluate_powerlaw_values


class TestBitcoinNetworkSimulation(unittest.TestCase):
    def test_simulation_slope_stays_near_bitcoin_default(self):
        dates = pd.date_range(start=GENESIS_DATE + pd.Timedelta(days=1), periods=4000, freq="D")
        base_df = pd.DataFrame(index=dates)
        base_df["AbsDays"] = (dates - GENESIS_DATE).days

        sim_df = build_bitcoin_network_simulation(base_df, seed=42)
        fitted_b, _, r2_score = fit_powerlaw_regression(
            sim_df["AbsDays"].to_numpy(dtype=float),
            np.log10(sim_df["Close"].to_numpy(dtype=float)),
            0,
        )

        self.assertAlmostEqual(fitted_b, DEFAULT_B, delta=0.2)
        self.assertGreater(r2_score, 0.5)

    def test_simulation_seed_changes_values(self):
        dates = pd.date_range(start=GENESIS_DATE + pd.Timedelta(days=1), periods=500, freq="D")
        base_df = pd.DataFrame(index=dates)
        base_df["AbsDays"] = (dates - GENESIS_DATE).days

        first_df = build_bitcoin_network_simulation(base_df, seed=1)
        second_df = build_bitcoin_network_simulation(base_df, seed=2)

        self.assertFalse(first_df["Close"].equals(second_df["Close"]))

    def test_simulation_ignores_source_price_values(self):
        dates = pd.date_range(start=GENESIS_DATE + pd.Timedelta(days=1), periods=2500, freq="D")
        base_df = pd.DataFrame(index=dates)
        base_df["AbsDays"] = (dates - GENESIS_DATE).days
        log_days = np.log10(base_df["AbsDays"].to_numpy(dtype=float))
        baseline, baseline_log, _ = evaluate_powerlaw_values(log_days, -16.0, DEFAULT_B)
        base_df["Close"] = baseline * 1000.0
        base_df["LogClose"] = baseline_log + 3.0
        abs_days_only_df = base_df[["AbsDays"]].copy()

        source_aware_df = build_bitcoin_network_simulation(
            base_df,
            seed=7,
            intercept_a=-16.0,
            slope_b=DEFAULT_B,
        )
        abs_days_only_sim_df = build_bitcoin_network_simulation(
            abs_days_only_df,
            seed=7,
            intercept_a=-16.0,
            slope_b=DEFAULT_B,
        )

        self.assertTrue(source_aware_df["Close"].equals(abs_days_only_sim_df["Close"]))

    def test_resolution_controls_path_detail(self):
        dates = pd.date_range(start=GENESIS_DATE + pd.Timedelta(days=1), periods=2500, freq="D")
        base_df = pd.DataFrame(index=dates)
        base_df["AbsDays"] = (dates - GENESIS_DATE).days

        high_resolution_df = build_bitcoin_network_simulation(
            base_df,
            seed=9,
            resolution_days=0.00001,
        )
        low_resolution_df = build_bitcoin_network_simulation(
            base_df,
            seed=9,
            resolution_days=0.01,
        )
        high_resolution_changes = high_resolution_df["LogClose"].diff().abs().mean()
        low_resolution_changes = low_resolution_df["LogClose"].diff().abs().mean()

        self.assertGreater(high_resolution_changes, low_resolution_changes)

    def test_subday_resolution_is_supported(self):
        dates = pd.date_range(start=GENESIS_DATE + pd.Timedelta(days=1), periods=1200, freq="D")
        base_df = pd.DataFrame(index=dates)
        base_df["AbsDays"] = (dates - GENESIS_DATE).days

        sim_df = build_bitcoin_network_simulation(
            base_df,
            seed=11,
            resolution_days=0.00001,
        )

        self.assertEqual(len(sim_df), len(base_df))
        self.assertTrue(np.isfinite(sim_df["Close"]).all())
        self.assertTrue((sim_df["Close"] > 0).all())


if __name__ == "__main__":
    unittest.main()
