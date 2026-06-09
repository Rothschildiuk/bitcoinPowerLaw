import unittest
from unittest.mock import patch

from core.constants import (
    KEY_SIGMA_BAND_HISTORY_YEARS,
    KEY_PORTFOLIO_FORECAST_UNIT,
    PORTFOLIO_VIEW_ACCUMULATION,
    PORTFOLIO_VIEW_PENSION,
)
from ui import sidebar


class TestUISidebarHelpers(unittest.TestCase):
    def test_sync_portfolio_forecast_unit_defaults_to_month_when_entering_accumulation(
        self,
    ):
        session_state = {
            sidebar.KEY_LAST_PORTFOLIO_VIEW: PORTFOLIO_VIEW_PENSION,
            KEY_PORTFOLIO_FORECAST_UNIT: "Year",
        }

        with patch.object(sidebar.st, "session_state", session_state):
            sidebar._sync_portfolio_forecast_unit_default(PORTFOLIO_VIEW_ACCUMULATION)

        self.assertEqual(session_state[KEY_PORTFOLIO_FORECAST_UNIT], "Month")
        self.assertEqual(
            session_state[sidebar.KEY_LAST_PORTFOLIO_VIEW], PORTFOLIO_VIEW_ACCUMULATION
        )

    def test_sync_portfolio_forecast_unit_preserves_manual_accumulation_choice(self):
        session_state = {
            sidebar.KEY_LAST_PORTFOLIO_VIEW: PORTFOLIO_VIEW_ACCUMULATION,
            KEY_PORTFOLIO_FORECAST_UNIT: "Year",
        }

        with patch.object(sidebar.st, "session_state", session_state):
            sidebar._sync_portfolio_forecast_unit_default(PORTFOLIO_VIEW_ACCUMULATION)

        self.assertEqual(session_state[KEY_PORTFOLIO_FORECAST_UNIT], "Year")

    def test_sigma_band_history_control_uses_native_select_slider_without_default_value(self):
        session_state = {KEY_SIGMA_BAND_HISTORY_YEARS: 99}
        captured = {}

        def capture_select_slider(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

        with (
            patch.object(sidebar.st, "session_state", session_state),
            patch.object(sidebar.st, "select_slider", side_effect=capture_select_slider),
        ):
            sidebar._render_sigma_band_history_sidebar_control(["2020-01-01", "2022-01-01"])

        self.assertEqual(session_state[KEY_SIGMA_BAND_HISTORY_YEARS], 3)
        self.assertEqual(captured["args"][0], "Sigma band history")
        self.assertEqual(captured["kwargs"]["key"], KEY_SIGMA_BAND_HISTORY_YEARS)
        self.assertNotIn("value", captured["kwargs"])


if __name__ == "__main__":
    unittest.main()
