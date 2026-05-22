import unittest
from unittest.mock import patch

from core.constants import (
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


if __name__ == "__main__":
    unittest.main()
