import unittest

from ui.viewport import user_agent_is_mobile


class TestUserAgentIsMobile(unittest.TestCase):
    def test_phone_user_agents_are_mobile(self):
        phones = (
            "Mozilla/5.0 (Linux; Android 14; Pixel 8 Pro) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/152.0.0.0 Mobile Safari/537.36",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 "
            "(KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1",
        )

        for user_agent in phones:
            with self.subTest(user_agent=user_agent):
                self.assertTrue(user_agent_is_mobile(user_agent))

    def test_desktop_and_tablet_user_agents_are_not_mobile(self):
        # An Android tablet omits "Mobile", which is what keeps it on the wide layout.
        wide_clients = (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Linux; Android 14; Pixel Tablet) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36",
        )

        for user_agent in wide_clients:
            with self.subTest(user_agent=user_agent):
                self.assertFalse(user_agent_is_mobile(user_agent))

    def test_missing_user_agent_falls_back_to_the_wide_layout(self):
        self.assertFalse(user_agent_is_mobile(None))
        self.assertFalse(user_agent_is_mobile(""))


if __name__ == "__main__":
    unittest.main()
