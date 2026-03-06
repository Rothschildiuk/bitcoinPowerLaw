import unittest

from scripts.update_powerlaw_defaults import update_constants_content


class TestUpdateDefaultsScript(unittest.TestCase):
    def test_update_constants_content_updates_scalars_and_mapping_entries(self):
        original = """DEFAULT_A = -16.511
OSC_DEFAULTS_HASHRATE = {
    "lambda_val": 4.71,
    "t1_age": 1.69,
    "amp_factor_top": 0.68,
}
"""
        updated = update_constants_content(
            original,
            {
                "DEFAULT_A": "-16.400",
                "OSC_DEFAULTS_HASHRATE.lambda_val": "5.32",
                "OSC_DEFAULTS_HASHRATE.t1_age": "1.51",
            },
        )

        self.assertIn("DEFAULT_A = -16.400", updated)
        self.assertIn('"lambda_val": 5.32,', updated)
        self.assertIn('"t1_age": 1.51,', updated)


if __name__ == "__main__":
    unittest.main()
