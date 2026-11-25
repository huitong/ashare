import unittest

import pandas as pd

from ashare.rotation import GridRanks, should_exit


class LowVolGridTest(unittest.TestCase):
    def test_sell_rule_boundaries(self):
        self.assertFalse(should_exit("top30", 30, None))
        self.assertTrue(should_exit("top30", 31, None))
        self.assertFalse(should_exit("sum50", 25, 25))
        self.assertTrue(should_exit("sum50", 26, 25))
        self.assertFalse(should_exit("sum50", 80, None))

    def test_grid_ranks_fall_back_to_anchor_and_exclude_missing_anchor(self):
        frame = pd.DataFrame({"close": [10, 10, 10, None], "low_volatility": [.3, .2, None, .4],
                              "ridge": [None, .1, .2, 1.]}, index=list("ABCD"))
        result = GridRanks(frame, {"ridge", "low_volatility"}).rank(
            dict.fromkeys(frame.index, {"volume": "证据"}), {"low_volatility": .35, "ridge": .65})
        result = result.set_index("code")
        self.assertEqual(result.loc["A", "rank"], 1)
        self.assertEqual(result.loc["A", "score"], 75.)
        self.assertEqual(result.loc["B", "rank"], 2)
        self.assertTrue(pd.isna(result.loc["C", "rank"]))
        self.assertTrue(pd.isna(result.loc["D", "rank"]))
