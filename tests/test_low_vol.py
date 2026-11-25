import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ashare.low_vol import low_vol_features, tradable_label


class LowVolTest(unittest.TestCase):
    def test_moving_average_slope_uses_at_most_sixty_prices(self):
        prices = pd.Series(np.exp(np.arange(75) * .001))
        financials = [np.ones(len(prices))] * 4
        before = low_vol_features(prices, *financials)
        changed = prices.copy()
        changed.iloc[10] *= 4
        changed.iloc[71] *= 3
        after = low_vol_features(changed, *financials)
        for column in ("slope_mean", "slope_std", "slope_up", "slope_quality", "vol60", "mom59"):
            self.assertAlmostEqual(before[column].iloc[70], after[column].iloc[70])
        self.assertAlmostEqual(before["slope_mean"].iloc[70], .001)
        self.assertEqual(before["slope_up"].iloc[70], 1)

    def test_label_skips_entry_and_exit_at_price_limits(self):
        opens = np.ones((3, 7)) * 10
        opens[:, 6] = [11, 11, 9]
        ceiling, floor = np.ones((3, 7)) * 12, np.ones((3, 7)) * 8
        ceiling[1, 1] = 10
        floor[2, 6] = 9
        features = SimpleNamespace(values={"open": opens, "limit_up": ceiling, "limit_down": floor,
                                           "adj": np.ones_like(opens)})
        result = tradable_label(features, np.arange(3), 0)
        self.assertAlmostEqual(result[0], .1)
        self.assertTrue(np.isnan(result[1:]).all())
