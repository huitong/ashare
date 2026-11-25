import unittest
from types import SimpleNamespace

import pandas as pd

from ashare.backtest import Position
from ashare.config import Strategy
from ashare.intraday import EXECUTION_RULES, limit_fill
from ashare.rotation import ReplayBook


def bar(**changes):
    fields = dict(open=10, high=10.12, low=9.88, close=10, adj=1, volume=1_000_000,
                  limit_up=11, limit_down=9, limit_known=True, official_limit=False)
    return SimpleNamespace(**(fields | changes))


class IntradayTest(unittest.TestCase):
    def test_limit_price_rounding_gaps_missing_data_and_price_limits(self):
        lag = bar(close=10.03)
        self.assertEqual(limit_fill(bar(low=9.93), lag, False), (None, 9.92, "intraday_target_not_reached"))
        self.assertEqual(limit_fill(bar(low=9.92), lag, False), (9.92, 9.92, None))
        self.assertEqual(limit_fill(bar(open=9.89, low=9.88), lag, False), (9.89, 9.92, None))
        self.assertEqual(limit_fill(bar(high=10.13), lag, True), (None, 10.14, "intraday_target_not_reached"))
        self.assertEqual(limit_fill(bar(open=10.25, high=10.26), lag, True), (10.25, 10.14, None))
        self.assertEqual(limit_fill(bar(adj=1.1), lag, True), (None, None, "no_comparable_previous_close"))
        self.assertEqual(limit_fill(bar(open=9.9, limit_up=9.9), bar(), False),
                         (None, 9.9, "intraday_fill_at_limit"))

    def test_sells_at_threshold_and_replaces_missed_entry_by_rank(self):
        for mode in ("both", "both_reserved"):
            with self.subTest(mode=mode):
                replay = ReplayBook(SimpleNamespace(strategy=Strategy(slippage_bps=0), securities={}),
                                    "sum50", EXECUTION_RULES[mode][1])
                codes = [f"000{i:03d}.SZ" for i in range(12)]
                forced, optional, keep_at_30, missed, backup, *keep = codes
                replay.book.positions = {code: Position(100, 1, 10) for code in (forced, optional, keep_at_30, *keep)}
                ranks = pd.DataFrame({"code": codes, "rank": [26, 25, 15, 2, 11, *range(3, 10)],
                                      "score": range(12), "routes": ["holdings"] * 12})
                ranks = ranks.sort_values("rank")
                replay.previous_ranks = {forced: 25, optional: 25, keep_at_30: 15}
                replay.record("2026-01-05", dict.fromkeys(codes, bar()), 1, ranks)
                replay.execute("2026-01-06", dict.fromkeys(codes, bar()) | {missed: bar(low=9.95)})
                expected = ([("sell", forced, 10), ("buy", backup, 9.9), ("sell", optional, 10.1)]
                            if mode == "both" else [("sell", forced, 10), ("sell", optional, 10.1)])
                self.assertEqual([(t[3], t[2], t[5]) for t in replay.trades], expected)
                self.assertIn(keep_at_30, replay.book.positions)
                self.assertNotIn(missed, replay.book.positions)

    def test_optional_sell_does_not_create_a_same_day_buy_slot(self):
        replay = ReplayBook(SimpleNamespace(strategy=Strategy(slippage_bps=0), securities={}),
                            "sum50", EXECUTION_RULES["both"][1])
        holdings = [f"000{i:03d}.SZ" for i in range(10)]
        new = "000011.SZ"
        replay.book.positions = {code: Position(100, 1, 10) for code in holdings}
        replay.previous_ranks = {holdings[0]: 15}
        ranks = pd.DataFrame({"code": [new, *holdings], "rank": [1, 16, *range(20, 29)],
                              "score": range(11), "routes": ["volume"] * 11})
        bars = dict.fromkeys([new, *holdings], bar())
        replay.record("2026-01-05", bars, 1, ranks)
        replay.execute("2026-01-06", bars)
        self.assertEqual([(t[2], t[3]) for t in replay.trades], [(holdings[0], "sell")])

    def test_slippage_is_charged_to_fills_without_crossing_order_limits(self):
        replay = ReplayBook(SimpleNamespace(strategy=Strategy(slippage_bps=10), securities={}),
                            "sum50", EXECUTION_RULES["both"][1])
        hard_exit, limit_exit, entry = "000001.SZ", "000002.SZ", "000003.SZ"
        replay.book.positions = {code: Position(100, 1, 10) for code in (hard_exit, limit_exit)}
        replay.previous_ranks = {hard_exit: 26, limit_exit: 16}
        ranks = pd.DataFrame({"code": [entry, limit_exit, hard_exit], "rank": [1, 16, 25],
                              "score": [.9, .8, .7], "routes": ["volume", "holdings", "holdings"]})
        replay.record("2026-01-05", dict.fromkeys(ranks["code"], bar()), 1, ranks)
        replay.execute("2026-01-06", {hard_exit: bar(), limit_exit: bar(open=10.11, high=10.12),
                                    entry: bar(open=9.9, low=9.88)})
        trades = {trade[2]: trade for trade in replay.trades}
        self.assertAlmostEqual(trades[hard_exit][5], 9.99)
        self.assertAlmostEqual(trades[hard_exit][-1], 1)
        self.assertAlmostEqual(trades[entry][5], 9.9)
        self.assertAlmostEqual(trades[entry][-1], 0)
        self.assertAlmostEqual(trades[limit_exit][5], 10.1)
        self.assertAlmostEqual(trades[limit_exit][-1], 1)
