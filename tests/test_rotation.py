import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ashare.config import Strategy
from ashare.rotation import MAIN, RECALL_SIZE, ReplayBook, capped_recall, executable_limits, index_benchmarks


class RotationTest(unittest.TestCase):
    def test_recall_capped_with_missing_quote_for_holder(self):
        names = [f"S{i:04d}" for i in range(RECALL_SIZE + 5)]
        signals = {names[4]: {"volume": "recent volume increase"}}
        found, _ = capped_recall(signals, {}, names, {"NOQUOTE": object()})
        self.assertEqual(len(found), RECALL_SIZE)
        self.assertEqual(list(found)[:2], ["NOQUOTE", names[4]])
        self.assertIn("holdings", found["NOQUOTE"])
        self.assertNotIn(names[-1], found)

    def test_execution_uses_preceding_st_flag_or_published_limit(self):
        dates = np.array(["2026-03-09", "2026-03-10", "2026-03-11"])
        status = np.full((3, 3), np.nan)
        status[2, 1:] = (10.8, 9.2)
        upper, lower, actual = executable_limits(
            dates, "2020-01-01", MAIN, np.array([10, 10, 10]), np.array([10, 6, 10]),
            np.ones(3), np.ones(3, dtype=bool), status, np.array([True, False, True]),
        )
        self.assertEqual(upper[1], 10.5)
        self.assertEqual(lower[1], 9.5)
        self.assertEqual(upper[2], 10.8)
        self.assertTrue(actual[2])

    def test_outside_top_20_sell_and_blocked_replacement(self):
        replay = ReplayBook(SimpleNamespace(strategy=Strategy(slippage_bps=0), securities={}))
        from ashare.backtest import Position
        keep, drop, blocked, buy = "000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"
        price = SimpleNamespace(close=10, open=10, limit_known=True, limit_up=11, limit_down=9,
                                adj=1, volume=1_000_000, official_limit=False)
        at_floor = SimpleNamespace(**{**vars(price), "open": 9})
        bars = dict.fromkeys((keep, drop, blocked, buy), price)
        replay.book.positions = {code: Position(100, 1, 10) for code in (keep, drop, blocked)}
        ranks = pd.DataFrame({"code": [buy, keep, drop, blocked], "rank": [10, 20, 21, 22],
                              "score": [.9, .8, .7, .6], "routes": ["volume", "holdings", "holdings", "holdings"]})
        replay.record("2026-01-05", bars, 1, ranks)
        replay.execute("2026-01-06", {**bars, blocked: at_floor})
        self.assertIn(keep, replay.book.positions)
        self.assertIn(blocked, replay.book.positions)
        self.assertIn(buy, replay.book.positions)
        self.assertNotIn(drop, replay.book.positions)
        self.assertEqual([(r[2], r[3]) for r in replay.trades], [(drop, "sell"), (buy, "buy")])
        self.assertEqual(replay.book.blocked[0][-1], "opening_at_limit")
        self.assertEqual(replay.book.positions[keep].shares, 100)
        for trade in replay.trades:
            self.assertEqual(trade[5], trade[8])
            self.assertAlmostEqual(trade[7], sum(trade[9:12]))
        self.assertEqual(replay.trades[1][11], 0)

    def test_index_calendar_does_not_fill_missing_closes(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "index.csv"
            path.write_text("day,code,close,source\n2026-01-05,000300,100,source\n"
                            "2026-01-07,000300,110,source\n"
                            "2026-01-05,H20269,100,CSI\n2026-01-07,H20269,120,CSI\n"
                            "2026-01-05,H30269,100,CSI\n2026-01-07,H30269,115,CSI\n",
                            encoding="utf-8")
            details, curves = index_benchmarks(path, ["2026-01-05", "2026-01-06", "2026-01-07"])
            self.assertAlmostEqual(details["csi300"]["total_return"], .1)
            self.assertFalse(details["csi300"]["daily"])
            self.assertEqual(curves, {})
            self.assertIsNone(details["star_composite"]["total_return"])
            self.assertAlmostEqual(details["dividend_low_vol_total"]["total_return"], .20)
            self.assertAlmostEqual(details["dividend_low_vol_price"]["total_return"], .15)
            _, sparse = index_benchmarks(path, ["2026-01-05", "2026-01-06", "2026-01-07"], include_sparse=True)
            self.assertEqual(sparse["csi300_reference"], [1, None, 1.1])
            self.assertEqual(sparse["dividend_low_vol_total_reference"], [1, None, 1.2])
            self.assertEqual(sparse["dividend_low_vol_price_reference"], [1, None, 1.15])


if __name__ == "__main__":
    unittest.main()
