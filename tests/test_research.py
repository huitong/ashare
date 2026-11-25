import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ashare.backtest import Book, trade_size
from ashare.config import Strategy
from ashare.demo import create_demo
from ashare.features import Features
from ashare.futu_feed import import_futu
from ashare.providers import FeedError, TushareFeed, synchronize
from ashare.rank import FactorModel, Ranker, Recall
from ashare.store import Store, cutoff
from ashare.temporal import Tokenizer, get_targets, partition


class TimingTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.store = Store(Path(self.temp_dir.name) / "test.sqlite")
        self.days = create_demo(self.store, "2026-11-20", days=65, stocks=12)
        self.code = "600000.SH"

    def test_news_memberships_and_empty_holdings_are_as_of(self):
        day = self.days[-2]
        tomorrow = self.days[-1]
        self.store.put_industries([
            (self.code, "未来行业", day, None, cutoff(tomorrow)),
        ])
        self.store.put_hotspots([
            ("test-before", "code", self.code, day + "T15:00:00+08:00", cutoff(day), 1, "截至收盘已知"),
            ("test-after", "code", self.code, day + "T15:00:00+08:00", cutoff(tomorrow), 50, "收盘以后才入库"),
        ])
        self.store.put_holdings(cutoff(day), [(self.code, 100)])
        self.store.put_holdings(cutoff(tomorrow), [])
        with self.store.snapshot() as market:
            features = Features(market, tomorrow)
            settings = Strategy(weights={"trend": 1}, industry_min_members=1, min_amount=0)
            recall = Recall(market, settings)
            found, industry = recall.candidates(day, features.day(day), market.holdings(day))
            self.assertNotEqual(industry[self.code], "未来行业")
            self.assertIn("截至收盘已知", found[self.code]["news"])
            self.assertIn("holdings", found[self.code])
            self.assertEqual(market.holdings(tomorrow), {})
            ranking = Ranker(market, features, settings, [FactorModel("trend")]).rank(day, market.holdings(day))
            self.assertTrue(ranking.set_index("code").loc[self.code, "held"])

    def test_partial_coverage_cannot_enter_rank(self):
        with self.store.snapshot() as market:
            day = self.days[-1]
            features = Features(market, day)
            settings = Strategy(weights={"trend": 0.35, "macd": 0.65}, min_weight_coverage=0.60)

            class Unavailable:
                model_id = "macd"

                def score(self, day, candidates, features):
                    return pd.Series(np.nan, index=candidates.index)

            ranking = Ranker(market, features, settings, [FactorModel("trend"), Unavailable()]).rank(day, {self.code})
            self.assertGreater(len(ranking), 0)
            self.assertTrue(ranking["score"].isna().all())

    def test_future_rows_do_not_change_factors_and_targets_are_next_session(self):
        with self.store.snapshot() as market:
            past = Features(market, self.days[-5])
            full = Features(market, self.days[-1])
            day = self.days[-10]
            self.assertTrue(past.day(day)["macd"].equals(full.day(day)["macd"]))
            self.assertTrue(past.day(day)["volume_ratio"].equals(full.day(day)["volume_ratio"]))
            context = 8
            groups, train_end, val_end = partition(full, self.days[0], self.days[-1], context, None, None)
            frame = full.frame
            targets = get_targets(full, context)
            for idx in targets:
                self.assertEqual(frame.at[idx, "code"], frame.at[idx - context - 1, "code"])
                self.assertEqual(frame.at[idx, "session_id"] - frame.at[idx - context - 1, "session_id"], context + 1)
            self.assertLessEqual(frame.iloc[groups[0]]["day"].max(), train_end)
            self.assertGreater(frame.iloc[groups[1]]["day"].min(), train_end)
            self.assertGreater(frame.iloc[groups[2]]["day"].min(), val_end)
            first, last = self.days[0], train_end
            tok = Tokenizer.fit(frame, first, last, groups[0])
            altered = frame.copy()
            altered.loc[altered["day"] > last, ["return_1", "log_cap"]] = 999
            self.assertEqual(tok, Tokenizer.fit(altered, first, last, groups[0]))

    def test_incomplete_day_cannot_be_used_as_an_observation(self):
        self.store.register_market("synthetic", [], ["2026-11-23"])
        with self.store.snapshot() as market:
            with self.assertRaisesRegex(ValueError, "未同步"):
                Features(market, "2026-11-23")


class TradingTests(unittest.TestCase):
    def test_opening_execution_uses_lagged_volume_and_blocks_limit_and_suspension(self):
        book = Book(Strategy(participation=0.01, initial_cash=250_000))
        day, signal = "2026-11-24", "2026-11-23"
        names = ["600000.SH", "600001.SH", "600002.SH"]
        normal = SimpleNamespace(open=10, limit_up=11, limit_down=9, limit_known=1, adj=1, volume=1)
        limit = SimpleNamespace(open=11, limit_up=11, limit_down=9, limit_known=1, adj=1, volume=10**9)
        target = dict.fromkeys(names, 10_000)
        yesterday = {code: SimpleNamespace(volume=30_000) for code in names}
        book.rebalance(day, signal, target, {names[0]: normal, names[1]: limit}, yesterday)
        self.assertEqual(len(book.trades), 1)
        self.assertEqual(book.trades[0][4], 300)
        self.assertEqual({row[-1] for row in book.blocked}, {"opening_at_limit", "no_trading_bar"})
        self.assertGreater(book.cash, 0)
        self.assertEqual(trade_size("688001.SH", 237, 400), 237)
        self.assertEqual(trade_size("000001.SZ", 237, 400), 200)


class ProviderTests(unittest.TestCase):
    def test_tushare_converts_units_and_rejects_missing_market_cap(self):
        class Client:
            missing_cap = False

            def query(self, endpoint, fields, **params):
                rows = {
                    "daily": [["600519.SH", 20, 21, 19, 20, 20, 120, 240]],
                    "daily_basic": [] if self.missing_cap else [["600519.SH", 120_000_000]],
                    "adj_factor": [["600519.SH", 3.0]],
                    "stk_limit": [["600519.SH", 22, 18]],
                }
                return pd.DataFrame(rows[endpoint], columns=fields.split(","))

        client = Client()
        adapter = TushareFeed(client=client)
        adapter.interval = 0
        row = adapter.bars("2026-11-23", {"600519.SH"})[0]
        self.assertEqual(row[7:11], (12_000, 240_000, 1_200_000_000_000, 3))
        client.missing_cap = True
        with self.assertRaisesRegex(FeedError, "缺市值"):
            adapter.bars("2026-11-23", {"600519.SH"})

    def test_tushare_resume_preserves_complete_day(self):
        class Feed:
            source = "tushare"
            fail = True

            def calendar(self, start, end):
                return [start, end]

            def securities(self):
                return [("600519.SH", "示例", "2001-01-01", None)]

            def bars(self, day, codes):
                if self.fail and day.endswith("24"):
                    raise FeedError("test")
                return [(day, "600519.SH", 10, 10, 10, 10, 10, 200, 2000, 1e9, 1, 11, 9, 1)]

        with tempfile.TemporaryDirectory() as temp:
            store = Store(Path(temp) / "test.sqlite")
            feed = Feed()
            with self.assertRaises(FeedError):
                synchronize(store, feed, "2026-11-23", "2026-11-24", progress=lambda _: None)
            with store.snapshot() as market:
                self.assertEqual(market.status()["complete_sessions"], 1)
            feed.fail = False
            synced = synchronize(store, feed, "2026-11-23", "2026-11-24", progress=lambda _: None)
            self.assertEqual(synced, 1)

    def test_futu_requires_per_day_reference_values(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            day = "2026-11-23"
            (path / "request.json").write_text(json.dumps({"codes": ["SH.600519"]}))
            (path / "sessions.csv").write_text(f"day\n{day}\n")
            for suffix in ("raw", "hfq"):
                (path / f"SH.600519.{suffix}.csv").write_text(
                    f"time_key,open,high,low,close,last_close,volume,turnover\n{day} 00:00:00,10,12,9,11,10,1000,11000\n"
                )
            securities = path / "securities.csv"
            securities.write_text("code,name,listed,delisted\n600519.SH,Test,2000-01-01,\n")
            ref = path / "ref.csv"
            header = "day,code,market_cap,adj,limit_up,limit_down,limit_known\n"
            ref.write_text(header)
            store = Store(path / "test.sqlite")
            with self.assertRaisesRegex(ValueError, "缺逐日参考"):
                import_futu(store, path, ref, securities)
            ref.write_text(header + f"{day},600519.SH,1000000000,2,11,9,1\n")
            info = import_futu(store, path, ref, securities)
            self.assertEqual(info["bars"], 1)
            with store.snapshot() as market:
                record = market.frame(day, day).iloc[0]
                self.assertEqual(record["market_cap"], 1e9)
                self.assertEqual(record["adj"], 2)


if __name__ == "__main__":
    unittest.main()
