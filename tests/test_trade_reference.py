import hashlib
import struct
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

from ashare.qlib_delta import read_updates
from ashare.trade_reference import DEFAULT_STRATEGY, portfolio_choices


class TradeReferenceTest(unittest.TestCase):
    def test_default_keeps_backtest_weights_without_duplicating_exit_rules(self):
        strategies = portfolio_choices()
        self.assertEqual(len(strategies), 16)
        self.assertEqual(strategies[DEFAULT_STRATEGY][1], {"low_volatility": .35, "mlp_value": .65})

    def test_delta_uses_original_offset_and_leaves_missing_observation_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = "features/sh600000/close.day.bin"
            old_path = root / "market" / path
            old_path.parent.mkdir(parents=True)
            old = struct.pack("<ff", 3, 10)
            payload = struct.pack("<ff", float("nan"), 12)
            old_path.write_bytes(old)
            checksum = lambda data: hashlib.sha256(data).hexdigest()
            item = dict(path=path, payload_path="payloads.bin", payload_offset=0, payload_size=len(payload),
                        operation="append", expected_old_size=len(old), expected_old_sha256=checksum(old),
                        new_size=len(old) + len(payload), new_sha256=checksum(old + payload), append_value_count=2)
            manifest = {"files": [item], "calendar_after_count": 6}
            known, updates = {}, defaultdict(dict)
            read_updates(manifest, payload, "market", root, {"close"}, known, updates)
            self.assertEqual(updates, {(5, "sh600000"): {"close": 12}})
            self.assertEqual(known["market", path], (3, 16, checksum(old + payload)))
            with self.assertRaisesRegex(ValueError, "上次同步不一致"):
                read_updates(manifest, payload, "market", root, {"close"}, known, defaultdict(dict))
