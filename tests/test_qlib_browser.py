"""Tests for Qlib calendar alignment and local archive browsing."""

import io
import json
import sqlite3
import tarfile
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import numpy as np

from ashare.qlib_data import digest, import_qlib, unpack
from ashare.web import Explorer, csv_bytes


def write_qlib_fixture(directory, invalid_calendar=False):
    root = Path(directory)
    market = root / "qlib_bin"
    extra = root / "increment"
    calendar = market / "calendars" / "day.txt"
    calendar.parent.mkdir(parents=True)
    days = ["2026-11-19", "2026-11-20", "2026-11-23"]
    calendar.write_text("\n".join(days) + "\n")
    instruments = market / "instruments"
    instruments.mkdir()
    (instruments / "all.txt").write_text(f"SH600519\t{days[0]}\t{days[-1]}\n")
    (instruments / "csi300.txt").write_text(f"SH600519\t{days[0]}\t{days[-1]}\n")
    fields = {
        "open": [1.1, 1.5, 1.55], "high": [1.6, 1.6, 1.8], "low": [1, 1.4, 1.5],
        "close": [1.5, 1.55, 1.6], "volume": [100, 1000, 1500], "factor": [0.075, 0.075, 0.075],
    }
    for key, values in fields.items():
        path = market / "features" / "sh600519" / f"{key}.day.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.array([0, *values], dtype="<f4").tofile(path)
    for key, values in {"total_mv": [10000, float("nan")], "pb": [1.2, 1.5]}.items():
        path = extra / "features" / "sh600519" / f"{key}.day.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.array([1, *values], dtype="<f4").tofile(path)
    manifest = {
        "base_calendar_sha256": "invalid" if invalid_calendar else digest(calendar),
        "base_calendar_start": days[0], "base_calendar_end": days[-1], "base_calendar_count": 3,
        "files": sorted(p.relative_to(extra).as_posix() for p in extra.rglob("*.bin")),
    }
    (extra / "manifest.json").write_text(json.dumps(manifest))
    packed = [root / name for name in ("qlib_bin.tar.gz", "daily_basic_qlib_features.tar.gz")]
    for path, folder in zip(packed, (market, extra)):
        with tarfile.open(path, "w:gz") as archive:
            archive.add(folder, arcname=folder.name)
    return packed


class QlibBrowserTests(unittest.TestCase):
    def test_backtest_trade_names_use_full_code_and_survive_missing_names(self):
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            trades = root / "runs" / "history" / "main"
            trades.mkdir(parents=True)
            (trades.parent / "summary.json").write_text(json.dumps({
                "kind": "rotation_top10_top20", "variants": {"main": {}},
            }), encoding="utf-8")
            (trades / "trades.csv").write_text(
                "code,side,price\n000001.SZ,buy,10\n000001.SH,buy,20\n600519.SH,sell,30\n", encoding="utf-8")
            (root / "security-names.csv").write_text("code,name\n000001.SZ,平安银行\n", encoding="utf-8")
            with closing(sqlite3.connect(root / "market.sqlite")) as db:
                db.execute("CREATE TABLE securities(code TEXT PRIMARY KEY, name TEXT NOT NULL)")
                db.execute("INSERT INTO securities VALUES (?,?)", ("600519.SH", "贵州茅台"))
                db.commit()
            api = Explorer(root / "market.sqlite", root / "qlib.sqlite", root / "runs")
            rows = api.dispatch("/api/backtest", {"run": "history", "table": "trades"})["rows"]
            self.assertEqual([row["stock_name"] for row in rows], ["平安银行", "", "贵州茅台"])
            self.assertEqual([row["price"] for row in rows], ["10", "20", "30"])
            self.assertIn("平安银行", csv_bytes(rows).decode("utf-8-sig"))

    def test_full_import_and_read_api(self):
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            db = root / "archive.sqlite"
            info = import_qlib(*write_qlib_fixture(root), db, extract_to=root / "extracted",
                               progress=lambda _: None)
            self.assertEqual((info["rows"], info["market_cap_rows"]), (3, 1))
            self.assertEqual(info["fields"]["pb"], 2)
            self.assertTrue((root / "extracted" / "market" / "qlib_bin" / "calendars" / "day.txt").is_file())
            explorer = Explorer(root / "missing-market.sqlite", db)
            catalog = explorer.dispatch("/api/catalog", {})
            self.assertEqual(catalog[0]["years"][0]["price_rows"], 3)
            data = explorer.dispatch("/api/history", {"symbol": "600519.SH"})
            self.assertEqual(data["universes"][0]["universe"], "csi300")
            self.assertEqual([row["market_cap"] for row in data["rows"]], [None, 1e8, None])
            self.assertIn(b"2026-11-20", csv_bytes(data["rows"]))
            cross = explorer.dispatch("/api/daily", {"day": "2026-11-20"})
            self.assertEqual(cross["rows"][0]["code"], "600519.SH")
            self.assertEqual(cross["cap_count"], 1)
            with closing(sqlite3.connect(db)) as connection:
                self.assertEqual(connection.execute("SELECT count(*) FROM qlib_metadata_files").fetchone()[0], 4)

    def test_mismatched_calendar_does_not_publish(self):
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            db = root / "archive.sqlite"
            with self.assertRaisesRegex(ValueError, "交易日历"):
                import_qlib(*write_qlib_fixture(root, invalid_calendar=True), db, progress=lambda _: None)
            self.assertFalse(db.exists())

    def test_unpack_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            with tarfile.open(root / "test.tar", "w") as archive:
                entry = tarfile.TarInfo("../unexpected")
                entry.size = 1
                archive.addfile(entry, io.BytesIO(b"x"))
            with self.assertRaisesRegex(ValueError, "路径越界"):
                unpack(root / "test.tar", root / "output")
            self.assertFalse((root / "unexpected").exists())


if __name__ == "__main__":
    unittest.main()
