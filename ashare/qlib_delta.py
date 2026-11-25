"""Apply paired Qlib temporal deltas to an independent recent-history database."""

import hashlib
import json
import math
import re
import sqlite3
import struct
import tempfile
from collections import defaultdict
from contextlib import closing
from pathlib import Path

from .qlib_data import code_for, date_value, digest, find_single, unpack
from .web import open_readonly

TAIL_SESSIONS = 100
FEATURE = re.compile(r"features/([a-z0-9._-]{1,40})/([a-z_][a-z0-9_]*)\.day\.bin")


def read_delta(archive, expected_hash, folder, dataset):
    if digest(archive) != expected_hash:
        raise ValueError(f"{dataset} 增量包 SHA-256 与 GitHub 发布记录不一致。")
    unpack(archive, folder)
    path = find_single(folder, "*/manifest.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest["dataset"], manifest["package_type"], manifest["schema_version"], manifest["frequency"]) != (
        dataset, "qlib_temporal_delta", 2, "day",
    ):
        raise ValueError(f"{dataset} 不是可用的日线时间增量包。")
    return manifest, (path.parent / "payloads.bin").read_bytes()


def read_updates(manifest, payload, dataset, extracted, fields, known, updates):
    root = find_single(extracted / dataset, "**/features")
    seen = set()
    for item in manifest["files"]:
        path = item["path"]
        match = FEATURE.fullmatch(path)
        if not match or match[2] not in fields or path in seen or item["payload_path"] != "payloads.bin":
            raise ValueError(f"无效或重复的增量字段路径：{path}")
        seen.add(path)
        symbol, field = match.groups()
        key = dataset, path
        operation = item["operation"]
        start, size = item["payload_offset"], item["payload_size"]
        if start < 0 or size <= 0 or size % 4 or start + size > len(payload):
            raise ValueError(f"增量数据范围无效：{path}")
        block = payload[start:start + size]
        if operation == "append":
            old_size, old_hash = item["expected_old_size"], item["expected_old_sha256"]
            prior = known.get(key)
            if prior is None:
                original = (root / symbol / Path(path).name).read_bytes()
                if len(original) < 8 or len(original) != old_size or hashlib.sha256(original).hexdigest() != old_hash:
                    raise ValueError(f"本地原始二进制文件与增量基线不一致：{path}")
                first, = struct.unpack_from("<f", original)
                if not first.is_integer() or first < 0:
                    raise ValueError(f"增量文件起始下标无效：{path}")
                prior = int(first), old_size, old_hash
                if hashlib.sha256(original + block).hexdigest() != item["new_sha256"]:
                    raise ValueError(f"追加后的二进制校验失败：{path}")
            first, local_size, local_hash = prior
            if local_size != old_size or local_hash != old_hash or old_size + size != item["new_size"]:
                raise ValueError(f"增量基线与上次同步不一致：{path}")
            position = first + old_size // 4 - 1
            new_size = old_size + size
        elif operation == "create":
            if key in known or size < 8 or size != item["new_size"] or hashlib.sha256(block).hexdigest() != item["new_sha256"]:
                raise ValueError(f"新证券文件校验失败：{path}")
            offset, = struct.unpack_from("<f", block)
            if not offset.is_integer() or offset < 0:
                raise ValueError(f"新证券起始下标无效：{path}")
            first = position = int(offset)
            new_size, block = size, block[4:]
        else:
            raise ValueError(f"不支持的增量操作：{operation}")
        if (len(block) // 4 != item["append_value_count"]
                or position + len(block) // 4 > manifest["calendar_after_count"]):
            raise ValueError(f"增量数据与目标日历不一致：{path}")
        for offset, (value,) in enumerate(struct.iter_unpack("<f", block), position):
            if math.isfinite(value):
                row = updates[offset, symbol]
                if field in row:
                    raise ValueError(f"两个增量包包含相同字段：{path}")
                row[field] = value
        known[key] = first, new_size, item["new_sha256"]


def apply_delta(base, output, extracted, market_archive, basic_archive, market_hash, basic_hash, progress):
    """Publish a verified append-only tail while keeping the historical database intact."""
    existing = output if output.is_file() else base
    if base.resolve() == output.resolve():
        raise ValueError("交易参考库需要使用与历史数据库不同的路径。")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".qlib-trade-", dir=output.parent) as temp:
        folder = Path(temp)
        market, market_payload = read_delta(market_archive, market_hash, folder / "market", "market")
        basic, basic_payload = read_delta(basic_archive, basic_hash, folder / "daily_basic", "daily_basic")
        alignment = ("release_tag", "base_release_tag", "calendar_before_count", "calendar_before_sha256",
                     "calendar_after_count", "calendar_after_sha256", "calendar_append")
        if any(market[key] != basic[key] for key in alignment):
            raise ValueError("两个增量包的发布标签或交易日历不一致。")
        release = date_value(market["release_tag"])
        with closing(open_readonly(existing)) as old:
            old.execute("BEGIN")
            summary = json.loads(old.execute("SELECT value FROM qlib_meta WHERE key='summary'").fetchone()[0])
            days = [row[0] for row in old.execute("SELECT day FROM qlib_calendar ORDER BY position")]
            if summary["release"] == release:
                return summary
            appended = [date_value(x) for x in market["calendar_append"]]
            calendar = lambda values: hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()
            if (summary["release"] != market["base_release_tag"] or release <= summary["release"]
                    or len(days) != market["calendar_before_count"] or calendar(days) != market["calendar_before_sha256"]
                    or not appended or appended[0] <= days[-1] or appended != sorted(set(appended)) or appended[-1] > release):
                raise ValueError("增量包基线、日期或日历不匹配。请按发布顺序补齐中间版本的增量。")
            days.extend(appended)
            if len(days) != market["calendar_after_count"] or calendar(days) != market["calendar_after_sha256"]:
                raise ValueError("合并后的交易日历与发布记录不一致。")
            fields = {row[1] for row in old.execute("PRAGMA table_info(qlib_daily)")} - {"day", "symbol"}
            known = {}
            if existing == output:
                known = {(dataset, path): (first, size, sha) for dataset, path, first, size, sha in
                         old.execute("SELECT dataset,path,first,size,sha256 FROM qlib_delta_features")}
            updates = defaultdict(dict)
            progress(f"正在验证 {release} 增量的二进制字段。")
            read_updates(market, market_payload, "market", extracted, fields, known, updates)
            read_updates(basic, basic_payload, "daily_basic", extracted, fields, known, updates)
            temporary = folder / "trade.sqlite"
            cutoff = days[max(0, len(days) - TAIL_SESSIONS)]
            with closing(sqlite3.connect(temporary)) as db:
                db.execute("ATTACH DATABASE ? AS prior", (existing.resolve().as_uri() + "?mode=ro",))
                for table in ("qlib_calendar", "qlib_symbols", "qlib_daily", "qlib_meta"):
                    schema = old.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()[0]
                    db.execute(schema)
                    where = " WHERE day >= ?" if table == "qlib_daily" else ""
                    db.execute(f"INSERT INTO {table} SELECT * FROM prior.{table}{where}", (cutoff,) if where else ())
                db.executemany("INSERT INTO qlib_calendar VALUES (?,?)", ((day, i) for i, day in enumerate(days) if day in appended))
                touched = set()
                for (position, symbol), values in sorted(updates.items()):
                    day = days[position]
                    if day < cutoff:
                        if existing == base:
                            continue
                        raise ValueError("增量修改目标超出参考库历史窗口。")
                    columns = ",".join(f'"{field}"' for field in values)
                    assignments = ",".join(f'"{field}"=excluded."{field}"' for field in values)
                    db.execute(f"INSERT INTO qlib_daily (symbol,day,{columns}) VALUES (?,?{',?' * len(values)})"
                               f" ON CONFLICT(symbol,day) DO UPDATE SET {assignments}", (symbol, day, *values.values()))
                    touched.add(symbol)
                for symbol in touched:
                    first, last, count, priced = db.execute(
                        "SELECT min(day),max(day),count(*),sum(close > 0) FROM qlib_daily WHERE symbol=?", (symbol,)).fetchone()
                    db.execute("INSERT INTO qlib_symbols VALUES (?,?,?,?,?,?) ON CONFLICT(symbol) DO UPDATE SET"
                               " first_day=min(first_day,excluded.first_day), last_day=max(last_day,excluded.last_day),"
                               " feature_rows=excluded.feature_rows, price_rows=excluded.price_rows",
                               (symbol, code_for(symbol), first, last, count, priced or 0))
                db.execute("CREATE INDEX qlib_daily_calendar ON qlib_daily(day, symbol)")
                db.execute("CREATE TABLE qlib_delta_features(dataset TEXT, path TEXT, first INTEGER, size INTEGER,"
                           " sha256 TEXT, PRIMARY KEY(dataset,path)) WITHOUT ROWID")
                db.executemany("INSERT INTO qlib_delta_features VALUES (?,?,?,?,?)",
                               ((*key, *value) for key, value in known.items()))
                last_prices, last_cap = db.execute("SELECT count(*), count(CASE WHEN total_mv>0 THEN 1 END)"
                                                   " FROM qlib_daily WHERE day=? AND close>0", (appended[-1],)).fetchone()
                if not last_prices or not last_cap:
                    raise ValueError("最后一个交易日缺少行情或估值基础数据。")
                result = {"source": summary["source"], "release": release,
                          "sha256": {"market": market_hash, "daily_basic": basic_hash},
                          "base_release": summary["release"], "last_day": appended[-1],
                          "last_day_prices": last_prices, "last_day_caps": last_cap}
                db.execute("UPDATE qlib_meta SET value=? WHERE key='summary'", (json.dumps(result, ensure_ascii=False),))
                db.commit()
                if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                    raise ValueError("交易参考库完整性校验失败。")
        temporary.replace(output)
    return result
