"""Import complete Qlib releases into a queryable local SQLite warehouse."""

import hashlib
import json
import os
import re
import shutil
import sqlite3
import tarfile
import tempfile
from contextlib import closing
from datetime import date
from pathlib import Path, PurePosixPath

import numpy as np

DEFAULT_QLIB_DB = Path("data/qlib.sqlite")
DEFAULT_RELEASE = "2026-11-23"
RELEASE_HASHES = {
    DEFAULT_RELEASE: (
        "c5f484c8d76442fd4adca2ce87ce9508851d437011103f923996fc9cec33872e",
        "daff41f1a5c2aa1040625745190ace75b5a75029c236a78557f3130d6ff3fcf7",
    ),
}
FEATURE_NAME = re.compile(r"[a-z_][a-z0-9_]*")
SYMBOL_NAME = re.compile(r"[A-Za-z0-9._-]{1,40}")
STOCK = re.compile(r"(sh|sz|bj)([0-9]{6})", re.I)
PRICE_FIELDS = ("open", "high", "low", "close")
DB_SCHEMA = """
CREATE TABLE qlib_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE qlib_calendar(day TEXT PRIMARY KEY, position INTEGER NOT NULL UNIQUE);
CREATE TABLE qlib_symbols(
    symbol TEXT PRIMARY KEY, code TEXT NOT NULL UNIQUE, first_day TEXT, last_day TEXT,
    feature_rows INTEGER NOT NULL, price_rows INTEGER NOT NULL
);
CREATE TABLE qlib_memberships(
    universe TEXT NOT NULL, symbol TEXT NOT NULL, start TEXT NOT NULL, end TEXT NOT NULL,
    PRIMARY KEY(universe, symbol, start, end)
) WITHOUT ROWID;
CREATE TABLE qlib_metadata_files(
    package TEXT NOT NULL, path TEXT NOT NULL, content BLOB NOT NULL,
    PRIMARY KEY(package, path)
) WITHOUT ROWID;
"""


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def unpack(path, destination):
    """Extract only regular files and directories. Never apply archive permissions."""
    destination.mkdir()
    root = destination.resolve()
    with tarfile.open(path, mode="r|*") as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError(f"压缩包路径越界：{member.name}")
            target = root.joinpath(*name.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isreg():
                raise ValueError(f"压缩包包含不支持的文件类型：{member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as data, target.open("xb") as output:
                shutil.copyfileobj(data, output)


def find_single(root, pattern):
    matches = list(root.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"{root.name} 中应有且仅有一个 {pattern}。")
    return matches[0]


def date_value(value):
    return date.fromisoformat(value).isoformat()


def calendar_at(calendar_path):
    days = [
        date_value(line.strip()) for line in calendar_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not days or any(a >= b for a, b in zip(days, days[1:])):
        raise ValueError("Qlib 交易日历为空、含重复日期或顺序错误。")
    return days


def index_features(root):
    entries = {}
    for path in root.glob("*/*.day.bin"):
        symbol = path.parent.name.lower()
        field = path.name.removesuffix(".day.bin").lower()
        if not SYMBOL_NAME.fullmatch(symbol) or not FEATURE_NAME.fullmatch(field) or field in {"day", "symbol"}:
            raise ValueError(f"不支持的 Qlib 文件名：{path.name}")
        columns = entries.setdefault(symbol, {})
        if field in columns:
            raise ValueError(f"{symbol} 的 {field} 重复。")
        columns[field] = path
    if not entries:
        raise ValueError(f"{root} 没有可读的 Qlib 日频特征文件。")
    unexpected = [path for path in root.rglob("*.bin") if len(path.relative_to(root).parts) != 2
                  or not path.name.endswith(".day.bin")]
    if unexpected:
        raise ValueError(f"数据包含有无法导入的二进制文件：{unexpected[0]}")
    return entries


def validate_pair(manifest, calendar_path, days, published_files, release):
    actual = (digest(calendar_path), days[0], days[-1], len(days))
    expected = tuple(manifest.get(f"base_calendar_{key}") for key in ("sha256", "start", "end", "count"))
    if expected != actual:
        raise ValueError("每日指标包的交易日历与行情主包不一致。请使用同一发布版本的全量包。")
    declared = manifest.get("files")
    if isinstance(declared, list) and all(isinstance(item, str) for item in declared):
        if sorted(declared) != sorted(published_files):
            raise ValueError("实际每日指标文件与 manifest 列出的文件不一致。")
    else:
        raise ValueError("每日指标 manifest 缺少文件清单。")
    compatible = manifest.get("compatible_base_release")
    if release and compatible and compatible != release:
        raise ValueError(f"每日指标包标注适用于 {compatible}，和所选 {release} 不一致。")


def code_for(symbol):
    match = STOCK.fullmatch(symbol)
    return f"{match[2]}.{match[1].upper()}" if match else symbol.upper()


def search_term(value):
    parts = value.strip().upper().split(".")
    if len(parts) == 2 and parts[1] in {"SH", "SZ", "BJ"} and parts[0].isdigit():
        return f"{parts[1]}{parts[0]}"
    return value.strip().upper()


def read_series(path, calendar_count):
    if path.stat().st_size % 4:
        raise ValueError(f"Qlib 二进制字节长度无效：{path.name}")
    values = np.fromfile(path, dtype="<f4")
    if len(values) < 2 or not np.isfinite(values[0]):
        raise ValueError(f"Qlib 二进制为空或缺少起始下标：{path.name}")
    offset = int(values[0])
    if offset != values[0] or offset < 0 or offset + len(values) - 1 > calendar_count:
        raise ValueError(f"Qlib 序列超出交易日历：{path.parent.name}/{path.name}")
    return offset, values[1:]


def membership_rows(instrument_root):
    for group in sorted(instrument_root.glob("*.txt")):
        for line in group.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 3 or not SYMBOL_NAME.fullmatch(parts[0]):
                raise ValueError(f"证券名单格式无效：{group.name}")
            start, end = map(date_value, parts[1:])
            if start > end:
                raise ValueError(f"证券名单日期逆序：{group.name}")
            yield group.stem, parts[0].lower(), start, end


def import_extracted(db, market, extra, source, progress):
    calendar_path = find_single(market, "**/calendars/day.txt")
    provider = calendar_path.parent.parent
    manifest_path = find_single(extra, "**/manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    days = calendar_at(calendar_path)
    market_features = index_features(provider / "features")
    extra_features = index_features(manifest_path.parent / "features")
    extra_files = [
        path.relative_to(manifest_path.parent).as_posix()
        for columns in extra_features.values() for path in columns.values()
    ]
    validate_pair(manifest, calendar_path, days, extra_files, source.get("release"))
    for symbol, fields in extra_features.items():
        combined = market_features.setdefault(symbol, {})
        if combined.keys() & fields.keys():
            raise ValueError(f"{symbol} 的字段同时出现在两个包中。")
        combined.update(fields)
    fields = sorted({field for cols in market_features.values() for field in cols})
    if not {*PRICE_FIELDS, "volume", "total_mv"}.issubset(fields):
        raise ValueError("数据包缺少 OHLC、成交量或总市值字段。")

    db.executescript(DB_SCHEMA)
    schema = ", ".join(f'"{field}" REAL' for field in fields)
    db.execute(f"""CREATE TABLE qlib_daily (
        symbol TEXT NOT NULL, day TEXT NOT NULL, {schema}, PRIMARY KEY(symbol, day)
    ) WITHOUT ROWID""")
    db.executemany("INSERT INTO qlib_calendar VALUES (?,?)", ((day, i) for i, day in enumerate(days)))
    db.executemany("INSERT OR IGNORE INTO qlib_memberships VALUES (?,?,?,?)",
                   membership_rows(provider / "instruments"))
    for package, root in (("market", market), ("daily_basic", extra)):
        db.executemany(
            "INSERT INTO qlib_metadata_files VALUES (?,?,?)",
            ((package, item.relative_to(root).as_posix(), item.read_bytes())
             for item in root.rglob("*") if item.is_file() and not item.name.endswith(".day.bin")),
        )
    placeholders = ",".join("?" for _ in range(len(fields) + 2))
    counters = np.zeros(len(fields), dtype=np.int64)
    field_position = {key: fields.index(key) for key in fields}
    close_idx, cap_idx = field_position["close"], field_position["total_mv"]
    op_idx, hi_idx, lo_idx = (field_position[k] for k in ("open", "high", "low"))
    rows_total = price_total = cap_total = bad_ohlc = price_symbols = 0
    for number, (symbol, columns) in enumerate(sorted(market_features.items()), start=1):
        values = np.full((len(days), len(fields)), np.nan, dtype=np.float64)
        for field, path in columns.items():
            offset, series = read_series(path, len(days))
            values[offset:offset + len(series), field_position[field]] = series
        valid = np.isfinite(values)
        counters += valid.sum(axis=0)
        present = np.flatnonzero(valid.any(axis=1))
        values[~valid] = np.nan
        close_mask = valid[:, close_idx] & (values[:, close_idx] > 0)
        cap_mask = valid[:, cap_idx] & (values[:, cap_idx] > 0)
        bounds = valid[:, [op_idx, hi_idx, lo_idx, close_idx]].all(axis=1)
        bad_ohlc += int((bounds & (
            (values[:, lo_idx] <= 0)
            | (values[:, hi_idx] < np.maximum(values[:, op_idx], values[:, close_idx]))
            | (values[:, lo_idx] > np.minimum(values[:, op_idx], values[:, close_idx]))
        )).sum())
        rows_total += len(present)
        price_rows = int(close_mask.sum())
        price_total += price_rows
        price_symbols += int(price_rows > 0)
        cap_total += int((close_mask & cap_mask).sum())
        first, last = (days[present[0]], days[present[-1]]) if len(present) else (None, None)
        db.execute("INSERT INTO qlib_symbols VALUES (?,?,?,?,?,?)",
                   (symbol, code_for(symbol), first, last, len(present), price_rows))
        db.executemany(
            f"INSERT INTO qlib_daily VALUES ({placeholders})",
            ((symbol, days[i], *row) for i, row in zip(present, values[present].tolist())),
        )
        if number % 500 == 0:
            db.commit()
            progress(f"已入库 {number:,}/{len(market_features):,} 个证券，共 {rows_total:,} 行。")

    progress("正在建立按日查询索引并汇总覆盖情况。")
    db.executescript("""
        CREATE INDEX qlib_daily_calendar ON qlib_daily(day, symbol);
        CREATE TABLE qlib_day_stats AS
        SELECT day, count(*) AS observations,
          count(CASE WHEN close > 0 THEN 1 END) AS prices,
          count(CASE WHEN close > 0 AND total_mv > 0 THEN 1 END) AS market_caps,
          sum(CASE WHEN close > 0 AND total_mv > 0 THEN total_mv END) AS total_mv
        FROM qlib_daily GROUP BY day;
        CREATE UNIQUE INDEX qlib_day_stats_by_day ON qlib_day_stats(day);
    """)
    db.execute("CREATE INDEX qlib_memberships_by_symbol ON qlib_memberships(symbol,universe,start)")
    db.execute("ANALYZE")
    observed = db.execute("SELECT min(day), max(day), count(*) FROM qlib_day_stats").fetchone()
    markets = db.execute("""
        SELECT CASE WHEN symbol GLOB 'sh[0-9]*' THEN 'SH'
                    WHEN symbol GLOB 'sz[0-9]*' THEN 'SZ'
                    WHEN symbol GLOB 'bj[0-9]*' THEN 'BJ' ELSE '其他' END,
               count(*), sum(price_rows)
        FROM qlib_symbols GROUP BY 1
    """).fetchall()
    stats = dict(
        source=source.get("source", "xu-duqing/investment_data"), release=source.get("release") or days[-1],
        calendar_start=days[0], calendar_end=days[-1], calendar_sessions=len(days),
        first_day=observed[0], last_day=observed[1], observed_sessions=observed[2],
        symbols=len(market_features), price_symbols=price_symbols, rows=rows_total,
        price_rows=price_total, market_cap_rows=cap_total,
        invalid_ohlc_rows=bad_ohlc, fields=dict(zip(fields, map(int, counters))), markets=markets,
        universes=db.execute("SELECT universe, count(*) FROM qlib_memberships GROUP BY 1 ORDER BY 1").fetchall(),
        sha256=source.get("sha256", {}),
    )
    db.execute("INSERT INTO qlib_meta VALUES ('summary',?)", (json.dumps(stats, ensure_ascii=False),))
    db.commit()
    if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
        raise ValueError("导入后 SQLite 完整性检查失败。")
    return stats


def import_qlib(market_path, basics_path, output, release=None, extract_to=None, progress=print):
    market_path, basics_path, output = map(Path, (market_path, basics_path, output))
    if output.exists():
        raise ValueError(f"数据库已存在：{output}。请给新版本指定独立路径。")
    if extract_to is not None and extract_to.exists():
        raise ValueError(f"解压目录已存在：{extract_to}")
    hashes = {"market": digest(market_path), "daily_basic": digest(basics_path)}
    if release in RELEASE_HASHES and tuple(hashes.values()) != RELEASE_HASHES[release]:
        raise ValueError(f"文件 SHA-256 不匹配 {release} 的已核对 GitHub Release。")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".qlib-import-", dir=output.parent) as staging:
        staging = Path(staging)
        root = staging / "extracted"
        root.mkdir()
        progress("正在解压行情包和每日指标包。")
        unpack(market_path, root / "market")
        unpack(basics_path, root / "daily_basic")
        database = staging / "data.sqlite"
        with closing(sqlite3.connect(database)) as db:
            db.execute("PRAGMA journal_mode=OFF")
            db.execute("PRAGMA synchronous=OFF")
            db.execute("PRAGMA temp_store=FILE")
            info = import_extracted(db, root / "market", root / "daily_basic",
                                    {"release": release, "sha256": hashes}, progress)
        if extract_to is not None:
            extract_to.parent.mkdir(parents=True, exist_ok=True)
            if extract_to.exists():
                raise ValueError(f"解压目录已存在：{extract_to}")
            shutil.move(str(root), str(extract_to))
        # Publish a completed database atomically without replacing an existing file.
        os.link(database, output)
    return info
