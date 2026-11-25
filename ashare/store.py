"""SQLite persistence for market data and point-in-time research inputs."""

import math
import sqlite3
from contextlib import closing, contextmanager
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

SHANGHAI = ZoneInfo("Asia/Shanghai")
CUTOFF = time(17, 30)
BAR_FIELDS = (
    "day", "code", "open", "high", "low", "close", "prev_close",
    "volume", "amount", "market_cap", "adj", "limit_up", "limit_down", "limit_known",
)


def iso_day(value):
    return date.fromisoformat(value).isoformat()


def timestamp(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=SHANGHAI)
    return parsed.astimezone(SHANGHAI).isoformat(timespec="seconds")


def cutoff(day):
    return datetime.combine(date.fromisoformat(day), CUTOFF, SHANGHAI).isoformat(timespec="seconds")


def latest_completed_wall_day():
    now = datetime.now(SHANGHAI)
    return (now.date() - timedelta(days=now.time() < CUTOFF)).isoformat()


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(Path(__file__).with_name("schema.sql").read_text())

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA foreign_keys=ON")
        return db

    @contextmanager
    def transaction(self):
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            yield db

    @contextmanager
    def snapshot(self):
        with closing(self._connect()) as db, db:
            db.execute("BEGIN")
            yield MarketSnapshot(db)

    def register_market(self, source, securities, sessions):
        with self.transaction() as db:
            current = db.execute("SELECT value FROM meta WHERE key='source'").fetchone()
            if current and current[0] != source:
                raise ValueError(f"行情库已绑定 {current[0]}，此数据源请使用独立数据库。")
            db.execute("INSERT OR IGNORE INTO meta VALUES('source', ?)", (source,))
            db.executemany(
                """INSERT INTO securities VALUES(?,?,?,?) ON CONFLICT(code) DO UPDATE
                   SET name=excluded.name, listed=excluded.listed, delisted=excluded.delisted""",
                securities,
            )
            db.executemany("INSERT OR IGNORE INTO sessions(day) VALUES(?)", ((d,) for d in sessions))

    def save_day(self, day, rows):
        if not rows or any(row[0] != day for row in rows):
            raise ValueError(f"{day} 行情为空或日期不一致。")
        for row in rows:
            _, code, op, high, low, close, prev, vol, amount, cap, adj, up, down, known = row
            required = (op, high, low, close, vol, amount, cap, adj)
            if any(not math.isfinite(x) for x in required):
                raise ValueError(f"{day} {code} 行情含非有限数值。")
            if not (0 < low <= min(op, close) <= max(op, close) <= high):
                raise ValueError(f"{day} {code} OHLC 不一致。")
            if any(x is not None and (not math.isfinite(x) or x <= 0) for x in (prev, up, down)):
                raise ValueError(f"{day} {code} 昨收或涨跌停价格异常。")
            if known and ((up is None) != (down is None) or (up is not None and up < down)):
                raise ValueError(f"{day} {code} 涨跌停价格不完整或顺序异常。")
        with self.transaction() as db:
            if db.execute("SELECT 1 FROM sessions WHERE day=?", (day,)).fetchone() is None:
                raise ValueError(f"{day} 不在交易日历。")
            db.execute("DELETE FROM bars WHERE day=?", (day,))
            placeholders = ",".join("?" for _ in BAR_FIELDS)
            db.executemany(f"INSERT INTO bars VALUES ({placeholders})", rows)
            db.execute("UPDATE sessions SET complete=1 WHERE day=?", (day,))

    def put_industries(self, rows):
        with self.transaction() as db:
            db.executemany(
                """INSERT INTO industry_membership VALUES(?,?,?,?,?)
                   ON CONFLICT(code,industry,effective_from) DO UPDATE
                   SET effective_to=excluded.effective_to, known_at=excluded.known_at""",
                rows,
            )

    def put_hotspots(self, rows):
        with self.transaction() as db:
            db.executemany(
                """INSERT INTO hotspots VALUES(?,?,?,?,?,?,?)
                   ON CONFLICT(event_id) DO UPDATE SET kind=excluded.kind, target=excluded.target,
                   published_at=excluded.published_at, available_at=excluded.available_at,
                   strength=excluded.strength, title=excluded.title""",
                rows,
            )

    def put_holdings(self, asof, rows):
        with self.transaction() as db:
            db.execute("INSERT OR IGNORE INTO holding_snapshots VALUES(?)", (asof,))
            db.execute("DELETE FROM holdings WHERE asof=?", (asof,))
            db.executemany("INSERT INTO holdings VALUES(?,?,?)", ((asof, *row) for row in rows))


class MarketSnapshot:
    def __init__(self, db):
        self.db = db

    def status(self):
        row = self.db.execute("SELECT value FROM meta WHERE key='source'").fetchone()
        total, complete, earliest, latest = self.db.execute(
            """SELECT count(*), coalesce(sum(complete),0),
               min(CASE WHEN complete THEN day END), max(CASE WHEN complete THEN day END)
               FROM sessions"""
        ).fetchone()
        return dict(
            source=row[0] if row else None, calendar_sessions=total, complete_sessions=complete,
            first_complete=earliest, last_complete=latest,
            bars=self.db.execute("SELECT count(*) FROM bars").fetchone()[0],
            industry_memberships=self.db.execute("SELECT count(*) FROM industry_membership").fetchone()[0],
            hotspot_events=self.db.execute("SELECT count(*) FROM hotspots").fetchone()[0],
        )

    def sessions(self, start, end, require_complete=True):
        rows = list(self.db.execute(
            "SELECT day,complete FROM sessions WHERE day BETWEEN ? AND ? ORDER BY day", (start, end)
        ))
        missing = [day for day, complete in rows if not complete]
        if require_complete and missing:
            raise ValueError(f"{len(missing)} 个交易日行情未同步，最早缺失 {missing[0]}。")
        return [row[0] for row in rows]

    def completed(self, start, end):
        return {row[0] for row in self.db.execute(
            "SELECT day FROM sessions WHERE day BETWEEN ? AND ? AND complete=1", (start, end)
        )}

    def frame(self, start, end):
        return pd.read_sql_query(
            """SELECT b.* FROM bars b JOIN sessions s USING(day)
               WHERE s.complete=1 AND b.day BETWEEN ? AND ? ORDER BY b.code,b.day""",
            self.db, params=(start, end),
        )

    def securities(self):
        return dict((code, (name, listed, delisted)) for code, name, listed, delisted
                    in self.db.execute("SELECT code,name,listed,delisted FROM securities"))

    def industry_rows(self):
        return list(self.db.execute(
            """SELECT code,industry,effective_from,effective_to,known_at
               FROM industry_membership ORDER BY effective_from,code,industry"""
        ))

    def hotspots(self, day, lookback):
        end = cutoff(day)
        beginning = (datetime.fromisoformat(end) - timedelta(days=lookback)).isoformat(timespec="seconds")
        return list(self.db.execute(
            """SELECT kind,target,strength,title,available_at FROM hotspots
               WHERE available_at > ? AND available_at <= ? ORDER BY available_at,event_id""",
            (beginning, end),
        ))

    def holdings(self, day):
        return dict(self.db.execute(
            """SELECT code,shares FROM holdings WHERE asof=(
               SELECT max(asof) FROM holding_snapshots WHERE asof<=?)""", (cutoff(day),)
        ))
