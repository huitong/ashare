"""Loopback-only HTTP server for the date and symbol views of the local databases."""

import csv
import io
import json
import math
import re
import sqlite3
from contextlib import closing
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .qlib_data import search_term

ASSETS = Path(__file__).with_name("web_assets")
PAGE_SIZE = 50
SORTS = {"cap": "market_cap DESC NULLS LAST, code",
         "volume": "volume DESC NULLS LAST, code", "code": "code"}
MIME = {"/": ("index.html", "text/html; charset=utf-8"),
        "/training": ("training.html", "text/html; charset=utf-8"),
        "/training.js": ("training.js", "application/javascript; charset=utf-8"),
        "/trade": ("trade.html", "text/html; charset=utf-8"),
        "/trade.js": ("trade.js", "application/javascript; charset=utf-8"),
        "/backtest": ("backtest.html", "text/html; charset=utf-8"),
        "/backtest.js": ("backtest.js", "application/javascript; charset=utf-8"),
        "/style.css": ("style.css", "text/css; charset=utf-8"),
        "/app.js": ("app.js", "application/javascript; charset=utf-8")}


def open_readonly(path):
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=15)
    db.row_factory = sqlite3.Row
    return db


def all_rows(db, statement, params=()):
    return [dict(row) for row in db.execute(statement, params)]


class Explorer:
    def __init__(self, market, qlib, runs=Path("runs"), *, trade_reference=None):
        self.runs = Path(runs)
        self.paths = {"qlib": Path(qlib), "market": Path(market)}
        self.trade_reference = Path(trade_reference) if trade_reference else self.paths["qlib"].with_name("trade-reference.json")

    def catalog(self):
        result = []
        for kind, path in self.paths.items():
            if not path.is_file():
                continue
            with closing(open_readonly(path)) as db:
                if kind == "qlib":
                    stats = self.qlib_status(db)
                    years = all_rows(db, """
                        SELECT substr(day,1,4) AS year, count(*) AS sessions, sum(prices) AS price_rows,
                               sum(market_caps) AS market_cap_rows
                        FROM qlib_day_stats GROUP BY 1 ORDER BY 1 DESC
                    """)
                else:
                    reason = db.execute("SELECT value FROM meta WHERE key='source'").fetchone()
                    first, last, count = db.execute("SELECT min(day),max(day),count(*) FROM sessions WHERE complete").fetchone()
                    stats = {"source": reason[0] if reason else None, "first_day": first, "last_day": last,
                             "observed_sessions": count, "symbols": db.execute("SELECT count(*) FROM securities").fetchone()[0],
                             "rows": db.execute("SELECT count(*) FROM bars").fetchone()[0]}
                    stats["price_rows"] = stats["market_cap_rows"] = stats["rows"]
                    years = []
                result.append(dict(kind=kind, stats=stats, years=years))
        return result

    @staticmethod
    def qlib_status(db):
        row = db.execute("SELECT value FROM qlib_meta WHERE key='summary'").fetchone()
        if row is None:
            raise ValueError("Qlib 数据库没有完整的导入记录。")
        return json.loads(row[0])

    def run_file(self, run, *parts):
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", run) or run in {".", ".."}:
            raise ValueError("运行名称无效。")
        path = self.runs
        for part in (run, *parts):
            path = path / part
            if path.is_symlink():
                raise LookupError("找不到运行记录。")
        if not path.is_file():
            raise LookupError("找不到运行记录。")
        return path

    def training_run(self, run):
        return json.loads(self.run_file(run, "progress.json").read_text(encoding="utf-8"))

    def training_runs(self):
        if not self.runs.is_dir():
            return []
        result = []
        for folder in self.runs.iterdir():
            if not (folder / "progress.json").is_file():
                continue
            try:
                data = self.training_run(folder.name)
            except (LookupError, OSError, ValueError):
                continue
            result.append({key: data.get(key) for key in ("run", "status", "started_at", "vocabulary")})
        return sorted(result, key=lambda run: run["started_at"] or "", reverse=True)

    def rotation(self, args):
        run = args.get("run")
        if not run:
            result = []
            for folder in sorted(self.runs.glob("*/summary.json")):
                try:
                    data = self.rotation({"run": folder.parent.name})
                except (LookupError, OSError, ValueError):
                    continue
                result.append({**{k: data[k] for k in ("run", "start", "end")},
                               "comparison_version": data.get("comparison_version", 1)})
            return sorted(result, key=lambda row: (row["comparison_version"], row["end"], row["run"]), reverse=True)
        summary = json.loads(self.run_file(run, "summary.json").read_text(encoding="utf-8"))
        if summary.get("kind") != "rotation_top10_top20":
            raise LookupError("找不到此选股回测。")
        if "table" not in args:
            return summary
        tables = {"equity", "trades", "blocked", "decisions", "events"}
        table, variant = args["table"], args.get("variant", "main")
        if (table not in tables or not re.fullmatch(r"[a-z0-9_]{1,40}", variant)
                or variant not in summary["variants"]):
            raise ValueError("回测数据表或方案无效。")
        file = self.run_file(run, variant, f"{table}.csv")
        with file.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        if table == "trades" and rows:
            names = self.security_names()
            for row in rows:
                row["stock_name"] = row.get("stock_name") or names.get(row["code"], "")
        return {"rows": rows, "filename": f"{variant}-{table}.csv"}

    def security_names(self):
        names = {}
        reference = self.paths["qlib"].with_name("security-names.csv")
        if reference.is_file():
            with reference.open(newline="", encoding="utf-8-sig") as source:
                names.update((row["code"], row["name"]) for row in csv.DictReader(source))
        market = self.paths["market"]
        if market.is_file():
            with closing(open_readonly(market)) as db:
                names.update(db.execute("SELECT code,trim(name) FROM securities WHERE trim(name) <> ''"))
        return names

    def dispatch(self, path, args):
        if path == "/api/catalog":
            return self.catalog()
        if path == "/api/training":
            return self.training_run(args["run"]) if "run" in args else self.training_runs()
        if path == "/api/backtest":
            return self.rotation(args)
        if path == "/api/trade-reference":
            if not self.trade_reference.is_file():
                raise LookupError("尚无交易参考排名，请先生成收盘评分。")
            return json.loads(self.trade_reference.read_text(encoding="utf-8"))
        source = args.get("source", "qlib")
        handlers = {"/api/symbols": self.symbols, "/api/daily": self.daily, "/api/history": self.history,
                    "/api/export": self.history}
        if source not in self.paths or not self.paths[source].is_file():
            raise LookupError("该数据源尚未入库。")
        handler = handlers.get(path)
        if handler is None:
            raise LookupError("找不到该接口。")
        with closing(open_readonly(self.paths[source])) as db:
            return handler(db, source, args)

    @staticmethod
    def symbols(db, source, args):
        q = args.get("q", "").strip()[:40]
        if source == "qlib":
            vendor = search_term(q).lower()
            return all_rows(db, """
                SELECT symbol,code,first_day,last_day,price_rows FROM qlib_symbols
                WHERE (symbol LIKE ? OR code LIKE ?) AND price_rows>0
                ORDER BY (symbol=?) DESC,(code=?) DESC,price_rows DESC,code LIMIT 20
            """, (f"%{vendor}%", f"%{q.upper()}%", vendor, q.upper()))
        return all_rows(db, """
            SELECT code AS symbol,code,name,listed AS first_day,delisted AS last_day FROM securities
            WHERE (code LIKE ? OR name LIKE ?) AND EXISTS(SELECT 1 FROM bars WHERE bars.code=securities.code)
            ORDER BY (code=?) DESC,code LIMIT 20
        """, (f"%{q}%", f"%{q}%", q.upper()))

    @staticmethod
    def daily(db, source, args):
        day = date.fromisoformat(args["day"]).isoformat()
        page = int(args.get("page", "1"))
        sort = SORTS.get(args.get("sort", "cap"))
        if not 1 <= page <= 100_000 or sort is None:
            raise ValueError("页码或排序方式无效。")
        limit = (PAGE_SIZE, (page - 1) * PAGE_SIZE)
        if source == "qlib":
            result = all_rows(db, f"""
                SELECT s.code,s.symbol,b.open,b.high,b.low,b.close,b.volume,b.total_mv*10000 AS market_cap
                FROM qlib_daily b JOIN qlib_symbols s USING(symbol)
                WHERE b.day=? AND b.close>0 ORDER BY {sort} LIMIT ? OFFSET ?
            """, (day, *limit))
            row = db.execute("SELECT prices,market_caps FROM qlib_day_stats WHERE day=?", (day,)).fetchone()
            count, caps = (row[0], row[1]) if row else (0, 0)
        else:
            result = all_rows(db, f"""
                SELECT b.code AS symbol,b.code,s.name,b.open,b.high,b.low,b.close,b.volume,b.market_cap
                FROM bars b JOIN securities s USING(code)
                WHERE b.day=? ORDER BY {sort} LIMIT ? OFFSET ?
            """, (day, *limit))
            count = caps = db.execute("SELECT count(*) FROM bars WHERE day=?", (day,)).fetchone()[0]
        return {"day": day, "count": count, "cap_count": caps, "page_size": PAGE_SIZE, "rows": result}

    @staticmethod
    def history(db, source, args):
        symbol = args.get("symbol", "").strip()
        if not symbol or len(symbol) > 40:
            raise ValueError("请选择证券。")
        start = date.fromisoformat(args.get("start") or "1900-01-01").isoformat()
        end = date.fromisoformat(args.get("end") or "9999-12-31").isoformat()
        if start > end:
            raise ValueError("起始日不能晚于结束日。")
        if len(symbol) == 6 and symbol.isascii() and symbol.isdigit():
            table = "qlib_symbols" if source == "qlib" else "securities"
            candidates = db.execute(f"SELECT code FROM {table} WHERE code LIKE ? LIMIT 2", (symbol + ".%",)).fetchall()
            if len(candidates) > 1:
                raise ValueError("匹配到多个市场，请在代码后加交易所后缀。")
            if candidates:
                symbol = candidates[0][0]
        if source == "qlib":
            info = db.execute(
                "SELECT * FROM qlib_symbols WHERE symbol=? OR code=?",
                (search_term(symbol).lower(), symbol.upper()),
            ).fetchone()
            if info is None:
                raise LookupError("未找到该证券。")
            rows = all_rows(db, """
                SELECT *,total_mv*10000 AS market_cap FROM qlib_daily
                WHERE symbol=? AND day BETWEEN ? AND ? ORDER BY day
            """, (info["symbol"], start, end))
            universes = all_rows(db, """
                SELECT universe,start,end FROM qlib_memberships
                WHERE symbol=? AND universe<>'all' ORDER BY universe,start
            """, (info["symbol"],))
        else:
            info = db.execute("SELECT code AS symbol,code,name FROM securities WHERE code=?", (symbol.upper(),)).fetchone()
            if info is None:
                raise LookupError("未找到该证券。")
            rows = all_rows(db, "SELECT day,open,high,low,close,prev_close,volume,amount,market_cap,"
                            "adj,limit_up,limit_down,limit_known FROM bars WHERE code=? AND day BETWEEN ? AND ? ORDER BY day",
                            (info["symbol"], start, end))
            universes = []
        return {"symbol": dict(info), "rows": rows, "universes": universes}


def csv_bytes(rows):
    if not rows:
        return b"\xef\xbb\xbf"
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8-sig")


def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    return value


def make_handler(explorer):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            address = urlsplit(self.path)
            if address.path in MIME:
                asset, content_type = MIME[address.path]
                self.reply(200, content_type, (ASSETS / asset).read_bytes())
                return
            if not address.path.startswith("/api/"):
                self.reply(404, "text/plain; charset=utf-8", "页面不存在。".encode())
                return
            params = {key: values[0] for key, values in parse_qs(address.query).items()}
            try:
                data = explorer.dispatch(address.path, params)
            except (KeyError, ValueError) as exc:
                self.json_reply(400, {"error": str(exc)})
            except LookupError as exc:
                self.json_reply(404, {"error": str(exc)})
            except sqlite3.Error:
                self.json_reply(500, {"error": "无法读取本地数据库，请检查是否已完整导入。"})
            else:
                if address.path == "/api/backtest" and params.get("download") == "1" and "table" in params:
                    self.reply(200, "text/csv; charset=utf-8", csv_bytes(data["rows"]),
                               f'attachment; filename="{data["filename"]}"')
                elif address.path == "/api/export":
                    self.reply(200, "text/csv; charset=utf-8", csv_bytes(data["rows"]),
                               'attachment; filename="ashare-history.csv"')
                else:
                    self.json_reply(200, data)

        def json_reply(self, status, value):
            self.reply(status, "application/json; charset=utf-8",
                       json.dumps(finite_json(value), ensure_ascii=False, allow_nan=False).encode())

        def reply(self, status, content_type, raw, disposition=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; frame-ancestors 'none'")
            if disposition:
                self.send_header("Content-Disposition", disposition)
            self.end_headers()
            self.wfile.write(raw)

    return Handler


def serve(market_path, qlib_path, port, progress=print, runs=Path("runs"), *, trade_reference=None):
    explorer = Explorer(market_path, qlib_path, runs, trade_reference=trade_reference)
    with ThreadingHTTPServer(("127.0.0.1", port), make_handler(explorer)) as server:
        progress(f"A 股行情浏览器 http://127.0.0.1:{server.server_port}/")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            return
