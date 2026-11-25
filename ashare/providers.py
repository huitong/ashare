"""Market-source adapter and CSV imports. Credentials are read from the environment."""

import csv
import hashlib
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import HTTPSHandler, HTTPRedirectHandler, Request, build_opener

import pandas as pd

from .store import BAR_FIELDS, SHANGHAI, cutoff, iso_day, timestamp


class FeedError(RuntimeError):
    pass


def optional_float(value):
    return None if pd.isna(value) or value == "" else float(value)


def optional_day(value):
    if pd.isna(value) or value == "":
        return None
    value = str(value)
    if len(value) == 8 and value.isdigit():
        value = f"{value[:4]}-{value[4:6]}-{value[6:]}"
    return iso_day(value)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # The credential is meant for the official API host only.
        return None


class TushareHTTP:
    def __init__(self, token):
        self.token = token
        self.opener = build_opener(HTTPSHandler(), NoRedirect())

    def query(self, endpoint, fields, **params):
        payload = dict(api_name=endpoint, token=self.token, params=params, fields=fields)
        request = Request(
            "https://api.tushare.pro",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with self.opener.open(request, timeout=25) as response:
                result = json.load(response)
        except HTTPError as exc:
            raise FeedError(f"API HTTP {exc.code}。") from None
        if result.get("code") != 0:
            raise FeedError(result.get("msg") or f"API status {result.get('code')}")
        data = result["data"]
        return pd.DataFrame(data["items"], columns=data["fields"])


def tushare_token():
    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if token:
        return token
    location = Path(os.environ.get("TUSHARE_TOKEN_FILE", "data/tushare.token"))
    if location.is_file():
        return location.read_text(encoding="utf-8").strip()
    return ""


class TushareFeed:
    source = "tushare"

    def __init__(self, calls_per_minute=45, client=None):
        if client is None:
            token = tushare_token()
            if not token:
                raise FeedError("缺少 TUSHARE_TOKEN 或 data/tushare.token 配置。")
            client = TushareHTTP(token)
        else:
            token = ""
        self.client = client
        self.token = token
        self.interval = 60 / calls_per_minute
        self.next_call = 0.0

    def _request(self, endpoint, fields, **params):
        for attempt in range(3):
            if self.next_call > time.monotonic():
                time.sleep(max(0, self.next_call - time.monotonic()))
            self.next_call = time.monotonic() + self.interval
            try:
                result = self.client.query(endpoint, fields=fields, **params)
            except Exception as exc:
                message = str(exc)
                if self.token:
                    message = message.replace(self.token, "[REDACTED]")
                obvious_config_error = any(x in message.lower() for x in ("token", "权限", "积分"))
                if obvious_config_error or attempt == 2:
                    raise FeedError(f"Tushare {endpoint} 请求失败。{message}") from None
                time.sleep(2 ** attempt)
            else:
                return result
        raise AssertionError("unreachable")

    def query(self, endpoint, fields, row_cap, *, page_by=None, **params):
        page = self._request(endpoint, fields, **params)
        if len(page) < row_cap:
            return page
        if not page_by:
            raise FeedError(f"Tushare {endpoint} 返回达到 {row_cap} 条上限，可能截断。{params}")
        pages = [page]
        keys = set(page[page_by])
        while len(page) >= row_cap:
            page = self._request(endpoint, fields, offset=sum(map(len, pages)), limit=row_cap, **params)
            new_keys = set(page[page_by]) if len(page) else set()
            if keys & new_keys or len(new_keys) != len(page):
                raise FeedError(f"Tushare {endpoint} 无法确认分页完整性。请分区获取后使用 import-market。{params}")
            keys.update(new_keys)
            pages.append(page)
        return pd.concat(pages, ignore_index=True)

    def calendar(self, start, end):
        frame = self.query(
            "trade_cal", "cal_date,is_open", 6000,
            exchange="SSE", start_date=start.replace("-", ""), end_date=end.replace("-", ""),
        )
        received = {optional_day(x) for x in frame["cal_date"]}
        expected = {x.date().isoformat() for x in pd.date_range(start, end)}
        if received != expected:
            raise FeedError("Tushare trade_cal 返回日期不完整。")
        return sorted(optional_day(r.cal_date) for r in frame.itertuples() if int(r.is_open) == 1)

    def securities(self):
        result = {}
        for status in ("L", "D", "P"):
            frame = self.query(
                "stock_basic", "ts_code,name,list_date,delist_date,exchange", 6000, list_status=status,
            )
            for row in frame.itertuples():
                if row.exchange in {"SSE", "SZSE", "BSE"} and optional_day(row.list_date):
                    result[row.ts_code] = (
                        row.ts_code, row.name, optional_day(row.list_date), optional_day(row.delist_date),
                    )
        if not result:
            raise FeedError("Tushare stock_basic 没有可用股票。")
        return list(result.values())

    def bars(self, day, valid_codes):
        d = day.replace("-", "")
        daily = self.query(
            "daily", "ts_code,open,high,low,close,pre_close,vol,amount", 6000, page_by="ts_code", trade_date=d,
        )
        if not valid_codes:
            raise FeedError(f"{day} 没有预期上市股票。")
        share = daily[daily["ts_code"].isin(valid_codes)]
        if len(share) < math.ceil(0.85 * len(valid_codes)):
            raise FeedError(f"{day} 行情只覆盖 {len(share)}/{len(valid_codes)} 只预期上市股票。")
        basic = self.query("daily_basic", "ts_code,total_mv", 6000, page_by="ts_code", trade_date=d)
        factors = self.query("adj_factor", "ts_code,adj_factor", 6000, page_by="ts_code", trade_date=d)
        limits = self.query("stk_limit", "ts_code,up_limit,down_limit", 5800, page_by="ts_code", trade_date=d)
        if limits.empty:
            raise FeedError(f"{day} stk_limit 返回空表。")
        merged = share.merge(basic, on="ts_code", how="left", validate="one_to_one")
        merged = merged.merge(factors, on="ts_code", how="left", validate="one_to_one")
        merged = merged.merge(limits, on="ts_code", how="left", validate="one_to_one")
        merged = merged[merged["vol"] > 0]
        incomplete = merged[["total_mv", "adj_factor"]].isna().any(axis=1)
        if incomplete.any():
            codes = merged.loc[incomplete, "ts_code"]
            raise FeedError(f"{day} {len(codes)} 只有成交股票缺市值或复权因子，例如 {codes.iloc[0]}。")
        rows = []
        for r in merged.itertuples(index=False):
            up, down = optional_float(r.up_limit), optional_float(r.down_limit)
            rows.append((
                day, r.ts_code, float(r.open), float(r.high), float(r.low), float(r.close),
                optional_float(r.pre_close), float(r.vol) * 100, float(r.amount) * 1000,
                float(r.total_mv) * 10000, float(r.adj_factor), up, down, int(up is not None and down is not None),
            ))
        return rows

    def industries(self, codes, assume_effective_known):
        today = datetime.now(SHANGHAI).isoformat(timespec="seconds")
        groups = self.query("index_classify", "index_code", 1000, level="L1", src="SW2021")
        results = {}
        for industry in groups["index_code"]:
            for current in ("N", "Y"):
                frame = self.query(
                    "index_member_all", "l1_name,ts_code,in_date,out_date", 2000,
                    l1_code=industry, is_new=current,
                )
                for row in frame.itertuples():
                    start, end = optional_day(row.in_date), optional_day(row.out_date)
                    if row.ts_code in codes and start and (not end or end > start):
                        known = cutoff(start) if assume_effective_known else today
                        results[row.ts_code, row.l1_name, start] = (
                            row.ts_code, row.l1_name, start, end, known,
                        )
        if not results:
            raise FeedError("Tushare 没有返回有效的申万行业记录。")
        return list(results.values())


def synchronize(store, feed, start, end, refresh=False, progress=print):
    with store.snapshot() as market:
        status = market.status()
    if status["source"] and status["source"] != feed.source:
        raise ValueError("此数据库已绑定其他行情数据源。")
    first, last = status["first_complete"], status["last_complete"]
    # Extend through the existing boundary so later runs cannot leave an unregistered calendar gap.
    if last and start > last:
        start = last
    if first and end < first:
        end = first
    calendar = feed.calendar(start, end)
    if not calendar:
        raise FeedError("所选区间没有交易日。")
    securities = feed.securities()
    store.register_market(feed.source, securities, calendar)
    with store.snapshot() as market:
        complete = market.completed(start, end)
    count = 0
    for day in calendar:
        if not refresh and day in complete:
            continue
        codes = {code for code, _, listed, delisted in securities if listed <= day <= (delisted or end)}
        rows = feed.bars(day, codes)
        store.save_day(day, rows)
        count += 1
        progress(f"{day} {len(rows)} 只")
    return count


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def import_industries(store, path):
    entries = []
    for row in read_csv(path):
        start = iso_day(row["effective_from"])
        end = optional_day(row["effective_to"])
        if end and start >= end:
            raise ValueError("行业区间应采用 [effective_from, effective_to)。")
        entries.append((row["code"], row["industry"], start, end, timestamp(row["known_at"])))
    store.put_industries(entries)
    return len(entries)


def import_hotspots(store, path):
    entries = []
    for row in read_csv(path):
        published = timestamp(row["published_at"])
        available = timestamp(row["available_at"])
        strength = float(row["strength"])
        kind, target, title = row["kind"].strip(), row["target"].strip(), row["title"].strip()
        if not (published <= available and math.isfinite(strength) and strength > 0 and target and title):
            raise ValueError("热点需要有效标题、目标、强度和发布时间/可用时间。")
        event_id = (row.get("event_id") or "").strip()
        if not event_id:
            event_id = hashlib.sha256(str((kind, target, title, published)).encode()).hexdigest()
        entries.append((event_id, kind, target, published, available, strength, title))
    store.put_hotspots(entries)
    return len(entries)


def import_holdings(store, path, asof):
    rows = [(r["code"], int(r["shares"])) for r in read_csv(path)]
    store.put_holdings(timestamp(asof), rows)
    return len(rows)


def import_market(store, folder, source):
    securities = [
        (r["code"], r["name"], iso_day(r["listed"]), optional_day(r["delisted"]))
        for r in read_csv(folder / "securities.csv")
    ]
    sessions = sorted({iso_day(r["day"]) for r in read_csv(folder / "sessions.csv")})
    if not securities or not sessions:
        raise ValueError("securities.csv 和 sessions.csv 不能为空。")
    # Stream a date-sorted full-market file to keep memory bounded.
    from itertools import groupby

    store.register_market(source, securities, sessions)
    seen = set()
    with (folder / "bars.csv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(BAR_FIELDS).issubset(reader.fieldnames or []):
            raise ValueError(f"bars.csv 缺列：要求 {', '.join(BAR_FIELDS)}。")
        for day, group in groupby(reader, key=lambda row: iso_day(row["day"])):
            if day in seen:
                raise ValueError("bars.csv 中同一日的记录必须连续。")
            rows = [
                (day, row["code"], *(optional_float(row[key]) for key in BAR_FIELDS[2:-1]), int(row["limit_known"]))
                for row in group
            ]
            store.save_day(day, rows)
            seen.add(day)
    if seen != set(sessions):
        raise ValueError("bars.csv 未覆盖 sessions.csv 的全部交易日。")
    return len(seen)
