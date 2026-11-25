"""Resumable OpenD quote downloads and audited import with daily reference data."""

import hashlib
import json
import re
import socket
import time

import pandas as pd

from .providers import FeedError, optional_day, optional_float, read_csv
from .store import iso_day, latest_completed_wall_day

# The installed OpenD SDK supports Shanghai and Shenzhen quote codes.
CODE_PATTERN = re.compile(r"(SH|SZ)\.\d{6}")


def canonical(code):
    market, number = code.split(".")
    return f"{number}.{market}"


def check_opend(host, port):
    try:
        with socket.create_connection((host, port), timeout=3):
            pass
    except PermissionError:
        raise FeedError(
            f"当前环境禁止连接 {host}:{port}，请在打开 OpenD 的电脑终端运行 futu-pull。"
        ) from None
    except OSError as exc:
        raise FeedError(f"无法连接 OpenD {host}:{port}。{exc}") from None


def write_atomic(df, path):
    temp = path.with_suffix(".csv.tmp")
    df.to_csv(temp, index=False)
    temp.replace(path)


def pull_futu(args, start, progress):
    code_lines = args.codes.read_text(encoding="utf-8-sig").splitlines() if args.codes else []
    codes = sorted({x.strip().upper() for x in [*args.code, *code_lines] if x.strip()})
    if not codes or any(not CODE_PATTERN.fullmatch(x) for x in codes):
        raise ValueError("请使用 --code 或 --codes 指定沪深股票，例如 SH.600519、SZ.000001。")
    if not start <= args.end <= latest_completed_wall_day():
        raise ValueError("OpenD 拉取结束日须在开始日之后，且已到上海时间 17:30 截止点。")
    if not 1 <= args.port <= 65535:
        raise ValueError("OpenD 端口无效。")
    folder = args.out
    manifest_path = folder / "request.json"
    manifest = dict(start=start, end=args.end, codes=codes)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("此下载目录属于其他代码或起止日期，请使用新目录。")
    pending = [code for code in codes if not (
        (folder / f"{code}.raw.csv").exists() and (folder / f"{code}.hfq.csv").exists()
    )]
    if not pending and (folder / "sessions.csv").exists():
        return {"start": start, "end": args.end, "symbols": len(codes), "downloaded_symbols": 0}
    check_opend(args.host, args.port)
    try:
        import futu
    except ImportError:
        raise FeedError("缺少富途 SDK，请在终端安装 pip install '.[futu]'。") from None
    folder.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    quote = futu.OpenQuoteContext(host=args.host, port=args.port)
    try:
        ret, quota = quote.get_history_kl_quota(get_detail=True)
        if ret != futu.RET_OK:
            raise FeedError(f"无法查询 OpenD 历史 K 线额度。{quota}")
        used, remaining, details = quota
        prior_codes = {row["code"] for row in details}
        new_codes = set(pending) - prior_codes
        available = set(sorted(new_codes)[:remaining]) | prior_codes
        batch = [code for code in pending if code in available]
        if not batch:
            raise FeedError(f"OpenD 的新标的额度已用完，仍有 {len(pending)} 个标的待下载。")
        if len(batch) < len(pending):
            progress(f"本次可下载 {len(batch)} 个标的，额度恢复后可使用同一命令续传。")
        progress(f"OpenD 历史 K 线额度：已用 {used}，剩余 {remaining}，本批新增 {len(new_codes)}。")
        ret, calendar = quote.request_trading_days(market=futu.TradeDateMarket.CN, start=start, end=args.end)
        if ret != futu.RET_OK or not calendar:
            raise FeedError(f"OpenD 无法提供所选区间的 A 股交易日历。{calendar}")
        sessions = sorted({iso_day(row["time"]) for row in calendar})
        write_atomic(pd.DataFrame({"day": sessions}), folder / "sessions.csv")
        for code in batch:
            for suffix, adj in (("raw", futu.AuType.NONE), ("hfq", futu.AuType.HFQ)):
                output = folder / f"{code}.{suffix}.csv"
                if output.exists():
                    continue
                pages, key, seen = [], None, set()
                while True:
                    ret, data, key = quote.request_history_kline(
                        code, start=start, end=args.end, ktype=futu.KLType.K_DAY,
                        autype=adj, max_count=500, page_req_key=key,
                    )
                    time.sleep(0.55)
                    if ret != futu.RET_OK:
                        raise FeedError(f"OpenD {code} {suffix} 拉取失败。{data}")
                    pages.append(data)
                    if key is None:
                        break
                    if key in seen:
                        raise FeedError(f"OpenD {code} 返回重复分页游标。")
                    seen.add(key)
                df = pd.concat(pages, ignore_index=True)
                required = {"time_key", "open", "high", "low", "close", "volume", "turnover"}
                if df.empty or not required.issubset(df.columns) or df["time_key"].duplicated().any():
                    raise FeedError(f"OpenD {code} 返回空日线、缺字段或重复日期。")
                write_atomic(df, output)
            progress(f"OpenD {code} 日 K 已下载。")
    finally:
        quote.close()
    return {"start": start, "end": args.end, "symbols": len(codes),
            "downloaded_symbols": len(batch), "pending_symbols": len(pending) - len(batch)}


def import_futu(store, folder, fundamentals, securities_path):
    manifest = json.loads((folder / "request.json").read_text(encoding="utf-8"))
    unique_codes = {canonical(x) for x in manifest["codes"]}
    sessions = sorted({iso_day(x["day"]) for x in read_csv(folder / "sessions.csv")})
    securities = [
        (r["code"], r["name"], iso_day(r["listed"]), optional_day(r["delisted"]))
        for r in read_csv(securities_path) if r["code"] in unique_codes
    ]
    if {r[0] for r in securities} != unique_codes or not sessions:
        raise ValueError("股票基础信息须包含全部下载标的，且交易日历不能为空。")
    meta = pd.read_csv(fundamentals, dtype={"day": str, "code": str})
    fields = ["market_cap", "adj", "limit_up", "limit_down", "limit_known"]
    required = {"day", "code", *fields}
    if not required.issubset(meta.columns):
        raise ValueError(f"逐日参考数据需要字段 {', '.join(sorted(required))}。")
    meta = meta[meta["code"].isin(unique_codes)].set_index(["day", "code"])
    if not meta.index.is_unique:
        raise ValueError("逐日参考数据存在重复的 (day, code)。")
    all_bars = []
    for vendor_code in manifest["codes"]:
        raw_path = folder / f"{vendor_code}.raw.csv"
        if not raw_path.is_file() or not (folder / f"{vendor_code}.hfq.csv").is_file():
            raise ValueError(f"{vendor_code} 的 OpenD 下载批次尚未完成。")
        raw = pd.read_csv(raw_path)
        raw["day"] = raw["time_key"].str.slice(0, 10).map(iso_day)
        raw["code"] = canonical(vendor_code)
        all_bars.append(raw[raw["volume"] > 0])
    raw = pd.concat(all_bars, ignore_index=True)
    if set(raw["day"]) != set(sessions):
        raise ValueError("全股票池日线没有覆盖下载批次内每个交易日。")
    merged = raw.join(meta[fields], on=["day", "code"], validate="one_to_one")
    missing = merged[["market_cap", "adj", "limit_known"]].isna().any(axis=1)
    if missing.any():
        row = merged.loc[missing, ["day", "code"]].iloc[0]
        raise ValueError(f"{int(missing.sum())} 根富途日线缺逐日参考字段，例如 {row.day} {row.code}。")
    pool = hashlib.sha256("\n".join(sorted(unique_codes)).encode()).hexdigest()[:16]
    source = f"futu:{pool}"
    store.register_market(source, securities, sessions)
    for day, frame in merged.groupby("day"):
        rows = [(
            day, r.code, float(r.open), float(r.high), float(r.low), float(r.close),
            optional_float(r.last_close) if getattr(r, "last_close", 0) > 0 else None,
            float(r.volume), float(r.turnover), float(r.market_cap), float(r.adj),
            optional_float(r.limit_up), optional_float(r.limit_down), int(r.limit_known),
        ) for r in frame.itertuples(index=False)]
        store.save_day(day, rows)
    return {"source": source, "sessions": len(sessions), "symbols": len(unique_codes), "bars": len(merged)}
