"""Causal board, security status and execution features for Qlib training."""

import csv
import json
import sqlite3
from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .joint_token import FEATURES, VOLUME_MIN, VOLUME_WINDOW
from .qlib_data import code_for, search_term

MAIN, STAR, CHINEXT, BEIJING, ST = range(5)
TYPE_LABELS = ["主板或板块未知", "科创板", "创业板", "北交所", "ST 或疑似 ST"]
AUX_FIELDS = ("volatility_20", "relative_return_5", "cap_age", "st_verified", "st_inferred",
              "close_limit_up", "close_limit_down", "limit_official", "limit_estimated", "industry_known")
FOOTPRINT = {"stock_types": TYPE_LABELS, "features": list(FEATURES), "auxiliary": list(AUX_FIELDS)}


@dataclass
class ResearchPanel:
    dates: np.ndarray
    symbols: list
    features: np.ndarray
    auxiliary: np.ndarray
    trade_return: np.ndarray
    kind: np.ndarray
    opening_blocked: np.ndarray
    source: dict
    coverage: dict


def board(symbol):
    if symbol.startswith(("sh688", "sh689")):
        return STAR
    if symbol.startswith(("sz300", "sz301")):
        return CHINEXT
    if symbol.startswith("bj"):
        return BEIJING
    return MAIN


def price_grid(value):
    return np.floor(value * 100 + 0.50001) / 100


def st_from_price(high, low, close, comparable, dates):
    """Detect recurring 5% bands using only observations visible at each close."""
    reference = np.r_[np.nan, close[:-1]]
    up, down = price_grid(reference * 1.05), price_grid(reference * 0.95)
    # A tick in very cheap shares can make separate bands indistinguishable.
    valid = comparable & (reference >= 1) & (dates < "2026-07-06")
    hit = valid & ((abs(close - up) < .0051) | (abs(close - down) < .0051))
    violated = ~valid | (high > up + .0051) | (low < down - .0051)
    last_violation = np.maximum.accumulate(np.where(violated, np.arange(len(close)), -1))
    prefix = np.r_[0, np.cumsum(hit)]
    left = np.maximum(np.arange(len(close)) - 14, last_violation + 1)
    return valid & ~violated & ((prefix[1:] - prefix[left]) >= 2)


def limits(raw, dates, group, listed, status):
    """Return as-of-close features and a next-open buy constraint for labels."""
    adjusted_open, adjusted_close, adjusted_high, adjusted_low, factor, _, _ = raw.T
    with np.errstate(divide="ignore", invalid="ignore"):
        open_, close, high, low = (x / factor for x in (adjusted_open, adjusted_close, adjusted_high, adjusted_low))
    prev = np.r_[np.nan, close[:-1]]
    previous_factor = np.r_[np.nan, factor[:-1]]
    comparable = np.isfinite(prev) & (prev > 0) & np.isfinite(high) & np.isfinite(low) & (close > 0)
    comparable &= np.isclose(factor, previous_factor, rtol=2e-5, atol=0)
    if listed >= dates[0]:
        comparable[:np.searchsorted(dates, listed) + 5] = False
    verified = status[:, 0]
    inferred = st_from_price(high, low, close, comparable, dates) if group == MAIN else np.zeros(len(dates), bool)
    inferred &= ~np.isfinite(verified)
    is_st = (verified == 1) | inferred
    rate = np.full(len(dates), [0.10, 0.20, 0.20, 0.30][group])
    if group == MAIN:
        rate[is_st & (dates < "2026-07-06")] = 0.05
    predicted_up, predicted_down = price_grid(prev * (1 + rate)), price_grid(prev * (1 - rate))
    official = np.isfinite(status[:, 1]) & np.isfinite(status[:, 2])
    # Rule-break observations indicate an exception or an unreliable inferred reference.
    estimated = comparable & ~official & (high <= predicted_up + .0101) & (low >= predicted_down - .0101)
    hi = np.where(official, status[:, 1], predicted_up)
    lo = np.where(official, status[:, 2], predicted_down)
    usable = official | estimated
    close_up = usable & (close >= hi - .0051)
    close_down = usable & (close <= lo + .0051)
    open_blocked = usable & ~np.isnan(open_) & (open_ >= hi - .0051)
    return is_st, inferred, close_up, close_down, official, estimated, open_blocked


def symbol_features(raw):
    with np.errstate(divide="ignore", invalid="ignore"):
        open_, close, _, _, factor, normalized_volume, observed_cap = raw.T
        volume = normalized_volume * factor
        traded = np.isfinite(close) & np.isfinite(volume) & (close > 0) & (volume > 0) & (factor > 0)
        trailing_vol = (pd.Series(volume).where(traded).shift(1)
                        .rolling(VOLUME_WINDOW, min_periods=VOLUME_MIN).median().to_numpy())
        cap_ok = np.isfinite(observed_cap) & (observed_cap > 0)
        cap = pd.Series(observed_cap).where(cap_ok).ffill(limit=3).to_numpy()
        latest = np.maximum.accumulate(np.where(cap_ok, np.arange(len(cap)), -99))
        cap_age = np.minimum(np.arange(len(cap)) - latest, 4).astype(np.float32) / 3
        ret = np.full(len(close), np.nan)
        adjacent = np.flatnonzero(traded[1:] & traded[:-1]) + 1
        ret[adjacent] = close[adjacent] / close[adjacent - 1] - 1
        volatility = pd.Series(ret).rolling(20, min_periods=15).std().to_numpy()
        ret5 = np.full(len(close), np.nan)
        consecutive = np.r_[0, np.cumsum(traded)]
        valid5 = np.flatnonzero(consecutive[6:] - consecutive[:-6] == 6) + 5
        ret5[valid5] = close[valid5] / close[valid5 - 5] - 1
        out = np.full((len(close), 3), np.nan, dtype=np.float32)
        valid = np.isfinite(ret) & np.isfinite(trailing_vol) & (trailing_vol > 0) & np.isfinite(cap)
        out[valid] = np.stack((np.log10(cap[valid] * 10000), np.log(volume[valid] / trailing_vol[valid]), ret[valid]), axis=1)
        open_return = np.full(len(close), np.nan, dtype=np.float32)
        good = traded[1:] & traded[:-1] & (open_[1:] > 0) & (open_[:-1] > 0)
        open_return[:-1][good] = open_[1:][good] / open_[:-1][good] - 1
    return out, volatility, ret5, cap_age, open_return


def load_status(path):
    dates = defaultdict(dict)
    if path is None:
        return dates
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        read = csv.DictReader(handle)
        required = {"day", "code", "is_st", "limit_up", "limit_down"}
        if not required.issubset(read.fieldnames or ()):
            raise ValueError("状态 CSV 需要 day,code,is_st,limit_up,limit_down。")
        for row in read:
            day = pd.Timestamp(row["day"]).date().isoformat()
            sym = search_term(row["code"]).lower()
            st, up, down = (float(row[field]) if row[field].strip() else np.nan
                            for field in ("is_st", "limit_up", "limit_down"))
            if (np.isfinite(st) and st not in (0, 1)) or (np.isfinite(up) != np.isfinite(down)):
                raise ValueError(f"非法状态值：{day} {sym}")
            if np.isfinite(up) and (down <= 0 or up <= down):
                raise ValueError(f"非法涨跌停价：{day} {sym}")
            if day in dates[sym]:
                raise ValueError(f"状态 CSV 含重复记录：{day} {sym}")
            dates[sym][day] = (st, up, down)
    return dates


def load_reference(market_path, statuses):
    industries = defaultdict(list)
    if market_path is None or not Path(market_path).is_file():
        return industries
    with closing(sqlite3.connect(Path(market_path).resolve().as_uri() + "?mode=ro", uri=True)) as db:
        for day, code, up, down in db.execute(
            "SELECT day,code,limit_up,limit_down FROM bars WHERE limit_known=1 AND limit_up>0 AND limit_down>0"
        ):
            statuses[search_term(code).lower()][day] = (np.nan, up, down)
        for code, industry, start, end, known in db.execute(
            "SELECT code,industry,effective_from,effective_to,known_at FROM industry_membership ORDER BY known_at"
        ):
            ts = pd.Timestamp(known)
            ts = ts.tz_localize("Asia/Shanghai") if ts.tzinfo is None else ts.tz_convert("Asia/Shanghai")
            same_day = (ts.hour, ts.minute, ts.second) <= (17, 30, 0)
            effective = ts.date().isoformat() if same_day else (ts.date() + pd.Timedelta(days=1)).isoformat()
            industries[search_term(code).lower()].append((industry, max(start, effective), end))
    return industries


def load_research_panel(path, market_path, status_csv, start, end, context, progress, *, on_symbol=None,
                        include_valuation=False):
    extra = defaultdict(dict)
    industries = load_reference(market_path, extra)
    for symbol, records in load_status(status_csv).items():
        for day, (st, up, down) in records.items():
            verified = extra[symbol].get(day, (np.nan, np.nan, np.nan))
            extra[symbol][day] = (st, up, down) if np.isfinite(up) else (st, *verified[1:])
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("BEGIN")
        source = json.loads(db.execute("SELECT value FROM qlib_meta WHERE key='summary'").fetchone()[0])
        days = [row[0] for row in db.execute("SELECT day FROM qlib_calendar WHERE day<=? ORDER BY day", (end,))]
        start_idx = next((i for i, day in enumerate(days) if day >= start), None)
        if start_idx is None:
            raise ValueError("指定训练区间内没有行情。")
        dates = np.array(days[max(0, start_idx - context - 21):])
        index = {day: i for i, day in enumerate(dates)}
        pairs = list(db.execute(
            """SELECT symbol,first_day FROM qlib_symbols WHERE price_rows>0 AND first_day<=? AND last_day>=?
               ORDER BY symbol""", (dates[-1], dates[0])
        ))
        pairs = [(symbol, listed) for symbol, listed in pairs
                 if symbol.startswith(("sh60", "sh68", "sz00", "sz30", "bj"))]
        if not pairs:
            raise ValueError("未找到 A 股训练标的。")
        shape = (len(pairs), len(dates))
        features = np.full((*shape, 3), np.nan, dtype=np.float32)
        aux = np.zeros((*shape, len(AUX_FIELDS)), dtype=np.float32)
        kind = np.empty(shape, dtype=np.int8)
        groups = np.empty(len(pairs), dtype=np.int8)
        ret5 = np.full(shape, np.nan, dtype=np.float32)
        open_returns = np.full(shape, np.nan, dtype=np.float32)
        blocked = np.zeros(shape, dtype=bool)
        industry_codes = np.zeros(shape, dtype=np.int16)
        industry_names = {}
        fields = ["open", "close", "high", "low", "factor", "volume", "total_mv"]
        if on_symbol is not None:
            columns = {row[1] for row in db.execute("PRAGMA table_info(qlib_daily)")}
            optional = ("amount", "pb", "pe_ttm", "dv_ttm") if include_valuation else ("amount",)
            fields.extend(name if name in columns else "NULL" for name in optional)
        for sid, (symbol, listing) in enumerate(pairs):
            history = np.full((len(dates), len(fields)), np.nan)
            for day, *values in db.execute(
                f"""SELECT day,{','.join(fields)}
                   FROM qlib_daily WHERE symbol=? AND day BETWEEN ? AND ?""",
                (symbol, dates[0], dates[-1])
            ):
                history[index[day]] = values
            raw = history[:, :7]
            status = np.full((len(dates), 3), np.nan)
            for day, values in extra.get(symbol, {}).items():
                if day in index:
                    status[index[day]] = values
            group = board(symbol)
            is_st, inferred, up, down, official, estimated, blocked[sid] = limits(raw, dates, group, listing, status)
            if on_symbol is not None:
                on_symbol(sid, shape, dates, symbol, listing, history, status, is_st)
            features[sid], vol, ret5[sid], cap_age, open_returns[sid] = symbol_features(raw)
            groups[sid] = group
            kind[sid] = np.where(is_st, ST, group)
            aux[sid, :, 0] = np.nan_to_num(vol, nan=0).clip(0, 0.15) * 20
            aux[sid, :, 2] = cap_age
            aux[sid, :, 3:9] = np.stack((np.isfinite(status[:, 0]), inferred, up, down, official, estimated), axis=1)
            for name, begin, finish in industries.get(symbol, []):
                a, b = np.searchsorted(dates, begin), np.searchsorted(dates, finish or "9999-12-31")
                industry_codes[sid, a:b] = industry_names.setdefault(name, len(industry_names) + 1)
            if (sid + 1) % 1500 == 0:
                progress(f"量价、板块与交易约束已读取 {sid+1:,}/{len(pairs):,} 个标的。")
    # Same-day, same-board fallback. True industry membership overrides it when available.
    for day in range(len(dates)):
        ind = industry_codes[:, day]
        key = np.where(ind > 0, ind + 3, groups)
        returns = ret5[:, day]
        valid = np.isfinite(returns)
        counts = np.bincount(key[valid], minlength=len(industry_names) + 4)
        sums = np.bincount(key[valid], weights=returns[valid].clip(-.35, .35), minlength=len(counts))
        industry_ok = (ind > 0) & (counts[key] >= 5)
        board_count = np.bincount(groups[valid], minlength=4)
        board_sum = np.bincount(groups[valid], weights=returns[valid].clip(-.35, .35), minlength=4)
        benchmark = np.where(industry_ok, sums[key] / counts[key].clip(1), board_sum[groups] / board_count[groups].clip(1))
        relative = returns - benchmark
        aux[:, day, 1] = np.nan_to_num(relative, nan=0).clip(-.35, .35) * 5
        aux[:, day, 9] = industry_ok
    candidates = aux[:, :, 7].astype(bool) | aux[:, :, 8].astype(bool)
    open_returns[~candidates | blocked] = np.nan
    based = np.isfinite(features[:, :, 2]) & (dates >= start)
    coverage = {
        "observations": int(based.sum()),
        "actual_st_status_rows": int((based & (aux[:, :, 3] > 0)).sum()),
        "inferred_st_rows": int((based & (aux[:, :, 4] > 0)).sum()),
        "actual_limit_rows": int((based & (aux[:, :, 7] > 0)).sum()),
        "estimated_limit_rows": int((based & (aux[:, :, 8] > 0)).sum()),
        "close_limit_up_rows": int((based & (aux[:, :, 5] > 0)).sum()),
        "opening_blocked_rows": int((based & blocked).sum()),
        "industry_relative_rows": int((based & (aux[:, :, 9] > 0)).sum()),
        "carried_cap_rows": int((based & (aux[:, :, 2] > 0)).sum()),
        "stock_type_counts": np.bincount(kind[based], minlength=5).tolist(),
    }
    return ResearchPanel(dates, [code_for(symbol) for symbol, _ in pairs], features, aux, open_returns,
                         kind, blocked, {k: source[k] for k in ("source", "release", "sha256") if k in source},
                         coverage)
