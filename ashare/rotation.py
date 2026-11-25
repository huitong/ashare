"""Portfolio replay for a daily recall and top-10/top-20 ranking strategy."""

import csv
import json
import math
from collections import Counter
from contextlib import closing, nullcontext
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .backtest import Book, stats
from .config import Strategy
from .features import macd_histogram
from .intraday import EXECUTION_RULES, limit_fill
from .joint_token import VOLUME_MIN, VOLUME_WINDOW, VOCABULARY, JointTokenizer
from .rank import FactorModel, Ranker
from .run_files import publish_directory
from .store import MarketSnapshot
from .temporal import sha256
from .typed_data import MAIN, board, load_research_panel, price_grid
from .typed_token import TypedTransformer, batch_inputs
from .web import open_readonly

RECALL_SIZE, BUY_RANK, SELL_RANK = 1000, 10, 20
FACTORS = {
    "trend": ("趋势", "5 日与 20 日复权收益加权，越高越优先"),
    "macd": ("MACD", "12/26/9 归一化柱值，越高越优先"),
    "token_transformer": ("Transformer", "用已有模型的可入场收益排序头打分"),
    "volume_ratio": ("成交量异动", "当日成交量相对此前 20 个交易日中位数的倍数，越高越优先"),
    "reversal_5": ("5 日反转", "最近 5 日复权收益取负，跌幅越大越优先"),
    "low_volatility": ("低波动", "20 日复权日收益波动率取负，波动越低越优先"),
}
PORTFOLIOS = {
    "main": ("趋势 + MACD + Transformer", {"trend": .45, "macd": .35, "token_transformer": .20}),
    "rules": ("趋势 + MACD", {"trend": .45, "macd": .35}),
    "trend_risk": ("趋势 + 低波动", {"trend": .60, "low_volatility": .40}),
    "trend_volume": ("趋势 + 成交量异动", {"trend": .70, "volume_ratio": .30}),
    **{key: (label, {key: 1.0}) for key, (label, _) in FACTORS.items()},
}
INDEX_BENCHMARKS = {
    "000300": ("csi300", "沪深 300"),
    "000680": ("star_composite", "科创综指"),
    "000688": ("star50", "科创 50"),
    "H20269": ("dividend_low_vol_total", "红利低波全收益"),
    "H30269": ("dividend_low_vol_price", "红利低波价格"),
}
EXIT_LABELS = {"top20": "跌出 Top 20", "top30": "跌出 Top 30", "sum50": "两日排名和 > 50"}


def index_benchmarks(path, days, *, include_sparse=False):
    """Only calculate a like-for-like return when both replay dates have closing points."""
    values = {}
    if path is not None:
        with path.open(newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source)
            if not {"day", "code", "close", "source"}.issubset(reader.fieldnames or ()):
                raise ValueError("指数 CSV 必须包含 day,code,close,source。")
            for row in reader:
                day, code, close = date.fromisoformat(row["day"]).isoformat(), row["code"].strip(), float(row["close"])
                key = code, day
                if code not in INDEX_BENCHMARKS or not math.isfinite(close) or close <= 0 or key in values:
                    raise ValueError(f"无效或重复的指数点位：{code} {day}")
                values[key] = (close, row["source"].strip())
    details, curves = {}, {}
    for code, (key, label) in INDEX_BENCHMARKS.items():
        quotes = [values.get((code, day)) for day in days]
        begin, end = (quote[0] if quote is not None else None for quote in (quotes[0], quotes[-1]))
        complete = all(quote is not None for quote in quotes)
        details[key] = {"label": label, "code": code, "start_close": begin, "end_close": end,
                        "total_return": end / begin - 1 if begin is not None and end is not None else None,
                        "observations": sum(quote is not None for quote in quotes), "daily": complete,
                        "source": "、".join(dict.fromkeys(quote[1] for quote in quotes if quote is not None))}
        if complete or (include_sparse and any(quote is not None for quote in quotes)):
            base = next(quote[0] for quote in quotes if quote is not None)
            curves[key + "_reference"] = [quote[0] / base if quote is not None else None for quote in quotes]
    return details, curves


def executable_limits(dates, listed, group, raw_open, raw_close, factor, traded, status, is_st):
    previous_close = np.r_[np.nan, raw_close[:-1]]
    bands = np.full(len(dates), [.10, .20, .20, .30][group])
    if group == MAIN:
        bands[np.r_[False, is_st[:-1]] & (dates < "2026-07-06")] = .05
    upper, lower = price_grid(previous_close * (1 + bands)), price_grid(previous_close * (1 - bands))
    official = np.isfinite(status[:, 1:]).all(1)
    upper = np.where(official, status[:, 1], upper)
    lower = np.where(official, status[:, 2], lower)
    stable_factor = np.isclose(factor, np.r_[np.nan, factor[:-1]], rtol=2e-5, atol=0)
    estimated = traded & np.r_[False, traded[:-1]] & stable_factor
    estimated &= (raw_open <= upper + .005) & (raw_open >= lower - .005)
    if listed >= dates[0]:
        estimated[:np.searchsorted(dates, listed) + 5] = False
    known = official | estimated
    return np.where(known, upper, np.nan), np.where(known, lower, np.nan), official & traded


class ReplayFeatures:
    def __init__(self, *, intraday=False):
        self.values = {}
        self.current_day = None
        self.intraday = intraday

    def collect(self, sid, shape, dates, symbol, listed, raw, status, is_st):
        adj_open, adj_close, adj_high, adj_low, factor, qvolume, cap, amount = raw.T
        volume = np.rint(qvolume * factor * 100)
        with np.errstate(divide="ignore", invalid="ignore"):
            open_, close = price_grid(adj_open / factor), price_grid(adj_close / factor)
        traded = np.isfinite(open_) & np.isfinite(close) & (open_ > 0) & (close > 0) & (volume > 0)
        amount = np.where(amount > 0, amount * 1000, close * volume)
        amount = np.where(traded, amount, np.nan)
        # Ratios compare with preceding sessions and never include the current row in the median.
        volume = np.where(traded, volume, np.nan)
        amount_median = pd.Series(amount).shift(1).rolling(VOLUME_WINDOW, min_periods=VOLUME_MIN).median()
        volume_median = pd.Series(volume).shift(1).rolling(VOLUME_WINDOW, min_periods=VOLUME_MIN).median()
        prices = pd.Series(np.where(traded, adj_close, np.nan))
        returns = {n: prices.pct_change(n, fill_method=None) for n in (1, 5, 20)}
        macd = pd.Series(np.nan, index=prices.index)
        macd.loc[traded] = macd_histogram(prices.loc[traded])
        upper, lower, official = executable_limits(dates, listed, board(symbol), open_, close, factor,
                                                     traded, status, is_st)
        values = {"open": np.where(traded, open_, np.nan), "close": np.where(traded, close, np.nan),
                  "adj": factor, "volume": volume, "amount": amount, "market_cap": cap * 10000,
                  "volume_ratio": volume / volume_median, "amount_ratio": amount / amount_median,
                  "return_1": returns[1], "trend": .35 * returns[5] + .65 * returns[20],
                  "reversal_5": -returns[5], "low_volatility": -returns[1].rolling(20, min_periods=15).std(),
                  "macd": macd, "limit_up": upper, "limit_down": lower,
                  "limit_known": np.isfinite(upper), "official_limit": official}
        if self.intraday:
            with np.errstate(divide="ignore", invalid="ignore"):
                values["high"] = price_grid(adj_high / factor)
                values["low"] = price_grid(adj_low / factor)
        for column, array in values.items():
            if sid == 0:
                self.values[column] = np.empty(shape, dtype=bool if column in ("limit_known", "official_limit")
                                               else np.float64 if column == "adj" else np.float32)
            self.values[column][sid] = array

    def select(self, day, index, symbols):
        self.current_day = day
        self.frame = pd.DataFrame({key: values[:, index] for key, values in self.values.items()},
                                  index=pd.Index(symbols, name="ticker"))
        self.frame["code"] = self.frame.index

    def day(self, day):
        if day != self.current_day:
            raise ValueError(f"行情截面日期不匹配：{day}")
        return self.frame


class TypedScores:
    def __init__(self, folder, panel, device, first_day):
        folder = Path(folder)
        meta = json.loads((folder / "model.json").read_text(encoding="utf-8"))
        status = json.loads((folder / "progress.json").read_text(encoding="utf-8"))
        self.validated_through = status["splits"]["validation"]["end"]
        if status["status"] != "finished" or first_day <= self.validated_through:
            raise ValueError("回放起始日须晚于已完成模型的验证截止日。")
        self.digest = sha256(folder / "weights.pt")
        if self.digest != meta["weights_sha256"] or panel.source != meta["source"]:
            raise ValueError("模型权重校验失败，或训练与回放的行情版本不一致。")
        self.model = TypedTransformer(**meta["architecture"]).to(device).eval()
        self.model.load_state_dict(torch.load(folder / "weights.pt", map_location=device, weights_only=True))
        self.context = meta["architecture"]["context"]
        self.device, self.panel = device, panel
        core = JointTokenizer(**meta["tokenizer"]).encode(panel.features)
        self.tokens = np.where(core >= 0, panel.kind.astype(np.int16) * VOCABULARY + core, -1)
        self.symbol_index = {code: i for i, code in enumerate(panel.symbols)}

    @torch.inference_mode()
    def add(self, frame, index, candidates):
        ids = np.array([self.symbol_index[code] for code in sorted(candidates)
                        if code in self.symbol_index], dtype=np.int64)
        window = self.tokens[ids, index + 1 - self.context:index + 1]
        ids = ids[(window >= 0).all(1)]
        result = np.full(len(frame), np.nan)
        for start in range(0, len(ids), 256):
            group = ids[start:start + 256]
            x, aux = batch_inputs(self.panel, self.tokens, group, np.full(len(group), index + 1),
                                  self.context, self.device)
            _, _, score = self.model(x, aux)
            result[group] = score.cpu().numpy()
        frame["token_transformer"] = result


def capped_recall(found, industries, liquidity, positions):
    # Holders are always present, even if their quote is missing on the signal day.
    result = {code: {**found.get(code, {}), "holdings": "已有持仓"} for code in sorted(positions)}
    for code in found:
        if len(result) == RECALL_SIZE:
            break
        result.setdefault(code, found[code])
    for code in liquidity:
        if len(result) == RECALL_SIZE:
            break
        result.setdefault(code, {"liquidity": "按当日成交额补齐"})
    return result, industries


@dataclass
class Decision:
    day: str
    ranks: dict
    buys: tuple
    exits: tuple
    passive_exits: tuple
    bars: dict
    previous_ranks: dict


class ReplayBook:
    def __init__(self, ranker, exit_rule=None, execution=None):
        self.ranker = ranker
        self.exit_rule = exit_rule
        self.execution = execution
        self.previous_ranks = {}
        self.book = Book(ranker.strategy)
        self.pending = None
        self.daily, self.decisions, self.trades = [], [], []

    def execute(self, day, bars):
        book = self.book
        book.mark_open(day, bars, self.ranker.securities)
        signal = self.pending
        if signal is None:
            return
        self.pending = None
        first_trade = len(book.trades)
        triggered = {}
        for code in signal.exits:
            if code in book.positions:
                book.order(day, signal.day, code, True, book.positions[code].shares, bars, signal.bars)
        budget = book.equity() / BUY_RANK
        depth = self.execution.buy_depth if self.execution else 0
        candidates = signal.buys
        if self.execution and self.execution.reserve_buy_slots:
            candidates = candidates[:max(0, BUY_RANK - len(book.positions))]
            if candidates:
                budget = min(budget, book.cash / len(candidates))
        for code in candidates:
            if len(book.positions) >= BUY_RANK:
                break
            if code in book.positions:
                continue
            if depth:
                price, target, reason = limit_fill(bars.get(code), signal.bars.get(code), False)
                if reason is None:
                    triggered[code] = ("dip_buy", target, price)
                    price = book.slipped_price(price, False, target)
            elif code in bars:
                price, reason = book.fill_price(bars[code], False)
            else:
                price, reason = None, "no_trading_bar"
            if reason is not None:
                book.blocked.append((day, signal.day, code, "buy", None, reason))
                continue
            wanted = math.floor(budget / price)
            while wanted > 0 and price * wanted + book.fee(price * wanted, False) > budget:
                wanted = math.floor((budget - book.fee(price * wanted, False)) / price)
            book.order(day, signal.day, code, False, max(wanted, 0), bars, signal.bars,
                       execution_price=price if depth else None)
        # An optional sell may happen after every buy trigger. Its proceeds and slots
        # are held back until the next session when timestamps are unavailable.
        for code in signal.passive_exits:
            position = book.positions.get(code)
            if position is None:
                continue
            price, target, reason = limit_fill(bars.get(code), signal.bars.get(code), True)
            if reason:
                book.blocked.append((day, signal.day, code, "sell", position.shares, reason))
                continue
            triggered[code] = ("rally_sell", target, price)
            price = book.slipped_price(price, True, target)
            book.order(day, signal.day, code, True, position.shares, bars, signal.bars, execution_price=price)
        for trade in book.trades[first_trade:]:
            _, _, code, side, shares, price, value, _ = trade
            bar = bars[code]
            fees = book.fee_components(value, side == "sell")
            rank, previous = signal.ranks.get(code), signal.previous_ranks.get(code)
            detail = (previous, previous + rank if previous is not None and rank is not None else None)
            row = (*trade, bar.open, *fees, rank, "official" if bar.official_limit else "estimated", *detail)
            kind, target, reference = triggered.get(code, ("open", None, bar.open))
            if self.execution:
                lag = signal.bars.get(code)
                row += (kind, lag.close if lag is not None else None, target)
            cost = (reference - price if side == "sell" else price - reference) * shares
            self.trades.append((*row, reference, cost))

    def record(self, day, bars, reference, ranked=None, *, star_reference=None):
        book = self.book
        book.mark_close(bars)
        self.daily.append((day, book.equity(), book.cash, len(book.positions), reference, star_reference,
                           sum(code not in bars for code in book.positions),
                           len(ranked) if ranked is not None else None,
                           int(ranked["rank"].count()) if ranked is not None else None))
        if ranked is None:
            return
        hold = book.positions
        watch_limit = 30 if self.exit_rule else SELL_RANK
        rows = ranked[ranked["rank"].le(watch_limit) | ranked["code"].isin(hold)]
        ranks = {r.code: int(r.rank) for r in rows.itertuples(index=False) if pd.notna(r.rank)}
        entry_depth = self.execution.buy_depth if self.execution and self.execution.buy_depth else BUY_RANK
        buys = tuple(code for code, rank in ranks.items() if rank <= entry_depth and code not in hold)
        limit = self.exit_rule or "top20"
        exits = tuple(code for code in hold if code in ranks and should_exit(
            limit, ranks[code], self.previous_ranks.get(code)))
        passive = ()
        if self.execution and self.execution.take_profit:
            passive = tuple(code for code in hold if code in ranks and code in self.previous_ranks
                            and 30 < ranks[code] + self.previous_ranks[code] <= 50)
        for row in rows.itertuples(index=False):
            rank = ranks.get(row.code)
            action = ("sell" if row.code in exits else "take_profit" if row.code in passive else "hold"
                      ) if row.code in hold else ("buy_candidate" if row.code in buys else "watch")
            previous = self.previous_ranks.get(row.code)
            rank_sum = previous + rank if previous is not None and rank is not None else None
            self.decisions.append((day, row.code, rank, row.score, row.routes, action, previous, rank_sum))
        self.pending = Decision(day, ranks, buys, exits, passive,
                                {code: bars[code] for code in (*buys, *exits, *passive) if code in bars},
                                self.previous_ranks)
        self.previous_ranks = ranks if self.exit_rule else {}

    def results(self):
        book = self.book
        frames = {
            "equity": pd.DataFrame(self.daily, columns=["day", "equity", "cash", "positions", "universe_reference",
                                                        "star_board_reference",
                                                        "stale_positions", "recall_count", "scored_count"]),
            "trades": pd.DataFrame(self.trades, columns=["day", "signal_day", "code", "side", "shares", "price",
                                                        "value", "fee", "open", "commission", "transfer",
                                                        "stamp_duty", "rank", "limit_source",
                                                        "previous_rank", "rank_2day"] +
                                       (["fill_rule", "previous_close", "trigger_price"] if self.execution else []) +
                                       ["reference_price", "slippage"]),
            "blocked": pd.DataFrame(book.blocked, columns=["day", "signal_day", "code", "side", "wanted", "reason"]),
            "events": pd.DataFrame(book.events, columns=["day", "code", "event", "value"]),
            "decisions": pd.DataFrame(self.decisions, columns=["signal_day", "code", "rank", "score", "routes", "action",
                                                              "previous_rank", "rank_2day"]),
        }
        if not self.exit_rule:
            for name in ("trades", "decisions"):
                frames[name].drop(columns=["previous_rank", "rank_2day"], inplace=True)
        equity, trades = frames["equity"], frames["trades"]
        summary = stats(equity, trades, book.settings.initial_cash)
        fee_total = {name: float(trades[name].sum()) for name in ("commission", "transfer", "stamp_duty")}
        summary.update(fee_components=fee_total, slippage=float(trades["slippage"].sum()),
                       slippage_bps=book.settings.slippage_bps,
                       average_positions=float(equity["positions"].iloc[1:].mean()),
                       ending_stale=int(equity["stale_positions"].iloc[-1]),
                       stale_position_days=int(equity["stale_positions"].sum()),
                       buys=int((trades["side"] == "buy").sum()), sells=int((trades["side"] == "sell").sum()),
                       limit_estimated_fills=int((trades["limit_source"] == "estimated").sum()),
                       blocked_orders=dict(Counter(frames["blocked"]["reason"])),
                       position_events=dict(Counter(frames["events"]["event"])))
        if self.execution:
            blocked = frames["blocked"]
            for side, kind, direction in (("buy", "dip_buy", -1), ("sell", "rally_sell", 1)):
                filled = trades[trades["side"] == side]
                hits = filled[filled["fill_rule"] == kind]
                attempts = len(filled) + int(blocked["side"].eq(side).sum())
                summary[side + "_fill_rate"] = len(filled) / attempts if attempts else None
                summary[kind + "_fills"] = len(hits)
                summary[kind + "_vs_open"] = (float((direction * (hits["price"] / hits["open"] - 1)).mean())
                                               if len(hits) else None)
            summary["outside_top10_buys"] = int((trades["side"].eq("buy") & trades["rank"].gt(BUY_RANK)).sum())
        return summary, frames


def should_exit(rule, rank, previous_rank):
    if rule == "sum50":
        return previous_rank is not None and rank + previous_rank > 50
    return rank > (30 if rule == "top30" else 20)


def low_vol_grid(existing):
    portfolios, exits = {}, {}
    for exit_rule, exit_name in EXIT_LABELS.items():
        for key, (label, weights) in existing.items():
            for share in ((100,) if key == "main" else (65, 50, 35)):
                variant = "main" if key == "main" and exit_rule == "top20" else f"{key}_{share}_{exit_rule}"
                mix = ({factor: (share if factor == "low_volatility" else 100 - share) / 100
                        for factor in weights})
                portfolios[variant] = (f"{label} · 原低波 {share}% · {exit_name}", mix)
                exits[variant] = exit_rule
    return portfolios, exits


class GridRanks:
    """Share cross-sectional factor ranks among portfolios with identical recall sets."""

    def __init__(self, current, fields):
        self.current, self.fields = current, sorted(fields)
        self.cache = {}

    def rank(self, found, weights, *, include_components=False):
        codes = tuple(sorted(found))
        if codes not in self.cache:
            frame = self.current.reindex(list(codes))
            raw = frame[self.fields].where(frame["close"].notna(), np.nan, axis=0)
            raw = raw.where(np.isfinite(raw))
            percentiles = ((raw.rank() - .5) / raw.count() * 100)
            self.cache[codes] = {name: percentiles[name].to_numpy() for name in percentiles}
        factors = self.cache[codes]
        base = factors["low_volatility"]
        if len(weights) == 1:
            score, coverage = base, np.isfinite(base).astype(float)
        else:
            other = next(key for key in weights if key != "low_volatility")
            base_weight = weights["low_volatility"]
            has_other = np.isfinite(factors[other])
            coverage = np.where(has_other, 1.0, base_weight)
            # Keep the low-vol anchor available for every weight and fall back to it if needed.
            total = base_weight * base + weights[other] * np.nan_to_num(factors[other], nan=0)
            score = total / coverage
        order = np.lexsort((np.arange(len(codes)), -coverage, -score))
        ordered_codes = np.asarray(codes, dtype=str)[order]
        ranks = pd.array(np.arange(1, len(codes) + 1), dtype="Int64")
        ranks[~np.isfinite(score[order])] = pd.NA
        result = pd.DataFrame({"code": ordered_codes, "rank": ranks, "score": score[order],
                               "routes": ["|".join(found[code]) for code in ordered_codes]})
        if include_components:
            for field in weights:
                result[field] = factors[field][order]
        return result


class EmptyReference:
    def industry_rows(self):
        return []

    def securities(self):
        return {}

    def hotspots(self, day, lookback):
        return []


def rotate(args, progress):
    if args.out.exists() or args.start >= args.end:
        raise ValueError("回放日期无效，或输出目录已存在。")
    low_vol_path = getattr(args, "low_vol_model", None)
    grid = getattr(args, "low_vol_grid", False)
    intraday = getattr(args, "intraday", False)
    slippage_bps = getattr(args, "slippage_bps", 0)
    compare_slippage = getattr(args, "compare_slippage", False)
    if not math.isfinite(slippage_bps) or not 0 <= slippage_bps < 10000:
        raise ValueError("滑点须在 0–10000 bp 之间。")
    if compare_slippage and not slippage_bps:
        raise ValueError("滑点对照需要指定大于 0 的 --slippage-bps。")
    if (grid or intraday) and not low_vol_path:
        raise ValueError("--low-vol-grid 和 --intraday 需要 --low-vol-model。")
    if grid and intraday:
        raise ValueError("--low-vol-grid 与 --intraday 请选择其一。")
    exit_rules = {}
    if low_vol_path:
        from .low_vol import LowVolFeatures, LowVolScores, low_vol_portfolios
        context, features, portfolios = 60, LowVolFeatures(intraday=intraday), low_vol_portfolios()
        if grid:
            portfolios, exit_rules = low_vol_grid(portfolios)
        if intraday:
            weights = {"low_volatility": .35, "mlp_value": .65}
            portfolios = {key: (label, weights) for key, (label, _) in EXECUTION_RULES.items()}
            exit_rules = dict.fromkeys(portfolios, "sum50")
    else:
        status = json.loads((args.model / "progress.json").read_text(encoding="utf-8"))
        context = status["architecture"]["context"]
        if status["status"] != "finished" or args.start <= status["splits"]["validation"]["end"]:
            raise ValueError("回放起始日须晚于已完成模型的验证截止日。")
        features, portfolios = ReplayFeatures(), PORTFOLIOS
    market_path = args.db if args.db.is_file() else None
    panel = load_research_panel(args.qlib_db, market_path, args.status_csv, args.start, args.end,
                                max(context, 60), progress, on_symbol=features.collect,
                                include_valuation=bool(low_vol_path))
    days = np.flatnonzero(panel.dates >= args.start)
    if len(days) < 2:
        raise ValueError("回放区间至少需要两个有报价的市场交易日。")
    labels = [str(panel.dates[index]) for index in days]
    indices, index_curves = index_benchmarks(args.benchmark_csv, labels, include_sparse=True)
    if low_vol_path:
        scorer = LowVolScores(low_vol_path, panel, labels[0])
        model_info = {"run": low_vol_path.name, **scorer.metadata}
    else:
        scorer = TypedScores(args.model, panel, args.device, labels[0])
        model_info = {"run": args.model.name, "weights_sha256": scorer.digest,
                      "validated_through": scorer.validated_through, "context": context}
    strategy = Strategy(weights=portfolios["main"][1], min_weight_coverage=.65, initial_cash=args.cash,
                        volume_count=RECALL_SIZE, positions=BUY_RANK, slippage_bps=slippage_bps)
    base_variants = {key: key for key in portfolios}
    if compare_slippage:
        base_variants.update({f"{key}_no_slip": key for key in portfolios})
    settings = {key: replace(strategy, weights=portfolios[original][1],
                             slippage_bps=slippage_bps if key == original else 0)
                for key, original in base_variants.items()}
    connection = closing(open_readonly(market_path)) if market_path else nullcontext(None)
    route_totals = Counter()
    with connection as db:
        if db is not None:
            db.execute("BEGIN")
        market = MarketSnapshot(db) if db is not None else EmptyReference()
        books = {key: ReplayBook(Ranker(market, features, rule, [FactorModel(k) for k in rule.weights]),
                                 exit_rules.get(base_variants[key]),
                                 EXECUTION_RULES[base_variants[key]][1] if intraday else None)
                 for key, rule in settings.items()}
        recall = books["main"].ranker.recall
        grid_ranking = grid or intraday
        grid_fields = set().union(*(rule.weights.keys() for rule in settings.values())) if grid_ranking else set()
        reference = star_reference = 1.0
        star_codes = [code for code in panel.symbols if code.startswith(("688", "689")) and code.endswith(".SH")]
        for offset, index in enumerate(days):
            day = str(panel.dates[index])
            features.select(day, index, panel.symbols)
            frame = features.day(day)
            quoted = frame[frame["open"].notna()]
            bars = {bar.code: bar for bar in quoted.itertuples(index=False)}
            for replay in books.values():
                replay.execute(day, bars)
            if offset and quoted["return_1"].notna().any():
                reference *= 1 + float(quoted["return_1"].mean())
            star_returns = frame.loc[star_codes, "return_1"].dropna()
            if offset and not star_returns.empty:
                star_reference *= 1 + float(star_returns.mean())
            if offset == len(days) - 1:
                for replay in books.values():
                    replay.record(day, bars, reference, star_reference=star_reference)
                break
            found, industries = recall.candidates(day, frame, {})
            liquidity = frame[frame["amount"] >= strategy.min_amount].sort_values(
                "amount", ascending=False, kind="stable").index[:RECALL_SIZE]
            pools = {}
            recalling = {}
            for key, replay in books.items():
                held = tuple(sorted(replay.book.positions))
                if held not in pools:
                    pools[held] = capped_recall(found, industries, liquidity, held)
                recalling[key] = pools[held]
            for routes in recalling["main"][0].values():
                route_totals.update(routes.keys())
            candidates = set().union(*(set(data[0]) for key, data in recalling.items()
                                       if low_vol_path or "token_transformer" in settings[key].weights))
            scorer.add(frame, index, candidates)
            ranker = GridRanks(frame, grid_fields) if grid_ranking else None
            for key, replay in books.items():
                ranked = (ranker.rank(recalling[key][0], settings[key].weights) if grid_ranking else
                          replay.ranker.rank(day, replay.book.positions, recalled=recalling[key]))
                replay.record(day, bars, reference, ranked, star_reference=star_reference)
            if (offset + 1) % 30 == 0:
                progress(f"已回放 {offset + 1}/{len(days)} 个交易日，最新 {day}。")
    variants, tables = {}, {}
    for key, replay in books.items():
        outcomes, tables[key] = replay.results()
        for column, levels in index_curves.items():
            tables[key]["equity"][column] = levels
        original = base_variants[key]
        label, weights = portfolios[original]
        if compare_slippage:
            label += f" · {settings[key].slippage_bps:g} bp"
        variants[key] = {"label": label, "weights": weights, "factor_count": len(weights), **outcomes}
        if grid_ranking:
            equity = tables[key]["equity"].set_index("day")["equity"]
            value = tables[key]["trades"].groupby("day")["value"].sum()
            turnover = value.reindex(equity.index, fill_value=0).div(equity.shift()).iloc[1:].mean()
            variants[key].update(exit_rule=exit_rules[original], exit_label=EXIT_LABELS[exit_rules[original]],
                                 baseline_share=weights["low_volatility"], daily_turnover=float(turnover))
        if intraday:
            variants[key]["execution"] = vars(replay.execution)
            if replay.execution.take_profit:
                variants[key]["exit_label"] += "时开盘卖出，30 < 两日排名和 ≤ 50 时挂 +1% 卖单"
    main_equity = tables["main"]["equity"]
    benchmarks = {
        **indices,
        "star_board": {"label": "科创板全板等权", "total_return": star_reference - 1, "daily": True,
                       "source": "Qlib，科创板有相邻交易日有效报价的股票等权"},
        "universe": {"label": "全市场等权", "total_return": reference - 1, "daily": True,
                     "source": "Qlib，A 股有相邻交易日有效报价的股票等权"},
    }
    config = {**strategy.as_dict(), "recall_size": RECALL_SIZE, "buy_rank": BUY_RANK, "sell_rank": SELL_RANK}
    report = {
        "run": args.out.name, "kind": "rotation_top10_top20",
        "comparison_version": 6 if slippage_bps else 5 if intraday else 4 if grid else 3 if low_vol_path else 2,
        "factors": {key: {"label": label, "definition": definition}
                    for key, (label, definition) in FACTORS.items()}, "start": str(panel.dates[days[0]]),
        "end": str(panel.dates[days[-1]]), "source": panel.source, "strategy": config,
        "model": model_info,
        "routes": dict(route_totals), "full_recall_days": int(main_equity["recall_count"].eq(RECALL_SIZE).sum()),
        "signal_days": len(days) - 1, "variants": variants, "benchmarks": benchmarks,
    }
    if compare_slippage:
        report["default_variant"] = "both" if intraday else "main"

    def writer(root):
        for key, frames in tables.items():
            subfolder = root / key
            subfolder.mkdir()
            for name, frame in frames.items():
                frame.to_csv(subfolder / f"{name}.csv", index=False)
        (root / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
                                           encoding="utf-8")

    publish_directory(args.out, writer)
    return report
