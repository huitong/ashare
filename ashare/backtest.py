"""Long-only opening-auction simulation for end-of-day signals."""

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Position:
    shares: int
    adj: float
    mark: float


def lot_rule(code):
    if code.startswith(("688", "689")) and code.endswith(".SH"):
        return 200, 1
    if code.endswith(".BJ"):
        return 100, 1
    return 100, 100


def trade_size(code, wanted, maximum, allow_odd_lot_exit=False):
    minimum, step = lot_rule(code)
    size = min(wanted, maximum)
    if size == wanted and allow_odd_lot_exit:
        return size
    if size < minimum:
        return 0
    return minimum + (size - minimum) // step * step


class Book:
    def __init__(self, settings):
        self.settings = settings
        self.cash = settings.initial_cash
        self.positions = {}
        self.trades = []
        self.blocked = []
        self.events = []

    def equity(self):
        return self.cash + sum(position.shares * position.mark for position in self.positions.values())

    def mark_open(self, day, bars, securities):
        for code, position in list(self.positions.items()):
            if code in bars:
                bar = bars[code]
                if not math.isclose(position.adj, bar.adj, rel_tol=1e-8):
                    # Approximate total return by reinvesting the factor change into economic shares.
                    economic_shares = position.shares * bar.adj / position.adj
                    position.shares = math.floor(economic_shares + 1e-9)
                    self.cash += (economic_shares - position.shares) * bar.open
                    self.events.append((day, code, "adjustment_factor_proxy", bar.adj / position.adj))
                    position.adj = bar.adj
                position.mark = bar.open
                if position.shares == 0:
                    del self.positions[code]
            else:
                delisted = securities.get(code, (None, None, None))[2]
                if delisted is not None and delisted < day:
                    del self.positions[code]
                    self.events.append((day, code, "delisted_zero_recovery", position.shares * position.mark))

    def mark_close(self, bars):
        for code, position in self.positions.items():
            if code in bars:
                position.mark = bars[code].close

    def fee_components(self, value, sell):
        s = self.settings
        return (max(s.min_commission, value * s.commission_bps / 10000),
                value * s.transfer_bps / 10000, value * s.sell_tax_bps / 10000 if sell else 0)

    def fee(self, value, sell):
        return sum(self.fee_components(value, sell))

    def slipped_price(self, reference, sell, bound):
        price = reference * (1 + (-1 if sell else 1) * self.settings.slippage_bps / 10000)
        return max(price, bound) if sell else min(price, bound)

    def fill_price(self, bar, sell):
        if not bar.limit_known:
            return None, "limit_unknown"
        bound = bar.limit_down if sell else bar.limit_up
        if bound is not None and not pd.isna(bound):
            if (sell and bar.open <= bound + 1e-6) or (not sell and bar.open >= bound - 1e-6):
                return None, "opening_at_limit"
        else:
            bound = 0 if sell else float("inf")
        return self.slipped_price(bar.open, sell, bound), None

    def rebalance(self, day, signal_day, target, bars, lag_bars):
        retained = set(target)
        sells = sorted(set(self.positions) - retained) + [code for code in target if code in self.positions]
        for code in sells:
            position = self.positions[code]
            if position.shares > target.get(code, 0):
                self.order(day, signal_day, code, True, position.shares - target.get(code, 0), bars, lag_bars)
        for code, goal in target.items():
            current = self.positions[code].shares if code in self.positions else 0
            if goal > current:
                self.order(day, signal_day, code, False, goal - current, bars, lag_bars)

    def order(self, day, signal, code, sell, wanted, bars, lag_bars, *, execution_price=None):
        side = "sell" if sell else "buy"

        def blocked(reason, amount=wanted):
            self.blocked.append((day, signal, code, side, amount, reason))

        if code not in bars:
            blocked("no_trading_bar")
            return
        bar = bars[code]
        price = execution_price
        if price is None:
            price, reason = self.fill_price(bar, sell)
            if reason:
                blocked(reason)
                return
        lag = lag_bars.get(code)
        if lag is None:
            blocked("no_signal_day_volume")
            return
        maximum = math.floor(lag.volume * self.settings.participation)
        if not sell:
            cash_after_minimum = self.cash - self.settings.min_commission
            cash_per_share = price * (1 + self.settings.transfer_bps / 10000)
            commission_per_share = price * self.settings.commission_bps / 10000
            affordable = math.floor(min(
                cash_after_minimum / cash_per_share, self.cash / (cash_per_share + commission_per_share),
            ) + 1e-9)
            maximum = min(maximum, affordable)
        position = self.positions.get(code)
        full_exit = sell and wanted == position.shares
        shares = trade_size(code, wanted, maximum, allow_odd_lot_exit=full_exit)
        if not shares:
            blocked("cash_liquidity_or_lot")
            return
        value = price * shares
        fee = self.fee(value, sell)
        if sell:
            if self.cash + value < fee:
                blocked("insufficient_cash_for_fee")
                return
            self.cash += value - fee
            position.shares -= shares
            if position.shares == 0:
                del self.positions[code]
        else:
            self.cash -= value + fee
            if position is None:
                self.positions[code] = Position(shares, bar.adj, bar.open if execution_price is None else price)
            else:
                position.shares += shares
        self.trades.append((day, signal, code, side, shares, price, value, fee))


def stats(equity, trades, initial_cash):
    series = equity["equity"]
    returns = series.pct_change().dropna()
    years = len(returns) / 252
    final_ratio = series.iloc[-1] / initial_cash
    stdev = returns.std()
    return dict(
        sessions=len(equity), final_equity=float(series.iloc[-1]),
        total_return=float(final_ratio - 1),
        annualized_return=float(final_ratio ** (1 / years) - 1) if len(returns) >= 20 else None,
        max_drawdown=float((series / series.cummax() - 1).min()),
        annualized_volatility=float(stdev * np.sqrt(252)) if len(returns) > 1 else None,
        sharpe_zero_rf=float(returns.mean() / stdev * np.sqrt(252)) if stdev > 0 else None,
        universe_reference_return=float(equity["universe_reference"].iloc[-1] - 1),
        fees=float(trades["fee"].sum()), orders_filled=len(trades),
        gross_turnover=float(trades["value"].sum() / series.mean()),
    )


def simulate(ranker, days):
    settings, features = ranker.strategy, ranker.features
    book = Book(settings)
    pnl, decisions = [], []
    target, signal_day, lag_bars = None, None, None
    reference = 1.0
    for i, day in enumerate(days):
        current = features.day(day)
        bars = {bar.code: bar for bar in current.itertuples(index=False)}
        book.mark_open(day, bars, ranker.securities)
        if target is not None:
            book.rebalance(day, signal_day, target, bars, lag_bars)
            target = None
        book.mark_close(bars)
        if i and current["return_1"].notna().any():
            reference *= 1 + current["return_1"].mean()
        nav = book.equity()
        pnl.append((day, nav, book.cash, len(book.positions), reference))
        if i < len(days) - 1 and i % settings.rebalance_every == 0:
            scored = ranker.rank(day, book.positions)
            selected = scored.dropna(subset=["score"]).head(settings.positions)["code"].tolist()
            budget = nav / len(selected) if selected else 0
            target = {code: math.floor(budget / bars[code].close) for code in selected}
            signal_day, lag_bars = day, bars
            decisions.append(scored.assign(signal_day=day, selected=scored["code"].isin(target)))
    equity = pd.DataFrame(pnl, columns=["day", "equity", "cash", "positions", "universe_reference"])
    trades = pd.DataFrame(book.trades, columns=["day", "signal_day", "code", "side", "shares", "price", "value", "fee"])
    blocked = pd.DataFrame(book.blocked, columns=["day", "signal_day", "code", "side", "wanted", "reason"])
    events = pd.DataFrame(book.events, columns=["day", "code", "event", "value"])
    ranks = pd.concat([frame for frame in decisions if not frame.empty], ignore_index=True) if any(
        not frame.empty for frame in decisions
    ) else pd.DataFrame(columns=["signal_day", "code", "selected"])
    summary = stats(equity, trades, settings.initial_cash)
    summary["blocked_orders"] = dict(Counter(blocked["reason"]))
    summary["position_events"] = dict(Counter(events["event"]))
    summary["scored_signal_days"] = int(ranks.loc[ranks["selected"], "signal_day"].nunique())
    return dict(equity=equity, trades=trades, blocked=blocked, events=events, decisions=ranks, summary=summary)
