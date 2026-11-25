"""Execution rules for limit orders replayed from daily OHLC bars."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class IntradayRule:
    buy_depth: int = 0
    take_profit: bool = False
    reserve_buy_slots: bool = False


EXECUTION_RULES = {
    "main": ("开盘对照", IntradayRule()),
    "dip_buy": ("仅 −1% 买入 · 顺位至 Top 30", IntradayRule(buy_depth=30)),
    "rally_sell": ("仅 +1% 提前卖出", IntradayRule(take_profit=True)),
    "both": ("−1% 买入＋1% 提前卖出 · 顺位至 Top 30", IntradayRule(30, True)),
    "both_reserved": ("−1% 买入＋1% 提前卖出 · 开盘固定买单", IntradayRule(10, True, True)),
}


def limit_fill(bar, lag, sell):
    """Return the simulated fill, order limit and any reason for declining the order."""
    if bar is None:
        return None, None, "no_trading_bar"
    if not bar.limit_known:
        return None, None, "limit_unknown"
    if lag is None or not math.isclose(bar.adj, lag.adj, rel_tol=2e-5):
        return None, None, "no_comparable_previous_close"
    extreme = bar.high if sell else bar.low
    if not math.isfinite(extreme) or extreme <= 0:
        return None, None, "no_intraday_range"
    previous_cents = round(lag.close * 100)
    target_cents = (previous_cents * 101 + 99) // 100 if sell else previous_cents * 99 // 100
    opening = round(bar.open * 100)
    intraday = round(extreme * 100)
    target = target_cents / 100
    if (intraday < target_cents if sell else intraday > target_cents) or target_cents <= 0:
        return None, target, "intraday_target_not_reached"
    fill = (max(opening, target_cents) if sell else min(opening, target_cents)) / 100
    lower, upper = round(bar.limit_down * 100) / 100, round(bar.limit_up * 100) / 100
    if not (lower <= fill <= upper) or (fill == lower if sell else fill == upper):
        return None, target, "intraday_fill_at_limit"
    return fill, target, None
