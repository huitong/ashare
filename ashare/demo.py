"""Deterministic synthetic input for smoke tests and the local demo."""

from datetime import date, datetime

import numpy as np
import pandas as pd

from .store import SHANGHAI, cutoff


def create_demo(store, end, days=165, stocks=48, seed=42):
    if days < 1 or stocks < 8:
        raise ValueError("模拟数据至少需要 1 日和 8 只股票。")
    sessions = [x.date().isoformat() for x in pd.bdate_range(end=end, periods=days)]
    rng = np.random.default_rng(seed)
    codes = [f"{600000 + i:06d}.SH" for i in range(stocks)]
    industries = ("电子", "计算机", "银行", "医药生物", "电力设备", "食品饮料")
    sectors = np.arange(stocks) % len(industries)
    listings = [(code, f"模拟股票{i + 1:03d}", "2000-01-01", None) for i, code in enumerate(codes)]
    store.register_market("synthetic", listings, sessions)
    store.put_industries([
        (code, industries[sectors[i]], "2000-01-01", None, cutoff("2000-01-01"))
        for i, code in enumerate(codes)
    ])
    price = rng.uniform(10, 120, stocks)
    outstanding = rng.uniform(0.2, 4, stocks) * 1e9
    regular_volume = rng.integers(1_000_000, 15_000_000, stocks)
    momentum = np.zeros(stocks)
    for t, day in enumerate(sessions):
        previous = price.copy()
        theme = rng.normal(0, 0.008, len(industries))
        market_move = rng.normal(0, 0.006)
        shock = rng.normal(0, 0.016, stocks)
        returns = np.clip(0.24 * momentum + theme[sectors] + market_move + shock, -0.095, 0.095)
        momentum = returns
        price = np.round(previous * (1 + returns), 2)
        change = np.clip(0.35 * returns + rng.normal(0, 0.006, stocks), -0.09, 0.09)
        opens = np.round(previous * (1 + change), 2)
        lows = np.round(np.minimum(opens, price) * 0.997, 2)
        highs = np.round(np.maximum(opens, price) * 1.003, 2)
        volume = (regular_volume * rng.lognormal(0, 0.5, stocks) * (1 + 12 * np.abs(returns))).astype(int)
        rows = [
            (day, code, float(opens[i]), float(highs[i]), float(lows[i]), float(price[i]),
             float(previous[i]), float(volume[i]), float(volume[i] * price[i]), float(outstanding[i] * price[i]),
             1.0, round(float(previous[i]) * 1.1, 2), round(float(previous[i]) * 0.9, 2), 1)
            for i, code in enumerate(codes)
        ]
        store.save_day(day, rows)
        if t % 7 == 0:
            event_time = datetime.combine(date.fromisoformat(day), datetime.min.time(), SHANGHAI).isoformat()
            store.put_hotspots([
                (f"synthetic-{day}", "industry", industries[theme.argmax()], event_time, event_time, 2.0, "模拟行业热点"),
            ])
    store.put_holdings(cutoff(sessions[-1]), [(codes[0], 1000)])
    return sessions
