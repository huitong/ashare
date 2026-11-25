"""Causal daily factors shared by training, ranking and simulation."""

import numpy as np
import pandas as pd

TOKEN_FEATURES = ("return_1", "log_cap", "log_volume", "log_volume_ratio")
ELIGIBLE_VOLUME_WINDOW = 20


def macd_histogram(prices):
    fast = prices.ewm(span=12, adjust=False, min_periods=12).mean()
    slow = prices.ewm(span=26, adjust=False, min_periods=26).mean()
    diff = fast - slow
    signal = diff.ewm(span=9, adjust=False, min_periods=9).mean()
    return (diff - signal) / prices


class Features:
    def __init__(self, market, end):
        first = market.status()["first_complete"]
        if first is None or first > end:
            raise ValueError("请求区间内没有已同步行情。")
        self.sessions = market.sessions(first, end)
        self.calendar = {day: i for i, day in enumerate(self.sessions)}
        self.frame = self._build(market.frame(first, end))
        self.by_day = self.frame.groupby("day", sort=False).indices

    def _build(self, frame):
        frame["session_id"] = frame["day"].map(self.calendar).astype("int32")
        frame["active_price"] = frame["close"] * frame["adj"]
        grouped = frame.groupby("code", sort=False)
        for period in (1, 5, 20):
            lag = grouped["active_price"].shift(period)
            age = frame["session_id"] - grouped["session_id"].shift(period)
            frame[f"return_{period}"] = (frame["active_price"] / lag - 1).where(age == period)
        for col in ("volume", "amount"):
            frame[f"prior_{col}"] = grouped[col].transform(
                lambda s: s.shift(1).rolling(ELIGIBLE_VOLUME_WINDOW, min_periods=ELIGIBLE_VOLUME_WINDOW).median()
            )
        frame["volume_ratio"] = frame["volume"] / frame["prior_volume"].replace(0, np.nan)
        frame["amount_ratio"] = frame["amount"] / frame["prior_amount"].replace(0, np.nan)
        frame["log_cap"] = np.log10(frame["market_cap"])
        frame["log_volume"] = np.log10(frame["volume"].where(frame["volume"] > 0))
        frame["log_volume_ratio"] = np.log(frame["volume_ratio"].where(frame["volume_ratio"] > 0))
        frame["trend"] = 0.35 * frame["return_5"] + 0.65 * frame["return_20"]

        frame["macd"] = grouped["active_price"].transform(macd_histogram)
        return frame

    def day(self, day):
        indices = self.by_day.get(day, [])
        return self.frame.iloc[indices].set_index("code", drop=False)

    def window_ends(self, context):
        frame = self.frame
        if len(frame) < context:
            return np.empty(0, dtype=np.int64)
        codes = frame["code"].to_numpy()
        ordinal = frame["session_id"].to_numpy()
        # Return for the first context day also needs its preceding session.
        eligible = (codes[context:] == codes[:-context]) & (ordinal[context:] - ordinal[:-context] == context)
        eligible &= np.isfinite(frame["return_1"].to_numpy()[context:])
        return np.flatnonzero(eligible) + context
