"""Recall providers and an extensible score-model interface."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from .store import cutoff


class ScoreModel(Protocol):
    model_id: str

    def score(self, day: str, candidates: pd.DataFrame, features) -> pd.Series:
        """Return raw scores indexed by stock code. Missing values mean no score."""


@dataclass(frozen=True)
class FactorModel:
    model_id: str

    def score(self, day, candidates, features):
        return candidates[self.model_id]


class Recall:
    def __init__(self, market, strategy):
        self.market = market
        self.settings = strategy
        self.memberships = market.industry_rows()

    def _industries(self, day):
        known = cutoff(day)
        active = {}
        for code, industry, start, end, available in self.memberships:
            if start > day:
                break
            if available <= known and (end is None or day < end):
                active[code] = industry
        return active

    def candidates(self, day, current, holdings):
        settings = self.settings
        industry = self._industries(day)
        eligible = current[(current["volume"] > 0) & (current["amount"] >= settings.min_amount)].copy()
        eligible["industry"] = eligible["code"].map(industry)
        found = {}

        def add(code, source, evidence):
            found.setdefault(code, {})[source] = evidence

        with_industry = eligible.dropna(subset=["industry", "return_1", "amount_ratio"]).copy()
        with_industry["breadth"] = (with_industry["return_1"] > 0).astype(float)
        with_industry["direction"] = with_industry["return_1"].clip(-0.10, 0.10)
        with_industry["activity"] = np.log(with_industry["amount_ratio"].clip(0.25, 4))
        heat = with_industry.groupby("industry").agg(
            members=("code", "size"), direction=("direction", "mean"),
            breadth=("breadth", "mean"), activity=("activity", "mean"),
        )
        heat = heat[heat["members"] >= settings.industry_min_members].copy()
        heat["heat"] = (
            0.5 * heat["direction"].rank(pct=True) + 0.25 * heat["breadth"].rank(pct=True)
            + 0.25 * heat["activity"].rank(pct=True)
        )
        hot_industries = heat["heat"].sort_values(ascending=False, kind="stable").head(settings.industry_count)
        active_names = eligible.sort_values("amount", ascending=False, kind="stable")
        for group, score in hot_industries.items():
            for code in active_names[active_names["industry"] == group].head(settings.industry_members).index:
                add(code, "industry", f"{group} 热度 {score:.2f}")

        news = {}
        for kind, target, strength, title, available in self.market.hotspots(day, settings.news_days):
            key = kind, target
            total, _ = news.get(key, (0.0, ""))
            news[key] = (total + strength, title)
        for (kind, target), (strength, title) in sorted(news.items(), key=lambda item: (-item[1][0], item[0])):
            evidence = f"{target} | {title} | 热度 {strength:g}"
            if kind == "code" and target in eligible.index:
                add(target, "news", evidence)
            if kind == "industry":
                for code in active_names[active_names["industry"] == target].head(settings.industry_members).index:
                    if "news" not in found.get(code, {}):
                        add(code, "news", evidence)

        unusual = eligible[eligible["volume_ratio"] >= settings.volume_ratio]
        unusual = unusual["volume_ratio"].sort_values(ascending=False, kind="stable")
        for code, ratio in unusual.head(settings.volume_count).items():
            add(code, "volume", f"成交量为过去 20 个有成交日中位数的 {ratio:.2f} 倍")
        for code in sorted(holdings):
            add(code, "holdings", "已有持仓")
        return found, industry


class Ranker:
    def __init__(self, market, features, strategy, models):
        self.features = features
        self.recall = Recall(market, strategy)
        self.strategy = strategy
        self.models = {model.model_id: model for model in models}
        if len(self.models) != len(models):
            raise ValueError("模型 ID 必须唯一。")
        missing = {key for key, value in strategy.weights.items() if value > 0} - self.models.keys()
        if missing:
            raise ValueError(f"缺少正权重模型：{', '.join(sorted(missing))}。")
        self.securities = market.securities()

    def rank(self, day, holdings, *, recalled=None):
        current = self.features.day(day)
        found, industries = recalled if recalled is not None else self.recall.candidates(day, current, holdings)
        columns = ["rank", "code", "name", "industry", "has_bar", "held", "score", "routes", "evidence"]
        if not found:
            return pd.DataFrame(columns=columns)
        frame = current.reindex(sorted(found)).rename_axis(None)
        frame["code"] = frame.index
        frame["name"] = [self.securities.get(code, (code,))[0] for code in frame.index]
        frame["industry"] = frame["code"].map(industries)
        frame["has_bar"] = frame["close"].notna()
        frame["held"] = frame["code"].isin(holdings)
        frame["routes"] = ["|".join(found[code]) for code in frame.index]
        frame["evidence"] = [" / ".join(found[code].values()) for code in frame.index]
        weighted = pd.Series(0.0, index=frame.index)
        available = weighted.copy()
        scored = frame[frame["has_bar"]].copy()
        for key, weight in self.strategy.weights.items():
            if not weight:
                continue
            raw = self.models[key].score(day, scored, self.features).reindex(frame.index)
            raw = raw.where(np.isfinite(raw))
            count = int(raw.count())
            percentile = (raw.rank(method="average") - 0.5) / count if count else raw
            raw_col = "expected_return" if key == "token_transformer" else f"{key}_raw"
            frame[raw_col] = raw
            frame[f"{key}_score"] = percentile * 100
            columns.extend([raw_col, f"{key}_score"])
            weighted += weight * frame[f"{key}_score"].fillna(0)
            available += weight * raw.notna()
        frame["weight_coverage"] = available / sum(self.strategy.weights.values())
        sufficient = frame["weight_coverage"] >= self.strategy.min_weight_coverage - 1e-12
        frame["score"] = weighted / available.where(sufficient)
        frame = frame.sort_values(["score", "weight_coverage", "code"], ascending=[False, False, True])
        frame["rank"] = pd.Series(np.arange(len(frame)) + 1, index=frame.index).where(frame["score"].notna()).astype("Int64")
        return frame[[*columns, "weight_coverage", "close", "volume", "volume_ratio", "market_cap"]].reset_index(drop=True)
