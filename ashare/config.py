"""User-facing ranking and simulation configuration."""

import json
import math
from dataclasses import asdict, dataclass, field

DEFAULT_WEIGHTS = {"trend": 0.35, "macd": 0.25, "token_transformer": 0.40}


@dataclass(frozen=True)
class Strategy:
    weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    industry_count: int = 4
    industry_members: int = 25
    industry_min_members: int = 5
    volume_count: int = 80
    volume_ratio: float = 1.8
    min_amount: float = 10_000_000
    news_days: int = 3
    min_weight_coverage: float = 0.60
    positions: int = 12
    rebalance_every: int = 1
    initial_cash: float = 1_000_000
    commission_bps: float = 3
    min_commission: float = 5
    sell_tax_bps: float = 5
    transfer_bps: float = 0.1
    slippage_bps: float = 10
    participation: float = 0.01

    def __post_init__(self):
        if not self.weights or any(not math.isfinite(w) or w < 0 for w in self.weights.values()):
            raise ValueError("模型权重必须是有限非负数。")
        if sum(self.weights.values()) <= 0:
            raise ValueError("至少需要一个正权重模型。")
        integers = (
            self.industry_count, self.industry_members, self.industry_min_members, self.volume_count,
            self.news_days, self.positions, self.rebalance_every,
        )
        if any(type(x) is not int or x < 1 for x in integers):
            raise ValueError("召回数量、持仓数、回看天数与调仓间隔必须是正整数。")
        money = (self.min_amount, self.initial_cash, self.min_commission)
        if any(not math.isfinite(x) or x < 0 for x in money) or not self.initial_cash:
            raise ValueError("成交额、资金和佣金配置无效。")
        bps = (self.commission_bps, self.sell_tax_bps, self.transfer_bps, self.slippage_bps)
        if any(not math.isfinite(x) or not 0 <= x < 10_000 for x in bps):
            raise ValueError("费率和滑点必须在 0–10000 bp 之间。")
        if not 0 < self.min_weight_coverage <= 1:
            raise ValueError("最小权重覆盖率必须在 (0, 1]。")
        if not math.isfinite(self.volume_ratio) or self.volume_ratio <= 0 or not 0 < self.participation <= 1:
            raise ValueError("放量阈值必须为正，参与率必须在 (0, 1]。")

    @classmethod
    def load(cls, path):
        if path is None:
            return cls()
        with open(path, encoding="utf-8") as handle:
            return cls(**json.load(handle))

    def as_dict(self):
        return asdict(self)
