"""Train and score incremental low-volatility signals with chronological splits."""

import json
import re
from contextlib import closing, nullcontext
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .config import Strategy
from .joint_token import rank_ic
from .rank import Recall
from .rotation import EmptyReference, RECALL_SIZE, ReplayFeatures, capped_recall
from .run_files import publish_directory
from .store import MarketSnapshot
from .temporal import sha256
from .typed_data import load_research_panel
from .web import open_readonly

WINDOW = 60
EXIT_LAG = 6
PRICE = ("vol20", "vol60", "down60", "drawdown60", "mom5", "mom20", "mom59", "slope_mean",
         "slope_std", "slope_quality", "slope_up", "log_size")
VALUE = ("bp", "ep", "dividend", "has_bp", "has_ep", "has_dividend")
MODELS = {"ridge": ("Ridge · 量价", PRICE), "ridge_value": ("Ridge · 量价＋估值", PRICE + VALUE),
          "mlp": ("MLP · 量价", PRICE), "mlp_value": ("MLP · 量价＋估值", PRICE + VALUE)}


def low_vol_features(prices, cap, pb, pe, dividend):
    returns = prices.pct_change(fill_method=None)
    prices60 = prices.rolling(WINDOW, min_periods=45)
    avg = prices.rolling(5).mean()
    slope = np.log(avg).diff()
    # A slope consumes six closes, so 55 slopes use at most 60 observations.
    slopes = slope.rolling(WINDOW - 5, min_periods=35)
    slope_mean, slope_std = slopes.mean(), slopes.std()

    def positive_inverse(values):
        values = pd.Series(values, index=prices.index)
        return 1 / values.where(values > 0)

    bp, ep = positive_inverse(pb), positive_inverse(pe)
    dividend = pd.Series(dividend, index=prices.index).where(lambda x: x >= 0)
    up = (slope > 0).where(slope.notna()).rolling(WINDOW - 5, min_periods=35).mean()
    return {
        "vol20": returns.rolling(20, min_periods=15).std(),
        "vol60": returns.rolling(WINDOW - 1, min_periods=44).std(),
        "down60": returns.clip(upper=0).pow(2).rolling(WINDOW - 1, min_periods=44).mean().pow(.5),
        "drawdown60": prices / prices60.max() - 1,
        "mom5": prices.pct_change(5, fill_method=None),
        "mom20": prices.pct_change(20, fill_method=None),
        "mom59": prices.pct_change(WINDOW - 1, fill_method=None),
        "slope_mean": slope_mean, "slope_std": slope_std,
        "slope_quality": slope_mean / slope_std.clip(lower=.0005), "slope_up": up,
        "log_size": np.log(pd.Series(cap, index=prices.index).where(lambda x: x > 0)),
        "bp": bp, "ep": ep, "dividend": dividend,
        "has_bp": bp.notna(), "has_ep": ep.notna(), "has_dividend": dividend.notna(),
    }


class LowVolFeatures(ReplayFeatures):
    def collect(self, sid, shape, dates, symbol, listed, raw, status, is_st):
        super().collect(sid, shape, dates, symbol, listed, raw[:, :8], status, is_st)
        quoted = np.isfinite(self.values["close"][sid])
        prices = pd.Series(np.where(quoted, raw[:, 1], np.nan))
        for name, values in low_vol_features(prices, raw[:, 6], *raw[:, 8:].T).items():
            if sid == 0:
                self.values[name] = np.empty(shape, dtype=np.float32)
            self.values[name][sid] = values

    def as_matrix(self, fields, stock, day):
        return np.column_stack([self.values[key][stock, day] for key in fields])


def tradable_label(features, stock, day):
    values = features.values
    begin, end = day + 1, day + EXIT_LAG
    enter, exit_ = values["open"][stock, begin], values["open"][stock, end]
    valid = (enter < values["limit_up"][stock, begin]) & (exit_ > values["limit_down"][stock, end])
    adjusted = values["adj"]
    return np.where(valid, exit_ * adjusted[stock, end] / (enter * adjusted[stock, begin]) - 1, np.nan)


def fit_transform(x, normalizer=None):
    if normalizer is None:
        quantiles = np.nanquantile(x, [.1, .5, .9], axis=0)
        low, center, high = np.nan_to_num(quantiles).astype(np.float32)
        scale = high - low
        normalizer = center, np.where(scale > 1e-6, scale, 1).astype(np.float32)
    center, scale = normalizer
    return np.nan_to_num(((x - center) / scale).clip(-4, 4)).astype(np.float32), normalizer


def mlp(width):
    return nn.Sequential(nn.Linear(width, 24), nn.GELU(), nn.Linear(24, 12), nn.GELU(), nn.Linear(12, 1))


def fit(name, x, target, validation=None, epochs=8):
    if name.startswith("ridge"):
        design = np.column_stack((x, np.ones(len(x), dtype=np.float32)))
        penalty = np.eye(design.shape[1]) * (len(x) * .003)
        penalty[-1, -1] = 0
        return np.linalg.solve(design.T @ design + penalty, design.T @ target).astype(np.float32), [], None
    torch.manual_seed(42)
    model = mlp(x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.003)
    inputs, truth = torch.from_numpy(x), torch.from_numpy(target)
    best, best_score, steps, chosen = None, -np.inf, [], epochs
    for epoch in range(1, epochs + 1):
        model.train()
        for idx in torch.randperm(len(x)).split(4096):
            loss = nn.functional.mse_loss(model(inputs[idx]).flatten(), truth[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if validation is not None:
            score = validation(lambda data: predict(model, data))
            steps.append(score)
            if score > best_score:
                best, best_score, chosen = {k: v.detach().clone() for k, v in model.state_dict().items()}, score, epoch
    if best is not None:
        model.load_state_dict(best)
    return model.eval(), steps, chosen


@torch.inference_mode()
def predict(model, x):
    if isinstance(model, np.ndarray):
        return x @ model[:-1] + model[-1]
    return model(torch.from_numpy(x)).flatten().numpy()


def candidate_samples(panel, features, market_path, start, split, end):
    reference = closing(open_readonly(market_path)) if market_path else nullcontext(None)
    arrays = [[], []]
    dates = panel.dates
    with reference as db:
        recall = Recall(MarketSnapshot(db) if db else EmptyReference(), Strategy(volume_count=RECALL_SIZE))
        for i in range(len(dates) - EXIT_LAG):
            day = str(dates[i])
            if day < start or dates[i + EXIT_LAG] > end:
                continue
            side = int(day > split)
            if side == 0 and dates[i + EXIT_LAG] > split:
                continue
            features.select(day, i, panel.symbols)
            current = features.day(day)
            found, industries = recall.candidates(day, current, {})
            liquidity = current[current["amount"] >= recall.settings.min_amount].nlargest(RECALL_SIZE, "amount").index
            candidates, _ = capped_recall(found, industries, liquidity, {})
            stock = np.flatnonzero(current.index.isin(candidates))
            data = features.as_matrix(PRICE + VALUE, stock, i)
            returns = tradable_label(features, stock, i)
            base = features.values["low_volatility"][stock, i]
            valid = np.isfinite(returns) & np.isfinite(base) & np.isfinite(data[:, [1, 8]]).all(axis=1)
            if valid.sum() < 10:
                continue
            ranks = pd.Series(returns[valid]).rank(pct=True).to_numpy(dtype=np.float32) - .5
            arrays[side].append((data[valid], ranks, base[valid]))
    if not all(arrays):
        raise ValueError("低波训练或验证期没有可成交的样本。")
    return [tuple(map(np.concatenate, zip(*part))) for part in arrays], [list(map(lambda row: len(row[1]), p)) for p in arrays]


def level_ic(pred, target, coverage, base=None):
    values = []
    start = 0
    for n in coverage:
        end = start + n
        raw = pred[start:end]
        if base is not None:
            ranking = pd.DataFrame({"model": raw, "base": base[start:end]}).rank(pct=True)
            raw = (.35 * ranking.model + .65 * ranking.base).to_numpy()
        score = rank_ic(raw, target[start:end])
        if score is not None:
            values.append(score)
        start = end
    return float(np.mean(values)) if values else 0.


def train_low_vol(args, progress):
    if args.out.exists() or not args.start < args.split < args.end:
        raise ValueError("训练时间切分无效，或输出目录已存在。")
    market = args.db if args.db.is_file() else None
    features = LowVolFeatures()
    panel = load_research_panel(args.qlib_db, market, args.status_csv, args.start, args.end,
                                WINDOW, progress, on_symbol=features.collect, include_valuation=True)
    (train, valid), (train_days, val_days) = candidate_samples(panel, features, market, args.start, args.split, args.end)
    progress(f"低波训练 {len(train[1]):,} 条，验证 {len(valid[1]):,} 条。")
    output = {"start": args.start, "selection_train_end": args.split, "validated_through": args.end,
              "train_samples": len(train[1]), "validation_samples": len(valid[1]),
              "train_sessions": len(train_days), "validation_sessions": len(val_days),
              "window": WINDOW, "entry_lag": 1, "exit_lag": EXIT_LAG,
              "source": panel.source, "baseline_rank_ic": level_ic(valid[2], valid[1], val_days), "models": {}}

    def writer(folder):
        for key, (label, fields) in MODELS.items():
            width = len(fields)
            train_x, normalizer = fit_transform(train[0][:, :width])
            val_x, _ = fit_transform(valid[0][:, :width], normalizer)

            def ic(fn):
                return level_ic(fn(val_x), valid[1], val_days, valid[2])

            model, history, epochs = fit(key, train_x, train[1], validation=ic)
            predicted = predict(model, val_x)
            metrics = {"label": label, "features": fields, "epochs": epochs, "validation_curve": history,
                       "model_rank_ic": level_ic(predicted, valid[1], val_days),
                       "blend_rank_ic": level_ic(predicted, valid[1], val_days, valid[2])}
            progress(f"{label} 验证混合 Rank IC {metrics['blend_rank_ic']:+.4f}。")
            del train_x, val_x
            all_x, normalizer = fit_transform(np.vstack((train[0][:, :width], valid[0][:, :width])))
            final, _, _ = fit(key, all_x, np.concatenate((train[1], valid[1])), epochs=epochs or 1)
            path = folder / f"{key}.pt"
            weights = torch.from_numpy(final) if isinstance(final, np.ndarray) else final.state_dict()
            torch.save({"weights": weights, "center": torch.from_numpy(normalizer[0]),
                        "scale": torch.from_numpy(normalizer[1])}, path)
            output["models"][key] = {**metrics, "sha256": sha256(path)}
        (folder / "model.json").write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False),
                                          encoding="utf-8")

    publish_directory(args.out, writer)
    return output


class LowVolScores:
    def __init__(self, folder, panel, first_day, *, allow_newer_release=False):
        folder = Path(folder)
        self.metadata = json.loads((folder / "model.json").read_text(encoding="utf-8"))
        trained_source = self.metadata["source"]
        newer = (
            allow_newer_release
            and panel.source["source"] == trained_source["source"]
            and date.fromisoformat(panel.source["release"]) > date.fromisoformat(trained_source["release"])
            and all(re.fullmatch(r"[0-9a-f]{64}", panel.source["sha256"].get(key, ""))
                    for key in ("market", "daily_basic"))
        )
        if first_day <= self.metadata["validated_through"] or (panel.source != trained_source and not newer):
            raise ValueError("回放起始日须晚于低波模型所用数据，且行情来源一致。")
        self.models = {}
        for key, (_, fields) in MODELS.items():
            path = folder / f"{key}.pt"
            meta = self.metadata["models"][key]
            if list(fields) != meta["features"] or sha256(path) != meta["sha256"]:
                raise ValueError("低波模型校验失败。")
            saved = torch.load(path, map_location="cpu", weights_only=True)
            model = saved["weights"].numpy() if key.startswith("ridge") else mlp(len(fields)).eval()
            if not key.startswith("ridge"):
                model.load_state_dict(saved["weights"])
            self.models[key] = (model, fields, (saved["center"].numpy(), saved["scale"].numpy()))

    def add(self, frame, index, candidates):
        eligible = frame.index.isin(candidates) & frame["vol60"].notna() & frame["slope_std"].notna()
        frame["steady_rule"] = (-.50 * frame["vol60"] / .02 + .30 * frame["slope_quality"].clip(-3, 3)
                                + .20 * frame["mom59"] / .15)
        for key, (model, fields, normalizer) in self.models.items():
            x, _ = fit_transform(frame.loc[eligible, list(fields)].to_numpy(), normalizer)
            frame.loc[eligible, key] = predict(model, x)


def low_vol_portfolios():
    return {"main": ("原版 20 日低波", {"low_volatility": 1.0}),
            **{key: (label, {"low_volatility": .65, key: .35})
               for key, label in (("steady_rule", "低波＋平稳上涨规则"),
                                  *((key, value[0]) for key, value in MODELS.items()))}}
