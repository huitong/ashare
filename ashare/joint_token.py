"""Joint market-cap, relative-volume and return tokens for Qlib daily data."""

import json
import math
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from .run_files import publish_directory
from .temporal import sha256

BINS = 10
VOCABULARY = BINS ** 3
FEATURES = ("log_market_cap_cny", "log_volume_ratio_20", "return_1_adj")
VOLUME_WINDOW = 20
VOLUME_MIN = 15


@dataclass
class Panel:
    dates: np.ndarray
    symbols: list
    features: np.ndarray
    source: dict


def daily_features(close, normalized_volume, factor, cap):
    close, normalized_volume, factor, cap = map(lambda x: np.asarray(x, dtype=np.float64),
                                                  (close, normalized_volume, factor, cap))
    volume = normalized_volume * factor
    trading = np.isfinite(close) & np.isfinite(volume) & (close > 0) & (volume > 0) & (factor > 0)
    trailing = (pd.Series(volume).where(trading).shift(1)
                .rolling(VOLUME_WINDOW, min_periods=VOLUME_MIN).median().to_numpy())
    result = np.full((len(close), len(FEATURES)), np.nan, dtype=np.float32)
    valid = np.flatnonzero(trading[1:] & trading[:-1]) + 1
    valid = valid[np.isfinite(trailing[valid]) & (trailing[valid] > 0) & np.isfinite(cap[valid]) & (cap[valid] > 0)]
    result[valid, 0] = np.log10(cap[valid] * 10000)
    result[valid, 1] = np.log(volume[valid] / trailing[valid])
    result[valid, 2] = close[valid] / close[valid - 1] - 1
    return result


def load_panel(path, start, end, context, progress):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"找不到 Qlib 数据库：{path}")
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("BEGIN")
        info = json.loads(db.execute("SELECT value FROM qlib_meta WHERE key='summary'").fetchone()[0])
        days = [row[0] for row in db.execute("SELECT day FROM qlib_calendar WHERE day<=? ORDER BY day", (end,))]
        first = next((i for i, day in enumerate(days) if day >= start), None)
        if first is None:
            raise ValueError("指定范围内没有 Qlib 交易日。")
        days = days[max(0, first - context - VOLUME_WINDOW - 1):]
        position = {day: i for i, day in enumerate(days)}
        symbols = [row[0] for row in db.execute(
            """SELECT symbol FROM qlib_symbols WHERE price_rows>0 AND first_day<=? AND last_day>=?
               AND (symbol GLOB 'sh[0-9]*' OR symbol GLOB 'sz[0-9]*' OR symbol GLOB 'bj[0-9]*')
               ORDER BY symbol""", (days[-1], days[0])
        )]
        data = np.full((len(symbols), len(days), len(FEATURES)), np.nan, dtype=np.float32)
        for i, symbol in enumerate(symbols):
            raw = np.full((len(days), 4), np.nan, dtype=np.float64)
            for day, *values in db.execute(
                """SELECT day,close,volume,factor,total_mv FROM qlib_daily
                   WHERE symbol=? AND day BETWEEN ? AND ? ORDER BY day""", (symbol, days[0], days[-1])
            ):
                raw[position[day]] = values
            data[i] = daily_features(*raw.T)
            if (i + 1) % 1500 == 0:
                progress(f"已提取 {i+1:,}/{len(symbols):,} 个证券的时序特征。")
    return Panel(np.asarray(days), symbols, data, {k: info[k] for k in ("source", "release", "sha256") if k in info})


@dataclass
class JointTokenizer:
    edges: list
    return_values: list

    @classmethod
    def fit(cls, training):
        valid = training[np.isfinite(training).all(axis=-1)].astype(np.float64)
        if len(valid) < 1000:
            raise ValueError("拟合分桶至少需要 1,000 条完整训练记录。")
        edges = np.quantile(valid, np.arange(1, BINS) / BINS, axis=0).T
        if (np.diff(edges, axis=1) <= 0).any():
            raise ValueError("训练特征的相邻分位点重合，无法使用十等分。")
        realized = valid[:, 2]
        bins = np.searchsorted(edges[2], realized, side="right")
        reference = [float(realized[bins == i].mean()) for i in range(BINS)]
        return cls(edges.tolist(), reference)

    def encode(self, values):
        parts = [np.searchsorted(edges, values[..., i], side="right").astype(np.int16)
                 for i, edges in enumerate(self.edges)]
        tokens = (parts[0] * BINS + parts[1]) * BINS + parts[2]
        tokens[~np.isfinite(values).all(axis=-1)] = -1
        return tokens


class JointTransformer(nn.Module):
    def __init__(self, context, width, heads, layers=5, markov_residual=False):
        super().__init__()
        self.joint = nn.Embedding(VOCABULARY, width)
        self.size = nn.Embedding(BINS, width)
        self.volume = nn.Embedding(BINS, width)
        self.change = nn.Embedding(BINS, width)
        self.positions = nn.Embedding(context, width)
        self.blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(width, heads, 2 * width, dropout=0.1, activation="gelu",
                                       batch_first=True, norm_first=True),
            layers, norm=nn.LayerNorm(width), enable_nested_tensor=False,
        )
        self.output = nn.Linear(width, VOCABULARY)
        self.register_buffer("mask", torch.ones(context, context, dtype=torch.bool).triu(1))
        self.register_buffer("log_transition_prior",
                             torch.zeros(VOCABULARY, VOCABULARY) if markov_residual else None)
        if markov_residual:
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)


    def forward(self, token, tail=1):
        size = token // (BINS ** 2)
        volume = (token // BINS) % BINS
        change = token % BINS
        embedding = (self.joint(token) + self.size(size) + self.volume(volume) + self.change(change)) / 2
        length = token.size(1)
        embedding = embedding + self.positions(torch.arange(length, device=token.device))
        hidden = self.blocks(embedding, mask=self.mask[:length, :length], is_causal=True)
        logits = self.output(hidden[:, -tail:])
        if self.log_transition_prior is not None:
            logits = logits + self.log_transition_prior[token[:, -tail:]]
        return logits


def eligible_targets(tokens, context):
    valid = np.zeros(tokens.shape, dtype=bool)
    counts = np.pad(np.cumsum(tokens >= 0, axis=1, dtype=np.int32), ((0, 0), (1, 0)))
    valid[:, context:] = (counts[:, context + 1:] - counts[:, :-(context + 1)]) == context + 1
    return valid


def sample_days(days, limit):
    if not limit or len(days) <= limit:
        return days
    indices = (np.arange(limit) * len(days) // limit + (len(days) // limit) // 2)
    return days[indices]


def training_blocks(available, days, count, rng, context):
    """Space apart the sampled label blocks within each security."""
    by_symbol, label = np.nonzero(available[:, days])
    label = days[label]
    if len(label) == 0:
        raise ValueError("训练期没有连续的完整上下文。")
    phase = rng.integers(context, size=available.shape[0])
    spaced = (label % context) == phase[by_symbol]
    by_symbol, label = by_symbol[spaced], label[spaced]
    if not len(label):
        raise ValueError("训练期没有可抽取的非重叠块。")
    indices = rng.choice(len(label), min(count, len(label)), replace=False)
    return by_symbol[indices], label[indices]


def baseline(tokens, days):
    previous, target = tokens[:, days - 1].ravel(), tokens[:, days].ravel()
    keep = (previous >= 0) & (target >= 0)
    counts = np.bincount(previous[keep].astype(np.int64) * VOCABULARY + target[keep],
                        minlength=VOCABULARY ** 2).reshape(VOCABULARY, VOCABULARY).astype(np.float64)
    prior = (counts.sum(axis=0) + 0.5) / (counts.sum() + 0.5 * VOCABULARY)
    transitions = counts + 10 * prior[None, :]
    transitions /= transitions.sum(axis=1, keepdims=True)
    return prior.astype(np.float32), transitions.astype(np.float32)


def tensors(tokens, sid, tid, context, device):
    offsets = np.arange(-context, 0)
    history = tokens[sid[:, None], tid[:, None] + offsets].astype(np.int64)
    return torch.from_numpy(history).to(device)


def distribution_scores(probabilities, labels, representatives):
    probabilities = probabilities.astype(np.float64, copy=False)
    by_return = probabilities.reshape(-1, BINS * BINS, BINS).sum(axis=1)
    indices = np.arange(len(labels))
    joint_loss = -np.log(probabilities[indices, labels].clip(1e-12))
    return_loss = -np.log(by_return[indices, labels % BINS].clip(1e-12))
    score = np.sum(by_return * representatives, axis=1)
    return joint_loss, return_loss, probabilities.argmax(axis=1) == labels, score


def rank_ic(predicted, realized):
    if len(realized) < 5 or np.ptp(predicted) == 0 or np.ptp(realized) == 0:
        return None
    ordered = pd.DataFrame({"predicted": predicted, "realized": realized}).rank()
    return float(ordered["predicted"].corr(ordered["realized"]))


@torch.inference_mode()
def evaluate(net, panel, tokens, eligible, days, tokenizer, priors, context, batch, device):
    net.eval()
    names = ("model", "markov", "unconditional")
    counters = {key: np.zeros(4) for key in names}
    all_ic = {key: [] for key in names}
    per_day = []
    n = 0
    representatives = np.asarray(tokenizer.return_values)
    for day in days:
        sid = np.flatnonzero(eligible[:, day])
        if len(sid) == 0:
            continue
        actual = panel.features[sid, day, 2]
        labels = tokens[sid, day]
        result = {key: [] for key in names}
        for offset in range(0, len(sid), batch):
            current = sid[offset:offset + batch]
            histories = tensors(tokens, current, np.full(len(current), day), context, device)
            probabilities = net(histories)[:, 0].softmax(dim=-1).cpu().numpy()
            markov = priors[1][tokens[current, day - 1]]
            neutral = np.broadcast_to(priors[0], probabilities.shape)
            selected = slice(offset, offset + len(current))
            for key, p in zip(names, (probabilities, markov, neutral)):
                joint, marginal, correct, score = distribution_scores(p, labels[selected], representatives)
                counters[key] += (joint.sum(), marginal.sum(), correct.sum(), np.abs(score - actual[selected]).sum())
                result[key].append(score)
        metrics = {}
        for key in names:
            value = rank_ic(np.concatenate(result[key]), actual)
            if value is not None:
                all_ic[key].append(value)
            metrics[key + "_rank_ic"] = value
        n += len(sid)
        per_day.append({"date": str(panel.dates[day]), "samples": len(sid), **metrics})
    if not n:
        raise ValueError("评估期没有可用样本。")
    return {
        "samples": n, "sessions": len(per_day), "daily": per_day,
        **{name: {**dict(zip(("joint_nll", "return_nll", "joint_accuracy", "return_mae"),
                             (counters[name] / n).tolist())),
                  "mean_daily_rank_ic": float(np.mean(all_ic[name])) if all_ic[name] else None}
           for name in names},
    }


def train_joint(panel, output, *, start, train_end, val_end, end, context, width, heads, epochs,
                batch, train_blocks, val_days, test_days, seed, device, progress,
                markov_residual=False, exclude_test=()):
    if min(context, width, heads, epochs, batch, train_blocks) < 1 or width % heads or context < 4:
        raise ValueError("参数须为正整数，上下文至少 4，宽度须能被注意力头数整除。")
    if min(val_days, test_days) < 0 or not start <= train_end < val_end < end:
        raise ValueError("时间切分或评估日数量不合法。")
    dates = panel.dates
    split = [np.flatnonzero(mask) for mask in (
        (dates >= start) & (dates <= train_end),
        (dates > train_end) & (dates <= val_end),
        (dates > val_end) & (dates <= end),
    )]
    if any(len(group) == 0 for group in split):
        raise ValueError("有样本分段缺少交易日。")
    tokenizer = JointTokenizer.fit(panel.features[:, split[0], :])
    tokens = tokenizer.encode(panel.features)
    eligible = eligible_targets(tokens, context)
    counts = [int(eligible[:, group].sum()) for group in split]
    train_observed = tokens[:, split[0]]
    histogram = np.bincount(train_observed[train_observed >= 0], minlength=VOCABULARY)
    prior = baseline(tokens, split[0][(split[0] > 0) & (dates[split[0]] > dates[split[0][0]])])
    valid_dates = eligible.any(axis=0)
    validation = sample_days(split[1][valid_dates[split[1]]], val_days)
    untested = split[2][valid_dates[split[2]] & ~np.isin(dates[split[2]], exclude_test)]
    test = sample_days(untested, test_days)
    if len(validation) == 0 or len(test) == 0:
        raise ValueError("排除未覆盖或已评估日期后，没有可用的验证日或测试日。")
    evaluation_batch = max(batch, 256)
    architecture = dict(context=context, width=width, heads=heads, layers=5,
                        markov_residual=markov_residual)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = JointTransformer(**architecture).to(device)
    if markov_residual:
        net.log_transition_prior.copy_(torch.from_numpy(np.log(prior[1])))
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.0003 if markov_residual else 0.001)
    tail = (context + 1) // 2
    history = []
    best_state = None
    best_value = math.inf
    seen = np.zeros((len(panel.symbols), len(dates)), dtype=bool)
    progress(f"词表 {VOCABULARY}，训练实际出现 {np.count_nonzero(histogram)} 类；完整窗口 {counts}。")
    for epoch in range(1, epochs + 1):
        net.train()
        sid, tid = training_blocks(eligible, split[0][split[0] >= split[0][0] + tail - 1], train_blocks, rng, context)
        y_offsets = np.arange(1 - tail, 1)
        seen[sid[:, None], tid[:, None] + y_offsets] = True
        loss_sum = torch.zeros((), device=device)
        for offset in range(0, len(sid), batch):
            take = slice(offset, offset + batch)
            x = tensors(tokens, sid[take], tid[take], context, device)
            y = torch.from_numpy(tokens[sid[take, None], tid[take, None] + y_offsets].astype(np.int64)).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = net(x, tail=tail).reshape(-1, VOCABULARY)
            y = y.flatten()
            ret = logits.reshape(-1, BINS * BINS, BINS).logsumexp(dim=1)
            loss = F.cross_entropy(logits, y) + 0.25 * F.cross_entropy(ret, y % BINS)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.detach() * y.numel()
        train_loss = float(loss_sum.cpu()) / (len(sid) * tail)
        if not math.isfinite(train_loss):
            raise ValueError("训练损失出现非有限值。")
        progress(f"第 {epoch} 轮已训练 {len(sid) * tail:,} 个 next-token 标签，正在验证。")
        val = evaluate(net, panel, tokens, eligible, validation, tokenizer, prior, context, evaluation_batch, device)
        loss_value = val["model"]["return_nll"]
        history.append({"epoch": epoch, "train_objective": train_loss, "training_labels": len(sid) * tail,
                        "validation_joint_nll": val["model"]["joint_nll"], "validation_return_nll": loss_value})
        progress(f"第 {epoch} 轮：训练损失 {train_loss:.4f}，验证收益桶 NLL {loss_value:.4f}。")
        if loss_value < best_value:
            best_value = loss_value
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
            best_val = val
    net.load_state_dict(best_state)
    progress("验证集已选定权重，正在评估独立测试期。")
    holdout = evaluate(net, panel, tokens, eligible, test, tokenizer, prior, context, evaluation_batch, device)
    metadata = {
        "format": 1, "model_id": "qlib_joint_token", "source": panel.source, "seed": seed,
        "vocabulary": VOCABULARY, "seen_training_tokens": int(np.count_nonzero(histogram)), "features": FEATURES,
        "feature_spec": {"volume_window": VOLUME_WINDOW, "volume_min": VOLUME_MIN, "bins_per_feature": BINS,
                         "auxiliary_return_loss": 0.25, "training_label_tail": tail},
        "architecture": architecture, "parameters": sum(p.numel() for p in net.parameters()),
        "splits": {key: dict(start=str(dates[group[0]]), end=str(dates[group[-1]]), sessions=len(group),
                             eligible_windows=count)
                   for key, group, count in zip(("train", "validation", "test"), split, counts)},
        "test_day_sampling": "calendar-spaced eligible dates excluding prior evaluations, all eligible symbols per date",
        "excluded_test_dates": list(exclude_test),
        "training_setting": {"epochs": epochs, "batch": batch, "blocks_per_epoch": train_blocks,
                             "validation_day_limit": val_days, "test_day_limit": test_days, "device": device},
        "unique_training_labels": int(seen.sum()), "best_epoch": best_epoch, "tokenizer": asdict(tokenizer),
        "epochs": history, "validation": best_val, "test": holdout,
    }

    def writer(folder):
        torch.save(best_state, folder / "weights.pt")
        metadata["weights_sha256"] = sha256(folder / "weights.pt")
        (folder / "model.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        pd.DataFrame(holdout["daily"]).to_csv(folder / "test_rank_ic.csv", index=False)

    publish_directory(output, writer)
    return metadata

def load_joint_checkpoint(folder, device="cpu"):
    folder = Path(folder)
    metadata = json.loads((folder / "model.json").read_text(encoding="utf-8"))
    if metadata.get("format") != 1 or metadata.get("model_id") != "qlib_joint_token":
        raise ValueError("模型文件版本不兼容。")
    weights = folder / "weights.pt"
    if sha256(weights) != metadata["weights_sha256"]:
        raise ValueError("模型权重完整性校验失败。")
    model = JointTransformer(**metadata["architecture"]).to(device)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.eval()
    return model, JointTokenizer(**metadata["tokenizer"]), metadata
