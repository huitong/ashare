"""Causal Transformer for next-session return tokens."""

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .features import TOKEN_FEATURES

MODEL_ID = "token_transformer"
FORMAT_VERSION = 1
OFFSETS = np.array([0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95])


@dataclass
class Tokenizer:
    edges: list
    return_values: list
    prior: list

    @classmethod
    def fit(cls, frame, first_train_day, last_train_day, train_targets):
        mask = frame["day"].between(first_train_day, last_train_day)
        fit = frame.loc[mask, TOKEN_FEATURES]
        edges = []
        for col in TOKEN_FEATURES:
            finite = fit[col].dropna().to_numpy()
            if not finite.size:
                raise ValueError(f"训练集缺少特征 {col}。")
            edges.append(np.unique(np.quantile(finite, OFFSETS)).tolist())
        target = frame["return_1"].to_numpy()[train_targets]
        tokens = np.searchsorted(edges[0], target, side="right")
        values, counts = [], []
        for index in range(len(edges[0]) + 1):
            values_in_bin = target[tokens == index]
            counts.append(len(values_in_bin))
            values.append(float(values_in_bin.mean()) if len(values_in_bin) else float(np.median(target)))
        prior = (np.asarray(counts) + 0.5) / (len(tokens) + 0.5 * len(counts))
        return cls(edges, values, prior.tolist())

    def encode(self, frame):
        result = []
        for name, edges in zip(TOKEN_FEATURES, self.edges):
            values = frame[name].to_numpy()
            codes = np.searchsorted(edges, values, side="right") + 1
            codes[~np.isfinite(values)] = 0
            result.append(codes)
        return np.stack(result, axis=1).astype(np.int16)

    @property
    def vocab_sizes(self):
        return [len(edges) + 2 for edges in self.edges]


class CausalTransformer(nn.Module):
    def __init__(self, sizes, context, width, heads, layers):
        super().__init__()
        self.embeddings = nn.ModuleList(nn.Embedding(size, width) for size in sizes)
        self.position = nn.Embedding(context, width)
        layer = nn.TransformerEncoderLayer(
            width, heads, width * 3, dropout=0.1, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(width), enable_nested_tensor=False)
        self.output = nn.Linear(width, sizes[0] - 1)
        self.register_buffer("causal_mask", torch.ones(context, context, dtype=torch.bool).triu(1))

    def forward(self, tokens):
        steps = tokens.size(1)
        emb = torch.stack([layer(tokens[:, :, i]) for i, layer in enumerate(self.embeddings)])
        states = emb.sum(dim=0) / math.sqrt(len(self.embeddings))
        states = states + self.position(torch.arange(steps, device=tokens.device))
        states = self.encoder(states, mask=self.causal_mask[:steps, :steps], is_causal=True)
        return self.output(states[:, -1, :])


class Windows(Dataset):
    def __init__(self, tokens, targets, context):
        self.tokens = tokens
        self.targets = targets
        self.context = context

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        target = self.targets[index]
        return self.tokens[target - self.context:target].astype(np.int64), int(self.tokens[target, 0] - 1)


def get_targets(features, context):
    frame = features.frame
    eligible = features.window_ends(context) + 1
    next_rows = eligible[eligible < len(frame)]
    codes = frame["code"].to_numpy()
    ids = frame["session_id"].to_numpy()
    return next_rows[(codes[next_rows] == codes[next_rows - 1]) & (ids[next_rows] == ids[next_rows - 1] + 1)]


def partition(features, start, end, context, train_end, val_end):
    days = [day for day in features.sessions if start <= day <= end]
    if len(days) < 12:
        raise ValueError("训练区间至少需要 12 个交易日，另需上下文预热数据。")
    train_end = train_end or days[int(len(days) * 0.65) - 1]
    val_end = val_end or days[int(len(days) * 0.80) - 1]
    if train_end not in days or val_end not in days or not start <= train_end < val_end < end:
        raise ValueError("训练和验证截止日须是区间内的交易日，并依次早于测试区间。")
    targets = get_targets(features, context)
    target_dates = features.frame["day"].to_numpy()[targets]
    groups = [
        targets[(target_dates >= start) & (target_dates <= train_end)],
        targets[(target_dates > train_end) & (target_dates <= val_end)],
        targets[(target_dates > val_end) & (target_dates <= end)],
    ]
    if any(len(group) == 0 for group in groups):
        raise ValueError("训练、验证或测试分段无连续窗口。请检查数据覆盖、预热与上下文长度。")
    return groups, train_end, val_end


@torch.inference_mode()
def probabilities(net, loader, device):
    net.eval()
    values = []
    for x, _ in loader:
        values.append(net(x.to(device)).softmax(dim=1).cpu().numpy())
    return np.concatenate(values)


def evaluate(predicted, tokens, targets, frame, tokenizer):
    labels = tokens[targets, 0] - 1
    reps = np.asarray(tokenizer.return_values)
    prior = np.asarray(tokenizer.prior)
    realized = frame["return_1"].to_numpy()[targets]
    expected = np.sum(predicted * reps, axis=1)
    table = pd.DataFrame({
        "day": frame["day"].to_numpy()[targets], "prediction": expected, "realized": realized,
    })
    grouped = table.groupby("day", sort=False)
    correlations = []
    for _, group in grouped:
        if len(group) >= 5 and group["prediction"].nunique() > 1 and group["realized"].nunique() > 1:
            ranked = group[["prediction", "realized"]].rank()
            correlations.append(float(ranked["prediction"].corr(ranked["realized"])))
    return {
        "samples": len(labels),
        "sessions": table["day"].nunique(),
        "nll": float(-np.log(predicted[np.arange(len(labels)), labels].clip(1e-12)).mean()),
        "prior_nll": float(-np.log(prior[labels]).mean()),
        "accuracy": float((predicted.argmax(axis=1) == labels).mean()),
        "prior_accuracy": float((labels == prior.argmax()).mean()),
        "expected_return_mae": float(np.abs(expected - realized).mean()),
        "prior_mae": float(np.abs(np.sum(prior * reps) - realized).mean()),
        "mean_daily_rank_ic": float(np.mean(correlations)) if correlations else None,
        "rank_ic_sessions": len(correlations),
    }


def sha256(path):
    with path.open("rb") as handle:
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
        return digest.hexdigest()


def train(features, source, start, end, output, *, train_end=None, val_end=None,
          context=20, width=32, heads=4, layers=2, epochs=5, batch=256,
          seed=42, device="cpu", progress=print):
    if min(context, width, heads, layers, epochs, batch) < 1 or width % heads:
        raise ValueError("模型参数须为正整数，宽度须能被注意力头数整除。")
    if output.exists():
        raise ValueError(f"模型目录已存在：{output}。请指定新目录保留版本。")
    torch.manual_seed(seed)
    groups, train_end, val_end = partition(features, start, end, context, train_end, val_end)
    train_targets, val_targets, test_targets = groups
    tokenizer = Tokenizer.fit(features.frame, start, train_end, train_targets)
    tokens = tokenizer.encode(features.frame)
    architecture = dict(context=context, width=width, heads=heads, layers=layers)
    net = CausalTransformer(tokenizer.vocab_sizes, **architecture).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)
    generator = torch.Generator().manual_seed(seed)
    train_loader, val_loader, test_loader = [
        DataLoader(Windows(tokens, group, context), batch_size=batch, shuffle=(i == 0),
                   generator=generator if i == 0 else None)
        for i, group in enumerate(groups)
    ]
    history = []
    best_state, best_loss = None, float("inf")
    for epoch in range(1, epochs + 1):
        net.train()
        loss_total = torch.zeros((), device=device)
        for x, labels in train_loader:
            x, labels = x.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(net(x), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            loss_total += loss.detach() * len(x)
        val_probs = probabilities(net, val_loader, device)
        val_labels = tokens[val_targets, 0] - 1
        val_loss = float(-np.log(val_probs[np.arange(len(val_labels)), val_labels].clip(1e-12)).mean())
        train_loss = float(loss_total.cpu()) / len(train_targets)
        if not math.isfinite(train_loss) or not math.isfinite(val_loss):
            raise ValueError("模型训练损失非有限数。")
        history.append(dict(epoch=epoch, train_nll=train_loss, validation_nll=val_loss))
        progress(f"Epoch {epoch}/{epochs}  train NLL={train_loss:.5f}  val NLL={val_loss:.5f}")
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {name: value.detach().cpu().clone() for name, value in net.state_dict().items()}
    net.load_state_dict(best_state)
    metrics = {
        "train_samples": len(train_targets),
        "best_epoch": best_epoch,
        "epochs": history,
        "validation": evaluate(probabilities(net, val_loader, device), tokens, val_targets, features.frame, tokenizer),
        "test": evaluate(probabilities(net, test_loader, device), tokens, test_targets, features.frame, tokenizer),
    }
    labels = features.frame["day"].to_numpy()[train_targets]
    metadata = dict(
        first_train_label=str(labels.min()), last_train_label=str(labels.max()),
        format=FORMAT_VERSION, model_id=MODEL_ID, source=source, training_start=start, trained_through=train_end,
        validated_through=val_end, tested_through=end, feature_order=TOKEN_FEATURES, seed=seed,
        architecture=architecture, tokenizer=tokenizer.__dict__, metrics=metrics,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".model-", dir=output.parent))
    try:
        weights = staging / "weights.pt"
        torch.save(best_state, weights)
        metadata["weights_sha256"] = sha256(weights)
        (staging / "model.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return metadata


class TemporalScorer:
    model_id = MODEL_ID

    def __init__(self, path, source, device="cpu"):
        self.path = path
        self.metadata = json.loads((path / "model.json").read_text(encoding="utf-8"))
        meta = self.metadata
        if (meta["format"] != FORMAT_VERSION or meta["model_id"] != MODEL_ID
                or meta["feature_order"] != list(TOKEN_FEATURES)):
            raise ValueError("模型文件版本或特征顺序不兼容。")
        if meta["source"] != source:
            raise ValueError(f"模型数据源 {meta['source']} 与行情库 {source} 不一致。")
        if sha256(path / "weights.pt") != meta["weights_sha256"]:
            raise ValueError("模型权重校验失败。")
        self.tokenizer = Tokenizer(**meta["tokenizer"])
        self.context = meta["architecture"]["context"]
        self.net = CausalTransformer(self.tokenizer.vocab_sizes, **meta["architecture"]).to(device)
        self.net.load_state_dict(torch.load(path / "weights.pt", map_location=device, weights_only=True))
        self.net.eval()
        self.device = device
        self.features = None

    def require_out_of_sample(self, day):
        if day < self.metadata["validated_through"]:
            raise ValueError("第一个信号日在模型验证截止日前，存在未来数据泄漏。")

    @torch.inference_mode()
    def distribution(self, day, candidates, features):
        self.require_out_of_sample(day)
        if features is not self.features:
            self.features = features
            self.encoded = self.tokenizer.encode(features.frame)
            self.valid = set(features.window_ends(self.context).tolist())
        indices = features.by_day[day] if day in features.by_day else []
        stocks = features.frame.iloc[indices]["code"]
        chosen = stocks[stocks.isin(candidates.index) & stocks.index.isin(self.valid)]
        if chosen.empty:
            return pd.DataFrame(index=pd.Index([], name="code"), columns=range(len(self.tokenizer.return_values)))
        results = []
        ends = chosen.index.to_numpy()
        for offset in range(0, len(ends), 256):
            batch = ends[offset:offset + 256]
            windows = batch[:, None] + np.arange(1 - self.context, 1)
            x = torch.from_numpy(self.encoded[windows].astype(np.int64)).to(self.device)
            results.append(self.net(x).softmax(dim=1).cpu().numpy())
        return pd.DataFrame(np.concatenate(results), index=pd.Index(chosen.to_numpy(), name="code"))

    def score(self, day, candidates, features):
        probs = self.distribution(day, candidates, features)
        if probs.empty:
            return pd.Series(index=candidates.index, dtype=float)
        return pd.Series(np.sum(probs.to_numpy() * self.tokenizer.return_values, axis=1), index=probs.index)
