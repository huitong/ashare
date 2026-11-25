"""5,000-token, six-layer Qlib model with execution-aware cross-sectional training."""

import copy
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from .joint_token import BINS, VOCABULARY as CORE, JointTokenizer, eligible_targets, rank_ic, sample_days
from .temporal import sha256
from .training_log import write_json
from .typed_data import AUX_FIELDS, FOOTPRINT, TYPE_LABELS, load_research_panel

VOCABULARY = len(TYPE_LABELS) * CORE
SCALING = 0.02


def lookup(keys, values, queries):
    index = np.searchsorted(keys, queries)
    keep = index < len(keys)
    if len(keys):
        keep &= keys[index.clip(0, len(keys) - 1)] == queries
    out = np.zeros(queries.shape, dtype=values.dtype)
    out[keep] = values[index[keep]]
    return out


class TransitionOrder:
    def __init__(self, history, labels, tradable, strength):
        pairs, weights = np.unique(history * VOCABULARY + labels, return_counts=True)
        contexts, targets = pairs // VOCABULARY, pairs % VOCABULARY
        self.keys, row, group = np.unique(contexts, return_index=True, return_inverse=True)
        self.total = np.add.reduceat(weights, row)
        self.pairs, self.counts = pairs, weights
        self.bins = np.zeros((len(row), BINS), dtype=np.int32)
        np.add.at(self.bins, (group, targets % BINS), weights)
        self.expected = np.add.reduceat(weights * tradable[targets], row)
        self.strength = strength

    def step(self, history, labels, previous):
        pos = np.searchsorted(self.keys, history)
        if not len(self.keys):
            return previous
        idx = pos.clip(0, len(self.keys) - 1)
        known = (pos < len(self.keys)) & (self.keys[idx] == history)
        same = lookup(self.pairs, self.counts, history * VOCABULARY + labels)
        numerator = np.column_stack((same, self.bins[idx, labels % BINS], self.expected[idx]))
        denominator = self.total[idx, None] + self.strength
        predictions = (numerator + self.strength * previous) / denominator
        return np.where(known[:, None], predictions, previous)


def markov_fit(tokens, trade_return, train_dates):
    complete = (tokens[:, train_dates] >= 0)
    symbols, days = np.nonzero(complete)
    times = train_dates[days]
    labels = tokens[symbols, times]
    histogram = np.bincount(labels, minlength=VOCABULARY) + 0.5
    prior = histogram / histogram.sum()
    target_trade = trade_return[symbols, times]
    valid = np.isfinite(target_trade) & (times < train_dates[-1])
    sums = np.bincount(labels[valid], weights=target_trade[valid].clip(-0.30, 0.30), minlength=VOCABULARY)
    counts = np.bincount(labels[valid], minlength=VOCABULARY)
    tradable = sums / (counts + 30)
    orders = []
    for depth, strength in [(1, 12), (2, 24), (3, 36)]:
        valid_time = times >= train_dates[0] + depth
        sid, tid = symbols[valid_time], times[valid_time]
        histories = tokens[sid[:, None], tid[:, None] - np.arange(depth, 0, -1)]
        present = (histories >= 0).all(axis=1)
        key = histories[present] @ (VOCABULARY ** np.arange(depth - 1, -1, -1, dtype=np.int64))
        orders.append(TransitionOrder(key, tokens[sid[present], tid[present]], tradable, strength))
    found = orders[0]
    prev, label = found.pairs // VOCABULARY, found.pairs % VOCABULARY
    core = np.full((CORE, CORE), 0.03, dtype=np.float32)
    np.add.at(core, (prev % CORE, label % CORE), found.counts)
    core /= core.sum(1, keepdims=True)
    return prior, tradable, orders, np.log(core)


class TypedTransformer(nn.Module):
    def __init__(self, context, width=32, heads=4, layers=6):
        super().__init__()
        self.core = nn.Embedding(CORE, width)
        self.kind = nn.Embedding(len(TYPE_LABELS), width)
        self.features = nn.Linear(len(AUX_FIELDS), width)
        self.positions = nn.Embedding(context, width)
        self.blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(width, heads, 2 * width, dropout=0.1, activation="gelu",
                                       batch_first=True, norm_first=True),
            layers, norm=nn.LayerNorm(width), enable_nested_tensor=False,
        )
        self.market_head = nn.Linear(width, CORE)
        self.type_head = nn.Linear(width, len(TYPE_LABELS))
        self.rank_head = nn.Linear(width, 1)
        self.register_buffer("mask", torch.ones(context, context, dtype=torch.bool).triu(1))
        self.register_buffer("core_prior", torch.zeros(CORE, CORE))
        nn.init.zeros_(self.market_head.weight)
        nn.init.zeros_(self.market_head.bias)

    def forward(self, token, auxiliary, tail=1):
        core = token % CORE
        steps = token.size(1)
        x = (self.core(core) + self.kind(token // CORE) + self.features(auxiliary)) / 1.732
        x = x + self.positions(torch.arange(steps, device=token.device))
        h = self.blocks(x, mask=self.mask[:steps, :steps], is_causal=True)[:, -tail:]
        market = self.market_head(h) + self.core_prior[core[:, -tail:]]
        return market, self.type_head(h), self.rank_head(h[:, -1]).squeeze(-1)


def batch_inputs(panel, tokens, stock, day, context, device):
    days = day[:, None] + np.arange(-context, 0)
    return (
        torch.from_numpy(tokens[stock[:, None], days].astype(np.int64)).to(device),
        torch.from_numpy(panel.auxiliary[stock[:, None], days]).to(device),
    )


def cross_section_loss(score, realized):
    """Compare opposite quintiles and tilt the loss toward strong tradable winners."""
    valid = torch.isfinite(realized)
    realized, score = realized[valid], score[valid]
    count = len(realized) // 5
    if count < 2:
        return score.sum() * 0
    order = realized.argsort()
    winners, losers = order[-count:], order[:count]
    excess = realized[winners] - realized.median()
    extra = 1 + (excess / 0.05).clamp(0, 1)
    gaps = score[winners, None] - score[losers][None, :]
    return (F.softplus(-gaps) * extra[:, None]).mean()


def split_days(dates, start, train_end, val_end, end):
    train = np.flatnonzero((dates >= start) & (dates <= train_end))
    val = np.flatnonzero((dates > train_end) & (dates <= val_end))
    test = np.flatnonzero((dates > val_end) & (dates <= end))
    if min(len(train), len(val), len(test)) < 2:
        raise ValueError("训练、验证和测试各需至少两个交易日。")
    return train, val, test


def checked_dates(group, eligibility, limit):
    # Reserve the last session of a split so the rank exit stays inside the split.
    return sample_days(group[:-1][eligibility[:, group[:-1]].any(0)], limit)


@torch.inference_mode()
def evaluate(model, panel, tokens, eligible, days, priors, tokenizer, context, batch, device):
    model.eval()
    prior, trade_expectation, orders, _ = priors
    return_prior = prior.reshape(-1, BINS).sum(0)
    names = ("model", "markov1", "markov3")
    metrics = {key: np.zeros(3) for key in names}
    daily = []
    total = trade_total = 0
    representatives = np.asarray(tokenizer.return_values, dtype=np.float64)
    for day in days:
        stocks = np.flatnonzero(eligible[:, day])
        outcome = panel.trade_return[stocks, day]
        tradable = np.isfinite(outcome)
        scores = {key: [] for key in names}
        distribution_scores = []
        for off in range(0, len(stocks), batch):
            sid = stocks[off:off + batch]
            inputs = batch_inputs(panel, tokens, sid, np.full(len(sid), day), context, device)
            core, kind, score = model(*inputs)
            core, kind = core[:, 0].softmax(-1).cpu().numpy(), kind[:, 0].softmax(-1).cpu().numpy()
            target = tokens[sid, day].astype(np.int64)
            nd = np.arange(len(sid))
            p_ret = core.reshape(len(sid), -1, BINS).sum(1)
            prob = core[nd, target % CORE] * kind[nd, target // CORE]
            stats = [
                (prob, p_ret[nd, target % BINS], score.cpu().numpy(),
                 (kind.argmax(1) * CORE + core.argmax(1)) == target)
            ]
            p = np.column_stack((prior[target], return_prior[target % BINS], np.full(len(sid), prior @ trade_expectation)))
            for i, order in enumerate(orders):
                hist = tokens[sid[:, None], day - np.arange(i + 1, 0, -1)]
                keys = hist @ (VOCABULARY ** np.arange(i, -1, -1, dtype=np.int64))
                p = order.step(keys, target, p)
                if i in (0, 2):
                    stats.append((p[:, 0], p[:, 1], p[:, 2], None))
            for name, (joint, marginal, values, correct) in zip(names, stats):
                metrics[name][:2] += (-np.log(joint.clip(1e-12)).sum(), -np.log(marginal.clip(1e-12)).sum())
                if correct is not None:
                    metrics[name][2] += correct.sum()
                scores[name].append(values)
            distribution_scores.append(p_ret.astype(np.float64) @ representatives)
        entry = {"date": str(panel.dates[day]), "samples": len(stocks), "tradable_samples": int(tradable.sum())}
        for name in names:
            entry[name + "_rank_ic"] = rank_ic(np.concatenate(scores[name])[tradable], outcome[tradable])
        entry["model_return_rank_ic"] = rank_ic(np.concatenate(distribution_scores), panel.features[stocks, day, 2])
        daily.append(entry)
        total += len(stocks)
        trade_total += int(tradable.sum())
    if not total:
        raise ValueError("评估期没有可用样本。")
    out = {"sessions": len(daily), "samples": total, "tradable_samples": trade_total, "daily": daily}
    for key in names:
        records = [day[key + "_rank_ic"] for day in daily if day[key + "_rank_ic"] is not None]
        out[key] = dict(joint_nll=float(metrics[key][0] / total), return_nll=float(metrics[key][1] / total),
                        rank_ic=float(np.mean(records)) if records else None)
    baseline_ic = [d["model_return_rank_ic"] for d in daily if d["model_return_rank_ic"] is not None]
    out["model"]["return_token_rank_ic"] = float(np.mean(baseline_ic)) if baseline_ic else None
    out["model"]["joint_accuracy"] = float(metrics["model"][2] / total)
    return out


def clipped_metrics(result):
    return {k: v for k, v in result.items() if k != "daily"}


def train_typed(args, progress):
    if min(args.epochs, args.steps, args.batch, args.context) < 1 or args.context < 6 or args.batch < 20:
        raise ValueError("训练步数和轮数必须为正。上下文至少 6，批量至少 20。")
    if min(args.val_days, args.test_days) < 0 or not args.start <= args.train_end < args.val_end < args.end:
        raise ValueError("无效的时间切分或评估日数量。")
    device = args.device
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    structure = dict(context=args.context, width=32, heads=4, layers=6)
    log = {"run": out.name, "status": "loading", "started_at": datetime.now(timezone.utc).isoformat(),
           "architecture": structure, "vocabulary": VOCABULARY, "features": FOOTPRINT,
           "config": {"epochs": args.epochs, "steps": args.steps, "batch": args.batch, "device": device,
                      "rank_weight": 0.5, "return_loss_weight": 0.25, "seed": args.seed},
           "epochs": [], "steps": []}

    def state(status, message):
        log.update(status=status, message=message, elapsed_s=round(time.perf_counter() - started, 2))
        write_json(out / "progress.json", log)

    try:
        state("loading", "正在读取行情、板块和交易状态。")
        panel = load_research_panel(args.qlib_db, args.db, args.status_csv, args.start, args.end, args.context, progress)
        train, validation, test = split_days(panel.dates, args.start, args.train_end, args.val_end, args.end)
        tokenizer = JointTokenizer.fit(panel.features[:, train, :])
        core = tokenizer.encode(panel.features)
        tokens = np.where(core >= 0, panel.kind.astype(np.int16) * CORE + core, -1)
        eligible = eligible_targets(tokens, args.context)
        seen = np.bincount(tokens[:, train][tokens[:, train] >= 0], minlength=VOCABULARY)
        log.update(source=panel.source, coverage=panel.coverage, seen_tokens=int(np.count_nonzero(seen)),
                   seen_tokens_by_type=np.count_nonzero(seen.reshape(-1, CORE), axis=1).tolist(),
                   splits={key: {"start": str(panel.dates[group[0]]), "end": str(panel.dates[group[-1]]),
                                 "sessions": len(group), "windows": int(eligible[:, group].sum())}
                           for key, group in zip(("train", "validation", "test"), (train, validation, test))})
        state("preparing", "拟合一阶、二阶及三阶 Markov 转移。")
        priors = markov_fit(tokens, panel.trade_return, train)
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        model = TypedTransformer(**structure).to(device)
        model.core_prior.copy_(torch.from_numpy(priors[3]).to(device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0006)
        tail = min(6, args.context)
        valid_train_days = train[(train >= train[0] + tail - 1) & (train < train[-1])]
        valid_train_days = valid_train_days[eligible[:, valid_train_days].sum(0) >= 20]
        val_days = checked_dates(validation, eligible, args.val_days)
        test_days = checked_dates(test, eligible, args.test_days)
        if min(len(valid_train_days), len(val_days), len(test_days)) < 1:
            raise ValueError("连续交易日或特征覆盖不足。")
        best_ic = -float("inf")
        best_checkpoint = best_validation = None
        for epoch in range(1, args.epochs + 1):
            state("training", f"第 {epoch}/{args.epochs} 轮训练。")
            model.train()
            epoch_started = time.perf_counter()
            epochs_loss = []
            for iteration in range(1, args.steps + 1):
                step_started = time.perf_counter()
                day = int(rng.choice(valid_train_days))
                pool = np.flatnonzero(eligible[:, day])
                stock = rng.choice(pool, min(len(pool), args.batch), replace=False)
                dates = np.full(len(stock), day)
                model_input = batch_inputs(panel, tokens, stock, dates, args.context, device)
                y = torch.from_numpy(tokens[stock[:, None], day + np.arange(1 - tail, 1)].astype(np.int64)).to(device)
                actual = torch.from_numpy(panel.trade_return[stock, day]).to(device)
                optimizer.zero_grad(set_to_none=True)
                market, kinds, scores = model(*model_input, tail=tail)
                core_logits = market.reshape(-1, CORE)
                flat_y = y.flatten()
                joint = F.cross_entropy(core_logits, flat_y % CORE) + F.cross_entropy(kinds.reshape(-1, len(TYPE_LABELS)), flat_y // CORE)
                returns = F.cross_entropy(core_logits.reshape(-1, CORE // BINS, BINS).logsumexp(1), flat_y % BINS)
                rank = cross_section_loss(scores, actual)
                loss = joint + 0.25 * returns + 0.5 * rank
                if not torch.isfinite(loss):
                    raise ValueError("训练损失不是有限值。")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                values = [float(x.detach().cpu()) for x in (loss, joint, returns, rank)]
                log["steps"].append({"epoch": epoch, "step": iteration, "date": str(panel.dates[day]),
                                     "seconds": round(time.perf_counter() - step_started, 4),
                                     **dict(zip(("loss", "joint_loss", "return_loss", "rank_loss"), values))})
                epochs_loss.append(values)
                # Keep cross-section batches observable, with only a bounded pending write interval.
                if iteration % 5 == 0 or iteration == args.steps:
                    state("training", f"第 {epoch}/{args.epochs} 轮 · 批次 {iteration}/{args.steps}。")
            train_sec = time.perf_counter() - epoch_started
            state("validating", f"第 {epoch} 轮训练完成，正在计算验证集 Rank IC。")
            stamp = time.perf_counter()
            val = evaluate(model, panel, tokens, eligible, val_days, priors, tokenizer, args.context,
                           max(args.batch, 256), device)
            record = {"epoch": epoch, "train_seconds": round(train_sec, 2),
                      "validation_seconds": round(time.perf_counter()-stamp, 2),
                      **dict(zip(("loss", "joint_loss", "return_loss", "rank_loss"), np.mean(epochs_loss, axis=0).tolist())),
                      "validation": clipped_metrics(val)}
            log["epochs"].append(record)
            ic = val["model"]["rank_ic"]
            if best_checkpoint is None or (ic is not None and ic > best_ic):
                best_ic = ic if ic is not None else -float("inf")
                best_checkpoint = copy.deepcopy({name: weight.detach().cpu() for name, weight in model.state_dict().items()})
                best_validation = val
                log["best_epoch"] = epoch
            progress(f"第 {epoch} 轮：loss={record['loss']:.4f}，验证 Rank IC={ic}，训练 {train_sec:.1f}s。")
        model.load_state_dict(best_checkpoint)
        state("testing", "正在用已选权重评估测试期与两类 Markov。")
        stamp = time.perf_counter()
        final = evaluate(model, panel, tokens, eligible, test_days, priors, tokenizer, args.context,
                         max(args.batch, 256), device)
        log["test_seconds"] = round(time.perf_counter() - stamp, 2)
        log["validation"] = clipped_metrics(best_validation)
        log["test"] = clipped_metrics(final)
        torch.save(best_checkpoint, out / "weights.pt")
        pd.DataFrame(final["daily"]).to_csv(out / "test_daily.csv", index=False)
        write_json(out / "model.json", {"architecture": structure, "tokenizer": asdict(tokenizer),
                                        "features": FOOTPRINT, "trade_rank_horizon": "next open to following open",
                                        "weights_sha256": sha256(out / "weights.pt"), "source": panel.source,
                                        "validation": best_validation, "test": final})
        state("finished", "训练完成，已保存模型与逐日测试结果。")
        return clipped_metrics(final)
    except (Exception, KeyboardInterrupt) as exc:
        state("failed", str(exc) or "训练中断。")
        raise
