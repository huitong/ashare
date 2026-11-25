"""CLI for data import, training, inference and backtesting."""

import argparse
import json
import sqlite3
import sys
import tarfile
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .backtest import simulate
from .config import Strategy
from .demo import create_demo
from .features import Features
from .models import load_models
from .providers import (
    TushareFeed, import_holdings, import_hotspots, import_industries, import_market, synchronize,
)
from .qlib_data import DEFAULT_QLIB_DB
from .rank import Ranker
from .run_files import publish_directory
from .store import Store, iso_day, latest_completed_wall_day
from .temporal import MODEL_ID, train


def json_text(value):
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)


def progress(message):
    print(message, file=sys.stderr, flush=True)


def result_paths(paths):
    artifacts = {}
    for value in paths:
        key, sep, path = value.partition("=")
        if not sep:
            key, path = MODEL_ID, value
        if key in artifacts or not path:
            raise ValueError(f"模型路径无效或重复：{value}。")
        artifacts[key] = Path(path)
    return artifacts


def configure_run(args, market):
    strategy = Strategy.load(args.config)
    if args.rules_only:
        from .models import FACTORS
        strategy = replace(strategy, weights={k: v for k, v in strategy.weights.items() if k in FACTORS})
    artifacts = result_paths(args.model)
    models = load_models(strategy, artifacts, market.status()["source"], args.device)
    return strategy, models


def add_rank_args(command):
    command.add_argument("--config", type=Path, help="策略 JSON，默认使用项目内配置")
    command.add_argument("--model", action="append", default=[], help="模型目录，或 MODEL_ID=目录。可重复。")
    command.add_argument("--rules-only", action="store_true", help="仅运行趋势和 MACD")


def add_feed_args(command):
    command.add_argument("--rpm", type=int, default=45, help="接口总请求数上限/分钟")


def parser():
    cli = argparse.ArgumentParser(description="A 股日频量化研究系统")
    cli.add_argument("--db", type=Path, default=Path("data/ashare.sqlite"))
    cli.add_argument("--threads", type=int, default=2)
    cli.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    commands = cli.add_subparsers(dest="command", required=True)

    commands.add_parser("status", help="查看行情、日历、行业和新闻覆盖")
    demo = commands.add_parser("demo-data", help="生成独立模拟行情库")
    demo.add_argument("--end", type=iso_day, default="2026-11-20")
    demo.add_argument("--days", type=int, default=165)
    demo.add_argument("--stocks", type=int, default=48)

    sync = commands.add_parser("sync", help="按交易日同步 Tushare A 股行情")
    sync.add_argument("--start", type=iso_day, help="默认过去一年加 120 个自然日预热")
    sync.add_argument("--end", type=iso_day, default=latest_completed_wall_day())
    sync.add_argument("--refresh", action="store_true", help="重新下载已完成交易日")
    add_feed_args(sync)
    industry = commands.add_parser("sync-industries", help="同步申万一级行业")
    industry.add_argument("--assume-effective-known", action="store_true", help="以纳入生效日代替历史可用时间")
    add_feed_args(industry)

    qlib = commands.add_parser("import-qlib", help="解压并导入完整 Qlib 日线及每日指标包")
    qlib.add_argument("market", type=Path, help="qlib_bin.tar.gz")
    qlib.add_argument("daily_basic", type=Path, help="daily_basic_qlib_features.tar.gz")
    qlib.add_argument("--out", type=Path, default=DEFAULT_QLIB_DB)
    qlib.add_argument("--release", help="GitHub 发布标签，传入 2026-11-23 时校验发布文件 SHA-256")
    qlib.add_argument("--extract-to", type=Path, help="保留解包文件的目录")
    web = commands.add_parser("serve", help="启动本地 A 股行情和数据质量浏览器")
    web.add_argument("--qlib-db", type=Path, default=DEFAULT_QLIB_DB)
    web.add_argument("--port", type=int, default=8765)
    web.add_argument("--runs-dir", type=Path, default=Path("runs"))
    web.add_argument("--trade-reference", type=Path, help="可选交易参考排名 JSON 路径")

    for name, help_ in (
        ("rank-trade-reference", "用最近已完成交易日生成前 30 名交易参考"),
        ("sync-trade-reference", "校验并同步 GitHub 最新 Qlib 增量，再生成交易参考"),
    ):
        command = commands.add_parser(name, help=help_)
        command.add_argument("--qlib-db", type=Path, default=DEFAULT_QLIB_DB)
        command.add_argument("--trade-db", type=Path, default=Path("data/qlib-trade.sqlite"))
        command.add_argument("--low-vol-model", type=Path, default=Path("runs/low-vol-2020-2023-v1"))
        command.add_argument("--out", type=Path, default=Path("data/trade-reference.json"))
        if name == "sync-trade-reference":
            command.add_argument("--extracted", type=Path, default=Path("data/qlib-extracted"))
            command.add_argument("--release", type=iso_day, help="按指定发布标签同步，默认检查最新标签")
            command.add_argument("--release-metadata", type=Path, help="使用本地 GitHub 发布 JSON 和同目录增量包")

    bulk = commands.add_parser("import-market", help="导入标准化的行情 CSV 目录")
    bulk.add_argument("folder", type=Path)
    bulk.add_argument("--source", required=True)
    for name in ("industries", "hotspots", "holdings"):
        command = commands.add_parser(f"import-{name}", help=f"导入 {name} CSV")
        command.add_argument("file", type=Path)
        if name == "holdings":
            command.add_argument("--asof", required=True, help="持仓快照可用时间，北京时间或 ISO 时区")

    futu = commands.add_parser("futu-pull", help="通过 OpenD 下载指定 A 股的原始/后复权日 K")
    futu.add_argument("--host", default="127.0.0.1")
    futu.add_argument("--port", type=int, default=11111)
    futu.add_argument("--code", action="append", default=[], help="如 SH.600519，可重复")
    futu.add_argument("--codes", type=Path, help="代码清单，每行一个富途代码")
    futu.add_argument("--start", type=iso_day)
    futu.add_argument("--end", type=iso_day, default=latest_completed_wall_day())
    futu.add_argument("--out", type=Path, required=True)
    assemble = commands.add_parser("futu-import", help="合并富途行情与逐日市值、涨跌停价")
    assemble.add_argument("--pull", required=True, type=Path)
    assemble.add_argument("--fundamentals", required=True, type=Path)
    assemble.add_argument("--securities", required=True, type=Path)

    joint = commands.add_parser("train-joint", help="从 Qlib 训练 5 层、1000 类联合 token 模型")
    joint.add_argument("--qlib-db", type=Path, default=DEFAULT_QLIB_DB)
    joint.add_argument("--out", type=Path, required=True)
    joint.add_argument("--start", type=iso_day, default="2022-01-01")
    joint.add_argument("--train-end", type=iso_day, default="2024-12-31")
    joint.add_argument("--val-end", type=iso_day, default="2025-12-31")
    joint.add_argument("--end", type=iso_day, default="9999-12-31")
    joint.add_argument("--context", type=int, default=24)
    joint.add_argument("--width", type=int, default=32)
    joint.add_argument("--heads", type=int, default=4)
    joint.add_argument("--epochs", type=int, default=3)
    joint.add_argument("--batch", type=int, default=128)
    joint.add_argument("--train-blocks", type=int, default=4096)
    joint.add_argument("--val-days", type=int, default=8, help="验证日数量，0 为全部")
    joint.add_argument("--test-days", type=int, default=20, help="测试日数量，0 为全部")
    joint.add_argument("--seed", type=int, default=42)
    joint.add_argument("--markov-residual", action="store_true", help="让 Transformer 修正一阶 Markov 先验")
    joint.add_argument("--exclude-tested", type=Path, help="排除指定 model.json 中已经评估过的测试日期")

    typed = commands.add_parser("train-typed", help="训练 6 层、5000 token、带可交易排序损失的模型")
    typed.add_argument("--qlib-db", type=Path, default=DEFAULT_QLIB_DB)
    typed.add_argument("--out", type=Path, required=True)
    typed.add_argument("--status-csv", type=Path, help="可选历史状态 CSV")
    typed.add_argument("--start", type=iso_day, default="2022-01-01")
    typed.add_argument("--train-end", type=iso_day, default="2024-12-31")
    typed.add_argument("--val-end", type=iso_day, default="2025-12-31")
    typed.add_argument("--end", type=iso_day, default="9999-12-31")
    typed.add_argument("--context", type=int, default=24)
    typed.add_argument("--epochs", type=int, default=4)
    typed.add_argument("--steps", type=int, default=128)
    typed.add_argument("--batch", type=int, default=96)
    typed.add_argument("--val-days", type=int, default=12)
    typed.add_argument("--test-days", type=int, default=0)
    typed.add_argument("--seed", type=int, default=42)

    model = commands.add_parser("train", help="训练并按时间验证下一日收益 token 模型")
    model.add_argument("--start", type=iso_day, help="默认截止日前一年")
    model.add_argument("--end", type=iso_day)
    model.add_argument("--train-end", type=iso_day)
    model.add_argument("--val-end", type=iso_day)
    model.add_argument("--out", type=Path, required=True)
    model.add_argument("--context", type=int, default=20)
    model.add_argument("--width", type=int, default=32)
    model.add_argument("--heads", type=int, default=4)
    model.add_argument("--layers", type=int, default=2)
    model.add_argument("--epochs", type=int, default=5)
    model.add_argument("--batch", type=int, default=256)
    model.add_argument("--seed", type=int, default=42)

    predict = commands.add_parser("predict", help="收盘后生成候选股排名")
    predict.add_argument("--asof", type=iso_day)
    predict.add_argument("--out", type=Path, required=True)
    predict.add_argument("--probabilities", action="store_true", help="另存候选股票的下一日收益分桶概率")
    add_rank_args(predict)

    bt = commands.add_parser("backtest", help="用下一交易日开盘执行历史信号")
    bt.add_argument("--start", type=iso_day)
    bt.add_argument("--end", type=iso_day)
    bt.add_argument("--out", type=Path, required=True)
    bt.add_argument("--compare-baseline", action="store_true", help="同时运行仅趋势与 MACD 的基线")

    rotation = commands.add_parser("backtest-rotation", help="用 Qlib 回放 Top 10 买入、跌出 Top 20 卖出的组合")
    rotation.add_argument("--qlib-db", type=Path, default=DEFAULT_QLIB_DB)
    rotation.add_argument("--model", type=Path, default=Path("runs/typed-token-v1"))
    rotation.add_argument("--low-vol-model", type=Path, help="回放低波增强模型及原版低波对照")
    rotation.add_argument("--low-vol-grid", action="store_true", help="交叉比较原低波权重与三类卖出规则")
    rotation.add_argument("--intraday", action="store_true", help="用日 K 对照低波 MLP 的开盘和日内限价策略")
    rotation.add_argument("--slippage-bps", type=float, default=0, help="买入加价、卖出减价的单边滑点，单位基点")
    rotation.add_argument("--compare-slippage", action="store_true", help="同场回测各方案的 0 bp 对照")
    rotation.add_argument("--status-csv", type=Path, help="可选逐日 ST 及涨跌停价")
    rotation.add_argument("--benchmark-csv", type=Path,
                          help="指数收盘数据，默认读取 data/benchmark-index-closes.csv（如有）")
    rotation.add_argument("--start", type=iso_day, default="2026-01-01")
    rotation.add_argument("--end", type=iso_day, default="9999-12-31")
    rotation.add_argument("--cash", type=float, default=1_000_000)
    rotation.add_argument("--out", type=Path, required=True)
    low_vol = commands.add_parser("train-low-vol", help="训练 60 日低波增强模型")
    low_vol.add_argument("--qlib-db", type=Path, default=DEFAULT_QLIB_DB)
    low_vol.add_argument("--status-csv", type=Path)
    low_vol.add_argument("--start", type=iso_day, default="2020-01-01")
    low_vol.add_argument("--split", type=iso_day, default="2022-12-31")
    low_vol.add_argument("--end", type=iso_day, default="2023-12-31")
    low_vol.add_argument("--out", type=Path, required=True)
    add_rank_args(bt)
    return cli


def latest(market):
    info = market.status()
    end = info["last_complete"]
    if end is None:
        raise ValueError("数据库没有已同步的行情。")
    pending = market.sessions(end, "9999-12-31", require_complete=False)
    if len(pending) > 1 and pending[-1] <= latest_completed_wall_day():
        raise ValueError(f"{pending[-1]} 已在交易日历中，但行情尚未同步。请同步或明确指定历史日期。")
    return end


def lookback(end, warmup=0):
    return (pd.Timestamp(end) - pd.DateOffset(years=1) - pd.Timedelta(days=warmup)).date().isoformat()


def run_prediction(args, market, features, strategy, models, day):
    if day not in features.calendar:
        raise ValueError(f"{day} 没有已同步的交易日行情。")
    for model in models:
        if hasattr(model, "require_out_of_sample"):
            model.require_out_of_sample(day)
    ranker = Ranker(market, features, strategy, models)
    holdings = market.holdings(day)
    ranks = ranker.rank(day, holdings)
    temporal = next((model for model in models if model.model_id == MODEL_ID), None)
    if args.probabilities and temporal is None:
        raise ValueError("导出收益概率需要加载时序模型。")
    reports = dict(
        asof=day, data=market.status(), candidates=len(ranks), scored=int(ranks["score"].notna().sum()),
        holding_count=len(holdings), strategy=strategy.as_dict(),
        model_versions={m.model_id: m.metadata for m in models if hasattr(m, "metadata")},
    )

    def writer(folder):
        ranks.to_csv(folder / "ranking.csv", index=False)
        if args.probabilities:
            eligible = ranks.loc[ranks["has_bar"], "code"]
            probs = temporal.distribution(day, features.day(day).loc[eligible], features)
            probs.add_prefix("return_bin_").to_csv(folder / "probabilities.csv")
        (folder / "summary.json").write_text(json_text(reports), encoding="utf-8")

    publish_directory(args.out, writer)
    return {key: reports[key] for key in ("asof", "candidates", "scored", "holding_count")}


def run_backtest(args, market, features, strategy, models, end):
    ranker = Ranker(market, features, strategy, models)
    learned = [model for model in models if hasattr(model, "require_out_of_sample")]
    start = args.start or (max(m.metadata["validated_through"] for m in learned)
                           if learned else features.sessions[0])
    if not features.sessions[0] <= start < end:
        raise ValueError("回测开始日须在数据覆盖内并早于结束日。")
    days = [day for day in features.sessions if start <= day <= end]
    if len(days) < 2:
        raise ValueError("回测至少需要两个交易日。")
    for model in learned:
        model.require_out_of_sample(days[0])
    runs = {"strategy": simulate(ranker, days)}
    if args.compare_baseline:
        from .models import FACTORS
        if not learned:
            raise ValueError("基线对比需要至少一个训练模型。")
        baseline = replace(strategy, weights={k: v for k, v in strategy.weights.items() if k in FACTORS})
        baseline_ranker = Ranker(market, features, baseline, [m for m in models if m.model_id in FACTORS])
        runs["rules_only"] = simulate(baseline_ranker, days)
    reports = {
        "start": days[0], "end": days[-1], "data": market.status(), "strategy_config": strategy.as_dict(),
        "model_versions": {m.model_id: m.metadata for m in learned},
        "runs": {key: run["summary"] for key, run in runs.items()},
    }

    def writer(folder):
        for key, run in runs.items():
            subfolder = folder / key
            subfolder.mkdir()
            for name, table in run.items():
                if name != "summary":
                    table.to_csv(subfolder / f"{name}.csv", index=False)
        (folder / "summary.json").write_text(json_text(reports), encoding="utf-8")

    publish_directory(args.out, writer)
    return reports["runs"]


def run(args):
    command = args.command
    if command in {"rank-trade-reference", "sync-trade-reference"}:
        import torch
        from .trade_reference import score_reference, sync_reference
        if args.threads < 1:
            raise ValueError("线程数必须为正数。")
        torch.set_num_threads(args.threads)
        if command == "sync-trade-reference":
            result = sync_reference(args.qlib_db, args.trade_db, args.extracted, args.db, args.low_vol_model,
                                    args.out, progress, release_tag=args.release, metadata=args.release_metadata)
        else:
            qlib = args.trade_db if args.trade_db.is_file() and args.qlib_db == DEFAULT_QLIB_DB else args.qlib_db
            result = score_reference(qlib, args.db, args.low_vol_model, args.out, progress)
        return {key: result[key] for key in ("day", "release", "candidate_count", "default_strategy")}
    if command == "train-low-vol":
        import torch
        from .low_vol import train_low_vol
        if args.threads < 1:
            raise ValueError("线程数必须为正数。")
        torch.set_num_threads(args.threads)
        return train_low_vol(args, progress)
    if command == "backtest-rotation":
        import torch
        from .rotation import rotate
        if args.threads < 1:
            raise ValueError("线程数必须为正数。")
        torch.set_num_threads(args.threads)
        default_benchmarks = Path("data/benchmark-index-closes.csv")
        if args.benchmark_csv is None and default_benchmarks.is_file():
            args.benchmark_csv = default_benchmarks
        return rotate(args, progress)
    if command == "train-typed":
        import torch
        from .typed_token import train_typed
        if args.threads < 1:
            raise ValueError("线程数必须为正数。")
        torch.set_num_threads(args.threads)
        return train_typed(args, progress)
    if command == "train-joint":
        import torch
        from .joint_token import load_panel, train_joint
        if args.threads < 1 or args.out.exists():
            raise ValueError("线程数必须为正，且输出目录不得已存在。")
        torch.set_num_threads(args.threads)
        panel = load_panel(args.qlib_db, args.start, args.end, args.context, progress)
        prior_dates = ()
        if args.exclude_tested:
            prior = json.loads(args.exclude_tested.read_text(encoding="utf-8"))
            prior_dates = tuple(iso_day(row["date"]) for row in prior["test"]["daily"])
        metadata = train_joint(
            panel, args.out, start=args.start, train_end=args.train_end, val_end=args.val_end,
            end=args.end, context=args.context, width=args.width, heads=args.heads, epochs=args.epochs,
            batch=args.batch, train_blocks=args.train_blocks, val_days=args.val_days,
            test_days=args.test_days, seed=args.seed, device=args.device, progress=progress,
            markov_residual=args.markov_residual, exclude_test=prior_dates,
        )
        return {key: metadata[key] if key not in {"validation", "test"} else
                {k: v for k, v in metadata[key].items() if k != "daily"}
                for key in ("splits", "parameters", "seen_training_tokens", "best_epoch", "validation", "test")}

    if command == "import-qlib":
        from .qlib_data import import_qlib
        return import_qlib(args.market, args.daily_basic, args.out, args.release, args.extract_to, progress)
    if command == "serve":
        from .web import serve
        serve(args.db, args.qlib_db, args.port, progress, runs=args.runs_dir, trade_reference=args.trade_reference)
        return {}
    if command == "futu-pull":
        from .futu_feed import pull_futu
        return pull_futu(args, args.start or lookback(args.end, warmup=120), progress)
    store = Store(args.db)
    if command == "status":
        with store.snapshot() as market:
            return market.status()
    if command == "demo-data":
        create_demo(store, args.end, args.days, args.stocks)
        with store.snapshot() as market:
            return market.status()
    if command == "sync":
        if not 1 <= args.rpm <= 500:
            raise ValueError("--rpm 必须在 1–500 之间。")
        start = args.start or lookback(args.end, warmup=120)
        if not start <= args.end <= latest_completed_wall_day():
            raise ValueError("同步结束日须晚于开始日且行情已到上海时间 17:30 截止点。")
        feed = TushareFeed(args.rpm)
        return {"downloaded_sessions": synchronize(store, feed, start, args.end, args.refresh, progress)}
    if command == "sync-industries":
        if not 1 <= args.rpm <= 500:
            raise ValueError("--rpm 必须在 1–500 之间。")
        with store.snapshot() as market:
            codes = market.securities().keys()
        if not codes:
            raise ValueError("请先同步股票基础信息。")
        rows = TushareFeed(args.rpm).industries(codes, args.assume_effective_known)
        store.put_industries(rows)
        return {"industry_memberships": len(rows)}
    if command == "import-market":
        return {"imported_sessions": import_market(store, args.folder, args.source)}
    if command == "futu-import":
        from .futu_feed import import_futu
        return import_futu(store, args.pull, args.fundamentals, args.securities)
    imports = {
        "import-industries": import_industries,
        "import-hotspots": import_hotspots,
        "import-holdings": import_holdings,
    }
    if command in imports:
        extra = [args.asof] if command == "import-holdings" else []
        return {"imported_rows": imports[command](store, args.file, *extra)}
    import torch
    if args.threads < 1:
        raise ValueError("--threads 必须为正整数。")
    torch.set_num_threads(args.threads)
    with store.snapshot() as market:
        requested = args.asof if command == "predict" else args.end
        end = requested or latest(market)
        features = Features(market, end)
        if end > features.sessions[-1]:
            if command == "predict" or end > market.status()["last_complete"]:
                raise ValueError(f"{end} 超出已同步交易日范围。")
        if command == "train":
            end = features.sessions[-1]
            metadata = train(
                features, market.status()["source"], args.start or lookback(end), end, args.out,
                train_end=args.train_end, val_end=args.val_end, context=args.context, width=args.width,
                heads=args.heads, layers=args.layers, epochs=args.epochs, batch=args.batch, seed=args.seed,
                device=args.device, progress=progress,
            )
            return {key: metadata[key] for key in ("trained_through", "validated_through", "tested_through", "metrics")}
        strategy, models = configure_run(args, market)
        if command == "predict":
            return run_prediction(args, market, features, strategy, models, end)
        return run_backtest(args, market, features, strategy, models, end)


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = run(args)
    except (OSError, ValueError, KeyError, RuntimeError, sqlite3.Error, tarfile.TarError) as exc:
        progress(f"错误：{exc}")
        return 1
    print(json_text(result))
    return 0
