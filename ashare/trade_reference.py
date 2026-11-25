"""Generate the latest close-based research rankings with the backtest score engine."""

import json
import os
import tempfile
from contextlib import closing, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from shutil import copyfileobj
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .config import Strategy
from .low_vol import LowVolFeatures, LowVolScores, low_vol_portfolios
from .qlib_data import date_value
from .qlib_delta import apply_delta
from .rank import Recall
from .rotation import RECALL_SIZE, EmptyReference, GridRanks, capped_recall, low_vol_grid
from .store import MarketSnapshot, latest_completed_wall_day
from .typed_data import load_research_panel
from .web import Explorer, open_readonly

DEFAULT_STRATEGY = "mlp_value_35"
SHOW_COUNT = 30
RELEASE_API = "https://api.github.com/repos/xu-duqing/investment_data/releases/"
DELTA_ASSETS = ("qlib_bin_delta.tar.gz", "daily_basic_qlib_features_delta.tar.gz")


def portfolio_choices():
    choices, exits = low_vol_grid(low_vol_portfolios())
    return {key.removesuffix("_top20"): (label.removesuffix(" · 跌出 Top 20"), weights)
            for key, (label, weights) in choices.items() if exits[key] == "top20"}


def score_reference(qlib, market, model, output, progress):
    with closing(open_readonly(qlib)) as db:
        latest = db.execute("SELECT day FROM qlib_calendar c WHERE day <= ? AND EXISTS"
                            "(SELECT 1 FROM qlib_daily d WHERE d.day=c.day AND close>0)"
                            " ORDER BY day DESC LIMIT 1", (latest_completed_wall_day(),)).fetchone()
    if latest is None:
        raise ValueError("行情库没有已完成的交易日。")
    day = latest[0]
    progress(f"正在计算 {day} 的交易参考排名。")
    features = LowVolFeatures()
    available_market = market if market.is_file() else None
    panel = load_research_panel(qlib, available_market, None, day, day, 60, progress,
                                on_symbol=features.collect, include_valuation=True)
    scorer = LowVolScores(model, panel, day, allow_newer_release=True)
    index = len(panel.dates) - 1
    features.select(day, index, panel.symbols)
    frame = features.day(day)
    settings = Strategy(volume_count=RECALL_SIZE)
    connection = closing(open_readonly(market)) if available_market else nullcontext(None)
    with connection as db:
        if db:
            db.execute("BEGIN")
        reference = MarketSnapshot(db) if db else EmptyReference()
        holdings = reference.holdings(day) if db else {}
        holdings = {code: shares for code, shares in holdings.items() if shares > 0}
        found, industries = Recall(reference, settings).candidates(day, frame, {})
        liquidity = frame[frame["amount"] >= settings.min_amount].sort_values(
            "amount", ascending=False, kind="stable").index[:RECALL_SIZE]
        pool, _ = capped_recall(found, industries, liquidity, holdings)
    if not pool:
        raise ValueError(f"{day} 没有可参与评分的股票。")
    scorer.add(frame, index, pool)
    choices = portfolio_choices()
    fields = set().union(*(weights for _, weights in choices.values()))
    ranker = GridRanks(frame, fields)
    names = Explorer(market, qlib).security_names()
    strategies = {}
    for key, (label, weights) in choices.items():
        scored = ranker.rank(pool, weights, include_components=True).dropna(subset=["rank"])
        top = scored.head(SHOW_COUNT).set_index("code")
        top["name"] = [names.get(code, "") for code in top.index]
        top["close"] = frame["close"].reindex(top.index).round(2)
        top["change"] = frame["return_1"].reindex(top.index)
        values = json.loads(top.reset_index().to_json(orient="records", force_ascii=False))
        strategies[key] = {"label": label, "weights": weights, "scored_count": len(scored), "rows": values}
    report = {"day": day, "source": panel.source["source"], "release": panel.source["release"],
              "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "candidate_count": len(pool), "holding_count": len(holdings),
              "model": model.name, "validated_through": scorer.metadata["validated_through"],
              "default_strategy": DEFAULT_STRATEGY, "strategies": strategies}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, allow_nan=False)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return report


def sync_reference(base, trade_db, extracted, market, model, output, progress, *, release_tag=None, metadata=None):
    headers = {"User-Agent": "ashare-quant"}
    if metadata is not None:
        release = json.loads(metadata.read_text(encoding="utf-8"))
    else:
        locator = "tags/" + date_value(release_tag) if release_tag else "latest"
        request = Request(RELEASE_API + locator, headers={**headers, "Accept": "application/vnd.github+json"})
        try:
            with urlopen(request, timeout=30) as response:
                release = json.load(response)
        except HTTPError as exc:
            if exc.code == 429 or (exc.code == 403 and exc.headers.get("X-RateLimit-Remaining") == "0"):
                raise ValueError("GitHub API 暂时限流。可通过 --release-metadata 使用已保存的发布元数据和配套增量包。") from None
            raise
    tag = date_value(release["tag_name"])
    if release_tag and tag != release_tag:
        raise ValueError("GitHub 发布元数据与所选标签不一致。")
    local = trade_db if trade_db.is_file() else base
    with closing(open_readonly(local)) as db:
        current = Explorer.qlib_status(db)["release"]
    if tag > current:
        assets = {asset["name"]: asset for asset in release["assets"]}
        if not all(name in assets for name in DELTA_ASSETS):
            raise ValueError(f"{tag} 未发布配对增量，交易参考保留上一次已校验的排名。")
        with tempfile.TemporaryDirectory(prefix="ashare-release-") as scratch:
            paths, checksums = [], []
            for name in DELTA_ASSETS:
                asset = assets[name]
                download = f"https://github.com/xu-duqing/investment_data/releases/download/{tag}/{name}"
                checksum = asset.get("digest") or ""
                if asset["browser_download_url"] != download or not checksum.startswith("sha256:"):
                    raise ValueError(f"{tag} 的增量地址或校验信息无效。")
                saved = metadata.with_name(name) if metadata is not None else None
                if saved is not None and saved.is_file():
                    path = saved
                else:
                    path = Path(scratch) / name
                    progress(f"正在下载 {name}。")
                    with urlopen(Request(download, headers=headers), timeout=45) as response, path.open("wb") as target:
                        copyfileobj(response, target, length=1024 * 1024)
                if path.stat().st_size != asset["size"]:
                    raise ValueError(f"{name} 下载不完整。")
                paths.append(path)
                checksums.append(checksum.removeprefix("sha256:"))
            apply_delta(base, trade_db, extracted, *paths, *checksums, progress)
        local = trade_db
    else:
        progress(f"本地 {current} 已覆盖 GitHub 所选发布 {tag}。")
    return score_reference(local, market, model, output, progress)
