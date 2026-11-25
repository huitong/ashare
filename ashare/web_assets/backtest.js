"use strict";

const byId = id => document.getElementById(id);
const money = x => Number(x).toLocaleString("zh-CN", {style: "currency", currency: "CNY", maximumFractionDigits: 0});
const moneyCents = x => Number(x).toLocaleString("zh-CN", {style: "currency", currency: "CNY"});
const pct = x => x === null || x === "" ? "—" : (100 * Number(x)).toFixed(2) + "%";
const points = x => x === null ? "—" : (100 * x).toFixed(2) + " 个百分点";
const num = x => x === null || x === "" ? "—" : Number(x).toLocaleString("zh-CN");
const base = "/api/backtest";
const pageSize = 35;
const comparisonYears = ["2024", "2025", "2026"];
const indexColumns = [
  ["dividend_low_vol_total_reference", "红利低波全收益", "#a74062"],
  ["dividend_low_vol_price_reference", "红利低波价格", "#866c39"],
  ["csi300_reference", "沪深 300", "#526785"],
  ["star_composite_reference", "科创综指", "#ab6b29"],
  ["star50_reference", "科创 50", "#8664a1"],
];
const lowVolControlVariants = [
  ["main_100_sum50", "100% 原版低波"],
  ["mlp_35_sum50", "35% 原版低波＋65% 量价 MLP"],
  ["mlp_value_35_sum50", "35% 原版低波＋65% 量价与估值 MLP"],
];
const levelReturn = (begin, end) => {
  const from = Number(begin), to = Number(end);
  return from > 0 && end != null && end !== "" && to >= 0 && Number.isFinite(from) && Number.isFinite(to)
    ? to / from - 1 : null;
};
const terms = {
  "buy": "买入", "sell": "卖出", "estimated": "推算", "official": "实录", "hold": "保留",
  "opening_at_limit": "开盘触及涨跌停", "limit_unknown": "当日限价未知", "no_trading_bar": "缺少交易行情",
  "no_signal_day_volume": "缺少信号日成交量", "cash_liquidity_or_lot": "现金、流动性或最小手数不足",
  "insufficient_cash_for_fee": "现金不足以支付费用", "buy_candidate": "候选买入", "watch": "观察",
  "adjustment_factor_proxy": "复权因子调整", "delisted_zero_recovery": "退市零回收",
  "open": "开盘成交", "dip_buy": "−1% 限价买入", "rally_sell": "+1% 限价卖出", "take_profit": "候选止盈",
  "intraday_target_not_reached": "日内未触价", "no_comparable_previous_close": "缺少可比昨收",
  "no_intraday_range": "缺少日内高低价", "intraday_fill_at_limit": "触及不利限价或超出涨跌停范围",
  "industry": "行业", "news": "新闻", "volume": "量能异动", "holdings": "已有持仓", "liquidity": "成交额补齐",
};
const term = x => terms[x] ?? x;
const fixed = digits => x => x === "" ? "—" : Number(x).toFixed(digits);
const columns = {
  trades: [["day","成交日"],["signal_day","信号日"],["code","股票代码"],["stock_name","股票名称",x => x || "—"],
    ["side","方向",term],["rank","信号排名",num],
    ["shares","股数",num],["open","开盘价",fixed(2)],["reference_price","未计滑点参考价",fixed(3)],
    ["price","成交价",fixed(3)],["value","成交额",money],
    ["fee","费用",moneyCents],["commission","佣金",moneyCents],["transfer","过户费",moneyCents],
    ["stamp_duty","印花税",moneyCents],["slippage","滑点成本",moneyCents],["limit_source","限价来源",term],
    ["previous_rank","前一日名次",num],["rank_2day","两日排名和",num],
    ["fill_rule","成交规则",term],["previous_close","昨收",fixed(2)],["trigger_price","挂单价",fixed(2)]],
  blocked: [["day","尝试成交日"],["signal_day","信号日"],["code","股票"],["side","方向",term],
    ["wanted","股数",num],["reason","原因",term]],
  equity: [["day","交易日"],["equity","净资产",money],["cash","现金",money],["positions","持仓数",num],
    ["universe_reference","全市场等权",x => pct(Number(x) - 1)],
    ...indexColumns.map(([column, label]) => [column, label, x => pct(levelReturn(1, x))]),
    ["star_board_reference","科创板等权",x => pct(Number(x) - 1)],["stale_positions","沿用旧价持仓",num],
    ["recall_count","候选数",num],["scored_count","有分数",num]],
  decisions: [["signal_day","信号日"],["code","股票"],["rank","排名",num],["score","分数",fixed(2)],
    ["previous_rank","前一日名次",num],["rank_2day","两日排名和",num],
    ["routes","召回来源",x => x.split("|").map(term).join("、")],["action","信号",term]],
  events: [["day","交易日"],["code","股票"],["event","事件",term],["value","数值",fixed(4)]],
};
let report, equity = [], rows = [], page = 0, hover = null, ledgerVersion = 0, viewVersion = 0;
let comparisonReports = [], comparisonRuns = [], periodFrequency = "year";

function endpoint(params = {}) {
  return base + "?" + new URLSearchParams(params);
}

async function get(params) {
  const response = await fetch(endpoint(params), {cache: "no-store"});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "无法读取回测记录");
  return result;
}

function table(container, schema, entries) {
  const documentTable = document.createElement("table");
  const head = documentTable.createTHead().insertRow();
  schema.forEach(([, title]) => {
    const th = document.createElement("th");
    th.scope = "col";
    th.textContent = title;
    head.append(th);
  });
  const body = documentTable.createTBody();
  entries.forEach(entry => {
    const row = body.insertRow();
    schema.forEach(([key, , format]) => {
      row.insertCell().textContent = format ? format(entry[key]) : entry[key];
    });
  });
  if (!entries.length) {
    const cell = body.insertRow().insertCell();
    cell.colSpan = schema.length;
    cell.className = "empty";
    cell.textContent = "没有记录";
  }
  container.replaceChildren(documentTable);
}

function options(control, entries) {
  control.replaceChildren(...entries.map(([id, label]) => new Option(label, id)));
}

function setRunOptions(runs) {
  const byYear = new Map();
  const slippage = [];
  const intraday = [];
  const grid = [];
  const enhanced = [];
  const others = [];
  for (const run of runs) {
    const year = run.start.slice(0, 4);
    if (run.comparison_version === 6) {
      slippage.push([run.run, `${run.start} 至 ${run.end} · 滑点回测`]);
    } else if (run.comparison_version === 5) {
      intraday.push([run.run, `${year} 年 · 日内买卖对照`]);
    } else if (run.comparison_version === 4) {
      grid.push([run.run, `${year} 年 · 权重与卖出对照`]);
    } else if (run.comparison_version === 3) {
      enhanced.push([run.run, `${year} 年 · 低波增强对照`]);
    } else if (run.comparison_version >= 2 && run.end.startsWith(year + "-") && !byYear.has(year)) {
      byYear.set(year, run);
    } else {
      others.push(run);
    }
  }
  const years = [...byYear.entries()].sort((a, b) => b[0].localeCompare(a[0]));
  const yearly = document.createElement("optgroup");
  yearly.label = "按年度";
  options(yearly, years.map(([year, run]) => [run.run, `${year} 年 · ${run.start.slice(5)} 至 ${run.end.slice(5)}`]));
  const historical = document.createElement("optgroup");
  historical.label = "其他记录";
  options(historical, others.map(run => [run.run, `${run.start} 至 ${run.end} · ${run.run}`]));
  const experiments = document.createElement("optgroup");
  experiments.label = "低波增强回测";
  options(experiments, enhanced);
  const comparison = document.createElement("optgroup");
  comparison.label = "低波权重与卖出对照";
  options(comparison, grid);
  const execution = document.createElement("optgroup");
  execution.label = "低波策略日内买卖";
  options(execution, intraday);
  const executionCost = document.createElement("optgroup");
  executionCost.label = "滑点回测";
  options(executionCost, slippage);
  comparisonRuns = runs.filter(run => run.comparison_version === 4 || run.comparison_version === 5);
  byId("run").replaceChildren(...[executionCost, execution, comparison, experiments, yearly, historical]
    .filter(group => group.children.length));
}

function set(id, value) {
  byId(id).textContent = value;
}

function showError(error) {
  set("alert", error.message);
  byId("alert").hidden = false;
}

function dailyCorrelation(daily, referenceColumn) {
  const pairs = [];
  for (let i = 1; i < daily.length; i++) {
    const previous = daily[i - 1], current = daily[i];
    const strategy = levelReturn(previous.equity, current.equity);
    const reference = levelReturn(previous[referenceColumn], current[referenceColumn]);
    if (strategy !== null && reference !== null) pairs.push([strategy, reference]);
  }
  if (pairs.length < 2) return "—";
  const [sumStrategy, sumReference] = pairs.reduce(([a, b], [x, y]) => [a + x, b + y], [0, 0]);
  let varianceStrategy = 0, varianceReference = 0, covariance = 0;
  for (const [strategy, reference] of pairs) {
    const x = strategy - sumStrategy / pairs.length, y = reference - sumReference / pairs.length;
    varianceStrategy += x * x;
    varianceReference += y * y;
    covariance += x * y;
  }
  const denominator = Math.sqrt(varianceStrategy * varianceReference);
  return denominator ? (covariance / denominator).toFixed(2) : "—";
}

function renderComparisons() {
  if (!report) return;
  const grid = report.comparison_version === 4;
  const slippage = report.comparison_version === 6;
  const intraday = !!report.variants.main?.execution;
  const annual = grid || report.comparison_version === 5;
  byId("grid-filters").hidden = !grid;
  const weight = byId("grid-weight").value;
  const exit = byId("grid-exit").value;
  const complete = annual && comparisonYears.every((year, index) => comparisonReports[index]?.start.startsWith(year + "-"));
  const yearlyColumns = comparisonReports.flatMap(period => {
    const year = period.start.slice(0, 4);
    const label = period.end.endsWith("-12-31") || year !== comparisonYears.at(-1) ? year : `${year} 年初至今`;
    return [[`return_${year}`, `${label} 净收益`, pct], [`drawdown_${year}`, `${label} 最大回撤`, pct]];
  });
  set("comparison-basis", annual
    ? (complete
      ? (`按三段净收益几何串联后的累计收益降序排列。各年独立空仓起算。2026 年行情截至 ${comparisonReports.at(-1).end}。`
         + (intraday ? "顺位版按日 K 判断整日是否触价，成交先后需分钟行情验证。" : ""))
      : "按所选年度净收益排序。三年记录齐全后显示综合收益。")
    : slippage ? "同一期内持续持仓，展示各成交方案的累计净收益和滑点成本。"
      : "使用相同的候选规则、初始资金和成交口径，按累计净收益排序");
  const comparisons = Object.entries(report.variants)
    .map(([key, v]) => {
      const entry = {...v, type: v.factor_count === 1 ? "单因子" : "多因子",
        label: grid ? v.label.split(" · 原低波")[0] : v.label, total_cost: v.fees + (v.slippage ?? 0)};
      const annualReturns = comparisonReports.map(period => {
        const year = period.start.slice(0, 4), variant = period.variants[key];
        entry[`return_${year}`] = variant?.total_return ?? null;
        entry[`drawdown_${year}`] = variant?.max_drawdown ?? null;
        return entry[`return_${year}`];
      });
      entry.combined_return = complete && annualReturns.every(Number.isFinite)
        ? annualReturns.reduce((growth, annualReturn) => growth * (1 + annualReturn), 1) - 1 : null;
      return entry;
    })
    .sort((a, b) => complete
      ? (b.combined_return ?? -Infinity) - (a.combined_return ?? -Infinity)
      : b.total_return - a.total_return)
    .map((v, index) => ({...v, rank: v.combined_return === null ? null : index + 1}))
    .filter(v => !grid || ((!weight || v.baseline_share === Number(weight)) && (!exit || v.exit_rule === exit)));
  const typeColumn = slippage ? [["slippage_bps","单边滑点（bp）",num],["slippage","滑点成本",money]]
    : grid ? [["baseline_share","原低波权重",pct],["exit_label","卖出条件"]]
    : !intraday && report.comparison_version >= 2 ? [["type","类型"]] : [];
  const rankColumn = complete ? [["rank","三年名次",num]] : [];
  const returnColumns = annual ? [
    ...(complete ? [["combined_return","三年综合净收益",pct]] : []),
    ...yearlyColumns,
  ] : [["total_return","累计净收益",pct],["max_drawdown","最大回撤",pct]];
  const selectedYear = annual ? report.start.slice(0, 4) + " " : "";
  const turnover = annual || intraday ? [["daily_turnover", selectedYear + "日均双边换手", pct]] : [];
  const executionColumns = intraday ? [
    ["buy_fill_rate", selectedYear + "买单成交率", pct], ["sell_fill_rate", selectedYear + "卖单成交率", pct],
    ["outside_top10_buys", selectedYear + "顺位买入 11–30 名", num],
  ] : [];
  table(byId("comparisons"), [...rankColumn,["label","方案"],...typeColumn,...returnColumns,
    ["annualized_return",selectedYear + "年化收益",pct],
    ["sharpe_zero_rf",selectedYear + "夏普（零无风险利率）",x => x === null ? "—" : x.toFixed(2)],
    ...turnover,...executionColumns,["orders_filled",selectedYear + "成交笔数",num],
    ["total_cost",selectedYear + "总交易成本",money]], comparisons);
}

function renderSummary(summary) {
  const settings = report.strategy;
  set("basis", `${report.start} 至 ${report.end} · ${num(summary.sessions)} 个交易日 · Qlib 历史回放`);
  set("return", pct(summary.total_return));
  byId("return").className = summary.total_return >= 0 ? "bt-up" : "bt-down";
  set("final", `期末资产 ${money(summary.final_equity)}`);
  set("drawdown", pct(summary.max_drawdown));
  set("annual", `年化收益 ${pct(summary.annualized_return)}`);
  set("cost", money(summary.fees + (summary.slippage ?? 0)));
  const fees = summary.fee_components;
  const slippageBps = summary.slippage_bps ?? settings.slippage_bps ?? 0;
  const slipDetail = report.comparison_version === 6 ? ` · 滑点 ${money(summary.slippage)}` : "";
  set("cost-detail", fees ? `佣金 ${money(fees.commission)} · 过户 ${money(fees.transfer)} · 印花税 ${money(fees.stamp_duty)}`
    + slipDetail : `费用 ${money(summary.fees)} · 价格偏移 ${money(summary.slippage)}`);
  set("positions", summary.average_positions.toFixed(1) + " 只");
  set("orders", `买入 ${num(summary.buys)} 笔 · 卖出 ${num(summary.sells)} 笔`);
  const newReport = report.comparison_version >= 2;
  const execution = summary.execution;
  const executionLabel = execution && (execution.buy_depth || execution.take_profit) ? "日 K 模拟限价成交" : "开盘撮合";
  const executionCost = `${executionLabel} · 单边滑点 ${slippageBps} bp`;
  set("chart-summary", `初始资金 ${money(settings.initial_cash)} · ${newReport ? executionCost + "，扣除佣金、过户费和印花税" : "历史方案含价格偏移"}`);
  const grid = report.comparison_version === 4;
  renderComparisons();
  renderPeriodReturns();
  byId("benchmark-panel").hidden = !report.benchmarks;
  if (report.benchmarks) {
    const benchmarkRows = Object.entries(report.benchmarks).map(([key, benchmark]) => ({
      ...benchmark, code: benchmark.code ?? "—",
      excess: benchmark.total_return === null ? null : summary.total_return - benchmark.total_return,
      start_close: benchmark.start_close?.toFixed(2) ?? "—", end_close: benchmark.end_close?.toFixed(2) ?? "—",
      correlation: dailyCorrelation(equity, key + "_reference"),
      source: benchmark.source || "收盘点位待补",
    }));
    table(byId("benchmarks"), [["label","基准"],["code","代码"],["start_close","期初点位"],["end_close","期末点位"],
      ["total_return","同期收益",pct],["excess","策略超额",points],["correlation","日收益相关性"],["source","来源"]], benchmarkRows);
  }
  const routes = report.routes;
  const weightSum = Object.values(summary.weights).reduce((a, b) => a + b, 0);
  const oldFactorLabels = {trend: "趋势", macd: "MACD", token_transformer: "Transformer"};
  const weights = Object.entries(summary.weights).map(([key, weight]) =>
    `${report.factors?.[key]?.label ?? report.model?.models?.[key]?.label ?? oldFactorLabels[key] ?? (key === "steady_rule" ? "平稳上涨规则" : key)} ${(weight * 100 / weightSum).toFixed(1)}%`);
  const blocked = Object.entries(summary.blocked_orders).map(([reason, count]) =>
    `${term(reason)} ${num(count)} 笔`).join("、") || "无";
  const assumptions = [
    ["起算", `所选区间从 ${money(settings.initial_cash)} 空仓起算。`],
    ["候选与排序", `每天最多 ${num(settings.recall_size)} 只，含已有持仓。${report.full_recall_days}/${report.signal_days} 个信号日达到上限。各模型按截面分位加权，${weights.join("、")}。`],
    ["召回命中", `量能异动 ${num(routes.volume || 0)} 次，成交额补齐 ${num(routes.liquidity || 0)} 次，行业 ${num(routes.industry || 0)} 次，新闻 ${num(routes.news || 0)} 次。`],
    ["调仓", `收盘打分。卖出条件为${summary.exit_label ?? `跌出 Top ${settings.sell_rank}`}。${execution?.buy_depth
      ? (execution.reserve_buy_slots ? "开盘后按空位从 Top 10 固定 −1% 买单，未成交空位留到下一日。" : "从 Top 30 按名次检查 −1% 买单，未触价顺延。")
      : `开盘卖出后，从 Top ${settings.buy_rank} 补空位。`}每笔新仓预算上限为净值的 10%。`],
    ["费用", `${executionCost}。佣金 ${settings.commission_bps} bp（最低 ${settings.min_commission} 元），双边过户费各 ${settings.transfer_bps} bp，卖出印花税 ${settings.sell_tax_bps} bp。`],
    ["成交约束", `单笔股数上限为信号日成交量的 ${pct(settings.participation)}。开盘封板和缺少可核实或可推算限价时保守跳过。实际成交中 ${num(summary.limit_estimated_fills)} 笔使用推算限价。`],
    ["未成交与估值", `${blocked}。期末 ${summary.ending_stale} 只沿用最后可用价格。未排名持仓等有可用分数后再判断是否出局，期末持仓按收盘价估值。`],
    ["基准", "指数用收盘点位。红利低波全收益计入分红再投资，价格指数反映价格变动。科创综指编制口径包含分红。等权参考按本地有相邻交易日有效价格的股票计算。"],
  ];
  if (execution) {
    assumptions.splice(4, 0, ["日内撮合", "以昨收为基准，日 K 高低点判断是否触价。盘中按挂单价成交，开盘已越过挂单价时按开盘价。开盘后的买入只用开盘时可用的现金和空位。顺位回补使用整天是否触价，单靠日 K 不能还原成交先后。成交率为成交笔数除以实际尝试的候选单数。"]);
  }
  if (slippageBps > 0) {
    assumptions.push(["滑点", "买入价向上、卖出价向下偏移，受涨跌停价约束。日内限价单的成交价受挂单价约束。净收益已经包含滑点。"]);
  }
  if (report.model?.models) {
    const m = report.model;
    byId("enhanced-guide").hidden = false;
    set("enhanced-split", `${m.start.slice(0,4)}–${m.selection_train_end.slice(0,4)} 年拟合 · ${m.validated_through.slice(0,4)} 年验证后合并重训 · 年度回测共用固定权重`);
    set("enhanced-mix", grid ? "对照采用 35%、50%、65%、100% 原低波权重。增强分缺失时用原低波排名。两日规则要求相邻交易日均有有效名次，25＋26＝51 会触发，和恰好等于 50 时继续持有。日均双边换手为每日买卖成交额之和除以上一日净值后取平均。"
      : execution ? "日内对照统一使用 35% 原版低波与 65% 量价＋估值 MLP。"
        : "增强组合保留 65% 原版低波，使用 35% 模型或规则排序。");
    const records = [{label:"原版 20 日低波", model_rank_ic:m.baseline_rank_ic, blend_rank_ic:m.baseline_rank_ic},
      ...Object.values(m.models)];
    table(byId("enhanced-models"), [["label","模型"],["model_rank_ic","验证分数 Rank IC",x => x.toFixed(4)],
      ["blend_rank_ic","65/35 混合 Rank IC",x => x.toFixed(4)]], records);
  } else {
    byId("enhanced-guide").hidden = true;
  }
  if (report.model?.validated_through && !report.model.models) {
    assumptions.splice(1, 0, ["Transformer", `模型验证截止日为 ${report.model.validated_through}。`]);
  }
  byId("assumptions").replaceChildren(...assumptions.flatMap(([label, value]) => {
    const dt = document.createElement("dt"), dd = document.createElement("dd");
    dt.textContent = label;
    dd.textContent = value;
    return [dt, dd];
  }));
}

function periodReturnRows(daily, frequency) {
  const periods = [];
  for (let i = 0; i < daily.length; i++) {
    const current = daily[i], year = current.day.slice(0, 4), month = Number(current.day.slice(5, 7));
    const key = frequency === "quarter" ? `${year} Q${Math.ceil(month / 3)}` : year;
    if (periods.at(-1)?.key !== key) periods.push({key, begin: daily[Math.max(0, i - 1)]});
    periods.at(-1).end = current;
  }
  return periods.map(({key, begin, end}) => {
    const strategy = levelReturn(begin.equity, end.equity);
    const row = {key, period: `${begin.day} 至 ${end.day}`, strategy};
    for (const [column] of indexColumns) {
      row[column] = levelReturn(begin[column], end[column]);
      row[column + "_excess"] = strategy === null || row[column] === null ? null : strategy - row[column];
    }
    return row;
  });
}

function renderPeriodReturns() {
  for (const frequency of ["year", "quarter"]) {
    byId("period-" + frequency).setAttribute("aria-pressed", String(periodFrequency === frequency));
  }
  const indexSchema = indexColumns
    .filter(([column]) => equity.some(row => levelReturn(1, row[column]) !== null))
    .flatMap(([column, label]) =>
      [[column, label + " 收益", pct], [column + "_excess", "超额 vs " + label, points]]);
  table(byId("period-returns"), [
    ["key", periodFrequency === "quarter" ? "季度" : "年度"], ["period","基准收盘日至期末"],
    ["strategy","策略净收益",pct], ...indexSchema,
  ], periodReturnRows(equity, periodFrequency));
}

async function loadLowVolControls(runs) {
  const peers = comparisonYears.map(year => runs.find(run =>
    run.comparison_version === 4 && run.start.startsWith(year + "-") && run.end.startsWith(year + "-")));
  if (!peers.every(Boolean)) return;
  const reports = await Promise.all(peers.map(peer => get({run: peer.run})));
  if (!reports.every(data => lowVolControlVariants.every(([key]) => data.variants[key]))) return;
  const entries = reports.map(data => {
    const benchmark = data.benchmarks?.dividend_low_vol_total?.total_return ?? null;
    const returns = Object.fromEntries(lowVolControlVariants.map(([key]) =>
      [key, data.variants[key].total_return]));
    return {period: `${data.start} 至 ${data.end}`, benchmark, ...returns,
      valuation_delta: returns.mlp_value_35_sum50 - returns.mlp_35_sum50};
  });
  table(byId("low-vol-controls"), [
    ["period", "回测区间"], ["benchmark", "红利低波全收益", pct],
    ...lowVolControlVariants.map(([key, label]) => [key, label, pct]),
    ["valuation_delta", "加入估值特征后收益差", points],
  ], entries);
  byId("low-vol-control-panel").hidden = false;
}

function chart(canvas, lines, marker, percent, shade = false) {
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height || !lines[0].values.length) return;
  const ratio = window.devicePixelRatio || 1;
  const w = rect.width, h = rect.height;
  canvas.width = Math.round(w * ratio);
  canvas.height = Math.round(h * ratio);
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  const box = {l: 54, r: w - 12, t: 12, b: h - 24};
  const values = lines.flatMap(line => line.values);
  const low = Math.min(0, ...values), high = Math.max(0, ...values);
  const span = Math.max(high - low, .01);
  const n = lines[0].values.length;
  const x = i => box.l + (box.r - box.l) * i / Math.max(1, n - 1);
  const y = value => box.b - (box.b - box.t) * (value - low) / span;
  ctx.font = "11px system-ui";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let i = 0; i <= 4; i++) {
    const value = low + span * i / 4;
    ctx.strokeStyle = "#dbe5e7";
    ctx.beginPath();
    ctx.moveTo(box.l, y(value));
    ctx.lineTo(box.r, y(value));
    ctx.stroke();
    ctx.fillStyle = "#506770";
    ctx.fillText(percent(value), box.l - 7, y(value));
  }
  for (const [index, textAlign] of [[0, "left"], [n - 1, "right"]]) {
    ctx.fillStyle = "#506770";
    ctx.textAlign = textAlign;
    ctx.fillText(equity[index].day, x(index), h - 10);
  }
  for (const line of lines) {
    ctx.beginPath();
    line.values.forEach((value, i) => i ? ctx.lineTo(x(i), y(value)) : ctx.moveTo(x(i), y(value)));
    if (shade) {
      ctx.lineTo(x(n - 1), y(0));
      ctx.lineTo(x(0), y(0));
      ctx.closePath();
      ctx.fillStyle = "#f4dddd";
      ctx.fill();
    }
    ctx.strokeStyle = line.color;
    ctx.setLineDash(line.dash ? [4, 4] : []);
    ctx.lineWidth = 1.7;
    ctx.stroke();
    ctx.setLineDash([]);
    if (marker !== null) {
      ctx.strokeStyle = "#8ca0a3";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x(marker), box.t);
      ctx.lineTo(x(marker), box.b);
      ctx.stroke();
      ctx.fillStyle = line.color;
      ctx.beginPath();
      ctx.arc(x(marker), y(line.values[marker]), 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  canvas.plot = box;
}

function renderChart() {
  if (!equity.length) return;
  const capital = report.strategy.initial_cash;
  const returns = equity.map(row => Number(row.equity) / capital - 1);
  const references = indexColumns.filter(([column]) =>
    equity.every(row => levelReturn(1, row[column]) !== null));
  if (!references.length) references.push(["universe_reference", "全市场等权", "#526785"]);
  const series = references.map(([column, label, color]) =>
    ({values: equity.map(row => Number(row[column]) - 1), label, color, dash: true}));
  byId("chart-legend").replaceChildren(...[{label: "所选方案", color: "#14665d"}, ...series]
    .map(({label, color, dash}) => {
    const span = document.createElement("span");
    span.textContent = label;
    span.style.setProperty("--legend-color", color);
    if (dash) span.className = "bt-index-key";
    return span;
  }));
  let peak = capital;
  const drawdown = equity.map(row => Number(row.equity) / (peak = Math.max(peak, Number(row.equity))) - 1);
  chart(byId("equity-chart"), [...series, {values: returns, color: "#14665d"}], hover, pct);
  chart(byId("drawdown-chart"), [{values: drawdown, color: "#b94337"}], hover, pct, true);
  const index = hover ?? equity.length - 1;
  set("chart-hover", [equity[index].day, `方案 ${pct(returns[index])}`,
    ...series.map(line => `${line.label} ${pct(line.values[index])}`), `回撤 ${pct(drawdown[index])}`].join(" · "));
}

function renderLedger() {
  const schema = columns[byId("table").value].filter(([key]) => !rows.length || key in rows[0]);
  table(byId("ledger"), schema, rows.slice(page * pageSize, (page + 1) * pageSize));
  const pager = byId("pager");
  const count = Math.ceil(rows.length / pageSize);
  if (count < 2) {
    pager.replaceChildren();
    return;
  }
  const previous = document.createElement("button"), next = document.createElement("button");
  previous.textContent = "上一页";
  next.textContent = "下一页";
  previous.disabled = page === 0;
  next.disabled = page + 1 >= count;
  previous.addEventListener("click", () => { page--; renderLedger(); });
  next.addEventListener("click", () => { page++; renderLedger(); });
  pager.replaceChildren(previous, `第 ${page + 1} / ${count} 页`, next);
}

async function loadLedger() {
  if (!report) return;
  const version = ++ledgerVersion;
  const params = {run: report.run, variant: byId("variant").value, table: byId("table").value};
  byId("download").hidden = true;
  set("ledger-caption", "正在读取明细");
  const response = await get(params);
  if (version !== ledgerVersion) return;
  page = 0;
  rows = response.rows.reverse();
  set("ledger-caption", `共 ${num(rows.length)} 条，最近日期优先${params.table === "trades" ? " · 名称取自证券名录" : ""}`);
  byId("download").href = endpoint({...params, download: 1});
  byId("download").hidden = false;
  renderLedger();
}

async function loadVariant() {
  const version = ++viewVersion;
  const variant = byId("variant").value;
  const run = report.run;
  const response = await get({run, variant, table: "equity"});
  if (version !== viewVersion) return;
  equity = response.rows;
  hover = null;
  renderSummary(report.variants[variant]);
  byId("result").hidden = false;
  renderChart();
  await loadLedger();
}

async function loadRun() {
  const version = ++viewVersion;
  ++ledgerVersion;
  const previousVariant = byId("variant").value;
  byId("variant").disabled = true;
  const run = byId("run").value;
  const selected = comparisonRuns.find(item => item.run === run);
  const peers = selected ? comparisonYears.filter(year => !selected.start.startsWith(year + "-"))
    .map(year => comparisonRuns.find(item => item.comparison_version === selected.comparison_version
      && item.start.startsWith(year + "-") && item.end.startsWith(year + "-")))
    .filter(Boolean) : [];
  const [data, ...comparative] = await Promise.all([get({run}), ...peers.map(peer => get({run: peer.run}).catch(() => null))]);
  if (version !== viewVersion) return;
  report = data;
  comparisonReports = selected ? [data, ...comparative.filter(Boolean)].sort((a, b) => a.start.localeCompare(b.start)) : [];
  options(byId("variant"), Object.entries(report.variants).map(([key, value]) => [key, value.label]));
  if (Object.hasOwn(report.variants, previousVariant)) byId("variant").value = previousVariant;
  else if (report.default_variant) byId("variant").value = report.default_variant;
  byId("variant").disabled = false;
  byId("alert").hidden = true;
  await loadVariant();
}

for (const id of ["equity-chart", "drawdown-chart"]) {
  const canvas = byId(id);
  canvas.addEventListener("pointermove", event => {
    if (!canvas.plot || !equity.length) return;
    const p = (event.clientX - canvas.getBoundingClientRect().left - canvas.plot.l) / (canvas.plot.r - canvas.plot.l);
    const index = Math.max(0, Math.min(equity.length - 1, Math.round(p * (equity.length - 1))));
    if (hover !== index) { hover = index; renderChart(); }
  });
  canvas.addEventListener("pointerleave", () => { hover = null; renderChart(); });
  canvas.addEventListener("keydown", event => {
    if (!["ArrowLeft", "ArrowRight"].includes(event.key) || !equity.length) return;
    event.preventDefault();
    hover = Math.max(0, Math.min(equity.length - 1, (hover ?? equity.length - 1) + (event.key === "ArrowLeft" ? -1 : 1)));
    renderChart();
  });
}
window.addEventListener("resize", renderChart);
byId("run").addEventListener("change", () => loadRun().catch(showError));
byId("variant").addEventListener("change", () => loadVariant().catch(showError));
byId("table").addEventListener("change", () => loadLedger().catch(showError));
byId("grid-weight").addEventListener("change", renderComparisons);
byId("grid-exit").addEventListener("change", renderComparisons);
for (const frequency of ["year", "quarter"]) {
  byId("period-" + frequency).addEventListener("click", () => {
    periodFrequency = frequency;
    renderPeriodReturns();
  });
}
get().then(async runs => {
  if (!runs.length) {
    set("basis", "还没有组合回测记录");
    return;
  }
  setRunOptions(runs);
  await Promise.all([loadRun(), loadLowVolControls(runs).catch(() => {})]);
}).catch(showError);
