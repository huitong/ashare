"use strict";

const el = id => document.getElementById(id);
const states = {loading:"读取数据", preparing:"计算基线", training:"训练中", validating:"验证中",
                testing:"测试中", finished:"已完成", failed:"训练失败"};
const models = {model:"6 层模型", markov1:"一阶 Markov", markov3:"三阶 Markov"};
const stageNames = {train:"训练", validation:"验证", test:"测试"};
const coverageNames = {
  actual_st_status_rows:"有确证 ST 状态", inferred_st_rows:"疑似 5% 限幅", actual_limit_rows:"官方涨跌停价",
  estimated_limit_rows:"推算涨跌停价", close_limit_up_rows:"收盘涨停", opening_blocked_rows:"开盘达到上限",
  industry_relative_rows:"有历史行业标签", carried_cap_rows:"市值短期沿用",
};
const integers = new Intl.NumberFormat("zh-CN");
let detail = null;

const fmt = (v, n = 4) => Number.isFinite(v) ? v.toFixed(n) : "—";
const count = v => Number.isFinite(v) ? integers.format(v) : "—";
const text = (id, value) => {el(id).textContent = value;};
const seconds = v => Number.isFinite(v) ? v.toFixed(2) + " 秒" : "—";

function add(node, tag, value) {
  const item = document.createElement(tag);
  if (value !== undefined) item.textContent = value;
  node.append(item);
  return item;
}

function table(id, columns, rows, best = []) {
  const root = el(id);
  root.replaceChildren();
  if (rows.length === 0) {
    add(root, "p", "等待训练记录。").className = "empty";
    return;
  }
  const view = add(root, "table");
  const head = add(add(view, "thead"), "tr");
  columns.forEach(c => { add(head, "th", c).scope = "col"; });
  const body = add(view, "tbody");
  rows.forEach((row, i) => {
    const record = add(body, "tr");
    row.forEach((v, j) => {
      const cell = add(record, "td", v);
      if (best[j] === i) cell.className = "best";
    });
  });
}

async function api(url) {
  const response = await fetch(url, {cache:"no-store"});
  if (!response.ok) throw new Error("无法读取训练记录");
  return response.json();
}

function chart(id, values, color) {
  const canvas = el(id);
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (!width) return;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  ctx.font = "11px system-ui";
  ctx.fillStyle = "#506770";
  const good = values.filter(v => Number.isFinite(v.y));
  if (good.length === 0) {
    ctx.fillText("等待首轮验证", 18, 31);
    return;
  }
  const left = 61, right = width - 17, top = 15, bottom = height - 20;
  const low = Math.min(...good.map(v => v.y)), high = Math.max(...good.map(v => v.y));
  const spread = Math.max(high-low, .00001);
  const y = v => bottom - .1 * (bottom-top) - .8 * (v-low) / spread * (bottom-top);
  const x = v => good.length === 1 ? (left+right)/2 : left + (right-left) * (v-good[0].e)/(good.at(-1).e-good[0].e);
  for (const value of [...new Set([low, (low+high)/2, high])]) {
    ctx.fillStyle = "#d3e5e1"; ctx.fillRect(left, y(value), right-left, 1);
    ctx.fillStyle = "#506770"; ctx.fillText(fmt(value, 4), 2, y(value)+4);
  }
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  good.forEach((v,i) => i ? ctx.lineTo(x(v.e),y(v.y)) : ctx.moveTo(x(v.e),y(v.y)));
  ctx.stroke();
  for (const v of good) {
    ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x(v.e),y(v.y),3,0,2*Math.PI); ctx.fill();
    ctx.fillStyle = "#506770"; ctx.fillText(String(v.e),x(v.e)-3,height-4);
  }
}

function drawCharts() {
  const rounds = detail?.epochs || [];
  chart("loss-chart", rounds.map(r => ({e:r.epoch,y:r.loss})), "#186d67");
  chart("nll-chart", rounds.map(r => ({e:r.epoch,y:r.validation.model.return_nll})), "#4777ad");
  chart("ic-chart", rounds.map(r => ({e:r.epoch,y:r.validation.model.rank_ic})), "#b3602b");
}

function resultTable() {
  const result = detail?.[el("result-set").value];
  text("test-caption", result
    ? count(result.sessions) + " 个交易日 · " + count(result.samples) + " 个概率预测 · " + count(result.tradable_samples) + " 个可入场排序样本"
    : "验证选定权重后生成结果。");
  const names = Object.keys(models);
  const columns = ["模型", "联合 token NLL ↓", "涨跌幅 NLL ↓", "可入场 Rank IC ↑"];
  const metrics = ["joint_nll", "return_nll", "rank_ic"];
  const best = [null, ...metrics.map((key,k) => {
    if (!result) return null;
    const scored = names.map(name => result[name]?.[key]);
    const eligible = scored.filter(Number.isFinite);
    return eligible.length ? scored.indexOf((k === 2 ? Math.max : Math.min)(...eligible)) : null;
  })];
  table("score-table", columns, result ? names.map(name => [
    models[name], ...metrics.map(metric => fmt(result[name]?.[metric])),
  ]) : [], best);
}

function batchTable() {
  const steps = detail?.steps || [];
  const picked = Number(el("batch-epoch").value);
  const rows = steps.filter(s => s.epoch === picked);
  text("batch-caption", steps.length ? "本轮已记录 " + count(rows.length) + " 个批次" : "训练开始后显示每个优化步骤。");
  table("step-table", ["批次", "样本交易日", "耗时", "总 loss", "联合 loss", "涨跌幅 loss", "排序 loss"],
    rows.map(r => [count(r.step),r.date,seconds(r.seconds),fmt(r.loss),fmt(r.joint_loss),fmt(r.return_loss),fmt(r.rank_loss)]));
}

function render(data) {
  detail = data;
  const rounds = data?.epochs || [];
  const best = rounds.find(r => r.epoch === data.best_epoch);
  text("run-message", data?.message || "尚无训练记录。");
  text("state", states[data?.status] || "—");
  text("runtime", Number.isFinite(data?.elapsed_s) ? "累计 " + seconds(data.elapsed_s) : "");
  text("vocabulary", count(data?.vocabulary));
  text("tokens-seen", data?.seen_tokens ? "训练中实际出现 " + count(data.seen_tokens) + " 类" : "");
  text("epochs", data ? rounds.length + " / " + data.config.epochs : "—");
  text("configuration", data ? data.architecture.layers + " 层 · 上下文 " + data.architecture.context + " 日" : "");
  text("best-ic", fmt(best?.validation?.model?.rank_ic));
  text("best-epoch", best ? "第 " + best.epoch + " 轮" : "");
  table("round-table", ["轮次", "训练耗时", "验证耗时", "总 loss", "联合 loss", "涨跌幅 loss", "排序 loss", "验证 IC"],
    rounds.map(r => [r.epoch,seconds(r.train_seconds),seconds(r.validation_seconds),
      fmt(r.loss),fmt(r.joint_loss),fmt(r.return_loss),fmt(r.rank_loss),fmt(r.validation.model.rank_ic)]));
  table("split-table", ["阶段","起始","结束","交易日","可用时序窗口"], Object.entries(data?.splits || {}).map(([key,s]) =>
    [stageNames[key],s.start,s.end,count(s.sessions),count(s.windows)]));
  const c = data?.coverage;
  table("type-table", ["股票类型", "训练中出现的 token", "全期记录数"],
    (data?.features?.stock_types || []).map((label,i)=>[label,count(data?.seen_tokens_by_type?.[i]),count(c?.stock_type_counts?.[i])]));
  text("source-caption", data?.source
    ? data.source.source + " / " + data.source.release + " · 官方状态缺失时使用明确区分的规则推算与同板块相对收益。"
    : "");
  table("coverage-table", ["字段","记录数"], c ? Object.entries(coverageNames).map(([k,name])=>[name,count(c[k])]) : []);
  const drop = el("batch-epoch");
  const previous = drop.value;
  drop.replaceChildren();
  const epochs = [...new Set((data?.steps || []).map(s => s.epoch))];
  for (const epoch of epochs) {
    const option = add(drop,"option", "第 " + epoch + " 轮");
    option.value = epoch;
  }
  drop.value = epochs.includes(Number(previous)) ? previous : (epochs.at(-1) || "");
  resultTable();
  batchTable();
  drawCharts();
}

let loading = false;
async function refresh() {
  if (loading) return;
  loading = true;
  const select = el("run");
  const chosen = select.value;
  try {
    const recorded = await api("/api/training");
    const active = recorded.find(r => r.run === chosen) || recorded[0];
    select.replaceChildren();
    for (const run of recorded) {
      const option = add(select, "option", run.run + " · " + (states[run.status] || run.status));
      option.value = run.run;
    }
    select.value = active?.run || "";
    render(active ? await api("/api/training?run=" + encodeURIComponent(active.run)) : null);
  } catch(error) {
    text("run-message", error.message);
  } finally {
    loading = false;
  }
}

el("run").onchange = refresh;
el("refresh").onclick = refresh;
el("result-set").onchange = resultTable;
el("batch-epoch").onchange = batchTable;
window.addEventListener("resize", drawCharts);
refresh();
window.setInterval(refresh, 3000);
