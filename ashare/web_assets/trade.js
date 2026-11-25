"use strict";

const element = id => document.getElementById(id);
const strategy = element("strategy");
const routeLabels = {industry:"热门行业", news:"新闻热点", volume:"量能异动", holdings:"已有持仓", liquidity:"成交额补齐"};
const number = value => value.toLocaleString("zh-CN");
const fixed = value => value == null ? "—" : value.toFixed(2);
let snapshot;

function render() {
  const selected = snapshot.strategies[strategy.value];
  const factor = Object.keys(selected.weights).find(key => key !== "low_volatility");
  element("factor-head").hidden = !factor;
  element("pool").textContent = `召回 ${number(snapshot.candidate_count)} 只 · 有分数 ${number(selected.scored_count)} 只 · 召回持仓 ${number(snapshot.holding_count)} 只`;
  element("scoring-note").textContent = factor
    ? "综合分按候选股的低波分和增强分加权，分值为 0–100。缺少增强分时沿用低波分。"
    : "低波分为候选股中近 20 日波动率的反向分位，分值为 0–100。";
  const data = selected.rows.map(entry => {
    const row = document.createElement("tr");
    const change = entry.change == null ? "—" : `${entry.change > 0 ? "+" : ""}${fixed(entry.change * 100)}%`;
    const routes = entry.routes.split("|").map(route => routeLabels[route] ?? route).join("、");
    const values = [entry.rank, entry.code, entry.name || "—", fixed(entry.score), fixed(entry.low_volatility),
      ...(factor ? [fixed(entry[factor])] : []), fixed(entry.close), change, routes];
    for (const value of values) row.insertCell().textContent = value;
    row.cells[3].className = "trade-score";
    return row;
  });
  element("ranking").replaceChildren(...data);
  element("trade-results").hidden = false;
}

async function load() {
  const refresh = element("refresh-trade");
  const previous = strategy.value;
  refresh.disabled = true;
  element("alert").hidden = true;
  try {
    const response = await fetch("/api/trade-reference", {cache:"no-store"});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "无法读取收盘排名。");
    snapshot = data;
    const keys = [data.default_strategy, ...Object.keys(data.strategies).filter(key => key !== data.default_strategy)];
    strategy.replaceChildren(...keys.map(key => new Option(data.strategies[key].label, key)));
    strategy.value = Object.hasOwn(data.strategies, previous) ? previous : data.default_strategy;
    element("reference-day").textContent = `${data.day} 收盘 · Qlib ${data.release} · 供下一交易日参考`;
    element("trade-source").href = "https://github.com/xu-duqing/investment_data/releases/tag/" + encodeURIComponent(data.release);
    strategy.disabled = false;
    render();
  } catch (error) {
    if (!snapshot) element("reference-day").textContent = "暂无收盘排名";
    element("alert").textContent = error.message;
    element("alert").hidden = false;
  } finally {
    refresh.disabled = false;
  }
}

strategy.addEventListener("change", render);
element("refresh-trade").addEventListener("click", load);
load();
