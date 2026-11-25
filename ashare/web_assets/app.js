"use strict";

const el = id => document.getElementById(id);
const integer = new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 0 });
const numeric = new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 5 });
const state = { catalog: [], active: null, page: 1, history: [], stockPage: 1, chart: [], generation: 0 };
const fieldLabel = {
  open: "开盘", high: "最高", low: "最低", close: "收盘", volume: "成交量", amount: "成交额",
  market_cap: "总市值（亿元）", total_mv: "总市值（万元）", circ_mv: "流通市值（万元）",
  total_share: "总股本", float_share: "流通股本", free_share: "自由流通股本",
  prev_close: "昨收", adj: "复权因子", factor: "归一化因子", change: "涨跌幅",
  vwap: "均价", turnover_rate: "换手率", turnover_rate_f: "自由流通换手率", limit_up: "涨停价",
  limit_down: "跌停价", limit_known: "已知限价",
};

function asText(value) {
  return value == null ? "—" : String(value);
}

function num(value, format = numeric) {
  return typeof value === "number" ? format.format(value) : "—";
}

function rate(top, bottom) {
  return bottom ? (100 * top / bottom).toFixed(2) + "%" : "—";
}

function money(value) {
  return value == null ? "—" : num(value / 1e8) + " 亿";
}

function add(parent, tag, value) {
  const node = document.createElement(tag);
  if (value !== undefined) node.textContent = value;
  parent.append(node);
  return node;
}

function status(message = "") {
  el("alert").hidden = !message;
  el("alert").textContent = message;
}

function query(params) {
  return new URLSearchParams({source: state.active.kind, ...params});
}

async function api(path, params) {
  const response = await fetch(path + (params ? "?" + query(params) : ""));
  const data = await response.json();
  if (!response.ok) throw new Error(data.error);
  return data;
}

function switchView(view) {
  for (const section of ["market", "stock", "quality"]) {
    el(section).hidden = section !== view;
    el("tab-" + section).setAttribute("aria-pressed", String(section === view));
  }
  if (view === "stock") drawChart();
}

function table(id, headers, rows, empty = "没有符合条件的记录。") {
  const node = el(id);
  node.replaceChildren();
  if (!rows.length) {
    add(node, "p", empty).className = "empty";
    return;
  }
  const element = add(node, "table");
  const head = add(add(element, "thead"), "tr");
  for (const label of headers) add(head, "th", label).scope = "col";
  const body = add(element, "tbody");
  for (const row of rows) {
    const tr = add(body, "tr");
    for (const value of row) {
      const td = add(tr, "td");
      if (value instanceof Node) td.append(value);
      else td.textContent = asText(value);
    }
  }
}

function pager(id, current, pages, onPage) {
  const container = el(id);
  container.replaceChildren();
  if (pages < 2) return;
  const previous = add(container, "button", "上一页");
  previous.disabled = current <= 1;
  previous.onclick = () => onPage(current - 1);
  add(container, "span", current + " / " + pages);
  const next = add(container, "button", "下一页");
  next.disabled = current >= pages;
  next.onclick = () => onPage(current + 1);
}

function showSummary() {
  const s = state.active?.stats;
  el("basis").textContent = state.active?.kind === "qlib"
    ? "Qlib 复权归一化行情 · 成交量沿用发布口径 · 总市值折算为人民币"
    : "本地原始行情 · 价格为人民币 · 成交量为股";
  el("coverage").textContent = s?.source ? "来源 " + s.source + (s.release ? " / " + s.release : "") : "";
  el("m-symbols").textContent = num(s?.price_symbols ?? s?.symbols, integer);
  el("m-prices").textContent = num(s?.price_rows, integer);
  el("m-days").textContent = num(s?.observed_sessions, integer);
  el("m-cap").textContent = rate(s?.market_cap_rows, s?.price_rows);
  el("m-exchanges").textContent = (s?.markets || []).map(row => row[0] + " " + integer.format(row[1])).join(" · ");
  el("m-rows").textContent = s ? "含指标的证券日记录 " + integer.format(s.rows) : "";
  el("m-range").textContent = s?.first_day ? s.first_day + " 至 " + s.last_day : "";
  el("quality-caption").textContent = s?.fields
    ? "OHLC 异常 " + integer.format(s.invalid_ohlc_rows) + " 行 · 字段覆盖分母为所有证券日记录"
    : "Qlib 全量导入后提供年份及字段覆盖统计。";
  table("year-table", ["年份", "交易日", "有收盘价", "市值覆盖"], (state.active?.years || []).map(y => [
    y.year, num(y.sessions, integer), num(y.price_rows, integer), rate(y.market_cap_rows, y.price_rows),
  ]));
  table("field-table", ["字段", "有效值", "覆盖率"], Object.entries(s?.fields || {}).map(([name, count]) => [
    (fieldLabel[name] || name) + " · " + name, num(count, integer), rate(count, s.rows),
  ]));
}

function stockLink(item) {
  const cell = document.createElement("span");
  const button = add(cell, "button", item.code);
  button.onclick = () => {
    el("ticker").value = item.code;
    switchView("stock");
    guarded(() => loadHistory(item.symbol));
  };
  if (item.name) add(cell, "small", item.name);
  return cell;
}

async function loadDaily(page) {
  const request = ++state.generation;
  const day = el("day").value;
  if (!state.active || !day) {
    el("daily-caption").textContent = "尚无可浏览的行情。";
    table("daily-table", [], []);
    el("daily-pager").replaceChildren();
    return;
  }
  const data = await api("/api/daily", {day, sort: el("sort").value, page});
  if (request !== state.generation) return;
  state.page = page;
  el("daily-caption").textContent = data.day + " · " + num(data.count, integer) + " 条行情 · 市值覆盖 " + rate(data.cap_count, data.count);
  table("daily-table", ["证券", "开盘", "最高", "最低", "收盘", "成交量", "总市值"], data.rows.map(item => [
    stockLink(item), num(item.open), num(item.high), num(item.low), num(item.close), num(item.volume, integer), money(item.market_cap),
  ]));
  pager("daily-pager", page, Math.ceil(data.count / data.page_size), p => guarded(() => loadDaily(p)));
}

function previousYear(day) {
  if (!day) return "";
  const [y, m, d] = day.split("-").map(Number);
  const cutoff = new Date(Date.UTC(y - 1, m, 0)).getUTCDate();
  return String(y - 1) + day.slice(4, 8) + String(Math.min(d, cutoff)).padStart(2, "0");
}

function setSource(kind) {
  state.active = state.catalog.find(item => item.kind === kind) || null;
  const end = state.active?.stats.last_day || "";
  el("day").value = end;
  el("end").value = end;
  el("start").value = previousYear(end);
  el("ticker").value = "";
  el("ticker-options").replaceChildren();
  el("stock-heading").textContent = "个股历史";
  el("stock-caption").textContent = "";
  el("chart-wrap").hidden = el("export").hidden = el("stock-membership").hidden = true;
  el("stock-pager").replaceChildren();
  state.history = [];
  state.chart = [];
  table("stock-table", [], [], "输入证券代码开始查询。");
  showSummary();
}

async function refresh() {
  const current = state.active?.kind;
  state.catalog = await api("/api/catalog");
  const select = el("source");
  select.replaceChildren();
  for (const item of state.catalog) {
    const option = add(select, "option", item.kind === "qlib" ? "Qlib 历史档案" : "本地原始行情");
    option.value = item.kind;
  }
  if (!state.catalog.length) add(select, "option", "尚未入库");
  const best = state.catalog.find(item => item.stats.price_rows && item.kind === current)
    || state.catalog.find(item => item.stats.price_rows) || state.catalog[0];
  select.value = best?.kind || "";
  setSource(best?.kind);
  select.disabled = state.catalog.length < 2;
  await loadDaily(1);
  if (!best?.stats.price_rows) status("尚未导入可浏览的行情。导入 Qlib 双包后点击「刷新」。");
}

function renderHistory(page) {
  state.stockPage = page;
  const count = 50;
  const fields = ["open", "high", "low", "close", "volume", "market_cap"];
  const first = state.history[0];
  if (first) fields.push(...Object.keys(first).filter(key => !["symbol", "day", ...fields].includes(key)));
  const rows = state.history.slice((page - 1) * count, page * count);
  table("stock-table", ["交易日", ...fields.map(key => fieldLabel[key] || key)], rows.map(item => [
    item.day, ...fields.map(key => key === "market_cap" ? money(item[key]) : num(item[key])),
  ]));
  pager("stock-pager", page, Math.ceil(state.history.length / count), renderHistory);
}

async function loadHistory(symbol) {
  if (!state.active) throw new Error("尚未导入行情。");
  const request = ++state.generation;
  const filters = {symbol, start: el("start").value, end: el("end").value};
  const result = await api("/api/history", filters);
  if (request !== state.generation) return;
  state.history = result.rows.slice().reverse();
  el("stock-heading").textContent = result.symbol.code + (result.symbol.name ? " · " + result.symbol.name : "");
  el("stock-caption").textContent = num(result.rows.length, integer) + " 条记录";
  if (result.rows.length) el("stock-caption").textContent += " · " + result.rows[0].day + " 至 " + result.rows.at(-1).day;
  el("export").hidden = !result.rows.length;
  el("export").href = "/api/export?" + query(filters);
  el("stock-membership").hidden = !result.universes.length;
  table("membership-table", ["指数", "开始", "结束"], result.universes.map(item => [item.universe, item.start, item.end]));
  state.chart = result.rows.filter(row => ["open","high","low","close"].every(key =>
    Number.isFinite(row[key]) && row[key] > 0)).slice(-250);
  el("chart-wrap").hidden = !state.chart.length;
  drawChart();
  renderHistory(1);
}

function drawChart(highlight = -1) {
  const canvas = el("chart");
  const rows = state.chart;
  const width = canvas.clientWidth;
  if (!width || !rows.length) return;
  const height = canvas.clientHeight;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  const left = 58, right = width - 6, top = 15, bottom = height - 82;
  const min = Math.min(...rows.map(row => row.low)), max = Math.max(...rows.map(row => row.high));
  const maxVolume = Math.max(1, ...rows.map(row => row.volume || 0));
  const y = value => bottom - (value - min) / (max - min || 1) * (bottom - top);
  const step = (right - left) / rows.length;
  ctx.font = "11px system-ui";
  ctx.textBaseline = "middle";
  for (let index = 0; index < 5; index++) {
    const value = min + (max - min) * index / 4;
    const line = y(value);
    ctx.fillStyle = "#526b70";
    ctx.fillText(num(value), 2, line);
    ctx.fillStyle = "#deeaeb";
    ctx.fillRect(left, line, right - left, 1);
  }
  const barWidth = Math.max(1, Math.min(10, step * .62));
  rows.forEach((row, index) => {
    const x = left + step * (index + .5);
    const color = row.close >= row.open ? "#b94337" : "#177454";
    ctx.fillStyle = color;
    ctx.fillRect(x, y(row.high), 1, Math.max(1, y(row.low) - y(row.high)));
    ctx.fillRect(x - barWidth / 2, Math.min(y(row.open), y(row.close)), barWidth, Math.max(1, Math.abs(y(row.open) - y(row.close))));
    const volumeHeight = Math.max(0, (row.volume || 0) / maxVolume * 48);
    ctx.globalAlpha = .68;
    ctx.fillRect(x - barWidth / 2, height - 20 - volumeHeight, barWidth, volumeHeight);
    ctx.globalAlpha = 1;
  });
  ctx.fillStyle = "#526b70";
  ctx.fillText(rows[0].day, left, height - 5);
  ctx.textAlign = "right";
  ctx.fillText(rows.at(-1).day, right, height - 5);
  const selected = rows[highlight];
  if (selected) {
    ctx.fillStyle = "#526b70";
    ctx.fillRect(left + step * (highlight + .5), top, 1, height - top - 19);
  }
  el("chart-date").textContent = selected
    ? selected.day + " · 开 " + num(selected.open) + " · 收 " + num(selected.close) + " · 量 " + num(selected.volume, integer)
    : "图示最近 " + rows.length + " 根有效日 K";
}

async function guarded(action) {
  status();
  try { await action(); }
  catch (error) { status(error.message); }
}

el("source").onchange = () => guarded(async () => {setSource(el("source").value); await loadDaily(1);});
el("refresh").onclick = () => guarded(refresh);
el("daily-form").onsubmit = event => {event.preventDefault(); guarded(() => loadDaily(1));};
el("stock-form").onsubmit = event => {event.preventDefault(); guarded(() => loadHistory(el("ticker").value));};
for (const view of ["market", "stock", "quality"]) el("tab-" + view).onclick = () => switchView(view);
let searchTimer;
el("ticker").oninput = () => {
  clearTimeout(searchTimer);
  const value = el("ticker").value;
  if (!state.active || value.trim().length < 1) return;
  searchTimer = setTimeout(async () => {
    try {
      const matches = await api("/api/symbols", {q: value});
      if (el("ticker").value !== value) return;
      el("ticker-options").replaceChildren();
      for (const item of matches) {
        const opt = add(el("ticker-options"), "option");
        opt.value = item.code;
        if (item.name) opt.label = item.name;
      }
    } catch (_) { el("ticker-options").replaceChildren(); }
  }, 150);
};
el("chart").onpointermove = event => {
  const width = el("chart").clientWidth;
  const index = Math.floor((event.offsetX - 58) / (width - 64) * state.chart.length);
  drawChart(index);
};
el("chart").onpointerleave = () => drawChart();
window.addEventListener("resize", () => drawChart());
guarded(refresh);
