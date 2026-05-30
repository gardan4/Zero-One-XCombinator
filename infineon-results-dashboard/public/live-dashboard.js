/**
 * Live registry, comparison, disagreements, and inference — backed by FastAPI.
 */

import {
  CONTROL_PLANE_BASE,
  controlPlaneOnline,
  getCompareExamples,
  getCompareReport,
  getRun,
  getRunConfusion,
  getRunExamples,
  getRunMetrics,
  getRuns,
  probeControlPlane,
  startInferenceJob,
} from "./control-plane.js";
import { dataBadge } from "./badges.js";

const FAMILIES = ["MOSFET", "IGBT", "IC"];
const HEADLINE_KEYS = ["top1", "top3", "mrr", "em", "anomaly_f1", "anomaly_auc"];
const INFER_ROW_LIMIT = 100;

const INFER_INPUT_HINTS = {
  nextstep: "Partial route — one step per line (defaults: MOSFET, 60% observed).",
  completion: "Partial route to complete — one step per line (defaults: MOSFET, 60% observed).",
  anomaly: "Full route to validate — one step per line (defaults: MOSFET).",
};

const $ = (id) => document.getElementById(id);

let compareState = {
  report: null,
  allRuns: [],
  filterSplit: "",
  filterRole: "",
  filterFamily: "",
  taskGroup: "all",
  selectedRunIds: [],
  selectionInitialized: false,
  selectedAnomalyRun: null,
  confusion: { tp: 0, fp: 0, tn: 0, fn: 0 },
  lossSeries: [],
};

let inferencePoll = null;
let inferState = { rows: [], task: null, total: 0, batch: false };

export async function initLiveDashboard(escapeHtml, flushBars, queueBar) {
  _escapeHtml = escapeHtml;
  _flushBars = flushBars;
  _queueBar = queueBar;

  bindCompareFilters();
  bindDiffControls();
  bindInferenceForm();

  const online = await probeControlPlane();
  setControlPlaneStatus(online);

  const banner = $("cpOfflineBanner");
  const liveRoot = $("liveControl");
  if (!online) {
    if (banner) banner.hidden = false;
    if (liveRoot) liveRoot.classList.add("cp-degraded");
    return;
  }
  if (banner) banner.hidden = true;
  if (liveRoot) liveRoot.classList.remove("cp-degraded");

  await refreshLiveData();
}

let _escapeHtml = (s) => String(s);
let _flushBars = () => {};
let _queueBar = () => {};

function setControlPlaneStatus(online) {
  const el = $("controlPlaneStatus");
  if (!el) return;
  el.querySelector(".status-text").textContent = online
    ? `Control plane · ${CONTROL_PLANE_BASE.replace(/^https?:\/\//, "")}`
    : "Control plane offline";
  el.classList.toggle("online", online);
  el.classList.toggle("live", !online);
}

async function refreshLiveData() {
  if (!controlPlaneOnline()) return;
  try {
    const [runs, report] = await Promise.all([getRuns(), getCompareReport({})]);
    compareState.allRuns = runs;
    compareState.report = report;
    initRunSelection(report?.rows ?? []);
    renderRegistry(runs);
    renderRunPicker(report?.rows ?? []);
    renderCompareSection();
  } catch (err) {
    setControlPlaneStatus(false);
    const banner = $("cpOfflineBanner");
    if (banner) {
      banner.hidden = false;
      banner.querySelector(".cp-offline-msg").textContent =
        `Could not reach control plane: ${err.message || "unknown error"}`;
    }
  }
}

function bindCompareFilters() {
  $("cpFilterSplit")?.addEventListener("change", (e) => {
    compareState.filterSplit = e.target.value;
    loadCompareReport();
  });
  $("cpFilterRole")?.addEventListener("change", (e) => {
    compareState.filterRole = e.target.value;
    loadCompareReport();
  });
  $("cpFilterFamily")?.addEventListener("change", (e) => {
    compareState.filterFamily = e.target.value;
    loadCompareReport();
  });
  $("cpFilterTask")?.addEventListener("change", (e) => {
    compareState.taskGroup = e.target.value;
    renderCompareViews();
  });
  $("cpRefreshBtn")?.addEventListener("click", () => refreshLiveData());
  $("cpRunPicker")?.addEventListener("change", (e) => {
    const cb = e.target.closest('input[type="checkbox"][data-run-id]');
    if (!cb) return;
    toggleRunSelection(cb.dataset.runId, cb.checked);
  });
  $("cpAnomalyRunSelect")?.addEventListener("change", async (e) => {
    compareState.selectedAnomalyRun = e.target.value || null;
    if (compareState.selectedAnomalyRun) {
      compareState.confusion = await getRunConfusion(compareState.selectedAnomalyRun);
      renderLiveConfusion();
    }
  });
}

async function loadCompareReport() {
  if (!controlPlaneOnline()) return;
  compareState.report = await getCompareReport({
    split: compareState.filterSplit || undefined,
    role: compareState.filterRole || undefined,
    family: compareState.filterFamily || undefined,
  });
  initRunSelection(compareState.report?.rows ?? [], true);
  renderRunPicker(compareState.report?.rows ?? []);
  renderCompareSection();
}

function sortedCompareRows(rows) {
  return [...rows].sort((a, b) => String(b.created_at || "").localeCompare(String(a.created_at || "")));
}

function metricKeys(row) {
  return Object.keys(row?.metrics_flat || {});
}

function isProxyOnly(row) {
  const keys = metricKeys(row);
  return keys.length > 0 && keys.every((k) => k.startsWith("proxy_"));
}

function hasScoredMetrics(row) {
  const flat = row?.metrics_flat || {};
  return Object.keys(flat).some((k) => !k.startsWith("proxy_") && typeof flat[k] === "number");
}

function pickDefaultSelection(rows) {
  return sortedCompareRows(rows)
    .filter(hasScoredMetrics)
    .slice(0, 2)
    .map((r) => r.run_id);
}

function initRunSelection(rows, preserve = false) {
  const ids = new Set(rows.map((r) => r.run_id));
  if (preserve && compareState.selectedRunIds.length) {
    const kept = compareState.selectedRunIds.filter((id) => ids.has(id));
    if (kept.length) {
      compareState.selectedRunIds = kept;
      return;
    }
  }
  if (!compareState.selectionInitialized || !preserve) {
    compareState.selectedRunIds = pickDefaultSelection(rows);
    compareState.selectionInitialized = true;
  }
}

function toggleRunSelection(runId, checked) {
  const set = new Set(compareState.selectedRunIds);
  if (checked) set.add(runId);
  else set.delete(runId);
  compareState.selectedRunIds = [...set];
  updateRunPickerHint();
  renderCompareViews();
}

function selectedRows() {
  const rows = compareState.report?.rows ?? [];
  const sel = new Set(compareState.selectedRunIds);
  return rows.filter((r) => sel.has(r.run_id));
}

function visibleSpecs(rows, specs) {
  return specs.filter((s) => rows.some((r) => flatMetric(r, s.key) !== null));
}

function runBadge(row) {
  if (isProxyOnly(row)) return dataBadge("proxy");
  if (hasScoredMetrics(row)) return dataBadge("live");
  return dataBadge("mock");
}

function renderRunPicker(rows) {
  const host = $("cpRunPicker");
  if (!host) return;
  const sorted = sortedCompareRows(rows);
  if (!sorted.length) {
    host.innerHTML = `<p class="cp-empty">No eval runs match filters — run <code>zo-track predict</code> with gold labels.</p>`;
    updateRunPickerHint();
    return;
  }
  host.innerHTML = sorted
    .map((r) => {
      const m = r.model || {};
      const d = r.dataset || {};
      const headline = headlineSummary(r);
      const checked = compareState.selectedRunIds.includes(r.run_id) ? "checked" : "";
      return `<label class="run-pick">
        <input type="checkbox" data-run-id="${_escapeHtml(r.run_id)}" ${checked} />
        <span class="run-pick-body">
          <span class="run-pick-title">${_escapeHtml(r.run_name)} ${runBadge(r)}</span>
          <span class="run-pick-meta">${_escapeHtml(m.model_ref || m.display || "—")} · ${_escapeHtml(m.role || r.kind)} · ${_escapeHtml(d.split || "—")}/${_escapeHtml(d.family || "—")}</span>
          <span class="run-pick-metrics">${_escapeHtml(headline)}</span>
        </span>
      </label>`;
    })
    .join("");
  updateRunPickerHint();
}

function headlineSummary(row) {
  const parts = [];
  for (const k of HEADLINE_KEYS) {
    const v = flatMetric(row, k);
    if (v !== null) parts.push(`${k}=${fmtMetric(v, k.includes("top") || k === "em" || k.startsWith("anomaly_f") ? "pct" : "dec")}`);
  }
  if (!parts.length) {
    const flat = row?.metrics_flat || {};
    const first = Object.entries(flat).find(([, v]) => typeof v === "number");
    return first ? `${first[0]}=${first[1]}` : "no metrics logged";
  }
  return parts.slice(0, 4).join(" · ");
}

function updateRunPickerHint() {
  const hint = $("cpRunPickerHint");
  if (!hint) return;
  const n = compareState.selectedRunIds.length;
  const total = compareState.report?.rows?.length ?? 0;
  hint.textContent =
    n === 0
      ? "Select at least one run to populate charts and tables."
      : `${n} of ${total} run${total === 1 ? "" : "s"} selected for comparison.`;
}

function renderSelectedCompare(rows) {
  const host = $("cpSelectedCompare");
  if (!host) return;
  if (!rows.length) {
    host.innerHTML = `<p class="cp-empty">Select runs above to compare headline metrics.</p>`;
    return;
  }
  const specs = (compareState.report?.metric_specs ?? []).filter((s) => HEADLINE_KEYS.includes(s.key));
  const cols = visibleSpecs(rows, specs);
  if (!cols.length) {
    host.innerHTML = `<p class="cp-empty">Selected runs have no headline task metrics (proxy-only or empty). ${dataBadge("proxy")}</p>`;
    return;
  }
  const head = cols.map((c) => `<th>${_escapeHtml(c.label)}</th>`).join("");
  const body = rows
    .map((r) => {
      const m = r.model || {};
      const cells = cols.map((c) => `<td class="num">${fmtMetric(flatMetric(r, c.key), c.format)}</td>`).join("");
      return `<tr>
        <td><span class="cp-run-name">${_escapeHtml(r.run_name)}</span> ${runBadge(r)}</td>
        <td class="truncate">${_escapeHtml(m.model_ref || m.display || "—")}</td>
        ${cells}
      </tr>`;
    })
    .join("");
  host.innerHTML = `<div class="table-wrap"><table class="table cp-matrix cp-headline-table">
    <thead><tr><th>Run</th><th>Model</th>${head}</tr></thead>
    <tbody>${body}</tbody></table></div>`;
}

function renderRegistry(runs) {
  const host = $("cpRunRegistry");
  if (!host) return;
  if (!runs?.length) {
    host.innerHTML = `<p class="cp-empty">No experiment runs yet. Use the <b>Compare runs</b> tab to start an inference job or run <code>zo-track predict</code>.</p>`;
    return;
  }
  const rows = runs
    .slice(0, 40)
    .map((r) => {
      const metric = topMetric(r.metrics);
      const statusCls = `cp-status cp-status-${r.status || "created"}`;
      return `<tr>
        <td><span class="cp-run-name">${_escapeHtml(r.name)}</span><span class="cp-run-id">${_escapeHtml(r.id.slice(-14))}</span></td>
        <td>${_escapeHtml(r.kind)}</td>
        <td><span class="${statusCls}">${_escapeHtml(r.status)}</span></td>
        <td>${_escapeHtml(r.git_branch || "—")}</td>
        <td class="mono-col">${_escapeHtml(metric)}</td>
      </tr>`;
    })
    .join("");
  host.innerHTML = `<div class="table-wrap"><table class="table cp-run-table">
    <thead><tr><th>Run</th><th>Kind</th><th>Status</th><th>Branch</th><th>Metric</th></tr></thead>
    <tbody>${rows}</tbody></table></div>
    <p class="cp-meta">${runs.length} run${runs.length === 1 ? "" : "s"} in registry</p>`;
}

function topMetric(m) {
  if (!m || typeof m !== "object") return "—";
  for (const k of ["accuracy", "top1", "anomaly_auc", "success_rate", "reward", "loss"]) {
    if (k in m) return `${k}=${m[k]}`;
  }
  const first = Object.entries(m)[0];
  return first ? `${first[0]}=${first[1]}` : "—";
}

function flatMetric(row, key) {
  const v = row?.metrics_flat?.[key];
  return typeof v === "number" ? v : null;
}

function fmtMetric(v, format) {
  if (v === null || v === undefined) return "·";
  if (format === "pct") return `${(v * 100).toFixed(1)}%`;
  return v.toFixed(3);
}

function isAnomalyRow(r) {
  return flatMetric(r, "anomaly_auc") !== null || flatMetric(r, "anomaly_f1") !== null;
}

function renderCompareSection() {
  const allRows = compareState.report?.rows ?? [];
  renderRunPicker(allRows);
  renderCompareViews();
}

function renderCompareViews() {
  const report = compareState.report;
  const allRows = report?.rows ?? [];
  const rows = selectedRows();
  const specsAll = (report?.metric_specs ?? []).filter(
    (s) => compareState.taskGroup === "all" || s.task === compareState.taskGroup,
  );
  const specs = visibleSpecs(rows.length ? rows : allRows, specsAll);
  renderSelectedCompare(rows);
  renderCompareMatrix(rows, specs, report?.deltas_vs_baseline);
  renderLiveHeroBars(rows);
  renderLivePredictorBars(rows);
  renderAnomalyRunSelect(rows.filter(isAnomalyRow));
  renderLiveLossChart(rows.length ? rows : allRows);
  populateDiffRunSelects(rows);
}

function renderCompareMatrix(rows, specs, deltas) {
  const host = $("cpCompareMatrix");
  if (!host) return;
  if (!rows.length) {
    host.innerHTML = `<p class="cp-empty">Select at least one run in the picker above.</p>`;
    return;
  }
  if (!specs.length) {
    host.innerHTML = `<p class="cp-empty">Selected runs have no metrics for this task filter. Try <b>Metrics → all</b> or pick runs with gold-label eval. ${dataBadge("proxy")}</p>`;
    return;
  }
  const head = specs.map((c) => `<th title="${_escapeHtml(c.task)}">${_escapeHtml(c.label)}</th>`).join("");
  const body = rows
    .map((r) => {
      const d = r.dataset || {};
      const m = r.model || {};
      const delta = deltas?.[r.run_id] || {};
      const cells = specs
        .map((c) => {
          const v = flatMetric(r, c.key);
          const dv = delta[c.key];
          let deltaHtml = "";
          if (typeof dv === "number" && Math.abs(dv) > 0.0001) {
            const sign = dv > 0 ? "+" : "";
            const cls = dv > 0 ? "gain" : "loss-delta";
            const fmt = c.format === "pct" ? (dv * 100).toFixed(1) : dv.toFixed(2);
            deltaHtml = `<span class="${cls}">${sign}${fmt}</span>`;
          }
          return `<td class="num">${fmtMetric(v, c.format)}${deltaHtml}</td>`;
        })
        .join("");
      return `<tr>
        <td><span class="cp-run-name">${_escapeHtml(r.run_name)}</span><span class="cp-run-id">${_escapeHtml(r.run_id.slice(-12))}</span> ${runBadge(r)}</td>
        <td class="truncate" title="${_escapeHtml(m.display || "")}">${_escapeHtml(m.model_ref || m.display || "—")}</td>
        <td>${_escapeHtml(m.role || "—")}</td>
        <td class="split-${_escapeHtml(d.split || "na")}">${_escapeHtml(d.split || "—")}</td>
        <td>${_escapeHtml(d.family || "—")}</td>
        ${cells}
      </tr>`;
    })
    .join("");
  host.innerHTML = `<div class="table-wrap"><table class="table cp-matrix">
    <thead><tr><th>Run</th><th>Model</th><th>Role</th><th>Split</th><th>Family</th>${head}</tr></thead>
    <tbody>${body}</tbody></table></div>`;
}

function renderLiveHeroBars(rows) {
  const host = $("cpHeroBars");
  if (!host) return;
  const anomalyRows = rows.filter(isAnomalyRow);
  const data = FAMILIES.map((fam) => {
    const idR = anomalyRows.find((x) => x.dataset?.split === "id" && x.dataset?.family === fam);
    const oodR = anomalyRows.find((x) => x.dataset?.split === "ood" && x.dataset?.family === fam);
    const id = idR ? flatMetric(idR, `anomaly_auc_${fam}`) ?? flatMetric(idR, "anomaly_auc") : null;
    const ood = oodR ? flatMetric(oodR, `anomaly_auc_${fam}`) ?? flatMetric(oodR, "anomaly_auc") : null;
    return { fam, id, ood };
  }).filter((x) => x.id !== null || x.ood !== null);

  if (!data.length) {
    host.innerHTML = `<p class="cp-empty">No per-family anomaly AUC in registry yet.</p>`;
    return;
  }
  host.innerHTML = data
    .map(
      (row) => `<div class="compare-row">
        <div class="cr-head"><span class="cr-name">${_escapeHtml(row.fam)} · ROC-AUC</span></div>
        <div class="track"><span class="base" style="width:${(row.id ?? 0) * 100}%"></span></div>
        <div class="track" style="margin-top:5px"><span class="tuned" style="width:${(row.ood ?? 0) * 100}%"></span></div>
        <div class="cr-vals" style="margin-top:6px;font-size:12px">
          <span class="b">ID ${fmtMetric(row.id, "dec")}</span>
          <span class="sep"> · </span>
          <span class="t">OOD ${fmtMetric(row.ood, "dec")}</span>
        </div>
      </div>`,
    )
    .join("");
}

function renderLivePredictorBars(rows) {
  const host = $("cpPredictorBars");
  if (!host) return;
  const byPred = new Map();
  for (const r of rows.filter(isAnomalyRow)) {
    const p = r.model?.predictor ?? r.kind ?? "?";
    const auc = flatMetric(r, "anomaly_auc");
    const f1 = flatMetric(r, "anomaly_f1");
    const e = byPred.get(p) ?? { auc: [], f1: [] };
    if (auc !== null) e.auc.push(auc);
    if (f1 !== null) e.f1.push(f1);
    byPred.set(p, e);
  }
  const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);
  const entries = [...byPred.entries()].map(([predictor, v]) => ({
    predictor,
    auc: mean(v.auc),
    f1: mean(v.f1),
  }));
  if (!entries.length) {
    host.innerHTML = `<p class="cp-empty">No anomaly predictors in registry.</p>`;
    return;
  }
  host.innerHTML = entries
    .map(
      (e) => `<div class="compare-row">
        <div class="cr-head"><span class="cr-name">${_escapeHtml(e.predictor)}</span>
          <span class="cr-vals"><span class="b">AUC ${fmtMetric(e.auc, "dec")}</span><span class="sep"> · </span><span class="t">F1 ${fmtMetric(e.f1, "pct")}</span></span>
        </div>
        <div class="track"><span class="base" style="width:${(e.auc ?? 0) * 100}%"></span></div>
        <div class="track" style="margin-top:5px"><span class="tuned" style="width:${(e.f1 ?? 0) * 100}%"></span></div>
      </div>`,
    )
    .join("");
}

function renderAnomalyRunSelect(anomalyRows) {
  const sel = $("cpAnomalyRunSelect");
  if (!sel) return;
  const prev = compareState.selectedAnomalyRun;
  sel.innerHTML =
    `<option value="">Select anomaly run…</option>` +
    anomalyRows
      .map(
        (r) =>
          `<option value="${_escapeHtml(r.run_id)}">${_escapeHtml(r.run_name)} [${r.dataset?.split ?? "?"}/${r.model?.predictor ?? "?"}]</option>`,
      )
      .join("");
  if (prev && anomalyRows.some((r) => r.run_id === prev)) {
    sel.value = prev;
  } else if (anomalyRows.length) {
    const ood = anomalyRows.find((r) => r.dataset?.split === "ood") ?? anomalyRows[0];
    sel.value = ood.run_id;
    compareState.selectedAnomalyRun = ood.run_id;
    getRunConfusion(ood.run_id).then((c) => {
      compareState.confusion = c;
      renderLiveConfusion();
    });
  }
}

function renderLiveConfusion() {
  const host = $("cpLiveConfusion");
  if (!host) return;
  const c = compareState.confusion;
  const total = c.tp + c.fp + c.tn + c.fn || 1;
  const cell = (n, label, cls) =>
    `<div class="cm-cell ${cls}"><span class="cm-n">${n}</span><span class="cm-l">${label} · ${((n / total) * 100).toFixed(1)}%</span></div>`;
  host.innerHTML =
    cell(c.tp, "anomaly · flagged", "hit") +
    cell(c.fn, "anomaly · missed", "miss") +
    cell(c.fp, "valid · false alarm", "miss") +
    cell(c.tn, "valid · passed", "hit");
}

async function renderLiveLossChart(rows) {
  const host = $("cpLossChart");
  if (!host) return;
  const evalIds = new Set(rows.map((r) => r.run_id));
  let runs;
  try {
    runs = await getRuns();
  } catch {
    host.innerHTML = `<p class="cp-empty">Could not load training metrics.</p>`;
    return;
  }
  const trainish = runs
    .filter((r) => (r.kind === "sft" || r.kind === "grpo") && !evalIds.has(r.id))
    .slice(0, 6);
  const series = await Promise.all(
    trainish.map(async (r) => ({ name: r.name, data: await getRunMetrics(r.id) })),
  );
  const withLoss = series.filter((s) => s.data.some((m) => typeof m.loss === "number"));
  if (withLoss.length < 1) {
    host.innerHTML = `<p class="cp-empty">No training loss curves in registry yet.</p>`;
    return;
  }
  const maxLen = Math.max(...withLoss.map((s) => s.data.length));
  const maxStep = Math.max(1, maxLen - 1);
  const allLoss = withLoss.flatMap((s) => s.data.map((m) => m.loss).filter((v) => typeof v === "number"));
  const yMax = Math.max(1, ...allLoss) * 1.1;
  const W = 680,
    H = 260,
    padL = 36,
    padR = 12,
    padT = 14,
    padB = 26;
  const iW = W - padL - padR,
    iH = H - padT - padB;
  const x = (s) => padL + (s / maxStep) * iW;
  const y = (v) => padT + (1 - v / yMax) * iH;
  const colors = ["#008a4b", "#006b39", "#00a85a", "#46585c", "#b9740f", "#c2403a"];
  const paths = withLoss
    .map((s, idx) => {
      const pts = s.data
        .map((m, i) => (typeof m.loss === "number" ? `${i ? "L" : "M"}${x(i).toFixed(1)} ${y(m.loss).toFixed(1)}` : null))
        .filter(Boolean)
        .join(" ");
      return `<path class="line-train" stroke="${colors[idx % colors.length]}" d="${pts}"/>`;
    })
    .join("");
  const legend = withLoss
    .map(
      (s, idx) =>
        `<span><i style="background:${colors[idx % colors.length]}"></i>${_escapeHtml(s.name.slice(0, 28))}</span>`,
    )
    .join("");
  host.innerHTML = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Training loss curves">${paths}</svg>
    <div class="legend" style="margin:14px 0 0">${legend}</div>`;
}

function populateDiffRunSelects(rows) {
  const opts = rows.map((r) => `<option value="${_escapeHtml(r.run_id)}">${_escapeHtml(r.run_name)}</option>`).join("");
  const ids = rows.map((r) => r.run_id);
  const defaults = compareState.selectedRunIds.filter((id) => ids.includes(id));
  for (const [id, fallbackIdx] of [
    ["cpDiffRunA", 0],
    ["cpDiffRunB", 1],
  ]) {
    const sel = $(id);
    if (!sel) continue;
    const prev = sel.value;
    sel.innerHTML = `<option value="">Select run…</option>${opts}`;
    if (prev && ids.includes(prev)) sel.value = prev;
    else if (defaults[fallbackIdx]) sel.value = defaults[fallbackIdx];
  }
}

function bindDiffControls() {
  $("cpDiffLoadBtn")?.addEventListener("click", loadDiffExamples);
}

async function loadDiffExamples() {
  const runA = $("cpDiffRunA")?.value;
  const runB = $("cpDiffRunB")?.value;
  const task = $("cpDiffTask")?.value || "nextstep";
  const mode = $("cpDiffMode")?.value || "different";
  const host = $("cpDiffExamples");
  if (!host) return;
  if (!runA || !runB) {
    host.innerHTML = `<p class="cp-empty">Select two runs with examples.jsonl artifacts.</p>`;
    return;
  }
  host.innerHTML = `<div class="loading-row"><span class="spin"></span>Loading disagreements…</div>`;
  try {
    const res = await getCompareExamples(runA, runB, task, mode, 25);
    const examples = res?.examples ?? [];
    if (!examples.length) {
      host.innerHTML = `<p class="cp-empty">No examples matched this filter.</p>`;
      return;
    }
    host.innerHTML = examples.map((ex) => renderDiffCard(ex)).join("");
  } catch (err) {
    host.innerHTML = `<p class="cp-empty cp-error">${_escapeHtml(err.message || "Failed to load examples")}</p>`;
  }
}

function renderDiffCard(ex) {
  const reasonA = ex.a?.trace?.reasoning;
  const reasonB = ex.b?.trace?.reasoning;
  return `<div class="diff-card">
    <div class="diff-id">${_escapeHtml(ex.example_id)} · ${_escapeHtml(ex.family || "")}</div>
    <div class="diff-grid">
      ${diffSide("A", ex.a, reasonA)}
      ${diffSide("B", ex.b, reasonB)}
    </div>
    ${ex.gold != null ? `<div class="diff-gold">gold: <code>${_escapeHtml(JSON.stringify(ex.gold).slice(0, 120))}</code></div>` : ""}
  </div>`;
}

function diffSide(label, side, reason) {
  const ok = side?.correct;
  const mark = ok === true ? "✓" : ok === false ? "✗" : "";
  const markCls = ok === true ? "ok" : ok === false ? "bad" : "";
  return `<div class="diff-side">
    <div class="diff-label">${label} <span class="${markCls}">${mark}</span></div>
    <pre class="diff-pred">${_escapeHtml(JSON.stringify(side?.prediction ?? null))}</pre>
    ${reason ? `<p class="diff-reason">${_escapeHtml(String(reason).slice(0, 280))}</p>` : ""}
  </div>`;
}

function bindInferenceForm() {
  $("cpInferTask")?.addEventListener("change", updateInferInputHint);
  $("cpInferCsv")?.addEventListener("change", onInferCsvSelected);
  $("cpInferRun")?.addEventListener("click", () => withInferBusy(runInferJob));
  $("cpInferExportCsv")?.addEventListener("click", downloadInferCsv);
  updateInferInputHint();
}

function updateInferInputHint() {
  const task = $("cpInferTask")?.value || "nextstep";
  const hint = $("cpInferInputHint");
  const input = $("cpInferInput");
  if (hint) hint.textContent = INFER_INPUT_HINTS[task] || "";
  if (input && !input.value.trim()) {
    input.placeholder =
      task === "anomaly"
        ? "RECEIVE WAFER LOT\nLOT IDENTIFICATION\n…"
        : "SPIN COAT PHOTORESIST\nEXPOSE LITHO LEVEL 1\n…";
  }
}

function onInferCsvSelected() {
  const file = $("cpInferCsv")?.files?.[0];
  const nameEl = $("cpInferFileName");
  if (!file) {
    if (nameEl) nameEl.textContent = "No CSV — using text input above";
    return;
  }
  if (nameEl) nameEl.textContent = file.name;
}

function inferUsesCsv() {
  return Boolean($("cpInferCsv")?.files?.[0]);
}

function buildInferFormData() {
  const fd = new FormData();
  const predictor = $("cpInferModelSelect")?.value || "ngram";
  const task = $("cpInferTask")?.value || "nextstep";
  fd.set("predictor", predictor);
  fd.set("model", "default");
  fd.set("version", "dashboard-v1");
  fd.set("tasks", task);
  fd.set("tags", "source:dashboard");

  const csvFile = $("cpInferCsv")?.files?.[0];
  if (csvFile) {
    if (task === "anomaly") fd.set("anomaly_csv", csvFile);
    else fd.set("valid_csv", csvFile);
    return fd;
  }

  const steps = ($("cpInferInput")?.value || "")
    .split("\n")
    .map((s) => s.trim())
    .filter(Boolean);
  if (!steps.length) {
    throw new Error("Enter steps in the text box or choose a CSV file.");
  }
  const manual = {
    task,
    family: "MOSFET",
    completion_fraction: 0.6,
  };
  if (task === "anomaly") manual.sequence = steps;
  else manual.partial_sequence = steps;
  fd.set("manual_json", JSON.stringify(manual));
  return fd;
}

async function withInferBusy(fn) {
  const status = $("cpInferStatus");
  const btn = $("cpInferRun");
  btn.disabled = true;
  status.textContent = "Running…";
  hideInferOutput();
  try {
    await fn();
  } finally {
    btn.disabled = false;
  }
}

function hideInferOutput() {
  const out = $("cpInferOutput");
  const exportRow = $("cpInferExportRow");
  if (out) out.hidden = true;
  if (exportRow) exportRow.hidden = true;
  inferState = { rows: [], task: null, total: 0, batch: false };
}

async function runInferJob() {
  const status = $("cpInferStatus");
  try {
    const fd = buildInferFormData();
    const batch = inferUsesCsv();
    const task = $("cpInferTask")?.value || "nextstep";
    const res = await startInferenceJob(fd);
    status.innerHTML = `Running… <span class="cp-run-id">${_escapeHtml(res.run_id.slice(-14))}</span>`;
    pollInferenceJob(res.run_id, task, batch);
    setTimeout(refreshLiveData, 2000);
  } catch (err) {
    status.textContent = err.message || "Run failed";
  }
}

function pollInferenceJob(runId, task, batch) {
  if (inferencePoll) clearInterval(inferencePoll);
  const tick = async () => {
    try {
      const meta = await getRun(runId);
      const statusEl = $("cpInferStatus");
      if (statusEl) {
        statusEl.innerHTML = `<span class="cp-status cp-status-${meta.status}">${_escapeHtml(meta.status)}</span> · ${_escapeHtml(runId.slice(-14))}`;
      }
      if (meta.status === "completed" || meta.status === "failed") {
        clearInterval(inferencePoll);
        inferencePoll = null;
        if (meta.status === "completed") {
          const ex = await getRunExamples(runId, task, INFER_ROW_LIMIT);
          renderInferResults(ex.examples || [], task, batch, ex.total ?? ex.count ?? 0);
        } else if (statusEl) {
          statusEl.textContent = meta.notes || "Job failed";
        }
        refreshLiveData();
      }
    } catch {
      /* keep polling */
    }
  };
  tick();
  inferencePoll = setInterval(tick, 2500);
}

function renderInferResults(examples, task, batch, total) {
  const host = $("cpInferOutput");
  const exportRow = $("cpInferExportRow");
  const exportMeta = $("cpInferExportMeta");
  if (!host) return;

  inferState = { rows: examples, task, total: total || examples.length, batch };

  if (!examples.length) {
    host.innerHTML = `<p class="cp-empty">Job finished but no example rows were returned.</p>`;
    host.hidden = false;
    return;
  }

  const truncated = total > examples.length;
  const truncNote = truncated
    ? `<p class="cp-meta">Showing first ${examples.length} of ${total} rows (frontend limit ${INFER_ROW_LIMIT}).</p>`
    : "";

  if (!batch && examples.length === 1) {
    host.innerHTML = truncNote + renderInferSingle(examples[0], task);
    host.hidden = false;
    if (exportRow) exportRow.hidden = true;
    return;
  }

  host.innerHTML = truncNote + renderInferTable(examples, task);
  host.hidden = false;
  if (exportRow) exportRow.hidden = false;
  if (exportMeta) {
    exportMeta.textContent = `${examples.length} row${examples.length === 1 ? "" : "s"} ready to export`;
  }
}

function renderInferSingle(ex, task) {
  const pred = ex.prediction;
  const input = ex.input || {};
  let body = "";

  if (task === "nextstep") {
    const partial = (input.partial_sequence || []).join("\n");
    const ranked = Array.isArray(pred) ? pred : [];
    body = `<div class="infer-block"><span class="infer-label">Partial route</span><pre class="infer-text">${_escapeHtml(partial || "—")}</pre></div>
      <div class="infer-block"><span class="infer-label">Predicted next steps (ranked)</span><ol class="infer-ranked">${ranked.map((s, i) => `<li>${_escapeHtml(s)}</li>`).join("") || "<li class='empty'>—</li>"}</ol></div>`;
  } else if (task === "completion") {
    const partial = (input.partial_sequence || []).join("\n");
    const completed = Array.isArray(pred) ? pred.join("\n") : String(pred ?? "—");
    body = `<div class="infer-block"><span class="infer-label">Partial route</span><pre class="infer-text">${_escapeHtml(partial || "—")}</pre></div>
      <div class="infer-block"><span class="infer-label">Completed suffix</span><pre class="infer-text">${_escapeHtml(completed)}</pre></div>`;
  } else {
    const seq = (input.sequence || []).join("\n");
    const valid = pred?.is_valid === 1 || pred?.is_valid === true;
    body = `<div class="infer-block"><span class="infer-label">Route</span><pre class="infer-text">${_escapeHtml(seq || "—")}</pre></div>
      <div class="infer-block"><span class="infer-label">Verdict</span><p class="infer-verdict ${valid ? "valid" : "invalid"}">${valid ? "Valid" : "Invalid"}</p>
      ${pred?.rule ? `<p class="cp-meta">Rule: <code>${_escapeHtml(pred.rule)}</code></p>` : ""}</div>`;
  }

  const goldLine =
    ex.gold != null
      ? `<div class="infer-block"><span class="infer-label">Gold (if available)</span><pre class="infer-text infer-muted">${_escapeHtml(formatGold(ex.gold, task))}</pre></div>`
      : "";

  return `<div class="infer-single">${body}${goldLine}</div>`;
}

function formatGold(gold, task) {
  if (task === "nextstep" && typeof gold === "string") return gold;
  if (Array.isArray(gold)) return gold.join("\n");
  if (typeof gold === "object" && gold) return JSON.stringify(gold, null, 2);
  return String(gold);
}

function renderInferTable(examples, task) {
  const { head, rowFn } = inferTableSpec(task);
  const rows = examples.map((ex) => `<tr>${rowFn(ex)}</tr>`).join("");
  return `<div class="table-wrap"><table class="table infer-table"><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table></div>`;
}

function inferTableSpec(task) {
  if (task === "nextstep") {
    return {
      head: "<th>Example</th><th>Family</th><th>Top prediction</th><th>Top 5</th><th>Correct</th>",
      rowFn: (ex) => {
        const pred = Array.isArray(ex.prediction) ? ex.prediction : [];
        const ok = ex.correct === true ? "✓" : ex.correct === false ? "✗" : "·";
        return `<td>${_escapeHtml(ex.example_id)}</td><td>${_escapeHtml(ex.family || "—")}</td>
          <td class="tuned-col">${_escapeHtml(pred[0] || "—")}</td>
          <td class="infer-wrap">${_escapeHtml(pred.slice(0, 5).join(" · ") || "—")}</td>
          <td>${ok}</td>`;
      },
    };
  }
  if (task === "completion") {
    return {
      head: "<th>Example</th><th>Family</th><th>Completed steps</th><th>Exact match</th>",
      rowFn: (ex) => {
        const pred = Array.isArray(ex.prediction) ? ex.prediction.join(" | ") : "—";
        const ok = ex.correct === true ? "✓" : ex.correct === false ? "✗" : "·";
        return `<td>${_escapeHtml(ex.example_id)}</td><td>${_escapeHtml(ex.family || "—")}</td>
          <td class="infer-wrap">${_escapeHtml(pred)}</td><td>${ok}</td>`;
      },
    };
  }
  return {
    head: "<th>Example</th><th>Family</th><th>Valid?</th><th>Rule</th><th>Correct</th>",
    rowFn: (ex) => {
      const pred = ex.prediction || {};
      const valid = pred.is_valid === 1 || pred.is_valid === true;
      const ok = ex.correct === true ? "✓" : ex.correct === false ? "✗" : "·";
      return `<td>${_escapeHtml(ex.example_id)}</td><td>${_escapeHtml(ex.family || "—")}</td>
        <td>${valid ? "Valid" : "Invalid"}</td><td>${_escapeHtml(pred.rule || "—")}</td><td>${ok}</td>`;
    },
  };
}

function inferRowToCsvCells(ex, task) {
  if (task === "nextstep") {
    const pred = Array.isArray(ex.prediction) ? ex.prediction : [];
    return [
      ex.example_id ?? "",
      ex.family ?? "",
      pred[0] ?? "",
      pred.slice(0, 5).join("|"),
      ex.correct === true ? "1" : ex.correct === false ? "0" : "",
    ];
  }
  if (task === "completion") {
    const pred = Array.isArray(ex.prediction) ? ex.prediction : [];
    return [
      ex.example_id ?? "",
      ex.family ?? "",
      pred.join("|"),
      ex.correct === true ? "1" : ex.correct === false ? "0" : "",
    ];
  }
  const pred = ex.prediction || {};
  const valid = pred.is_valid === 1 || pred.is_valid === true;
  return [
    ex.example_id ?? "",
    ex.family ?? "",
    valid ? "1" : "0",
    pred.rule ?? "",
    ex.correct === true ? "1" : ex.correct === false ? "0" : "",
  ];
}

function inferCsvHeader(task) {
  if (task === "nextstep") return ["EXAMPLE_ID", "FAMILY", "PREDICTED_NEXT", "TOP5", "CORRECT"];
  if (task === "completion") return ["EXAMPLE_ID", "FAMILY", "COMPLETED_STEPS", "EXACT_MATCH"];
  return ["EXAMPLE_ID", "FAMILY", "IS_VALID", "RULE", "CORRECT"];
}

function csvCell(value) {
  const s = String(value ?? "");
  return /[",\n]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s;
}

function downloadInferCsv() {
  const { rows, task } = inferState;
  if (!rows.length || !task) return;
  const header = inferCsvHeader(task);
  const lines = rows.map((ex) => inferRowToCsvCells(ex, task).map(csvCell).join(","));
  const blob = new Blob([[header.join(","), ...lines].join("\n")], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `inference-${task}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}
