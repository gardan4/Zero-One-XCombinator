import { RESULTS as R } from "./results.js";

/* ------------------------------------------------------------------ *
 *  Formatting helpers
 * ------------------------------------------------------------------ */
const pct = (x) => `${(x * 100).toFixed(1)}%`;
const dec = (x) => x.toFixed(3);
const pts = (a, b) => `${a - b >= 0 ? "+" : ""}${((a - b) * 100).toFixed(1)} pts`;
const $ = (id) => document.getElementById(id);
const el = (tag, cls, html) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html != null) n.innerHTML = html;
  return n;
};

/* queue bar widths so the CSS transition animates from 0 */
const pending = [];
const queueBar = (node, width) => pending.push([node, width]);
function flushBars() {
  requestAnimationFrame(() =>
    requestAnimationFrame(() => pending.forEach(([n, w]) => (n.style.width = w))),
  );
}

/* ------------------------------------------------------------------ *
 *  Dashboard
 * ------------------------------------------------------------------ */
renderDashboard();
initDemo().catch(() => setStatus("Demo backend offline", false));

function renderDashboard() {
  const c = R.copy;

  $("modelName").textContent = c.modelName;
  $("demoModelName").textContent = c.modelName;
  $("batchModelName").textContent = c.modelName;
  $("heroEyebrow").textContent = c.heroEyebrow;
  $("heroTitle").innerHTML = `${escapeHtml(c.heroTitlePre)} <em>${escapeHtml(
    c.heroTitleEmph,
  )}</em> ${escapeHtml(c.heroTitlePost || "")}`;
  $("heroLede").textContent = c.heroLede;
  $("modelBlurb").textContent = c.modelBlurb;
  if (c.sections) {
    $("blurbNextstep").textContent = c.sections.nextstep || "";
    $("blurbCompletion").textContent = c.sections.completion || "";
    $("blurbAnomaly").textContent = c.sections.anomaly || "";
    $("blurbTraining").textContent = c.sections.training || "";
    $("blurbOod").textContent = c.sections.ood || "";
  }

  renderDeltas();
  renderCompare("nextstepCompare", [
    row("Top-1 accuracy", R.nextstep.overall.baseline.top1, R.nextstep.overall.finetuned.top1),
    row("Top-3 accuracy", R.nextstep.overall.baseline.top3, R.nextstep.overall.finetuned.top3),
    row("Top-5 accuracy", R.nextstep.overall.baseline.top5, R.nextstep.overall.finetuned.top5),
    row("MRR", R.nextstep.overall.baseline.mrr, R.nextstep.overall.finetuned.mrr, "dec"),
  ]);
  renderFamilyTable();
  renderCompare("completionCompare", [
    row("Exact match", R.completion.overall.baseline.exactMatch, R.completion.overall.finetuned.exactMatch),
    row("Token accuracy", R.completion.overall.baseline.tokenAcc, R.completion.overall.finetuned.tokenAcc),
    row("Block accuracy", R.completion.overall.baseline.blockAcc, R.completion.overall.finetuned.blockAcc),
    row("Edit distance", R.completion.overall.baseline.normEditDistance, R.completion.overall.finetuned.normEditDistance, "dec", true),
  ]);
  renderCompare("anomalyCompare", [
    row("Binary accuracy", R.anomaly.overall.baseline.binAcc, R.anomaly.overall.finetuned.binAcc),
    row("Precision", R.anomaly.overall.baseline.precision, R.anomaly.overall.finetuned.precision),
    row("Recall", R.anomaly.overall.baseline.recall, R.anomaly.overall.finetuned.recall),
    row("F1", R.anomaly.overall.baseline.f1, R.anomaly.overall.finetuned.f1),
    row("ROC-AUC", R.anomaly.overall.baseline.rocAuc, R.anomaly.overall.finetuned.rocAuc, "dec"),
  ]);
  renderConfusion();
  renderRuleTable();
  renderLossChart();
  renderTrainMeta();
  renderOod();
  renderTakeaways();

  flushBars();
}

function row(name, base, tuned, kind = "pct", lowerBetter = false) {
  return { name, base, tuned, kind, lowerBetter };
}

function renderDeltas() {
  const cards = [
    { k: "Next-step Top-1", base: R.nextstep.overall.baseline.top1, tuned: R.nextstep.overall.finetuned.top1, kind: "pct" },
    { k: "Next-step MRR", base: R.nextstep.overall.baseline.mrr, tuned: R.nextstep.overall.finetuned.mrr, kind: "dec" },
    { k: "Completion exact match", base: R.completion.overall.baseline.exactMatch, tuned: R.completion.overall.finetuned.exactMatch, kind: "pct" },
    { k: "Anomaly F1", base: R.anomaly.overall.baseline.f1, tuned: R.anomaly.overall.finetuned.f1, kind: "pct" },
  ];
  const host = $("deltas");
  host.innerHTML = "";
  cards.forEach((card) => {
    const val = card.kind === "dec" ? dec(card.tuned) : pct(card.tuned);
    const from = card.kind === "dec" ? dec(card.base) : pct(card.base);
    const delta = card.kind === "dec" ? `+${(card.tuned - card.base).toFixed(3)}` : pts(card.tuned, card.base);
    host.append(
      el(
        "div",
        "delta-card",
        `<span class="k">${escapeHtml(card.k)}</span>
         <span class="v">${val}</span>
         <span class="d">▲ ${delta} <span class="from">from ${from}</span></span>`,
      ),
    );
  });
}

function renderCompare(hostId, rows) {
  const host = $(hostId);
  host.innerHTML = "";
  rows.forEach((r) => {
    const f = r.kind === "dec" ? dec : pct;
    const hint = r.lowerBetter ? ' <span class="from">(lower is better)</span>' : "";
    const node = el(
      "div",
      "compare-row",
      `<div class="cr-head">
         <span class="cr-name">${escapeHtml(r.name)}${hint}</span>
         <span class="cr-vals"><span class="b">${f(r.base)}</span><span class="sep">→</span><span class="t">${f(r.tuned)}</span></span>
       </div>
       <div class="track"><span class="base"></span></div>
       <div class="track" style="margin-top:5px"><span class="tuned"></span></div>`,
    );
    host.append(node);
    queueBar(node.querySelector(".base"), `${r.base * 100}%`);
    queueBar(node.querySelector(".tuned"), `${r.tuned * 100}%`);
  });
}

function renderFamilyTable() {
  const tb = $("familyTable").querySelector("tbody");
  tb.innerHTML = "";
  R.nextstep.perFamily.forEach((f) => {
    tb.append(
      el(
        "tr",
        null,
        `<td>${escapeHtml(f.family)}</td>
         <td>${pct(f.baselineTop1)}</td>
         <td class="tuned-col">${pct(f.finetunedTop1)}</td>
         <td class="tuned-col">${pct(f.finetunedTop5)}</td>
         <td class="gain">${pts(f.finetunedTop1, f.baselineTop1)}</td>`,
      ),
    );
  });
}

function renderConfusion() {
  const cm = R.anomaly.confusion;
  const host = $("confusion");
  host.style.display = "grid";
  host.style.gridTemplateColumns = "1fr 1fr";
  host.style.gap = "8px";
  const cell = (n, label, cls) =>
    `<div class="cm-cell ${cls}"><span class="cm-n">${n}</span><span class="cm-l">${label}</span></div>`;
  host.innerHTML =
    cell(cm.tp, "anomaly · flagged", "hit") +
    cell(cm.fn, "anomaly · missed", "miss") +
    cell(cm.fp, "valid · false alarm", "miss") +
    cell(cm.tn, "valid · passed", "hit");
}

function renderRuleTable() {
  const shortById = new Map(R.rules.map((r) => [r.id, r.short]));
  const tb = $("ruleTable").querySelector("tbody");
  tb.innerHTML = "";
  R.anomaly.perRule.forEach((r) => {
    tb.append(
      el(
        "tr",
        null,
        `<td style="font-family:var(--mono);font-size:11.5px">${escapeHtml(r.rule)}</td>
         <td style="text-align:left;font-family:var(--sans);font-weight:400;color:var(--ink-2)">${escapeHtml(
           shortById.get(r.rule) || "",
         )}</td>
         <td class="tuned-col">${pct(r.attrAcc)}</td>`,
      ),
    );
  });
}

function renderLossChart() {
  const t = R.training;
  const W = 680, H = 260, padL = 36, padR = 12, padT = 14, padB = 26;
  const iW = W - padL - padR, iH = H - padT - padB;
  const maxStep = t.steps[t.steps.length - 1].step || 1;
  const yMax = 3.8;
  const x = (s) => padL + (s / maxStep) * iW;
  const y = (v) => padT + (1 - v / yMax) * iH;

  const line = (key) =>
    t.steps
      .filter((p) => p[key] != null)
      .map((p, i) => `${i ? "L" : "M"}${x(p.step).toFixed(1)} ${y(p[key]).toFixed(1)}`)
      .join(" ");

  const trainPath = line("loss");
  const valPath = line("valLoss");
  const area = `${trainPath} L${x(maxStep).toFixed(1)} ${y(0).toFixed(1)} L${x(0).toFixed(1)} ${y(0).toFixed(1)} Z`;

  const gridY = [0, 1, 2, 3]
    .map(
      (v) =>
        `<line class="grid-line" x1="${padL}" y1="${y(v).toFixed(1)}" x2="${W - padR}" y2="${y(v).toFixed(1)}"/>
         <text class="axis-label" x="4" y="${(y(v) + 3).toFixed(1)}">${v.toFixed(1)}</text>`,
    )
    .join("");
  const gridX = [0, Math.round(maxStep / 2), maxStep]
    .map((s) => `<text class="axis-label" x="${x(s).toFixed(1)}" y="${H - 8}" text-anchor="middle">${s}</text>`)
    .join("");

  $("lossChart").innerHTML = `
    <svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Training loss curve">
      <defs>
        <linearGradient id="lossGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#00a85a" stop-opacity="0.22"/>
          <stop offset="100%" stop-color="#00a85a" stop-opacity="0"/>
        </linearGradient>
      </defs>
      ${gridY}${gridX}
      <path class="area" d="${area}"/>
      <path class="line-val" d="${valPath}"/>
      <path class="line-train" d="${trainPath}"/>
    </svg>
    <div class="legend" style="margin:14px 0 0">
      <span><i class="tuned"></i>Train loss</span>
      <span><i class="base"></i>Validation loss</span>
    </div>`;
}

function renderTrainMeta() {
  const t = R.training;
  const items = [
    ["Parameters", t.params || "—"],
    ["Epochs", t.epochs ?? "—"],
    ["Final train loss", t.finalLoss?.toFixed(2) ?? "—"],
    ["Final val loss", t.finalValLoss?.toFixed(2) ?? "—"],
    ["Pretraining", "none"],
  ];
  $("trainMeta").innerHTML = items
    .map(([k, v]) => `<span class="tm">${escapeHtml(k)}<b>${escapeHtml(String(v))}</b></span>`)
    .join("");
}

function renderOod() {
  const o = R.ood;
  const cards = [
    { k: "Anomaly F1", id: o.idF1, ood: o.oodF1 },
    { k: "Next-step Top-1", id: o.idTop1, ood: o.oodTop1 },
    { k: "Completion exact match", id: o.idExactMatch, ood: o.oodExactMatch },
  ].filter((c) => c.id != null && c.ood != null);

  $("oodGrid").innerHTML = cards
    .map(
      (c) =>
        `<div class="ood-card">
           <span class="ood-k">${escapeHtml(c.k)}</span>
           <div class="ood-pair">
             <span class="ood-id">${pct(c.id)}</span>
             <span class="ood-arrow">→</span>
             <span class="ood-ood">${pct(c.ood)}</span>
           </div>
           <span class="ood-drop">−${((c.id - c.ood) * 100).toFixed(1)} pts on unseen ${escapeHtml(o.heldOutFamily)}</span>
         </div>`,
    )
    .join("");
}

function renderTakeaways() {
  $("takeaways").innerHTML = (R.copy.takeaways || [])
    .map((t, i) => `<div class="takeaway"><span class="n">0${i + 1}</span>${escapeHtml(t)}</div>`)
    .join("");
}

/* ------------------------------------------------------------------ *
 *  Live demo — interactive route builder (retrieval baseline)
 * ------------------------------------------------------------------ */
const familyEl = $("family");
const fractionEl = $("fraction");
const sequenceEl = $("sequence");
const stepPickerEl = $("stepPicker");
const statusEl = $("modelStatus");
const CTRL_IDS = ["predict", "complete", "anomaly", "loadSample", "loadInvalid", "addStep", "addTop"];
let lastTopStep = "";
let busy = false;

$("loadSample").addEventListener("click", () => loadSample(false));
$("loadInvalid").addEventListener("click", () => loadSample(true));
$("predict").addEventListener("click", () => withButton("predict", "Predicting", runPredict));
$("complete").addEventListener("click", () => withButton("complete", "Completing", runComplete));
$("anomaly").addEventListener("click", () => withButton("anomaly", "Validating", runAnomaly));
$("addTop").addEventListener("click", () => appendStep(lastTopStep));
$("addStep").addEventListener("click", addPickedStep);
stepPickerEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    addPickedStep();
  }
});
familyEl.addEventListener("change", () => loadSample(false));
fractionEl.addEventListener("change", () => loadSample(false));
sequenceEl.addEventListener("input", () => {
  renderRouteList();
  updateRouteMeta();
});

async function initDemo() {
  const health = await api("/api/health");
  setStatus(health.modelConfigured ? "Live · retrieval + Qwen rerank" : "Live · retrieval baseline", true);
  await loadVocab();
  await loadSample(false);
}

async function loadVocab() {
  try {
    const { steps } = await api("/api/vocab");
    $("vocabList").innerHTML = steps.map((s) => `<option value="${escapeHtml(s)}"></option>`).join("");
  } catch {
    /* picker stays usable as free text */
  }
}

/* ---- busy orchestration: one action at a time, with live feedback ---- */
function setAllDisabled(on) {
  CTRL_IDS.forEach((id) => {
    const b = $(id);
    if (b) b.disabled = on;
  });
  stepPickerEl.disabled = on;
  familyEl.disabled = on;
  fractionEl.disabled = on;
  $("demo").classList.toggle("is-busy", on);
}

async function withBusy(fn, restore) {
  if (busy) return;
  busy = true;
  setAllDisabled(true);
  statusEl.dataset.busy = "true";
  try {
    await fn();
  } catch (e) {
    surfaceError(e);
  } finally {
    if (restore) restore();
    busy = false;
    statusEl.dataset.busy = "";
    setAllDisabled(false);
    $("addTop").disabled = !lastTopStep;
  }
}

async function withButton(id, label, fn) {
  if (busy) return;
  const btn = $(id);
  const prev = btn.innerHTML;
  btn.innerHTML = `<span class="spin"></span>${label}`;
  await withBusy(fn, () => (btn.innerHTML = prev));
}

function surfaceError(e) {
  $("predictions").innerHTML = `<li class="empty">Request failed: ${escapeHtml(e.message || "error")}</li>`;
  $("predHint").hidden = true;
}

/* ---- route state (textarea is the source of truth) ---- */
function getSteps() {
  return sequenceEl.value
    .split(/\n|\|/)
    .map((s) => s.trim())
    .filter(Boolean);
}

function cleanStep(s) {
  return String(s || "").trim().replace(/\s+/g, " ").toUpperCase();
}

function setSteps(steps) {
  sequenceEl.value = steps.join("\n");
  renderRouteList();
  updateRouteMeta();
}

function appendStep(step) {
  const s = cleanStep(step);
  if (!s || busy) return;
  setSteps([...getSteps(), s]);
  withBusy(runPredict);
}

function removeStep(i) {
  if (busy) return;
  const steps = getSteps();
  steps.splice(i, 1);
  setSteps(steps);
}

function addPickedStep() {
  const s = cleanStep(stepPickerEl.value);
  if (!s) return;
  stepPickerEl.value = "";
  appendStep(s);
}

function renderRouteList() {
  const steps = getSteps();
  const host = $("routeList");
  host.innerHTML = "";
  if (!steps.length) {
    host.innerHTML = `<li class="route-empty">No steps yet — load a sample, or add steps below.</li>`;
    return;
  }
  steps.forEach((step, i) => {
    const li = el(
      "li",
      "route-step",
      `<span class="rs-n">${i + 1}</span><span class="rs-name">${escapeHtml(step)}</span><button class="rs-x" type="button" aria-label="Remove step" title="Remove">×</button>`,
    );
    li.querySelector(".rs-x").addEventListener("click", () => removeStep(i));
    host.append(li);
  });
  host.scrollTop = host.scrollHeight;
}

function loadSample(invalid) {
  return withBusy(async () => {
    const sample = await api(`/api/sample?family=${familyEl.value}&fraction=${fractionEl.value}`);
    const seq = invalid ? sample.invalid_sequence : sample.partial_sequence;
    clearOutputs();
    setSteps(seq);
    updateRouteMeta(sample);
    if (invalid) await runAnomaly();
    else await runPredict();
  });
}

async function runPredict() {
  if (!getSteps().length) {
    renderPredictionsEmpty("Add at least one step to predict the next one.");
    return;
  }
  showPredLoading();
  const result = await api("/api/predict", { family: familyEl.value, partial_sequence: getSteps() });
  renderPredictions(result);
}

async function runComplete() {
  if (!getSteps().length) {
    renderPredictionsEmpty("Add at least one step before completing the route.");
    return;
  }
  $("completionMeta").textContent = "running…";
  $("completion").innerHTML = `<span class="loading-row"><span class="spin"></span>Completing route to SHIP LOT…</span>`;
  const result = await api("/api/complete", { family: familyEl.value, partial_sequence: getSteps(), max_steps: 90 });
  const added = result.predicted_sequence.length;
  const total = result.full_sequence?.length || getSteps().length + added;
  $("completionMeta").textContent = `${added} added · ${total} total`;
  $("completion").textContent = added ? result.predicted_sequence.join("\n") : "Route is already complete.";
  renderAnomaly(result.anomaly);
}

async function runAnomaly() {
  $("validity").textContent = "checking…";
  $("validity").className = "verdict";
  $("violations").innerHTML = `<div class="loading-row"><span class="spin"></span>Checking process rules…</div>`;
  const result = await api("/api/anomaly", { family: familyEl.value, sequence: getSteps() });
  renderAnomaly(result);
}

function showPredLoading() {
  $("sourceLabel").textContent = "…";
  $("predHint").hidden = true;
  $("reason").textContent = "";
  $("predictions").innerHTML = `<li class="skel"></li><li class="skel"></li><li class="skel"></li>`;
}

function renderPredictionsEmpty(message) {
  $("sourceLabel").textContent = "—";
  $("predHint").hidden = true;
  $("predictions").innerHTML = `<li class="empty">${escapeHtml(message)}</li>`;
  $("reason").textContent = "";
  lastTopStep = "";
}

function renderPredictions(result) {
  $("sourceLabel").textContent =
    result.source === "model_reranked_retrieval" ? "Qwen reranked" : "retrieval";
  const host = $("predictions");
  host.innerHTML = "";
  lastTopStep = result.predictions[0]?.step || "";
  $("addTop").disabled = !lastTopStep;
  $("predHint").hidden = !result.predictions.length;

  if (!result.predictions.length) {
    host.innerHTML = `<li class="empty">No candidate — the route may already end at SHIP LOT.</li>`;
  }
  result.predictions.forEach((p) => {
    const li = el(
      "li",
      "pickable",
      `<span class="rank">${p.rank}</span>
       <span class="body">
         <span class="step-name">${escapeHtml(p.step)}</span>
         <span class="bar"><span></span></span>
       </span>
       <span class="pct">${Math.round(p.confidence * 100)}%<span class="src">${escapeHtml(p.source || "")}</span></span>`,
    );
    li.title = `Add “${p.step}” to the route`;
    li.addEventListener("click", () => appendStep(p.step));
    host.append(li);
    queueBar(li.querySelector(".bar > span"), `${Math.min(100, p.confidence * 100)}%`);
  });
  $("reason").textContent = result.reason || "";
  flushBars();
}

function renderAnomaly(result) {
  const v = $("validity");
  v.textContent = result.is_valid ? "Valid" : "Invalid";
  v.className = "verdict " + (result.is_valid ? "valid" : "invalid");
  const host = $("violations");
  host.innerHTML = "";

  if (!result.violations.length) {
    host.innerHTML = `<div class="ok">No process-rule violations detected.</div>`;
    return;
  }
  result.violations.slice(0, 8).forEach((vi) => {
    host.append(
      el(
        "div",
        "violation",
        `<span class="rule">${escapeHtml(vi.rule)}</span>
         <div class="where">step ${vi.step_index ?? "—"} · ${escapeHtml(vi.step_name || "")}</div>
         <div class="desc">${escapeHtml(vi.description || "")}</div>`,
      ),
    );
  });
}

function clearOutputs() {
  $("predictions").innerHTML = "";
  $("sourceLabel").textContent = "—";
  $("predHint").hidden = true;
  $("reason").textContent = "";
  $("validity").textContent = "—";
  $("validity").className = "verdict";
  $("violations").innerHTML = "";
  $("completionMeta").textContent = "—";
  $("completion").textContent = "";
  lastTopStep = "";
  $("addTop").disabled = true;
}

function updateRouteMeta(sample) {
  const steps = getSteps();
  $("routeMeta").textContent = `${steps.length} ${steps.length === 1 ? "step" : "steps"}`;
  const frac = sample && sample.fraction ? sample.fraction : Number(fractionEl.value);
  $("progressBar").style.width = `${Math.round(frac * 100)}%`;
}

function setStatus(text, online) {
  statusEl.querySelector(".status-text").textContent = text;
  statusEl.classList.toggle("online", !!online);
}

/* ------------------------------------------------------------------ *
 *  Batch CSV — predict next step for every row
 * ------------------------------------------------------------------ */
const csvFileEl = $("csvFile");
const csvTextEl = $("csvText");
let batchResults = [];

csvFileEl.addEventListener("change", async () => {
  const file = csvFileEl.files?.[0];
  if (!file) return;
  $("csvFileName").textContent = file.name;
  csvTextEl.value = await file.text();
});
$("runBatch").addEventListener("click", () => withButton("runBatch", "Predicting", runBatch));
$("downloadBatch").addEventListener("click", downloadBatchCsv);

async function runBatch() {
  const csv = csvTextEl.value.trim();
  if (!csv) {
    setBatchStatus("Choose a CSV file or paste CSV text first.", true);
    return;
  }
  setBatchStatus("Predicting next step for each row…");
  $("downloadBatch").disabled = true;

  const result = await api("/api/predict-batch", { csv });
  batchResults = result.results || [];
  renderBatch(result);
}

function renderBatch(result) {
  const table = $("batchTable");
  const tb = table.querySelector("tbody");
  tb.innerHTML = "";

  if (!batchResults.length) {
    table.hidden = true;
    setBatchStatus(result.parseErrors?.[0] || "No rows found in the CSV.", true);
    return;
  }

  batchResults.forEach((r) => {
    const predicted = r.error
      ? `<span class="batch-err">${escapeHtml(r.error)}</span>`
      : escapeHtml(r.predicted_next_step || "—");
    const conf = r.confidence != null ? `${Math.round(r.confidence * 100)}%` : "—";
    const unknown = r.unknown_steps?.length
      ? ` <span class="batch-warn" title="Steps not in vocabulary: ${escapeHtml(r.unknown_steps.join(", "))}">⚠ ${r.unknown_steps.length}</span>`
      : "";
    tb.append(
      el(
        "tr",
        r.error ? "batch-row-err" : null,
        `<td>${escapeHtml(r.example_id || "—")}</td>
         <td>${escapeHtml(r.family || "—")}</td>
         <td>${escapeHtml(String(r.completion_fraction ?? "—"))}</td>
         <td>${r.partial_length ?? "—"}${unknown}</td>
         <td class="tuned-col" style="text-align:left">${predicted}</td>
         <td>${conf}</td>`,
      ),
    );
  });

  table.hidden = false;
  const ok = batchResults.filter((r) => !r.error).length;
  const failed = batchResults.length - ok;
  setBatchStatus(
    `${ok} of ${batchResults.length} rows predicted${failed ? ` · ${failed} could not be parsed` : ""}.`,
  );
  $("downloadBatch").disabled = !ok;
}

function downloadBatchCsv() {
  const header = "EXAMPLE_ID,FAMILY,COMPLETION_FRACTION,PARTIAL_LENGTH,PREDICTED_NEXT_STEP,CONFIDENCE,TOP5";
  const rows = batchResults.map((r) =>
    [
      r.example_id ?? "",
      r.family ?? "",
      r.completion_fraction ?? "",
      r.partial_length ?? "",
      r.error ? `ERROR: ${r.error}` : r.predicted_next_step ?? "",
      r.confidence != null ? r.confidence : "",
      r.top5 ? r.top5.map((p) => p.step).join("|") : "",
    ]
      .map(csvCell)
      .join(","),
  );
  const blob = new Blob([[header, ...rows].join("\n")], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "next-step-predictions.csv";
  a.click();
  URL.revokeObjectURL(url);
}

function csvCell(value) {
  const s = String(value);
  return /[",\n]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s;
}

function setBatchStatus(text, isError = false) {
  const node = $("batchStatus");
  node.textContent = text;
  node.classList.toggle("is-error", isError);
}

async function api(path, body) {
  const init = body
    ? { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) }
    : undefined;
  const response = await fetch(path, init);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
