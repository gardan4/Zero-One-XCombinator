"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  type CompareReport,
  type CompareReportRow,
  type Confusion,
  type ExampleComparison,
  type Metric,
  type MetricSpec,
  getCompare,
  getCompareExamples,
  getCompareReport,
  getConfusion,
  getMetrics,
} from "@/lib/api";

const FAMILIES = ["MOSFET", "IGBT", "IC"] as const;
const ID_COLOR = "#34d399";
const OOD_COLOR = "#fb7185";
const TASK_GROUPS = ["all", "nextstep", "completion", "anomaly"] as const;

function num(m: Record<string, unknown>, key: string): number | null {
  const v = m[key];
  return typeof v === "number" ? v : null;
}

function fmtMetric(v: number | null, format: string): string {
  if (v === null) return "·";
  if (format === "pct") return `${(v * 100).toFixed(1)}%`;
  return v.toFixed(3);
}

function flatMetric(row: CompareReportRow, key: string): number | null {
  const v = row.metrics_flat?.[key];
  return typeof v === "number" ? v : null;
}

function isAnomalyRow(r: CompareReportRow): boolean {
  return flatMetric(r, "anomaly_auc") !== null || flatMetric(r, "anomaly_f1") !== null;
}

export default function ComparePage() {
  const [report, setReport] = useState<CompareReport | null>(null);
  const [legacyRuns, setLegacyRuns] = useState<Awaited<ReturnType<typeof getCompare>>>([]);
  const [loaded, setLoaded] = useState(false);
  const [taskGroup, setTaskGroup] = useState<(typeof TASK_GROUPS)[number]>("all");
  const [filterSource, setFilterSource] = useState("local");
  const [filterSplit, setFilterSplit] = useState("");
  const [filterRole, setFilterRole] = useState("");
  const [filterFamily, setFilterFamily] = useState("");
  const [selected, setSelected] = useState<string | null>(null);
  const [confusion, setConfusion] = useState<Confusion>({ tp: 0, fp: 0, tn: 0, fn: 0 });
  const [lossSeries, setLossSeries] = useState<{ runId: string; name: string; data: Metric[] }[]>([]);
  const [diffRunA, setDiffRunA] = useState("");
  const [diffRunB, setDiffRunB] = useState("");
  const [diffTask, setDiffTask] = useState("nextstep");
  const [diffMode, setDiffMode] = useState("different");
  const [diffExamples, setDiffExamples] = useState<ExampleComparison[]>([]);
  const [diffLoading, setDiffLoading] = useState(false);

  const refresh = useCallback(() => {
    const tags: string[] = [];
    getCompareReport({
      tags,
      source: filterSource,
      split: filterSplit || undefined,
      role: filterRole || undefined,
      family: filterFamily || undefined,
    }).then((r) => {
      setReport(r);
      setLoaded(true);
    });
    getCompare([]).then(setLegacyRuns);
  }, [filterSource, filterSplit, filterRole, filterFamily]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const rows = report?.rows ?? [];
  const metricSpecs = report?.metric_specs ?? [];
  const visibleSpecs = useMemo(
    () =>
      metricSpecs.filter((s: MetricSpec) => taskGroup === "all" || s.task === taskGroup),
    [metricSpecs, taskGroup],
  );

  const anomalyRows = useMemo(() => rows.filter(isAnomalyRow), [rows]);

  useEffect(() => {
    if (selected || anomalyRows.length === 0) return;
    const ood = anomalyRows.find((r) => r.dataset?.split === "ood");
    setSelected((ood ?? anomalyRows[0]).run_id);
  }, [anomalyRows, selected]);

  useEffect(() => {
    if (!selected) return;
    getConfusion(selected).then(setConfusion);
  }, [selected]);

  useEffect(() => {
    if (!loaded) return;
    const trainish = legacyRuns.filter(
      (r) => r.kind === "sft" || r.kind === "grpo" || num(r.metrics, "loss") !== null,
    );
    Promise.all(
      trainish.slice(0, 6).map(async (r) => ({ runId: r.id, name: r.name, data: await getMetrics(r.id) })),
    ).then((all) => setLossSeries(all.filter((s) => s.data.some((m) => typeof m.loss === "number"))));
  }, [legacyRuns, loaded]);

  const loadDiff = () => {
    if (!diffRunA || !diffRunB) return;
    setDiffLoading(true);
    getCompareExamples(diffRunA, diffRunB, diffTask, diffMode, 25)
      .then((res) => setDiffExamples(res?.examples ?? []))
      .finally(() => setDiffLoading(false));
  };

  const heroData = useMemo(() => {
    const pick = (s: string, fam: string) => {
      const r = anomalyRows.find((x) => x.dataset?.split === s && x.dataset?.family === fam);
      if (!r) return null;
      return flatMetric(r, `anomaly_auc_${fam}`) ?? flatMetric(r, "anomaly_auc");
    };
    return FAMILIES.map((fam) => ({
      family: fam,
      ID: pick("id", fam),
      OOD: pick("ood", fam),
    })).filter((row) => row.ID !== null || row.OOD !== null);
  }, [anomalyRows]);

  const predictorData = useMemo(() => {
    const byPred = new Map<string, { auc: number[]; f1: number[] }>();
    for (const r of anomalyRows) {
      const p = r.model?.predictor ?? r.kind ?? "?";
      const auc = flatMetric(r, "anomaly_auc");
      const f1 = flatMetric(r, "anomaly_f1");
      const e = byPred.get(p) ?? { auc: [], f1: [] };
      if (auc !== null) e.auc.push(auc);
      if (f1 !== null) e.f1.push(f1);
      byPred.set(p, e);
    }
    const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);
    return [...byPred.entries()].map(([predictor, v]) => ({
      predictor,
      AUC: mean(v.auc),
      F1: mean(v.f1),
    }));
  }, [anomalyRows]);

  const lossChart = useMemo(() => {
    if (lossSeries.length === 0)
      return { rows: [] as Record<string, number>[], keys: [] as string[], labels: {} as Record<string, string> };
    const keys = lossSeries.map((s) => s.runId);
    const labels = Object.fromEntries(lossSeries.map((s) => [s.runId, s.name || s.runId]));
    const maxLen = Math.max(...lossSeries.map((s) => s.data.length));
    const chartRows: Record<string, number>[] = [];
    for (let i = 0; i < maxLen; i++) {
      const row: Record<string, number> = { step: i };
      lossSeries.forEach((s) => {
        const v = s.data[i]?.loss;
        if (typeof v === "number") row[s.runId] = v;
      });
      chartRows.push(row);
    }
    return { rows: chartRows, keys, labels };
  }, [lossSeries]);

  const selectedRun = anomalyRows.find((r) => r.run_id === selected) ?? null;
  const lineColors = ["#38bdf8", "#a78bfa", "#f472b6", "#fbbf24", "#34d399", "#f87171"];
  const evalRunIds = rows.map((r) => r.run_id);

  return (
    <div className="mx-auto max-w-7xl">
      <Link href="/" className="text-sm text-neutral-500 hover:underline">
        ← all runs
      </Link>
      <h1 className="mt-2 text-2xl font-semibold">Comparison dashboard</h1>
      <p className="mt-1 text-sm text-neutral-500">
        Compare eval runs by model ref, predictor, version, and tags. Use disagreements to inspect
        per-example differences and reasoning traces.
      </p>

      <div className="mt-4 flex flex-wrap gap-2 text-xs">
        <FilterSelect label="Source" value={filterSource} onChange={setFilterSource} options={["local", "wandb", "repo"]} />
        <FilterSelect label="Split" value={filterSplit} onChange={setFilterSplit} options={["", "id", "ood"]} />
        <FilterSelect label="Role" value={filterRole} onChange={setFilterRole} options={["", "baseline", "finetuned", "oracle"]} />
        <FilterSelect label="Family" value={filterFamily} onChange={setFilterFamily} options={["", ...FAMILIES]} />
        <FilterSelect
          label="Metrics"
          value={taskGroup}
          onChange={(v) => setTaskGroup(v as (typeof TASK_GROUPS)[number])}
          options={[...TASK_GROUPS]}
        />
      </div>

      <RunMatrix rows={rows} specs={visibleSpecs} deltas={report?.deltas_vs_baseline} />

      <section className="mt-8 rounded-lg border border-neutral-800 bg-neutral-950 p-4">
        <h2 className="mb-3 text-sm font-medium text-neutral-300">Per-example disagreements</h2>
        <div className="mb-3 flex flex-wrap gap-2 text-xs">
          <select
            value={diffRunA}
            onChange={(e) => setDiffRunA(e.target.value)}
            className="rounded border border-neutral-700 bg-neutral-900 px-2 py-1"
          >
            <option value="">Run A…</option>
            {evalRunIds.map((id) => (
              <option key={id} value={id}>
                {rows.find((r) => r.run_id === id)?.run_name ?? id}
              </option>
            ))}
          </select>
          <select
            value={diffRunB}
            onChange={(e) => setDiffRunB(e.target.value)}
            className="rounded border border-neutral-700 bg-neutral-900 px-2 py-1"
          >
            <option value="">Run B…</option>
            {evalRunIds.map((id) => (
              <option key={id} value={id}>
                {rows.find((r) => r.run_id === id)?.run_name ?? id}
              </option>
            ))}
          </select>
          <select
            value={diffTask}
            onChange={(e) => setDiffTask(e.target.value)}
            className="rounded border border-neutral-700 bg-neutral-900 px-2 py-1"
          >
            <option value="nextstep">nextstep</option>
            <option value="completion">completion</option>
            <option value="anomaly">anomaly</option>
          </select>
          <select
            value={diffMode}
            onChange={(e) => setDiffMode(e.target.value)}
            className="rounded border border-neutral-700 bg-neutral-900 px-2 py-1"
          >
            <option value="different">different</option>
            <option value="both_wrong">both wrong</option>
            <option value="only_a_correct">only A correct</option>
            <option value="only_b_correct">only B correct</option>
          </select>
          <button
            type="button"
            onClick={loadDiff}
            disabled={!diffRunA || !diffRunB || diffLoading}
            className="rounded bg-neutral-700 px-3 py-1 text-neutral-100 hover:bg-neutral-600 disabled:opacity-40"
          >
            {diffLoading ? "Loading…" : "Compare examples"}
          </button>
        </div>
        {diffExamples.length === 0 ? (
          <p className="text-xs text-neutral-600">
            Select two runs with <code className="text-neutral-400">examples.jsonl</code> (eval with traces enabled).
          </p>
        ) : (
          <div className="max-h-96 space-y-3 overflow-auto text-xs">
            {diffExamples.map((ex) => (
              <DiffCard key={ex.example_id} ex={ex} />
            ))}
          </div>
        )}
      </section>

      {loaded && anomalyRows.length === 0 && (
        <div className="mt-6 rounded-lg border border-neutral-800 p-6 text-sm text-neutral-400">
          No anomaly eval runs match filters. Run{" "}
          <code className="text-neutral-200">zo-track predict -p ngram … --gold … --tasks anomaly</code>
        </div>
      )}

      <section className="mt-8">
        <h2 className="mb-1 text-sm font-medium text-neutral-300">Anomaly ROC-AUC — ID vs OOD</h2>
        <div className="rounded-lg border border-neutral-800 bg-neutral-950 p-4">
          {heroData.length === 0 ? (
            <div className="py-12 text-center text-sm text-neutral-600">no per-family anomaly AUC yet</div>
          ) : (
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={heroData} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
                <XAxis dataKey="family" stroke="#a3a3a3" fontSize={12} />
                <YAxis domain={[0, 1]} stroke="#a3a3a3" fontSize={12} />
                <Tooltip contentStyle={{ background: "#0a0a0a", border: "1px solid #404040", borderRadius: 8, fontSize: 12 }} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="ID" name="In-distribution" fill={ID_COLOR} radius={[3, 3, 0, 0]} />
                <Bar dataKey="OOD" name="Out-of-distribution" fill={OOD_COLOR} radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      </section>

      <section className="mt-8">
        <h2 className="mb-3 text-sm font-medium text-neutral-300">Anomaly by predictor</h2>
        <div className="rounded-lg border border-neutral-800 bg-neutral-950 p-4">
          {predictorData.length === 0 ? (
            <div className="py-12 text-center text-sm text-neutral-600">no anomaly predictors yet</div>
          ) : (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={predictorData} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
                <XAxis dataKey="predictor" stroke="#a3a3a3" fontSize={12} />
                <YAxis domain={[0, 1]} stroke="#a3a3a3" fontSize={12} />
                <Tooltip contentStyle={{ background: "#0a0a0a", border: "1px solid #404040", borderRadius: 8, fontSize: 12 }} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="AUC" name="ROC-AUC" fill="#38bdf8" radius={[3, 3, 0, 0]} />
                <Bar dataKey="F1" name="F1" fill="#a78bfa" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      </section>

      <section className="mt-8">
        <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
          <h2 className="text-sm font-medium text-neutral-300">Anomaly confusion matrix</h2>
          {anomalyRows.length > 0 && (
            <select
              value={selected ?? ""}
              onChange={(e) => setSelected(e.target.value)}
              className="rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1 text-xs text-neutral-200"
            >
              {anomalyRows.map((r) => (
                <option key={r.run_id} value={r.run_id}>
                  {r.run_name} [{r.dataset?.split ?? "?"}/{r.model?.predictor ?? "?"}]
                </option>
              ))}
            </select>
          )}
        </div>
        <div className="rounded-lg border border-neutral-800 bg-neutral-950 p-4">
          {selectedRun ? (
            <>
              <div className="mb-3 text-xs text-neutral-500">
                {selectedRun.model?.display} · {selectedRun.run_id}
              </div>
              <ConfusionGrid c={confusion} />
            </>
          ) : (
            <div className="py-8 text-center text-sm text-neutral-600">select an anomaly run</div>
          )}
        </div>
      </section>

      <section className="mt-8 mb-12">
        <h2 className="mb-3 text-sm font-medium text-neutral-300">Training loss curves</h2>
        <div className="rounded-lg border border-neutral-800 bg-neutral-950 p-4">
          {lossChart.rows.length < 2 ? (
            <div className="py-12 text-center text-sm text-neutral-600">no training loss curves yet</div>
          ) : (
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={lossChart.rows} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
                <XAxis dataKey="step" stroke="#a3a3a3" fontSize={12} />
                <YAxis stroke="#a3a3a3" fontSize={12} />
                <Tooltip contentStyle={{ background: "#0a0a0a", border: "1px solid #404040", borderRadius: 8, fontSize: 12 }} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                {lossChart.keys.map((k, i) => (
                  <Line
                    key={k}
                    type="monotone"
                    dataKey={k}
                    name={lossChart.labels[k]}
                    stroke={lineColors[i % lineColors.length]}
                    dot={false}
                    strokeWidth={2}
                    connectNulls
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>
      </section>
    </div>
  );
}

function FilterSelect({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: string[];
}) {
  return (
    <label className="flex items-center gap-1 text-neutral-500">
      {label}
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded border border-neutral-700 bg-neutral-900 px-2 py-1 text-neutral-200"
      >
        {options.map((o) => (
          <option key={o || "_"} value={o}>
            {o || "any"}
          </option>
        ))}
      </select>
    </label>
  );
}

function RunMatrix({
  rows,
  specs,
  deltas,
}: {
  rows: CompareReportRow[];
  specs: MetricSpec[];
  deltas?: Record<string, Record<string, number | null>>;
}) {
  if (rows.length === 0) {
    return (
      <section className="mt-6">
        <div className="rounded-lg border border-neutral-800 bg-neutral-950 p-6 text-sm text-neutral-500">
          No eval runs match filters — run <code className="text-neutral-200">zo-track predict</code> with gold.
        </div>
      </section>
    );
  }
  const splitTone = (s: string | null | undefined) =>
    s === "ood" ? "text-rose-300" : s === "id" ? "text-emerald-300" : "text-neutral-400";

  return (
    <section className="mt-6">
      <h2 className="mb-2 text-sm font-medium text-neutral-300">Run comparison — all documented metrics</h2>
      <div className="overflow-auto rounded-lg border border-neutral-800">
        <table className="w-full text-xs">
          <thead className="bg-neutral-900 text-left text-neutral-400">
            <tr>
              <th className="px-3 py-2 font-medium">run</th>
              <th className="px-3 py-2 font-medium">model</th>
              <th className="px-3 py-2 font-medium">role</th>
              <th className="px-3 py-2 font-medium">split</th>
              <th className="px-3 py-2 font-medium">family</th>
              {specs.map((c) => (
                <th key={c.key} className="px-3 py-2 text-right font-medium" title={c.task}>
                  {c.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const d = r.dataset;
              const delta = deltas?.[r.run_id];
              return (
                <tr key={r.run_id} className="border-t border-neutral-800 hover:bg-neutral-900/40">
                  <td className="px-3 py-1.5">
                    <Link href={`/runs/${r.run_id}`} className="text-neutral-200 hover:underline">
                      {r.run_name}
                    </Link>
                    <div className="font-mono text-[10px] text-neutral-600">{r.run_id.slice(-12)}</div>
                  </td>
                  <td className="max-w-[140px] truncate px-3 py-1.5 text-neutral-300" title={r.model?.display}>
                    {r.model?.model_ref || r.model?.display || "—"}
                  </td>
                  <td className="px-3 py-1.5 text-neutral-400">{r.model?.role ?? "—"}</td>
                  <td className={`px-3 py-1.5 font-medium ${splitTone(d?.split)}`}>{d?.split ?? "—"}</td>
                  <td className="px-3 py-1.5 text-neutral-400">{d?.family ?? "—"}</td>
                  {specs.map((c) => {
                    const v = flatMetric(r, c.key);
                    const dv = delta?.[c.key];
                    return (
                      <td key={c.key} className="px-3 py-1.5 text-right tabular-nums text-neutral-300">
                        {fmtMetric(v, c.format)}
                        {typeof dv === "number" && Math.abs(dv) > 0.0001 && (
                          <span className={`ml-1 text-[10px] ${dv > 0 ? "text-emerald-500" : "text-rose-500"}`}>
                            {dv > 0 ? "+" : ""}
                            {c.format === "pct" ? (dv * 100).toFixed(1) : dv.toFixed(2)}
                          </span>
                        )}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function DiffCard({ ex }: { ex: ExampleComparison }) {
  const reasonA = (ex.a?.trace as Record<string, string>)?.reasoning;
  const reasonB = (ex.b?.trace as Record<string, string>)?.reasoning;
  return (
    <div className="rounded border border-neutral-800 bg-neutral-900/50 p-3">
      <div className="font-mono text-neutral-400">
        {ex.example_id} · {ex.family}
      </div>
      <div className="mt-2 grid gap-2 sm:grid-cols-2">
        <Side label="A" pred={ex.a?.prediction} ok={ex.a?.correct} reason={reasonA} />
        <Side label="B" pred={ex.b?.prediction} ok={ex.b?.correct} reason={reasonB} />
      </div>
      {ex.gold != null && (
        <div className="mt-2 text-neutral-500">
          gold: <code className="text-neutral-400">{JSON.stringify(ex.gold).slice(0, 120)}</code>
        </div>
      )}
    </div>
  );
}

function Side({
  label,
  pred,
  ok,
  reason,
}: {
  label: string;
  pred: unknown;
  ok?: boolean | null;
  reason?: string;
}) {
  return (
    <div>
      <span className="font-medium text-neutral-300">
        {label}{" "}
        {ok === true && <span className="text-emerald-400">✓</span>}
        {ok === false && <span className="text-rose-400">✗</span>}
      </span>
      <pre className="mt-1 max-h-24 overflow-auto whitespace-pre-wrap text-[10px] text-neutral-400">
        {JSON.stringify(pred, null, 0)}
      </pre>
      {reason && (
        <p className="mt-1 text-[10px] italic text-neutral-500">reasoning: {reason.slice(0, 280)}</p>
      )}
    </div>
  );
}

function ConfusionGrid({ c }: { c: Confusion }) {
  const total = c.tp + c.fp + c.tn + c.fn || 1;
  const cell = (label: string, value: number, tone: string, sub: string) => (
    <div className={`rounded-lg border p-4 ${tone}`}>
      <div className="text-xs uppercase tracking-wide opacity-70">{label}</div>
      <div className="mt-1 text-2xl font-semibold tabular-nums">{value}</div>
      <div className="text-xs opacity-60">
        {sub} · {((value / total) * 100).toFixed(1)}%
      </div>
    </div>
  );
  return (
    <div>
      <div className="mb-2 grid grid-cols-[auto_1fr_1fr] gap-2 text-xs text-neutral-500">
        <div />
        <div className="text-center">pred INVALID</div>
        <div className="text-center">pred VALID</div>
      </div>
      <div className="grid grid-cols-[auto_1fr_1fr] items-center gap-2">
        <div className="text-xs text-neutral-500">true INVALID</div>
        {cell("TP", c.tp, "border-emerald-700/50 bg-emerald-950/40 text-emerald-200", "caught")}
        {cell("FN", c.fn, "border-rose-700/50 bg-rose-950/40 text-rose-200", "missed")}
        <div className="text-xs text-neutral-500">true VALID</div>
        {cell("FP", c.fp, "border-amber-700/50 bg-amber-950/40 text-amber-200", "false alarm")}
        {cell("TN", c.tn, "border-neutral-700/50 bg-neutral-900 text-neutral-200", "correct")}
      </div>
    </div>
  );
}
