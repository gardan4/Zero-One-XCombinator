"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  type CompareRun,
  type Confusion,
  type Metric,
  getCompare,
  getConfusion,
  getMetrics,
} from "@/lib/api";

const FAMILIES = ["MOSFET", "IGBT", "IC"] as const;
const ID_COLOR = "#34d399"; // emerald — in-distribution
const OOD_COLOR = "#fb7185"; // rose — out-of-distribution (the collapse)

function num(m: Record<string, unknown>, key: string): number | null {
  const v = m[key];
  return typeof v === "number" ? v : null;
}

function tagValue(tags: string[], prefix: string): string | null {
  const t = tags.find((x) => x.startsWith(`${prefix}:`));
  return t ? t.slice(prefix.length + 1) : null;
}

// A run is an anomaly run if it carries an anomaly metric (the driver writes anomaly_auc/acc).
function isAnomaly(r: CompareRun): boolean {
  return num(r.metrics, "anomaly_auc") !== null || num(r.metrics, "anomaly_acc") !== null;
}

function split(r: CompareRun): "id" | "ood" | null {
  const s = tagValue(r.tags, "split");
  return s === "id" || s === "ood" ? s : null;
}

export default function ComparePage() {
  const [runs, setRuns] = useState<CompareRun[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);
  const [confusion, setConfusion] = useState<Confusion>({ tp: 0, fp: 0, tn: 0, fn: 0 });
  const [lossSeries, setLossSeries] = useState<{ runId: string; name: string; data: Metric[] }[]>([]);

  // 1) all runs (no tag filter) — group client-side by split / predictor / family.
  useEffect(() => {
    getCompare([]).then((rs) => {
      setRuns(rs);
      setLoaded(true);
    });
  }, []);

  const anomalyRuns = useMemo(() => runs.filter(isAnomaly), [runs]);

  // Default the confusion-matrix selection to the first OOD anomaly run (the interesting case),
  // else the first anomaly run.
  useEffect(() => {
    if (selected || anomalyRuns.length === 0) return;
    const ood = anomalyRuns.find((r) => split(r) === "ood");
    setSelected((ood ?? anomalyRuns[0]).id);
  }, [anomalyRuns, selected]);

  // 2) confusion matrix for the selected run (via the dedicated endpoint).
  useEffect(() => {
    if (!selected) return;
    getConfusion(selected).then(setConfusion);
  }, [selected]);

  // 3) loss curves: any run whose metric stream has a numeric `loss` (training runs).
  useEffect(() => {
    if (!loaded) return;
    const trainish = runs.filter((r) => r.kind === "sft" || r.kind === "grpo" || num(r.metrics, "loss") !== null);
    Promise.all(
      trainish.slice(0, 6).map(async (r) => ({ runId: r.id, name: r.name, data: await getMetrics(r.id) })),
    ).then((all) => setLossSeries(all.filter((s) => s.data.some((m) => typeof m.loss === "number"))));
  }, [runs, loaded]);

  // --- Hero: ID vs OOD anomaly AUC per family (the headline collapse) ---
  const heroData = useMemo(() => {
    const pick = (s: "id" | "ood", fam: string) => {
      const r = anomalyRuns.find((x) => split(x) === s && tagValue(x.tags, "family") === fam);
      if (!r) return null;
      // Prefer the family-suffixed metric, fall back to the overall AUC for a single-family run.
      return num(r.metrics, `anomaly_auc_${fam}`) ?? num(r.metrics, "anomaly_auc");
    };
    return FAMILIES.map((fam) => ({
      family: fam,
      ID: pick("id", fam),
      OOD: pick("ood", fam),
    })).filter((row) => row.ID !== null || row.OOD !== null);
  }, [anomalyRuns]);

  // --- Predictor comparison: anomaly AUC + F1 by predictor (baseline vs trained) ---
  const predictorData = useMemo(() => {
    const byPred = new Map<string, { auc: number[]; f1: number[] }>();
    for (const r of anomalyRuns) {
      const p = tagValue(r.tags, "predictor") ?? r.kind ?? "?";
      const auc = num(r.metrics, "anomaly_auc");
      const f1 = num(r.metrics, "anomaly_f1");
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
  }, [anomalyRuns]);

  // --- Loss-curve overlay: align series by step index ---
  const lossChart = useMemo(() => {
    if (lossSeries.length === 0)
      return { rows: [] as Record<string, number>[], keys: [] as string[], labels: {} as Record<string, string> };
    const keys = lossSeries.map((s) => s.runId); // unique — run names can collide across sessions
    const labels = Object.fromEntries(lossSeries.map((s) => [s.runId, s.name || s.runId]));
    const maxLen = Math.max(...lossSeries.map((s) => s.data.length));
    const rows: Record<string, number>[] = [];
    for (let i = 0; i < maxLen; i++) {
      const row: Record<string, number> = { step: i };
      lossSeries.forEach((s) => {
        const v = s.data[i]?.loss;
        if (typeof v === "number") row[s.runId] = v;
      });
      rows.push(row);
    }
    return { rows, keys, labels };
  }, [lossSeries]);

  const selectedRun = anomalyRuns.find((r) => r.id === selected) ?? null;
  const lineColors = ["#38bdf8", "#a78bfa", "#f472b6", "#fbbf24", "#34d399", "#f87171"];

  return (
    <div className="mx-auto max-w-6xl">
      <Link href="/" className="text-sm text-neutral-500 hover:underline">
        ← all runs
      </Link>
      <h1 className="mt-2 text-2xl font-semibold">Comparison dashboard</h1>
      <p className="mt-1 text-sm text-neutral-500">
        Industrial AI track — process-logic generalization. The headline is the anomaly detector&apos;s
        in-distribution vs out-of-distribution collapse.
      </p>

      {/* (0) Run comparison matrix — every eval run, key metrics side by side */}
      <RunMatrix runs={runs} />

      {loaded && anomalyRuns.length === 0 && (
        <div className="mt-6 rounded-lg border border-neutral-800 p-6 text-sm text-neutral-400">
          No anomaly runs yet. Generate some with{" "}
          <code className="text-neutral-200">
            zo-track predict -p ngram --anomaly … --gold … --tasks anomaly --tags split:id,family:MOSFET
          </code>{" "}
          (and again with <code className="text-neutral-200">--train-families IGBT,IC --tags split:ood,…</code> for OOD).
        </div>
      )}

      {/* (a) HERO — ID vs OOD anomaly AUC per family */}
      <section className="mt-8">
        <h2 className="mb-1 text-sm font-medium text-neutral-300">
          Anomaly ROC-AUC — in-distribution vs out-of-distribution, per family
        </h2>
        <p className="mb-3 text-xs text-neutral-500">
          The learned detector knows process validity in-distribution but collapses toward chance (0.5)
          on a held-out family — logic vs. memorization.
        </p>
        <div className="rounded-lg border border-neutral-800 bg-neutral-950 p-4">
          {heroData.length === 0 ? (
            <div className="py-12 text-center text-sm text-neutral-600">no per-family anomaly AUC yet</div>
          ) : (
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={heroData} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
                <XAxis dataKey="family" stroke="#a3a3a3" fontSize={12} />
                <YAxis domain={[0, 1]} stroke="#a3a3a3" fontSize={12} />
                <Tooltip
                  contentStyle={{ background: "#0a0a0a", border: "1px solid #404040", borderRadius: 8, fontSize: 12 }}
                  formatter={(v) => (typeof v === "number" ? v.toFixed(4) : String(v))}
                />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="ID" name="In-distribution" fill={ID_COLOR} radius={[3, 3, 0, 0]} />
                <Bar dataKey="OOD" name="Out-of-distribution" fill={OOD_COLOR} radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      </section>

      {/* (b) Predictor comparison — baseline vs trained */}
      <section className="mt-8">
        <h2 className="mb-3 text-sm font-medium text-neutral-300">
          Anomaly detector by predictor — baseline (n-gram / oracle) vs trained
        </h2>
        <div className="rounded-lg border border-neutral-800 bg-neutral-950 p-4">
          {predictorData.length === 0 ? (
            <div className="py-12 text-center text-sm text-neutral-600">no anomaly predictors yet</div>
          ) : (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={predictorData} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
                <XAxis dataKey="predictor" stroke="#a3a3a3" fontSize={12} />
                <YAxis domain={[0, 1]} stroke="#a3a3a3" fontSize={12} />
                <Tooltip
                  contentStyle={{ background: "#0a0a0a", border: "1px solid #404040", borderRadius: 8, fontSize: 12 }}
                  formatter={(v) => (typeof v === "number" ? v.toFixed(4) : String(v))}
                />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="AUC" name="ROC-AUC" fill="#38bdf8" radius={[3, 3, 0, 0]} />
                <Bar dataKey="F1" name="F1 (invalid class)" fill="#a78bfa" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      </section>

      {/* (c) Confusion matrix (2×2 Tailwind grid) + run picker */}
      <section className="mt-8">
        <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
          <h2 className="text-sm font-medium text-neutral-300">Anomaly confusion matrix</h2>
          {anomalyRuns.length > 0 && (
            <select
              value={selected ?? ""}
              onChange={(e) => setSelected(e.target.value)}
              className="rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1 text-xs text-neutral-200"
            >
              {anomalyRuns.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.name} [{tagValue(r.tags, "split") ?? "?"}/{tagValue(r.tags, "predictor") ?? r.kind}]
                </option>
              ))}
            </select>
          )}
        </div>
        <div className="rounded-lg border border-neutral-800 bg-neutral-950 p-4">
          {selectedRun ? (
            <>
              <div className="mb-3 text-xs text-neutral-500">
                {selectedRun.name} · {selectedRun.id}
              </div>
              <ConfusionGrid c={confusion} />
            </>
          ) : (
            <div className="py-8 text-center text-sm text-neutral-600">select an anomaly run</div>
          )}
        </div>
      </section>

      {/* (d) Loss-curve overlay */}
      <section className="mt-8 mb-12">
        <h2 className="mb-3 text-sm font-medium text-neutral-300">Training loss curves</h2>
        <div className="rounded-lg border border-neutral-800 bg-neutral-950 p-4">
          {lossChart.rows.length < 2 ? (
            <div className="py-12 text-center text-sm text-neutral-600">
              no training runs with a loss curve yet (SFT/GRPO runs appear here)
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={lossChart.rows} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
                <XAxis dataKey="step" stroke="#a3a3a3" fontSize={12} />
                <YAxis stroke="#a3a3a3" fontSize={12} />
                <Tooltip
                  contentStyle={{ background: "#0a0a0a", border: "1px solid #404040", borderRadius: 8, fontSize: 12 }}
                />
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

const METRIC_COLS: { key: string; label: string; group: string }[] = [
  { key: "top1", label: "Top-1", group: "next-step" },
  { key: "top5", label: "Top-5", group: "next-step" },
  { key: "mrr", label: "MRR", group: "next-step" },
  { key: "block_acc", label: "Block", group: "completion" },
  { key: "ned", label: "NED", group: "completion" },
  { key: "anomaly_f1", label: "Anom F1", group: "anomaly" },
  { key: "anomaly_auc", label: "Anom AUC", group: "anomaly" },
];

// All eval runs, their key metrics side by side, ID/OOD-colored. The "compare runs" surface.
function RunMatrix({ runs }: { runs: CompareRun[] }) {
  const evalRuns = runs.filter(
    (r) => r.kind === "eval" || METRIC_COLS.some((c) => num(r.metrics, c.key) !== null),
  );
  if (evalRuns.length === 0) {
    return (
      <section className="mt-6">
        <div className="rounded-lg border border-neutral-800 bg-neutral-950 p-6 text-sm text-neutral-500">
          No eval runs yet — run the OOD eval (or an n-gram baseline) to populate the comparison.
        </div>
      </section>
    );
  }
  const splitTone = (s: string | null) =>
    s === "ood" ? "text-rose-300" : s === "id" ? "text-emerald-300" : "text-neutral-400";
  return (
    <section className="mt-6">
      <h2 className="mb-2 text-sm font-medium text-neutral-300">Run comparison — eval metrics</h2>
      <div className="overflow-auto rounded-lg border border-neutral-800">
        <table className="w-full text-xs">
          <thead className="bg-neutral-900 text-left text-neutral-400">
            <tr>
              <th className="px-3 py-2 font-medium">run</th>
              <th className="px-3 py-2 font-medium">split</th>
              <th className="px-3 py-2 font-medium">family</th>
              <th className="px-3 py-2 font-medium">predictor</th>
              {METRIC_COLS.map((c) => (
                <th key={c.key} className="px-3 py-2 text-right font-medium" title={c.group}>
                  {c.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {evalRuns.map((r) => {
              const s = tagValue(r.tags, "split");
              return (
                <tr key={r.id} className="border-t border-neutral-800 hover:bg-neutral-900/40">
                  <td className="px-3 py-1.5">
                    <Link href={`/runs/${r.id}`} className="text-neutral-200 hover:underline">
                      {r.name}
                    </Link>
                  </td>
                  <td className={`px-3 py-1.5 font-medium ${splitTone(s)}`}>{s ?? "—"}</td>
                  <td className="px-3 py-1.5 text-neutral-400">{tagValue(r.tags, "family") ?? "—"}</td>
                  <td className="px-3 py-1.5 text-neutral-400">{tagValue(r.tags, "predictor") ?? r.kind}</td>
                  {METRIC_COLS.map((c) => {
                    const v = num(r.metrics, c.key);
                    return (
                      <td key={c.key} className="px-3 py-1.5 text-right tabular-nums text-neutral-300">
                        {v === null ? "·" : v.toFixed(3)}
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
