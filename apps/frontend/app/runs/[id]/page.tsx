"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { type Metric, type RunMeta, getMetrics, getRun } from "@/lib/api";

const STATUS_COLORS: Record<string, string> = {
  completed: "text-emerald-400",
  running: "text-sky-400",
  queued: "text-amber-400",
  failed: "text-rose-400",
};

export default function RunDetailPage() {
  const params = useParams<{ id: string }>();
  const id = params.id;
  const [meta, setMeta] = useState<RunMeta | null>(null);
  const [metrics, setMetrics] = useState<Metric[]>([]);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    if (!id) return;
    Promise.all([getRun(id), getMetrics(id)]).then(([m, ms]) => {
      setMeta(m);
      setMetrics(ms);
      setLoaded(true);
    });
  }, [id]);

  const lossRows = useMemo(
    () =>
      metrics
        .filter((m) => typeof m.loss === "number")
        .map((m, i) => ({ step: typeof m.step === "number" ? m.step : i, loss: m.loss as number })),
    [metrics],
  );
  const accRows = useMemo(
    () =>
      metrics
        .filter((m) => typeof m.mean_token_accuracy === "number")
        .map((m, i) => ({ step: typeof m.step === "number" ? m.step : i, acc: m.mean_token_accuracy as number })),
    [metrics],
  );

  if (loaded && !meta) {
    return (
      <div className="mx-auto max-w-5xl">
        <Link href="/" className="text-sm text-neutral-500 hover:underline">
          ← all runs
        </Link>
        <div className="mt-6 rounded-lg border border-neutral-800 p-8 text-center text-neutral-400">
          Run <code className="text-neutral-200">{id}</code> not found.
        </div>
      </div>
    );
  }

  const summary = (meta?.metrics ?? {}) as Record<string, unknown>;

  return (
    <div className="mx-auto max-w-5xl pb-16">
      <Link href="/" className="text-sm text-neutral-500 hover:underline">
        ← all runs
      </Link>

      <div className="mt-2 flex flex-wrap items-baseline justify-between gap-2">
        <h1 className="text-2xl font-semibold">{meta?.name ?? id}</h1>
        <span className={`text-sm font-medium ${STATUS_COLORS[meta?.status ?? ""] ?? "text-neutral-400"}`}>
          {meta?.status ?? "…"}
        </span>
      </div>
      <div className="mt-1 font-mono text-xs text-neutral-500">{id}</div>

      {/* meta chips */}
      <div className="mt-4 flex flex-wrap gap-2 text-xs">
        {meta?.kind && <Chip k="kind" v={meta.kind} />}
        {meta?.slurm_job_id && <Chip k="slurm" v={meta.slurm_job_id} />}
        {meta?.git_branch && <Chip k="branch" v={meta.git_branch} />}
        {meta?.git_sha && <Chip k="sha" v={meta.git_sha} />}
        {meta?.cluster && <Chip k="cluster" v={meta.cluster} />}
        {meta?.tags?.map((t) => (
          <span key={t} className="rounded bg-neutral-800 px-2 py-0.5 font-mono text-neutral-300">
            {t}
          </span>
        ))}
      </div>

      {/* metric summary */}
      {Object.keys(summary).length > 0 && (
        <section className="mt-6">
          <h2 className="mb-2 text-sm font-medium text-neutral-300">Metrics summary</h2>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            {Object.entries(summary)
              .filter(([, v]) => typeof v === "number")
              .slice(0, 16)
              .map(([k, v]) => (
                <div key={k} className="rounded-lg border border-neutral-800 bg-neutral-950 p-3">
                  <div className="truncate text-[11px] text-neutral-500">{k}</div>
                  <div className="mt-0.5 text-lg font-semibold tabular-nums text-neutral-100">
                    {typeof v === "number" ? (v as number).toFixed(4).replace(/\.?0+$/, "") : String(v)}
                  </div>
                </div>
              ))}
          </div>
        </section>
      )}

      {/* loss + accuracy curves */}
      {lossRows.length > 1 && (
        <section className="mt-6 grid grid-cols-1 gap-4 lg:grid-cols-2">
          <Curve title="Loss" rows={lossRows} dataKey="loss" color="#38bdf8" />
          {accRows.length > 1 && (
            <Curve title="Mean token accuracy" rows={accRows} dataKey="acc" color="#34d399" domain={[0, 1]} />
          )}
        </section>
      )}

      {/* config */}
      {meta?.config && Object.keys(meta.config).length > 0 && (
        <section className="mt-6">
          <h2 className="mb-2 text-sm font-medium text-neutral-300">Config</h2>
          <div className="overflow-hidden rounded-lg border border-neutral-800">
            <table className="w-full text-xs">
              <tbody>
                {Object.entries(meta.config).map(([k, v]) => (
                  <tr key={k} className="border-t border-neutral-800 first:border-t-0">
                    <td className="w-48 bg-neutral-900/50 px-3 py-1.5 font-mono text-neutral-400">{k}</td>
                    <td className="px-3 py-1.5 font-mono text-neutral-300">
                      {typeof v === "object" ? JSON.stringify(v) : String(v)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
}

function Chip({ k, v }: { k: string; v: string }) {
  return (
    <span className="rounded border border-neutral-800 px-2 py-0.5 text-neutral-400">
      <span className="text-neutral-600">{k}:</span> <span className="text-neutral-300">{v}</span>
    </span>
  );
}

function Curve({
  title,
  rows,
  dataKey,
  color,
  domain,
}: {
  title: string;
  rows: Record<string, number>[];
  dataKey: string;
  color: string;
  domain?: [number, number];
}) {
  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-950 p-4">
      <div className="mb-2 text-sm font-medium text-neutral-300">{title}</div>
      <ResponsiveContainer width="100%" height={240}>
        <LineChart data={rows} margin={{ top: 8, right: 12, bottom: 8, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
          <XAxis dataKey="step" stroke="#a3a3a3" fontSize={11} />
          <YAxis stroke="#a3a3a3" fontSize={11} domain={domain ?? ["auto", "auto"]} />
          <Tooltip
            contentStyle={{ background: "#0a0a0a", border: "1px solid #404040", borderRadius: 8, fontSize: 12 }}
            formatter={(v) => (typeof v === "number" ? v.toFixed(4) : String(v))}
          />
          <Line type="monotone" dataKey={dataKey} stroke={color} dot={false} strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
