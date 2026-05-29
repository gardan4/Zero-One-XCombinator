import Link from "next/link";

import { getMetrics, getRun, type Metric } from "@/lib/api";

export const dynamic = "force-dynamic";

function series(metrics: Metric[], key: string): number[] {
  return metrics.map((m) => m[key]).filter((v) => typeof v === "number") as number[];
}

function Sparkline({ data }: { data: number[] }) {
  if (data.length < 2) return <span className="text-neutral-600">not enough data yet</span>;
  const w = 600;
  const h = 120;
  const pad = 8;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const span = max - min || 1;
  const points = data
    .map((v, i) => {
      const x = pad + (i / (data.length - 1)) * (w - 2 * pad);
      const y = h - pad - ((v - min) / span) * (h - 2 * pad);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full text-sky-400">
      <polyline points={points} fill="none" stroke="currentColor" strokeWidth="2" />
    </svg>
  );
}

export default async function RunPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const run = await getRun(id);
  const metrics = await getMetrics(id);

  if (!run) {
    return (
      <div className="text-neutral-400">
        Run not found.{" "}
        <Link href="/" className="underline">
          Back to all runs
        </Link>
      </div>
    );
  }

  const plotKey =
    ["loss", "reward", "running_acc", "success", "accuracy"].find((k) => series(metrics, k).length >= 2) ?? null;
  const data = plotKey ? series(metrics, plotKey) : [];

  const facts: [string, string][] = [
    ["kind", run.kind],
    ["status", run.status],
    ["branch", run.git_branch ?? "—"],
    ["slurm job", run.slurm_job_id ?? "—"],
  ];

  return (
    <div className="mx-auto max-w-4xl">
      <Link href="/" className="text-sm text-neutral-500 hover:underline">
        ← all runs
      </Link>
      <h1 className="mt-2 text-2xl font-semibold">{run.name}</h1>
      <div className="mt-1 text-sm text-neutral-500">{run.id}</div>

      <div className="mt-4 grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
        {facts.map(([k, v]) => (
          <div key={k} className="rounded-lg border border-neutral-800 p-3">
            <div className="text-xs uppercase tracking-wide text-neutral-500">{k}</div>
            <div className="mt-1 text-neutral-100">{v}</div>
          </div>
        ))}
      </div>

      <section className="mt-8">
        <h2 className="mb-2 text-sm font-medium text-neutral-400">
          {plotKey ? `${plotKey} over ${data.length} steps` : "metrics"}
        </h2>
        <div className="rounded-lg border border-neutral-800 p-4">
          <Sparkline data={data} />
        </div>
      </section>

      <section className="mt-8">
        <h2 className="mb-2 text-sm font-medium text-neutral-400">summary</h2>
        <pre className="overflow-auto rounded-lg border border-neutral-800 bg-neutral-900 p-4 text-xs text-neutral-300">
          {JSON.stringify(run.metrics, null, 2)}
        </pre>
      </section>
    </div>
  );
}
