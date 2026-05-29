import Link from "next/link";

import { getRuns, type RunMeta } from "@/lib/api";

export const dynamic = "force-dynamic";

const STATUS_COLORS: Record<string, string> = {
  completed: "text-emerald-400",
  running: "text-sky-400",
  queued: "text-amber-400",
  failed: "text-rose-400",
  created: "text-neutral-400",
  cancelled: "text-neutral-500",
};

function topMetric(m: Record<string, unknown>): string {
  for (const k of ["accuracy", "success_rate", "reward", "loss"]) {
    if (k in m) return `${k}=${m[k]}`;
  }
  const first = Object.entries(m)[0];
  return first ? `${first[0]}=${first[1]}` : "—";
}

export default async function Home() {
  const runs = await getRuns();
  return (
    <div className="mx-auto max-w-6xl">
      <div className="mb-6 flex items-baseline justify-between">
        <h1 className="text-2xl font-semibold">Experiment runs</h1>
        <span className="text-sm text-neutral-500">
          {runs.length} run{runs.length === 1 ? "" : "s"}
        </span>
      </div>

      {runs.length === 0 ? (
        <div className="rounded-lg border border-neutral-800 p-8 text-center text-neutral-400">
          No runs yet. Try{" "}
          <code className="text-neutral-200">
            uv run zo-train sft -c packages/training/configs/sft_smoke.yaml --dry-run
          </code>
        </div>
      ) : (
        <div className="overflow-hidden rounded-lg border border-neutral-800">
          <table className="w-full text-sm">
            <thead className="bg-neutral-900 text-left text-neutral-400">
              <tr>
                <th className="px-4 py-2 font-medium">Run</th>
                <th className="px-4 py-2 font-medium">Kind</th>
                <th className="px-4 py-2 font-medium">Status</th>
                <th className="px-4 py-2 font-medium">Branch</th>
                <th className="px-4 py-2 font-medium">Metric</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r: RunMeta) => (
                <tr key={r.id} className="border-t border-neutral-800 hover:bg-neutral-900/50">
                  <td className="px-4 py-2">
                    <Link href={`/runs/${r.id}`} className="font-medium text-neutral-100 hover:underline">
                      {r.name}
                    </Link>
                    <div className="text-xs text-neutral-500">{r.id}</div>
                  </td>
                  <td className="px-4 py-2 text-neutral-300">{r.kind}</td>
                  <td className={`px-4 py-2 ${STATUS_COLORS[r.status] ?? "text-neutral-400"}`}>{r.status}</td>
                  <td className="px-4 py-2 text-neutral-400">{r.git_branch ?? "—"}</td>
                  <td className="px-4 py-2 font-mono text-xs text-neutral-300">{topMetric(r.metrics)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
