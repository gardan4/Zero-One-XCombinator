export const BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type RunMeta = {
  id: string;
  name: string;
  kind: string;
  status: string;
  created_at: string;
  updated_at: string;
  git_branch?: string | null;
  git_sha?: string | null;
  cluster?: string | null;
  slurm_job_id?: string | null;
  tags: string[];
  notes: string;
  config: Record<string, unknown>;
  metrics: Record<string, unknown>;
};

export type Metric = Record<string, number | string>;

// Trimmed projection returned by GET /api/compare (id/name/kind/status/tags/metrics only).
export type CompareRun = {
  id: string;
  name: string;
  kind: string;
  status: string;
  tags: string[];
  metrics: Record<string, unknown>;
};

export type Confusion = { tp: number; fp: number; tn: number; fn: number };

export async function getRuns(): Promise<RunMeta[]> {
  try {
    const r = await fetch(`${BASE}/api/runs`, { cache: "no-store" });
    return r.ok ? r.json() : [];
  } catch {
    return [];
  }
}

export async function getRun(id: string): Promise<RunMeta | null> {
  try {
    const r = await fetch(`${BASE}/api/runs/${id}`, { cache: "no-store" });
    return r.ok ? r.json() : null;
  } catch {
    return null;
  }
}

export async function getMetrics(id: string): Promise<Metric[]> {
  try {
    const r = await fetch(`${BASE}/api/runs/${id}/metrics`, { cache: "no-store" });
    return r.ok ? r.json() : [];
  } catch {
    return [];
  }
}

// Runs matching ALL given tags (repeatable ?tag= query param). [] → all runs.
export async function getCompare(tags: string[]): Promise<CompareRun[]> {
  try {
    const qs = tags.filter(Boolean).map((t) => `tag=${encodeURIComponent(t)}`).join("&");
    const r = await fetch(`${BASE}/api/compare${qs ? `?${qs}` : ""}`, { cache: "no-store" });
    return r.ok ? r.json() : [];
  } catch {
    return [];
  }
}

// The anomaly confusion matrix {tp,fp,tn,fn} for a run (zeros if it has no anomaly metrics).
export async function getConfusion(id: string): Promise<Confusion> {
  try {
    const r = await fetch(`${BASE}/api/runs/${id}/confusion`, { cache: "no-store" });
    return r.ok ? r.json() : { tp: 0, fp: 0, tn: 0, fn: 0 };
  } catch {
    return { tp: 0, fp: 0, tn: 0, fn: 0 };
  }
}

// --- Process-Logic Copilot (symbolic verifier, GPU-free) ------------------------------------
export type Violation = {
  rule: string;
  description: string;
  step_index: number;
  step_name: string;
};

export type ValidateResult = {
  valid: boolean;
  n_steps: number;
  n_violations: number;
  violations: Violation[];
  explanation: string | null;
  steps: string[];
};

export type Grammar = {
  categories: Record<string, string[]>;
  steps: string[];
  n_steps: number;
  n_categories: number;
};

export type RuleInfo = { rule: string; description: string };

export type InvalidExample = {
  steps: string[];
  target_rule: string;
  repair: string | null;
  explanation: string | null;
  violation_index: number | null;
};

export type Examples = { family: string; valid: string[][]; invalid: InvalidExample[] };

export async function validateRecipe(text: string): Promise<ValidateResult | null> {
  try {
    const r = await fetch(`${BASE}/api/validate`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ text }),
      cache: "no-store",
    });
    return r.ok ? r.json() : null;
  } catch {
    return null;
  }
}

export async function getGrammar(): Promise<Grammar | null> {
  try {
    const r = await fetch(`${BASE}/api/grammar`, { cache: "no-store" });
    return r.ok ? r.json() : null;
  } catch {
    return null;
  }
}

export async function getRules(): Promise<RuleInfo[]> {
  try {
    const r = await fetch(`${BASE}/api/rules`, { cache: "no-store" });
    return r.ok ? r.json() : [];
  } catch {
    return [];
  }
}

export async function getExamples(family: string): Promise<Examples | null> {
  try {
    const r = await fetch(`${BASE}/api/examples?family=${encodeURIComponent(family)}`, { cache: "no-store" });
    return r.ok ? r.json() : null;
  } catch {
    return null;
  }
}
