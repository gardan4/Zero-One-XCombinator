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
  compare_row?: CompareReportRow;
};

export type Metric = Record<string, number | string>;

export type CompareRun = {
  id: string;
  name: string;
  kind: string;
  status: string;
  tags: string[];
  metrics: Record<string, unknown>;
};

export type ModelIdentity = {
  display: string;
  model_ref: string | null;
  predictor: string | null;
  version: string | null;
  train_run_id: string | null;
  role: string | null;
  served_model?: string | null;
};

export type DatasetContext = {
  eval_set: string | null;
  split: string | null;
  family: string | null;
  suite: string | null;
  suite_run: string | null;
};

export type ArtifactLinks = {
  results_dir: string;
  metrics_report_json?: string | null;
  metrics_report_md?: string | null;
  manifest_json?: string | null;
  examples_jsonl?: string | null;
  proxy_report_json?: string | null;
  official_scores_txt?: string | null;
  nextstep_csv?: string | null;
  completion_csv?: string | null;
  anomaly_csv?: string | null;
};

export type MetricSpec = {
  key: string;
  label: string;
  task: string;
  direction: string;
  format: string;
};

export type CompareReportRow = {
  run_id: string;
  run_name: string;
  kind: string;
  status: string;
  created_at: string;
  notes: string;
  tags: string[];
  parsed_tags: Record<string, string | null>;
  model: ModelIdentity;
  dataset: DatasetContext;
  metrics_flat: Record<string, number>;
  metrics_structured: Record<string, unknown> | null;
  artifacts: ArtifactLinks;
  git_branch?: string | null;
  git_sha?: string | null;
};

export type CompareReport = {
  source?: string;
  metric_specs: MetricSpec[];
  rows: CompareReportRow[];
  deltas_vs_baseline: Record<string, Record<string, number | null>>;
  count: number;
};

export type ExampleComparison = {
  example_id: string;
  family?: string;
  input?: Record<string, unknown>;
  gold?: unknown;
  a: { prediction: unknown; correct?: boolean | null; trace?: Record<string, unknown> };
  b: { prediction: unknown; correct?: boolean | null; trace?: Record<string, unknown> };
  score_gap?: number;
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

export async function getCompare(tags: string[]): Promise<CompareRun[]> {
  try {
    const qs = tags.filter(Boolean).map((t) => `tag=${encodeURIComponent(t)}`).join("&");
    const r = await fetch(`${BASE}/api/compare${qs ? `?${qs}` : ""}`, { cache: "no-store" });
    return r.ok ? r.json() : [];
  } catch {
    return [];
  }
}

export async function getCompareReport(params: {
  tags?: string[];
  kind?: string;
  role?: string;
  split?: string;
  family?: string;
  suite?: string;
  eval_set?: string;
  model_ref?: string;
  source?: string;
  refresh?: boolean;
  include_tests?: boolean;
  include_proxy?: boolean;
}): Promise<CompareReport | null> {
  try {
    const q = new URLSearchParams();
    (params.tags || []).forEach((t) => q.append("tag", t));
    if (params.kind) q.set("kind", params.kind);
    if (params.role) q.set("role", params.role);
    if (params.split) q.set("split", params.split);
    if (params.family) q.set("family", params.family);
    if (params.suite) q.set("suite", params.suite);
    if (params.eval_set) q.set("eval_set", params.eval_set);
    if (params.model_ref) q.set("model_ref", params.model_ref);
    if (params.source) q.set("source", params.source);
    if (params.refresh) q.set("refresh", "true");
    if (params.include_tests) q.set("include_tests", "true");
    if (params.include_proxy) q.set("include_proxy", "true");
    const r = await fetch(`${BASE}/api/compare/report?${q}`, { cache: "no-store" });
    return r.ok ? r.json() : null;
  } catch {
    return null;
  }
}

export async function getCompareExamples(
  runA: string,
  runB: string,
  task: string,
  mode: string,
  limit = 30,
): Promise<{ examples: ExampleComparison[]; run_a: CompareReportRow; run_b: CompareReportRow } | null> {
  try {
    const q = new URLSearchParams({ run_a: runA, run_b: runB, task, mode, limit: String(limit) });
    const r = await fetch(`${BASE}/api/compare/examples?${q}`, { cache: "no-store" });
    return r.ok ? r.json() : null;
  } catch {
    return null;
  }
}

export async function getRunExamples(
  runId: string,
  task: string,
  outcome?: string,
  limit = 30,
): Promise<{ examples: Record<string, unknown>[] } | null> {
  try {
    const q = new URLSearchParams({ task, limit: String(limit) });
    if (outcome) q.set("outcome", outcome);
    const r = await fetch(`${BASE}/api/runs/${runId}/examples?${q}`, { cache: "no-store" });
    return r.ok ? r.json() : null;
  } catch {
    return null;
  }
}

export async function getConfusion(id: string): Promise<Confusion> {
  try {
    const r = await fetch(`${BASE}/api/runs/${id}/confusion`, { cache: "no-store" });
    return r.ok ? r.json() : { tp: 0, fp: 0, tn: 0, fn: 0 };
  } catch {
    return { tp: 0, fp: 0, tn: 0, fn: 0 };
  }
}

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

export type InferenceJobResponse = {
  run_id: string;
  status: string;
  input_summary: Record<string, unknown>;
  links: { run: string; examples: string; compare_report: string };
};

export type InferencePreviewResponse = {
  ok: boolean;
  predictor: string;
  tasks: string[];
  input_summary: Record<string, unknown>;
  valid_csv: string | null;
  anomaly_csv: string | null;
};

type ApiError = { detail: string };

async function parseInferenceResponse<T>(r: Response): Promise<T | ApiError | null> {
  if (!r.ok) {
    try {
      const body = await r.json();
      const detail = typeof body?.detail === "string" ? body.detail : JSON.stringify(body?.detail ?? body);
      return { detail };
    } catch {
      return { detail: r.statusText || `HTTP ${r.status}` };
    }
  }
  return r.json();
}

export async function startInferenceJob(
  formData: FormData,
): Promise<InferenceJobResponse | ApiError | null> {
  try {
    const r = await fetch(`${BASE}/api/inference/jobs`, {
      method: "POST",
      body: formData,
      cache: "no-store",
    });
    return parseInferenceResponse<InferenceJobResponse>(r);
  } catch {
    return null;
  }
}

export async function previewInferenceJob(
  formData: FormData,
): Promise<InferencePreviewResponse | ApiError | null> {
  try {
    const r = await fetch(`${BASE}/api/inference/preview`, {
      method: "POST",
      body: formData,
      cache: "no-store",
    });
    return parseInferenceResponse<InferencePreviewResponse>(r);
  } catch {
    return null;
  }
}

export async function getInferenceJob(
  runId: string,
): Promise<{ run_id: string; status: string; links: InferenceJobResponse["links"] } | null> {
  try {
    const r = await fetch(`${BASE}/api/inference/jobs/${runId}`, { cache: "no-store" });
    return r.ok ? r.json() : null;
  } catch {
    return null;
  }
}
