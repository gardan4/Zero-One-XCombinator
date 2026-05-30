/**
 * FastAPI control-plane client (localhost:8000 by default).
 * Separate from the Worker-local api() helper in app.js.
 */

export const CONTROL_PLANE_BASE =
  (typeof window !== "undefined" && window.CONTROL_PLANE_URL) || "http://localhost:8000/api";

let _online = null;

export function controlPlaneOnline() {
  return _online === true;
}

export async function probeControlPlane() {
  try {
    const r = await fetch(`${CONTROL_PLANE_BASE.replace(/\/api$/, "")}/health`, {
      method: "GET",
      cache: "no-store",
    });
    _online = r.ok;
  } catch {
    _online = false;
  }
  return _online;
}

async function cpFetch(path, init = {}) {
  const url = path.startsWith("http") ? path : `${CONTROL_PLANE_BASE}${path.startsWith("/") ? path : `/${path}`}`;
  const response = await fetch(url, { cache: "no-store", ...init });
  let data = null;
  try {
    data = await response.json();
  } catch {
    data = null;
  }
  if (!response.ok) {
    const detail = data?.detail;
    const msg =
      typeof detail === "string" ? detail : detail ? JSON.stringify(detail) : response.statusText || "Request failed";
    throw new Error(msg);
  }
  return data;
}

export function getRuns(params = {}) {
  const q = new URLSearchParams();
  if (params.source) q.set("source", params.source);
  if (params.includeTests) q.set("include_tests", "true");
  if (params.includeProxy) q.set("include_proxy", "true");
  const qs = q.toString();
  return cpFetch(`/runs${qs ? `?${qs}` : ""}`);
}

export function getCompareReport(params = {}) {
  const q = new URLSearchParams();
  (params.tags || []).forEach((t) => q.append("tag", t));
  if (params.source) q.set("source", params.source);
  if (params.split) q.set("split", params.split);
  if (params.role) q.set("role", params.role);
  if (params.family) q.set("family", params.family);
  if (params.kind) q.set("kind", params.kind);
  if (params.refresh) q.set("refresh", "true");
  if (params.includeTests) q.set("include_tests", "true");
  if (params.includeProxy) q.set("include_proxy", "true");
  const qs = q.toString();
  return cpFetch(`/compare/report${qs ? `?${qs}` : ""}`);
}

export function refreshCompareCache(source = "wandb") {
  return cpFetch(`/compare/refresh?source=${encodeURIComponent(source)}`, { method: "POST" });
}

export function getRun(runId) {
  return cpFetch(`/runs/${encodeURIComponent(runId)}`);
}

export function getCompareExamples(runA, runB, task, mode, limit = 25) {
  const q = new URLSearchParams({
    run_a: runA,
    run_b: runB,
    task,
    mode,
    limit: String(limit),
  });
  return cpFetch(`/compare/examples?${q}`);
}

export function getRunConfusion(runId) {
  return cpFetch(`/runs/${encodeURIComponent(runId)}/confusion`);
}

export function getRunMetrics(runId) {
  return cpFetch(`/runs/${encodeURIComponent(runId)}/metrics`);
}

export async function previewInference(formData) {
  return cpFetch("/inference/preview", { method: "POST", body: formData });
}

export async function startInferenceJob(formData) {
  return cpFetch("/inference/jobs", { method: "POST", body: formData });
}

export function getInferenceJob(runId) {
  return cpFetch(`/inference/jobs/${encodeURIComponent(runId)}`);
}

export function getRunExamples(runId, task, limit = 5) {
  const q = new URLSearchParams({ task, limit: String(limit) });
  return cpFetch(`/runs/${encodeURIComponent(runId)}/examples?${q}`);
}
