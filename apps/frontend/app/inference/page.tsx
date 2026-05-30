"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";

import {
  type InferenceJobResponse,
  getRun,
  getRunExamples,
  previewInferenceJob,
  startInferenceJob,
} from "@/lib/api";

const PREDICTORS = [
  "ngram",
  "freq",
  "oracle",
  "base",
  "base-hf",
  "hf",
  "llm",
  "likelihood-ngram",
  "classifier",
] as const;

const FAMILIES = ["MOSFET", "IGBT", "IC"] as const;
type InputMode = "upload" | "manual";

export default function InferencePage() {
  const [inputMode, setInputMode] = useState<InputMode>("manual");
  const [predictor, setPredictor] = useState("ngram");
  const [model, setModel] = useState("default");
  const [version, setVersion] = useState("dashboard-v1");
  const [tags, setTags] = useState("source:dashboard");
  const [tasks, setTasks] = useState("nextstep,completion,anomaly");
  const [name, setName] = useState("");
  const [notes, setNotes] = useState("");

  const [manualTask, setManualTask] = useState("nextstep");
  const [family, setFamily] = useState<(typeof FAMILIES)[number]>("MOSFET");
  const [completionFraction, setCompletionFraction] = useState("0.6");
  const [sequenceText, setSequenceText] = useState("SPIN COAT PHOTORESIST\nEXPOSE LITHO LEVEL 1");

  const [validFile, setValidFile] = useState<File | null>(null);
  const [anomalyFile, setAnomalyFile] = useState<File | null>(null);

  const [submitting, setSubmitting] = useState(false);
  const [previewing, setPreviewing] = useState(false);
  const [preview, setPreview] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [job, setJob] = useState<InferenceJobResponse | null>(null);
  const [runStatus, setRunStatus] = useState<string | null>(null);
  const [examplePreview, setExamplePreview] = useState<Record<string, unknown>[]>([]);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const buildFormData = useCallback((): FormData => {
    const fd = new FormData();
    fd.set("predictor", predictor);
    fd.set("model", model);
    fd.set("version", version);
    fd.set("tasks", tasks);
    fd.set("tags", tags);
    if (name.trim()) fd.set("name", name.trim());
    if (notes.trim()) fd.set("notes", notes.trim());

    if (inputMode === "upload") {
      if (validFile) fd.set("valid_csv", validFile);
      if (anomalyFile) fd.set("anomaly_csv", anomalyFile);
    } else {
      const steps = sequenceText
        .split("\n")
        .map((s) => s.trim())
        .filter(Boolean);
      const manual: Record<string, unknown> = {
        task: manualTask,
        family,
        completion_fraction: parseFloat(completionFraction) || 0.6,
      };
      if (manualTask === "anomaly") {
        manual.sequence = steps;
      } else {
        manual.partial_sequence = steps;
      }
      fd.set("manual_json", JSON.stringify(manual));
    }
    return fd;
  }, [
    predictor,
    model,
    version,
    tasks,
    tags,
    name,
    notes,
    inputMode,
    validFile,
    anomalyFile,
    manualTask,
    family,
    completionFraction,
    sequenceText,
  ]);

  const onPreview = async () => {
    setPreviewing(true);
    setError(null);
    setPreview(null);
    const res = await previewInferenceJob(buildFormData());
    setPreviewing(false);
    if (!res) {
      setError("Preview failed — check backend and inputs");
      return;
    }
    if ("detail" in res) {
      setError(String(res.detail));
      return;
    }
    setPreview(res as Record<string, unknown>);
  };

  const onSubmit = async () => {
    setSubmitting(true);
    setError(null);
    setJob(null);
    setRunStatus(null);
    setExamplePreview([]);
    const res = await startInferenceJob(buildFormData());
    setSubmitting(false);
    if (!res) {
      setError("Failed to start job — is the backend running?");
      return;
    }
    if ("detail" in res) {
      setError(String(res.detail));
      return;
    }
    setJob(res);
    setRunStatus(res.status);
  };

  useEffect(() => {
    if (!job?.run_id) return;
    const poll = async () => {
      const meta = await getRun(job.run_id);
      if (!meta) return;
      setRunStatus(meta.status);
      if (meta.status === "completed" || meta.status === "failed") {
        if (pollRef.current) clearInterval(pollRef.current);
        if (meta.status === "completed") {
          const task =
            manualTask === "anomaly" ? "anomaly" : manualTask === "completion" ? "completion" : "nextstep";
          const ex = await getRunExamples(job.run_id, task, undefined, 5);
          setExamplePreview(ex?.examples ?? []);
        }
      }
    };
    poll();
    pollRef.current = setInterval(poll, 2500);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [job?.run_id, manualTask]);

  return (
    <main className="mx-auto max-w-3xl space-y-8 px-4 py-10">
      <header>
        <h1 className="text-2xl font-semibold text-neutral-100">Inference</h1>
        <p className="mt-1 text-sm text-neutral-400">
          Run track eval from the dashboard — upload organizer CSVs or a single manual example.
          Results land in the run registry like <code className="text-neutral-300">zo-track predict</code>.
        </p>
      </header>

      <section className="space-y-4 rounded-lg border border-neutral-800 bg-neutral-950/50 p-5">
        <h2 className="text-sm font-medium text-neutral-200">Model</h2>
        <div className="grid gap-3 sm:grid-cols-2">
          <label className="block text-xs text-neutral-400">
            Predictor
            <select
              value={predictor}
              onChange={(e) => setPredictor(e.target.value)}
              className="mt-1 w-full rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1.5 text-sm text-neutral-100"
            >
              {PREDICTORS.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </label>
          <label className="block text-xs text-neutral-400">
            Model (HF id or served name)
            <input
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="mt-1 w-full rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1.5 text-sm text-neutral-100"
            />
          </label>
          <label className="block text-xs text-neutral-400">
            Version tag
            <input
              value={version}
              onChange={(e) => setVersion(e.target.value)}
              className="mt-1 w-full rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1.5 text-sm text-neutral-100"
            />
          </label>
          <label className="block text-xs text-neutral-400">
            Tasks (comma-separated)
            <input
              value={tasks}
              onChange={(e) => setTasks(e.target.value)}
              className="mt-1 w-full rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1.5 text-sm text-neutral-100"
            />
          </label>
          <label className="col-span-2 block text-xs text-neutral-400">
            Tags
            <input
              value={tags}
              onChange={(e) => setTags(e.target.value)}
              className="mt-1 w-full rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1.5 text-sm text-neutral-100"
            />
          </label>
        </div>
        <p className="text-xs text-neutral-500">
          Baselines (ngram, freq, oracle) run by default. <code className="text-neutral-400">base</code> (served)
          / <code className="text-neutral-400">base-hf</code> (local) are an un-fine-tuned LLM given full
          context; these and HF / LLM / classifier need{" "}
          <code className="text-neutral-400">ZO_ALLOW_DASHBOARD_INFERENCE=1</code> on the backend. Leave Model
          as <code className="text-neutral-400">default</code> to use{" "}
          <code className="text-neutral-400">ZO_BASE_LLM_MODEL</code> (Qwen2.5-7B-Instruct).
        </p>
      </section>

      <section className="space-y-4 rounded-lg border border-neutral-800 bg-neutral-950/50 p-5">
        <div className="flex gap-2">
          {(["manual", "upload"] as const).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setInputMode(m)}
              className={`rounded-md px-3 py-1.5 text-sm ${
                inputMode === m
                  ? "bg-neutral-700 text-neutral-100"
                  : "text-neutral-400 hover:bg-neutral-900"
              }`}
            >
              {m === "manual" ? "Manual example" : "Upload CSV"}
            </button>
          ))}
        </div>

        {inputMode === "manual" ? (
          <div className="grid gap-3">
            <label className="block text-xs text-neutral-400">
              Task
              <select
                value={manualTask}
                onChange={(e) => setManualTask(e.target.value)}
                className="mt-1 w-full rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1.5 text-sm"
              >
                <option value="nextstep">nextstep</option>
                <option value="completion">completion</option>
                <option value="anomaly">anomaly</option>
              </select>
            </label>
            <label className="block text-xs text-neutral-400">
              Family
              <select
                value={family}
                onChange={(e) => setFamily(e.target.value as (typeof FAMILIES)[number])}
                className="mt-1 w-full rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1.5 text-sm"
              >
                {FAMILIES.map((f) => (
                  <option key={f} value={f}>
                    {f}
                  </option>
                ))}
              </select>
            </label>
            {manualTask !== "anomaly" && (
              <label className="block text-xs text-neutral-400">
                Completion fraction
                <input
                  value={completionFraction}
                  onChange={(e) => setCompletionFraction(e.target.value)}
                  className="mt-1 w-full rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1.5 text-sm"
                />
              </label>
            )}
            <label className="block text-xs text-neutral-400">
              {manualTask === "anomaly" ? "Full sequence (one step per line)" : "Partial sequence"}
              <textarea
                value={sequenceText}
                onChange={(e) => setSequenceText(e.target.value)}
                rows={6}
                className="mt-1 w-full rounded-md border border-neutral-700 bg-neutral-900 px-2 py-2 font-mono text-sm text-neutral-100"
              />
            </label>
          </div>
        ) : (
          <div className="grid gap-3">
            <label className="block text-xs text-neutral-400">
              eval_input_valid.csv (Tasks 1 &amp; 2)
              <input
                type="file"
                accept=".csv"
                onChange={(e) => setValidFile(e.target.files?.[0] ?? null)}
                className="mt-1 block w-full text-sm text-neutral-300"
              />
            </label>
            <label className="block text-xs text-neutral-400">
              eval_input_anomaly.csv (Task 3)
              <input
                type="file"
                accept=".csv"
                onChange={(e) => setAnomalyFile(e.target.files?.[0] ?? null)}
                className="mt-1 block w-full text-sm text-neutral-300"
              />
            </label>
          </div>
        )}
      </section>

      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          onClick={onPreview}
          disabled={previewing || submitting}
          className="rounded-md border border-neutral-600 px-4 py-2 text-sm text-neutral-200 hover:bg-neutral-800 disabled:opacity-50"
        >
          {previewing ? "Validating…" : "Validate inputs"}
        </button>
        <button
          type="button"
          onClick={onSubmit}
          disabled={submitting || previewing}
          className="rounded-md bg-sky-700 px-4 py-2 text-sm font-medium text-white hover:bg-sky-600 disabled:opacity-50"
        >
          {submitting ? "Starting…" : "Run inference"}
        </button>
      </div>

      {error && (
        <p className="rounded-md border border-rose-900/50 bg-rose-950/30 px-3 py-2 text-sm text-rose-300">
          {error}
        </p>
      )}

      {preview && (
        <pre className="overflow-x-auto rounded-md border border-neutral-800 bg-neutral-900/50 p-3 text-xs text-neutral-300">
          {JSON.stringify(preview, null, 2)}
        </pre>
      )}

      {job && (
        <section className="space-y-3 rounded-lg border border-emerald-900/40 bg-emerald-950/20 p-5">
          <h2 className="text-sm font-medium text-emerald-200">Job started</h2>
          <p className="font-mono text-sm text-neutral-200">
            {job.run_id}{" "}
            <span className="text-neutral-400">({runStatus ?? job.status})</span>
          </p>
          <ul className="flex flex-wrap gap-3 text-sm">
            <Link href={`/runs/${job.run_id}`} className="text-sky-400 hover:underline">
              Run detail
            </Link>
            <Link href="/compare" className="text-sky-400 hover:underline">
              Compare
            </Link>
          </ul>
          {examplePreview.length > 0 && (
            <pre className="max-h-48 overflow-auto rounded border border-neutral-800 bg-neutral-900/60 p-2 text-xs">
              {JSON.stringify(examplePreview, null, 2)}
            </pre>
          )}
        </section>
      )}
    </main>
  );
}
