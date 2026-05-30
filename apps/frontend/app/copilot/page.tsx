"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  type Examples,
  type RuleInfo,
  type ValidateResult,
  getExamples,
  getRules,
  validateRecipe,
} from "@/lib/api";

const FAMILIES = ["MOSFET", "IGBT", "IC"] as const;
type Family = (typeof FAMILIES)[number];

export default function CopilotPage() {
  const [family, setFamily] = useState<Family>("MOSFET");
  const [text, setText] = useState<string>("");
  const [result, setResult] = useState<ValidateResult | null>(null);
  const [checking, setChecking] = useState(false);
  const [examples, setExamples] = useState<Examples | null>(null);
  const [rules, setRules] = useState<RuleInfo[]>([]);
  const [validCursor, setValidCursor] = useState(0);
  const [invalidCursor, setInvalidCursor] = useState(0);
  const debounce = useRef<ReturnType<typeof setTimeout> | null>(null);

  // load the rule catalogue once
  useEffect(() => {
    getRules().then(setRules);
  }, []);

  // load examples per family; seed the editor with a valid recipe the first time
  useEffect(() => {
    getExamples(family).then((ex) => {
      setExamples(ex);
      if (ex && ex.valid.length && !text.trim()) setText(ex.valid[0].join("\n"));
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [family]);

  // live validation (debounced) whenever the recipe text changes
  useEffect(() => {
    if (debounce.current) clearTimeout(debounce.current);
    if (!text.trim()) {
      setResult(null);
      return;
    }
    setChecking(true);
    debounce.current = setTimeout(async () => {
      const r = await validateRecipe(text);
      setResult(r);
      setChecking(false);
    }, 300);
    return () => {
      if (debounce.current) clearTimeout(debounce.current);
    };
  }, [text]);

  const loadValid = useCallback(() => {
    if (!examples?.valid.length) return;
    const i = validCursor % examples.valid.length;
    setText(examples.valid[i].join("\n"));
    setValidCursor(i + 1);
  }, [examples, validCursor]);

  const injectViolation = useCallback(() => {
    if (!examples?.invalid.length) return;
    const i = invalidCursor % examples.invalid.length;
    setText(examples.invalid[i].steps.join("\n"));
    setInvalidCursor(i + 1);
  }, [examples, invalidCursor]);

  const steps = useMemo(
    () => text.split("\n").map((s) => s.trim()).filter(Boolean),
    [text],
  );
  const violIndices = useMemo(
    () => new Set((result?.violations ?? []).map((v) => v.step_index)),
    [result],
  );
  // the repair hint comes from the matching seed example (symbolic repair the agent would apply)
  const repairHint = useMemo(() => {
    if (!result || result.valid || !examples) return null;
    const rule = result.violations[0]?.rule;
    return examples.invalid.find((e) => e.target_rule === rule)?.repair ?? null;
  }, [result, examples]);

  return (
    <div className="mx-auto max-w-6xl pb-16">
      <h1 className="text-2xl font-semibold">Process-Logic Copilot</h1>
      <p className="mt-1 max-w-3xl text-sm text-neutral-400">
        Paste or edit a semiconductor fab route. The verifier checks all{" "}
        <span className="text-neutral-200">{rules.length || 10}</span> process-logic rules{" "}
        <span className="text-neutral-200">instantly</span> — no GPU — flags the first violating step,
        explains it in plain English, and suggests a repair. This is the ground-truth oracle our
        trained model is measured against.
      </p>

      {/* family tabs + example loaders */}
      <div className="mt-5 flex flex-wrap items-center gap-2">
        <div className="flex rounded-lg border border-neutral-800 p-0.5">
          {FAMILIES.map((f) => (
            <button
              key={f}
              onClick={() => setFamily(f)}
              className={`rounded-md px-3 py-1 text-sm transition ${
                family === f ? "bg-neutral-800 text-neutral-100" : "text-neutral-400 hover:text-neutral-200"
              }`}
            >
              {f}
            </button>
          ))}
        </div>
        <button
          onClick={loadValid}
          className="rounded-md border border-emerald-800/60 bg-emerald-950/40 px-3 py-1.5 text-sm text-emerald-200 transition hover:bg-emerald-900/40"
        >
          ↻ Load valid recipe
        </button>
        <button
          onClick={injectViolation}
          className="rounded-md border border-rose-800/60 bg-rose-950/40 px-3 py-1.5 text-sm text-rose-200 transition hover:bg-rose-900/40"
        >
          ⚡ Inject a violation
        </button>
        <button
          onClick={() => setText("")}
          className="rounded-md border border-neutral-800 px-3 py-1.5 text-sm text-neutral-400 transition hover:text-neutral-200"
        >
          Clear
        </button>
      </div>

      <div className="mt-5 grid grid-cols-1 gap-5 lg:grid-cols-2">
        {/* editor */}
        <section className="flex flex-col">
          <div className="mb-2 flex items-baseline justify-between">
            <h2 className="text-sm font-medium text-neutral-300">Recipe — one step per line</h2>
            <span className="text-xs text-neutral-500">{steps.length} steps</span>
          </div>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            spellCheck={false}
            placeholder="RECEIVE WAFER LOT&#10;LOT IDENTIFICATION&#10;..."
            className="h-[460px] w-full resize-none rounded-lg border border-neutral-800 bg-neutral-950 p-3 font-mono text-xs leading-relaxed text-neutral-200 outline-none focus:border-neutral-600"
          />
        </section>

        {/* verdict */}
        <section className="flex flex-col gap-4">
          <Verdict result={result} checking={checking} repairHint={repairHint} />
          <RouteViewer steps={result?.steps ?? steps} violIndices={violIndices} />
        </section>
      </div>

      {/* rules reference */}
      <section className="mt-8">
        <h2 className="mb-3 text-sm font-medium text-neutral-300">
          The {rules.length || 10} process-logic rules
        </h2>
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
          {rules.map((r) => {
            const hit = result?.violations.some((v) => v.rule === r.rule);
            return (
              <div
                key={r.rule}
                className={`rounded-lg border p-3 text-xs transition ${
                  hit ? "border-rose-700/70 bg-rose-950/30" : "border-neutral-800 bg-neutral-950"
                }`}
              >
                <div className={`font-mono ${hit ? "text-rose-300" : "text-neutral-300"}`}>{r.rule}</div>
                <div className="mt-1 text-neutral-500">{r.description}</div>
              </div>
            );
          })}
        </div>
      </section>
    </div>
  );
}

function Verdict({
  result,
  checking,
  repairHint,
}: {
  result: ValidateResult | null;
  checking: boolean;
  repairHint: string | null;
}) {
  if (!result) {
    return (
      <div className="rounded-lg border border-neutral-800 bg-neutral-950 p-6 text-sm text-neutral-500">
        {checking ? "Checking…" : "Enter a recipe to validate."}
      </div>
    );
  }
  if (result.valid) {
    return (
      <div className="rounded-lg border border-emerald-800/60 bg-emerald-950/30 p-6">
        <div className="flex items-center gap-3">
          <span className="text-2xl">✓</span>
          <div>
            <div className="text-lg font-semibold text-emerald-200">Valid route</div>
            <div className="text-xs text-emerald-400/70">
              {result.n_steps} steps · all process-logic rules satisfied
            </div>
          </div>
        </div>
      </div>
    );
  }
  return (
    <div className="rounded-lg border border-rose-800/60 bg-rose-950/30 p-5">
      <div className="flex items-center gap-3">
        <span className="text-2xl">⚠</span>
        <div>
          <div className="text-lg font-semibold text-rose-200">
            {result.n_violations} violation{result.n_violations === 1 ? "" : "s"}
          </div>
          <div className="text-xs text-rose-400/70">{result.n_steps} steps · process logic broken</div>
        </div>
      </div>
      <ul className="mt-4 space-y-3">
        {result.violations.map((v, i) => (
          <li key={i} className="rounded-md border border-rose-900/50 bg-neutral-950/60 p-3">
            <div className="flex items-center gap-2">
              <span className="rounded bg-rose-900/50 px-1.5 py-0.5 font-mono text-[11px] text-rose-200">
                {v.rule}
              </span>
              <span className="text-xs text-neutral-500">
                step {v.step_index} · {v.step_name}
              </span>
            </div>
            <p className="mt-2 text-xs leading-relaxed text-neutral-300">{v.description}</p>
          </li>
        ))}
      </ul>
      {result.explanation && (
        <p className="mt-3 border-t border-rose-900/40 pt-3 text-xs italic text-neutral-400">
          {result.explanation}
        </p>
      )}
      {repairHint && (
        <div className="mt-3 rounded-md border border-sky-900/50 bg-sky-950/30 p-3">
          <div className="text-[11px] font-medium uppercase tracking-wide text-sky-400/80">
            Suggested repair
          </div>
          <p className="mt-1 text-xs text-sky-200">{repairHint}</p>
        </div>
      )}
    </div>
  );
}

function RouteViewer({ steps, violIndices }: { steps: string[]; violIndices: Set<number> }) {
  if (!steps.length) return null;
  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-950">
      <div className="border-b border-neutral-800 px-3 py-2 text-xs text-neutral-500">Route ({steps.length} steps)</div>
      <ol className="max-h-[260px] overflow-auto p-2 font-mono text-[11px]">
        {steps.map((s, i) => {
          const bad = violIndices.has(i);
          return (
            <li
              key={i}
              className={`flex gap-2 rounded px-2 py-0.5 ${
                bad ? "bg-rose-950/50 text-rose-200" : "text-neutral-400"
              }`}
            >
              <span className="w-8 shrink-0 text-right text-neutral-600">{i}</span>
              <span>{s}</span>
              {bad && <span className="ml-auto text-rose-400">← violates</span>}
            </li>
          );
        })}
      </ol>
    </div>
  );
}
