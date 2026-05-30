# XCombinator — Fab Process Copilot

A dark, liquid-glass operator dashboard for the **Industrial AI (Infineon)** track. Import a
partial semiconductor fab route, watch it as a connected-node process flow, and have the model
**predict the next process step** — with live process-logic (anomaly) checking.

Standalone Vite + React + TypeScript app. No backend required: predictions are **simulated**
locally until the team's fine-tuned checkpoint is hosted, then swap in with two env vars.

## Run

```bash
npm install
npm run dev        # http://localhost:5173
# or a production build:
npm run build && npm run preview   # http://localhost:4173
```

## Deploy

Live at **https://infineon-fab-copilot.alexreinicke2004.workers.dev/copilot/** — served as a path on
the existing `infineon-fab-copilot` Cloudflare Worker via Static Assets. The worker's own results
dashboard (`/`) and API (`/api/*`) are untouched; the build just lands in
`../infineon-results-dashboard/public/copilot/`.

Redeploy after changes (Cloudflare creds in env, e.g. `set -a; . /root/RD/.env; set +a`):

```bash
npm run deploy   # vite build (base /copilot/) → copy into the worker's public/copilot → wrangler deploy
```

## Swap the simulator for the real model

`src/lib/model.ts` is the single seam. The simulator is a back-off n-gram over a few bundled
reference routes (labelled "Simulated"). To use the served checkpoint (OpenAI-compatible, e.g.
vLLM from `just serve`), set:

```bash
# .env.local
VITE_MODEL_BASE_URL=http://localhost:8001/v1
VITE_MODEL_NAME=XCombinator/sft-fab-all
# VITE_MODEL_API_KEY=...        # optional
```

Nothing else changes — same UI, same shapes. The status pill flips to **"Live · model"**. On any
network error it falls back to the simulator so the demo never stalls.

## How it's built

| Path | Role |
|---|---|
| `src/lib/model.ts` | `predictNextStep()` — the simulated/served model seam |
| `src/lib/rules.ts` | `validateRoute()` — the 10 process-logic rules, ported verbatim from the track's `validate_sequence()` (anomaly truth) |
| `src/lib/grammar.ts` | phase segmentation (collapse long routes), per-family roadmap, step categories |
| `src/data/fab.json` | bundled reference routes (n-gram corpus), import samples, step descriptions, 198-token vocab |
| `src/components/ProcessRoute.tsx` | the centerpiece: collapsed phases + expanded current phase + predicted ghost |

## What's deliberately not here (yet)

Metrics/benchmark, top-5 ranking, training curves, OOD panels, and the **Auto-run** loop (designed
in, shown as "Soon"). The data layer and seam are shaped so these slot in without rework.
