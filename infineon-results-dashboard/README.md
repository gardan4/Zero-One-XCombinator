# Infineon Fab Route Copilot — Results Dashboard

Submission for the **Industrial-AI (Infineon) track** of the Zero One Hack.

The track theme is *real model training, not prompt engineering*: train a sequence
model on semiconductor fab process routes and show it has **learned the process
grammar** — not memorized a lookup table.

This folder is a self-contained [Cloudflare Workers](https://workers.cloudflare.com/)
app with two halves:

| Half | Path | What it does |
| --- | --- | --- |
| **Results dashboard** (frontend) | `public/` | Presents the trained model's eval results — baseline vs fine-tuned — for next-step prediction, route completion, anomaly detection, the training loss curve, and out-of-distribution (OOD) generalization. |
| **Live-demo backend** (Worker) | `src/worker.js` | A retrieval-over-training-routes baseline with optional LLM reranking. Powers the interactive "try it" strip and serves as the *before* (baseline) the trained model is measured against. |

## How the two halves relate

- The **dashboard numbers are the headline result.** They live in a single file,
  `public/results.js`, and the whole UI renders from that one object.
- ⚠️ **The numbers in `results.js` are currently ILLUSTRATIVE placeholders.** They are
  stand-ins for the trained model. To go live, replace that object with real output
  from the eval harness (per-task, per-family) plus the training log. See
  [`public/README.md`](public/README.md) for the exact shape.
- The **backend is the baseline**, not the trained model. With no model secret set it
  answers purely by retrieval over the shipped training routes — enough to make the UI
  interactive while a hosted/trained model is wired in.

## Product families

The app is product-line aware: `IC`, `IGBT`, and `MOSFET` are ranked against separate
route pools. A hidden fourth family (`BJT`) is the held-out OOD test.

## Local run

```bash
cd infineon-results-dashboard
npm install
npm run build:data   # generates src/data/fab-data.js from the training routes
npm run dev          # wrangler dev — open the URL it prints
```

Without any model secret the backend uses the retrieval baseline. That is enough for
the demo and keeps the UI usable while a hosted model is being selected.

## API

| Method & path | Purpose |
| --- | --- |
| `GET /api/health` | Product-line stats and model status |
| `GET /api/sample?family=MOSFET&fraction=0.6` | Sample a partial route |
| `POST /api/predict` | Top-5 next-step prediction |
| `POST /api/complete` | Complete a partial route |
| `POST /api/anomaly` | Validate a full route against the process rules |

## Optional model reranking

Copy `.dev.vars.example` → `.dev.vars` and set an OpenAI-compatible provider. The model
**only reranks backend-generated candidates** — it cannot invent step names outside the
allowed candidate list, so it can't hallucinate process steps.

```bash
MODEL_BASE_URL=https://api.featherless.ai/v1
MODEL_NAME=Qwen/Qwen3.6-27B
MODEL_API_KEY=...
```

`.dev.vars` is gitignored — never commit real keys.

## Deploy

```bash
npm run deploy

# for deployed model reranking:
printf '%s' "$MODEL_API_KEY" | wrangler secret put MODEL_API_KEY
wrangler deploy --var MODEL_NAME:Qwen/Qwen3.6-27B --var MODEL_BASE_URL:https://api.featherless.ai/v1
```

## Going live (placeholder → real)

1. Run the eval harness to produce real per-task / per-family metrics and the training log.
2. Replace the `RESULTS` object in `public/results.js` with those numbers (same shape).
3. Reload — the dashboard re-renders entirely from that object. No other code changes needed.
