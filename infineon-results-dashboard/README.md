# Infineon Fab Route Copilot — Results Dashboard

Submission for the **Industrial-AI (Infineon) track** of the Zero One Hack.

The track theme is *real model training, not prompt engineering*: train a sequence
model on semiconductor fab process routes and show it has **learned the process
grammar** — not memorized a lookup table.

This folder is a self-contained [Cloudflare Workers](https://workers.cloudflare.com/)
app with three layers:

| Layer | Path / port | What it does |
| --- | --- | --- |
| **Results dashboard** (frontend) | `public/` · `:8787` | Polished submission UI — static headline numbers from `results.js`, plus live registry/comparison/inference when the control plane is up. |
| **Live-demo backend** (Worker) | `src/worker.js` · `:8787/api/*` | Retrieval-over-training-routes baseline with optional LLM reranking. Powers the interactive “try it” strip. |
| **Control plane** (FastAPI) | repo root · `:8000` | Run registry, normalized compare report, per-example disagreements, and async inference jobs (`zo-track`). Optional but enables the live sections. |

## How the layers relate

- The **dashboard numbers are the headline result.** They live in `public/results.js`
  and render even when nothing else is running.
- ⚠️ **Finetuned / anomaly / training / OOD blocks in `results.js` may still be
  illustrative** until replaced from eval output. Baseline next-step/completion numbers
  can be real via `scripts/eval-baseline.mjs`.
- The **Worker API** is the retrieval baseline for the live demo — it works without
  the FastAPI backend.
- The **control plane sections** (registry, compare matrix, disagreements, inference
  jobs) call `http://localhost:8000/api` via CORS. If the backend is offline, the
  page shows a banner and keeps static results + the Worker demo working.

## Local run (full experience)

**Terminal 1 — control plane** (from repo root):

```bash
uv run python scripts/dev.py
# or: uv run uvicorn zo_backend.main:app --port 8000
```

**Terminal 2 — polished dashboard**:

```bash
cd infineon-results-dashboard
npm install
npm run build:data   # first time: generates src/data/fab-data.js
npm run dev -- --port 8787
```

Open **http://localhost:8787**

| URL | What you get |
| --- | --- |
| http://localhost:8787 | Full polished dashboard (static + live sections) |
| http://localhost:8000/health | FastAPI health check |
| http://localhost:3000 | Legacy Next.js dashboard (optional) |

### What needs the control plane?

| Feature | Needs `:8000`? |
| --- | --- |
| Static results from `results.js` | No |
| Live route demo (`/api/predict`, etc.) | No (Worker on `:8787`) |
| Batch CSV next-step (Worker) | No |
| Run registry table | Yes |
| Live compare matrix / charts | Yes |
| Per-example disagreements | Yes |
| Inference job form | Yes |

Set `ZO_ALLOW_DASHBOARD_INFERENCE=1` in `.env` for HF/LLM predictors in inference jobs
(baselines `ngram` / `freq` / `oracle` work by default).

Override the control-plane URL in the browser console if needed:

```js
window.CONTROL_PLANE_URL = "http://localhost:8000/api";
```

## Worker API (live demo)

| Method & path | Purpose |
| --- | --- |
| `GET /api/health` | Product-line stats and model status |
| `GET /api/vocab` | Full step catalog |
| `GET /api/sample?family=MOSFET&fraction=0.6` | Sample a partial route |
| `POST /api/predict` | Top-5 next-step prediction |
| `POST /api/predict-batch` | Batch CSV next-step |
| `POST /api/complete` | Complete a partial route |
| `POST /api/anomaly` | Validate a full route against the process rules |

## Control plane API (live sections)

Proxied from the browser to `http://localhost:8000/api`:

| Endpoint | Used for |
| --- | --- |
| `GET /runs` | Registry table |
| `GET /compare/report` | Metric matrix, charts |
| `GET /compare/examples` | Disagreement viewer |
| `GET /runs/{id}/confusion` | Live confusion matrix |
| `GET /runs/{id}/metrics` | Training loss curves |
| `POST /inference/preview` | Validate inference inputs |
| `POST /inference/jobs` | Start async eval job |

## Optional model reranking (Worker demo)

Copy `.dev.vars.example` to `.dev.vars` and set an OpenAI-compatible provider. The model
**only reranks Worker-generated candidates** — it cannot invent step names outside the
allowed candidate list.

```bash
MODEL_BASE_URL=http://localhost:8001/v1
MODEL_NAME=XCombinator/sft-fab-all
MODEL_API_KEY=...
```

`.dev.vars` is gitignored — never commit real keys.

## Deploy

```bash
npm run deploy

# for deployed model reranking:
printf '%s' "$MODEL_API_KEY" | wrangler secret put MODEL_API_KEY
wrangler deploy --var MODEL_NAME:XCombinator/sft-fab-all --var MODEL_BASE_URL:https://your-openai-compatible-endpoint/v1
```

Deployed Workers cannot reach your laptop's `:8000` — live control-plane sections require
a hosted FastAPI backend or will show the offline banner (static + Worker demo still work).

## Going live (placeholder → real)

1. Run the eval harness to produce real per-task / per-family metrics and the training log.
2. Replace the `RESULTS` object in `public/results.js` (or run `npm run build:results`).
3. Reload — static sections re-render from that object.
4. With the control plane running, live compare/inference sections pick up registry runs automatically.
