# Infineon Fab Route Copilot

Small Cloudflare Workers demo for the Industrial AI track.

It serves one UI and one backend:

- `GET /api/health` — product-line stats and model status
- `GET /api/sample?family=MOSFET&fraction=0.6` — sample partial route
- `POST /api/predict` — top-5 next-step prediction
- `POST /api/complete` — route completion
- `POST /api/anomaly` — process-rule validation

The app is product-line aware: `IC`, `IGBT`, and `MOSFET` are ranked against separate route pools.

## Local Run

```bash
cd /root/RD/Project-3-Infineon-Fab-Copilot
npm install
npm run build:data
npm run dev
```

Open the Wrangler URL printed by the terminal.

Without model secrets, the backend uses retrieval over the shipped training routes. That is enough for
the demo and keeps the UI usable while a hosted model is being selected.

## Optional Model Reranking

Copy `.dev.vars.example` to `.dev.vars` and add an OpenAI-compatible provider:

```bash
MODEL_BASE_URL=https://api.featherless.ai/v1
MODEL_NAME=Qwen/Qwen3.6-27B
MODEL_API_KEY=...
```

or:

```bash
MODEL_BASE_URL=https://api.deepseek.com
MODEL_NAME=deepseek-chat
MODEL_API_KEY=...
```

The model only reranks backend-generated candidates. It cannot invent step names outside the allowed
candidate list.

## Deploy

```bash
cd /root/RD/Project-3-Infineon-Fab-Copilot
set -a; . /root/RD/.env; set +a
npm run deploy
```

For deployed model reranking:

```bash
printf '%s' "$MODEL_API_KEY" | wrangler secret put MODEL_API_KEY
wrangler deploy --var MODEL_NAME:Qwen/Qwen3.6-27B --var MODEL_BASE_URL:https://api.featherless.ai/v1
```

If using Featherless, keep the default `MODEL_BASE_URL` in `wrangler.jsonc`. The tested catalog
model name is `Qwen/Qwen3.6-27B`.
