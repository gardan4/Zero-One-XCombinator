# Local setup for graders (Windows, macOS, Linux)

This is the shortest path to run the dashboard and light local checks. It installs the **light**
workspace only: no GPU stack, no torch, no vLLM, no WSL/bash requirement.

For Leonardo-only judge flows that do not need the dashboard, see
[judge-quickstart.md](judge-quickstart.md). Those can run with Python + pip only.

---

## Minimum requirements

| Tool | Version | Purpose |
|------|---------|---------|
| **Python** | 3.11+ | Runs the setup/dev scripts and backend tools |
| **uv** | latest | Installs the Python workspace from `uv.lock` |
| **Node.js + npm** | 20+ | Runs the Next.js dashboard |
| **Git** | any recent version | Clone the repository |

You do **not** need bash, WSL, `just`, `mise`, a global `next` install, or GPU libraries for the
dashboard smoke path.

Optional teammate conveniences:

- `mise` pins Python/Node/uv/just from `mise.toml`.
- `just` provides shortcuts like `just dev`, but every required command has a plain `uv run python ...` equivalent.

---

## 1. Install the required tools

- **Node.js 20+:** https://nodejs.org/
- **Python 3.11+:** https://www.python.org/downloads/
- **uv:** https://docs.astral.sh/uv/getting-started/installation/

Quick `uv` install commands from the official docs:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

If you already use `mise`, this also works:

```bash
mise trust
mise install
```

---

## 2. Clone and optionally create `.env`

```bash
git clone <repo-url>
cd Zero-One-XCombinator
```

Creating `.env` is optional for the first local dashboard smoke. Copy it only if you need private
Hugging Face models, cluster submission, or non-default paths:

```bash
cp .env.example .env
```

Common optional values:

- `ZO_EXPERIMENTS_DIR` — shared run store across worktrees (optional)
- `HF_TOKEN` — private Hugging Face models (optional)
- `ZO_ALLOW_DASHBOARD_INFERENCE=1` — allow HF/LLM predictors on `/inference` (optional)
- Cluster vars — only for Leonardo SSH jobs

---

## 3. Install project dependencies

Run one command from the repo root:

```bash
uv run python scripts/setup.py
```

This installs:

- Python workspace packages (`zo-common`, `zo-train`, `zo-eval`, `zo-agent`, `zo-backend`, …)
- Frontend `node_modules` under `apps/frontend/`

If you have `just`, `just setup` is the same thing.

Heavy ML (`torch`, `trl`, `vllm`) is **not** installed. On the cluster:

```bash
uv sync --extra gpu
```

---

## 4. Run the stack

**Both servers (backend + frontend):**

```bash
uv run python scripts/dev.py
```

- Backend API: http://localhost:8000 (override with `ZO_API_PORT`)
- Dashboard: http://localhost:3000 (`NEXT_PUBLIC_API_URL` defaults to the API)

If you have `just`, `just dev` is the same thing.

Separate terminals:

```bash
uv run uvicorn zo_backend.main:app --reload --port 8000
uv run python scripts/frontend.py dev
```

On Windows PowerShell you can also run `.\scripts\dev.ps1`, which wraps `dev.py`.

---

## 5. Verify

```bash
uv run pytest
uv run python scripts/frontend.py build
```

Optional developer checks:

```bash
uv run ruff check .
uv run pytest -m integration   # live Hugging Face/network test
```

Open the dashboard → **Runs**, **Compare**, **Inference**, **Copilot**.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `'next' is not recognized` | Run `uv run python scripts/setup.py`; do not install `next` globally |
| `uv: command not found` | Install uv, or use `mise trust && mise install` |
| `npm: command not found` | Install Node.js 20+, or use `mise trust && mise install` |
| `just dev` fails on Windows with bash errors | Use `uv run python scripts/dev.py` (no bash); `just` is optional |
| `npm install` fails inside `uv run python scripts/setup.py` | Fixed in repo: scripts resolve `npm.CMD` to a full path on Windows |
| `test_xcombinator_model_live` fails | Use `uv run pytest` for the default unit suite; run `uv run pytest -m integration` only for live HF/network checks |
| Frontend cannot reach API | Start backend; check `NEXT_PUBLIC_API_URL` in `apps/frontend/.env.local` if set |

---

## What still needs bash (optional)

- `just wt <name>` — git worktree helper (`scripts/wt.sh`)
- Some cluster shell snippets in docs

For graders and day-to-day local dashboard work, the main commands are:

```bash
uv run python scripts/setup.py
uv run python scripts/dev.py
uv run pytest
```
