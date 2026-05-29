# Agents (tool-use)

Code: `packages/agent/zo_agent/` (`tools.py`, `rollout.py`, `harness.py`, `cli.py`).
Scenarios: `packages/agent/scenarios/*.yaml`.

## Pieces
- **`tools.py`** — a `@tool(name, desc, params)` registry. `specs()` exports OpenAI-style tool
  schemas; `dispatch(name, args)` runs the matching function. Ships a **safe calculator** that
  evaluates arithmetic via an AST walk (`_safe_eval`) — **no Python `eval()`**. Add tools by writing
  a function and decorating it; keep them side-effect-free / sandboxed.
- **`rollout.py`** — `run_episode(system, task, model, base_url, max_steps=6)` runs the tool-calling
  loop: model proposes tool calls → we dispatch → feed results back → repeat until a final answer or
  `max_steps`. Returns `{final, steps, tool_calls, trace}`.
- **`harness.py`** — `Scenario` / `ScenarioSuite` models. `run_suite()` runs each scenario,
  `_judge()`s success, and writes `success_rate`, `avg_steps`, `avg_tool_calls` to the registry.

## Running
- `just agent <scenario> <model>` → `uv run zo-agent run --scenario <suite.yaml> --model <model>`.
- Needs an OpenAI-compatible endpoint that supports **tool calling** (`ZO_MODEL_BASE_URL`). vLLM
  serves tool-calling for many chat models — confirm the served model actually emits tool calls.

## Adding a scenario
Copy `scenarios/example.yaml` (`tool-use-arithmetic`): a system prompt, a task, and a success
criterion the judge checks against the final answer / trace.

## Ideas for "agentic" hackathon points
- Train (SFT or GRPO) a model to use tools *better*, then measure the lift with a scenario suite —
  that closes the loop between [training](training.md) and agent eval and is a strong demo.

## Append below as you learn
- (tools we added + which models tool-call reliably: TBD)
