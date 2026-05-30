#!/usr/bin/env python3
"""Run HF inference without uv or the full monorepo install.

Prerequisites (once per machine):
  python -m pip install -r requirements-inference.txt

Optional: copy .env.example to .env and set HF_TOKEN for private XCombinator models.

Examples:
  python scripts/hub_infer.py
  python scripts/hub_infer.py --model XCombinator/leonardo-smoke-qwen-0.5b-lora --prompt "Say OK"
  python scripts/hub_infer.py --model path/to/local/lora --base-model Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _bootstrap() -> None:
    import types

    common = _repo_root() / "packages" / "common"
    pkg_root = common / "zo_common"
    if str(common) not in sys.path:
        sys.path.insert(0, str(common))
    # Stub zo_common so hub_inference loads without pulling yaml/registry via __init__.py
    if "zo_common" not in sys.modules:
        pkg = types.ModuleType("zo_common")
        pkg.__path__ = [str(pkg_root)]
        pkg.__package__ = "zo_common"
        sys.modules["zo_common"] = pkg
    env_file = _repo_root() / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        key, _, val = s.partition("=")
        key = key.strip()
        if key:
            os.environ.setdefault(key, val.strip().strip('"').strip("'"))


def main() -> int:
    _bootstrap()
    parser = argparse.ArgumentParser(description="Hugging Face local inference (full or LoRA).")
    parser.add_argument(
        "--model",
        default=os.environ.get("ZO_INFER_MODEL", "XCombinator/leonardo-smoke-qwen-0.5b-lora"),
        help="HF repo id, local path, or base:adapter (default: ZO_INFER_MODEL or smoke LoRA).",
    )
    parser.add_argument(
        "--base-model",
        default=os.environ.get("ZO_INFER_BASE_MODEL"),
        help="Base model for LoRA adapters (auto-detected from adapter_config when possible).",
    )
    parser.add_argument(
        "--prompt",
        default="Say hello in one short sentence.",
        help="User message for a single-turn completion.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    from zo_common.hub_inference import HubInferenceClient, ensure_inference_deps

    ensure_inference_deps()
    client = HubInferenceClient(args.model, base_model=args.base_model, max_new_tokens=args.max_new_tokens)
    print(f"model={client.model_id!r} base={client.spec.base_model!r}")
    out = client.complete(args.prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
