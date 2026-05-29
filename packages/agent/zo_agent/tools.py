"""A tiny, safe tool registry. Add your track's tools here (API calls, retrieval, etc.)."""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from typing import Any

_TOOLS: dict[str, dict[str, Any]] = {}


def tool(name: str, description: str, parameters: dict[str, Any]) -> Callable:
    def deco(fn: Callable) -> Callable:
        _TOOLS[name] = {
            "fn": fn,
            "spec": {
                "type": "function",
                "function": {"name": name, "description": description, "parameters": parameters},
            },
        }
        return fn

    return deco


def specs() -> list[dict[str, Any]]:
    return [t["spec"] for t in _TOOLS.values()]


def dispatch(name: str, args: dict[str, Any]) -> str:
    if name not in _TOOLS:
        return f"error: unknown tool {name!r}"
    try:
        return str(_TOOLS[name]["fn"](**args))
    except Exception as e:  # tools must never crash the rollout
        return f"error: {e}"


# --- example tool: a safe arithmetic evaluator (no eval()) -------------------

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


@tool(
    "calculator",
    "Evaluate a basic arithmetic expression and return the number.",
    {
        "type": "object",
        "properties": {"expression": {"type": "string", "description": "e.g. '2 * (3 + 4)'"}},
        "required": ["expression"],
    },
)
def _calculator(expression: str) -> str:
    return str(_safe_eval(ast.parse(expression, mode="eval").body))
