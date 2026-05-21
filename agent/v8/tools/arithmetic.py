"""Restricted-eval arithmetic verifier.

Given an expression string, safely evaluate basic arithmetic. Pure deterministic
helper — emits tool.called / tool.returned events but no source.opened (no
external content).
"""
from __future__ import annotations

import ast
import math
from typing import Any

from agent.events import EventLog


_ALLOWED_FUNCS = {
    "abs": abs, "min": min, "max": max, "round": round, "sum": sum,
    "len": len, "sqrt": math.sqrt, "pow": pow, "log": math.log,
    "log2": math.log2, "log10": math.log10, "exp": math.exp,
    "ceil": math.ceil, "floor": math.floor,
}
_ALLOWED_NAMES = {"pi": math.pi, "e": math.e, "inf": math.inf, "nan": math.nan}


class _SafeEval(ast.NodeVisitor):
    def __init__(self):
        self.errors: list[str] = []

    def visit(self, node):  # type: ignore[override]
        method = "visit_" + type(node).__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def visit_Module(self, node):
        if len(node.body) != 1 or not isinstance(node.body[0], ast.Expr):
            raise ValueError("expression only")
        return self.visit(node.body[0].value)

    def visit_Constant(self, node):
        if isinstance(node.value, (int, float, complex, bool)):
            return node.value
        raise ValueError(f"unsupported constant {type(node.value).__name__}")

    def visit_BinOp(self, node):
        l = self.visit(node.left)
        r = self.visit(node.right)
        ops = {
            ast.Add: lambda a, b: a + b,
            ast.Sub: lambda a, b: a - b,
            ast.Mult: lambda a, b: a * b,
            ast.Div: lambda a, b: a / b,
            ast.FloorDiv: lambda a, b: a // b,
            ast.Mod: lambda a, b: a % b,
            ast.Pow: lambda a, b: a ** b,
        }
        fn = ops.get(type(node.op))
        if fn is None:
            raise ValueError(f"op {type(node.op).__name__} not allowed")
        return fn(l, r)

    def visit_UnaryOp(self, node):
        v = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.UAdd):
            return +v
        raise ValueError("unary op not allowed")

    def visit_Name(self, node):
        if node.id in _ALLOWED_NAMES:
            return _ALLOWED_NAMES[node.id]
        raise ValueError(f"name {node.id} not allowed")

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
            raise ValueError("function not allowed")
        args = [self.visit(a) for a in node.args]
        return _ALLOWED_FUNCS[node.func.id](*args)

    def visit_List(self, node):
        return [self.visit(e) for e in node.elts]

    def visit_Tuple(self, node):
        return tuple(self.visit(e) for e in node.elts)

    def generic_visit(self, node):
        raise ValueError(f"node {type(node).__name__} not allowed")


def safe_arith(expr: str) -> tuple[bool, Any, str]:
    """Returns (ok, value, error_msg)."""
    try:
        tree = ast.parse(expr, mode="exec")
        return True, _SafeEval().visit(tree), ""
    except Exception as e:
        return False, None, str(e)


def verify_arithmetic(
    expression: str,
    expected: str | None = None,
    *,
    log: EventLog,
    parent: str,
) -> dict[str, Any]:
    call_id = log.emit(
        "tool.called",
        parent=parent,
        tool="verify_arithmetic",
        args={"expression": expression[:200], "expected": expected},
    )
    ok, value, err = safe_arith(expression)
    matches = None
    if ok and expected is not None:
        try:
            expected_v = float(expected)
            matches = abs(float(value) - expected_v) < 1e-6
        except Exception:
            matches = str(value).strip() == str(expected).strip()
    log.emit(
        "tool.returned",
        parent=call_id,
        tool="verify_arithmetic",
        ok=ok,
        value=str(value) if ok else None,
        matches_expected=matches,
        error=err if not ok else None,
    )
    return {"ok": ok, "value": value, "matches_expected": matches, "error": err}
