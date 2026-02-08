#src/tools_calc.py
from __future__ import annotations
import ast
import math
import operator as op
from typing import Callable, Dict

_ALLOWED_BIN_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
}
_ALLOWED_UNARY_OPS = {ast.USub: op.neg, ast.UAdd: op.pos}

def _sin_deg(x: float) -> float:
    return math.sin(math.radians(x))

def _cos_deg(x: float) -> float:
    return math.cos(math.radians(x))

_ALLOWED_FUNCS: Dict[str, Callable[[float], float]] = {
    "sin": _sin_deg,
    "cos": _cos_deg,
}

def safe_eval(expr: str) -> float:
    expr = expr.strip().replace("−", "-").replace("–", "-")
    node = ast.parse(expr, mode="eval")

    def _eval(n) -> float:
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float)):
                return float(n.value)
            raise ValueError("Bad constant")
        if isinstance(n, ast.Num):
            return float(n.n)
        if isinstance(n, ast.BinOp):
            t = type(n.op)
            if t not in _ALLOWED_BIN_OPS:
                raise ValueError("Bad operator")
            return _ALLOWED_BIN_OPS[t](_eval(n.left), _eval(n.right))
        if isinstance(n, ast.UnaryOp):
            t = type(n.op)
            if t not in _ALLOWED_UNARY_OPS:
                raise ValueError("Bad unary operator")
            return _ALLOWED_UNARY_OPS[t](_eval(n.operand))
        if isinstance(n, ast.Call):
            if not isinstance(n.func, ast.Name):
                raise ValueError("Bad function")
            fname = n.func.id
            if fname not in _ALLOWED_FUNCS:
                raise ValueError(f"Function {fname} not allowed")
            if len(n.args) != 1:
                raise ValueError("Only 1-arg funcs allowed")
            return float(_ALLOWED_FUNCS[fname](_eval(n.args[0])))
        raise ValueError("Unsupported expression")

    return float(_eval(node))

def eval_equality(expr: str, eps: float = 1e-6) -> bool:
    expr = expr.strip().replace("−", "-").replace("–", "-")
    if "=" in expr:
        left, right = expr.split("=", 1)
        lv = safe_eval(left.strip())
        rv = safe_eval(right.strip())
        return abs(lv - rv) <= eps
    return abs(safe_eval(expr)) > eps
