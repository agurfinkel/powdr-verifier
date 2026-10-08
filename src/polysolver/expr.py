"""The input grammar: expressions exactly as ``json.load`` returns them.

    Expr = int | str | ["-", e] | [a, op, b]    with op in "+", "-", "*"

A ``str`` is a variable name. Anything else is rejected.
"""

from __future__ import annotations

from typing import Any

Expr = Any  # int | str | list, as loaded from JSON

OPS = ("+", "-", "*")


class UnsupportedExpr(ValueError):
    """The value is not in the expression grammar."""


def variables(e: Expr) -> set[str]:
    """The variable names occurring in ``e``."""
    out: set[str] = set()
    _walk(e, out)
    return out


def _walk(e: Expr, out: set[str]) -> None:
    if isinstance(e, bool):  # bool is an int in Python; never valid here
        raise UnsupportedExpr(f"not an expression: {e!r}")
    if isinstance(e, int):
        return
    if isinstance(e, str):
        out.add(e)
        return
    if isinstance(e, list):
        if len(e) == 2 and e[0] == "-":
            _walk(e[1], out)
            return
        if len(e) == 3 and e[1] in OPS:
            _walk(e[0], out)
            _walk(e[2], out)
            return
    raise UnsupportedExpr(f"not an expression: {e!r}")


def product_factors(e: Expr) -> list[Expr]:
    """The operands of the top-level ``*`` chain; ``[e]`` if ``e`` is not a product."""
    if isinstance(e, list) and len(e) == 3 and e[1] == "*":
        return product_factors(e[0]) + product_factors(e[2])
    return [e]
