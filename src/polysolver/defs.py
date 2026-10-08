"""Definitions ``x := t`` for variables that are not declared.

A ``Defs`` is built and validated by ``PolySolver.make_defs`` (checks D1-D3).
It caches the expansion of each right-hand side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class InvalidDefs(ValueError):
    """A definition breaks D1 (fresh) or D2 (flat)."""


@dataclass(frozen=True)
class Opaque:
    """A right-hand side we never look inside, e.g. a QuotientOrZero recipe."""

    content: Any


@dataclass(frozen=True)
class Defs:
    """Validated definitions; only valid with the solver that made them."""

    polys: dict = field(default_factory=dict)  # Var -> Poly, or None if too big
    opaque: frozenset = frozenset()  # Vars defined as Opaque
    owner: int = 0  # id() of the solver's VarTable

    def defines(self, v: int) -> bool:
        return v in self.polys or v in self.opaque


NO_DEFS = Defs()
