"""Search for one witness shared by several undefined variables.

A client of ``PolySolver.implies``, not part of it. Any witness it returns is
correct, because each candidate is checked with ``implies``; the shortlist
order only affects speed.
"""

from __future__ import annotations

from collections.abc import Iterable

from .defs import InvalidDefs, Opaque
from .expr import Expr, variables
from .solver import Implied, PolySolver


def find_uniform_witness(
    solver: PolySolver,
    base_w: dict[str, Expr | Opaque],
    unknowns: Iterable[str],
    cand_constraints: Iterable[Expr],
    shortlist: Iterable[Expr],
) -> Expr | None:
    """The first ``r`` in ``shortlist`` such that setting every unknown to ``r``
    makes all constraints that mention an unknown implied, or None."""
    unknowns = set(unknowns)
    targets = [c for c in cand_constraints if variables(c) & unknowns]
    for r in shortlist:
        try:
            defs = solver.make_defs({**base_w, **{u: r for u in unknowns}})
        except InvalidDefs:
            continue  # r is not usable as a definition (e.g. not declared)
        if all(isinstance(solver.implies(c, defs), Implied) for c in targets):
            return r
    return None
