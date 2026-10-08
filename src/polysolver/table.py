"""Names <-> integer variables.

Behind the public API the solver works on small ints, not column names.
Several solvers may share one table; their keys are comparable only then.
"""

from __future__ import annotations

Var = int


class VarTable:
    """Interns names as ``Var`` ints; the same name always gives the same Var."""

    def __init__(self) -> None:
        self._ids: dict[str, Var] = {}
        self._names: list[str] = []
        self._fresh: set[str] = set()

    def var(self, name: str) -> Var:
        """The Var of ``name``, interning it on first use."""
        if name in self._fresh:
            raise KeyError(f"{name!r} is a fresh variable, not a name")
        v = self._ids.get(name)
        if v is None:
            v = self._intern(name)
        return v

    def name(self, v: Var) -> str:
        """The name of ``v``, for reports."""
        return self._names[v]

    def fresh(self, hint: str) -> Var:
        """A new Var whose name ``var()`` will refuse."""
        n = 0
        while f"{hint}#{n}" in self._ids:
            n += 1
        name = f"{hint}#{n}"
        self._fresh.add(name)
        return self._intern(name)

    def _intern(self, name: str) -> Var:
        v = len(self._names)
        self._ids[name] = v
        self._names.append(name)
        return v

    def __len__(self) -> int:
        return len(self._names)
