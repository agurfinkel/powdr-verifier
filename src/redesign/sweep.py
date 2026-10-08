"""Step B: one obligation per candidate constraint, discharged up a ladder.

For a candidate constraint c the obligation is c[w]: it must follow from the
reference side's constraints and byte ranges. The rungs, cheapest first:

  trivial    - c[w] is identically zero (canonical key is empty)
  identical  - c[w] has the same canonical key as a reference constraint
  normalize / nowrap-split - PolySolver proves it
  undecided  - nothing applied; left for the SMT rungs (not built yet)

Results are cached by canonical key. A direction is verified only if no
obligation is undecided. "Verified" covers the algebraic constraints only:
bus interactions are not checked yet.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace

from src.lens.canon import CanonError, canon_zero
from src.polysolver import (
    BABYBEAR,
    Expr,
    Implied,
    Opaque,
    PolySolver,
    UndefinedVariable,
    poly,
    variables,
)

from .circuit import Circuit, substitute
from .mapping import Mapping

# Rungs that prove an obligation; anything else is "undecided".
DISCHARGED = ("trivial", "identical", "normalize", "nowrap-split")


@dataclass(frozen=True)
class Obligation:
    """The verdict for one candidate constraint under the mapping."""

    name: str  # "<label>:c<i>", or "<column>:def<i>" for an extra definition
    rung: str  # one of DISCHARGED, or "undecided"
    detail: str = ""  # premises used, or why it is undecided
    cached: bool = False  # answered from an identical earlier obligation
    # Undecided only: a short, fixed category (see REASONS).
    reason: str = ""
    # True if c mentions a column defined by the mapping, so c[w] differs from c.
    via_mapping: bool = False


# Why an obligation can stay undecided. Fixed strings, so reports can group them.
REASONS = {
    "unmapped column": "mentions a column the mapping could not define",
    "mapping gap": "mentions a column that is neither shared nor defined",
    "opaque QuotientOrZero": "needs the meaning of a QuotientOrZero column",
    "opaque IfEqZero": "needs the meaning of an IfEqZero column",
    "opaque other": "needs the meaning of a non-polynomial derived column",
    "opaque extra definition": "a second, non-polynomial definition of a column",
    "near miss: range missing": "rule S fits, but a summand has no range",
    "near miss: sum may wrap": "rule S fits, but the ranges are too wide",
    "nonzero constant": "c[w] is a nonzero constant: only contradictory premises imply it",
    "expansion limit": "too many terms to expand",
    "bus-only columns": "mentions a column the reference constrains only via"
    " bus interactions; likely needs bus facts (Step C)",
    "linear, no match": "linear; may follow from a combination of reference"
    " constraints (elimination rule not built)",
    "nonlinear, no match": "nonlinear; needs several premises or SMT",
}


@dataclass
class Report:
    """All obligations of one step in one direction."""

    direction: str
    mapping: Mapping
    obligations: list[Obligation] = field(default_factory=list)
    premises_expanded: int = 0

    @property
    def verified(self) -> bool:
        """Every obligation discharged and every column mapped (algebraic part only)."""
        return (
            all(o.rung in DISCHARGED for o in self.obligations)
            and not self.mapping.unmapped
        )

    def counts(self) -> Counter:
        """Counts by rung."""
        return Counter(o.rung for o in self.obligations)

    def categories(self) -> Counter:
        """Counts by (rung, subcategory)."""
        return Counter(category(o) for o in self.obligations)


def category(o: Obligation) -> tuple[str, str]:
    """(rung, subcategory): the undecided reason, or whether the mapping was used."""
    if o.rung == "undecided":
        return o.rung, o.reason
    if o.rung == "trivial":
        return o.rung, ""
    return o.rung, "via mapping" if o.via_mapping else "direct"


def sweep(
    direction: str, ref: Circuit, cand: Circuit, mapping: Mapping, solver: PolySolver
) -> Report:
    """Check every candidate constraint against the reference under ``mapping``."""
    report = Report(direction, mapping)
    defs = solver.make_defs(mapping.defs)
    # Polynomial definitions can be written into c to get c[w] for the
    # canonical rungs. Opaque or unmapped columns cannot, so they skip to PolySolver.
    poly_defs = {c: d for c, d in mapping.defs.items() if not isinstance(d, Opaque)}
    blocked = mapping.unmapped | {
        c for c, d in mapping.defs.items() if isinstance(d, Opaque)
    }
    # Canonical keys of the reference constraints, for the "identical" rung.
    ref_keys = {_key(c) for c in ref.constraints} - {None}
    # Reference columns that no reference constraint mentions: only buses touch them.
    in_constraints = (
        set().union(*map(variables, ref.constraints)) if ref.constraints else set()
    )
    bus_only = ref.columns - in_constraints
    cache: dict = {}  # canonical key of c[w] -> its verdict

    def check(name: str, c: Expr) -> Obligation:
        """Run one obligation up the ladder, cheapest rung first."""
        cols = variables(c)
        mapped = bool(cols & mapping.defs.keys())
        # No witness for some column: nothing to prove it with.
        if cols & mapping.unmapped:
            missing = ", ".join(sorted(cols & mapping.unmapped))
            return Obligation(
                name, "undecided", f"unmapped: {missing}", reason="unmapped column"
            )
        key = None
        if not cols & blocked:  # canonical rungs need c[w] written out
            key = _key(substitute(c, poly_defs))
            # Rung 0: the same obligation was already decided in this step.
            if key is not None and key in cache:
                return replace(cache[key], name=name, cached=True, via_mapping=mapped)
            # Rung 1: c[w] is identically zero, e.g. a leftover literal 0.
            if key == ():
                return _remember(cache, key, Obligation(name, "trivial"))
            # Rung 2: the reference has this very constraint (up to spelling).
            if key in ref_keys:
                ob = Obligation(name, "identical", via_mapping=mapped)
                return _remember(cache, key, ob)
        # PolySolver rung. It takes c and the definitions, not c[w]: it
        # expands the definitions itself and also handles opaque columns.
        try:
            v = solver.implies(c, defs)
        except UndefinedVariable as exc:
            return Obligation(name, "undecided", str(exc), reason="mapping gap")
        if isinstance(v, Implied):
            detail = ", ".join(v.premises + tuple(f"range {r}" for r in v.ranges))
            ob = Obligation(name, v.rule, detail, via_mapping=mapped)
            return _remember(cache, key, ob)
        # Undecided: record why, so reports can group these.
        reason = _reason(v.kind, cols, mapping)
        if reason == "no single premise":
            reason = _no_match_reason(substitute(c, poly_defs), bus_only)
        ob = Obligation(name, "undecided", v.note, reason=reason, via_mapping=mapped)
        return _remember(cache, key, ob)

    for i, c in enumerate(cand.constraints):
        report.obligations.append(check(f"{cand.label}:c{i}", c))
    # A column with several recipes was mapped by the first; each other recipe
    # is a claim too: col - rhs must hold. Opaque ones cannot be checked here.
    for i, (col, rhs) in enumerate(mapping.extra):
        if isinstance(rhs, Opaque):
            report.obligations.append(
                Obligation(
                    f"{col}:def{i + 1}",
                    "undecided",
                    "opaque definition",
                    reason="opaque extra definition",
                )
            )
        else:
            report.obligations.append(check(f"{col}:def{i + 1}", [col, "-", rhs]))
    report.premises_expanded = solver.premises_expanded
    return report


# PolySolver's Unknown.kind -> our reason label. "no single premise" is a
# placeholder that ``_no_match_reason`` refines.
_KIND_REASON = {
    "expansion-limit": "expansion limit",
    "nonzero-constant": "nonzero constant",
    "nowrap-missing-range": "near miss: range missing",
    "nowrap-bound": "near miss: sum may wrap",
    "no-match": "no single premise",
}


def _reason(kind: str, cols: set[str], mapping: Mapping) -> str:
    """Turn PolySolver's Unknown.kind into one of REASONS."""
    if kind != "opaque":
        return _KIND_REASON[kind]
    # Name the kind of recipe hiding behind the opaque column(s).
    kinds = set()
    for c in cols:
        d = mapping.defs.get(c)
        if isinstance(d, Opaque) and isinstance(d.content, dict):
            kinds |= set(d.content)
    if kinds == {"QuotientOrZero"}:
        return "opaque QuotientOrZero"
    if kinds == {"IfEqZero"}:
        return "opaque IfEqZero"
    return "opaque other"


def _no_match_reason(cw: Expr, bus_only: set[str]) -> str:
    """Split "no single premise" by checkable facts about c[w]."""
    # A column only the buses constrain has no algebraic premise to come from.
    if variables(cw) & bus_only:
        return "bus-only columns"
    # Otherwise tell linear from nonlinear: a linear one could follow from a
    # combination of premises, which a future elimination rule would find.
    try:
        p = poly.expand(cw, lambda n: poly.var_poly(n), BABYBEAR)
    except poly.ExpansionLimit:
        return "nonlinear, no match"
    linear = all(poly.degree(m) <= 1 for m in p)
    return "linear, no match" if linear else "nonlinear, no match"


def _key(e: Expr):
    """lens's canonical key of ``e = 0``, or None if lens rejects the expression."""
    try:
        return canon_zero(e)
    except CanonError:
        return None


def _remember(cache: dict, key, ob: Obligation) -> Obligation:
    """Cache a verdict under its key (when there is one) and return it."""
    if key is not None:
        cache[key] = ob
    return ob
