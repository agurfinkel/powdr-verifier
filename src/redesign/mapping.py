"""Step A: the mapping w from candidate columns to terms over reference columns.

Sources, in priority order:
  1. same name     - the column exists on the reference side: maps to itself
  2. derived       - the candidate's own ``derived_columns`` recipe
  3. substitution  - ``_substitutions.json``, composed down to reference columns
  4. witness       - soundness only: one shared witness found by search

Any mapping is sound: a poor choice can only make an obligation unprovable,
never prove a wrong pass (``flip_quant``). So the heuristics here are free to
be heuristic. Same-name columns are declared in the solver and need no entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.polysolver import Expr, Opaque, PolySolver, find_uniform_witness, variables

from .circuit import Circuit, interaction_columns, recipe_columns, substitute


class MissingDefinition(Exception):
    """A new After column has no definition (completeness); powdr should export one."""


@dataclass
class Mapping:
    """The witness w for one direction, plus what we learned building it."""

    # Candidate-only column -> its term over reference columns.
    defs: dict[str, Expr | Opaque] = field(default_factory=dict)
    # Column -> which source defined it ("derived", "substitution", "witness").
    source: dict[str, str] = field(default_factory=dict)
    # Columns no source could define; their obligations stay undecided.
    unmapped: set[str] = field(default_factory=set)
    # Why each unmapped column is unmapped: "witness search failed" or "lookup-only".
    unmapped_reason: dict[str, str] = field(default_factory=dict)
    # Each uniform witness found: (the columns it covers, the shared term).
    witnesses: list[tuple[list[str], Expr]] = field(default_factory=list)
    # Extra definitions of a derived column beyond the first: (column, rhs) to check.
    extra: list[tuple[str, Expr | Opaque]] = field(default_factory=list)


def completeness_mapping(
    before: Circuit, after: Circuit, subs: dict[str, Expr]
) -> Mapping:
    """w: After columns -> terms over Before columns. Raises if a column has no source."""
    m = Mapping()
    # Only After-only columns need a definition; shared ones map to themselves.
    for col in _needed(after, before):
        if not _from_derived(m, col, after) and not _from_subs(m, col, subs, before):
            m.unmapped.add(col)
    # powdr exports a definition for every column it adds, so a gap here is a
    # broken dump, not something to search for.
    if m.unmapped:
        raise MissingDefinition(", ".join(sorted(m.unmapped)))
    return m


def soundness_mapping(
    after: Circuit, before: Circuit, subs: dict[str, Expr], solver: PolySolver
) -> Mapping:
    """w': Before columns -> terms over After columns. ``solver`` holds After's premises."""
    m = Mapping()
    # Before-only columns: try the recorded sources first.
    left = [
        col
        for col in _needed(before, after)
        if not _from_derived(m, col, before) and not _from_subs(m, col, subs, after)
    ]
    # A column used only by lookups (no constraint mentions it) has nothing for
    # the witness search to check against: with no targets, any shortlisted
    # column would "pass". Leave it unmapped. These are typically free columns
    # whose range check the pass dropped; proving them needs table semantics.
    in_constraints = _constraint_columns(before) # all candidate columns in algebraic constraints
    for col in left:
        if col not in in_constraints:
            m.unmapped.add(col)
            m.unmapped_reason[col] = "lookup-only"
    left = [col for col in left if col in in_constraints]
    # What is left was merged away with no hint (e.g. the is-zero inverse
    # markers). Search for one After column that works for a whole group.
    for group in _groups(left, before.constraints):
        targets = [c for c in before.constraints if variables(c) & group]
        shortlist = _shortlist(after, before, targets)
        r = find_uniform_witness(solver, m.defs, group, targets, shortlist)
        if r is None:
            m.unmapped |= group  # reported, never guessed
            for col in group:
                m.unmapped_reason[col] = "witness search failed"
            continue
        for col in group:
            m.defs[col] = r
            m.source[col] = "witness"
        m.witnesses.append((sorted(group), r))
    return m


def _needed(cand: Circuit, ref: Circuit) -> list[str]:
    """Candidate columns the obligations mention that the reference lacks.

    Obligations are the candidate's constraints and its stateless interactions.
    """
    cols = _constraint_columns(cand)
    for bi in cand.stateless:
        cols |= interaction_columns(bi)
    return sorted(cols - ref.columns)


def _constraint_columns(c: Circuit) -> set[str]:
    return (
        set().union(*(variables(e) for e in c.constraints)) if c.constraints else set()
    )


def _from_derived(m: Mapping, col: str, cand: Circuit) -> bool:
    """Define ``col`` by its ``derived_columns`` recipe, if it has one."""
    recipes = cand.derived.get(col)
    if not recipes:
        return False
    defs = [_recipe_def(r) for r in recipes]
    m.defs[col], m.source[col] = defs[0], "derived"
    m.extra += [(col, d) for d in defs[1:]]  # pick one; the rest become obligations
    return True


def _recipe_def(recipe: dict) -> Expr | Opaque:
    """A recipe as a definition PolySolver accepts.

    Constants are polynomials. QuotientOrZero and IfEqZero are not, so they
    become Opaque: still a valid witness, but PolySolver never looks inside.
    """
    if set(recipe) == {"Constant"}:
        return recipe["Constant"]
    return Opaque(recipe)


def _from_subs(m: Mapping, col: str, subs: dict[str, Expr], ref: Circuit) -> bool:
    """Define ``col`` from the substitutions file, if it resolves to ref columns."""
    rhs = _compose(col, subs, ref, set())
    if rhs is None:
        return False
    m.defs[col], m.source[col] = rhs, "substitution"
    return True


def _compose(
    col: str, subs: dict[str, Expr], ref: Circuit, seen: set[str]
) -> Expr | None:
    """Follow substitutions until only reference columns remain, or give up.

    The file covers the whole pipeline, so a replacement can mention a column
    that was itself eliminated later (x -> y, y -> z). We chase such chains;
    ``seen`` stops a cycle from looping forever.
    """
    if col in seen or col not in subs:
        return None
    rhs = subs[col]
    inner = {}
    # Resolve every column of the replacement the reference doesn't have.
    for v in variables(rhs) - ref.columns:
        sub = _compose(v, subs, ref, seen | {col})
        if sub is None:
            return None
        inner[v] = sub
    return substitute(rhs, inner)


def _groups(cols: list[str], constraints: list[Expr]) -> list[set[str]]:
    """Split unknown columns into groups that share a constraint (union-find).

    One block can contain several merged gadgets, each needing its own
    witness. Columns that appear together in a constraint belong to the same
    gadget, so they share one search.
    """
    parent = {c: c for c in cols}

    def find(c: str) -> str:
        # Walk up to the group's root, shortening the path as we go.
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    # Merge all unknowns that occur in the same constraint.
    for e in constraints:
        hit = [c for c in variables(e) if c in parent]
        for c in hit[1:]:
            parent[find(c)] = find(hit[0])
    groups: dict[str, set[str]] = {}
    for c in cols:
        groups.setdefault(find(c), set()).add(c)
    return list(groups.values())


def _shortlist(ref: Circuit, cand: Circuit, targets: list[Expr]) -> list[str]:
    """Reference columns the candidate lacks; those whose recipe touches the targets first."""
    # A merged column is usually replaced by a column the pass added.
    new = sorted(ref.columns - cand.columns)
    touched = set().union(*(variables(t) for t in targets)) if targets else set()

    # The added column's recipe is a strong hint: free_var_123 is defined from
    # cmp and the limbs, the same columns the inverse markers' constraint uses.
    def related(col: str) -> bool:
        return any(recipe_columns(r) & touched for r in ref.derived.get(col, []))

    # Order only affects speed: every candidate is checked by implies anyway.
    return sorted(new, key=lambda c: not related(c))
