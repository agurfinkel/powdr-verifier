"""PolySolver: decides  P and R and Defs |= (q = 0)  over F_p. Sound, incomplete.

- P: premises, polynomials each read as "= 0".
- R: ranges lo <= x <= hi on the integer representative of a variable.
- Defs: definitions for variables that are not declared.

The answer is ``Implied`` (with a reason) or ``Unknown``. There is no
"not implied": Unknown only means no rule applied.

Rules, cheapest first:
- N (normalize): q has the same zero key as one premise.
- S (no-wrap split): a premise G * (c1*a1 + ... + cn*an) with ranged a_i whose
  weighted sum cannot wrap past p implies G * a_j = 0 for every j.

The solver knows nothing about circuits: only variables, polynomials,
ranges and definitions.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from . import poly as P
from .defs import NO_DEFS, Defs, InvalidDefs, Opaque
from .expr import Expr, product_factors, variables
from .table import Var, VarTable

BABYBEAR = 2013265921


class UndefinedVariable(LookupError):
    """A variable is neither declared nor defined: a caller error."""


@dataclass(frozen=True)
class Implied:
    rule: str  # "trivial" | "normalize" | "nowrap-split"
    premises: tuple[str, ...] = ()  # tags of the premises used
    ranges: tuple[str, ...] = ()  # variables whose ranges were used
    defs: tuple[str, ...] = ()  # defined variables that occur in q


@dataclass(frozen=True)
class Unknown:
    note: str  # why each rule declined, for triage
    # Machine-readable reason: "opaque", "expansion-limit", "nonzero-constant",
    # "no-match", or a near miss of rule S: "nowrap-missing-range", "nowrap-bound".
    kind: str = "no-match"


Verdict = Implied | Unknown


@dataclass
class _Premise:
    """One premise, kept as given. Its expansion is computed only when needed."""

    tag: str  # the caller's name for it, reported back in verdicts
    expr: Expr  # the expression exactly as given
    key: P.Key | None = None  # zero key; None until computed, or if too big
    expanded: bool = False  # True once we tried to compute the key
    # Each top-level factor expanded on its own (for rule S); None if too big.
    factors: list | None = None


class PolySolver:
    """Premises and ranges only grow; an Implied verdict therefore stays valid."""

    def __init__(
        self,
        table: VarTable,
        declared: Iterable[str],
        p: int = BABYBEAR,
        max_terms: int = P.MAX_TERMS,
    ):
        self.table = table
        self.p = p
        self.max_terms = max_terms
        self.declared: frozenset[Var] = frozenset(table.var(n) for n in declared)
        self._premises: list[_Premise] = []
        # Premise index for rule N: variable -> ids of the premises that mention it.
        # Built from the syntax alone, so adding a premise costs no expansion.
        self._by_var: dict[Var, list[int]] = defaultdict(list)
        # Sum index for rule S: variable -> (premise id, factor position) for each
        # factor that is a plain sum c1*a1 + ... + cn*an containing the variable.
        self._sum_index: dict[Var, list[tuple[int, int]]] = defaultdict(list)
        self._ranges: dict[Var, tuple[int, int, list[str]]] = {}  # v -> (lo, hi, tags)
        # Verdict caches, keyed by the query's zero key. Implied stays true as
        # premises grow; Unknown may not, so that cache is cleared on every add.
        self._implied: dict[P.Key, Implied] = {}
        self._unknown: dict[P.Key, Unknown] = {}
        self.premises_expanded = 0  # how many premise keys were computed

    # ------------------------------------------------------------ inputs

    def add_premise(self, e: Expr, tag: str) -> None:
        """Assert ``e = 0``. Trusted: the caller vouches for it."""
        vs = self._declared_vars(e, "premise")
        idx = len(self._premises)
        prem = _Premise(tag, e)
        self._premises.append(prem)
        # Index by the variables as written; the full key is computed lazily.
        for v in vs:
            self._by_var[v].append(idx)
        self._index_sums(idx, prem)
        self._unknown.clear()  # an earlier Unknown might now be provable

    def add_range(self, x: str, lo: int, hi: int, tag: str) -> None:
        """Assert ``lo <= x <= hi``; a second range on x intersects with the first."""
        if not 0 <= lo <= hi < self.p:
            raise ValueError(f"bad range [{lo}, {hi}] for {x}")
        (v,) = self._declared_vars(x, "range")
        if v in self._ranges:
            # Both ranges hold, so x lies in their intersection.
            old_lo, old_hi, tags = self._ranges[v]
            lo, hi, tags = max(lo, old_lo), min(hi, old_hi), tags + [tag]
        else:
            tags = [tag]
        self._ranges[v] = (lo, hi, tags)
        self._unknown.clear()

    def make_defs(self, mapping: dict[str, Expr | Opaque]) -> Defs:
        """Validate definitions (D1 fresh, D2 flat, D3 polynomial or opaque)."""
        polys, opaque = {}, set()
        for name, rhs in mapping.items():
            v = self._var(name)
            if v in self.declared:
                raise InvalidDefs(f"D1: {name} is declared, it cannot be defined")
            if isinstance(rhs, Opaque):
                opaque.add(v)
                continue
            for n in variables(rhs):  # raises UnsupportedExpr outside the grammar
                if self._var(n) not in self.declared:  # also rules out chains
                    raise InvalidDefs(f"D2: {name} := ... mentions undeclared {n}")
            try:
                polys[v] = P.expand(rhs, self._plain, self.p, self.max_terms)
            except P.ExpansionLimit:
                polys[v] = None  # queries through v report the limit
        return Defs(polys, frozenset(opaque), id(self.table))

    # ------------------------------------------------------------ query

    def implies(self, q: Expr, defs: Defs = NO_DEFS) -> Verdict:
        """Is ``q = 0`` implied? ``q`` itself is never modified."""
        if defs is not NO_DEFS and defs.owner != id(self.table):
            raise ValueError("defs were made by a solver with another table")
        names = variables(q)
        # Sort q's non-declared variables into defined (used) and opaque ones;
        # anything else is a caller error.
        used, opaque = [], []
        for n in sorted(names):
            v = self._var(n)
            if v in self.declared:
                continue
            if not defs.defines(v):
                raise UndefinedVariable(f"{n} is neither declared nor defined")
            (opaque if v in defs.opaque else used).append(n)
        if opaque:
            return Unknown(f"opaque: {', '.join(opaque)}", "opaque")

        # Expansion resolves each defined variable to its cached definition,
        # so q[Defs] is never built as an expression.
        def lookup(n: str) -> P.Poly:
            v = self._var(n)
            if v in defs.polys:
                if defs.polys[v] is None:
                    raise P.ExpansionLimit
                return defs.polys[v]
            return P.var_poly(v)

        try:
            k = P.zkey(P.expand(q, lookup, self.p, self.max_terms), self.p)
        except P.ExpansionLimit:
            return Unknown("expansion limit", "expansion-limit")

        # Queries with the same key are the same question; answer once.
        verdict = self._implied.get(k) or self._unknown.get(k)
        if verdict is None:
            verdict = self._decide(k)
            (self._implied if isinstance(verdict, Implied) else self._unknown)[k] = (
                verdict
            )
        if isinstance(verdict, Implied):
            # The cached verdict is shared; report this query's own definitions.
            return Implied(verdict.rule, verdict.premises, verdict.ranges, tuple(used))
        return verdict

    def _decide(self, k: P.Key) -> Verdict:
        """Try the rules in order, cheapest first; the first Implied wins."""
        declined = []
        for rule in (self._rule_n, self._rule_s):
            v = rule(k)
            if isinstance(v, Implied):
                return v
            declined.append(v)
        # Report the most specific reason any rule gave.
        kind = next((v.kind for v in declined if v.kind != "no-match"), "no-match")
        return Unknown("; ".join(v.note for v in declined), kind)

    # ------------------------------------------------------------ rule N

    def _rule_n(self, k: P.Key) -> Verdict:
        """Rule N: q has the same zero key as some premise."""
        if k == P.ZERO:
            return Implied("trivial")
        if P.is_constant(k):
            return Unknown("N: nonzero constant", "nonzero-constant")
        # Expansion only removes variables, so a premise with key k mentions
        # every variable of k syntactically: looking up one of them is enough.
        # Pick the rarest one, so we expand as few premises as possible.
        v = min(P.key_vars(k), key=lambda x: len(self._by_var.get(x, ())))
        for i in self._by_var.get(v, ()):
            if self._premise_key(i) == k:
                return Implied("normalize", (self._premises[i].tag,))
        return Unknown("N: no premise with this normal form")

    def _premise_key(self, i: int) -> P.Key | None:
        """The premise's zero key, expanded on first use and then cached."""
        prem = self._premises[i]
        if not prem.expanded:
            prem.expanded = True
            self.premises_expanded += 1
            try:
                prem.key = P.zkey(
                    P.expand(prem.expr, self._plain, self.p, self.max_terms), self.p
                )
            except P.ExpansionLimit:
                prem.key = None  # too big: this premise never matches
        return prem.key

    # ------------------------------------------------------------ rule S

    def _index_sums(self, idx: int, prem: _Premise) -> None:
        """Index linear factors with >= 2 variables and no constant term.

        Reads the premise's top-level product F1 * F2 * ... as written and
        expands each factor on its own, which is cheap (what if the factor has further products in it?). A factor that is a sum
        like a0 + a1 + a2 + a3 is what rule S splits, so it is indexed under
        each of its variables.
        """
        try:
            prem.factors = [
                P.expand(f, self._plain, self.p, self.max_terms)
                for f in product_factors(prem.expr)
            ]
        except P.ExpansionLimit:
            return  # leave the premise out of rule S; rule N can still use it
        for fi, f in enumerate(prem.factors):
            # Linear: each term is a single variable to the power 1. That also
            # excludes a constant term, whose monomial is empty.
            if len(f) >= 2 and all(len(m) == 1 and m[0][1] == 1 for m in f):
                for ((v, _),) in f:
                    self._sum_index[v].append((idx, fi))

    def _rule_s(self, k: P.Key) -> Verdict:
        """Rule S: q is G * a, where some premise is G * (sum containing a).

        If no summand can wrap past p, G * sum = 0 forces G * a = 0 for every
        summand a. Only ranged variables of q can play the role of a.
        """
        notes, kind = [], "no-match"
        for a in sorted(P.key_vars(k)):
            if a not in self._ranges:  # an unranged a can never pass the bound
                continue
            for idx, fi in self._sum_index.get(a, ()):
                prem = self._premises[idx]
                # Build G * a, where G is the product of the premise's other
                # factors, and check that it is q (up to a constant factor).
                g = P.var_poly(a)
                try:
                    for j, f in enumerate(prem.factors):
                        if j != fi:
                            g = P.mul(g, f, self.p, self.max_terms)
                except P.ExpansionLimit:
                    continue
                if P.zkey(g, self.p) != k:
                    continue
                # The shape matches; only the no-wrap condition is left.
                why = self._no_wrap(prem.factors[fi])
                if why:
                    kind, msg = why
                    notes.append(f"{prem.tag}: {msg}")
                    continue
                # Report every summand: all their ranges were used.
                ranged = sorted(self.table.name(v) for ((v, _),) in prem.factors[fi])
                return Implied("nowrap-split", (prem.tag,), tuple(ranged))
        return Unknown(
            "S: " + ("; ".join(notes) if notes else "no ranged sum premise"), kind
        )

    def _no_wrap(self, s: P.Poly) -> tuple[str, str] | None:
        """(kind, message) saying why the sum ``s`` might wrap past p, or None."""
        # Scale monic first: a unit multiple of s has the same zeros.
        s = dict(P.zkey(s, self.p))
        # Every summand needs an upper bound, or the sum could be anything.
        missing = [self.table.name(v) for ((v, _),) in s if v not in self._ranges]
        if missing:
            return "nowrap-missing-range", f"no range for {', '.join(sorted(missing))}"
        # With 0 <= a_i <= hi_i the integer sum is at most sum c_i * hi_i. If that
        # stays below p, "sum = 0 mod p" means the integer sum is 0, so each a_i = 0.
        bound = sum(c * self._ranges[v][1] for ((v, _),), c in s.items())
        if bound >= self.p:
            return "nowrap-bound", f"sum of c_i * hi_i is {bound} >= p"
        return None

    # ------------------------------------------------------------ helpers

    def _var(self, name: str) -> Var:
        """The Var of a name; a fresh variable's name is a caller error."""
        try:
            return self.table.var(name)
        except KeyError as exc:
            raise UndefinedVariable(str(exc)) from None

    def _plain(self, name: str) -> P.Poly:
        """Lookup for expanding premises: a name is just its variable."""
        return P.var_poly(self._var(name))

    def _declared_vars(self, e: Expr, what: str) -> list[Var]:
        """The Vars of ``e``; all must be declared (premises and ranges only)."""
        out = []
        for n in variables(e):
            v = self._var(n)
            if v not in self.declared:
                raise UndefinedVariable(f"{what} mentions undeclared {n}")
            out.append(v)
        return out
