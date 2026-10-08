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
from dataclasses import dataclass, field

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
    # Rule S's view, computed on first use: the top-level factors as written,
    # their expansions (filled in only as needed), and which ones are sums.
    factors: list | None = None  # the expressions of the top-level * chain
    polys: dict = field(default_factory=dict)  # factor position -> Poly (or None)
    sums: list | None = None  # [(factor position, Poly)] for the linear sums


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
        self._ranges: dict[Var, tuple[int, int, list[str]]] = {}  # v -> (lo, hi, tags)
        # Verdict caches, keyed by the query's zero key. Implied stays true as
        # premises grow; Unknown may not, so that cache is cleared on every add.
        self._implied: dict[P.Key, Implied] = {}
        self._unknown: dict[P.Key, Unknown] = {}
        self.premises_expanded = 0  # premise keys computed (rule N)
        self.premises_split = 0  # premises split into factors (rule S)

    # ------------------------------------------------------------ inputs

    def add_premise(self, e: Expr, tag: str) -> None:
        """Assert ``e = 0``. Trusted: the caller vouches for it."""
        vs = self._declared_vars(e, "premise")
        idx = len(self._premises)
        prem = _Premise(tag, e)
        self._premises.append(prem)
        # Index by the variables as written. Nothing is expanded here: rule N
        # and rule S expand a premise only when a query first needs it.
        for v in vs:
            self._by_var[v].append(idx)
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

    def _premise_sums(self, i: int) -> list:
        """The premise's top-level factors that are sums c1*a1 + ... + cn*an.

        Computed on first use and cached, so only premises that rule S actually
        looks at are ever split. Each factor is fully expanded, so a factor that
        becomes a sum only after cancellation (x*y - x*y + a + b) is still found.
        """
        prem = self._premises[i]
        if prem.sums is None:
            self.premises_split += 1
            prem.factors = product_factors(prem.expr)
            prem.sums = []
            for fi, f in enumerate(prem.factors):
                # Expansion never adds variables, so fewer than 2 means never a sum.
                if len(variables(f)) < 2:
                    continue
                poly = self._factor_poly(prem, fi)
                # Linear with no constant: every term is one variable to the power 1.
                if (
                    poly
                    and len(poly) >= 2
                    and all(len(m) == 1 and m[0][1] == 1 for m in poly)
                ):
                    prem.sums.append((fi, poly))
        return prem.sums

    def _factor_poly(self, prem: _Premise, fi: int) -> P.Poly | None:
        """Expansion of one factor, cached; None if it is too big."""
        if fi not in prem.polys:
            try:
                prem.polys[fi] = P.expand(
                    prem.factors[fi], self._plain, self.p, self.max_terms
                )
            except P.ExpansionLimit:
                prem.polys[fi] = None
        return prem.polys[fi]

    def _rule_s(self, k: P.Key) -> Verdict:
        """Rule S: q is G * a, where some premise is G * (sum containing a).

        If no summand can wrap past p, G * sum = 0 forces G * a = 0 for every
        summand a. Only ranged variables of q can play the role of a.
        """
        notes, kind = [], "no-match"
        for a in sorted(P.key_vars(k)):
            if a not in self._ranges:  # an unranged a can never pass the bound
                continue
            # A premise whose sum contains a mentions a syntactically, so the
            # premise index finds every candidate (as in rule N).
            for idx in self._by_var.get(a, ()):
                prem = self._premises[idx]
                for fi, s in self._premise_sums(idx):
                    if ((a, 1),) not in s:
                        continue
                    g = self._other_factors_times(prem, fi, a)
                    # The shape must match q (up to a constant factor) first.
                    if g is None or P.zkey(g, self.p) != k:
                        continue
                    # The shape matches; only the no-wrap condition is left.
                    why = self._no_wrap(s)
                    if why:
                        kind, msg = why
                        notes.append(f"{prem.tag}: {msg}")
                        continue
                    # Report every summand: all their ranges were used.
                    ranged = sorted(self.table.name(v) for ((v, _),) in s)
                    return Implied("nowrap-split", (prem.tag,), tuple(ranged))
        return Unknown(
            "S: " + ("; ".join(notes) if notes else "no ranged sum premise"), kind
        )

    def _other_factors_times(self, prem: _Premise, fi: int, a: Var) -> P.Poly | None:
        """G * a, where G is the product of the premise's factors other than ``fi``."""
        g = P.var_poly(a)
        try:
            for j in range(len(prem.factors)):
                if j != fi:
                    f = self._factor_poly(prem, j)
                    if f is None:
                        return None
                    g = P.mul(g, f, self.p, self.max_terms)
        except P.ExpansionLimit:
            return None
        return g

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
