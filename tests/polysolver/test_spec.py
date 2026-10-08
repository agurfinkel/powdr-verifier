"""The minimal working version of PolySolver, as written in the redesign spec (§5.6.8)."""
from functools import reduce

import pytest

from src.polysolver import (
    BABYBEAR, InvalidDefs, Opaque, PolySolver, UndefinedVariable, Unknown,
    UnsupportedExpr, VarTable, find_uniform_witness,
)

P = BABYBEAR


def add(*es):
    return reduce(lambda a, b: [a, "+", b], es)


def mul(*es):
    return reduce(lambda a, b: [a, "*", b], es)


def sub(a, b):
    return [a, "-", b]


def solver(given, premises=(), ranges=()):
    s = PolySolver(VarTable(), declared=given)
    for i, e in enumerate(premises):
        s.add_premise(e, tag=f"p{i}")
    for x, lo, hi in ranges:
        s.add_range(x, lo, hi, tag=f"r_{x}")
    return s


A = ["a0", "a1", "a2", "a3"]
SUMA = add(*A)


# ---------------------------------------------------------------- M4b

def test_table_interns():
    t = VarTable()
    assert t.var("x") == t.var("x") != t.var("y")
    v = t.fresh("aux")
    with pytest.raises(KeyError):
        t.var(t.name(v))


def test_key_ignores_representation():  # one premise, many spellings of it
    s = solver({"x", "y"}, [sub("x", "y")])
    for q in [add("x", mul(P - 1, "y")),  # constraints format: p-1 for -1
              sub("y", "x"),  # sign of the whole constraint
              mul(3, sub("x", "y")),  # any nonzero constant factor
              add(mul(P - 1, "y"), "x")]:  # order of summands
        assert s.implies(q).rule == "normalize"


def test_trivial_and_unknown():
    s = solver({"x", "y"}, [sub("x", "y")])
    assert s.implies(sub("x", "x")).rule == "trivial"
    assert isinstance(s.implies(sub("x", 1)), Unknown)  # not implied: Unknown, never "false"
    assert isinstance(s.implies(1), Unknown)  # nonzero constant


def test_expansion_is_needed():  # distributivity: f*(a+b) vs a*f + b*f
    s = solver({"f", "a", "b"}, [mul("f", add("a", "b"))])
    assert s.implies(add(mul("a", "f"), mul("b", "f"))).rule == "normalize"


def test_defs_checks():
    s = solver({"x"}, [sub("x", 1)])
    with pytest.raises(InvalidDefs):
        s.make_defs({"x": 0})  # D1: x is given
    with pytest.raises(InvalidDefs):
        s.make_defs({"u": "z"})  # D2: z is not given
    with pytest.raises(InvalidDefs):
        s.make_defs({"u": "x", "v": "u"})  # D2: chain
    with pytest.raises(UnsupportedExpr):
        s.make_defs({"u": ["QuotientOrZero", "x", "x"]})
    d = s.make_defs({"u": Opaque("QuotientOrZero(x, x)")})  # D3: opaque, not an error
    assert "opaque" in s.implies(sub("u", 1), d).note
    with pytest.raises(UndefinedVariable):
        s.implies(sub("w", 1))  # gap in the mapping


def test_premises_grow():
    s = solver({"x", "y"})
    q = sub("x", "y")
    assert isinstance(s.implies(q), Unknown)
    s.add_premise(sub("y", "x"), tag="later")
    assert s.implies(q).rule == "normalize"  # the Unknown was not cached
    with pytest.raises(UndefinedVariable):
        s.add_premise(sub("z", 1), tag="bad")


# The soundness direction of the is-zero step: given = After, candidate = Before.
AFTER_GIVEN = set(A) | {"cmp", "f"}
AFTER = [mul(sub(1, "cmp"), SUMA),  # (1 - cmp) * sum a
         sub(mul("f", SUMA), "cmp")]  # f * sum a - cmp
BEFORE_C = sub("cmp", add(*[mul(a, f"inv{i}") for i, a in enumerate(A)]))
BEFORE_PRODUCTS = [mul(sub(1, "cmp"), a) for a in A]
INVS = {f"inv{i}" for i in range(4)}


def test_is_zero_witness_search():
    s = solver(AFTER_GIVEN, AFTER)
    r = find_uniform_witness(s, {}, INVS, [BEFORE_C], shortlist=["cmp", "a0", "f"])
    assert r == "f"  # "cmp" and "a0" were tried and rejected
    d = s.make_defs({i: "f" for i in INVS})
    v = s.implies(BEFORE_C, d)
    assert v.rule == "normalize" and v.premises == ("p1",)
    assert all(isinstance(s.implies(q, d), Unknown) for q in BEFORE_PRODUCTS)  # until M4c


# ---------------------------------------------------------------- M4c

BYTES = [(a, 0, 255) for a in A]


def test_nowrap_split_with_bytes():
    s = solver(AFTER_GIVEN, AFTER, BYTES)
    for q in BEFORE_PRODUCTS:
        v = s.implies(q)
        assert v.rule == "nowrap-split" and v.premises == ("p0",) and set(v.ranges) == set(A)


def test_nowrap_split_needs_every_range():
    s = solver(AFTER_GIVEN, AFTER, BYTES[:3])  # a3 has no range
    v = s.implies(BEFORE_PRODUCTS[0])
    assert isinstance(v, Unknown) and "a3" in v.note


def test_nowrap_split_bound():  # sum of hi_i must stay below p
    edge = (P - 1) // 4  # 503316480: 4 * edge < p
    ok = solver(AFTER_GIVEN, AFTER, [(a, 0, edge) for a in A])
    bad = solver(AFTER_GIVEN, AFTER, [(a, 0, edge + 1) for a in A])
    assert ok.implies(BEFORE_PRODUCTS[0]).rule == "nowrap-split"
    assert isinstance(bad.implies(BEFORE_PRODUCTS[0]), Unknown)


def test_nowrap_split_declines():
    neg = [mul(sub(1, "cmp"), add("a0", mul(P - 1, "a1")))]  # a0 - a1: coefficient p-1
    cst = [mul(sub(1, "cmp"), add("a0", "a1", 1))]  # constant term in the sum
    for prem in (neg, cst):
        s = solver({"cmp", "a0", "a1"}, prem, BYTES[:2])
        assert isinstance(s.implies(mul(sub(1, "cmp"), "a0")), Unknown)


def test_opposite_polarity():  # cmp * sum a = 0, f * sum a + cmp - 1 = 0
    s = solver(AFTER_GIVEN, [mul("cmp", SUMA), add(mul("f", SUMA), "cmp", P - 1)], BYTES)
    d = s.make_defs({i: "f" for i in INVS})
    c = add(*[mul(a, f"inv{i}") for i, a in enumerate(A)], "cmp", P - 1)
    assert s.implies(c, d).rule == "normalize"
    assert all(s.implies(mul("cmp", a)).rule == "nowrap-split" for a in A)
