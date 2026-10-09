"""Stateless bus interactions (lookups) as obligations, discharged canonically."""

import collections
import copy

import pytest

from src.lens.loader import load, machine_of
from src.lens.resolve import resolve
from src.paths import POWDR_DUMPS_DIR
from src.redesign import mapping as mapping_module
from src.redesign.check import check, check_step
from src.redesign.circuit import STATELESS_BUSES, Circuit, load_step, load_substitutions


def bus_obligations(report, rung=None):
    return [
        o for o in report.obligations if o.kind == "bus" and rung in (None, o.rung)
    ]


def test_stateless_excludes_stateful_buses():
    c = load_step("keccak", "2106332", "009")
    ids = {bi["id"] for bi in c.bus_interactions}
    assert {0, 1} <= ids  # the dump has memory and execution-bridge traffic
    assert {bi["id"] for bi in c.stateless} <= STATELESS_BUSES
    assert len(c.stateless) == sum(bi["id"] in STATELESS_BUSES for bi in c.bus_interactions)


def test_is_zero_step_lookups_unchanged():
    for r in check("keccak", "2106332", "008", "009"):
        lookups = bus_obligations(r)
        assert lookups and all(o.rung == "identical" for o in lookups), r.direction
    _, snd = check("keccak", "2106332", "008", "009")
    assert snd.verified  # adding lookups does not break the soundness result


def test_noop_step_verified_with_lookups():
    for r in check("keccak", "2099512", "014", "015"):
        assert r.verified and r.counts("bus")["identical"] > 0, r.direction


def test_removed_constant_lookups_are_undecided():
    # remove_disconnected drops 14 lookups with constant args: 5 PcLookup,
    # 2 VariableRangeChecker, 7 BitwiseLookup (lens diff: "bus: -14").
    cmp, snd = check("keccak", "2099512", "005", "006")
    assert cmp.verified  # every After lookup is still in Before
    before = load_step("keccak", "2099512", "005")
    undecided = collections.Counter(
        (before.stateless[int(o.name.split(":b")[1])]["id"], o.reason)
        for o in bus_obligations(snd, "undecided")
    )
    assert undecided == {
        (6, "bus: args differ"): 7,
        (2, "bus: no same-mult reference"): 5,
        (3, "bus: args differ"): 2,
    }


def test_lookup_only_columns_skip_the_witness_search(monkeypatch):
    calls = []
    real = mapping_module.find_uniform_witness

    def spy(solver, base_w, unknowns, targets, shortlist):
        calls.append(list(targets))
        return real(solver, base_w, unknowns, targets, shortlist)

    monkeypatch.setattr(mapping_module, "find_uniform_witness", spy)
    _, snd = check("keccak", "2099512", "003", "004")
    assert all(calls_targets for calls_targets in calls)  # never an empty target list
    lookup_only = {
        "reads_aux__1__base__timestamp_lt_aux__lower_decomp__0_0@10",
        "reads_aux__1__base__timestamp_lt_aux__lower_decomp__1_0@11",
    }
    assert snd.mapping.unmapped == lookup_only
    assert set(snd.mapping.unmapped_reason.values()) == {"lookup-only"}
    assert not snd.mapping.witnesses
    stuck = bus_obligations(snd, "undecided")
    assert len(stuck) == 2 and {o.reason for o in stuck} == {"lookup-only column"}
    assert not snd.counts("constraint")["undecided"]  # constraints are unaffected


@pytest.fixture(scope="module")
def is_zero_dumps():
    def machine(step):
        return machine_of(load(resolve("keccak", "2106332", step, POWDR_DUMPS_DIR).path))

    return machine("008"), machine("009"), load_substitutions("keccak", "2106332")


def _completeness_with(dumps, edit):
    before, after, subs = (copy.deepcopy(d) for d in dumps)
    edit(after)
    (cmp,) = check_step(
        Circuit.from_machine("008", before),
        Circuit.from_machine("009", after),
        subs,
        ("completeness",),
    )
    return cmp


def _first_lookup(machine, bus):
    return next(bi for bi in machine["bus_interactions"] if bi["id"] == bus)


def test_changed_lookup_arg_is_caught(is_zero_dumps):
    def edit(after):
        vrc = _first_lookup(after, 3)  # VariableRangeChecker [x, bits]
        vrc["args"][1] = vrc["args"][1] + 1  # check a different bit width

    cmp = _completeness_with(is_zero_dumps, edit)
    stuck = bus_obligations(cmp, "undecided")
    assert [o.reason for o in stuck] == ["bus: args differ"]
    assert not cmp.verified


def test_zero_multiplicity_lookup_is_trivial(is_zero_dumps):
    def edit(after):
        vrc = _first_lookup(after, 3)
        vrc["mult"] = 0
        vrc["args"][1] = vrc["args"][1] + 1  # would not match, but asserts nothing

    cmp = _completeness_with(is_zero_dumps, edit)
    assert len(bus_obligations(cmp, "trivial")) == 1
    assert not bus_obligations(cmp, "undecided")
