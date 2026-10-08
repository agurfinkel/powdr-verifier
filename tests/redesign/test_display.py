"""The sweep records are consistent, and the report renders them faithfully."""

import json

from rich.console import Console

from src.redesign.check import check
from src.redesign.corpus import run_sweep
from src.redesign.display import render
from src.redesign.sweep import REASONS


def _sweep():
    return run_sweep("keccak", ["2106332"])


def test_records_add_up():
    data = _sweep()
    steps = data["steps"]
    assert len(steps) == 2 * 40  # dumps 000..040 -> 40 pairs, two directions each
    for s in steps:
        assert sum(s["categories"].values()) == s["obligations"]
        undecided = sum(n for k, n in s["categories"].items() if k.startswith("undecided|"))
        assert s["verified"] == (undecided == 0 and not s["unmapped"] and not s["error"])
        for k in s["categories"]:
            rung, sub = k.split("|")
            if rung == "undecided":
                assert sub in REASONS, sub  # every reason is a documented one
    json.dumps(data)  # saved as-is by --save


def test_bus_only_reason_on_memory_pass():
    (r,) = check("keccak", "2106332", "010", "011", ("completeness",))
    reasons = {o.reason for o in r.obligations if o.rung == "undecided"}
    assert "bus-only columns" in reasons  # values the memory pass read off the bus


def test_is_zero_reasons():
    cmp, snd = check("keccak", "2106332", "008", "009")
    assert sorted(o.reason for o in cmp.obligations if o.rung == "undecided") == [
        "nonlinear, no match",  # (1 - cmp) * sum a: a sum of four premises
        "opaque QuotientOrZero",  # f * sum a - cmp
    ]
    assert snd.categories()[("normalize", "via mapping")] == 1  # through the witness


def test_report_renders_the_numbers():
    data = _sweep()
    console = Console(record=True, width=200)
    render(data, console, examples=1, by_block=True)
    out = console.export_text()
    for title in ("Step pairs", "Obligations", "Recurring problem steps", "Per block"):
        assert title in out
    discharged = sum(s["verified"] for s in data["steps"] if s["direction"] == "soundness")
    assert f"{discharged:,}" in out


def test_cli_sweep_save_then_report(tmp_path, capsys):
    from src.redesign.cli import main

    out = tmp_path / "sweep.json"
    main(["sweep", "keccak", "2106332", "--save", str(out), "--no-legend"])
    swept = capsys.readouterr().out
    assert main(["report", str(out), "--no-legend"]) == 0
    reported = capsys.readouterr().out
    assert "Step pairs" in reported
    # Re-displaying the saved file gives exactly the same report.
    assert swept == reported
