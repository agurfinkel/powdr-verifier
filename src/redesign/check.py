"""Run the mapping and the sweep for one Before/After step, in either direction."""

from __future__ import annotations

from pathlib import Path

from src.polysolver import VarTable

from .adapter import reference_solver
from .circuit import Circuit, load_step, load_substitutions
from .mapping import completeness_mapping, soundness_mapping
from .sweep import Report, sweep

DIRECTIONS = ("completeness", "soundness")


def check_step(
    before: Circuit,
    after: Circuit,
    subs: dict,
    directions=DIRECTIONS,
    use_ranges: bool = True,
) -> list[Report]:
    """Completeness: Before is the reference. Soundness: After is."""
    table = VarTable()  # one per step, shared by both directions
    reports = []
    if "completeness" in directions:
        solver = reference_solver(table, before, use_ranges)
        m = completeness_mapping(before, after, subs)
        reports.append(sweep("completeness", before, after, m, solver))
    if "soundness" in directions:
        solver = reference_solver(table, after, use_ranges)
        m = soundness_mapping(after, before, subs, solver) # <- Polysolver used to find missing witnesses
        reports.append(sweep("soundness", after, before, m, solver))
    return reports


def check(
    group: str,
    block: str,
    step_from: str,
    step_to: str,
    directions=DIRECTIONS,
    use_ranges: bool = True,
    root: Path | None = None,
) -> list[Report]:
    before = load_step(group, block, step_from, root)
    after = load_step(group, block, step_to, root)
    subs = load_substitutions(group, block, root)
    return check_step(before, after, subs, directions, use_ranges)
