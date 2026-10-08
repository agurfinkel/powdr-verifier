"""Builds the PolySolver for one direction. The only place the sweep makes one."""

from __future__ import annotations

from src.polysolver import PolySolver, VarTable

from .circuit import Circuit, byte_ranges


def reference_solver(
    table: VarTable, ref: Circuit, use_ranges: bool = True
) -> PolySolver:
    """Premises and ranges from the reference side only (the polarity rule)."""
    s = PolySolver(table, declared=ref.columns)
    for i, c in enumerate(ref.constraints):
        s.add_premise(c, tag=f"{ref.label}:c{i}")
    if use_ranges:
        for x, lo, hi in byte_ranges(ref):
            s.add_range(x, lo, hi, tag=f"{ref.label}:bytes:{x}")
    return s
