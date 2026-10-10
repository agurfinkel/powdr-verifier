"""Ground evaluation of lookup tables: is a tuple of constants a row?

This is the only place the sweep uses what a lookup table contains. It is
used for lookups whose arguments are all constants (after the mapping and the
solved equations), so a row check decides them outright. The semantics follow
the SMT encoders in src/bus_interactions/.

PcLookup (bus 2) is deliberately not evaluated: its table is the program, and
the redesign does not trust it (yet). Unknown buses are not evaluated either.
"""

from __future__ import annotations

VARIABLE_RANGE_CHECKER = 3  # [x, bits]: x < 2^bits
BITWISE_LOOKUP = 6  # [x, y, z, op]: op 0 range-checks x, y; op 1 has z = x xor y
# Widest width the range checker is instantiated for
# (openvm_variable_range_checker.py MAX_BITS); wider is not decided here.
MAX_BITS = 25


def is_row(bus: int, args: list[int]) -> bool | None:
    """True or False if ``args`` (integers in [0, p)) is or is not a row; None if not decided."""
    if bus == VARIABLE_RANGE_CHECKER and len(args) == 2:
        x, bits = args
        if bits > MAX_BITS:
            return None
        return x < 2**bits
    if bus == BITWISE_LOOKUP and len(args) == 4:
        x, y, z, op = args
        if op == 0:
            return x < 256 and y < 256 and z == 0
        if op == 1:
            return x < 256 and y < 256 and z == x ^ y
        return False
    return None
