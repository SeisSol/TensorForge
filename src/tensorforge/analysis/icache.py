# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""How much instruction cache a kernel's code takes, and what does not fit.

The batch loop runs its body once per element and keeps doing so for the whole
launch, so the body is what has to stay resident.  Below the cache's capacity
its size costs next to nothing; past it every iteration fetches the difference
again, from a level of the hierarchy shared with everything else the SM or CU
is doing.  So this is a cliff, not a slope, and it is used as one: the tuning
scorer ranks by the excess right after the register file (`static_score`).

The emitter counts units, one a statement per copy the compiler lays down
(`pir.emit`, `Context.record_code`).  `INSTRUCTIONS_PER_UNIT` turns them into
machine instructions -- a statement may be several (an address and a load) or
share one (a folded offset) -- and the target's `instruction_bytes` into bytes.
Both factors are fitted rather than asserted: `tools/calibrate_icache.py`
counts what nvcc and hipcc actually emit for the case set and reports the fit
and its spread.
"""

from typing import Optional

#: Machine instructions per emitter unit, by architecture family; fitted by
#: `tools/calibrate_icache.py` against nvcc (SASS) and hipcc (ISA) over the 71
#: kernels of the case set, 2026-09-13, through the origin:
#:
#:   sm_120   0.43   median +8 %,  10/90 % -40 / +89 %
#:   gfx942   0.87   median +10 %, 10/90 % -30 / +71 %   (CDNA: `gfx9`)
#:   gfx1150  0.69   median +17 %, 10/90 %   0 / +82 %   (RDNA: `gfx1`)
#:
#: NVIDIA folds the most -- an offset into the load, a predicate into the
#: instruction -- and the largest kernels decide the fit, as they should for a
#: cliff.  The far tail is small kernels calling into the math library: `sin`,
#: `pow` and f128 are one unit apiece and twenty to forty instructions, so they
#: are counted short.  Intel is not fitted.
INSTRUCTIONS_PER_UNIT = {'nvidia': 0.43, 'gfx9': 0.87, 'gfx1': 0.69,
                         'intel': 1.0}


def _instructions_per_unit(hw) -> float:
    if hw.vendor == 'amd':
        family = 'gfx9' if str(hw.model).startswith('gfx9') else 'gfx1'
        return INSTRUCTIONS_PER_UNIT[family]
    return INSTRUCTIONS_PER_UNIT.get(hw.vendor, 1.0)


class ICacheBudgetWarning(UserWarning):
    """A kernel's code is larger than the target's instruction cache."""


def code_bytes(code_units: Optional[int], hw) -> Optional[int]:
    """The code a kernel's units become on `hw`, or None where either is
    unknown."""
    if code_units is None:
        return None
    return int(code_units * _instructions_per_unit(hw) * hw.instruction_bytes)


def icache_excess(code_units: Optional[int], hw) -> int:
    """Bytes of code past the instruction cache; 0 where it fits, or where the
    target states no capacity (`hw_descr.icache_size`) -- unknown is not
    over."""
    size = code_bytes(code_units, hw)
    capacity = getattr(hw, 'icache_size', None)
    if size is None or not capacity:
        return 0
    return max(0, size - capacity)


def hot_set(profile: Optional[dict], share: float = 0.9) -> Optional[tuple]:
    """`(code units carrying `share` of the executions, units in total)`.

    The instruction cache holds code; a kernel spends its time in statements.
    Those are the two numbers `Context.record_hot` keeps apart, and the
    distance between them is what says whether a body over the capacity is a
    problem: `elastic-o6s:derivative` lays down 195 742 B against 131 072 B of
    cache, and if nine tenths of what it runs sits in a tenth of that code,
    the fetch it pays for is the tenth and not the whole.

    None where nothing was recorded.  The units are the emitter's own, as
    `code_bytes` takes them.
    """
    if not profile:
        return None
    # Most-run statements first: the cheapest way to cover the executions.
    entries = sorted(((issued, copies, count)
                      for (issued, copies), count in profile.items()),
                     key=lambda e: -e[0])
    total_issues = sum(issued * copies * count for issued, copies, count in entries)
    total_units = sum(copies * count for _, copies, count in entries)
    if not total_issues:
        return 0, total_units
    seen, units = 0.0, 0
    for issued, copies, count in entries:
        if seen >= share * total_issues:
            break
        seen += issued * copies * count
        units += copies * count
    return units, total_units
