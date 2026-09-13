# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What a kernel's instructions occupy, and the least time that takes.

The emitter counts every statement it lays down by what it occupies
(`Context.record_mix`): floating point by precision, integer and address
arithmetic, the transcendental unit, moves between lanes, loads and stores by
memory space, barriers, branches -- and, beside them, the bytes a lane moves
through each space (`Context.record_bytes`).  All of it is derived from the
program, not predicted: the trip counts are constants and the spaces are
known.  What is not known is how many machine instructions one statement
becomes; `INSTRUCTIONS_PER_STATEMENT` is that, per category and family, and is
fitted by `tools/calibrate_mix.py` against what nvcc and hipcc emit.

Each pipe of an SM or CU gets through its instructions at a peak rate, and a
kernel takes at least as long as its busiest pipe needs.  That is a *lower
bound*, and it is meant as one: peak rates, perfect overlap, no stalls.  Used
as such it prunes a search correctly -- a candidate whose bound exceeds the
time the best one was measured at cannot be better -- which a point estimate
never could.  The same figures say which pipe binds, which is the answer to
"too many shared-memory instructions" or "too much besides the FMAs".

Rates are per SM (NVIDIA) or CU (AMD, Intel Xe-core) and clock, in warp or
wave instructions, or bytes for the bandwidth rows: peak figures from the
vendors' documentation, the optimistic end, so that the bound stays one.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

#: Machine instructions per emitted statement, by family and category: the
#: ratio of the totals over the case set (`tools/calibrate_mix.py`, nvcc and
#: hipcc, 71 kernels, 2026-09-14; sm_120 stands for NVIDIA, gfx942 for CDNA,
#: gfx1150 for RDNA).  A statement may fold into its neighbor -- two FMAs
#: into an FFMA2, four shared loads into one LDS.128 -- or become several: a
#: reduction is a ladder of shuffles, a branch brings its convergence
#: barrier.  Missing categories count 1.0; Intel is not fitted.
INSTRUCTIONS_PER_STATEMENT: Dict[str, Dict[str, float]] = {
    'nvidia': {'fp': 0.69, 'fp64': 0.60, 'int': 0.90, 'sfu': 0.94,
               'xlane': 18.5, 'global.load': 1.14, 'global.store': 1.05,
               'shared.load': 0.35, 'shared.store': 1.50,
               'async.copy': 4.56, 'barrier': 1.49, 'sync': 2.94,
               'branch': 6.57},
    'gfx9': {'fp': 3.73, 'fp64': 1.30, 'int': 1.10, 'sfu': 1.14,
             'xlane': 2.28, 'matrix': 1.0, 'global.load': 0.97,
             'global.store': 1.0, 'shared.load': 0.41, 'shared.store': 0.96,
             'barrier': 0.08, 'branch': 2.26},
    'gfx1': {'fp': 1.12, 'fp64': 0.65, 'int': 1.28, 'sfu': 1.14,
             'xlane': 1.51, 'global.load': 1.01, 'global.store': 1.0,
             'shared.load': 0.69, 'shared.store': 0.91, 'barrier': 0.08,
             'branch': 3.14},
    'intel': {},
}

#: The same, at the 10th percentile over the kernels rather than as a ratio
#: of totals: fewer instructions than nine kernels in ten compiled to, so a
#: bound taken with these stays below the time for them (`bound(...,
#: conservative=True)`), where the totals serve to rank.
INSTRUCTIONS_PER_STATEMENT_P10: Dict[str, Dict[str, float]] = {
    'nvidia': {'fp': 0.50, 'fp64': 0.50, 'int': 0.33, 'sfu': 0.0,
               'xlane': 8.0, 'global.load': 1.02, 'global.store': 1.0,
               'shared.load': 0.26, 'shared.store': 1.0, 'async.copy': 1.0,
               'barrier': 1.0, 'sync': 2.0, 'branch': 3.0},
    'gfx9': {'fp': 0.0, 'fp64': 1.0, 'int': 0.64, 'sfu': 0.0, 'xlane': 2.0,
             'matrix': 1.0, 'global.load': 1.0, 'global.store': 1.0,
             'shared.load': 0.0, 'shared.store': 0.75, 'barrier': 0.0,
             'branch': 1.71},
    'gfx1': {'fp': 0.50, 'fp64': 0.50, 'int': 0.55, 'sfu': 0.0,
             'xlane': 1.0, 'global.load': 1.01, 'global.store': 1.0,
             'shared.load': 0.0, 'shared.store': 0.50, 'barrier': 0.0,
             'branch': 1.78},
    'intel': {},
}

#: Every category the emitter counts under (`pir.emit._mix_category`).
CATEGORIES = ('fp', 'fp64', 'int', 'sfu', 'xlane', 'matrix',
              'global.load', 'global.store', 'global.prefetch',
              'constant.load', 'shared.load', 'shared.store',
              'local.load', 'local.store', 'async.copy',
              'barrier', 'sync', 'branch', 'other')

_MEMORY_INSTRUCTIONS = ('global.load', 'global.store', 'global.prefetch',
                        'constant.load', 'local.load', 'local.store',
                        'async.copy')
_LDS_INSTRUCTIONS = ('shared.load', 'shared.store')


@dataclass(frozen=True)
class Resource:
    """One pipe or path: what it gets through per clock, and what uses it
    (category -> instructions of this resource per instruction)."""
    rate: float
    uses: Tuple[Tuple[str, float], ...] = ()
    #: memory keys (`Context.record_bytes`) this path carries, if it is a
    #: bandwidth rather than an issue rate
    moves: Tuple[str, ...] = ()


def _uses(*categories, weight=1.0):
    return tuple((c, weight) for c in categories)


def _family(hw) -> str:
    if hw.vendor == 'amd':
        return 'gfx9' if str(hw.model).startswith('gfx9') else 'gfx1'
    return hw.vendor


def _sm(hw) -> int:
    model = str(hw.model)
    try:
        return int(model[3:]) if model.startswith('sm_') else 0
    except ValueError:
        return 0


def resources(hw) -> Dict[str, Resource]:
    """The pipes of one SM or CU of `hw`, with peak rates."""
    family = _family(hw)
    if family == 'nvidia':
        sm = _sm(hw)
        # FP32 lanes per SM: 64 on sm_70/75/80, 128 from sm_86 (and sm_90/100)
        fp32 = 2.0 if sm in (70, 72, 75, 80) else 4.0
        # FP64: half of 64 on the data-center parts, 1/64 on the others
        fp64 = 1.0 if sm in (70, 80, 90, 100) else 1.0 / 16
        # INT32: 64 lanes, unified with FP32 (128) from Blackwell on
        int32 = 4.0 if sm >= 100 else 2.0
        return {
            'issue': Resource(4.0, _uses(*CATEGORIES)),
            'fp32': Resource(fp32, _uses('fp')),
            'fp64': Resource(fp64, _uses('fp64')),
            'int': Resource(int32, _uses('int')),
            'sfu': Resource(0.5, _uses('sfu')),
            # the MIO path: shared memory, shuffles, and the LSU's address
            # issue for every other space
            'lsu': Resource(1.0, _uses(*_LDS_INSTRUCTIONS, *_MEMORY_INSTRUCTIONS,
                                       'xlane')),
            'shared_bytes': Resource(128.0, moves=('shared.read',
                                                   'shared.write')),
            'l1_bytes': Resource(128.0, moves=('global.read', 'global.write',
                                               'local.read', 'local.write',
                                               'constant.read')),
        }
    if family == 'gfx9':
        # CDNA: four SIMD16 per CU, a wave64 VALU instruction every clock
        # across them; FP64 FMA at the FP32 rate on gfx90a and later
        return {
            'valu': Resource(1.0, _uses('fp', 'fp64', 'int', 'xlane')
                             + _uses('sfu', weight=4.0)),
            'lds': Resource(0.5, _uses(*_LDS_INSTRUCTIONS)),
            'vmem': Resource(0.25, _uses(*_MEMORY_INSTRUCTIONS)),
            'shared_bytes': Resource(128.0, moves=('shared.read',
                                                   'shared.write')),
            'l1_bytes': Resource(64.0, moves=('global.read', 'global.write',
                                              'local.read', 'local.write',
                                              'constant.read')),
        }
    if family == 'gfx1':
        # RDNA: two SIMD32 per CU, a wave32 VALU instruction each per clock
        # (twice that for FP32 pairs under VOPD); FP64 at 1/16 of it
        return {
            'valu': Resource(4.0, _uses('fp') + _uses('int', 'xlane', weight=2.0)
                             + _uses('sfu', weight=8.0)
                             + _uses('fp64', weight=32.0)),
            'lds': Resource(1.0, _uses(*_LDS_INSTRUCTIONS)),
            'vmem': Resource(0.5, _uses(*_MEMORY_INSTRUCTIONS)),
            'shared_bytes': Resource(128.0, moves=('shared.read',
                                                   'shared.write')),
            'l1_bytes': Resource(128.0, moves=('global.read', 'global.write',
                                               'local.read', 'local.write',
                                               'constant.read')),
        }
    # Intel Xe-HPC, per Xe-core: eight vector engines, SIMD16 instructions
    return {
        'valu': Resource(8.0, _uses('fp', 'fp64', 'int', 'xlane')
                         + _uses('sfu', weight=4.0)),
        'send': Resource(2.0, _uses(*_LDS_INSTRUCTIONS, *_MEMORY_INSTRUCTIONS)),
        'shared_bytes': Resource(128.0, moves=('shared.read', 'shared.write')),
        'l1_bytes': Resource(128.0, moves=('global.read', 'global.write',
                                           'local.read', 'local.write',
                                           'constant.read')),
    }


@dataclass(frozen=True)
class Bound:
    """Clocks one SM or CU needs per element at least, and which pipe."""
    cycles: float
    binding: str
    per_resource: Dict[str, float] = field(default_factory=dict)

    def share(self, resource: str) -> float:
        """How busy `resource` is when the binding pipe is saturated."""
        return self.per_resource.get(resource, 0.0) / self.cycles if self.cycles else 0.0


def instructions_per_element(issue_mix: Dict[str, list], hw, lanes: int,
                             conservative: bool = False) -> Dict[str, float]:
    """Warp (wave) instructions per element, by category.

    The emitter counts per lane; a warp runs a statement once for all its
    lanes, which serve `wave / lanes` elements when a multiplication is
    narrower than the wave -- the same conversion `tuning.static_score` makes
    for its issue figure.
    """
    family = _family(hw)
    wave = max(1, getattr(hw, 'vec_unit_length', 32))
    table = (INSTRUCTIONS_PER_STATEMENT_P10 if conservative
             else INSTRUCTIONS_PER_STATEMENT)
    scale = table.get(family, {})
    per_warp = lanes / wave
    return {category: issued * scale.get(category, 1.0) * per_warp
            for category, (issued, _copies) in issue_mix.items()}


def stream_bytes(descr_list) -> float:
    """Bytes one element has to bring through DRAM at least: its own tensors
    read and written once each, the union of what the descriptors touch
    (`analysis.cost`).  A batch-invariant operator is not in it -- it is read
    once for the whole batch, and from a cache after that.  Not the loads the
    emitter counted, which include every reuse, and would make the bound one
    only by accident."""
    from tensorforge.analysis.cost import list_cost
    return list_cost(descr_list, batch=2).bytes - list_cost(descr_list,
                                                            batch=1).bytes


def bound(issue_mix: Optional[dict], memory_bytes: Optional[dict], hw,
          lanes: int, active_lanes: Optional[int] = None,
          conservative: bool = False, stream: Optional[float] = None,
          dram_bytes_per_clock: Optional[float] = None) -> Optional[Bound]:
    """The least clocks per element on one SM of `hw`, or None where the
    build counted nothing.

    `conservative` takes the instructions per statement at the tenth
    percentile over the calibration kernels rather than as a ratio of totals:
    the bound to prune a search with, where the default is the one to rank
    by.  `stream` (`stream_bytes`) against `dram_bytes_per_clock` -- the
    device's bandwidth over its SMs and clock, a property of the part and not
    of the architecture -- adds DRAM as one more pipe."""
    if not issue_mix:
        return None
    counts = instructions_per_element(issue_mix, hw, lanes, conservative)
    wave = max(1, getattr(hw, 'vec_unit_length', 32))
    # Per lane, times the lanes that access; a broadcast (`.bcast`) is one
    # transaction per warp, which serves `wave / lanes` elements.  Every key
    # also counts under its space and direction, so a resource names only
    # those (`global.read` carries `.bcast` and `.const` with it).
    moved: Dict[str, float] = {}
    for key, nbytes in (memory_bytes or {}).items():
        per_element = (nbytes * lanes / wave if key.endswith('.bcast')
                       else nbytes * (active_lanes or lanes))
        base = '.'.join(key.split('.')[:2])
        moved[base] = moved.get(base, 0) + per_element
        if key != base:
            moved[key] = moved.get(key, 0) + per_element
    per = {}
    for name, res in resources(hw).items():
        if res.moves:
            load = sum(moved.get(key, 0) for key in res.moves)
        else:
            load = sum(counts.get(category, 0.0) * weight
                       for category, weight in res.uses)
        if load:
            per[name] = load / res.rate
    if stream and dram_bytes_per_clock:
        per['dram'] = stream / dram_bytes_per_clock
    if not per:
        return None
    binding = max(per, key=per.get)
    return Bound(per[binding], binding, per)


def of(generator, conservative: bool = False,
       dram_bytes_per_clock: Optional[float] = None) -> Optional[Bound]:
    """`bound` for a generated kernel; with the device's DRAM bandwidth per
    SM and clock, DRAM included."""
    hw = generator._context.get_vm().get_hw_descr()
    stream = (stream_bytes(generator._given) if dram_bytes_per_clock
              else None)
    return bound(generator.issue_mix, generator.memory_bytes, hw,
                 max(1, generator._num_threads or 1),
                 generator._num_active_threads or None, conservative,
                 stream, dram_bytes_per_clock)


def prunable(least: float, incumbent: float) -> bool:
    """Whether a candidate whose least time is `least` can be dropped once
    another has been measured at `incumbent`: exactly when even its best case
    is slower.  Correct only with a bound that is one -- `conservative`, and
    in the same units as the measurement."""
    return least > incumbent


def persistent_efficiency(batch: int, mults_per_block: int,
                          blocks_per_sm: int, sms: int) -> float:
    """The share of the last round a persistent kernel spends busy.

    Every block walks the batch in steps of the whole grid, so the grid runs
    `ceil(batch / (blocks * mults))` rounds and the last is as partial as the
    remainder: the wave quantization of a grid-stride loop.
    """
    slots = max(1, mults_per_block) * max(1, blocks_per_sm) * max(1, sms)
    rounds = math.ceil(batch / slots) if batch else 0
    return batch / (rounds * slots) if rounds else 1.0


def least_seconds(b: Bound, batch: int, sms: int, clock_ghz: float,
                  efficiency: float = 1.0) -> float:
    """The least time `batch` elements take on `sms` SMs at `clock_ghz`: the
    per-SM bound, spread over the SMs, stretched by the rounds' quantization."""
    return b.cycles * batch / (max(1, sms) * clock_ghz * 1e9 * max(efficiency,
                                                                   1e-9))
