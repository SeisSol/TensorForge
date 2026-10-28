# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What a kernel's instructions occupy, and the least time that takes.

The emitter counts every statement it lays down by what it occupies
(`BuildMetrics.record_mix`): floating point by precision, integer and address
arithmetic, the transcendental unit, moves between lanes, loads and stores by
memory space, barriers, branches -- and, beside them, the bytes a lane moves
through each space (`BuildMetrics.record_bytes`).  All of it is derived from
the program, not predicted: the trip counts are constants and the spaces are
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
               'xlane': 18.5, 'xlane.wide': 18.5, 'global.load': 1.14,
               'global.store': 1.05,
               'shared.load': 0.35, 'shared.store': 1.50,
               'async.copy': 4.56, 'barrier': 1.49, 'sync': 2.94,
               'branch': 6.57},
    'gfx9': {'fp': 3.73, 'fp64': 1.30, 'int': 1.10, 'sfu': 1.14,
             'xlane': 2.28, 'xlane.wide': 1.0, 'matrix': 1.0,
             'global.load': 0.97, 'global.store': 1.0, 'shared.load': 0.41,
             'shared.store': 0.96, 'barrier': 0.08, 'branch': 2.26},
    'gfx1': {'fp': 1.12, 'fp64': 0.65, 'int': 1.28, 'sfu': 1.14,
             'xlane': 1.51, 'xlane.wide': 1.51, 'global.load': 1.01,
             'global.store': 1.0, 'shared.load': 0.69, 'shared.store': 0.91,
             'barrier': 0.08, 'branch': 3.14},
    # PVC, 81 kernels of the case set (`tools/calibrate_mix.py --arch pvc`,
    # IGC's shader dump, 2026-09-18).  `xlane` is the entry worth naming: 16938
    # cross-lane statements emitted and *no* instruction compiled, because a
    # broadcast of one lane's element is a region on the reading instruction
    # there and not an operation (`Target.folds_broadcast`).  `fp64` is the
    # ratio of totals like the rest and its median is 0.69: a handful of
    # kernels carry the difference, so a search ranking by it should know that
    # this row is the least settled of them.
    'intel': {'fp': 0.65, 'fp64': 6.38, 'int': 1.79, 'sfu': 1.31, 'xlane': 0.0,
              'xlane.wide': 0.0, 'global.load': 1.21, 'global.store': 0.99,
              'shared.load': 0.55, 'shared.store': 1.0, 'barrier': 5.13,
              'branch': 8.55},
}

#: The same, at the 10th percentile over the kernels rather than as a ratio
#: of totals: fewer instructions than nine kernels in ten compiled to, so a
#: bound taken with these stays below the time for them (`bound(...,
#: conservative=True)`), where the totals serve to rank.
INSTRUCTIONS_PER_STATEMENT_P10: Dict[str, Dict[str, float]] = {
    'nvidia': {'fp': 0.50, 'fp64': 0.50, 'int': 0.33, 'sfu': 0.0,
               'xlane': 8.0, 'xlane.wide': 8.0, 'global.load': 1.02,
               'global.store': 1.0,
               'shared.load': 0.26, 'shared.store': 1.0, 'async.copy': 1.0,
               'barrier': 1.0, 'sync': 2.0, 'branch': 3.0},
    'gfx9': {'fp': 0.0, 'fp64': 1.0, 'int': 0.64, 'sfu': 0.0, 'xlane': 2.0,
             'xlane.wide': 1.0, 'matrix': 1.0, 'global.load': 1.0,
             'global.store': 1.0,
             'shared.load': 0.0, 'shared.store': 0.75, 'barrier': 0.0,
             'branch': 1.71},
    'gfx1': {'fp': 0.50, 'fp64': 0.50, 'int': 0.55, 'sfu': 0.0,
             'xlane': 1.0, 'xlane.wide': 1.0, 'global.load': 1.01,
             'global.store': 1.0,
             'shared.load': 0.0, 'shared.store': 0.50, 'barrier': 0.0,
             'branch': 1.78},
    'intel': {'fp': 0.41, 'fp64': 0.62, 'int': 0.56, 'sfu': 0.0, 'xlane': 0.0,
              'xlane.wide': 0.0, 'global.load': 1.02, 'global.store': 1.0, 'shared.load': 0.26,
              'shared.store': 1.0, 'barrier': 2.0, 'branch': 1.67},
}

#: Clocks one MFMA holds its SIMD for, by the shape its builtin names, measured
#: on gfx942 (MI300A, 2026-10-27; `~/tf-probe/mi300a/coissue.hip`: four
#: independent accumulators, four waves a SIMD).  The 64-lane entries that
#: are not listed were not measured and take their nominal passes.  The
#: multi-block BF16 shapes run at half the rate of the single-block ones --
#: 109 to 119 MACs a clock against 234 to 244 -- which the catalog's names do
#: not say.
MFMA_CLOCKS: Dict[str, float] = {
    'f32_4x4x1f32': 10.0, 'f32_16x16x1f32': 41.0, 'f32_32x32x1f32': 64.0,
    'f32_16x16x4f32': 33.0, 'f32_32x32x2f32': 64.0,
    'f32_4x4x4bf16_1k': 9.4, 'f32_16x16x4bf16_1k': 34.5,
    'f32_32x32x4bf16_1k': 70.0, 'f32_16x16x16bf16_1k': 16.8,
    'f32_32x32x8bf16_1k': 35.0,
    'f64_4x4x4f64': 16.0, 'f64_16x16x4f64': 32.0,
}

#: Every category the emitter counts under (`pir.emit._mix_category`).
CATEGORIES = ('fp', 'fp64', 'int', 'sfu', 'xlane', 'xlane.wide', 'matrix',
              'global.load', 'global.store', 'global.prefetch',
              'constant.load', 'shared.load', 'shared.store',
              'local.load', 'local.store', 'async.copy',
              'barrier', 'sync', 'branch', 'other')
# ... and each AMD matrix instruction by its shape (`MFMA_CLOCKS`)
CATEGORIES += tuple(f'mfma.{shape}' for shape in MFMA_CLOCKS)

_MEMORY_INSTRUCTIONS = ('global.load', 'global.store', 'global.prefetch',
                        'constant.load', 'local.load', 'local.store',
                        'async.copy')
_LDS_INSTRUCTIONS = ('shared.load', 'shared.store')


#: Clocks of a SIMD one wave64 VALU instruction takes: the unit the `valu`
#: resource of CDNA counts in.
_VALU_CLOCKS = 4.0


def mfma_slots(shape: str) -> float:
    """VALU issue slots one MFMA of `shape` costs on CDNA.

    The two do not overlap on gfx942: an F32 MFMA chain beside a dependent F32
    FMA chain took 26.4 clocks a step against 10.9 and 9.4 alone, and four
    independent MFMAs beside four packed FMAs 90 against 40 and 22.  So an
    MFMA is priced on the pipe the VALU uses, at the clocks it holds the SIMD.
    An unknown shape counts as the 4x4x1 one.
    """
    return MFMA_CLOCKS.get(shape, MFMA_CLOCKS['f32_4x4x1f32']) / _VALU_CLOCKS


def _issue_slots(counts: Dict[str, float]) -> Dict[str, float]:
    """`counts` with every `mfma.<shape>` folded into `mfma`, in VALU issue
    slots (`mfma_slots`); the rest as it is."""
    out: Dict[str, float] = {}
    for category, n in counts.items():
        if category.startswith('mfma.'):
            out['mfma'] = out.get('mfma', 0.0) + n * mfma_slots(category[5:])
        else:
            out[category] = out.get(category, 0.0) + n
    return out


@dataclass(frozen=True)
class Resource:
    """One pipe or path: what it gets through per clock, and what uses it
    (category -> instructions of this resource per instruction)."""
    rate: float
    uses: Tuple[Tuple[str, float], ...] = ()
    #: memory keys (`BuildMetrics.record_bytes`) this path carries, if it is
    #: a bandwidth rather than an issue rate
    moves: Tuple[str, ...] = ()


def _uses(*categories, weight=1.0):
    return tuple((c, weight) for c in categories)


def _family(hw) -> str:
    return hw.family


def _sm(hw) -> int:
    return hw.sm_level() or 0


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
                                       'xlane', 'xlane.wide')),
            'shared_bytes': Resource(128.0, moves=('shared.read',
                                                   'shared.write')),
            'l1_bytes': Resource(128.0, moves=('global.read', 'global.write',
                                               'local.read', 'local.write',
                                               'constant.read')),
        }
    if family == 'gfx9':
        # CDNA: four SIMD16 per CU, a wave64 VALU instruction every clock
        # across them; FP64 FMA at the FP32 rate on gfx90a and later.  An
        # MFMA holds the SIMD the VALU issues to (`mfma_slots`), and a move
        # across 16-lane rows is a DS instruction (`ds_swizzle`,
        # `ds_bpermute`), so it waits for the LDS pipe and not for the VALU.
        return {
            'valu': Resource(1.0, _uses('fp', 'fp64', 'int', 'xlane', 'mfma')
                             + _uses('sfu', weight=4.0)),
            'lds': Resource(0.5, _uses(*_LDS_INSTRUCTIONS, 'xlane.wide')),
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
            'valu': Resource(4.0, _uses('fp') + _uses('int', 'xlane', 'xlane.wide',
                                                      weight=2.0)
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
        'valu': Resource(8.0, _uses('fp', 'fp64', 'int', 'xlane', 'xlane.wide')
                         + _uses('sfu', weight=4.0)),
        'send': Resource(2.0, _uses(*_LDS_INSTRUCTIONS, *_MEMORY_INSTRUCTIONS)),
        'shared_bytes': Resource(128.0, moves=('shared.read', 'shared.write')),
        'l1_bytes': Resource(128.0, moves=('global.read', 'global.write',
                                           'local.read', 'local.write',
                                           'constant.read')),
    }


def latency_waves(hw) -> Optional[float]:
    """Waves (warps) an SM or CU needs resident for its pipes to reach the
    bound, or None where that was never measured.

    NVIDIA: one warp per scheduler, the rate of the `issue` resource
    (`tuning._shortfall` says how that was measured).  CDNA: one wave per
    SIMD, the same statement for four SIMDs -- a SIMD with no wave issues
    nothing.  Not the two a lone dependent chain needs (an F32 FMA chain at
    one wave a SIMD ran at 14.1 clocks a step against 9.4 at four on gfx942):
    the kernels carry independent accumulators enough to fill their own
    gaps, and asking two of them stretched SeisSol's order-6 derivative and
    volume at 16 lanes past 32, which measured 31 % and 15 % slower.
    """
    family = _family(hw)
    if family == 'nvidia':
        return resources(hw)['issue'].rate
    if family == 'gfx9':
        return 4.0
    return None


def spills_are_a_cliff(hw) -> bool:
    """Whether a candidate that spills at all ranks behind every one that does
    not (`tuning.static_score`), or only by what its spilling costs.

    A cliff on NVIDIA and PVC, where it was measured to be one.  Not on CDNA:
    from gfx90a on the compiler spills into the AGPRs of the unified file
    before it reaches scratch, and what does reach scratch stays in the
    caches.  On MI300A the cliff took SeisSol's order-6 FP64 derivative to 64
    lanes, 44.5 % slower than the 32 that spill 108 bytes, and the
    viscoelastic derivative to 32 against a spilling 16 that is 6.5 % faster;
    over 26 kernels at 16, 32 and 64 lanes, ranking by the penalty alone picks
    the fastest in 18 against 15, at a geometric mean of 1.017 against 1.038
    of the fastest.
    """
    return _family(hw) != 'gfx9'


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


def executed_flops(generator) -> Optional[int]:
    """Floating-point operations the generated kernel performs, per element.

    The third of three figures and the only one taken from the program that
    runs.  `analysis.cost` has the other two -- what the operation needs
    (`nonzero_flops`) and what a dense implementation of it performs
    (`flops`) -- and both are of the *operation*: they count the destination
    box, where a kernel computes whatever its lanes cover.  A destination
    twenty rows wide on thirty-two lanes does a little over half its
    arithmetic on rows nobody reads, and nothing in a descriptor says so.

    The emitter counts per lane (`BuildMetrics.emitted_flops`), so the lanes
    of a multiplication bring it to the element -- and the hardware counter
    to compare it with counts threads too (`ncu`'s
    `smsp__sass_thread_inst_executed_op_*_pred_on`, with an `ffma` worth
    two).

    **An upper bound, tight where the tail is not guarded.**  Against ncu on
    sm_120, elastic order 4, batch 4096 -- the count here over the measured
    one:

    | `gpu_localFluxAll` | 1.000 |
    | `gpu_volume` | 1.067 |
    | `gpu_neighboringFlux[0]` | 1.093 |
    | `gpu_derivative` | 1.141 |
    | `gpu_derivativeTaylorExpansion` | 1.531 |

    Never below one, which is what makes it a bound: everything counted is
    written into the kernel, and what separates it from the measurement is
    arithmetic that reaches fewer lanes than the guard it sits under claims,
    or none.  So an efficiency taken against it is a floor, and a roofline
    plotting it is reporting at least that much arithmetic -- which is why
    `tools/bench` records it beside the other two rather than dividing by it.

    `None` where the build reached a statement whose arithmetic the emitter
    was not told -- a matrix instruction is one statement over a whole tile.
    A figure short by the bulk of the arithmetic is worse than none, because
    a rate divided by it looks like efficiency.
    """
    built = getattr(generator, 'built', None)
    metrics = getattr(built, 'metrics', None) if built is not None else None
    if metrics is None or metrics.emitted_flops is None:
        return None
    if not getattr(metrics, 'emitted_flops_exact', True):
        return None
    lanes = max(1, getattr(built, '_num_threads', 1) or 1)
    # A build that stacks elements counts the whole stack, as everything
    # else here does (`of`), and is divided by it for the same reason.
    stack = max(1, int(getattr(built, '_stack', 1) or 1))
    return int(round(metrics.emitted_flops * lanes / stack))


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


def spilled(issue_mix: Optional[dict], memory_bytes: Optional[dict],
            spill_bytes: float, word_bytes: int = 4):
    """The same two tables with `spill_bytes` of spilling written into them.

    A spilled value is not a property of the kernel the way a shared-memory
    round trip is -- the compiler puts it there, and only where the register
    file ran out -- but it costs what every other access costs: a store, a
    load back, and the bytes of both through the same L1 path.  Counting it
    anywhere else makes it incomparable with what it competes against; a rule
    that ranks "no spill" before every issue figure cannot say that 32 bytes
    of spilling is worth a reduction rolled by two.

    `spill_bytes` is per lane and per element, in the units the compiler
    reports (ptxas's spill stores plus spill loads, IGC's scratch).  One
    instruction per register word each way, because that is how a spill is
    spelled: there is no wide form of it.

    The categories are the ones the emitter already has and never fills
    (`local.load`, `local.store`) -- local memory is where a spill goes, and
    `resources` already routes those through the LSU and the L1 bandwidth.
    """
    if spill_bytes <= 0:
        return issue_mix, memory_bytes
    mix = dict(issue_mix or {})
    words = spill_bytes / max(1, word_bytes) / 2      # half stored, half read
    for category in ('local.store', 'local.load'):
        issued, copies = mix.get(category, [0, 0])
        mix[category] = [issued + words, copies + words]
    moved = dict(memory_bytes or {})
    for key in ('local.write', 'local.read'):
        moved[key] = moved.get(key, 0) + spill_bytes / 2
    return mix, moved


def bound(issue_mix: Optional[dict], memory_bytes: Optional[dict], hw,
          lanes: int, active_lanes: Optional[int] = None,
          conservative: bool = False, stream: Optional[float] = None,
          dram_bytes_per_clock: Optional[float] = None,
          spill_bytes: float = 0.0) -> Optional[Bound]:
    """The least clocks per element on one SM of `hw`, or None where the
    build counted nothing.

    `conservative` takes the instructions per statement at the tenth
    percentile over the calibration kernels rather than as a ratio of totals:
    the bound to prune a search with, where the default is the one to rank
    by.  `stream` (`stream_bytes`) against `dram_bytes_per_clock` -- the
    device's bandwidth over its SMs and clock, a property of the part and not
    of the architecture -- adds DRAM as one more pipe.  `spill_bytes`
    (`spilled`) writes what a compiler said it spilled into the same tables,
    so that it is weighed against the issue count instead of before it."""
    if not issue_mix:
        return None
    if spill_bytes:
        issue_mix, memory_bytes = spilled(issue_mix, memory_bytes, spill_bytes)
    counts = _issue_slots(instructions_per_element(issue_mix, hw, lanes,
                                                   conservative))
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
       dram_bytes_per_clock: Optional[float] = None,
       spill_bytes: float = 0.0) -> Optional[Bound]:
    """`bound` for a generated kernel; with the device's DRAM bandwidth per
    SM and clock, DRAM included, and with what a compiler said it spilled.

    The spilling is the caller's to pass because this module cannot know it:
    `BuildMetrics.peak_pressure` is the emitter's own live set and sits well
    under the file where ptxas reports spilling anyway -- 407 bytes a lane
    against a 1020-byte budget for `elastic-o6s:derivative`, which the compiler
    builds at 255 registers with spills.  What the compiler reports is the
    figure with the addressing and the schedule in it, and only
    `tuning.CompiledScore` has one.

    Per element of the batch, whatever a multiplication computes: a build
    that stacks elements (`Options.stack_elements`) counts every figure for
    the whole stack, the spilling a compiler reports for it included, and is
    divided by the stack here -- so that a stacked build and a plain one are
    compared in the same unit.
    """
    built = generator.built
    hw = built._context.target.hw
    stream = (stream_bytes(built._given) if dram_bytes_per_clock
              else None)
    metrics = built.metrics
    b = bound(metrics.issue_mix, metrics.memory_bytes, hw,
              max(1, built._num_threads or 1),
              built._num_active_threads or None, conservative,
              stream, dram_bytes_per_clock, spill_bytes)
    stack = max(1, int(getattr(built, '_stack', 1) or 1))
    if b is None or stack == 1:
        return b
    return Bound(b.cycles / stack, b.binding,
                 {name: v / stack for name, v in b.per_resource.items()})


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
