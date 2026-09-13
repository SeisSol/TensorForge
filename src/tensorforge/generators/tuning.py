# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The configurations a kernel can be generated in, and how to choose one.

`lanes.search` chooses the lane count and nothing else, and the measurements
say that is not the decision that stands alone.  `local_flux` on GB200
(package 4) was fastest at eight lanes up to b = 56 -- but with the four faces
merged, the operators prepared and, at 56, the reduction rolled by 28 -- and at
16 lanes of width 2 from b = 80 on.  Eight lanes without the rest is another
kernel, and the width is not in the lane candidates at all.  So the space is
the product: the lane geometry and the options that change what a geometry
compiles to.

Three parts, kept apart because they answer to different things:

* `space` says which configurations exist for a descriptor list -- every knob
  with the values that can make a difference here, and none that cannot (no
  merging where nothing repeats, no preparing where no operand is
  batch-constant, no matrix path off NVIDIA).
* A *scorer* says how good one built configuration is.  `static_score` reads
  the build alone; `CompiledScore` asks a compiler for registers and spills;
  `MeasuredScore` hands the build to a caller who times it.  Lower is better,
  and `None` is "could not score" rather than "scored zero".
* A *strategy* walks the space: `exhaustive` for a small one, `coordinate`
  (one knob at a time from the default, until nothing improves) otherwise.

The compiler is the caller's to name.  Nothing here assumes one is installed:
`Toolchain` takes paths, falls back on `TF_NVCC` / `TF_HIPCC` and then on
`PATH`, and a scorer that finds none says so instead of guessing.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import warnings
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from tensorforge.common.basic_types import Addressing
from tensorforge.common.context import Context, Options
from tensorforge.generators import lanes as lane_config
from tensorforge.generators.lanes import LaneConfig


# --------------------------------------------------------------------------- #
# Candidates
# --------------------------------------------------------------------------- #

def backend_of(context: Context) -> str:
    """The backend a context was made for, as `Context` was asked.

    Not `hw.backend`: that is the device's, and `esimd` runs on the `oneapi`
    device -- a context rebuilt from it is SPMD SYCL, so every ESIMD candidate
    was built, compiled and scored as the other lowering.
    """
    from tensorforge.common.vm.lexic import EXPLICIT_SIMD_BACKENDS
    backend = context.get_vm().get_hw_descr().backend
    if getattr(context.get_vm().get_lexic(), 'simd_mode', False):
        return next(k for k, v in EXPLICIT_SIMD_BACKENDS.items() if v == backend)
    return backend


@dataclass(frozen=True)
class Candidate:
    """One configuration: a lane geometry and the options it is built with.

    `lanes` of None is the generator's own deduction.  The options are the
    ones this candidate *sets*, on top of whatever the base context asked for.
    Hashable and plain, so that a tuner can keep a table of them and a result
    can be written down and rebuilt later.
    """
    lanes: Optional[LaneConfig] = None
    options: Tuple[Tuple[str, Any], ...] = ()

    def set(self, name: str, value: Any) -> 'Candidate':
        if name == 'lanes':
            return replace(self, lanes=value)
        opts = dict(self.options)
        opts[name] = value
        return replace(self, options=tuple(sorted(opts.items())))

    def get(self, name: str, default: Any = None) -> Any:
        if name == 'lanes':
            return self.lanes
        return dict(self.options).get(name, default)

    def context(self, base: Context) -> Context:
        """A context for this candidate: the base's target and options, with
        this candidate's on top."""
        hw = base.get_vm().get_hw_descr()
        asked = dict(base._asked_options.asked())
        asked.update(self.options)
        return Context(arch=hw.model, backend=backend_of(base),
                       fp_type=base.fp_type, options=Options(**asked))

    def label(self) -> str:
        geo = ('deduced' if self.lanes is None
               else f'{self.lanes.num_threads}w{self.lanes.lead_width}')
        opts = ','.join(f'{n}={_spell(v)}' for n, v in self.options)
        return f'{geo}' + (f' {opts}' if opts else '')

    def to_dict(self) -> Dict[str, Any]:
        return {'lanes': None if self.lanes is None else
                [self.lanes.num_threads, self.lanes.num_active_threads,
                 self.lanes.lead_width],
                'options': dict(self.options)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Candidate':
        lanes = data.get('lanes')
        return cls(lanes=None if lanes is None else LaneConfig(*lanes),
                   options=tuple(sorted(data.get('options', {}).items())))


def _spell(value: Any) -> str:
    if isinstance(value, bool):
        return '1' if value else '0'
    return str(value)


# --------------------------------------------------------------------------- #
# The space
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Knob:
    """One axis of the space.  `values(candidate)` may depend on the others:
    the prefetch depth of the matrix path is only a question where that path
    is taken."""
    name: str
    values: Callable[[Candidate], Sequence[Any]]


def _flat(descrs):
    return [op for d in descrs for op in d.operations()]


def _tensors(descrs):
    for d in _flat(descrs):
        for op in list(getattr(d, 'ops', [])) + [getattr(d, 'dest', None)]:
            tensor = getattr(op, 'tensor', None)
            if tensor is not None:
                yield tensor


def contraction_lengths(descrs) -> List[int]:
    """The extent of every contracted axis, one per multilinear operand axis
    that the destination does not have (`target` -1)."""
    out = []
    for d in _flat(descrs):
        target = getattr(d, 'target', None)
        if target is None or len(getattr(d, 'ops', ())) < 2:
            continue
        for op, axes in zip(d.ops, target):
            box = getattr(op, 'bbox', None)
            if box is None:
                continue
            for axis, t in enumerate(axes):
                if t == -1 and axis < box.rank():
                    out.append(int(box.size(axis)))
    return out


def _roll_values(descrs) -> List[int]:
    """0, and the largest divisors of the longest reduction at up to 32 and up
    to 8 steps -- the two ends of what rolling buys: a body the instruction
    cache holds, and one whose registers stay small."""
    ks = contraction_lengths(descrs)
    if not ks:
        return [0]
    k = max(ks)
    out = [0]
    for cap in (32, 8):
        divisors = [d for d in range(2, min(cap, k - 1) + 1) if k % d == 0]
        if divisors and divisors[-1] not in out:
            out.append(divisors[-1])
    return out


def _mergeable(descrs, context) -> bool:
    from tensorforge.generators.descriptions import ForDescr
    from tensorforge.generators.rolling import roll
    options = context.get_user_options()
    try:
        rolled = roll(list(descrs), min_count=options.merge_min_count,
                      max_arity=options.merge_max_arity)
    except Exception:
        return False
    return any(isinstance(d, ForDescr) for d in rolled)


def _geometries(descrs, context) -> List[LaneConfig]:
    """The lane candidates, and each power of two of them at width 2 where the
    backend spells a widened lead (CUDA and HIP) and the pair still fits the
    rows -- width 2 at 16 lanes was GB200's fastest from b = 80 on, and no
    lane candidate had it."""
    flat = _flat(descrs)
    out = list(lane_config.candidates(flat, context))
    backend = getattr(context.get_vm().get_lexic(), '_backend', None)
    if backend not in ('cuda', 'hip'):
        return out
    base = lane_config.deduce(flat, context)
    rows = base.num_active_threads or base.num_threads
    for config in list(out):
        t = config.num_threads
        if config.lead_width != 1 or t & (t - 1) or t < lane_config.MIN_LANES:
            continue
        if 2 * t > 2 * max(rows, t) or t >= rows:
            continue
        out.append(LaneConfig(num_threads=t,
                              num_active_threads=base.num_active_threads,
                              lead_width=2))
    return out


def space(descrs, context: Context) -> List[Knob]:
    """The knobs worth turning for this descriptor list on this target.

    Each with the values that can change the kernel here and only those, so
    that a strategy does not spend builds on a switch that does nothing.
    """
    hw = context.get_vm().get_hw_descr()
    geometries = _geometries(descrs, context)
    knobs = [Knob('lanes', lambda c, g=tuple(geometries): g)]
    if _mergeable(descrs, context):
        knobs.append(Knob('merge_variants', lambda c: (False, True)))
    if any(t.addressing == Addressing.NONE for t in _tensors(descrs)):
        knobs.append(Knob('prepare_operands', lambda c: (False, True)))
        knobs.append(Knob('preload_globals', lambda c: (False, True)))
    rolls = _roll_values(descrs)
    if len(rolls) > 1:
        knobs.append(Knob('k_roll', lambda c, r=tuple(rolls): r))
    if hw.vendor == 'nvidia' and contraction_lengths(descrs):
        knobs.append(Knob('tensor_cores', lambda c: (False, True)))
        knobs.append(Knob('mma_prefetch',
                          lambda c: ((1, 2) if c.get('tensor_cores') else (None,))))
    return knobs


def simple_space(descrs, context: Context) -> List[Knob]:
    """The knobs `Options.autotune` turns: only those whose every value is
    safe to ship without the caller knowing.

    Lane counts are the powers of two from the deduced one down to
    `lanes.MIN_LANES` -- the geometries measured on every target; the
    divisors of the extent (5 and 7 lanes for 20 and 35 rows) rank well on
    both scorers and have never been timed.  A width of two only where one
    instruction does two FMAs (`has_packed_fp32_fma`) and the extent is even:
    elsewhere it is two scalar FMAs, and at 35 rows on GB200 it was 70 %
    slower.  Merging where something repeats, and rolling by the largest
    divisor up to 32.  Not `prepare_operands`: the host packs for it.  Not
    `preload_globals` or the matrix path, whose defaults are per vendor and
    not yet measured.
    """
    from tensorforge.common.basic_types import Datatype
    from tensorforge.generators.descriptions import ElementwiseDescr
    flat = _flat(descrs)
    if any(isinstance(d, ElementwiseDescr) for d in flat):
        return []
    hw = context.get_vm().get_hw_descr()
    base = lane_config.deduce(flat, context)
    rows = base.num_active_threads or base.num_threads
    geometries = []
    t = base.num_threads
    while t >= lane_config.MIN_LANES:
        geometries.append(LaneConfig(t, base.num_active_threads, base.lead_width))
        if t & (t - 1):
            break
        t //= 2
    backend = getattr(context.get_vm().get_lexic(), '_backend', None)
    if (hw.has_packed_fp32_fma() and context.fp_type == Datatype.F32
            and backend in ('cuda', 'hip') and rows % 2 == 0
            and base.lead_width == 1):
        # On AMD only where a lane holds one pair per column.  hipcc spilled
        # every gfx942 build that needed two -- 16 lanes at 35 and 56 rows,
        # 32 at 80 and 120: 512 registers and 2 to 8 KB of scratch, where the
        # model saw 200 -- and none that needed one.  ptxas does not: 16 lanes
        # at width two was GB200's fastest at 80 and 120 rows.
        one_pair = hw.vendor == 'amd'
        geometries += [LaneConfig(g.num_threads, base.num_active_threads, 2)
                       for g in list(geometries)
                       if g.num_threads <= rows
                       and (not one_pair or 2 * g.num_threads >= rows)]
    knobs = [Knob('lanes', lambda c, g=tuple(geometries): g)] if len(geometries) > 1 else []
    if _mergeable(descrs, context):
        knobs.append(Knob('merge_variants', lambda c: (False, True)))
    rolls = _roll_values(descrs)[:2]
    if len(rolls) > 1:
        knobs.append(Knob('k_roll', lambda c, r=tuple(rolls): r))
    return knobs


def start(descrs, context: Context) -> Candidate:
    """The configuration the generator would pick unasked: the deduced lanes
    and every option at the base context's value."""
    base = lane_config.deduce(_flat(descrs), context)
    return Candidate(lanes=base)


def enumerate_space(knobs: Sequence[Knob], origin: Candidate) -> Iterable[Candidate]:
    """Every point, conditional knobs included.  For a small space."""
    def expand(i, cand):
        if i == len(knobs):
            yield cand
            return
        for v in knobs[i].values(cand):
            yield from expand(i + 1, cand if v is None else cand.set(knobs[i].name, v))
    yield from expand(0, origin)


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #

@dataclass
class Build:
    """One candidate, generated: the generator on success, the exception
    otherwise."""
    candidate: Candidate
    generator: Any = None
    context: Optional[Context] = None
    error: Optional[BaseException] = None

    @property
    def ok(self) -> bool:
        return self.error is None


def build(descr_factory, base: Context, candidate: Candidate) -> Build:
    """Generate `candidate` from a fresh descriptor list.

    Fresh every time, and not as a convenience: preparing an operand stores
    its order on the `Tensor` itself, so a build at eight lanes leaves an
    eight-lane interleave behind for the next build to inherit.
    """
    from tensorforge.generators.generator import Generator
    ctx = candidate.context(base)
    ctx.measure_pressure = True
    try:
        gen = Generator(descr_factory(), ctx, lanes=candidate.lanes)
        # asked a question, not emitting: nothing it builds reaches a file
        gen._announce_identity = False
        gen.generate()
    except Exception as exc:
        return Build(candidate, context=ctx, error=exc)
    return Build(candidate, generator=gen, context=ctx)


def kernel_source(result: Build) -> str:
    """The kernel as a translation unit a compiler can take on its own."""
    gen, ctx = result.generator, result.context
    headers = ctx.get_vm().get_headers() + gen.get_helper_headers()
    lines = [f'#include {h}' if h.startswith('<') else f'#include "{h}"'
             for h in headers]
    return '\n'.join(lines) + '\n' + gen.get_kernel()


# --------------------------------------------------------------------------- #
# Scorers
# --------------------------------------------------------------------------- #

def _geometry(result: Build) -> Tuple[int, int, int]:
    """`(lanes, wave, resident multiplications per SM from what is exact)`."""
    gen = result.generator
    hw = result.context.get_vm().get_hw_descr()
    mults = gen.launch_config().mults_per_block
    return gen._num_threads, hw.vec_unit_length, (gen.resident_blocks or 0) * mults


def static_score(result: Build):
    """What the build alone says, per multiplication rather than per block.

    Past the register file first (`_over_budget`).  Then multiplications
    resident per SM -- blocks times the multiplications a block holds, since
    eight lanes put four times as many in a block as 32.
    Then warp issue slots per multiplication: the arithmetic written out, times
    the share of a warp one multiplication takes.  Then the modelled register
    footprint, both in granules of sixteen registers (`_GRANULE`).  Last the
    length of the source: where nothing else differs, the smaller kernel --
    merged, on every measurement taken (GB200's winners, sm_120 by 5 to 24 %,
    and fewer spills from hipcc).  Per block and per lane, as `lanes.search`
    ranks, the default 32 lanes came first at every size GB200 measured, and
    it was the slowest at three of five.
    """
    if not result.ok:
        return None
    gen = result.generator
    lanes, wave, resident = _geometry(result)
    blocks = _register_blocks(result)
    if blocks is not None:
        mults = gen.launch_config().mults_per_block
        resident = min(resident, blocks * mults)
    issue = (gen.emitted_work or 0) * lanes / wave
    return (_granule(_over_budget(result)), -resident, issue,
            _granule(gen.peak_pressure or 0), len(gen.get_kernel() or ''))


#: Bytes below which two modelled footprints are the same footprint: sixteen
#: registers.  The model is within about forty registers of what a compiler
#: allocates, so a difference of eleven bytes -- merged and written-out
#: `local_flux` at b = 80, 1359 against 1348 -- is not one the compiler makes;
#: hipcc spilled 6 KB for the written-out one and nothing for the merged.
_GRANULE = 64


def _granule(value) -> int:
    return int(value // _GRANULE)


#: Registers per lane as a function of the modelled bytes, per vendor, fitted
#: over builds that did not spill: `(intercept, slope)` on bytes / 4.
#: NVIDIA: ptxas sm_100a, 52 builds, residuals within 40.  AMD: hipcc gfx942,
#: 29 builds, residuals within 38 -- no intercept worth the name.
_REGISTER_FIT = {'nvidia': (51.0, 1.05), 'amd': (0.0, 1.26)}


def register_estimate(result: Build) -> Optional[float]:
    """Registers per lane the target compiler is expected to allocate."""
    hw = result.context.get_vm().get_hw_descr()
    fit = _REGISTER_FIT.get(hw.vendor)
    peak = result.generator.peak_pressure
    if fit is None or not peak:
        return None
    return fit[0] + fit[1] * peak / 4


def _register_blocks(result: Build) -> Optional[int]:
    """Blocks per CU the register file admits, where that is what decides.

    AMD only.  A CDNA lane has 512 registers, VGPRs and AGPRs together, and a
    SIMD holds as many waves as fit -- eight at 64 registers, one above 256.
    On gfx942 over `local_flux` a third of the geometries sat at one wave per
    SIMD without spilling a byte, which blocks per CU from shared memory and
    threads alone never shows.  NVIDIA keeps what is exact: its fit has an
    intercept that the occupancy would inherit, and the eight lanes that GB200
    ran fastest sit right at the limit.
    """
    hw = result.context.get_vm().get_hw_descr()
    if hw.vendor != 'amd':
        return None
    regs = register_estimate(result)
    if regs is None:
        return None
    file = (hw.max_reg_per_thread or 1024) // 4
    granule = 8
    per_lane = max(granule, -(-int(regs) // granule) * granule)
    waves_per_simd = max(0, min(8, file // per_lane))
    gen = result.generator
    threads = gen._num_threads * gen.launch_config().mults_per_block
    waves_per_block = max(1, -(-threads // hw.vec_unit_length))
    return (4 * waves_per_simd) // waves_per_block


def _over_budget(result: Build) -> float:
    """How far the modelled footprint is past the register file a thread
    has, or 0 where it fits -- an amount and not a verdict, because where
    every candidate is past it (`local_flux` at b = 120 on gfx942, all of
    them spilling) the one past it least is the one that spills least: 880 B
    of scratch at 32 lanes against 11 KB at eight, which a yes/no left to the
    next key to decide the wrong way.

    First, before anything is ranked: a build that spills is slower than any
    difference the other keys can see.  Calibrated against ptxas on sm_100a
    (`local_flux`, 79 builds), registers come out at about 51 + 1.05 times the
    modelled bytes over four, and every build above the 1020 B a thread has
    there spilled -- eight lanes at b = 80 and 120, which GB200 then ran 75 %
    and 148 % behind the default.  Not a spill predictor in the other
    direction: ptxas also spills at 168 registers where it chooses occupancy,
    and nothing in the model says when.  Where the target states no budget,
    there is no guard.
    """
    hw = result.context.get_vm().get_hw_descr()
    budget = getattr(hw, 'max_reg_per_thread', None)
    peak = result.generator.peak_pressure
    if not (budget and peak):
        return 0
    if hw.vendor == 'amd':
        # hipcc allocates about 1.26 registers per modelled four bytes, so the
        # byte budget alone let eight lanes at b = 56 through (2449 B against
        # 2048) that gfx942 spilled 2 KB for.  In bytes, like the rest.
        return max(0.0, 4 * register_estimate(result) - budget)
    return max(0, peak - budget)


@dataclass
class Toolchain:
    """Where the compilers are.  Paths from the caller, then `TF_NVCC` /
    `TF_HIPCC`, then `PATH`; None where none of them has one."""
    nvcc: Optional[str] = None
    hipcc: Optional[str] = None
    #: oneAPI's `icpx`.  It needs the environment `setvars.sh` makes; a path
    #: alone reaches the driver, not the device compiler behind it.
    icpx: Optional[str] = None
    include: Optional[str] = None

    def compiler(self, vendor: str) -> Optional[str]:
        if vendor == 'nvidia':
            return (self.nvcc or os.environ.get('TF_NVCC')
                    or shutil.which('nvcc'))
        if vendor == 'amd':
            return (self.hipcc or os.environ.get('TF_HIPCC')
                    or shutil.which('hipcc'))
        if vendor == 'intel':
            return (self.icpx or os.environ.get('TF_ICPX')
                    or shutil.which('icpx'))
        return None

    def include_dir(self) -> str:
        if self.include:
            return self.include
        import tensorforge
        return os.path.join(os.path.dirname(tensorforge.__file__), 'include')


@dataclass(frozen=True)
class Resources:
    """What a compiler reports for one kernel."""
    registers: Optional[int]
    spill_bytes: int
    #: Blocks the register file admits per SM, where it can be said.
    register_blocks: Optional[int] = None


def parse_ptxas(log: str) -> Optional[Resources]:
    regs = re.search(r'Used (\d+) registers', log)
    spill = re.search(r'(\d+) bytes spill stores, (\d+) bytes spill loads', log)
    if regs is None:
        return None
    stores, loads = (int(spill.group(1)), int(spill.group(2))) if spill else (0, 0)
    return Resources(int(regs.group(1)), stores + loads)


#: IGC's word for a kernel it compiled twice: the first attempt blew the
#: register file and it retries with another strategy.  It says nothing else
#: per kernel -- no register count, and a spill size only where it gives up
#: (`tools/register_usage.py` reads the same stream).
_IGC_RETRY = re.compile(r'\[RetryManager\]\s+Start recompilation', re.I)
#: `Spill memory used = 33088 bytes for kernel ...` is what IGC 2026 prints,
#: for ESIMD and SPMD alike; the other two are older spellings.
_IGC_SPILL = re.compile(
    r"(?:kernel|Kernel)\s+.*?\bspill(?:s|ed)?\b.*?(?P<value>\d+)\s*bytes"
    r"|spill(?:ed)?\s+(?P<value2>\d+)\s*bytes"
    r"|spill memory used\s*=\s*(?P<value3>\d+)\s*bytes", re.I)


def parse_igc(log: str) -> Resources:
    """An Intel AOT build: whether it spilled, and how much where IGC says.

    Always a report, never None: IGC is silent about a kernel that fits, so
    silence is the answer "no spill" and not a failure to parse.  A retry
    without a size counts as one byte -- spilled, amount unknown -- so that it
    ranks behind every build that did not.
    """
    spill = 0
    for m in _IGC_SPILL.finditer(log):
        spill = max(spill, int(m.group('value') or m.group('value2')
                               or m.group('value3')))
    if not spill and _IGC_RETRY.search(log):
        spill = 1
    return Resources(None, spill)


def parse_amdgpu(log: str) -> Optional[Resources]:
    """`-Rpass-analysis=kernel-resource-usage`.  The occupancy it reports is
    waves per SIMD; kept as the register figure's blocks-equivalent."""
    vgprs = re.search(r'VGPRs: (\d+)', log)
    agprs = re.search(r'AGPRs: (\d+)', log)
    scratch = re.search(r'ScratchSize \[bytes/lane\]: (\d+)', log)
    occupancy = re.search(r'Occupancy \[waves/SIMD\]: (\d+)', log)
    if vgprs is None:
        return None
    return Resources(int(vgprs.group(1)) + (int(agprs.group(1)) if agprs else 0),
                     int(scratch.group(1)) if scratch else 0,
                     int(occupancy.group(1)) if occupancy else None)


class CompiledScore:
    """Registers and spills from the target's own compiler.

    The static model cannot say whether a configuration spills: on GB200 it
    gave eight lanes at b = 56 its highest figure and ptxas no spill, and at
    b = 80 a low one and ptxas 328 bytes.  The compiler can, so where one is
    available it decides that part: no spill before any spill, fewer spilled
    bytes before more, then multiplications resident per SM under shared
    memory, threads *and* registers, then issue slots per multiplication.
    """

    def __init__(self, toolchain: Optional[Toolchain] = None,
                 flags: Sequence[str] = (), timeout: float = 600):
        self.toolchain = toolchain or Toolchain()
        self.flags = tuple(flags)
        self.timeout = timeout
        self.reports: Dict[Candidate, Resources] = {}

    def available(self, context: Context) -> bool:
        return self.toolchain.compiler(context.get_vm().get_hw_descr().vendor) is not None

    def resources(self, result: Build) -> Optional[Resources]:
        hw = result.context.get_vm().get_hw_descr()
        compiler = self.toolchain.compiler(hw.vendor)
        if compiler is None:
            raise RuntimeError(
                f'no compiler for {hw.vendor}: pass one in `Toolchain`, or set '
                f'TF_NVCC / TF_HIPCC')
        with tempfile.TemporaryDirectory(prefix='tf-tune-') as tmp:
            src = os.path.join(tmp, 'kernel.cpp' if hw.vendor == 'intel'
                               else 'kernel.cu')
            with open(src, 'w') as f:
                f.write(kernel_source(result))
            inc = self.toolchain.include_dir()
            if hw.vendor == 'nvidia':
                arch = hw.model if hw.model.endswith('a') else (
                    hw.model + 'a' if hw.model in ('sm_90', 'sm_100', 'sm_101', 'sm_120')
                    else hw.model)
                cmd = [compiler, '-cubin', f'-arch={arch}', '--expt-relaxed-constexpr',
                       '-Xptxas', '-v', '-I', inc, '-o', os.path.join(tmp, 'k.cubin'), src]
                parse = parse_ptxas
            elif hw.vendor == 'intel':
                # Linked, as a shared object: the ahead-of-time device build
                # runs at link time, and `-c` leaves `-device` unused and IGC
                # silent.
                cmd = [compiler, '-fsycl', '-fsycl-targets=spir64_gen', '-O3',
                       '-shared', '-fPIC', '-Xsycl-target-backend',
                       f'-device {hw.model}', '-I', inc,
                       '-o', os.path.join(tmp, 'k.so'), src]
                parse = parse_igc
            else:
                cmd = [compiler, '-x', 'hip', '-c', f'--offload-arch={hw.model}',
                       '--offload-device-only', '-O3', '-I', inc,
                       '-Rpass-analysis=kernel-resource-usage',
                       '-o', os.path.join(tmp, 'k.o'), src]
                parse = parse_amdgpu
            run = subprocess.run(cmd + list(self.flags), capture_output=True,
                                 text=True, timeout=self.timeout)
        report = parse(run.stdout + run.stderr)
        if report is None or run.returncode:
            return None
        if hw.vendor == 'nvidia' and report.registers:
            gen = result.generator
            threads = gen._num_threads * gen.launch_config().mults_per_block
            per_thread = -(-report.registers // 8) * 8
            report = replace(report, register_blocks=hw.max_reg_per_block
                             // max(1, per_thread * threads))
        self.reports[result.candidate] = report
        return report

    def __call__(self, result: Build):
        if not result.ok:
            return None
        report = self.resources(result)
        if report is None:
            return None
        lanes, wave, resident = _geometry(result)
        mults = result.generator.launch_config().mults_per_block
        if report.register_blocks is not None and result.context.get_vm().get_hw_descr().vendor == 'nvidia':
            blocks = min(result.generator.resident_blocks or 0, report.register_blocks)
            resident = blocks * mults
        issue = (result.generator.emitted_work or 0) * lanes / wave
        return (report.spill_bytes > 0, report.spill_bytes, -resident, issue)


class MeasuredScore:
    """A caller's measurement: `run(result)` returns a time, or None where it
    could not take one.  What a real autotuner plugs in -- this module does
    not launch anything."""

    def __init__(self, run: Callable[[Build], Optional[float]]):
        self.run = run

    def __call__(self, result: Build):
        return self.run(result) if result.ok else None


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #

@dataclass
class Trial:
    candidate: Candidate
    score: Any
    error: Optional[BaseException] = None


@dataclass
class Outcome:
    best: Candidate
    score: Any
    trials: List[Trial] = field(default_factory=list)


def _evaluate(descr_factory, context, scorer, candidate, cache):
    if candidate in cache:
        return cache[candidate]
    result = build(descr_factory, context, candidate)
    if not result.ok:
        # Not the scorer's to judge: a build that failed has no figure, and a
        # scorer written for successes should not have to know that.
        trial = Trial(candidate, None, result.error)
    else:
        try:
            trial = Trial(candidate, scorer(result))
        except Exception as exc:
            trial = Trial(candidate, None, exc)
    cache[candidate] = trial
    return trial


def _better(a, b) -> bool:
    return b is None or (a is not None and a < b)


def exhaustive(descr_factory, context: Context, scorer,
               knobs: Optional[Sequence[Knob]] = None,
               origin: Optional[Candidate] = None) -> Outcome:
    """Every point.  Only for a space small enough to build whole."""
    descrs = descr_factory()
    knobs = space(descrs, context) if knobs is None else knobs
    origin = start(descrs, context) if origin is None else origin
    cache: Dict[Candidate, Trial] = {}
    best = None
    for cand in enumerate_space(knobs, origin):
        trial = _evaluate(descr_factory, context, scorer, cand, cache)
        if _better(trial.score, None if best is None else best.score):
            best = trial
    return _outcome(best, cache, origin)


def coordinate(descr_factory, context: Context, scorer,
               knobs: Optional[Sequence[Knob]] = None,
               origin: Optional[Candidate] = None,
               rounds: int = 3, budget: Optional[int] = None,
               seed: Optional[Dict[Candidate, Trial]] = None) -> Outcome:
    """One knob at a time, from the default, keeping whatever improves.

    A round costs the sum of the knobs' value counts rather than their
    product -- tens of builds instead of thousands -- and it misses a pair that
    only pays together.  Eight lanes without `k_roll` at b = 56 spills and with
    it does not, so the round is repeated until nothing moves.
    """
    descrs = descr_factory()
    knobs = space(descrs, context) if knobs is None else knobs
    origin = start(descrs, context) if origin is None else origin
    cache: Dict[Candidate, Trial] = dict(seed or {})
    best = _evaluate(descr_factory, context, scorer, origin, cache)
    for _ in range(rounds):
        moved = False
        for knob in knobs:
            for value in knob.values(best.candidate):
                cand = best.candidate.set(knob.name, value)
                if (budget is not None and cand not in cache
                        and len(cache) >= budget):
                    continue
                trial = _evaluate(descr_factory, context, scorer, cand, cache)
                if _better(trial.score, best.score):
                    best, moved = trial, True
        if not moved:
            break
    return _outcome(best, cache, origin)


def _outcome(best: Optional[Trial], cache, origin) -> Outcome:
    trials = list(cache.values())
    if best is None or best.score is None:
        failed = [t.error for t in trials if t.error is not None]
        if failed and len(failed) == len(trials):
            raise failed[0]
        return Outcome(origin, None, trials)
    return Outcome(best.candidate, best.score, trials)


def tune(descr_factory, context: Context, scorer=static_score,
         strategy: Callable = coordinate, **kwargs) -> Outcome:
    """Choose a configuration for `descr_factory()`'s kernel on `context`'s
    target.  `descr_factory` returns a fresh descriptor list per call."""
    return strategy(descr_factory, context, scorer, **kwargs)


# --------------------------------------------------------------------------- #
# Autotuning a kernel as it is generated
# --------------------------------------------------------------------------- #

#: Picks made in this process, by key (`_key`).  The file behind
#: `Options.autotune_cache` is read into it and written from it.
_PICKS: Dict[str, Dict[str, Any]] = {}
_LOADED: set = set()


def _key(result: Build, mode: str) -> str:
    """The default build's source and the target: what the pick depends on.

    The source rather than the descriptors, because it is the one thing that
    says everything -- shapes, addressing, the options already asked -- and
    it has been built anyway, as the walk's first point.
    """
    hw = result.context.get_vm().get_hw_descr()
    sha = hashlib.sha256()
    for part in (mode, hw.model, backend_of(result.context), str(result.context.fp_type),
                 kernel_source(result)):
        sha.update(part.encode())
        sha.update(b'\0')
    return sha.hexdigest()


def _load(path: Optional[str]) -> None:
    if not path or path in _LOADED:
        return
    _LOADED.add(path)
    try:
        with open(path) as f:
            _PICKS.update(json.load(f))
    except (OSError, ValueError):
        pass


def _store(path: Optional[str], key: str, pick: Candidate) -> None:
    _PICKS[key] = pick.to_dict()
    if not path:
        return
    tmp = f'{path}.{os.getpid()}.tmp'
    try:
        with open(tmp, 'w') as f:
            json.dump(_PICKS, f, indent=1, sort_keys=True)
        os.replace(tmp, path)
    except OSError as exc:
        warnings.warn(f'autotune: could not write {path}: {exc}')


def autotune(descr_factory, context: Context, mode: str = 'static',
             budget: Optional[int] = 24,
             cache: Optional[str] = None) -> Optional[Candidate]:
    """The configuration `Options.autotune` builds a kernel with.

    One build of the default first: its source is the cache key, and a kernel
    already tuned costs nothing more.  Otherwise a coordinate walk over
    `simple_space`, the default seeded in, within `budget` builds.  None where
    the default does not build -- the generator then fails the way it would
    have, instead of this failing differently.
    """
    descrs = descr_factory()
    # A measurement first: where one says what is best for this device and
    # this shape, it is taken as it is, whatever the scorers would say --
    # they rank what a build reports about itself, and the preference is what
    # the machine did.  One build, to check that it builds at all.
    from tensorforge.generators import preferences
    pref = preferences.lookup(descrs, context)
    if pref is not None:
        chosen = preferences.candidate(pref, descrs, context)
        if build(descr_factory, context, chosen).ok:
            return chosen
        warnings.warn(f'autotune: the preference from {pref.source} '
                      f'({pref.evidence}) does not build here; ranking instead')
    if mode == 'prefer':
        return None
    knobs = simple_space(descrs, context)
    origin = start(descrs, context)
    first = build(descr_factory, context, origin)
    if not first.ok:
        return None
    if not knobs:
        return origin
    if mode == 'compiled':
        scorer = CompiledScore()
        if not scorer.available(context):
            warnings.warn('autotune=compiled: no compiler for this target '
                          '(Toolchain, TF_NVCC, TF_HIPCC); ranking statically')
            scorer, mode = static_score, 'static'
    elif mode == 'static':
        scorer = static_score
    else:
        raise ValueError(f'autotune: unknown mode {mode!r}; off, prefer, '
                         f'static or compiled')
    _load(cache)
    key = _key(first, mode)
    if key in _PICKS:
        return Candidate.from_dict(_PICKS[key])
    try:
        seed = {origin: Trial(origin, scorer(first))}
    except Exception as exc:
        seed = {origin: Trial(origin, None, exc)}
    out = coordinate(descr_factory, context, scorer, knobs=knobs, origin=origin,
                     budget=budget, seed=seed)
    _store(cache, key, out.best)
    return out.best
