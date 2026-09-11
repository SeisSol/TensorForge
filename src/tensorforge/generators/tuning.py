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

import itertools
import os
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
        return Context(arch=hw.model, backend=hw.backend, fp_type=base.fp_type,
                       options=Options(**asked))

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
    mults = gen._section.shr_mem_obj.get_mults_per_block()
    return gen._num_threads, hw.vec_unit_length, (gen.resident_blocks or 0) * mults


def static_score(result: Build):
    """What the build alone says, per multiplication rather than per block.

    Multiplications resident per SM first -- blocks times the multiplications
    a block holds, since eight lanes put four times as many in a block as 32.
    Then warp issue slots per multiplication: the arithmetic written out, times
    the share of a warp one multiplication takes.  Then the modelled register
    footprint.  Per block and per lane, as `lanes.search` ranks, the default
    32 lanes came first at every size GB200 measured, and it was the slowest
    at three of five.
    """
    if not result.ok:
        return None
    gen = result.generator
    lanes, wave, resident = _geometry(result)
    issue = (gen.emitted_work or 0) * lanes / wave
    return (-resident, issue, gen.peak_pressure or 0)


@dataclass
class Toolchain:
    """Where the compilers are.  Paths from the caller, then `TF_NVCC` /
    `TF_HIPCC`, then `PATH`; None where none of them has one."""
    nvcc: Optional[str] = None
    hipcc: Optional[str] = None
    include: Optional[str] = None

    def compiler(self, vendor: str) -> Optional[str]:
        if vendor == 'nvidia':
            return (self.nvcc or os.environ.get('TF_NVCC')
                    or shutil.which('nvcc'))
        if vendor == 'amd':
            return (self.hipcc or os.environ.get('TF_HIPCC')
                    or shutil.which('hipcc'))
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
            src = os.path.join(tmp, 'kernel.cu')
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
            threads = gen._num_threads * gen._section.shr_mem_obj.get_mults_per_block()
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
        mults = result.generator._section.shr_mem_obj.get_mults_per_block()
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
               rounds: int = 3) -> Outcome:
    """One knob at a time, from the default, keeping whatever improves.

    A round costs the sum of the knobs' value counts rather than their
    product -- tens of builds instead of thousands -- and it misses a pair that
    only pays together.  Eight lanes without `k_roll` at b = 56 spills and with
    it does not, so the round is repeated until nothing moves.
    """
    descrs = descr_factory()
    knobs = space(descrs, context) if knobs is None else knobs
    origin = start(descrs, context) if origin is None else origin
    cache: Dict[Candidate, Trial] = {}
    best = _evaluate(descr_factory, context, scorer, origin, cache)
    for _ in range(rounds):
        moved = False
        for knob in knobs:
            for value in knob.values(best.candidate):
                cand = best.candidate.set(knob.name, value)
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
