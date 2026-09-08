# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What a descriptor list costs, before anything is generated or run.

Two numbers, and a roofline needs both: the arithmetic a kernel contains, and
the bytes it cannot avoid moving.  Neither is available anywhere else today --
`frontend/yateto.py` returns a hardcoded `0` where it means to return
`get_flops()`, and `hw_descr_db.yml` describes what the code generator may
emit, not what the machine can sustain.

The counts here are *of the operation*, not of the emitted code.  A kernel that
skips a structurally-zero block does less work than this reports, and one that
reloads an operand per iteration moves more bytes.  Both gaps are real and both
are worth knowing, which is the argument for keeping this model separate from
the measurement rather than folding one into the other: subtracting them is the
whole point.  `tools/bench/` measures, this states what was asked for.

## What is counted, and what is refused

Multiplications and additions are counted apart and summed into `flops`.
Transcendentals are counted apart and *not* summed in, because their cost is a
property of the target -- an `exp` is a handful of instructions on one machine
and a table lookup on another -- and a single weight would be an invention.
`min`, `max`, `abs`, `neg` and the comparisons land in `nonarith`: they are
work, they are not floating-point arithmetic, and counting them as flops makes
a max-reduction look like it computes something it does not.

Sparsity is reported per tensor and never applied to the arithmetic.  Which
structurally-zero entries actually get elided is a property of the backend and
of the block shape it chose, so folding a density into the flop count would
attribute a code-generation decision to the operation.  `density` is on the
record so a caller may scale, having decided what it is scaling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from typing import Dict, List, Optional

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.operation import Operation

#: Operations whose cost is a target property rather than a count.  Reported,
#: never summed into `flops`.
TRANSCENDENTAL = frozenset({
    Operation.SQRT, Operation.RSQRT, Operation.CBRT, Operation.RCBRT,
    Operation.RCP, Operation.DIV, Operation.MOD, Operation.POW,
    Operation.EXP, Operation.EXPM1, Operation.LOG, Operation.LOGP1,
    Operation.SIN, Operation.COS, Operation.TAN,
    Operation.ASIN, Operation.ACOS, Operation.ATAN,
    Operation.SINH, Operation.COSH, Operation.TANH,
    Operation.ASINH, Operation.ACOSH, Operation.ATANH,
    Operation.ERF, Operation.GAMMA,
})

#: Operations counted as additions.
ADDITIVE = frozenset({Operation.ADD, Operation.SUB})

#: Operations counted as multiplications.
MULTIPLICATIVE = frozenset({Operation.MUL})


@dataclass(frozen=True)
class TensorTraffic:
    """Compulsory global-memory traffic for one tensor.

    Held as the *region* each direction touches rather than as a byte count,
    because two descriptors touching two windows of one tensor move their
    union and not the larger of them.  Keeping boxes makes uniting exact;
    keeping bytes would make it a guess that is always low.

    `read_box` and `write_box` are both set for a tensor that is read and
    written, which is what a `SOURCESINK` accumulation does; they are not
    alternatives.

    Byte counts here are for *one* batch element.  The batch enters in
    `Cost.read_bytes` / `Cost.write_bytes`, which is the only place that knows
    whether `per_batch` applies -- putting it on the entry would mean either
    scaling a shared operator matrix or carrying a batch on a per-tensor record
    that has no other use for one.

    `per_batch` is False for `Addressing.NONE`, whose single storage block is
    shared across the batch -- the kernel-side pointer skips the
    `batchId * volume` term, and the host allocates once.  A model that scaled
    it by the batch would report a SeisSol operator matrix as the dominant
    cost of every kernel that touches one.
    """
    name: str
    read_box: Optional[BoundingBox]
    write_box: Optional[BoundingBox]
    elem_bytes: int
    per_batch: bool
    density: float
    stored: int           # elements the tensor occupies

    @property
    def read(self) -> int:
        return _volume(self.read_box) * self.elem_bytes

    @property
    def write(self) -> int:
        return _volume(self.write_box) * self.elem_bytes

    @property
    def touched(self) -> int:
        """Elements in the union of both directions."""
        box = _unite(self.read_box, self.write_box)
        return _volume(box)


@dataclass(frozen=True)
class Cost:
    """Arithmetic and traffic for one descriptor or one list of them."""
    mults: int = 0
    adds: int = 0
    transcendental: int = 0
    nonarith: int = 0
    read_bytes: int = 0
    write_bytes: int = 0
    batch: int = 1
    tensors: List[TensorTraffic] = field(default_factory=list)
    #: Why a figure is missing, where one is.  Empty when everything counted.
    unmodelled: List[str] = field(default_factory=list)

    @property
    def flops(self) -> int:
        return self.mults + self.adds

    @property
    def bytes(self) -> int:
        return self.read_bytes + self.write_bytes

    @property
    def intensity(self) -> Optional[float]:
        """Flops per byte moved, or None when nothing moves.

        The compulsory intensity, which is the *upper* bound: it assumes every
        byte is fetched once.  A measured intensity below it is cache misses,
        above it is impossible and means the model and the counters disagree
        about what the kernel is.
        """
        return self.flops / self.bytes if self.bytes else None

    def merged(self, other: 'Cost') -> 'Cost':
        """Two costs over the same batch, added.

        Traffic is *not* summed per tensor: a tensor two descriptors both read
        is fetched once, so the per-tensor entries are united and the byte
        totals recomputed from the union.  Summing them would count a chained
        temporary's storage twice and report an intensity that no arrangement
        of the kernel could reach.
        """
        if self.batch != other.batch:
            raise ValueError(f'cannot merge costs at batch {self.batch} and '
                             f'{other.batch}')
        by_name: Dict[str, TensorTraffic] = {t.name: t for t in self.tensors}
        for t in other.tensors:
            prev = by_name.get(t.name)
            by_name[t.name] = t if prev is None else TensorTraffic(
                name=t.name,
                read_box=_unite(prev.read_box, t.read_box),
                write_box=_unite(prev.write_box, t.write_box),
                elem_bytes=t.elem_bytes, per_batch=t.per_batch,
                density=t.density, stored=t.stored)
        tensors = [by_name[k] for k in sorted(by_name)]
        scale = {True: self.batch, False: 1}
        return Cost(
            mults=self.mults + other.mults,
            adds=self.adds + other.adds,
            transcendental=self.transcendental + other.transcendental,
            nonarith=self.nonarith + other.nonarith,
            read_bytes=sum(t.read * scale[t.per_batch] for t in tensors),
            write_bytes=sum(t.write * scale[t.per_batch] for t in tensors),
            batch=self.batch,
            tensors=tensors,
            unmodelled=self.unmodelled + other.unmodelled)


# -- helpers ---------------------------------------------------------------- #

def _volume(box: Optional[BoundingBox]) -> int:
    if box is None:
        return 0
    return prod(max(0, u - l) for l, u in zip(box.lower(), box.upper()))


def _unite(a: Optional[BoundingBox],
           b: Optional[BoundingBox]) -> Optional[BoundingBox]:
    if a is None:
        return b
    if b is None:
        return a
    return a.unite(b)


def _bucket(op: Operation, count: int) -> Dict[str, int]:
    if op in ADDITIVE:
        return {'adds': count}
    if op in MULTIPLICATIVE:
        return {'mults': count}
    if op in TRANSCENDENTAL:
        return {'transcendental': count}
    return {'nonarith': count}


def _elem_size(tensor, default: Optional[Datatype]) -> int:
    dt = tensor.datatype or default
    if dt is None:
        raise ValueError(
            f'tensor {tensor.alias or tensor.name!r} carries no datatype and '
            f'no default was given; pass datatype= (the context fp_type)')
    return dt.size()


def _name_of(tensor) -> str:
    return tensor.alias or tensor.name or f'anon@{id(tensor):x}'


def _multilinear_ranges(descr) -> Optional[Dict[int, tuple]]:
    """The iteration space `MultilinearInstruction._analyze` will run over.

    The same narrowing `OperationDescription.effective_boxes` performs, kept
    here rather than called: that one returns the *tensor* boxes it derived and
    drops the ranges, and the contracted axes -- which carry negative labels
    and appear on no tensor -- are exactly what a flop count needs.
    """
    ops = list(descr.ops or [])
    targets = list(descr.target or [])
    if descr.dest is None or len(ops) != len(targets):
        return None

    ranges: Dict[int, tuple] = {}

    def narrow(label, lo, hi):
        prev = ranges.get(label)
        ranges[label] = (max(prev[0], lo), min(prev[1], hi)) if prev else (lo, hi)

    for op, target in zip(ops, targets):
        box = getattr(op, 'bbox', None)
        if box is None or len(target) != box.rank():
            return None
        for j, label in enumerate(target):
            narrow(label, box.lower()[j], box.upper()[j])
    for j in range(descr.dest.bbox.rank()):
        narrow(j, descr.dest.bbox.lower()[j], descr.dest.bbox.upper()[j])
    return ranges


def _points(ranges: Dict[int, tuple]) -> int:
    return prod(max(0, hi - lo) for lo, hi in ranges.values()) if ranges else 1


def _traffic(descr, datatype) -> List[TensorTraffic]:
    """One entry per distinct non-temporary tensor this descriptor touches.

    Temporaries are dropped: `is_tmp` marks a tensor the generator keeps in
    shared memory or registers for the life of the kernel, so it never reaches
    global memory and a byte count including it would not be comparable with
    anything a DRAM counter reports.
    """
    boxes = descr.effective_boxes()
    if boxes is None:
        return []
    reads, write_box = boxes
    dest = descr.writes()

    out: Dict[str, TensorTraffic] = {}

    def add(tensor, box: BoundingBox, reading: bool):
        if tensor is None or tensor.is_tmp:
            return
        if tensor.addressing == Addressing.SCALAR:
            return
        name = _name_of(tensor)
        prev = out.get(name)
        read_box = _unite(prev.read_box if prev else None,
                          box if reading else None)
        write_box = _unite(prev.write_box if prev else None,
                           None if reading else box)
        out[name] = TensorTraffic(
            name=name, read_box=read_box, write_box=write_box,
            elem_bytes=_elem_size(tensor, datatype),
            per_batch=tensor.addressing != Addressing.NONE,
            density=tensor.density(), stored=tensor.get_real_volume())

    for tensor, box in reads.items():
        add(tensor, box, reading=True)
    if dest is not None:
        add(dest.tensor, write_box, reading=False)
    return [out[k] for k in sorted(out)]


def _finish(counts: Dict[str, int], tensors: List[TensorTraffic],
            batch: int, unmodelled: List[str]) -> Cost:
    scale = {True: batch, False: 1}
    return Cost(
        mults=counts.get('mults', 0) * batch,
        adds=counts.get('adds', 0) * batch,
        transcendental=counts.get('transcendental', 0) * batch,
        nonarith=counts.get('nonarith', 0) * batch,
        read_bytes=sum(t.read * scale[t.per_batch] for t in tensors),
        write_bytes=sum(t.write * scale[t.per_batch] for t in tensors),
        batch=batch, tensors=tensors, unmodelled=unmodelled)


# -- per descriptor --------------------------------------------------------- #

def descr_cost(descr, batch: int = 1,
               datatype: Optional[Datatype] = None) -> Cost:
    """Arithmetic and compulsory traffic for one descriptor.

    `datatype` is the fallback for a tensor that carries none, which is what a
    frontend leaves when the kernel's floating-point type is the context's.
    """
    from tensorforge.generators.descriptions import (
        BarrierDescription, ElementwiseDescr, MultilinearDescr, ReductionDescr)

    if isinstance(descr, BarrierDescription):
        return Cost(batch=batch)

    if isinstance(descr, MultilinearDescr):
        ranges = _multilinear_ranges(descr)
        if ranges is None:
            return Cost(batch=batch, unmodelled=[
                f'{type(descr).__name__}: operand ranks do not line up with '
                f'their target index lists; no iteration space to count'])
        points = _points(ranges)
        out_points = prod(
            max(0, ranges[j][1] - ranges[j][0])
            for j in range(descr.dest.bbox.rank())) if ranges else 0

        # A scalar operand is one multiply per *output* element, not one per
        # contraction point: `alpha * sum(...)` and `sum(alpha * ...)` are the
        # same value, and no implementation worth measuring picks the second.
        tensor_ops = [o for o in descr.ops
                      if getattr(o, 'tensor', None) is not None
                      and o.tensor.addressing != Addressing.SCALAR]
        scalars = len(descr.ops) - len(tensor_ops)

        counts = {
            'mults': max(0, len(tensor_ops) - 1) * points + scalars * out_points,
            # Accumulating reads the destination back, so every point is an
            # add.  Assigning writes the first term, so the destination's own
            # elements cost no add.
            'adds': points if descr.add else max(0, points - out_points),
        }
        return _finish(counts, _traffic(descr, datatype), batch, [])

    if isinstance(descr, ElementwiseDescr):
        points = prod(descr.dest.bbox.sizes())
        return _finish(_bucket(descr.op, points),
                       _traffic(descr, datatype), batch, [])

    if isinstance(descr, ReductionDescr):
        points = prod(descr.var.bbox.sizes())
        kept = prod(descr.dest.bbox.sizes())
        return _finish(_bucket(descr.op.operation(), max(0, points - kept)),
                       _traffic(descr, datatype), batch, [])

    return Cost(batch=batch, unmodelled=[
        f'{type(descr).__name__}: no cost model'])


def list_cost(descr_list: List, batch: int = 1,
              datatype: Optional[Datatype] = None) -> Cost:
    """The whole list, with traffic united rather than summed.

    Arithmetic adds up across descriptors; bytes do not.  Two descriptors
    reading the same operand fetch it once, and a chain whose middle tensor is
    a temporary moves it never -- so the totals come from the union of what
    each descriptor touches, not from adding their separate answers.
    """
    total = Cost(batch=batch)
    for descr in descr_list:
        total = total.merged(descr_cost(descr, batch=batch, datatype=datatype))
    return total


# -- how much code a list becomes ------------------------------------------- #

#: Emitted lines per lane-flop, and the fixed part of a kernel.
#:
#: Fitted against the order-6 viscoelastic set as SeisSol ships it: 57 kernels,
#: R^2 = 0.998 on the totals and 1.8 percent out on the largest, which is the
#: one this exists for.
#:
#: It does not hold across scales, and that is worth stating rather than
#: discovering.  Refitting against the case set -- 54 small kernels generated
#: on the spot -- gives roughly half this slope, so on a body of a few hundred
#: lines the figure comes out about twice too large.  The count ignores
#: everything but arithmetic, and what staging, index arithmetic and guards
#: cost relative to the arithmetic is simply not the same at the two ends.
#:
#: Which direction that errs in matters.  Overstating a small body makes a
#: budget more likely to roll it, and a small body is the one case where
#: rolling is clearly not worth it -- so the size a run must reach before it is
#: rolled at all is asked separately, and is not this number's job.  Here the
#: figure is used for the opposite question, whether a large body is too large,
#: and there it is worth what the fit above says it is.
#:
#: `tools/calibrate_code_size.py` redoes both fits.
LINES_PER_LANE_FLOP = 4.1
LINES_FIXED = 25


def estimated_lines(descr_list: List, num_threads: int,
                    datatype: Optional[Datatype] = None) -> int:
    """Roughly how many lines of kernel this list becomes.

    The arithmetic is spread over the lanes, so what one lane writes out is the
    iteration space divided by the lane count -- which is why the estimate is
    in lane-flops and not in flops, even though a corpus at one lane count
    cannot tell the two apart.

    An estimate and named as one.  It answers whether a body is large against
    an instruction cache; it does not answer how large.
    """
    if num_threads <= 0:
        raise ValueError('a kernel has at least one lane')
    flops = list_cost(descr_list, batch=1, datatype=datatype).flops
    return int(LINES_PER_LANE_FLOP * flops / num_threads) + LINES_FIXED
