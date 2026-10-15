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

Sparsity is counted *beside* the arithmetic and never folded into it.  Which
structurally-zero entries actually get elided is a property of the backend and
of the block shape it chose, so scaling `flops` by a density would attribute a
code-generation decision to the operation.  So there are two figures:
`flops`, what a dense implementation performs, and `nonzero_flops`, what the
operation needs once the zeros are taken out -- the count yateto's
`ComputeOptimalFlopCount` arrives at.  The difference between them is what a
backend has left to win, which is worth having as a number rather than as an
impression.

The zeros come from `Tensor.nonzero_pattern`, which is the *arithmetic*
pattern and not the storage one: an operator stored dense over its bounding
box is still mostly zero, and only the operand's own numbers say where.
Where an operand carries no numbers there is nothing to derive and the
storage pattern stands, so the count is then the dense one and
`nonzero_exact` says so.
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
    Operation.EXP, Operation.EXPM1, Operation.LOG, Operation.LOG1P,
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
    #: The same two over the points the sparsity patterns leave: what the
    #: operation *needs*, where `mults`/`adds` are what a dense
    #: implementation performs.  Equal where every operand is dense.  They are
    #: separate fields rather than a density applied to the first pair,
    #: because which structural zeros a backend actually elides is its
    #: decision and the difference between the two is the measurement of it.
    nonzero_mults: int = 0
    nonzero_adds: int = 0
    transcendental: int = 0
    nonarith: int = 0
    read_bytes: int = 0
    write_bytes: int = 0
    batch: int = 1
    tensors: List[TensorTraffic] = field(default_factory=list)
    #: Why a figure is missing, where one is.  Empty when everything counted.
    unmodeled: List[str] = field(default_factory=list)
    #: Whether `nonzero_mults`/`nonzero_adds` are a sparsity-aware count
    #: rather than the dense figures repeated.  False where an operand
    #: carried no pattern or the index space was too large to walk.
    nonzero_exact: bool = True

    @property
    def flops(self) -> int:
        return self.mults + self.adds

    @property
    def nonzero_flops(self) -> int:
        """The arithmetic the sparsity leaves, which is what yateto's
        `nonZeroFlops` counts and what an achieved rate should be measured
        against where the backend elides structural zeros.

        Checked against yateto over an elastic order-4 export: 113 of 119
        kernels agree exactly, against 59 for the dense `flops`.  The six that
        do not are the chains -- `derivative`, `volume`, `fsgKernel` -- whose
        operands include a matrix that carries no numbers and whose pattern
        the description therefore does not state.  Supplying yateto's own
        pattern for those brings `volume` onto its 4806 exactly, so what
        separates the two is the missing pattern and not the convention.
        """
        return self.nonzero_mults + self.nonzero_adds

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
            nonzero_mults=self.nonzero_mults + other.nonzero_mults,
            nonzero_adds=self.nonzero_adds + other.nonzero_adds,
            nonzero_exact=self.nonzero_exact and other.nonzero_exact,
            transcendental=self.transcendental + other.transcendental,
            nonarith=self.nonarith + other.nonarith,
            read_bytes=sum(t.read * scale[t.per_batch] for t in tensors),
            write_bytes=sum(t.write * scale[t.per_batch] for t in tensors),
            batch=self.batch,
            tensors=tensors,
            unmodeled=self.unmodeled + other.unmodeled)


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


#: Points of an iteration space this will walk before giving up.  The walk is
#: one `is_nz` per operand per point, which is cheap per point and unbounded
#: in the product; past this the dense figure stands and `unmodeled` says so.
NONZERO_POINT_CAP = 1 << 22


def _nonzero_points(descr, ranges: Dict[int, tuple]):
    """`(active, out_active)`: points where every tensor operand is nonzero,
    and destination points at least one such point reaches.  None where the
    space is too large to walk or an operand has no pattern to ask.

    This is the count yateto's `ComputeOptimalFlopCount` arrives at by
    propagating the equivalent sparsity patterns through the tree: a product
    contributes one multiplication per nonzero of its result, and a
    contraction one addition per nonzero it folds away.  Walking the index
    space reaches the same answer without building the patterns, which is
    worth it here only because the spaces are small -- a few thousand points
    for the kernels this is asked about.
    """
    from itertools import product as iproduct

    ops = [o for o in (descr.ops or [])
           if getattr(o, 'tensor', None) is not None
           and o.tensor.addressing != Addressing.SCALAR]
    targets = [t for o, t in zip(descr.ops or [], descr.target or [])
               if getattr(o, 'tensor', None) is not None
               and o.tensor.addressing != Addressing.SCALAR]
    if not ops or len(ops) != len(targets):
        return None

    patterns = []
    for op in ops:
        # The *non-zero* pattern and not the storage one: an operator stored
        # dense over its bounding box still has entries that are zero in
        # every batch element, and yateto's count skips the products that
        # read them.  Falls back to the storage pattern where the operand
        # carries no numbers, which is the same answer as before.
        spp = getattr(op.tensor, 'nonzero_pattern', None)
        if spp is None:
            spp = getattr(op.tensor, 'spp', None)
        if spp is None or not hasattr(spp, 'is_nz'):
            return None
        patterns.append(spp)

    labels = sorted(ranges)
    extents = [ranges[l] for l in labels]
    total = 1
    for lo, hi in extents:
        total *= max(0, hi - lo)
    if total == 0:
        return 0, 0
    if total > NONZERO_POINT_CAP:
        return None

    dest_rank = descr.dest.bbox.rank()
    at = {label: i for i, label in enumerate(labels)}
    # Which position of the full point each operand axis reads.
    picks = [[at[label] for label in target] for target in targets]
    dest_picks = [at[j] for j in range(dest_rank) if j in at]

    active = 0
    reached = set()
    for point in iproduct(*(range(lo, hi) for lo, hi in extents)):
        for spp, pick in zip(patterns, picks):
            if not spp.is_nz(tuple(point[i] for i in pick)):
                break
        else:
            active += 1
            reached.add(tuple(point[i] for i in dest_picks))
    return active, len(reached)


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
            batch: int, unmodeled: List[str]) -> Cost:
    scale = {True: batch, False: 1}
    return Cost(
        mults=counts.get('mults', 0) * batch,
        adds=counts.get('adds', 0) * batch,
        # Absent where nothing counted them: a descriptor whose patterns are
        # unavailable reports the dense figure for both rather than zero, so
        # that a sum over a list is never short by the kernels it could not
        # analyse.
        nonzero_mults=counts.get('nonzero_mults',
                                 counts.get('mults', 0)) * batch,
        nonzero_adds=counts.get('nonzero_adds',
                                counts.get('adds', 0)) * batch,
        nonzero_exact=not counts.get('nonzero_inexact'),
        transcendental=counts.get('transcendental', 0) * batch,
        nonarith=counts.get('nonarith', 0) * batch,
        read_bytes=sum(t.read * scale[t.per_batch] for t in tensors),
        write_bytes=sum(t.write * scale[t.per_batch] for t in tensors),
        batch=batch, tensors=tensors, unmodeled=unmodeled)


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
            return Cost(batch=batch, unmodeled=[
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
        # Not an `unmodeled` entry: that list is for a cost this cannot
        # state at all, and the dense figures are stated.  What is missing
        # here is only the *sparsity-aware* pair, which falls back to them
        # and says so through `nonzero_exact`.
        nonzero = _nonzero_points(descr, ranges)
        if nonzero is None:
            counts['nonzero_inexact'] = 1
        else:
            live, out_live = nonzero
            counts['nonzero_mults'] = (max(0, len(tensor_ops) - 1) * live
                                       + scalars * out_live)
            counts['nonzero_adds'] = (live if descr.add
                                      else max(0, live - out_live))
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

    return Cost(batch=batch, unmodeled=[
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
