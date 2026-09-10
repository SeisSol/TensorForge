# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Stating a repeated run once, and putting it back.

The first of the bindings: a run's varying operands bound to a counter.  The
analysis under :mod:`tensorforge.analysis` says which slices repeat, what a
repetition costs and what holds between its chunks; this turns that into a
descriptor and back again.

`unroll(roll(x))` is the property the rest of the work is checked against.  A
rewrite that cannot be undone exactly is a rewrite whose meaning is only
approximately the original's, and the whole approach -- one body, several
bindings -- rests on binding being the inverse of generalising.  Testing it on
the loop binder tests it for the binders that follow, since they differ only
in what the hole is bound *to*.

Two refusals are deliberate.  A run whose chunk contains a barrier is not
rolled: the descriptor list is split into sections at every barrier, so a
barrier inside a loop body is one the split cannot see, and hiding it would
turn a synchronisation into a silent reordering.  A run that varies in nothing
is not rolled either -- it is the same computation done several times, which is
a question for whoever wrote it and not something to make tidier.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from tensorforge.analysis.antiunify import substitute
from tensorforge.analysis.cost import estimated_lines
from tensorforge.analysis.dependence import (binding_period, carried,
                                             escapes, shifts)
from tensorforge.analysis.families import find_repeats
from tensorforge.generators.descriptions import ForDescr, OperationDescription


def _chunks(descrs: Sequence[OperationDescription], start: int, period: int,
            count: int) -> List[List[OperationDescription]]:
    return [list(descrs[start + period * i:start + period * (i + 1)])
            for i in range(count)]


def roll(descrs: Sequence[OperationDescription],
         min_count: int = 2,
         max_period: Optional[int] = None,
         max_arity: Optional[int] = None,
         allow_barriers: bool = False,
         keep_unrolled_under: Optional[int] = None,
         fit_within: Optional[int] = None,
         num_threads: int = 32) -> List[OperationDescription]:
    """Replace each repeated run of a descriptor list with a `ForDescr`.

    Runs that are not rolled are left where they are, so the result is the same
    list with some slices replaced and nothing reordered.

    Two different questions decide whether a run is rolled, and they are asked
    separately because they are not the same question.

    `keep_unrolled_under` asks whether *this* run is worth a loop at all.  A
    small body is better left alone: written out it costs a few hundred lines
    and keeps every operand at a compile-time address, while rolled it costs a
    counter, an indexed load per varying operand, and a residency that has to
    survive the back edge.

    `fit_within` asks whether the *list* is too large, in estimated lines, and
    is the one that answers to an instruction cache.  A cache holds the body
    that repeats, not one run of it, so every roll changes the same total and
    a threshold applied run by run answers a question nobody asked.  Runs are
    taken in order of what they save until the total fits, so a list that
    already fits keeps every operand where it was.

    Left unset neither is weighed and every run is rolled, which is what the
    tests want and not what a generator should do.
    """
    runs = find_repeats(descrs, min_count=min_count, max_period=max_period,
                        max_arity=max_arity)

    eligible = []
    for run in runs:
        chunks = _chunks(descrs, run.start, run.period, run.count)
        if not allow_barriers and any(d.barrier() for d in chunks[0]):
            continue
        if run.arity == 0:
            continue
        flat = [d for chunk in chunks for d in chunk]
        written_out = estimated_lines(flat, num_threads)
        if keep_unrolled_under is not None and written_out < keep_unrolled_under:
            continue
        saving = written_out - estimated_lines(chunks[0], num_threads)
        eligible.append((run, chunks, saving))

    if fit_within is not None:
        total = estimated_lines(
            [op for descr in descrs for op in descr.operations()], num_threads)
        # Largest saving first, and only as many as the total needs.  Ties go
        # to the earlier run so that the same list always rolls the same way.
        taken = []
        for run, chunks, saving in sorted(
                eligible, key=lambda e: (-e[2], e[0].start)):
            if total <= fit_within:
                break
            taken.append((run, chunks, saving))
            total -= saving
        eligible = sorted(taken, key=lambda e: e[0].start)

    out: List[OperationDescription] = []
    cursor = 0
    for run, chunks, _ in eligible:

        out.extend(descrs[cursor:run.start])
        holes = list(zip(*run.general.bindings))
        loop = ForDescr(
            run.general,
            carried(chunks),
            periods=tuple(binding_period(h) for h in holes),
            escaping=escapes(descrs, run.start, run.stop),
            shifts=shifts(run.general.bindings))
        out.append(loop)
        cursor = run.stop

    out.extend(descrs[cursor:])
    return out


def unroll(descrs: Sequence[OperationDescription]
           ) -> List[OperationDescription]:
    """Expand every loop back into the operations it stands for.

    Exact rather than equivalent: the descriptors that come out are the ones
    that went in, rebuilt operand by operand from the same views.
    """
    out: List[OperationDescription] = []
    for descr in descrs:
        if isinstance(descr, ForDescr):
            for body in descr.bodies():
                out.extend(body)
        else:
            out.append(descr)
    return out


# ---------------------------------------------------------------------------
# What a builder needs from a loop
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Variant:
    """One hole, as the thing a body names and the things it stands for.

    ``stand_in`` is a tensor with the members' shape, sparsity, addressing and
    type and a name of its own; ``members`` are the views the iterations
    actually bind, in order.

    The body is built against the stand-in and not against the first member.
    Building against a member would leave that member's name in the generated
    body, which is right for one iteration in four and wrong for the rest -- so
    it is given a name that belongs to no iteration, and what the name resolves
    to is the table's business.
    """

    stand_in: object
    members: Tuple[object, ...]

    @property
    def count(self) -> int:
        return len(self.members)


def _stand_in(view, name: str):
    """A tensor interchangeable with `view`'s, under a name of its own."""
    from tensorforge.common.matrix.tensor import SubTensor, Tensor
    source = view.tensor
    clone = Tensor(list(source.shape), source.addressing, source.bbox,
                   alias=name, is_tmp=source.is_tmp, spp=source.spp,
                   datatype=source.datatype,
                   alignment=getattr(source, 'alignment', 0))
    clone.direction = source.direction
    # The body reads every member through the stand-in, so the stand-in has to
    # read what the members store: an operand held as two TF32 halves per
    # element read as one scalar is a wrong kernel, not a slow one.  The
    # members agree on it -- `is_similar` compares it -- so the first speaks
    # for all.  The order is not copied: `_offer_order` decides it later, for
    # the stand-in and its members together.
    clone.storage_parts = getattr(source, 'storage_parts', 1)
    clone.is_variant = True
    return SubTensor(clone, view.bbox, list(view.offset),
                     getattr(view, 'sliced', False))


def variant_body(loop, prefix: str = 'variant'
                 ) -> Tuple[List[OperationDescription], List[Variant]]:
    """The one body a loop emits, and what each of its holes stands for.

    The step between a rewritten descriptor list and a builder: the loop knows
    which operands vary and what they are per iteration, the builder needs one
    body to build and one name per hole to resolve through a table.  Splitting
    it here keeps the substitution -- which is `antiunify`'s -- out of the
    backend, and keeps the backend's question to where a name resolves.
    """
    variants = []
    for index, column in enumerate(zip(*loop.general.bindings)):
        stand_in = _stand_in(column[0], f'{prefix}{index}')
        # What the stand-in resolves to, iteration by iteration.  Not needed
        # to build the body -- that is the point of a stand-in -- but needed
        # by a decision about the members' storage that the body's reads have
        # to follow (`MultilinearInstruction._offer_simt_order`).
        stand_in.tensor.variant_members = tuple(view.tensor for view in column)
        variants.append(Variant(stand_in, tuple(column)))
    body = substitute(loop.general, [v.stand_in for v in variants])
    return body, variants
