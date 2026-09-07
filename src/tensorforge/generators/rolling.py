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

from typing import List, Optional, Sequence

from tensorforge.analysis.dependence import binding_period, carried, escapes
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
         allow_barriers: bool = False) -> List[OperationDescription]:
    """Replace each repeated run of a descriptor list with a `ForDescr`.

    Runs that are not rolled are left where they are, so the result is the same
    list with some slices replaced and nothing reordered.  Whether rolling is
    worth it -- fewer instruction bytes against a counter and an indexed load --
    is a cost question and is not decided here; this states the run as a loop
    so that something else can weigh the two forms against each other.
    """
    runs = find_repeats(descrs, min_count=min_count, max_period=max_period,
                        max_arity=max_arity)

    out: List[OperationDescription] = []
    cursor = 0
    for run in runs:
        chunks = _chunks(descrs, run.start, run.period, run.count)
        if not allow_barriers and any(d.barrier() for d in chunks[0]):
            continue
        if run.arity == 0:
            continue

        out.extend(descrs[cursor:run.start])
        holes = list(zip(*run.general.bindings))
        loop = ForDescr(
            run.general,
            carried(chunks),
            periods=tuple(binding_period(h) for h in holes),
            escaping=escapes(descrs, run.start, run.stop))
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
