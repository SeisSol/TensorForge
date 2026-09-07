# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What several descriptor lists have in common, and where they part.

Two bodies that compute the same thing over different tensors can be generated
once instead of once each.  Deciding whether that is so is a structural
question and is answered here, in one place, because every consumer of the
answer wants it for a different reason: rolling a repeated body into a loop,
selecting a constant operand at runtime, or merely reporting that two kernels
are one kernel apart.

The question is *anti-unification* -- the least general form of which each
input is an instance.  It is the exact dual of common-subexpression
elimination: CSE shares subgraphs that are identical, this shares subgraphs
that are identical except at named positions, and pays for the difference with
a parameter at each such position.  Those positions are called holes.

A hole is legal only where the tensors filling it are *interchangeable to the
generated code*: same shape, same sparsity, same addressing, same element type,
same alignment, same box.  Everything the emitter reads off a tensor other than
which tensor it is has to agree, or the one body cannot stand for both.  That
is what :class:`OperandKey` collects, and it is deliberately conservative --
a field that turns out not to matter can be dropped once something proves it,
whereas a field wrongly left out produces a kernel that is quietly wrong for
half its inputs.

Sharing is checked as well as agreement.  ``C += A B`` and ``C += A A`` have
identical operand keys at every position and are not interchangeable, because
the second names one tensor twice and a body written for it would read the same
address for both operands.  So each body is reduced to a partition of its slots
by identity, and two bodies agree only if their partitions do.

What this module does not do: choose what to do with a hole.  Binding it to a
constant, to a loop counter or to a runtime index are three different
lowerings with three different costs, and none of them is a structural fact.
"""

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple, Union

from tensorforge.generators.descriptions import (BarrierDescription,
                                                 ElementwiseDescr,
                                                 MultilinearDescr,
                                                 OperationDescription,
                                                 ReductionDescr,
                                                 RegionDescription)


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------


def _spp_key(tensor) -> Tuple:
    """A canonical, hashable form of a tensor's sparsity pattern.

    Asked of the pattern rather than of its class: two patterns built by
    different constructors can describe the same set of nonzeros, and for the
    purpose of substituting one tensor for another it is the set that matters
    and not how it was spelled.  The enumeration is over the declared shape,
    which for the operators this is aimed at is a few thousand entries at
    worst and is computed once per tensor.
    """
    spp = tensor.spp
    if spp is None:
        return ('full', tuple(tensor.shape))
    nz = tuple(idx for idx in product(*(range(d) for d in tensor.shape))
               if spp.is_nz(idx))
    if len(nz) == _prod(tensor.shape):
        return ('full', tuple(tensor.shape))
    return ('nz', nz)


def _prod(shape) -> int:
    out = 1
    for d in shape:
        out *= d
    return out


def _box_key(box) -> Optional[Tuple]:
    if box is None:
        return None
    return (tuple(box.lower()), tuple(box.upper()))


@dataclass(frozen=True)
class OperandKey:
    """Everything about an operand except which operand it is.

    Two tensors with equal keys may fill one hole.  The fields are compared
    one at a time when they disagree, so that a near miss can say *which*
    property blocks the merge -- a family that differs only in sparsity is a
    candidate for a common pattern, and a family that differs in addressing is
    not a family at all.
    """

    shape: Tuple[int, ...]
    addressing: object
    datatype: object
    alignment: int
    is_tmp: bool
    spp: Tuple
    box: Optional[Tuple]
    offset: Tuple[int, ...]
    sliced: bool

    FIELDS = ('shape', 'addressing', 'datatype', 'alignment', 'is_tmp',
              'spp', 'box', 'offset', 'sliced')

    def first_difference(self, other: 'OperandKey') -> Optional[str]:
        for field in self.FIELDS:
            if getattr(self, field) != getattr(other, field):
                return field
        return None


def operand_key(view) -> OperandKey:
    """The key of a view onto a tensor.

    Both halves are read: the tensor supplies shape, sparsity, addressing and
    type, the view supplies the window.  A view is what a descriptor holds, and
    two descriptors reading different windows of equally shaped tensors are not
    interchangeable even though their tensors are.
    """
    tensor = getattr(view, 'tensor', view)
    box = getattr(view, 'bbox', None)
    if box is None:
        box = getattr(tensor, 'bbox', None)
    offset = tuple(getattr(view, 'offset', ()) or ())
    return OperandKey(
        shape=tuple(tensor.shape),
        addressing=tensor.addressing,
        datatype=tensor.datatype,
        alignment=getattr(tensor, 'alignment', 0) or 0,
        is_tmp=bool(getattr(tensor, 'is_tmp', False)),
        spp=_spp_key(tensor),
        box=_box_key(box),
        offset=offset,
        sliced=bool(getattr(view, 'sliced', False)),
    )


# ---------------------------------------------------------------------------
# Reading a descriptor
# ---------------------------------------------------------------------------


def _tuple2(rows) -> Tuple:
    return tuple(tuple(r) for r in rows) if rows is not None else None


def _attrs(descr) -> Tuple:
    """The part of a descriptor that no substitution may change.

    Index maps, permutations, operators, accumulation flags: anything the
    emitter branches on that is not a tensor.  A scalar operand belongs here
    too and not among the slots, since a literal is spelled into the code and
    two different literals are two different bodies.
    """
    if isinstance(descr, MultilinearDescr):
        return ('multilinear',
                _tuple2(descr.target),
                _tuple2(descr.permute),
                bool(descr.add),
                tuple(descr.add_dims) if descr.add_dims is not None else None,
                bool(getattr(descr, 'strict_match', False)),
                bool(getattr(descr, 'prefer_align', False)))
    if isinstance(descr, ElementwiseDescr):
        scalars = tuple(descr.scalar_srcs())
        return ('elementwise', descr.op, scalars,
                bool(getattr(descr, 'strict_match', False)),
                bool(getattr(descr, 'prefer_align', False)))
    if isinstance(descr, ReductionDescr):
        return ('reduction', descr.op, tuple(descr.dims),
                bool(getattr(descr, 'prefer_align', False)))
    if isinstance(descr, BarrierDescription):
        return ('barrier', bool(descr.trueBarrier()))
    if isinstance(descr, RegionDescription):
        return ('region', descr.name)
    return ('opaque', type(descr).__name__)


def _slots(descr) -> List[Tuple[str, object]]:
    """The tensor-carrying positions of a descriptor, in a fixed order.

    Position is what a hole is stated in terms of, so the order has to be a
    property of the descriptor kind and not of how the descriptor was built.
    """
    if isinstance(descr, MultilinearDescr):
        return ([('dest', descr.dest)] +
                [(f'op{i}', op) for i, op in enumerate(descr.ops)])
    if isinstance(descr, ElementwiseDescr):
        tensors = descr.tensor_srcs()
        srcs = [(f'src{i}', s) for i, s in enumerate(descr.srcs)
                if any(s is t for t in tensors)]
        return [('dest', descr.dest)] + srcs
    if isinstance(descr, ReductionDescr):
        return [('dest', descr.dest), ('var', descr.var)]
    return []


def _tensor_of(view):
    return getattr(view, 'tensor', view)


def _identity(view, slot: int) -> object:
    """What names this operand outside its own body.

    An alias is a name the frontend chose and the same alias in two bodies is
    the same tensor, which is what makes a difference in alias a hole.  A
    temporary has no name outside the body that creates it, so it is
    identified by where it sits instead: two bodies with a temporary in the
    same position hold the same thing, and a temporary is never a hole.
    """
    tensor = _tensor_of(view)
    alias = getattr(tensor, 'alias', None) or getattr(tensor, 'name', None)
    if alias is None:
        return ('#anon', slot)
    if getattr(tensor, 'is_tmp', False):
        return ('#tmp', slot)
    return ('#named', alias)


# ---------------------------------------------------------------------------
# Skeletons
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Skeleton:
    """One body reduced to what a substitution cannot change.

    ``shape`` is the descriptor sequence with its non-tensor attributes and its
    slot count; ``keys`` is one :class:`OperandKey` per slot across the whole
    body; ``sharing`` maps each slot to the first slot naming the same tensor,
    so that repeated operands are visible.

    The partition spans the body and not one descriptor: a temporary written by
    the first operation and read by the second is a sharing relation between
    two descriptors, and a per-descriptor partition would not see it.
    """

    shape: Tuple
    keys: Tuple[OperandKey, ...]
    sharing: Tuple[int, ...]

    def groups(self) -> List[List[int]]:
        """Slots gathered by the tensor they name, in order of first use."""
        out: Dict[int, List[int]] = {}
        for slot, rep in enumerate(self.sharing):
            out.setdefault(rep, []).append(slot)
        return [out[rep] for rep in sorted(out)]


def skeleton(descrs: Sequence[OperationDescription]) -> Tuple[Skeleton, List]:
    """Reduce a descriptor list to its skeleton and the tensors it binds.

    Returns the skeleton and, beside it, the view each slot holds, which is
    what a caller needs to read the binding of a hole back out.
    """
    shape: List = []
    keys: List[OperandKey] = []
    views: List = []
    identities: List = []
    for descr in descrs:
        slots = _slots(descr)
        shape.append((_attrs(descr), tuple(role for role, _ in slots)))
        for role, view in slots:
            slot = len(keys)
            keys.append(operand_key(view))
            views.append(view)
            identities.append(_identity(view, slot))

    first: Dict[object, int] = {}
    sharing: List[int] = []
    for slot, ident in enumerate(identities):
        sharing.append(first.setdefault(ident, slot))

    return Skeleton(tuple(shape), tuple(keys), tuple(sharing)), views


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Mismatch:
    """Why two bodies are not one body.

    ``where`` is machine-readable and stable; ``detail`` is for a person.  The
    distinction between a mismatch that a change of representation could remove
    -- ``'operand.spp'``, ``'operand.alignment'`` -- and one that could not --
    ``'length'``, ``'attrs'``, ``'sharing'`` -- is the whole reason the reason
    is reported at all rather than a bare ``None``.
    """

    where: str
    detail: str
    body: Optional[int] = None
    position: Optional[int] = None

    def __bool__(self) -> bool:
        return False


@dataclass(frozen=True)
class Generalization:
    """The common body, and the tensors each input puts in its holes.

    ``holes`` are slot groups, in the order the slots first appear.  ``bindings``
    has one entry per input body, each a tuple parallel to ``holes``.  A body
    with no holes is a body that was already the same in all inputs, which is
    the answer CSE would have given.
    """

    skeleton: Skeleton
    holes: Tuple[Tuple[int, ...], ...]
    bindings: Tuple[Tuple[object, ...], ...]
    template: Tuple[OperationDescription, ...] = ()
    template_views: Tuple[object, ...] = ()

    def __bool__(self) -> bool:
        return True

    @property
    def arity(self) -> int:
        return len(self.holes)

    def binding_names(self) -> Tuple[Tuple[Optional[str], ...], ...]:
        """The alias of each hole's tensor, per body -- for messages and tests."""
        return tuple(tuple(getattr(_tensor_of(v), 'alias', None) for v in row)
                     for row in self.bindings)


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------


def _compare(base: Skeleton, other: Skeleton, index: int) -> Optional[Mismatch]:
    if len(base.shape) != len(other.shape):
        return Mismatch('length',
                        f'body {index} has {len(other.shape)} operations '
                        f'against {len(base.shape)}', body=index)
    for pos, (a, b) in enumerate(zip(base.shape, other.shape)):
        if a[0] != b[0]:
            return Mismatch('attrs',
                            f'operation {pos} differs: {a[0]} against {b[0]}',
                            body=index, position=pos)
        if a[1] != b[1]:
            return Mismatch('arity',
                            f'operation {pos} takes {b[1]} against {a[1]}',
                            body=index, position=pos)
    for slot, (a, b) in enumerate(zip(base.keys, other.keys)):
        field = a.first_difference(b)
        if field is not None:
            return Mismatch(f'operand.{field}',
                            f'slot {slot} differs in {field}: '
                            f'{getattr(a, field)!r} against {getattr(b, field)!r}',
                            body=index, position=slot)
    if base.sharing != other.sharing:
        for slot, (a, b) in enumerate(zip(base.sharing, other.sharing)):
            if a != b:
                return Mismatch('sharing',
                                f'slot {slot} shares with slot {b} but does '
                                f'not in the first body, where it shares with '
                                f'{a}', body=index, position=slot)
    return None


def anti_unify(bodies: Sequence[Sequence[OperationDescription]]
               ) -> Union[Generalization, Mismatch]:
    """The least general body of which every input is an instance.

    Fewer than two bodies is not an error and not a special case: one body
    generalises to itself with no holes, which is the identity the callers of
    this want when a family turns out to have one member.
    """
    if not bodies:
        return Mismatch('empty', 'no bodies given')

    base, base_views = skeleton(bodies[0])
    all_views = [base_views]
    for index, body in enumerate(bodies[1:], start=1):
        other, views = skeleton(body)
        problem = _compare(base, other, index)
        if problem is not None:
            return problem
        all_views.append(views)

    holes: List[Tuple[int, ...]] = []
    for group in base.groups():
        slot = group[0]
        names = {_identity(views[slot], slot) for views in all_views}
        if len(names) > 1:
            holes.append(tuple(group))

    bindings = tuple(tuple(views[group[0]] for group in holes)
                     for views in all_views)
    return Generalization(base, tuple(holes), bindings,
                          tuple(bodies[0]), tuple(base_views))


# ---------------------------------------------------------------------------
# Putting a binding back in
# ---------------------------------------------------------------------------


def rebuild(descr: OperationDescription,
            views: Sequence[object]) -> OperationDescription:
    """A descriptor of the same kind with the given views in its slots.

    The inverse of :func:`_slots`, and the reason that function fixes an order
    rather than reporting whatever order the descriptor happens to store.  A
    descriptor with no slots is returned unchanged, since there is nothing in
    it a substitution could reach.
    """
    slots = _slots(descr)
    if not slots:
        return descr
    if len(views) != len(slots):
        raise ValueError(f'{type(descr).__name__} takes {len(slots)} operands, '
                         f'got {len(views)}')

    if isinstance(descr, MultilinearDescr):
        add = descr.add_dims if descr.add_dims is not None else descr.add
        return MultilinearDescr(views[0], list(views[1:]),
                                descr.target, descr.permute, add,
                                getattr(descr, 'strict_match', False),
                                getattr(descr, 'prefer_align', False))

    if isinstance(descr, ElementwiseDescr):
        tensors = descr.tensor_srcs()
        rest = list(views[1:])
        srcs = [rest.pop(0) if any(s is t for t in tensors) else s
                for s in descr.srcs]
        return ElementwiseDescr(descr.op, views[0], srcs,
                                getattr(descr, 'strict_match', False),
                                getattr(descr, 'prefer_align', False))

    if isinstance(descr, ReductionDescr):
        return ReductionDescr(views[0], views[1], descr.dims, descr.op,
                              getattr(descr, 'prefer_align', False))

    raise ValueError(f'no way to rebuild a {type(descr).__name__}')


def substitute(general: Generalization,
               bindings: Sequence[object]) -> List[OperationDescription]:
    """The generalised body with one tensor put in each hole.

    Binding every hole to a constant is what turns the common body back into a
    particular one, and the round trip -- generalise a family, bind hole by
    hole, get each member back -- is the property that says the generalisation
    kept everything it had to.  It is also the specialised form itself: a hole
    bound to a literal is a hole the emitter never sees.

    A hole covers every slot naming the same tensor, so one binding reaches all
    of them; that is what keeps a tensor used twice used twice.
    """
    if len(bindings) != len(general.holes):
        raise ValueError(f'{len(general.holes)} hole(s) to fill, '
                         f'got {len(bindings)}')

    filled = list(general.template_views)
    for group, view in zip(general.holes, bindings):
        for slot in group:
            filled[slot] = view

    out: List[OperationDescription] = []
    cursor = 0
    for descr in general.template:
        width = len(_slots(descr))
        out.append(rebuild(descr, filled[cursor:cursor + width]))
        cursor += width
    return out


def instantiate(general: Generalization, member: int
                ) -> List[OperationDescription]:
    """The generalised body bound as the ``member``-th input had it."""
    return substitute(general, general.bindings[member])
