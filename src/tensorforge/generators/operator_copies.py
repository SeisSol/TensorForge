# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Copies of a batch-constant operator for the readers on the matrix path
(`Options.mma_operator_copies`).

The matrix path reads its A operand best in the order its fragments come in:
stored so, a lane's fragments of a tile are one wide read and the warp's are
adjacent (`nvidia.fragment_order`).  That order is the reader's -- its
window, its tiles -- so an operator read by several operations is offered
none (`MultilinearInstruction._offer_order`), and each matrix product then
stages it through shared memory element by element.  SeisSol's derivative
reads each `kDivMT` once per order, over windows 35 x 54, 20 x 33, 10 x 18
at order 6: every one of them a matrix product with nothing prepared.

Where the description carries the operator's numbers, each such reader gets
a copy of its own (`Residence.MODULE`): a tensor with the operator's shape,
box and values, which the generated file defines as a table and the kernel
reads through an address into it.  It has one reader, so it is offered that
reader's order like any operator read once; the table holds the values in
that order (`Tensor.storage_values`), and the caller never sees it.  The
readers that stay off the matrix path -- the derivative's last orders, whose
contractions are too short -- keep the operator as it is passed, and share
its cache lines as before.

Which readers are the matrix path's is decided here from the descriptions,
by what the path asks of a product (`nvidia.min_depth`): an A operand on the
lead axis and a contraction, at least that deep.  A copy for a reader that
then does not take the path is a copy nobody needed, read the way the
original would have been -- costing its bytes, never a wrong number.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Tuple

import numpy as np

from tensorforge.common.basic_types import Addressing, Residence


#: The storage fields of a tensor as it is constructed
#: (`generator._STORAGE_FIELDS`).
_UNSTORED = {'storage_order': None, 'packing': None, 'storage_parts': 1,
             'storage_planar': False, 'simt_interleave': None,
             'slot_major': None, 'packing_table': None,
             'packing_table_param': True}


def _views(descr):
    """Every view `descr` reads, its guard's included."""
    out = []
    for name in ('reads', 'condition_reads'):
        reader = getattr(descr, name, None)
        if reader is not None:
            out.extend(reader())
    return out


def _takes_the_matrix_path(descr, view, depth: int) -> bool:
    """Whether `descr` reads `view` as the A operand of a product the matrix
    path may take: its first operand, its first axis the destination's lead
    and its second the contraction, at least `depth` steps of it."""
    ops = getattr(descr, 'ops', None)
    target = getattr(descr, 'target', None)
    if not ops or len(ops) < 2 or ops[0] is not view or not target:
        return False
    # A factor is a third operand without axes (`GemmDescr`'s alpha); a
    # third tensor is a product the path does not take.
    if any(len(getattr(op.tensor, 'shape', ()) or ()) != 0 for op in ops[2:]):
        return False
    axes = target[0]
    if len(axes) != 2 or axes[0] != 0 or axes[1] >= 0:
        return False
    if list(getattr(descr, 'permute', [[0, 1]])[0]) != [0, 1]:
        return False
    box = view.bbox
    return box.upper()[1] - box.lower()[1] >= depth and box.upper()[0] > box.lower()[0]


def _whole(view) -> bool:
    """Whether `view` reads its operator's whole box."""
    box, own = view.tensor.get_bbox(), view.storage_box()
    return (list(box.lower()) == list(own.lower())
            and list(box.upper()) == list(own.upper()))


def _copyable(tensor) -> bool:
    return (getattr(tensor, 'addressing', None) is Addressing.NONE
            and getattr(tensor, 'residence', Residence.MEMORY) is Residence.MEMORY
            and not getattr(tensor, 'is_tmp', False)
            and not getattr(tensor, 'is_variant', False)
            and tensor.has_values()
            and len(tensor.shape) == 2)


def confine(tensor, order) -> None:
    """Zero the cells of copy `tensor` that `order` leaves without a slot.

    A copy has one reader, and an order over its window says which cells it
    reads; the others are read by nobody, and an order may leave a cell out
    only where the numbers say it is zero (`Tensor.storage_order`).  So the
    numbers say it.  The pattern derived from them is derived again
    (`Tensor.nonzero_pattern`): inside the window it is what it was."""
    kept = {int(c) for c in order if int(c) >= 0}
    box = tuple(int(extent) for extent in tensor.get_actual_shape())
    lower = tuple(int(bound) for bound in tensor.bbox.lower())
    data = np.array(tensor.data, copy=True)
    for cell in range(int(np.prod(box))):
        if cell in kept:
            continue
        local = np.unravel_index(cell, box, order='F')
        data[tuple(lo + c for lo, c in zip(lower, local))] = 0
    tensor.data = data
    tensor._nonzero_pattern = None


def copy_operators(descrs: List, depth: int, prepared: bool = False
                   ) -> Tuple[List, List]:
    """`descrs` with each matrix-path reader of a batch-constant operator
    given a copy of it, and the copies.

    Where other operations read the operator too, since then none is offered
    an order.  Where this is its one reader, wherever the order would not be
    offered or not taken: over a window narrower than the operator, which is
    ordered only where the caller asks by name (`_offer_order`), or where
    the caller prepares no operands (`Options.prepare_operands`) and would
    not lay one out.  A run's members stay: `roll` has made them a run
    before, and the table that selects one per iteration is filled by the
    host.

    One copy per reader window: two readers of the same window share one, as
    they would share one order."""
    readers: Dict[int, set] = {}
    for n, descr in enumerate(descrs):
        for view in _views(descr):
            tensor = getattr(view, 'tensor', None)
            if tensor is not None:
                readers.setdefault(id(tensor), set()).add(n)
    copies: Dict[Tuple, object] = {}
    out = []
    for n, descr in enumerate(descrs):
        ops = getattr(descr, 'ops', None)
        if not ops:
            out.append(descr)
            continue
        view = ops[0]
        tensor = getattr(view, 'tensor', None)
        alone = len(readers.get(id(tensor), ())) < 2
        if (tensor is None or not _copyable(tensor)
                or (alone and prepared and _whole(view))
                or not _takes_the_matrix_path(descr, view, depth)):
            out.append(descr)
            continue
        key = (id(tensor), tuple(view.bbox.lower()), tuple(view.bbox.upper()),
               tuple(getattr(view, 'offset', ())))
        clone = copies.get(key)
        if clone is None:
            clone = copy.copy(tensor)
            clone.residence = Residence.MODULE
            clone.alias = f'{tensor.alias}@{len(copies)}'
            # Stored as a tensor nobody has built for yet: the original may
            # carry what an earlier build of the caller's list made of it.
            for field, value in _UNSTORED.items():
                setattr(clone, field, value)
            clone._nonzero_pattern = None
            copies[key] = clone
        new_view = copy.copy(view)
        new_view.tensor = clone
        new_descr = copy.copy(descr)
        new_descr.ops = [new_view] + list(ops[1:])
        out.append(new_descr)
    return out, list(copies.values())
