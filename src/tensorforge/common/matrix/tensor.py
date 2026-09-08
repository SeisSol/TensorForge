# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
from .spp import SparsityPattern, FullSPP
from .boundingbox import BoundingBox
from functools import reduce
from typing import List, Union

import numpy as np
from ..basic_types import Addressing, DataFlowDirection, Datatype
from tensorforge.common.exceptions import GenerationError

class Tensor:
    def __init__(self,
        shape: List[int],
        addressing: Addressing,
        bbox: Union[List[int], None]=None,
        alias: Union[str, None]=None,
        is_tmp: bool = False,
        spp: SparsityPattern = None,
        data: Union[np.ndarray, dict, None] = None,
        datatype: Datatype = None,
        alignment: int = 0):
        self.name = None
        self.alias = alias
        self.shape = tuple(shape)
        self.is_tmp = is_tmp
        #: A name a loop body uses where an operand varies between iterations.
        #:
        #: A third kind beside a parameter and a temporary, and it has to be
        #: one: it is not passed in, because it stands for several things that
        #: are; and it is not the generator's own scratch, because it is never
        #: written.  It resolves inside the loop to whichever member the
        #: counter names, so the signature leaves it out and the body binds it.
        self.is_variant = False
        #: Scalars one *logical* element occupies in memory.
        #:
        #: One for every tensor the frontend describes, and that is the whole
        #: of it as far as a frontend is concerned: this is a codegen
        #: decision, not a description of the operation.  The generator raises
        #: it when it decides to store an operand already prepared -- a value
        #: split into the halves a matrix instruction multiplies, say -- so
        #: that the kernel reads the parts instead of computing them.
        #:
        #: It is a count and not a flag because the schemes differ in it: two
        #: parts for a TF32 split, three for a BF16 one.  What the parts *mean*
        #: is the preparing function's business and not this attribute's; this
        #: one only says how much room they take, which is what an address
        #: needs to know.
        self.storage_parts = 1
        self.direction: Union[DataFlowDirection, None] = None
        self.data = data
        self.spp = spp
        self.datatype = datatype
        self.alignment = alignment

        if self.spp is None:
            self.spp = FullSPP(self.shape)

        if bbox is not None:
            self.bbox = bbox
        else:
            self.bbox = BoundingBox([0] * len(shape), shape)

        self.addressing = addressing
        self.ptr_type = self.addressing.to_pointer()

        if self.addressing == Addressing.SCALAR:
            # allow higher-order tensors, if they're effectively a scalar anyways
            assert all(d == 1 for d in self.shape)

        # `value()` indexes this by coordinate tuple, which a list answers with
        # a TypeError.  That went unnoticed for as long as `value()` asked
        # `realindex in self.data` first: on a list that tests the *elements*,
        # which a coordinate tuple never matches, so every lookup fell through
        # to `None` and the ill-typed access was never reached.  Asking the
        # sparsity pattern instead -- the right question -- reaches it.
        #
        # Checked rather than coerced.  A `np.asarray` here would accept the
        # callers that still hand over a list and leave them unfixed, which is
        # how the requirement got two homes in the first place.
        if self.data is not None:
            if isinstance(self.data, dict):
                # TODO: proper dtype
                data = np.zeros(self.shape, dtype=np.float64)
                for pos, value in self.data.items():
                    data[pos] = value
                self.data = data
            if not isinstance(self.data, np.ndarray):
                raise GenerationError(
                    f'Tensor {self}: data must be an ndarray of shape '
                    f'{self.shape}, got {type(self.data).__name__}')
            if self.data.shape != self.shape:
                raise GenerationError(
                    f'Tensor {self}: data has shape {self.data.shape}, '
                    f'tensor is {self.shape}')

        # check whether bbox was given correctly
        if any(dimshape < dimsize for dimshape, dimsize in zip(self.shape, self.bbox.sizes())):
            raise GenerationError(f'Tensor {self} is smaller than bounding box {self.bbox}')

        if any(dimshape < dimupper for dimshape, dimupper in zip(self.shape, self.bbox.upper())):
            raise GenerationError(f'Bounding box {self.bbox} is smaller than tensor {self}')

    def set_data_flow_direction(self, direction: DataFlowDirection):
        if self.direction is None or self.direction == direction:
            self.direction = direction
        else:
            self.direction = DataFlowDirection.SOURCESINK

    def has_values(self):
        return self.data is not None

    def get_values(self):
        return self.data

    def value(self, index):
        realindex = tuple(index)
        if self.spp.is_nz(realindex):
            return self.data[realindex]
        else:
            return None

    def linear_index(self, index):
        realindex = tuple(index)
        return self.spp.linear_index(realindex)

    def memory(self):
        return self.spp.count_nz()

    def storage_volume(self):
        """Scalars one batch element of this tensor occupies in memory.

        The one place the storage convention is decided, because it was
        previously decided twice and differently: the batch stride came from
        the bounding box while the staging loop copied ``count_nz`` cells, so
        a masked tensor was written densely by the host and read compressed by
        the kernel.

        A dense tensor is stored over its bounding box -- address zero is the
        box's lower corner and the buffer spans upper minus lower.  A sparse
        one is stored compressed, in the order ``linear_index`` assigns, and
        nothing is reserved for the structural zeros.  A bounding box is the
        same under either reading, which is why it needs no case of its own.
        """
        base = self.get_actual_volume() if self.is_dense() else self.memory()
        # `storage_parts` multiplies whichever of the two readings applies:
        # preparing an operand does not change which cells are stored, only
        # how many scalars each of them takes.
        return base * self.storage_parts

    def storage_map(self):
        """Which cell of the bounding box each storage slot holds.

        ``None`` when the tensor is stored dense, because then the two orders
        are the same thing and a map would only be a way to disagree with
        itself.  Otherwise a tuple of length ``storage_volume``, indexed by
        slot and giving the F-order position within the bounding box.

        Read off ``linear_index``, so everything that has to agree about the
        order -- the kernel, the test harness, the host oracle -- agrees by
        construction rather than by three parallel derivations.
        """
        if self.is_dense():
            return None
        # `linear_index` speaks full-tensor coordinates; the dense view a
        # caller compares against spans the bounding box.  So the slot is
        # asked of the one and the cell reported in the other, and the box's
        # lower corner is what separates them.
        box = tuple(self.get_actual_shape())
        lower = tuple(self.bbox.lower())
        strides, acc = [], 1
        for extent in box:
            strides.append(acc)
            acc *= extent
        slots = [-1] * int(self.storage_volume())
        for idx in _f_order(tuple(self.get_real_shape())):
            if not self.spp.is_nz(idx):
                continue
            cell = tuple(i - lo for i, lo in zip(idx, lower))
            if any(c < 0 or c >= extent for c, extent in zip(cell, box)):
                raise ValueError(
                    f'{self.alias!r}: a non-zero at {idx} lies outside the '
                    f'bounding box that is supposed to contain them')
            slots[self.linear_index(idx)] = sum(c * s for c, s
                                                in zip(cell, strides))
        if any(slot < 0 for slot in slots):
            raise ValueError(
                f'{self.alias!r}: linear_index left storage slots unassigned; '
                f'the pattern and the index map disagree')
        return tuple(slots)

    def storage_runs(self):
        """The stored cells as `(slot, cell, length)` runs, or `None`.

        A run is a stretch that is contiguous in both the compressed order
        and the bounding box at once, so copying one is a block copy with two
        constant bases and no per-element index.  That is what lets a sparse
        tensor be expanded into a dense image without an index table: the run
        list is the table, and it is spent at code-generation time.

        Worth it only when there are few runs.  The corpus splits sharply --
        `rDivM(2)` at order 8 is 2866 non-zeros in 62 runs, while `kDivM(0)`
        is 1446 in 1446, one per element -- and the second kind is cheaper
        staged dense in the first place.
        """
        pack = self.storage_map()
        if pack is None:
            return None
        runs = []
        start = 0
        for slot in range(1, len(pack) + 1):
            if (slot < len(pack)
                    and pack[slot] == pack[slot - 1] + 1):
                continue
            runs.append((start, pack[start], slot - start))
            start = slot
        return tuple(runs)

    def densified(self):
        """The same tensor with nothing left out, or `None` if already dense.

        Its bounding box, shape and type are this one's; only the pattern
        differs.  Somewhere to expand into: a consumer reading the image asks
        `is_dense()` and gets the answer that is true of the image rather than
        the one that is true of where it came from.
        """
        if self.is_dense():
            return None
        twin = Tensor(shape=self.shape, addressing=self.addressing,
                      bbox=self.bbox, alias=self.alias, is_tmp=self.is_tmp,
                      spp=None, data=self.data, datatype=self.datatype,
                      alignment=self.alignment)
        twin.name = self.name
        twin.direction = self.direction
        return twin

    def get_actual_shape(self):
        return self.bbox.sizes()

    def get_actual_volume(self):
        return reduce(lambda x,y:x*y, self.get_actual_shape(), 1)

    def get_real_shape(self):
        return self.shape

    def get_real_volume(self):
        return reduce(lambda x,y:x*y, self.get_real_shape(), 1)

    def get_offset_to_first_element(self):
        return '0' # self.bbox.first_element()

    def get_bbox(self):
        return self.bbox

    def _set_name(self, name):
        self.name = name

    def is_similar(self, other):
        is_similar = self.shape == other.shape
        is_similar &= self.addressing == other.addressing
        is_similar &= self.bbox == other.bbox
        return is_similar

    def is_same(self, other):
        return self.is_similar(other) # and self.alias == other.alias and self.is_tmp == other.is_tmp

    def __str__(self):
        return self.name

    def gen_descr(self):
        return f'{self.name} {"×".join(str(d) for d in self.shape)}({"×".join(str(d) for d in self.bbox.sizes())}) {self.bbox} {self.addressing}'

    def density(self):
        return self.spp.count_nz() / self.get_real_volume()

    def sparsity(self):
        return 1 - self.density()

    def is_dense(self):
        return self.spp.count_nz() == self.get_real_volume()

    def __str__(self):
        return self.gen_descr()

    def __repr__(self):
        return self.gen_descr()

def _f_order(shape):
    """Every index of ``shape``, first axis fastest."""
    if not shape:
        yield ()
        return
    for rest in _f_order(shape[1:]):
        for i in range(shape[0]):
            yield (i,) + rest


class TensorWrapper:
    pass

class SubTensor(TensorWrapper):
    def __init__(self,
        tensor: Tensor,
        bbox: Union[BoundingBox, None] = None,
        offset: Union[list[int], None] = None,
        sliced: bool = False):
        self.tensor = tensor
        self.bbox = bbox
        if bbox is None:
            self.bbox = self.tensor.bbox
        self.offset = offset or ([0] * self.bbox.rank())
        # Whether this names a *slice* of the tensor rather than the tensor
        # itself.  The two look alike -- both carry a box narrower than the
        # tensor's -- but they mean opposite things when the box is written:
        # a narrow box on the tensor itself is an eqspp window, so everything
        # outside it is zero and a write has to say so; a slice owns only what
        # it names.  A nonzero offset gives it away, but a slice starting at
        # index 0 has none, so the frontend states it instead of leaving the
        # backend to guess.
        self.sliced = sliced or any(o != 0 for o in self.offset)

    def storage_box(self) -> BoundingBox:
        """This view's box in the *tensor's* coordinates.

        A view states a box in its own index space plus the offset at which
        that space sits in the tensor.  Anything comparing two views of one
        tensor -- a union of reads, a coverage test -- has to do it in the
        tensor's coordinates, and doing the addition at each site is how two
        of them come to disagree about whether the offset is already included.
        """
        return BoundingBox(
            [l + o for l, o in zip(self.bbox.lower(), self.offset)],
            [u + o for u, o in zip(self.bbox.upper(), self.offset)])

    def __str__(self):
        return f'{self.tensor}({self.bbox})'

    def __repr__(self):
        return f'{self.tensor}({self.bbox})'

class FullTensor(TensorWrapper):
    def __init__(self, tensor: Tensor):
        self.tensor = tensor
