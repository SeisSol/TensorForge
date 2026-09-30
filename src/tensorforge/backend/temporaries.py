# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Buffers a section creates for itself, and their names.

Two kinds: a shared-memory symbol for a temporary the kernel produces, and a
register array to compute a result into before it goes anywhere.  Both are
made here rather than inside `MultilinearBuilder`: if only the contraction
builder could produce a temporary, an elementwise destination that no
contraction also writes would have no symbol at all, and the `SymbolView` built
over it would wrap `None`.

The counters live here for the same reason the record does: `s0`, `s1`, `r0`
have to be unique across a section, and two producers with a counter each would
collide the moment a second one appeared.

A shared symbol is created and registered but nothing is emitted for it.  That
is not an omission.  `ShrMemOpt` sizes each buffer from its *first user*, which
it requires to be a memory instruction that can report a size, so a shared
buffer has to be introduced by a store or a load.  A result computed into
registers and settled later gets exactly that -- the flush emits the store --
while a compute instruction writing shared memory directly would be a first
user that cannot answer the question the pass asks it.
"""

from typing import Tuple

from tensorforge.backend.data_types import RegMemObject
from tensorforge.backend.instructions.abstract_instruction import _explicit_simd
from tensorforge.backend.instructions.allocate import RegisterAlloc
from tensorforge.backend.pir.core import LaneAxis
from tensorforge.backend.symbol import slots_for, DataView, Symbol, SymbolType
from tensorforge.common.exceptions import InternalError
from tensorforge.common.matrix.boundingbox import BoundingBox


class Temporaries:
    """Names and symbols for one section's own buffers."""

    def __init__(self, context, scopes, num_threads: int):
        self._context = context
        self._scopes = scopes
        self._num_threads = num_threads
        self._shared_counter = 0
        self._register_counter = 0

    # -- names ----------------------------------------------------------- #

    def next_shared_name(self) -> str:
        name = f's{self._shared_counter}'
        self._shared_counter += 1
        return name

    #: How many adjacent lead-dimension elements one lane of a register image
    #: holds.  Set by whoever knows the compute arrangement; see
    #: `Symbol.lead_width`.
    _lead_width: int = 1

    def next_register_name(self) -> str:
        name = f'r{self._register_counter}'
        self._register_counter += 1
        return name

    # -- buffers --------------------------------------------------------- #

    def shared_symbol(self, tensor) -> Symbol:
        """The shared-memory symbol for `tensor`, created if it has none.

        Only for temporaries: anything the caller passes in stays live for the
        section, and a tensor that is a kernel argument already has a symbol.
        """
        existing = self._scopes.get_symbol(tensor)
        if existing is not None:
            return existing
        if not getattr(tensor, 'is_tmp', False):
            raise InternalError(
                f'{tensor}: not a temporary and not in scope, so there is '
                f'nothing to give it a buffer from')
        symbol = Symbol(name=self.next_shared_name(),
                        stype=SymbolType.SharedMem,
                        obj=tensor)
        self._scopes.add_symbol(symbol)
        return symbol

    @staticmethod
    def _strides(blocks):
        """Lane strides for a run of blocks, innermost first."""
        stride, out = 1, []
        for block in blocks:
            out.append(stride)
            stride *= block
        return out

    def _lead_axes(self, lead):
        """`lead` as an ordered `{dimension: block}`, whichever form it came in.

        A bare dimension index means the one-axis image every caller asks for
        today: the whole wave, cyclic, which is what `lead_dims` alone means
        and what `Symbol.lead_block` answers when no axes are written down.

        Refuses blocks that do not tile the wave.  They would leave lanes
        holding copies, `RegisterLayout.tiles` would then refuse the layout,
        `Symbol.lead_block` would fall back to the wave -- and the addressing
        would divide by a number this allocation did not size in.  Better to
        say so here, where the caller that chose the blocks can hear it, than
        to alias two dimensions onto each other and let the answer be wrong.
        """
        if isinstance(lead, int):
            return {lead: self._num_threads}
        axes = {}
        for dim, block in lead:
            if dim in axes:
                raise InternalError(
                    f'dimension {dim} given two lane axes')
            axes[dim] = block
        product = 1
        for block in axes.values():
            product *= block
        if self._num_threads and self._num_threads % product != 0:
            raise InternalError(
                f'lane blocks {list(axes.values())} multiply to {product}, '
                f'which does not divide a {self._num_threads}-lane wave; the '
                f'lanes would hold overlapping parts of the image rather than '
                f'whole copies of it')
        # A product *smaller* than the wave is allowed and means replication:
        # every run of `product` lanes holds one whole copy.  That is what a
        # multiplication spanning several sub-groups needs -- a broadcast
        # reaches only within one, so the image it reads has to be in each of
        # them (`Symbol.reading_lane`).  It costs a copy per run and buys the
        # register image being divided by the lanes that share a register
        # file rather than by the lanes of the whole multiplication.  What it
        # does not give is a single owner, which `owning_lane` answers `None`
        # to, so a store still refuses to guard itself to one of several.
        return axes

    def register_array(self, bbox: BoundingBox, lead,
                       shift: int = 0,
                       spp=None) -> Tuple[Symbol, RegisterAlloc]:
        """An array holding `bbox`, with the named axes spread over the lanes.

        `lead` is a dimension index for the one-axis image everything asks for
        today, or a sequence of `(dimension, block)` pairs for one that is
        spread over more than one.  The pairs are in lane order, innermost
        first: the first gets stride 1, and each later one a stride of the
        product of the blocks before it, so lane `t` holds
        ``(t % b0, (t // b0) % b1, ...)``.  That is the nesting, and it is not
        the dimension order -- a producer that wants a row in the high lane
        bits and a column pair in the low ones lists the column first.

        Sized and stated together, which is the point of the parameter.  The
        addressing divides a coordinate on each of these dimensions by that
        dimension's block (`Symbol.lead_block`), and an allocation in units
        the addressing does not divide by aliases the next dimension onto this
        one.  Passing the blocks here and writing them onto the symbol is what
        makes those the same number by construction rather than by both
        defaulting to the wave.

        `shift` moves the *first* axis' origin: the multilinear accumulator is
        indexed in the theta-shifted space, and straddling one more block
        boundary is the price of not needing a shuffle.  Everything else
        indexes at origin 0 and leaves it alone.
        """
        # A box without axes is one value, the same on every lane: nothing is
        # spread, whatever the caller named as its lead.
        axes = {} if bbox.rank() == 0 else self._lead_axes(lead)
        regsize = 1
        for d in range(bbox.rank()):
            block = axes.get(d)
            if block is None or self._num_threads == 0:
                regsize *= bbox.size(d)
            else:
                # The same rule addressing uses, called rather than restated:
                # a restatement without the width would size a three-slot
                # image for a four-wide read, and consecutive non-lead indices
                # would address overlapping windows.
                origin = shift if d == next(iter(axes)) else 0
                regsize *= slots_for(
                    bbox.lower()[d] + origin, bbox.upper()[d] + origin,
                    block, getattr(self, '_lead_width', 1))
        if axes and self._num_threads:
            # Once, not once per axis.  This is how many register entries one
            # slot occupies, which is a question about the lowering -- under
            # ESIMD the work-item holds the whole wave, so a slot is a run of
            # that many entries however many axes share the wave between them.
            regsize *= DataView.lead_lanes(
                None, _explicit_simd(self._context), self._num_threads)

        name = self.next_register_name()
        registers = Symbol(name=name, stype=SymbolType.Register,
                           obj=RegMemObject(name, regsize, spp=spp))
        registers.lead_dims = list(axes)
        registers.num_threads = self._num_threads
        # `None` means "one axis over the whole wave", which is the default
        # every reader assumes.  A single axis whose block is *narrower* than
        # the wave is not that: the lanes then hold copies, and a reader that
        # fell back to the wave would divide by the wrong number.  So the axes
        # are written down whenever they say something the default does not.
        default = (len(axes) == 1
                   and next(iter(axes.values())) == self._num_threads)
        registers.lead_axes = None if default else tuple(
            LaneAxis(block, stride) for block, stride in
            zip(axes.values(), self._strides(axes.values())))
        # The blocking of this image, set once here so every access resolves
        # positions the same way -- the loops that walk it, and the
        # fixed-element reads that go through the broadcast path.
        registers.lead_width = getattr(self, '_lead_width', 1)
        registers.datatype = self._context.fp_type
        self._scopes.add_symbol(registers)
        return registers, RegisterAlloc(self._context, registers, regsize, 0.0)
