# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Buffers a section creates for itself, and their names.

Two kinds: a shared-memory symbol for a temporary the kernel produces, and a
register array to compute a result into before it goes anywhere.  Both used to
be made inside `MultilinearBuilder`, which is why nothing else could produce a
temporary -- an elementwise destination that no contraction also wrote had no
symbol at all, and the `SymbolView` built over it wrapped `None`.

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

from typing import List, Optional, Tuple

from tensorforge.backend.data_types import RegMemObject
from tensorforge.backend.instructions.abstract_instruction import _explicit_simd
from tensorforge.backend.instructions.allocate import RegisterAlloc
from tensorforge.backend.symbol import DataView, Symbol, SymbolType
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

    def register_array(self, bbox: BoundingBox, lead_pos: int,
                       shift: int = 0,
                       spp=None) -> Tuple[Symbol, RegisterAlloc]:
        """An array holding `bbox`, with axis `lead_pos` spread over the lanes.

        `shift` moves the lane axis' origin: the multilinear accumulator is
        indexed in the theta-shifted space, and straddling one more block
        boundary is the price of not needing a shuffle.  Everything else
        indexes at origin 0 and leaves it alone.
        """
        regsize = 1
        threads = self._num_threads
        for d in range(bbox.rank()):
            dim = bbox.size(d)
            if d != lead_pos or threads == 0:
                regsize *= dim
            else:
                r_start = (bbox.lower()[d] + shift) // threads
                r_end = (bbox.upper()[d] + shift + threads - 1) // threads
                regsize *= (r_end - r_start) * DataView.lead_lanes(
                    None, _explicit_simd(self._context), threads)
                threads //= dim  # TODO?

        name = self.next_register_name()
        registers = Symbol(name=name, stype=SymbolType.Register,
                           obj=RegMemObject(name, regsize, spp=spp))
        registers.lead_dims = [lead_pos]
        registers.num_threads = self._num_threads
        registers.datatype = self._context.fp_type
        self._scopes.add_symbol(registers)
        return registers, RegisterAlloc(self._context, registers, regsize, 0.0)
