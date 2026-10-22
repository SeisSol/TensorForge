# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What a multiplication's register image holds past the rows it computed.

A register image spreads its lead dimension over the lanes, a round of them
at a time, so its last slot holds lanes that are none of its rows: past a
window that ends inside a round.  A multiplication without an epilogue that
computes that slot on every lane (`LeadLoop.full_lane`) writes those lanes
with what it accumulates there; where every operand that carries the lead
index reads a zero there -- memory, its accesses held to the window
(`LeadIndex.valid`), or a register image whose own rows end there -- every
product has a zero factor, and what it accumulates is a zero.

Nothing in the body shows it -- the arithmetic runs through cross-lane
primitives no pass looks into -- so the multiplication says so
(`IRBuilder.promise_zero`, `MultilinearBuilder._promised_zeros`): a
statement that emits nothing, and that the flow takes for a store of the
zero (`pir.flow`).  A load back from memory, which gives those lanes the
zero of its predicate, is then taken from the image on every lane, and no
select tells the lanes apart (`pir.forward`).
"""

from tensorforge.backend.pir.core import BOOL
from tensorforge.backend.symbol import (LeadLoop, Loop, SymbolType,
                                        write_loops)

from . import ComputeInstruction


class PromisedZeros(ComputeInstruction):
    """`image`, a register image, holds a zero in every lane past `rows` of
    its lead dimension, in the slot that holds the last of them.  Emits
    nothing (`IRBuilder.promise_zero`).

    `rows` is a range of the lead dimension in the image's own coordinates,
    those its writer indexes it in.  At one element a lane only: a lane of a
    wider slot can hold a row of the window and one outside it at once."""

    def __init__(self, context, image, rows, num_threads: int,
                 lead_width: int):
        super().__init__(context)
        self._image = image
        self._rows = tuple(rows)
        self._num_threads = num_threads
        self._lead_width = lead_width
        self._is_ready = True

    def defs(self):
        return ()

    def get_operands(self):
        return [self._image]

    def gen_code_inner(self, writer) -> None:
        image = self._image
        if (not hasattr(writer, 'promise_zero') or self._lead_width != 1
                or not self._num_threads
                or image.stype not in (SymbolType.Register,
                                       SymbolType.Scratch)
                or len(image.lead_dims) != 1):
            return
        lead = image.lead_dims[0]
        box = image.data_view.get_bbox()
        lo, hi = self._rows
        if lo >= hi:
            return
        loops = []
        for d in range(box.rank()):
            if d == lead:
                loops.append(LeadLoop(f'z{d}', lo, hi, self._num_threads, 1,
                                      unroll=True, full_lane=True))
            else:
                loops.append(Loop(f'z{d}', box.lower()[d], box.upper()[d], 1,
                                  unroll=True))

        def inner(index):
            at = index[lead]
            valid = getattr(at, '_valid', None)
            if valid is None:
                return
            lane = writer.lane_index(at._block, at._stride, hint='lead')
            past = writer.op('ge', BOOL, lane, valid, hint='o')
            image.promise_zero(writer, self._context, index, past)

        write_loops(self._context, writer, loops, inner)

    def __str__(self):
        return f'{self._image.name} = 0 past {self._rows}'
