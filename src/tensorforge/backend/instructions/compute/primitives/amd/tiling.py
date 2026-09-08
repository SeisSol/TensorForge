# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Which matrix scheme emits a contraction here.

`strategy.py` splits the choice of *arrangement* into what the target allows
and what it prefers.  This is the same split one level down: there are three
emitters behind `Strategy.MATRIX` on this target, they overlap, and which one
runs was an if-chain in `matmul` -- the shape that layer above was written to
remove, and it goes wrong the same way.

A scheme is a mapping from the generator's dimensions onto an instruction's,
and the three differ in it rather than in what they compute:

* `LANE_BATCHED` is `MatrixOp.lane_batched`'s subject -- ``k == 1``, the
  instruction broadcasting its own A operand, the accumulator already where
  the nest wants it;
* `EXCHANGE` is the ``k > 1`` shape, where A comes from one transpose per
  k-block, B from a swap sequence and the result from a gather;
* `EMULATED` reaches a wide accumulator through a narrow arithmetic, one
  issue per term product of the split.

They overlap: at 64 threads on CDNA all three serve an F32 contraction, and
only `EXCHANGE` serves an F64 one.  Which is faster is a measurement, so it is
an order here rather than a condition somewhere.
"""

from dataclasses import dataclass
from enum import Enum
from typing import FrozenSet, Optional, Tuple

from ... import packing
from .catalog import emu_tile_for, mfma_tile_for
from .exchange_codegen import exchange_op


class Scheme(Enum):
    """How the generator's dimensions are laid onto an instruction's."""

    LANE_BATCHED = 'lane-batched'
    EXCHANGE = 'exchange'
    EMULATED = 'emulated'


#: Whether the emulated scheme is deployed, as opposed to whether one exists
#: for a shape -- that second question is `emu_tile_for`.
#:
#: Parked for a run.  The operand stacking is derived from `layouts.position`
#: and checked against it, and the split is the runtime's; what is left is
#: whether three BF16 products through the matrix unit beat one F32 product
#: through it, on a machine.  Unlike the DPAS case the direct path here is
#: not merely slower but already fast, so the answer could well be no.
EMULATION = False

#: Whether the `k > 1` scheme is deployed.
#:
#: Turning it on is what makes F64 take a matrix instruction at all, and it
#: changes every F64 kernel on CDNA 2 and later.  The check available without
#: a machine is that the emitted calls match the plans, which says the emitter
#: is faithful and nothing about whether the instruction then computes the
#: right thing.
EXCHANGE = False


@dataclass(frozen=True)
class Fit:
    """One scheme, ready to emit: the instruction and what the emitter needs.

    Held together because the three selections answer in three shapes -- a
    tile, a bare `MatrixOp`, a tile and a term count -- and a caller that
    unpacks each differently is a caller that has to know which it asked.
    """

    scheme: Scheme
    #: The catalogue entry that will be issued.
    op: object
    #: The tile, where the scheme carries one beside the entry; the transpose
    #: that feeds A is not a property of the instruction.
    tile: Optional[object] = None
    #: Terms the operands are split into.  1 where the arithmetic is exact,
    #: which is every scheme but `EMULATED`.
    terms: int = 1
    #: The type the operands have to arrive as.  Not `op.a.dtype` for an
    #: emulated scheme: there the emitter splits, so what arrives is what the
    #: split reads.
    reads: object = None

    @property
    def width(self) -> int:
        """Output columns one issue covers, and the unit a boundary falls on."""
        return self.op.m


def offers(threads, accumulator, ctx) -> FrozenSet[Scheme]:
    """Which schemes can serve this shape on this target.

    Legality only.  A gate that is off removes a scheme here rather than at
    the call site, so that what is available and what is preferred stay two
    questions.
    """
    out = set()
    if mfma_tile_for(threads, accumulator, ctx) is not None:
        out.add(Scheme.LANE_BATCHED)
    if EXCHANGE and exchange_op(accumulator, threads, ctx) is not None:
        out.add(Scheme.EXCHANGE)
    if EMULATION and emu_tile_for(threads, accumulator, ctx) is not None:
        out.add(Scheme.EMULATED)
    return frozenset(out)


#: The order the legal schemes are taken in.
#:
#: The gated ones rank ahead of the deployed one, which is not a claim that
#: they are faster.  It is what makes a switch mean something: a gate that
#: left the deployed path in place wherever it also fits would answer nothing
#: about the scheme it turns on.  With both off this is `LANE_BATCHED` and
#: the order is not read.
ORDER: Tuple[Scheme, ...] = (Scheme.EMULATED, Scheme.EXCHANGE,
                             Scheme.LANE_BATCHED)


def choose(threads, accumulator, ctx) -> Optional[Fit]:
    """The scheme this target prefers among the ones that fit, ready to emit."""
    legal = offers(threads, accumulator, ctx)
    for scheme in ORDER:
        if scheme not in legal:
            continue
        if scheme is Scheme.LANE_BATCHED:
            tile = mfma_tile_for(threads, accumulator, ctx)
            return Fit(scheme, tile.op, tile=tile, reads=tile.op.a.dtype)
        if scheme is Scheme.EXCHANGE:
            op = exchange_op(accumulator, threads, ctx)
            return Fit(scheme, op, reads=op.a.dtype)
        tile, terms = emu_tile_for(threads, accumulator, ctx)
        # What the split reads, which is the accumulator: the narrow type is
        # what comes out of it, not what goes in.
        return Fit(scheme, tile.op, tile=tile, terms=terms,
                   reads=accumulator)
    return None


def boundary(fit: Fit, n: int) -> int:
    """Where the matrix span ends and a tail is left to something else.

    Only the lane-batched scheme draws one.  Its threshold is a measurement
    against a block of four: a tail of two or three columns is cheaper as one
    padded issue than as two or three passes of a chain, and a tail of one is
    not.  The other two pad a partial block inside the emitter and write back
    only the real columns, so a plan has no boundary to place for them -- and
    the same threshold would not carry over anyway, since their block is
    sixteen wide and nobody has measured what padding fifteen columns costs
    against a chain.
    """
    if fit.scheme is not Scheme.LANE_BATCHED:
        return n
    empty = packing.waste(n, fit.width)
    return ((n // fit.width) * fit.width) if empty in (0, fit.width - 1) else n
