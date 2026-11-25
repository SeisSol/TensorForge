# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""How an operation's rows fill a matrix unit's tile when they do not divide it.

A matrix instruction multiplies tiles of a fixed height on the axis that
holds an element's quantities: DPAS's eight rows (its repeat count), the
eight columns of a per-element `mma.sync` (its `n`), MFMA's 16 or 32, the
multi-element tiles' 64 or 128 rows of `wgmma` and `tcgen05`.  An element
brings `Q` of them -- SeisSol's elastic nine, more for viscoelastic -- and
where `Q` is no multiple of the height `H`, `r = Q mod H` rows are left
over.  What to do with them is one choice for every target, and this module
states it once (`Options.row_packing`), with each target saying which of the
answers it lowers and which it takes by itself (`row_caps`):

* `fma`: the remainder rows on the vector units' FMAs, over the operator
  values the tiles hold anyway (Intel's SPMD DPAS and NVIDIA's per-element
  `mma.sync` carry up to two such columns, AMD's stacked tiles a DPP span).
* `gather`: the remainders of `E = H / gcd(r, H)` elements stacked into
  whole tiles -- eight elements' ninth rows on DPAS -- exchanged through the
  block's shared memory among the multiplications of those elements
  (`shared`), or run as a work item of their own right after them (`item`,
  the hand-written PVC volume's q8 tile).
* `dense`: `E` elements' `Q E` rows on whole tiles, `Q E` near a multiple of
  `H` (63 of 64 for seven elements, 72 of 72 for eight) -- the multi-element
  layout's tiles (`generators.element_group`).
* `pad`: `ceil(Q / H)` tiles an element, the last one mostly zero (Intel's
  explicit-vector DPAS).
* `transposed`: the elements on the instruction's wide side and the basis
  on its rows: no remainder, and an element's whole output a lane.

What each costs depends on the binding resource, which is why the choice is
measured per target: on the Max 1550 the order-6 volume is bound by memory
and the order-8 one by the vector engines' issue, where `fma`'s extra FMAs
(2166 an element at order 8 on TF32 halves) compete with the mixing the
DPAS operand needs.

Today every target's `auto` is what it did before this choice existed.  The
packings no target lowers yet (`gather`, `dense` outside the multi-element
layout, `transposed`) are refused by name where they are asked for.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import gcd
from typing import FrozenSet, Optional

from tensorforge.common.exceptions import GenerationError


class Packing(Enum):
    """`Options.row_packing`, the strategies."""

    AUTO = 'auto'
    FMA = 'fma'
    GATHER = 'gather'
    DENSE = 'dense'
    PAD = 'pad'
    TRANSPOSED = 'transposed'


class Exchange(Enum):
    """`Options.row_packing_exchange`: how `gather` brings the elements'
    remainders together."""

    AUTO = 'auto'
    #: Through the block's shared memory, among the multiplications of the
    #: elements a block already holds side by side.
    SHARED = 'shared'
    #: A work item of its own right after its elements' items, which needs
    #: the items in that order and a thread count coprime to the block
    #: (`tensorforge::coreGrid`).
    ITEM = 'item'


@dataclass(frozen=True)
class RowCaps:
    """What a target lowers for one product (`MatrixPaths.row_caps`): the
    tile's height on the quantity axis, the packings and the exchanges it
    has, and the one it takes by itself."""

    height: int
    packings: FrozenSet[Packing]
    auto: Packing
    exchanges: FrozenSet[Exchange] = frozenset()


@dataclass(frozen=True)
class RowPlan:
    """The packing of one product: `quantities` rows an element on tiles of
    `height`, `elements` elements sharing tiles (`gather`, `dense`), and why
    -- a refusal names what was asked and what the target has."""

    packing: Packing
    height: int
    quantities: int
    elements: int = 1
    exchange: Optional[Exchange] = None
    why: str = ''

    @property
    def remainder(self) -> int:
        return self.quantities % self.height if self.height else 0


def _parse(value, kind, name):
    text = str(value).strip().lower()
    try:
        return kind(text)
    except ValueError:
        allowed = ', '.join(k.value for k in kind)
        raise GenerationError(f'{name}={value!r}: one of {allowed}') from None


def asked(ctx) -> Packing:
    """`Options.row_packing`, `auto` where there is no context."""
    if ctx is None:
        return Packing.AUTO
    return _parse(ctx.get_user_options().row_packing, Packing, 'row_packing')


def gather_elements(quantities: int, height: int) -> int:
    """Elements whose remainders fill whole tiles: `H / gcd(r, H)`."""
    r = quantities % height
    return height // gcd(r, height) if r else 1


def plan_rows(quantities: int, caps: Optional[RowCaps], ctx) -> RowPlan:
    """The packing of a product of `quantities` rows an element (its output
    columns) on a target that answers `caps`, as `Options.row_packing` asks.

    No remainder, or a target with no answer (`caps` None), packs nothing
    and asks nothing; `auto` is the target's own; any other packing has to
    be one the target lowers, or generation stops and says which ones it
    has."""
    wanted = asked(ctx)
    if caps is None:
        if wanted is not Packing.AUTO:
            raise GenerationError(
                f'row_packing={wanted.value}: this target states no packing '
                f'of rows on its matrix path; only auto')
        return RowPlan(Packing.AUTO, 0, quantities, why='no matrix unit')
    if quantities % caps.height == 0:
        return RowPlan(caps.auto if wanted is Packing.AUTO else wanted,
                       caps.height, quantities, why='no remainder')
    packing = caps.auto if wanted is Packing.AUTO else wanted
    if packing not in caps.packings:
        have = ', '.join(sorted(p.value for p in caps.packings))
        raise GenerationError(
            f'row_packing={packing.value}: {quantities} rows on tiles of '
            f'{caps.height} leave {quantities % caps.height}, and this target '
            f'lowers {have} for them')
    options = ctx.get_user_options() if ctx is not None else None
    elements = 1
    exchange = None
    if packing in (Packing.GATHER, Packing.DENSE):
        given = int(getattr(options, 'row_packing_elements', 0) or 0)
        elements = given or gather_elements(quantities, caps.height)
        if packing is Packing.GATHER:
            exchange = _parse(getattr(options, 'row_packing_exchange', 'auto'),
                              Exchange, 'row_packing_exchange')
            if exchange is Exchange.AUTO:
                exchange = next(iter(sorted(caps.exchanges,
                                            key=lambda e: e.value)), None)
            if exchange not in caps.exchanges:
                raise GenerationError(
                    f'row_packing=gather through {exchange}: this target has '
                    f'{sorted(e.value for e in caps.exchanges)}')
    return RowPlan(packing, caps.height, quantities, elements, exchange,
                   why='asked' if wanted is not Packing.AUTO else 'the target')


def check(ctx) -> None:
    """Refuse `row_packing` where nothing could take it: a target whose
    matrix module states no packing (`MatrixPaths.row_caps`).  Asked once
    a build, ahead of the products, so that an option no product reads does
    not pass unseen."""
    wanted = asked(ctx)
    if wanted is Packing.AUTO:
        return
    from .paths import matrix_paths
    module = matrix_paths(ctx.target)
    if module is None or not hasattr(module, 'row_caps'):
        raise GenerationError(
            f'row_packing={wanted.value}: {ctx.target.hw.vendor} states no '
            f'packing of rows on its matrix path; only auto')
