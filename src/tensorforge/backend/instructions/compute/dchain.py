# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Products that add into one output, summed where the matrix instruction
holds its sum (`Options.mma_d_layout`).

`Q += A_1 B_1`, `Q += A_2 B_2`, ... is built product by product: each one
accumulates into fragments laid out the instruction's way -- the D layout,
lane `t` holding rows `t / 4 + 8 g` and columns `2 (t % 4) + e` -- hands
them through a shared tile (`ctile`) to the lanes that own the rows, adds
the output's old value read by rows, and stores the sum by rows.  The next
product reads it back.  For SeisSol's volume that is three tiles, three
reads and three writes of `Q` for one sum.

A chain keeps the sum in the D layout from the first product to the last:
the output is read into the accumulators once, by the addresses the D
layout gives each lane, every product adds into them, and the last writes
them back by the same addresses.  No tile, no rows.  Nothing between the
first product and the last may read or write the output, which is what
makes holding it in registers the same as writing it every time.

Three parties.  The builder records what it put around a product for its
result (`Companions`): the registers it accumulates into, the load of the
old value, the epilogue that adds it, the store.  `link` finds the
products whose companions a chain replaces and takes those out of the
stream; each product is told its place in the chain (`DChain`).  And the
target's emitter does the rest, asked through `MatmulOperands.chain`:
`primitives/nvidia.py` declares the accumulators at the first product,
reads the output into them, and writes them at the last.

A product whose window is narrower than the chain's adds zeros to the rows
it does not own -- where its operand is known to be zero there
(`MultilinearInstruction._known_zero_rows`); otherwise it is not a member.
Its tiles still cover only its own rows, so the instructions it issues are
the ones it issues on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from tensorforge.common.exceptions import InternalError


@dataclass
class Companions:
    """What the builder put around one product for its result
    (`MultilinearBuilder`): what a chain replaces.

    `alloc` is the result registers' allocation, `bias` the load of the
    destination's old value behind the product, `epilogue` what adds it,
    `promise` what the result says it holds past its rows, and `store` the
    write of the destination -- recorded once the builder makes it."""
    alloc: Tuple = ()
    bias: Tuple = ()
    epilogue: object = None
    promise: object = None
    store: object = None

    def instructions(self) -> Tuple:
        return tuple(x for x in (self.alloc + self.bias
                                 + (self.epilogue, self.promise, self.store))
                     if x is not None)


@dataclass
class DChain:
    """The products of one chain, in the order the stream runs them, and
    what they share.

    `dest` is the global symbol they add into, `offset` where the
    accumulators' frame sits in it (`StoreRegToGlb._dest_offset`), `rows`
    the window of the lead dimension the chain reads and writes -- the
    union of the members' -- in that frame, and `columns` the output
    columns, which every member has the same.

    `state` is what the emitter keeps between the members of one build of
    the body: the accumulators, which the first member declares.  A body is
    built more than once (`Generator._place_section`, `_fused_if_over_budget`),
    and every build starts at the first member, which starts it over."""
    members: List
    dest: object
    offset: Tuple[int, ...]
    rows: Tuple[int, int]
    columns: Tuple[int, int]
    state: Dict = field(default_factory=dict)

    def opens(self, member) -> bool:
        return member is self.members[0]

    def closes(self, member) -> bool:
        return member is self.members[-1]

    def __str__(self) -> str:
        return (f'chain of {len(self.members)} into {self.dest.name} over '
                f'rows {self.rows}')


def _flat(instrs) -> list:
    """Every instruction of `instrs`, and of their regions."""
    out = []
    for instr in instrs:
        out.append(instr)
        for region in instr.regions():
            out.extend(_flat(region))
    return out


def _reads_or_writes(instr, symbols, tensor=None) -> bool:
    """Whether `instr` itself reads or writes one of `symbols` -- or any
    symbol of `tensor`."""
    for sym in tuple(instr.uses()) + tuple(instr.defs()):
        if sym is None:
            continue
        if any(sym is s for s in symbols):
            return True
        if tensor is not None and getattr(sym, 'obj', None) is tensor:
            return True
    return False


def _candidate(instr, module, everything) -> Optional[str]:
    """Why `instr` cannot be a member of a chain, or None where it can.

    Besides what the product says of itself (`d_chain_refusal`), its result
    and the image of the old value have to be its companions' alone: the
    chain writes neither, so anything else that reads them would read
    nothing."""
    from .multilinear import MultilinearInstruction
    if not isinstance(instr, MultilinearInstruction):
        return 'not a product'
    parts = getattr(instr, 'companions', None)
    if parts is None or parts.store is None or parts.epilogue is None:
        return 'no recorded epilogue and store'
    why = instr.d_chain_refusal(module)
    if why is not None:
        return why
    own = (instr,) + parts.instructions()
    symbols = instr.d_chain_symbols()
    if any(_reads_or_writes(x, symbols) for x in everything
           if not any(x is o for o in own)):
        return 'its result or its old value is read elsewhere'
    return None


def _rows_ok(members, rows, module) -> bool:
    """Whether every member adds only zeros to the rows of the chain's
    window outside its own."""
    lo, hi = rows
    for member in members:
        own = member.d_chain_rows()
        tiles = member.d_chain_tile_rows(module)
        foreign = [r for r in tiles if lo <= r < hi
                   and not own[0] <= r < own[1]]
        if foreign and not member._known_zero_rows(foreign):
            return False
    return True


def _compatible(chain: List, instr, module) -> bool:
    """Whether `instr` may join the chain whose members are `chain`."""
    if instr.d_chain_key(module) != chain[0].d_chain_key(module):
        return False
    rows = _union([m.d_chain_rows() for m in chain + [instr]])
    return _rows_ok(chain + [instr], rows, module)


def _union(windows: Sequence[Tuple[int, int]]) -> Tuple[int, int]:
    return (min(lo for lo, _ in windows), max(hi for _, hi in windows))


def link(context, instrs: Sequence) -> list:
    """`instrs` with the products that add into one global output linked
    into chains (see the module): their companions taken out, and each told
    its chain.

    One list at a time, and a chain within one: a product inside a loop of
    the section is left as it is.  A chain ends at anything else that
    touches its output, and at a product that cannot join it; a product on
    its own is a chain of one, which still reads and writes its output in
    the D layout rather than through the tile."""
    from .multilinear import _vendor_module
    module = _vendor_module(context)
    instrs = list(instrs)
    everything = _flat(instrs)
    open_chains: Dict[int, List] = {}
    done: List[List] = []

    def close(key):
        chain = open_chains.pop(key, None)
        if chain:
            done.append(chain)

    for instr in instrs:
        if _candidate(instr, module, everything) is None:
            key = id(instr.d_chain_dest())
            chain = open_chains.get(key)
            if chain is not None and not _compatible(chain, instr, module):
                close(key)
                chain = None
            if chain is None:
                open_chains[key] = [instr]
            else:
                chain.append(instr)
            continue
        # Anything else that touches an open chain's output ends it -- but
        # for a member's companions, which the chain takes out.
        for key, chain in list(open_chains.items()):
            if any(instr is x for m in chain
                   for x in m.companions.instructions()):
                continue
            dest = chain[0].d_chain_dest()
            if any(_reads_or_writes(x, (dest,), dest.obj)
                   for x in _flat([instr])):
                close(key)
    for key in list(open_chains):
        close(key)

    drop = set()
    for members in done:
        first = members[0]
        chain = DChain(members=members, dest=first.d_chain_dest(),
                       offset=tuple(first.companions.store._dest_offset),
                       rows=_union([m.d_chain_rows() for m in members]),
                       columns=first.d_chain_columns())
        for member in members:
            member.join_d_chain(chain)
            drop.update(id(x) for x in member.companions.instructions())
    return [x for x in instrs if id(x) not in drop]


def check(chain: DChain, member) -> None:
    """Raise where `member` is emitted out of its chain's order: the first
    member declares what the others add into."""
    if not chain.opens(member) and not chain.state:
        raise InternalError(
            f'{chain}: a member emitted before the first, which declares the '
            f'accumulators the others add into')
