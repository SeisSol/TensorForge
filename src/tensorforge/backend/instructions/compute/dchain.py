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
    and every build starts at the first member, which starts it over.

    `loop` is the merged run (`VariantLoop`) whose every trip runs the one
    member, where the chain spans its trips: the loop opens the chain ahead
    of its header and closes it behind, and carries the accumulators from
    trip to trip; `phase` is which of those the member is emitted for
    (`MatmulOperands.chain`)."""
    members: List
    dest: object
    offset: Tuple[int, ...]
    rows: Tuple[int, int]
    columns: Tuple[int, int]
    state: Dict = field(default_factory=dict)
    loop: object = None
    phase: str = 'all'

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
    # A loop's reads are its body's, which are asked themselves: a merged
    # run would otherwise read every member's result in its own name.
    if any(_reads_or_writes(x, symbols) for x in everything
           if not any(x is o for o in own) and not any(x.regions())):
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


def _read_after(instrs: Sequence, members: List) -> bool:
    """Whether anything after the chain's last member reads its output.

    Without the chain that read takes the last member's register image,
    which its store leaves holding the output (`GlbToRegLoader`); with it
    there is no such image, and the read goes to memory.  Where the
    accumulator is that image anyway (`MatrixPaths.D_CHAIN_READ_AFTER`),
    such a chain buys less than it costs: SeisSol's time derivative reads
    every order's sum for the next."""
    last = max(i for i, x in enumerate(instrs)
               if any(x is m for m in members))
    dest = members[0].d_chain_dest()
    own = {id(x) for m in members for x in m.companions.instructions()}
    for instr in _flat(instrs[last + 1:]):
        if id(instr) in own:
            continue
        if any(sym is not None and (sym is dest or getattr(sym, 'obj', None)
                                    is dest.obj)
               for sym in instr.uses()):
            return True
    return False


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
    the D layout rather than through the tile -- where the target keeps
    those (`MatrixPaths.D_CHAIN_SINGLES`)."""
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
    if not getattr(module, 'D_CHAIN_READ_AFTER', True):
        done = [members for members in done
                if not _read_after(instrs, members)]
    if (getattr(module, 'D_CHAIN_LOOPS', False)
            and context.get_user_options().mma_d_layout_loops):
        for instr in instrs:
            if hasattr(instr, 'counter') and hasattr(instr, 'replace_region'):
                _link_loop(instr, module, everything)

    drop = set()
    for members in done:
        if len(members) == 1 and not getattr(module, "D_CHAIN_SINGLES", True):
            # A product on its own adds through its epilogue where the
            # target says a chain of one is not worth it.
            continue
        first = members[0]
        chain = DChain(members=members, dest=first.d_chain_dest(),
                       offset=tuple(first.companions.store._dest_offset),
                       rows=_union([m.d_chain_rows() for m in members]),
                       columns=first.d_chain_columns())
        for member in members:
            member.join_d_chain(chain)
            drop.update(id(x) for x in member.companions.instructions())
    return [x for x in instrs if id(x) not in drop]


def _link_loop(loop, module, everything) -> None:
    """A chain over the trips of a merged run: where its body holds one
    product into an output in global memory that every trip adds into --
    not a member its counter selects -- and nothing else of the body
    touches that output, the trips add into one set of accumulators
    (`DChain.loop`).  SeisSol's `localFluxAll`, its four faces merged into
    one loop, is the case.  The member's companions leave the body."""
    region = list(loop.region)
    if loop.count < 2:
        return
    found = []
    for instr in region:
        if _candidate(instr, module, everything) is not None:
            continue
        dest = instr.d_chain_dest()
        if getattr(dest, 'is_variant', False) or getattr(
                getattr(dest, 'obj', None), 'is_variant', False):
            continue
        found.append(instr)
    chains = []
    drop = set()
    for member in found:
        dest = member.d_chain_dest()
        if sum(1 for x in found if x.d_chain_dest() is dest) != 1:
            continue
        own = {id(member)} | {id(x) for x in
                              member.companions.instructions()}
        if any(_reads_or_writes(x, (dest,), dest.obj)
               for x in _flat(region) if id(x) not in own):
            continue
        chain = DChain(members=[member], dest=dest,
                       offset=tuple(member.companions.store._dest_offset),
                       rows=member.d_chain_rows(),
                       columns=member.d_chain_columns(), loop=loop)
        member.join_d_chain(chain)
        drop.update(id(x) for x in member.companions.instructions())
        chains.append(chain)
    if chains:
        loop.replace_region(0, [x for x in region if id(x) not in drop])
        loop.chains = list(getattr(loop, 'chains', ())) + chains


def check(chain: DChain, member) -> None:
    """Raise where `member` is emitted out of its chain's order: the first
    member declares what the others add into."""
    if chain.loop is not None:
        if chain.phase != 'open' and not chain.state:
            raise InternalError(
                f'{chain}: a trip emitted before the loop opened the chain')
        return
    if not chain.opens(member) and not chain.state:
        raise InternalError(
            f'{chain}: a member emitted before the first, which declares the '
            f'accumulators the others add into')
