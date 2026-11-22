# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT

"""Pseudo-IR: the roles of a block.

A block in roles runs a body of its own on each of N disjoint ranges of its
warps -- a producer filling a ring, the warps that compute an operand, the
ones that issue a matrix engine's products and write what they made -- and
the roles meet on the barriers of rings alone (`IRBuilder.mbarrier_ring`),
each within itself on its own named barrier (`Participants.ROLE`): a barrier
of the block's inside a role would wait for warps that never come.

**The statement.**  `Op.IF` with the attribute `roles`, as two roles have
been since `IRBuilder.roles`; N of them are a chain of such two-region IFs.
Link k is `IF(c_k, roles=(R_k, ..., R_{N-1}))`: its first region is role
k's, its second role N-1's at the last link and otherwise the next link,
alone.  Every pass that reads an IF as a condition and two regions -- forty
places and more -- reads the chain right as it stands, and two roles are the
statement they always were.  What reads roles as roles -- `verify`, the
barrier placement -- reads them through `regions` and `rebuild`, and
`chain_refusal` holds the chain to its shape.

**The barrier ids.**  Every thread of the block computes its role's barrier
id from its warp (`RoleSplit.ident`): the first id, plus how many of the
roles' ranges start at or below its warp past the lowest -- the same text in
every role's region, so that a barrier is its role's wherever the role's
threads execute it, and the host oracle tells the roles' threads apart by
it.  On NVIDIA the roles take the top ids (`first_id`), apart from the
multiplications' ids from 1 on.

**Register budgets.**  A role may keep fewer registers than the kernel's
entry gives and hand the rest to the others (`setmaxnreg`):
`register_budgets` splits the block's pool among the roles -- zero-sum per
block, an `inc` past it waits forever -- and `emit_budgets` writes the
moves, whole warpgroups each.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, \
    Tuple

from .core import BOOL, IRError, Op, Participants, Region, Stmt

#: Warps a warpgroup holds: what `setmaxnreg` moves the registers of.
WARPGROUP_WARPS = 4


class Role(NamedTuple):
    """A role of the block (`IRBuilder.roles`, `IRBuilder.split_roles`):
    `threads` of its threads, meeting on a named barrier of their own, whose
    id `ident` is text every thread of the block evaluates to its own
    role's -- one id a role, so that a thread of one role is no member of
    another's barrier.  `warps` is the range of the block's warps the role
    holds, `(first, count)` along its split's axis, where a split was made
    of them (`RoleSplit`); `budget` the registers a thread of the role
    keeps from its region's head on (`register_budgets`), None where it
    keeps the kernel's entry's."""
    name: str
    threads: int
    ident: str = ''
    warps: Optional[Tuple[int, int]] = None
    budget: Optional[int] = None


#: The named barriers of an NVIDIA block, 0 to 15: 0 is `__syncthreads()`'s,
#: and the multiplications of the block take theirs from 1 on
#: (`CudaLexic.sync_mult`).
NAMED_IDS = 16


def first_id(count: int) -> int:
    """The first of the ids `count` roles take on NVIDIA: the top ones,
    apart from the multiplications' -- `14` for two roles, as the
    multi-element producer has always taken them."""
    if not 2 <= int(count) < NAMED_IDS:
        raise IRError(f'{count} roles, where a block has {NAMED_IDS - 1} '
                      f'named barriers besides its own')
    return NAMED_IDS - int(count)


@dataclass(frozen=True)
class RoleSplit:
    """N roles over the warps of a block (see the module): `roles` in the
    order their regions are built and the chain's links stand, each with
    its range of warps; `warp` the text a thread's warp is along the split's
    axis, which its barrier id is computed from, and `axis` the thread id
    the builder reads it as -- a row of the block, `threadIdx.y`, in the
    multi-element layout; `first` the first named barrier id the roles take.

    The ranges are disjoint and cover the warps from 0 on, in any order of
    the roles; each role comes out with its `ident` (`ident`)."""

    roles: Tuple[Role, ...]
    warp: str = 'threadIdx.y'
    axis: str = 'y'
    first: int = NAMED_IDS - 2

    def __post_init__(self):
        roles = tuple(self.roles)
        if len(roles) < 2:
            raise IRError(f'a split of {len(roles)} role(s): a block in roles '
                          f'has two or more')
        names = [r.name for r in roles]
        if len(set(names)) != len(names):
            raise IRError(f'two roles of one name in {names}')
        at = 0
        for role in sorted(roles, key=lambda r: (r.warps or (-1, 0))[0]):
            if role.warps is None:
                raise IRError(f'the role {role.name} names no warps')
            lo, count = role.warps
            if count < 1:
                raise IRError(f'the role {role.name} holds {count} warps')
            if lo != at:
                raise IRError(f'the roles\' warps leave warp {at} out or '
                              f'take it twice (role {role.name} starts at '
                              f'{lo})')
            at = lo + count
        if self.first < 0:
            raise IRError(f'a first barrier id of {self.first}')
        ident = self.ident
        object.__setattr__(self, 'roles',
                           tuple(r._replace(ident=ident) for r in roles))

    @property
    def warps(self) -> int:
        """The block's warps, every role's together."""
        return sum(r.warps[1] for r in self.roles)

    @property
    def starts(self) -> Tuple[int, ...]:
        """Where the roles' ranges start, past the lowest, in order."""
        return tuple(sorted(r.warps[0] for r in self.roles))[1:]

    @property
    def ident(self) -> str:
        """The barrier id a thread computes as its role's: `first` and a
        step at every range's start at or below its warp -- for two roles
        `(14 + (threadIdx.y >= E))`."""
        steps = ''.join(f' + ({self.warp} >= {lo})' for lo in self.starts)
        return f'({self.first}{steps})'

    def role(self, name: str) -> Role:
        for r in self.roles:
            if r.name == name:
                return r
        raise IRError(f'no role {name!r} in the split; it has '
                      f'{", ".join(r.name for r in self.roles)}')

    def index(self, name: str) -> int:
        return [r.name for r in self.roles].index(self.role(name).name)

    def id_of(self, name: str) -> int:
        """The barrier id the threads of role `name` compute."""
        lo = self.role(name).warps[0]
        return self.first + sum(1 for s in self.starts if s <= lo)

    def member(self, b, name: str, warp) -> Any:
        """Whether a thread whose warp is `warp` is of role `name`: below
        the range's end where it starts at warp 0, at or past its start
        where it ends with the block, both otherwise -- built by `b` under
        the role's name."""
        role = self.role(name)
        lo, count = role.warps
        hi = lo + count
        if lo == 0:
            return b.op('lt', BOOL, warp, hi, hint=role.name)
        if hi == self.warps:
            return b.op('ge', BOOL, warp, lo, hint=role.name)
        return b.op('and', BOOL, b.op('ge', BOOL, warp, lo, hint=role.name),
                    b.op('lt', BOOL, warp, hi, hint=role.name),
                    hint=role.name)


def chain(conds: Sequence[Any], roles: Sequence[Role],
          regions: Sequence[Region]) -> Stmt:
    """The roles' regions as the chain of links they are (see the module):
    `conds[k]` holds for the threads of role k, the regions in the roles'
    order.  Two roles are one link, the statement `IRBuilder.roles` has
    always built."""
    n = len(roles)
    if n < 2 or len(regions) != n or len(conds) < n - 1:
        raise IRError(f'{n} roles, {len(regions)} regions and {len(conds)} '
                      f'conditions do not make a split')
    link = Stmt(op=Op.IF, args=(conds[n - 2],),
                regions=(regions[n - 2], regions[n - 1]), pure=False,
                movable=False, attrs=(('roles', tuple(roles[n - 2:])),))
    for k in range(n - 3, -1, -1):
        link = Stmt(op=Op.IF, args=(conds[k],),
                    regions=(regions[k], Region(body=(link,))), pure=False,
                    movable=False, attrs=(('roles', tuple(roles[k:])),))
    return link


def is_split(s: Stmt) -> bool:
    """Whether `s` is a split into roles: a chain's first link, or one
    further in."""
    return s.op == Op.IF and bool(s.attr('roles'))


def chain_refusal(s: Stmt) -> Optional[str]:
    """What is wrong with the shape of the link `s`, or None: two regions,
    two roles at least, and past the last link's two roles a second region
    that is the next link alone -- the rest of the roles, and nothing
    beside it.  Strict, so that a pass that moved a statement in between
    is named rather than read as a role's."""
    roles = tuple(s.attr('roles') or ())
    if len(roles) < 2:
        return f'a split into {len(roles)} role(s)'
    if len(s.regions) != 2:
        return (f'a split into {len(roles)} roles in {len(s.regions)} '
                f'region(s): a link of the chain has two')
    if len(roles) == 2:
        return None
    rest = s.regions[1].body
    if (len(rest) != 1 or not is_split(rest[0])
            or tuple(rest[0].attr('roles')) != roles[1:]):
        return (f'the split into {", ".join(r.name for r in roles)} goes on '
                f'in its second region with something other than the next '
                f'link alone: a chain of roles is broken')
    return None


def regions(s: Stmt) -> List[Tuple[Role, Region]]:
    """The roles of the split `s` and their regions, in order, the chain's
    links flattened."""
    out = []
    while True:
        why = chain_refusal(s)
        if why is not None:
            raise IRError(why)
        roles = s.attr('roles')
        out.append((roles[0], s.regions[0]))
        if len(roles) == 2:
            out.append((roles[1], s.regions[1]))
            return out
        s = s.regions[1].body[0]


def rebuild(s: Stmt, placed: Sequence[Region]) -> Stmt:
    """The split `s` with its roles' regions `placed` in their order (as
    `regions` lists them), the chain's links as they were."""
    roles = s.attr('roles')
    if len(placed) != len(roles):
        raise IRError(f'{len(placed)} regions for a split into '
                      f'{len(roles)} roles')
    if len(roles) == 2:
        return replace(s, regions=(placed[0], placed[1]))
    inner = rebuild(s.regions[1].body[0], placed[1:])
    return replace(s, regions=(placed[0],
                               replace(s.regions[1], body=(inner,))))


def roles_of(s: Stmt) -> Tuple[Optional[Role], ...]:
    """The role each region of the link `s` is, None for a second region
    that goes on to the next link."""
    roles = s.attr('roles')
    return (roles[0], roles[1] if len(roles) == 2 else None)


class SplitHandle:
    """The regions of a split being built (`IRBuilder.split_roles`): one
    `with handle.role(name):` a role, in the split's order, each with its
    role current -- whose barrier `Participants.ROLE` names -- and the
    chain emitted where the last of them closes."""

    def __init__(self, builder, split: RoleSplit, conds: Sequence[Any]):
        self.builder = builder
        self.split = split
        self.conds = tuple(conds)
        self._regions: List[Region] = []

    @contextmanager
    def role(self, name: str):
        split, b = self.split, self.builder
        i = split.index(name)
        if i != len(self._regions):
            raise IRError(f'the role {name} built as the split\'s '
                          f'{len(self._regions) + 1}. of '
                          f'{", ".join(r.name for r in split.roles)}: the '
                          f'regions go in the split\'s order')
        b.push(kind='if')
        b._roles.append(split.roles[i])
        try:
            yield self
        finally:
            b._roles.pop()
        self._regions.append(b.pop())
        if len(self._regions) == len(split.roles):
            b.emit(chain(self.conds, split.roles, self._regions))


# --------------------------------------------------------------------------- #
# Register budgets
# --------------------------------------------------------------------------- #

#: Registers of the SM a block holds where one block is resident, which its
#: warpgroups share and `setmaxnreg` moves between them.
SM_REGISTERS = 65536

#: The most registers a thread holds at a kernel's entry (`__maxnreg__`),
#: and the most `setmaxnreg` raises a warpgroup's to.
ENTRY_REGISTERS = 255
MOST_REGISTERS = 256


def register_budgets(threads: Sequence[Tuple[str, int]],
                     keep: Mapping[str, int], *,
                     registers: int = SM_REGISTERS
                     ) -> Optional[Tuple[int, Dict[str, int]]]:
    """`(entry, budgets)`: the registers a thread holds at the kernel's
    entry -- every thread of the block alike, of the SM's `registers` --
    and from the moves on, by role: the roles in `keep` keep that many, and
    the others -- `threads` gives every role's threads -- share alike what
    those give up, a multiple of eight to `MOST_REGISTERS`.  None where
    nothing moves: no role keeps fewer than the entry, or the others would
    gain nothing.

    What the moves draw on is the block's own allocation, `entry` a thread
    of it, not the SM's: the pool is the block's and zero-sum, a role's
    `inc` waits until the others' `dec` gave the registers back, and
    forever past the pool."""
    if not keep:
        return None
    total = sum(t for _, t in threads)
    takers = [(name, t) for name, t in threads if name not in keep]
    if not total or not takers:
        return None
    entry = min(ENTRY_REGISTERS, registers // total // 8 * 8)
    given = sum(t * int(keep[name]) for name, t in threads if name in keep)
    gained = (entry * total - given) // sum(t for _, t in takers) // 8 * 8
    share = min(MOST_REGISTERS, gained)
    if (any(int(k) >= entry for k in keep.values()) or share <= entry
            or entry < 24):
        return None
    budgets = {name: int(keep[name]) for name, _ in threads if name in keep}
    budgets.update({name: share for name, _ in takers})
    return entry, budgets


def budget_refusal(split: RoleSplit, budgets: Mapping[str, int],
                   entry: int, wave: int = 32) -> Optional[str]:
    """Why the roles of `split` cannot move their registers to `budgets`
    from `entry`, or None: a role that moves is whole warpgroups on a
    warpgroup's boundary -- `setmaxnreg` is the warpgroup's -- and the
    block holds no more after the moves than its threads hold at the
    entry: the pool is the block's own allocation, zero-sum, and an `inc`
    past it waits forever."""
    held = 0
    for role in split.roles:
        count = int(budgets.get(role.name, entry))
        lo, warps = role.warps
        if count != entry and (lo % WARPGROUP_WARPS
                               or warps % WARPGROUP_WARPS):
            return (f'the role {role.name} moves its registers, and its warps '
                    f'{lo} to {lo + warps - 1} are no whole warpgroups')
        held += count * warps * wave
    if held > entry * split.warps * wave:
        return (f'the roles hold {held} registers after the moves, past the '
                f'{entry * split.warps * wave} the block holds at its entry')
    return None


def emit_budgets(b, split: RoleSplit, budgets: Mapping[str, int],
                 entry: int) -> None:
    """The moves of the roles' registers to `budgets`, ahead of the split,
    by every warp of the block: each role that gives registers lowers its
    count (`dec`), the block meets, and each role that takes them raises
    its own (`inc`) -- whole warpgroups each, as the collective is
    (`engines.register_budget`) -- refused where they do not fit the
    block's pool (`budget_refusal`)."""
    from .engines import register_budget
    why = budget_refusal(split, budgets, entry)
    if why is not None:
        raise IRError(why)
    warp = b.thread_id(split.axis)
    for role in split.roles:
        count = int(budgets.get(role.name, entry))
        if count < entry:
            with b.if_(split.member(b, role.name, warp)):
                b.emit(register_budget(count, 'dec'))
    b.barrier(Participants.BLOCK)
    for role in split.roles:
        count = int(budgets.get(role.name, entry))
        if count > entry:
            with b.if_(split.member(b, role.name, warp)):
                b.emit(register_budget(count, 'inc'))


__all__ = ['ENTRY_REGISTERS', 'MOST_REGISTERS', 'NAMED_IDS', 'Role',
           'RoleSplit', 'SM_REGISTERS', 'SplitHandle', 'WARPGROUP_WARPS',
           'budget_refusal', 'chain', 'chain_refusal', 'emit_budgets',
           'first_id', 'is_split', 'rebuild', 'register_budgets', 'regions',
           'roles_of']
