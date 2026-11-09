# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT

"""Pseudo-IR: wave priorities on AMD (`Options.amd_setprio`).

A CDNA SIMD issues from the eligible wave of the highest priority
(`s_setprio`, 0 to 3), the oldest among equals.  Priorities change nothing
for a wave alone on its SIMD, and nothing about what a wave waits for; what
they decide is which of several eligible waves goes first.  HipKittens
(Hazy Research, 2025) pairs two waves a SIMD, one loading while the other
multiplies, and raises the multiplying one: the matrix unit stays fed while
the loads go out in its shadow.

The modes, for measuring which order pays here:

* ``loads``: high while a wave issues a transfer (`transfers`), low
  otherwise -- the loads leave first, and the arithmetic fills the gaps.
* ``compute``: the reverse -- the arithmetic first, the loads in its gaps.
* ``alternate``: every other wave high for the whole kernel, a fixed
  order rather than a first-come one.
* ``flat`` and ``sched``, the controls: where ``loads`` sets a priority,
  priority 0 throughout, or a scheduling barrier and no priority -- for
  telling what the priorities do from what an instruction the compiler
  will not move across does.

Placed behind every pass that moves a statement, so the markers stand where
the transfers ended up; pinned, with no effect of their own.
"""

from __future__ import annotations

from dataclasses import replace
from typing import List, Sequence, Tuple

from .core import Effect, Op, Stmt
from .transfers import glue, neutral, transfers

HIGH = 2
LOW = 0

MODES = ('loads', 'compute', 'alternate', 'flat', 'sched')

#: What the controls put where `loads` puts a priority: the same
#: instruction at priority 0 throughout (`flat`), and a scheduling barrier
#: alone (`sched`) -- `s_setprio` is an instruction with side effects, which
#: the compiler does not move anything across.
_FLAT = '__builtin_amdgcn_s_setprio(0);'
_SCHED = '__builtin_amdgcn_sched_barrier(0);'


def _prio(level) -> Stmt:
    text = level if isinstance(level, str) else (
        f'__builtin_amdgcn_s_setprio({level});')
    return Stmt(op=Op.RAWSTMT, target=(), args=(), pure=False, movable=False,
                effect=Effect.NONE, text=text)


def _alternate() -> Stmt:
    # The wave's index in the block, read from the first lane so that the
    # branch is uniform, and every odd one raised.
    return Stmt(op=Op.RAWSTMT, target=(), args=(), pure=False, movable=False,
                effect=Effect.NONE,
                text=('if (__builtin_amdgcn_readfirstlane((threadIdx.x + '
                      'blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) '
                      f'/ warpSize) & 1) {{ __builtin_amdgcn_s_setprio({HIGH}); }}'))


def _quiet(s: Stmt) -> bool:
    """What a run of transfers may hold without ending: a comment, a mark,
    an allocation, the arithmetic of an address or a guard -- nothing that
    is the work the priorities are about."""
    return neutral(s) or glue(s) or s.op in (Op.MARK, Op.ALLOC)


def _mark(body: Sequence[Stmt], during: int, between: int) -> Tuple[Stmt, ...]:
    """`body` with the priority `during` ahead of every run of transfer
    statements and `between` behind it, in this list and every one nested."""
    members = set()
    for t in transfers(body):
        members.update(id(s) for s in t.stmts)
    out: List[Stmt] = []
    inside = False
    for n, s in enumerate(body):
        if id(s) in members:
            if not inside:
                out.append(_prio(during))
                inside = True
            out.append(s)
            continue
        if inside and _quiet(s):
            # A run ends only where no transfer follows before the next
            # statement that issues.
            ahead = next((x for x in body[n + 1:] if not _quiet(x)), None)
            if ahead is not None and id(ahead) in members:
                out.append(s)
                continue
        if inside:
            out.append(_prio(between))
            inside = False
        if s.regions:
            s = replace(s, regions=tuple(
                replace(r, body=_mark(r.body, during, between))
                for r in s.regions))
        out.append(s)
    if inside:
        out.append(_prio(between))
    return tuple(out)


def set_priority(body: Tuple[Stmt, ...], mode: str) -> Tuple[Stmt, ...]:
    """`body` with the wave priorities `mode` asks for (see the module)."""
    if mode == 'alternate':
        return (_alternate(),) + tuple(body)
    if mode == 'loads':
        return _mark(body, HIGH, LOW)
    if mode == 'flat':
        return _mark(body, _FLAT, _FLAT)
    if mode == 'sched':
        return _mark(body, _SCHED, _SCHED)
    if mode == 'compute':
        return (_prio(HIGH),) + _mark(body, LOW, HIGH)
    raise ValueError(f'amd_setprio: {mode!r} is none of {", ".join(MODES)}')
