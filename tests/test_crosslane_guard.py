# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Cross-lane reads leave the guards that split the lanes.

A lane broadcast reads another lane's register, so it is defined only where
that lane executes it too.  The generic nest at lead width two put the
broadcast of `B` inside its tail guard, `if (lead < 6)`: on gfx1150 the DPP
row share read zero from the lanes the guard had turned off, and
`slice_offset_a` came out wrong by 8.8.  `passes.converge_crosslane` takes such
a read out of the guard, with the register read it broadcasts.
"""
import re

from tensorforge.backend.pir import BOOL, IRBuilder, MemSpace, Op, ScalarType
from tensorforge.backend.pir.core import walk_stmts
from tensorforge.backend.pir.passes import converge_crosslane
from tensorforge.common.basic_types import Datatype

from test_amd_packed_dpp import _kernel

F32 = ScalarType(Datatype.F32)


def _guarded(space):
    """`if (tid < 6) acc += bcast(buf[3]);` with `buf` in `space`."""
    b = IRBuilder(fptype=Datatype.F32, scratch=('tempShrMem', 64))
    buf = b.alloc(Datatype.F32, (16,), space, hint='r')
    acc = b.declare(F32, hint='acc')
    cond = b.op('lt', BOOL, b.thread_id('x'), 6, hint='g')
    with b.if_(cond):
        v = b.load(buf, 3, hint='data')
        b.accumulate(acc, b.rawexpr('bcast({0})', v, pure=True, movable=True,
                                    crosslane=True, hint='bc'))
    return b.finish()


def _inside_guards(body):
    found = []
    for s in walk_stmts(body):
        if s.op == Op.IF:
            found.extend(walk_stmts(tuple(x for r in s.regions
                                          for x in r.body)))
    return found


def test_a_broadcast_of_a_register_read_leaves_the_lane_guard():
    body = _guarded(MemSpace.REGISTER)
    assert any(s.attr('crosslane') for s in _inside_guards(body))
    moved = converge_crosslane(body)
    inside = _inside_guards(moved)
    assert not any(s.attr('crosslane') for s in inside)
    assert not any(s.op == Op.LOAD for s in inside), (
        'the register read the broadcast needs goes with it')
    assert any(s.op == Op.ACCUM for s in inside), 'the accumulation stays'
    assert any(s.attr('crosslane') for s in walk_stmts(moved))


def test_a_broadcast_of_a_memory_read_stays_in_the_guard():
    """The guard may be what keeps a memory read in bounds, so neither the
    read nor the broadcast of it moves."""
    moved = converge_crosslane(_guarded(MemSpace.SHARED))
    assert any(s.attr('crosslane') for s in _inside_guards(moved))


def _broadcasts_in_lane_guards(src):
    guards = inside = depth = 0
    open_guard = False
    for line in src.splitlines():
        if not open_guard and re.search(r'if \(v\d+_g\) \{', line):
            open_guard, depth = True, 0
            guards += 1
        if open_guard:
            depth += line.count('{') - line.count('}')
            inside += line.count('broadcast<')
            if depth <= 0:
                open_guard = False
    return guards, inside


def test_the_nest_broadcasts_outside_its_lane_guards():
    """`slice_offset_a` and `offset_a` at lead width two on gfx1150, through
    the nest (`FUSED_WIDE` off): every tail block is guarded, and none of the
    broadcasts of `B` is in one."""
    from tensorforge.backend.instructions.compute.primitives.amd import codegen
    saved, codegen.FUSED_WIDE = codegen.FUSED_WIDE, False
    try:
        for stem in ('slice_offset_a', 'offset_a'):
            src = _kernel(stem, 'gfx1150', width=2)
            guards, inside = _broadcasts_in_lane_guards(src)
            assert guards and 'broadcast<' in src, stem
            assert inside == 0, stem
    finally:
        codegen.FUSED_WIDE = saved
