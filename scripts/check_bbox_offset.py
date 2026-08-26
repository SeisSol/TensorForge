# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Consistency checks for the bbox/offset split.  Run from the tensorforge root.

Pure arithmetic --- no codegen, no GPU.  Covers the two invariants the patch
depends on:

  1. allocation (_iregs / _alloc_register_array) and addressing
     (access_address / build_address) must agree on how many per-thread slots a
     thread-distributed dimension occupies;
  2. LeadLoop's block/guard structure must cover exactly [start, end).
"""
import sys
sys.path.insert(0, '.')
from tensorforge.backend.symbol import DataView
from tensorforge.common.matrix.boundingbox import BoundingBox

fail = 0


def check(name, got, want):
    global fail
    ok = got == want
    fail += not ok
    print(f'{"ok  " if ok else "FAIL"} {name}')
    if not ok:
        print(f'       got  {got}\n       want {want}')


def alloc_slots(l, u, T):
    """Verbatim from MultilinearInstruction._iregs."""
    return -(-u // T) - l // T


def old_addr_slots(l, u, T):
    """What access_address used to compute."""
    return (u - l + T - 1) // T


def covered(start, end, T, patched=True):
    """Replay LeadLoop.write's block/guard structure, collect visited indices."""
    a_s, a_e = start // T, -(-end // T)
    r_s, r_e = -(-start // T), end // T
    tail = end - r_e * T
    seen = set()
    if a_s == r_e:
        lo = max(start - a_s * T, 0)
        seen |= {a_s * T + t for t in range(lo, min(tail, T))}
    else:
        if start % T:
            lo = start - a_s * T if patched else start - a_s
            seen |= {a_s * T + t for t in range(lo, T)}
        for b in range(r_s, r_e):
            seen |= {b * T + t for t in range(T)}
        if end % T:
            seen |= {(a_e - 1) * T + t for t in range(tail)}
    return seen


CASES = [(0, 32, 32), (1, 22, 32), (31, 33, 32), (0, 70, 32),
         (37, 70, 32), (64, 96, 32), (5, 5, 32), (5, 70, 32), (64, 100, 32)]

print('--- slot count: allocation vs addressing')
for l, u, T in CASES:
    dv = DataView(shape=[max(u, 1)], permute=None, bbox=BoundingBox([l], [u]))
    check(f'get_dim_slots(l={l}, u={u}, T={T})',
          dv.get_dim_slots(0, T), alloc_slots(l, u, T))
    old = old_addr_slots(l, u, T)
    if old != alloc_slots(l, u, T):
        print(f'       (old addressing formula said {old} -> dimension aliasing)')

print('\n--- LeadLoop coverage')
for start, end, T in CASES:
    if start >= end:
        continue
    check(f'LeadLoop({start}, {end}, T={T})',
          covered(start, end, T), set(range(start, end)))
    lost = set(range(start, end)) - covered(start, end, T, patched=False)
    if lost:
        print(f'       (unpatched head guard silently lost {len(lost)} elements)')

lanes = sorted(x % 32 for x in covered(1, 22, 32))
print(f'\nmask example: LeadLoop(1, 22, T=32) -> lanes {lanes[0]}..{lanes[-1]} '
      f'({len(lanes)} of 32 active)')


# --- register addressing under a slicing offset ---------------------------
# A register operand distributes the lead dimension across lanes, so only
# whole thread-blocks can be re-indexed by an address; non-lead dimensions
# take any offset.  `access_address` with writer=None exercises the string
# path, which needs no Writer or Context for register symbols.
from tensorforge.backend.symbol import (Symbol, SymbolType, LeadIndex,
                                        add_offset)
from tensorforge.backend.data_types import RegMemObject
from tensorforge.common.exceptions import GenerationError

print('\n--- register addressing with slicing offset')


def reg_symbol(lower, upper, threads):
    shape = [u for u in upper]
    sym = Symbol(name='r', stype=SymbolType.Register,
                 obj=RegMemObject('r', 1))
    sym.num_threads = threads
    sym.data_view = DataView(shape=shape, permute=None,
                             bbox=BoundingBox(list(lower), list(upper)))
    return sym


T = 32
sym = reg_symbol([0, 0], [64, 8], T)          # 2 lead slots, 8 non-lead

base = sym.access_address(None, [LeadIndex('nl', T, 1), 3])
lead_block = sym.access_address(None,
                                [add_offset(LeadIndex('nl', T, 1), T), 3])
nonlead = sym.access_address(None, [LeadIndex('nl', T, 1), add_offset(3, 4)])

check('lead offset of one whole block shifts the block index by 1',
      lead_block, base.replace('(nl)', '(nl + 1)'))
check('non-lead offset shifts the plain index',
      nonlead, base.replace('(3)', '(7)'))

try:
    sym.access_address(None, [add_offset(LeadIndex('nl', T, 1), 5), 3])
    check('lead offset of 5 (not a multiple of 32) is rejected', 'accepted',
          'rejected')
except AssertionError:
    check('lead offset of 5 (not a multiple of 32) is rejected', 'rejected',
          'rejected')

sys.exit(1 if fail else 0)
