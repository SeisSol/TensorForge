# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What the register model counts, against what a compiler allocates.

Calibrated against ptxas on sm_100a over `local_flux` at b = 20..120, four lane
geometries and four option sets.  Two things read far high, and neither was a
matter of degree:

* Address offsets.  `lane + 2800` feeds a load, and the compiler keeps `lane`
  and puts 2800 in the instruction; the model kept each offset as a register.
  Where `licm` had hoisted them out of the loop over merged faces, the merged
  kernel peaked at 14864 B for a kernel ptxas fits in 255 registers.
* A peeled element written as text.  The last row of an odd extent at width
  two was a raw `r[k] = ...` statement, and an array named in raw text is
  taken as live whole for the whole body: 2312 B at 35 rows for 213
  registers.

Both are checked here on the smallest body that shows them.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from tensorforge.backend.pir import passes
from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import INDEX, MemSpace
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context
from tensorforge.generators.generator import Generator
from tensorforge.generators.lanes import LaneConfig

CASES = Path(__file__).resolve().parent / "cases"


def _offsets_then_loads(offset):
    """A hundred addresses computed up front, as `licm` leaves them, each
    read once afterwards into a running sum."""
    b = IRBuilder(fptype=Datatype.F32, scratch=('tempShrMem', 1 << 16))
    lane = b.rawexpr('threadIdx.x', type_=INDEX, hint='lane')
    with b.scratch_scope():
        tile = b.alloc(Datatype.F32, (1 << 15,), MemSpace.SHARED, hint='t')
        addrs = [offset(b, lane, i) for i in range(100)]
        total = b.load(tile, addrs[0], hint='d')
        for a in addrs[1:]:
            total = b.op('add', total.type, total, b.load(tile, a, hint='d'),
                         hint='s')
        b.store(tile, total, lane)
    return b.finish()


def test_an_address_offset_is_not_a_register():
    body = _offsets_then_loads(
        lambda b, lane, i: b.op('add', INDEX, lane, 56 * i, hint='a'))
    # the lane, the running sum and the load in flight -- not a hundred offsets
    assert passes.pressure(body, in_bytes=True, explicit_simd=False) < 64


def test_a_scaled_offset_is_one_register_per_scale():
    """`(lane + 64) * 4 + 3584 + i` is `lane * 4` and an immediate."""
    def scaled(b, lane, i):
        v = b.op('add', INDEX, lane, 64, hint='a')
        v = b.op('mul', INDEX, v, 4, hint='a')
        return b.op('add', INDEX, v, 3584 + i, hint='a')
    one = passes.pressure(_offsets_then_loads(scaled), in_bytes=True,
                          explicit_simd=False)
    assert one < 64

    def two_scales(b, lane, i):
        v = b.op('mul', INDEX, lane, 4 if i % 2 else 8, hint='a')
        return b.op('add', INDEX, v, i, hint='a')
    two = passes.pressure(_offsets_then_loads(two_scales), in_bytes=True,
                          explicit_simd=False)
    assert two == one + 4, 'a second scale is a second register, and one only'


def test_an_offset_keeps_what_it_offsets_live():
    """The register is the root's: folding the offset away must not let the
    root die before the last load through it."""
    body = _offsets_then_loads(
        lambda b, lane, i: b.op('add', INDEX, lane, 56 * i, hint='a'))
    order, _ = passes._index(body)
    keep, carriers = passes._affine_costs(order)
    assert keep and not carriers
    assert len(set(keep.values())) == 1


def _local_flux(rows):
    spec = importlib.util.spec_from_file_location(
        "tf_pressure__local_flux", CASES / "local_flux.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._M, mod._PAD = rows, rows + 1
    return mod


def _peak(mod, width):
    ctx = Context(arch='sm_100', backend='cuda', fp_type=mod.DTYPE)
    ctx.measure_pressure = True
    gen = Generator(mod.descr_list(), ctx,
                    lanes=LaneConfig(16, mod._M, width))
    gen.generate()
    return gen.peak_pressure


def test_a_peeled_element_is_one_slot_and_not_the_whole_array():
    """35 rows at width two peel the last one.  Its store names one slot; as
    raw text it made every register array of the kernel live throughout, four
    times the figure at width one (ptxas: 213 registers against 146)."""
    mod = _local_flux(35)
    assert _peak(mod, 2) < 2 * _peak(mod, 1)


def _mixed(uniform: int, lane: int):
    """`uniform` values read from one address and `lane` values read from a
    lane-dependent one, all live at once and summed at the end."""
    b = IRBuilder(fptype=Datatype.F32, scratch=('tempShrMem', 1 << 16))
    # `lane_index`, not the raw text the helper above uses: a raw expression
    # takes the uniformity of its arguments and has none here, so it would
    # claim to be the same on every lane.
    here = b.lane_index(32, 1, hint='lead')
    with b.scratch_scope():
        tile = b.alloc(Datatype.F32, (1 << 15,), MemSpace.SHARED, hint='t')
        held = [b.load(tile, b.const(7 * i, INDEX), hint='u')
                for i in range(uniform)]
        held += [b.load(tile, b.op('add', INDEX, here, 4096 * i, hint='a'),
                        hint='l') for i in range(lane)]
        total = held[0]
        for v in held[1:]:
            total = b.op('add', total.type, total, v, hint='s')
        b.store(tile, total, here)
    return b.finish()


def test_the_two_register_files_are_counted_apart():
    """A value a whole wave agrees on is a scalar register on AMD, and the
    lane-varying ones are the vector file; a single figure sees neither fill.
    The two peaks are taken at their own program points, so each is at most
    the total and they need not add up to it."""
    split: list = []
    body = _mixed(uniform=8, lane=8)
    total = passes.pressure(body, in_bytes=True, explicit_simd=False,
                            by_file=split)
    lane, uniform = split
    assert uniform >= 8 * 4, 'the values read from one address are uniform'
    assert lane >= 8 * 4, 'the lane-dependent ones are not'
    assert max(lane, uniform) <= total <= lane + uniform


def test_what_counts_as_uniform_follows_the_geometry():
    """Mult-uniform is wave-uniform only where a multiplication is at least a
    wave wide; below that a wave holds several and they disagree.  Asked at
    the threshold rather than assumed, so a narrow geometry puts those values
    back in the vector file."""
    from tensorforge.backend.pir.core import Participants, Uniformity

    assert Participants.WAVE.arrival(32, 32) is Uniformity.MULT
    assert Participants.WAVE.arrival(16, 32) is Uniformity.MULTGROUP

    body = _mixed(uniform=8, lane=8)
    wide: list = []
    narrow: list = []
    passes.pressure(body, in_bytes=True, explicit_simd=False, by_file=wide,
                    wave_uniform=Uniformity.MULT)
    passes.pressure(body, in_bytes=True, explicit_simd=False, by_file=narrow,
                    wave_uniform=Uniformity.GRID)
    assert narrow[1] <= wide[1], 'a stricter threshold cannot hold more'
