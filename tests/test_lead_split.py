# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The two-level lead (`Options.lead_split`).

A fused build of eight simulations puts them on the lanes, and a sub-group
of 16 then leaves half of them idle.  Two blocks of the destination's axis 1
on the high lanes keep every lane busy and halve what a lane holds: lane `t`
holds simulation `t % 8` and the rows `t // 8, t // 8 + 2, ...`.  A product
over the split axis reads the other block from the partner lane, `t ^ 8`
(`Symbol._exchanged`).

The numbers are checked on the CPU against NumPy (`~/tf-probe/esimd`'s
emulator): every s8 kernel of SeisSol's elastic system at orders 3 and 4
agrees, bit for bit in double precision.  What is checked here is the code
that makes it so.
"""
from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import pytest

from tensorforge.backend.pir import BOOL, IRBuilder, MemSpace, Op, ScalarType
from tensorforge.backend.pir.core import walk_stmts
from tensorforge.backend.pir.passes import if_convert
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context, Options
from tensorforge.generators.generator import Generator

F64 = ScalarType(Datatype.F64)
BENCH = Path(__file__).resolve().parent.parent / 'tools' / 'bench'


def _kernel(capture, workload, **opts):
    if str(BENCH) not in sys.path:
        sys.path.insert(0, str(BENCH))
    import suite as suite_mod
    w = {x.name: x for x in suite_mod.from_dump(
        BENCH / 'data' / f'{capture}.json')}[workload]
    ctx = Context(arch='pvc', backend='oneapi', fp_type=w.datatype,
                  options=Options(**opts))
    gen = Generator(w.descrs(), ctx, attrs=w.attrs)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        gen.generate()
    return gen


def test_eight_simulations_fill_the_sixteen_lanes_of_local_flux():
    """`localFluxAll` runs sixteen lanes (one of its products leads with the
    twenty basis functions), and the products that lead with the eight
    simulations left half of them idle."""
    plain = _kernel('elastic-4-d-s8', 'gpu_localFluxAll', lead_split=1)
    split = _kernel('elastic-4-d-s8', 'gpu_localFluxAll', lead_split=2)
    assert plain.lanes.num_threads == split.lanes.num_threads == 16
    src = split.get_kernel()
    # `Q`'s twenty rows by nine quantities: ten rows a lane, not twenty.
    assert re.search(r'double r0\[90\]', src)
    assert re.search(r'double r0\[180\]', plain.get_kernel())
    # The product over the split axis reads the partner's block.
    assert re.search(r'permute_group_by_xor\([^;]*, 8\)', src)


def test_a_section_whose_lead_fills_its_lanes_is_left_alone():
    """Eight lanes for eight simulations, two multiplications a sub-group:
    nothing is idle, and splitting cost the derivative 1.3 to 2.2 times its
    time on the Max 1550 (sng2 560392)."""
    plain = _kernel('elastic-4-d-s8', 'gpu_volume', lead_split=1)
    split = _kernel('elastic-4-d-s8', 'gpu_volume', lead_split=2)
    assert plain.lanes.num_threads == split.lanes.num_threads == 8
    # The same code; the name and the banner carry the options.
    def code(gen):
        src = re.sub(r'kernel_kernel_[0-9a-f]+', 'k', gen.get_kernel())
        return re.sub(r'options: .*', '', src)
    assert code(split) == code(plain)


@pytest.mark.parametrize('arch,backend,split', [
    ('pvc', 'oneapi', 2), ('pvc', 'esimd', 1), ('pvc', 'acpp', 1),
    ('sm_90', 'cuda', 1), ('gfx942', 'hip', 1)])
def test_on_by_default_under_dpcpp_spmd_alone(arch, backend, split):
    ctx = Context(arch=arch, backend=backend, fp_type=Datatype.F64)
    assert ctx.get_user_options().lead_split == split


def test_off_where_the_lead_does_not_fit():
    # Order-4 single-simulation kernels lead with twenty rows: no split.
    gen = _kernel('elastic-4-d-s1', 'gpu_volume', lead_split=2)
    assert 'permute_group_by_xor' not in gen.get_kernel()


def test_no_collective_inside_a_lane_guard():
    """A sub-group collective in divergent code is undefined in SYCL: the
    split axis's ragged windows guard their writes by lane, and neither a
    broadcast nor an exchange may be written into one."""
    src = _kernel('elastic-4-d-s8', 'gpu_localFluxAll',
                  lead_split=2).get_kernel()
    depth, guarded = 0, False
    for line in src.splitlines():
        if re.search(r'^\s*if \(v\d+(_g)?\) \{', line):
            guarded, depth = True, 0
        if guarded:
            depth += line.count('{') - line.count('}')
            assert 'group_broadcast' not in line, line
            assert 'permute_group_by_xor' not in line, line
            if depth <= 0:
                guarded = False


def _exchange_in_guard():
    """`if (lane < 1) out[0] = xor(r[0]) * 2;` -- a register read taken
    across lanes inside a lane guard."""
    b = IRBuilder(fptype=Datatype.F64, arena='shrMem')
    regs = b.alloc(Datatype.F64, (4,), MemSpace.REGISTER, hint='r')
    out = b.alloc(Datatype.F64, (4,), MemSpace.REGISTER, hint='o')
    cond = b.op('lt', BOOL, b.thread_id('x'), 1, hint='g')
    with b.if_(cond):
        v = b.load(regs, 0, hint='data')
        x = b.rawexpr('xor({0})', v, type_=F64, pure=True, crosslane=True,
                      hint='x')
        b.store(out, 0, b.op('mul', F64, x, 2.0))
    return b.finish()


def test_if_conversion_leaves_a_register_read_for_another_lane_unmasked():
    """The partner needs the value, not the zero a masked-out lane loads."""
    body = if_convert(_exchange_in_guard())
    loads = [s for s in walk_stmts(body) if s.op == Op.LOAD]
    stores = [s for s in walk_stmts(body) if s.op == Op.STORE]
    assert loads and all(s.predicate is None for s in loads)
    assert stores and all(s.predicate is not None for s in stores)


def test_the_intermediates_stay_in_registers():
    """A product's result is read by the next in the lanes that wrote it,
    so the store into shared memory and the read back are taken away
    (`pir.forward`) -- as they are without the split.  Written as text, the
    two-level image's stores named the whole array, nothing could be taken
    from a register, and order-4 `neighboringFlux` went through shared
    memory and ran 16 to 28 % behind the default on the Max 1550."""
    plain = _kernel('elastic-4-d-s8', 'gpu_neighboringFlux[0]',
                    lead_split=1).get_kernel()
    split = _kernel('elastic-4-d-s8', 'gpu_neighboringFlux[0]',
                    lead_split=2).get_kernel()
    shared_write = re.compile(r'\bs\d+\[[^\]]*\] = ')
    assert not shared_write.search(plain)
    assert not shared_write.search(split)
    assert 'permute_group_by_xor' in split
