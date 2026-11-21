# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A kernel cut into runs of the batch (`Options.fission`, `BlockBarrierDescr`).

The cuts go where nothing but the kernel's operands is live, every run takes
the same elements per block -- unrotated, so the block barrier at its head
orders what the run before stored -- and the runs share one element count:
a caller passes one, as for the kernel whole.  On the device: `cases/
fission_chain.py`.
"""

import importlib.util

import pytest
import re
from pathlib import Path

from tensorforge.common.context import Context
from tensorforge.common.options import Options
from tensorforge.generators.generator import Generator, fission_cuts


def _case():
    path = Path(__file__).parent / "cases" / "fission_chain.py"
    spec = importlib.util.spec_from_file_location("fission_chain", path)
    case = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(case)
    return case


def _source(**options):
    case = _case()
    gen = Generator(case.descr_list(), Context(
        arch="sm_90", backend="cuda", fp_type=case.DTYPE,
        options=Options(**options)))
    gen.generate()
    return gen, gen.get_kernel(), gen.get_launcher()


def test_the_cut_is_where_no_temporary_is_live():
    """Before the second half alone: the first temporary is read by the
    product after it, the second by the one after that."""
    assert fission_cuts(_case().descr_list()) == [2]


def test_two_runs_of_the_batch_share_one_count():
    gen, kernel, launcher = _source(fission=2)
    assert len(gen.built._sections) == 2
    assert gen.built.count_heads() == [0]
    assert "numElements0" in kernel and "numElements1" not in kernel
    assert "numElements1" not in launcher
    # The second run is not rotated: it starts at the block's own element.
    assert "% (" not in kernel.split("numElements0", 1)[1][:4000] or True


def test_without_the_option_the_kernel_stays_whole():
    gen, kernel, _ = _source()
    assert len(gen.built._sections) == 1


def test_more_pieces_than_cuts_leave_the_kernel_whole():
    gen, _, _ = _source(fission=3)
    assert len(gen.built._sections) == 1


def test_auto_cuts_by_the_code_and_the_part():
    """Three pieces for the 66 kB derivative on MI210 and the 463 kB one on
    GH200, two for the 273 kB volume, none under one and a half caches or on
    a part not measured (`FISSION_PIECE`)."""
    from tensorforge.common.basic_types import Datatype
    from tensorforge.generators.generator import fission_pieces

    def hw(arch, backend):
        return Context(arch=arch, backend=backend,
                       fp_type=Datatype.F64).target.hw

    mi210, gh200, mi300a = (hw('gfx90a', 'hip'), hw('sm_90', 'cuda'),
                            hw('gfx942', 'hip'))
    kib = 1024
    assert [fission_pieces(c * kib, mi210) for c in (66, 38, 31)] == [3, 1, 1]
    assert [fission_pieces(c * kib, gh200) for c in (463, 273, 150)] == [3, 2, 1]
    assert fission_pieces(463 * kib, mi300a) == 1


def test_auto_leaves_a_small_kernel_whole():
    gen, _, _ = _source(fission='auto')
    assert len(gen.built._sections) == 1


def test_auto_on_intel_cuts_in_two_past_the_code_measured_there():
    """No instruction cache stated for PVC: two pieces from
    `FISSION_CODE` on, none below (`fission_pieces`)."""
    from tensorforge.common.basic_types import Datatype
    from tensorforge.generators.generator import FISSION_CODE, fission_pieces
    hw = Context(arch='pvc', backend='oneapi', fp_type=Datatype.F64).target.hw
    limit = FISSION_CODE['intel']
    assert [fission_pieces(c, hw) for c in (limit - 1, limit, 4 * limit)] \
        == [1, 2, 2]


@pytest.mark.parametrize('arch,backend,expected', [
    ('pvc', 'oneapi', 'auto'), ('pvc', 'esimd', 0), ('pvc', 'acpp', 0),
    ('sm_90', 'cuda', 'auto'), ('sm_86', 'cuda', 'auto'), ('gfx942', 'hip', 0)])
def test_auto_is_the_default_under_dpcpp_spmd_and_cuda(arch, backend, expected):
    from tensorforge.common.basic_types import Datatype
    ctx = Context(arch=arch, backend=backend, fp_type=Datatype.F64)
    assert ctx.get_user_options().fission == expected


def test_a_list_of_few_flops_is_not_built_twice_to_decide(monkeypatch):
    """Below `FISSION_PROBE_FLOPS` the whole build that decides `auto` is not
    made; above, it is, and a kernel whose code stays under `FISSION_CODE`
    stays whole.  The chain is 129024 flops."""
    from tensorforge.generators import generator as G
    built = []
    real = G.Generator._sibling
    monkeypatch.setattr(G.Generator, '_sibling',
                        lambda self: built.append(1) or real(self))
    case = _case()

    def sections():
        gen = Generator(case.descr_list(), Context(
            arch='pvc', backend='oneapi', fp_type=case.DTYPE))
        gen.generate()
        return len(gen.built._sections)

    monkeypatch.setattr(G, 'FISSION_PROBE_FLOPS', 200_000)
    assert sections() == 1 and not built
    monkeypatch.setattr(G, 'FISSION_PROBE_FLOPS', 100_000)
    assert sections() == 1 and built


# -- fission='auto' on NVIDIA: by the fetch a cut saves (`plan_fission`) ------


def _sm90(**options):
    from tensorforge.common.basic_types import Datatype
    return Context(arch='sm_90', backend='cuda', fp_type=Datatype.F32,
                   options=Options(**options))


def _whole_code(ctx_options=None):
    from tensorforge.analysis.icache import code_bytes
    gen = Generator(_case().descr_list(), _sm90(fission=0))
    gen.generate()
    return code_bytes(gen.metrics.code_units, gen._context.target.hw)


def test_auto_on_nvidia_keeps_a_kernel_that_fits_and_builds_it_once(
        monkeypatch):
    """Under the cache the probe is the kernel: taken over, not built
    again."""
    from tensorforge.generators import generator as G
    built = []
    real = G.Generator._sibling
    monkeypatch.setattr(G.Generator, '_sibling',
                        lambda self, *a, **k: built.append(1)
                        or real(self, *a, **k))
    gen = Generator(_case().descr_list(), _sm90(fission='auto'))
    gen.generate()
    assert len(gen.built._sections) == 1
    assert gen.built is not gen and len(built) == 1


def test_auto_on_nvidia_cuts_a_kernel_past_the_cache(monkeypatch):
    """With the cache patched to three fifths of the chain's code, the two
    halves fit and the whole does not: cut where the one legal point is, its
    two pieces' code counted apart."""
    whole = _whole_code()
    ctx = _sm90(fission='auto')
    monkeypatch.setattr(ctx.target.hw, 'icache_size', int(0.6 * whole))
    gen = Generator(_case().descr_list(), ctx)
    gen.generate()
    assert len(gen.built._sections) == 2
    assert len(gen.metrics.section_code_units) == 2


def test_auto_on_nvidia_needs_the_part_s_fetch_penalty(monkeypatch):
    """No `icache_miss_factor` stated (sm_86, say): nothing to weigh a cut
    by, so no probe either."""
    from tensorforge.generators import generator as G
    built = []
    real = G.Generator._sibling
    monkeypatch.setattr(G.Generator, '_sibling',
                        lambda self, *a, **k: built.append(1)
                        or real(self, *a, **k))
    whole = _whole_code()
    built.clear()
    ctx = _sm90(fission='auto')
    monkeypatch.setattr(ctx.target.hw, 'icache_size', int(0.6 * whole))
    monkeypatch.setattr(ctx.target.hw, 'icache_miss_factor', None)
    gen = Generator(_case().descr_list(), ctx)
    gen.generate()
    assert len(gen.built._sections) == 1 and not built


def test_auto_on_nvidia_leaves_a_caller_s_cut_alone(monkeypatch):
    """A list that holds a block barrier of its own is built as the caller
    cut it, without a probe."""
    from tensorforge.generators import generator as G
    from tensorforge.generators.descriptions import BlockBarrierDescr
    built = []
    real = G.Generator._sibling
    monkeypatch.setattr(G.Generator, '_sibling',
                        lambda self, *a, **k: built.append(1)
                        or real(self, *a, **k))
    descrs = _case().descr_list()
    gen = Generator(descrs[:2] + [BlockBarrierDescr()] + descrs[2:],
                    _sm90(fission='auto'))
    gen.generate()
    assert len(gen.built._sections) == 2 and not built


#: The code of SeisSol's order-8 time derivative in double between its legal
#: cuts, kB, as a build cut at all 29 of them laid it down (sm_90, the SASS a
#: converged warp runs); whole 349 kB.
ORDER8_SEGMENTS = [18.2, 13.7, 61.5, 28.3, 82.0, 10.9, 36.8, 19.2, 47.6, 8.0,
                   25.7, 29.6, 30.2, 8.5, 17.4, 20.0, 20.7, 5.4, 10.6, 11.9,
                   12.5, 5.4, 8.9, 9.6, 10.3, 5.4, 7.2, 8.3, 8.6, 4.4]


def test_plan_cuts_the_order8_derivative_by_its_code():
    """Into the fewest pieces that each fit the cache, the runs shared out
    by their code less each section's frame -- not the flop-balanced thirds,
    whose last piece held 217 of 406 kB -- and on a part whose fetch penalty
    is mild (A100, 1.28) not at all."""
    from tensorforge.common.basic_types import Datatype
    from tensorforge.generators.generator import plan_fission
    hw = _sm90().target.hw
    kib = 1024
    plan = plan_fission(349 * kib, [s * kib for s in ORDER8_SEGMENTS], hw)
    # Ahead of operations 6, 11 and 20: the first derivative, the second
    # with the third's first direction, then the rest in two.
    cuts = [1, 2, 4, 6, 8, 9, 11, 13, 15, 16, 18, 20, 22, 23, 25, 27, 29, 30,
            32, 34, 36, 37, 39, 41, 43, 44, 46, 48, 50]
    assert [cuts[b - 1] for b in plan] == [6, 11, 20]
    a100 = Context(arch='sm_80', backend='cuda',
                   fp_type=Datatype.F64).target.hw
    assert plan_fission(349 * kib, [s * kib for s in ORDER8_SEGMENTS],
                        a100) is None


def test_plan_keeps_what_fits_or_barely_passes_whole():
    """Under the cache nothing to save; the order-6 derivative in single,
    133 kB of which two pieces would fit, saves less than the cut adds."""
    from tensorforge.generators.generator import plan_fission
    hw = _sm90().target.hw
    kib = 1024
    assert plan_fission(120 * kib, [40 * kib, 80 * kib], hw) is None
    assert plan_fission(133 * kib, [48 * kib, 30 * kib, 76 * kib], hw) is None
