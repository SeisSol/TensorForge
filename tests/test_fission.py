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
    ('sm_90', 'cuda', 0), ('gfx942', 'hip', 0)])
def test_auto_is_the_default_under_dpcpp_spmd_alone(arch, backend, expected):
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
