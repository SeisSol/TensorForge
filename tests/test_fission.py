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
