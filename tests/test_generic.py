# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The architecture ``generic``: any device AdaptiveCpp compiles the kernels
for when they run, the host included.

Nothing about that device is known while the code is generated, so its limits
are ones any SYCL device meets, and its wave is one lane: the size of a
sub-group is the device's (one on the host), so no lane may read the values of
another, and one lane holds a whole multiplication.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context
from tensorforge.common.target import Target
from tensorforge.generators.generator import Generator

CASES = Path(__file__).parent / "cases"

# what SYCL reads the values of another lane with
CROSS_LANE = re.compile(r'select_from_group|shift_group_\w+|permute_group_by_xor'
                        r'|group_broadcast|reduce_over_group')


def _case(name):
    spec = importlib.util.spec_from_file_location(
        f"_generic_{name.replace('/', '_')}", CASES / f"{name}.py")
    case = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(case)
    return case


def test_only_adaptivecpp_compiles_for_it():
    assert Target('generic', 'acpp').hw.vendor == 'generic'
    for backend in ('cuda', 'hip', 'oneapi'):
        with pytest.raises(ValueError, match='generic'):
            Target('generic', backend)


def test_its_limits_are_those_of_any_sycl_device():
    hw = Target('generic', 'acpp').hw
    assert hw.vec_unit_length == 1
    assert hw.max_local_mem_size_per_block == 32 * 1024


def test_its_reductions_are_rolled():
    """One lane would otherwise unroll every product whole."""
    rolled = Context(arch='generic', backend='acpp', fp_type=Datatype.F32)
    assert rolled.get_user_options().k_roll == 1
    device = Context(arch='pvc', backend='acpp', fp_type=Datatype.F32)
    assert device.get_user_options().k_roll == 0


@pytest.mark.parametrize('name', ['chain', 'f64', 'addressing_ptr_based',
                                  'reduction/max_all'])
def test_no_lane_reads_the_values_of_another(name):
    case = _case(name)
    for arch, crossing in (('generic', False), ('pvc', True)):
        gen = Generator(case.descr_list(),
                        Context(arch=arch, backend='acpp', fp_type=case.DTYPE))
        gen.generate()
        assert bool(CROSS_LANE.search(gen.get_kernel())) is crossing, arch
