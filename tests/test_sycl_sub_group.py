# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The sub-group an SPMD SYCL multiplication lives in, and how a broadcast
reads its lanes.

The kernel required a sub-group of 16 whatever the multiplication, while the
lane search puts 32 lanes on one (`lanes.DEFAULT_LANE_CEILING`, deliberately
not the 16-wide vector unit).  A multiplication then spanned two sub-groups,
and `group_broadcast` -- one index for the whole sub-group -- read lanes that
were not there: on the OpenCL CPU device local_flux wrote nothing and
chain_five 1e27.  Now the sub-group follows the multiplication where the
kernel states it (oneAPI on Intel), and where it cannot -- acpp, a plug-in --
each multiplication reads its lanes at its own base.
"""
import re

import pytest

from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context
from tensorforge.common.exceptions import GenerationError

SUB_GROUP = re.compile(r'reqd_sub_group_size\((\d+)\)')


def _lexic(backend):
    return Context(arch='pvc', backend=backend,
                   fp_type=Datatype.F32).get_vm().get_lexic()


@pytest.mark.parametrize('lanes,size', [(32, 32), (16, 16), (8, 16), (4, 16),
                                        (24, None), (64, None)])
def test_a_multiplication_is_held_by_one_sub_group(lanes, size):
    assert _lexic('oneapi').sub_group_for(lanes) == size


def _oneapi_kernel(threads=None):
    import contextlib
    import importlib.util
    import io
    from tensorforge.common.context import Options
    from tensorforge.generators.generator import Generator
    from tensorforge.generators.lanes import LaneConfig
    from test_amd_packed_dpp import CASES
    path = next(CASES.rglob('local_flux.py'))
    spec = importlib.util.spec_from_file_location('tf_sg__lf', path)
    case = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(case)
    ctx = Context(arch='pvc', backend='oneapi', fp_type=case.DTYPE,
                  options=Options())
    lanes = (None if threads is None else
             LaneConfig(num_threads=threads, num_active_threads=threads,
                        lead_width=1))
    gen = Generator(case.descr_list(), ctx, lanes=lanes)
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    return gen.get_kernel()


@pytest.mark.parametrize('threads,size', [(None, 32), (16, 16)])
def test_the_oneapi_kernel_requires_its_multiplications_sub_group(threads,
                                                                  size):
    src = _oneapi_kernel(threads)
    assert SUB_GROUP.findall(src) == [str(size)]
    # the sub-group is the multiplication, so one index serves it
    assert 'sycl::group_broadcast(item.get_sub_group()' in src
    assert 'select_from_group' not in src


def test_where_the_size_is_the_devices_each_multiplication_reads_at_its_base():
    lexic = _lexic('acpp')
    text = lexic.broadcast('v', 3, 16)
    assert text == ('sycl::select_from_group(item.get_sub_group(), v, '
                    '(item.get_sub_group().get_local_linear_id() / 16) * 16 '
                    '+ (3))')


def test_a_narrow_multiplication_under_a_stated_sub_group_reads_at_its_base():
    lexic = _lexic('oneapi')
    assert 'select_from_group' in lexic.broadcast('v', 3, 8)
    assert lexic.broadcast('v', 3, 16).startswith('sycl::group_broadcast(')


def test_a_multiplication_no_sub_group_holds_is_refused():
    with pytest.raises(GenerationError, match='24 lanes'):
        _lexic('oneapi').broadcast('v', 3, 24)
