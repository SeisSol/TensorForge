# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""`Options.sycl_core_threads`: the grid-stride traversal's grid sized for a
number of hardware threads an Xe-core under DPC++ on Intel.

The launcher asks the device for its Xe-cores and hands them with the
work-group's shape to `tensorforge::coreGrid` (`base.h`, checked by
`tests/cpp/core_grid.cpp`); the kernel loops over its share of the batch in
the batch's parts.  0, the default, leaves every launcher as it was.
"""

import contextlib
import io
import re
import warnings

import pytest

import seissol_suite as suite
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context, Options
from tensorforge.common.exceptions import GenerationError
from tensorforge.frontend.yateto import DescriptionReader
from tensorforge.generators.generator import Generator


def _generate(backend, arch='pvc', **options):
    descrs = DescriptionReader(None, {}).read(suite.description(
        'elastic-linearck', 'elastic-linearck-o4-s', 'gpu_volume'))[0]
    gen = Generator(descrs, Context(arch=arch, backend=backend,
                                    fp_type=Datatype.F32,
                                    options=Options(**options)))
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        gen.generate()
    return gen


_CALL = re.compile(r'tensorforge::coreGrid\(cores, (\d+), (\d+), (\d+), (\d+), '
                   r'(\d+)\);')


def _block(gen):
    x, y, z = gen.launch_config().block
    return x * y * z


@pytest.mark.parametrize('backend', ['oneapi', 'esimd'])
def test_the_default_launcher_is_unchanged(backend):
    """Unset and set to 0 are one build, and neither asks for the
    Xe-cores."""
    plain, zero = _generate(backend), _generate(backend, sycl_core_threads=0)
    assert plain.get_launcher() == zero.get_launcher()
    assert plain.get_kernel() == zero.get_kernel()
    assert 'coreGrid' not in plain.get_launcher()


def test_spmd_counts_a_sub_group_a_thread():
    gen = _generate('oneapi', sycl_core_threads=32)
    launcher = gen.get_launcher()
    m = _CALL.search(launcher)
    assert m, 'the grid by the Xe-cores'
    threads, group_threads, rows, parts, block = (int(x) for x in m.groups())
    sub = gen._context.target.lexic.sub_group_size(gen._num_threads)
    assert (threads, group_threads, rows) == (
        32, _block(gen) // sub, gen.launch_config().mults_per_block)
    assert (parts, block) == (2, 1)
    assert 'gpu_eu_count_per_subslice' in launcher
    assert 'config.grid[0] = std::min(gridsize, numElements0);' in launcher


def test_the_explicit_vector_counts_a_work_item_a_thread():
    gen = _generate('esimd', sycl_core_threads=16)
    m = _CALL.search(gen.get_launcher())
    assert m
    threads, group_threads, rows, _, _ = (int(x) for x in m.groups())
    assert (threads, group_threads) == (16, _block(gen))
    assert rows == gen.launch_config().mults_per_block


def test_the_kernel_loops_over_its_part_of_the_batch():
    """Under SYCL a grid otherwise covers the batch in one round; sized by
    the Xe-cores it strides over the parts `batch_parts` cuts (2 under
    SYCL)."""
    one_round = _generate('oneapi', sycl_dpas=False).get_kernel()
    looped = _generate('oneapi', sycl_core_threads=16).get_kernel()
    assert 'partStride' not in one_round
    assert 'partStride' in looped


@pytest.mark.parametrize('other', [{'one_round_grid': True},
                                   {'resident_grid': 1.0}])
def test_one_sizing_at_a_time(other):
    with pytest.raises(GenerationError, match='sycl_core_threads'):
        _generate('oneapi', sycl_core_threads=32, **other)


def test_auto_one_round_yields_to_the_xe_cores(monkeypatch):
    """`one_round_grid=auto`, the default, covers the batch in one round
    where the kernel takes the slot form (`element_group.slot_one_round`,
    held true here: no Intel kernel takes it yet).  Asked for a grid by the
    Xe-cores, the grid loops, and the launch says so."""
    from tensorforge.generators import element_group
    monkeypatch.setattr(element_group, 'slot_one_round',
                        lambda group, context: True)
    plain = _generate('oneapi')
    assert plain._grid_one_round()
    assert not plain.launch_config().persistent
    sized = _generate('oneapi', sycl_core_threads=16, one_round_grid='auto')
    assert not sized._grid_one_round()
    assert sized.launch_config().persistent
    assert _CALL.search(sized.get_launcher())


def test_only_dpcpp_on_intel():
    with pytest.raises(GenerationError, match='DPC\\+\\+ grid on Intel'):
        _generate('cuda', arch='sm_86', sycl_core_threads=32)


def test_a_count_of_threads():
    with pytest.raises(GenerationError, match='a count of threads'):
        _generate('oneapi', sycl_core_threads=-1)
