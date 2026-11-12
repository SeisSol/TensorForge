# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The broadcast of a multiplication narrower than its stated sub-group,
split: one `group_broadcast` per block of lanes, from a lane the whole
sub-group agrees on, and a select by the work-item's own block
(`tensorforge::split_broadcast`, `Options.sycl_split_broadcast`).

The other spelling of the same read, `select_from_group` at an index per
work-item, is what IGC turns into indirect register accesses; that one stays
for blocks beyond the option, at 0, and where the device picks the sub-group.
"""
import pytest

from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context, Options

from harness import syntax
from test_sycl_sub_group import _oneapi_kernel

SPLIT = 'tensorforge::split_broadcast'


def _lexic(backend='oneapi', **opts):
    return Context(arch='pvc', backend=backend, fp_type=Datatype.F32,
                   options=Options(**opts)).target.lexic


def test_two_multiplications_a_sub_group_read_by_split_broadcast():
    assert _lexic().broadcast('v', 3, 8) == (
        'tensorforge::split_broadcast<8, 16>(item.get_sub_group(), v, 3)')


def test_more_blocks_than_asked_read_at_an_index_per_work_item():
    assert 'select_from_group' in _lexic().broadcast('v', 3, 4)
    assert _lexic(sycl_split_broadcast=4).broadcast('v', 3, 4) == (
        'tensorforge::split_broadcast<4, 16>(item.get_sub_group(), v, 3)')


def test_zero_keeps_the_index_per_work_item():
    lexic = _lexic(sycl_split_broadcast=0)
    assert 'select_from_group' in lexic.broadcast('v', 3, 8)
    assert 'tensorforge_device/sycl_spmd.h' not in lexic.get_headers()


def test_a_whole_sub_group_needs_no_split():
    assert _lexic().broadcast('v', 3, 16).startswith('sycl::group_broadcast(')


def test_where_the_device_picks_the_sub_group_nothing_is_split():
    # Without a stated size the blocks of a sub-group are not known.
    lexic = _lexic('acpp')
    assert 'select_from_group' in lexic.broadcast('v', 3, 8)
    assert 'tensorforge_device/sycl_spmd.h' not in lexic.get_headers()


def test_the_split_brings_its_header():
    assert 'tensorforge_device/sycl_spmd.h' in _lexic().get_headers()


@pytest.fixture(scope='module')
def eight_lanes():
    return _oneapi_kernel(8)


def test_an_eight_lane_kernel_reads_by_split_broadcast(eight_lanes):
    assert f'{SPLIT}<8, 16>(item.get_sub_group()' in eight_lanes
    assert 'select_from_group' not in eight_lanes


@pytest.mark.skipif(syntax.compiler() is None,
                    reason='no host C++ compiler for the syntax check')
def test_the_eight_lane_kernel_parses(eight_lanes):
    result = syntax.check_source(eight_lanes, shim=syntax.SYCL_SHIM)
    assert result.ok, '\n'.join(result.errors())


# -- the lanes rule's floor ----------------------------------------------------

def _floor(arch='pvc', backend='oneapi', **opts):
    from tensorforge.generators.tuning import lane_floor
    return lane_floor(Context(arch=arch, backend=backend,
                              fp_type=Datatype.F32, options=Options(**opts)))


def test_the_floor_is_the_vector_unit_unless_split_lanes_are_allowed():
    assert _floor(lanes_split_floor=False) == 16
    assert _floor() == 8
    assert _floor(lanes_split_floor=True) == 8
    # Never below `lanes.MIN_LANES`, however many blocks a split serves.
    assert _floor(lanes_split_floor=True, sycl_split_broadcast=4) == 8


def test_without_a_split_the_floor_stays_at_the_vector_unit():
    assert _floor(lanes_split_floor=True, sycl_split_broadcast=0) == 16


def test_where_the_device_picks_the_sub_group_the_floor_does_not_move():
    from tensorforge.generators import lanes
    assert _floor('sm_90', 'cuda', lanes_split_floor=True) == lanes.MIN_LANES


def test_the_issue_mix_counts_every_broadcast_and_select_of_a_split():
    from types import SimpleNamespace
    from tensorforge.backend.pir.emit import _split_issues
    text = 'tensorforge::split_broadcast<8, 16>(item.get_sub_group(), {0}, 3)'
    assert _split_issues(SimpleNamespace(text=text)) == 3
    assert _split_issues(SimpleNamespace(
        text=text.replace('<8, 16>', '<4, 16>'))) == 7
    assert _split_issues(SimpleNamespace(
        text='sycl::group_broadcast(item.get_sub_group(), {0}, 3)')) == 1
