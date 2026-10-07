# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What a target is and can, asked of the one object that knows.

`common.target.Target` holds the device (`hw`), the spelling (`lexic`), and
answers every question whose answer decides what is emitted.  The questions
that have a test of their own elsewhere -- the nontemporal hint, the prefetch,
the packed FMA, the atomics, the sub-groups -- are pinned there; this file has
the rest, and the property the split is for: an answer that turns on the
device and the language together comes out right for each pair, including the
ones a vendor string or a backend string alone gets wrong.
"""

from __future__ import annotations

import pytest

from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context
from tensorforge.common.target import Target


# --------------------------------------------------------------------------- #
# The parts
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('backend,canonical', [('hipsycl', 'acpp'),
                                               ('dpcpp', 'oneapi'),
                                               ('esimd', 'esimd')])
def test_the_backend_is_named_as_asked(backend, canonical):
    """`esimd` stays apart from the `oneapi` device it runs on: a context
    rebuilt from the device's name would be the other lowering."""
    target = Target('pvc', backend)
    assert target.backend == canonical
    assert target.explicit_simd is (canonical == 'esimd')
    assert target.hw.backend in ('acpp', 'oneapi')


def test_a_context_holds_one():
    ctx = Context(arch='gfx942', backend='hip', fp_type=Datatype.F32)
    assert isinstance(ctx.target, Target)
    assert ctx.target.hw.model == 'gfx942'


def test_the_headers_are_the_runtimes_and_the_lowerings():
    headers = Target('sm_86', 'cuda').headers()
    assert headers[0] == 'tensorforge_aux.h'
    assert 'tensorforge_device/cuda.h' in headers


# --------------------------------------------------------------------------- #
# A rendezvous of one multiplication
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('threads,expected', [(8, True), (16, True),
                                              (12, False), (64, True),
                                              (48, False)])
def test_cuda_meets_a_run_of_lanes_or_whole_warps(threads, expected):
    """`__syncwarp` with a mask below the warp, `barrier.sync` with a count
    in whole warps above it."""
    assert Target('sm_86', 'cuda').sync_mult(threads) is expected


@pytest.mark.parametrize('arch,threads,expected', [
    ('gfx942', 16, True), ('gfx942', 64, True), ('gfx942', 128, False),
    ('gfx1250', 32, True), ('gfx1250', 64, False)])
def test_hip_meets_nothing_wider_than_a_wave(arch, threads, expected):
    """The barrier objects of gfx12.5 would need a prologue nothing emits."""
    assert Target(arch, 'hip').sync_mult(threads) is expected


def test_sycl_meets_a_multiplication_where_the_kernel_states_it():
    """oneAPI on Intel states the sub-group, so a 32-lane multiplication is
    one; AdaptiveCpp does not, and the device picks."""
    assert Target('pvc', 'oneapi').sync_mult(32)
    assert not Target('pvc', 'oneapi').sync_mult(8)
    assert not Target('pvc', 'acpp').sync_mult(32)
    assert Target('pvc', 'esimd').sync_mult(48), 'one work-item, any width'


@pytest.mark.parametrize('arch,backend', [('sm_86', 'omptarget')])
def test_the_rest_meet_the_block(arch, backend):
    assert not Target(arch, backend).sync_mult(32)


def test_an_exchange_reaches_the_wave_or_the_stated_sub_group():
    assert Target('sm_86', 'cuda').exchange_reach(64) == 32
    assert Target('gfx942', 'hip').exchange_reach(16) == 64
    assert Target('pvc', 'oneapi').exchange_reach(32) == 32, (
        'the stated sub-group, not the 16-wide vector unit')
    assert Target('pvc', 'acpp').exchange_reach(32) == 16
    assert Target('pvc', 'esimd').exchange_reach(48) == 48


# --------------------------------------------------------------------------- #
# Memory and launch
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('arch,backend,sizes', [
    ('sm_86', 'cuda', (4, 8, 16)),
    ('sm_70', 'cuda', ()),
    ('gfx942', 'hip', (1, 2, 4)),
    ('gfx1100', 'hip', ()),
    ('sm_86', 'hip', (4, 8, 16)),
    ('sm_86', 'acpp', ()),
    ('pvc', 'oneapi', ()),
])
def test_an_async_copy_needs_the_device_and_the_spelling(arch, backend, sizes):
    """The path is the device's -- `cp.async` from sm_80, `global_load_lds`
    on CDNA2/3 -- and the sizes are the language's; either missing is none."""
    assert Target(arch, backend).copy_async_sizes() == sizes


def test_the_path_and_the_spelling_are_asked_apart():
    """AdaptiveCpp on an sm_86 has the hardware and no way to name it, which
    is a different note in the output than a part without the path."""
    target = Target('sm_86', 'acpp')
    assert target.async_copy_path() and not target.copy_async_sizes()
    assert not Target('sm_70', 'cuda').async_copy_path()


def test_only_nvidia_stages_through_async_copies():
    assert Target('sm_86', 'cuda').async_staging()
    assert Target('sm_86', 'hip').async_staging()
    assert not Target('gfx942', 'hip').async_staging()


def test_one_wait_counter_on_amd():
    assert Target('gfx942', 'hip').one_wait_counter()
    assert not Target('sm_86', 'cuda').one_wait_counter()


@pytest.mark.parametrize('arch,backend,expected', [
    ('sm_86', 'cuda', 128), ('gfx1200', 'hip', 128), ('pvc', 'oneapi', 64),
    ('pvc', 'esimd', 31 * 64)])
def test_one_prefetch_covers_a_line_or_a_run(arch, backend, expected):
    assert Target(arch, backend).prefetch_line_bytes() == expected


def test_a_wide_lead_is_spelled_by_cuda_and_hip_alone():
    assert Target('sm_86', 'cuda').lead_vectors()
    assert Target('gfx942', 'hip').lead_vectors()
    assert not Target('pvc', 'oneapi').lead_vectors()
    assert not Target('sm_86', 'acpp').lead_vectors()


def test_sycl_covers_the_batch_in_one_round():
    assert Target('sm_86', 'cuda').bounds_grid()
    assert not Target('pvc', 'oneapi').bounds_grid()
    assert not Target('sm_86', 'acpp').bounds_grid()


@pytest.mark.parametrize('arch,expected', [('sm_90', False), ('sm_100', True),
                                           ('sm_120', True)])
def test_launch_control_from_sm_100(arch, expected):
    assert Target(arch, 'cuda').launch_control() is expected
    assert not Target('gfx950', 'hip').launch_control()


def test_launch_bounds_name_resident_blocks_on_nvidia():
    assert Target('sm_86', 'cuda').min_blocks_bound()
    assert not Target('gfx942', 'hip').min_blocks_bound()
