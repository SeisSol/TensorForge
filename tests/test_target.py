# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What a target is, can and prefers, asked of the one object that knows.

`common.target.Target` holds the device (`hw`), the spelling (`lexic`), and
answers every question whose answer decides what is emitted.  The questions
that have a test of their own elsewhere -- the nontemporal hint, the prefetch,
the packed FMA, the atomics, the sub-groups -- are pinned there; this file has
the rest, and the property the split is for: an answer that turns on the
device and the language together comes out right for each pair, including the
ones a vendor string or a backend string alone gets wrong.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context
from tensorforge.common.exceptions import GenerationError
from tensorforge.common.target import PREFERENCES, Preferences, Target

CASES = Path(__file__).parent / "cases"


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


@pytest.mark.parametrize('arch,family', [('sm_86', 'nvidia'),
                                         ('gfx942', 'gfx9'),
                                         ('gfx90a', 'gfx9'),
                                         ('gfx1150', 'gfx1'),
                                         ('gfx1250', 'gfx1'),
                                         ('pvc', 'intel')])
def test_the_calibration_family(arch, family):
    backend = ('cuda' if arch.startswith('sm_') else
               'hip' if arch.startswith('gfx') else 'oneapi')
    assert Target(arch, backend).hw.family == family


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


def test_intel_spmd_lanes_share_a_thread():
    assert Target('pvc', 'oneapi').lanes_share_register_file()
    assert not Target('pvc', 'esimd').lanes_share_register_file()
    assert not Target('gfx942', 'hip').lanes_share_register_file()


# --------------------------------------------------------------------------- #
# A grid-wide barrier
# --------------------------------------------------------------------------- #

def test_sycl_has_no_grid_barrier():
    assert Target('sm_86', 'cuda').grid_barrier()
    assert not Target('pvc', 'oneapi').grid_barrier()
    assert not Target('pvc', 'esimd').grid_barrier()


@pytest.mark.parametrize('backend', ['acpp', 'esimd'])
def test_a_kernel_that_needs_one_is_refused_by_name(backend):
    """Refused by `verify` before emission, with the reason, and not by an
    unimplemented spelling halfway through it."""
    spec = importlib.util.spec_from_file_location(
        'barrier_two_gemms', CASES / 'barrier' / 'barrier_two_gemms.py')
    case = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(case)
    from tensorforge.generators.generator import Generator
    ctx = Context(arch='pvc', backend=backend,
                  fp_type=getattr(case, 'DTYPE', None))
    with pytest.raises(GenerationError, match='grid barrier'):
        Generator(case.descr_list(), ctx).generate()


# --------------------------------------------------------------------------- #
# Preferences
# --------------------------------------------------------------------------- #

def test_a_vendor_without_a_row_takes_the_plain_preferences():
    """Correct and slow: nothing staged, nothing kept, nothing tuned."""
    plain = Preferences()
    assert not plain.placement.preload_operands_into_registers
    assert not plain.preload_globals and not plain.tune_matrix_path
    assert plain.register_fit is None


def test_every_vendor_has_a_row():
    assert set(PREFERENCES) == {'nvidia', 'amd', 'intel'}


def test_the_explicit_vector_changes_what_it_cannot_express():
    """Staging is the default there, the broadcast in place is not available,
    and the lane count is the vector's -- the rest of the row stands."""
    spmd, esimd = Target('pvc', 'oneapi').prefs, Target('pvc', 'esimd').prefs
    assert not spmd.preload_globals and esimd.preload_globals
    assert spmd.placement.broadcast_without_staging
    assert not esimd.placement.broadcast_without_staging
    assert spmd.tune_preload_globals and not esimd.tune_preload_globals
    assert spmd.lanes_down_to_wave and not esimd.lanes_down_to_wave
    assert spmd.split_predicated_load == esimd.split_predicated_load


@pytest.mark.parametrize('arch,backend,option,expected', [
    ('gfx942', 'hip', 'preload_globals', True),
    ('sm_86', 'cuda', 'preload_globals', False),
    ('pvc', 'esimd', 'preload_globals', True),
    ('pvc', 'oneapi', 'preload_globals', False),
    ('sm_86', 'cuda', 'inline_constants', 4096),
    ('gfx942', 'hip', 'inline_constants', 64),
    ('sm_86', 'cuda', 'argument_constants', True),
    ('pvc', 'oneapi', 'split_predicated_load', True),
])
def test_an_option_nobody_set_takes_the_preference(arch, backend, option,
                                                   expected):
    ctx = Context(arch=arch, backend=backend, fp_type=Datatype.F32)
    assert getattr(ctx.get_user_options(), option) == expected
