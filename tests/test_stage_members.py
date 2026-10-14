# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""`stage_members`: a merged run's batch-constant operand, staged per
iteration into a buffer the block shares.

The operators a merged run walks (`merge_variants`) are read from global memory
by every multiplication -- the preload cannot take them, since a table over
shared copies would claim global memory.  Staged, the block copies the current
one cooperatively between two block barriers, and the block is one group on
the same trips through the batch loop.
"""
import re

import pytest

from test_amd_blgp import _kernel_at

STAGED = re.compile(r'float \* __restrict__ glb_(v\d+) = &totalShrMem\[\d+\];')
BINDING = re.compile(r'const glb_(v\d+)g = ')


def _variant_loop(src):
    """The merged run's loop body, up to the end of the first product."""
    start = src.index('for (int32_t batchIdv')
    return src[start:start + 4000]


@pytest.mark.parametrize('arch', ['gfx942', 'gfx1150'])
def test_the_current_member_is_copied_by_the_block_between_barriers(arch):
    src = _kernel_at('local_flux', arch, merge_variants=True,
                     stage_members=True)
    body = _variant_loop(src)
    # the buffer's window is declared ahead of the loop, as every window
    # its body fills (`VariantLoop._declare_windows_early`)
    staged = STAGED.findall(src[:src.index('for (int32_t batchIdv')])
    assert staged and BINDING.findall(body) == staged
    # the table still selects global pointers; the copy sits between barriers
    copy = body.index(f'// glb_{staged[0]} = load{{g>s}}(glb_{staged[0]}g')
    assert body[:copy].count('__syncthreads();') == 1
    assert '__syncthreads();' in body[copy:]
    # every multiplication of the block on the same trips: one group of 8 --
    # spelled in the loop start where it is IR (a multiplication narrower
    # than the wave) and through `batchIdLane0` where it is text
    assert 'threadIdx.y % 8' in src


def test_a_smaller_group_is_asked_for_by_stage_group():
    src = _kernel_at('local_flux', 'gfx942', merge_variants=True,
                     stage_members=True, stage_group=4)
    assert 'threadIdx.y % 4' in src
    assert 'threadIdx.y % 8' not in src


def test_without_it_members_are_read_from_global():
    src = _kernel_at('local_flux', 'gfx942', merge_variants=True)
    assert not STAGED.search(src)
    assert not BINDING.search(src)


def test_the_group_stays_within_the_named_barriers():
    """On CUDA a multiplication wider than a warp meets on a named barrier
    of its own, `threadIdx.y + 1`, and a CTA has fifteen of them besides the
    block's: a group of sixteen would have its last multiplication ask for
    barrier 16, which does not exist.  So the group is held to fifteen."""
    import contextlib
    import importlib.util
    import io
    import json
    import warnings
    from pathlib import Path

    from tensorforge.common.context import Context
    from tensorforge.common.options import Options
    from tensorforge.generators.generator import Generator

    path = next((Path(__file__).resolve().parent / 'cases').rglob(
        'local_flux.py'))
    spec = importlib.util.spec_from_file_location('tf_stage__' + path.stem,
                                                  path)
    case = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(case)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        ctx = Context(arch='sm_86', backend='cuda', fp_type=case.DTYPE,
                      options=Options(merge_variants=True, stage_members=True,
                                      stage_group=16, lanes_per_mult=64))
        gen = Generator(case.descr_list(), ctx)
        gen.generate()
    src = gen.get_kernel()
    assert STAGED.search(src), 'no member staged; nothing to test'
    launch = json.loads(re.search(r'tensorforge-meta: (\{.*\})',
                                  src).group(1))['launch']
    assert launch['threads_per_mult'] == 64 and 'barrier.sync' in src
    assert launch['block'][1] == 15, launch
