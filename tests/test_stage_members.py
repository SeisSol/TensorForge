# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""`stage_members`: a merged run's batch-constant operands, copied into the
block's shared memory a trip at a time (`pir.shards`).

The operators a merged run walks (`merge_variants`) are read from global memory
by every multiplication -- the preload cannot take them, since a table selects
one per trip.  Streamed, the block copies the shards the trip reads of the
current one at the head of the trip, between two block barriers, under the
budget the shards ahead of the batch loop take theirs from; and the block is
one group on the same trips through the batch loop.
"""
import contextlib
import importlib.util
import io
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

from test_amd_blgp import _kernel_at

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.context import Context
from tensorforge.common.helper import generate_tmp_matrix
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.common.options import Options
from tensorforge.generators.descriptions import GemmDescr
from tensorforge.generators.generator import Generator
from tensorforge.reference import kernel_eval as ke

SHARD = re.compile(r'(?:float|double) \* (v\d+_shard) = &totalShrMem\[\d+\];')
BINDING = re.compile(r'const (glb_v\d+) = ')


def _variant_loop(src):
    """The merged run's loop body, up to the end of the first product."""
    start = src.index('for (int32_t batchIdv')
    return src[start:start + 4000]


@pytest.mark.parametrize('arch', ['gfx942', 'gfx1150'])
def test_the_current_member_is_copied_by_the_block_between_barriers(arch):
    src = _kernel_at('local_flux', arch, merge_variants=True,
                     preload_shard_budget=65536)
    body = _variant_loop(src)
    # the buffer is the arena's, declared ahead of the batch loop
    shards = SHARD.findall(src[:src.index('for (int32_t batchIdv')])
    assert len(shards) == 1, shards
    # behind the binding of the trip's member: a barrier, the copy, a barrier
    first = body.index('__syncthreads();')
    second = body.index('__syncthreads();', first + 1)
    copy = body[first:second]
    copied = [b for b in BINDING.findall(body[:first]) if f'{b}[' in copy]
    assert f'{shards[0]}[' in copy and len(copied) == 1
    # and the products read the copy
    assert f'{shards[0]}[' in body[second:]
    # every multiplication of the block on the same trips: one group of 8
    row = 'tfMult' if arch == 'gfx1150' else 'threadIdx.y'
    assert f'{row} % 8' in src


def test_a_smaller_group_is_asked_for_by_stage_group():
    src = _kernel_at('local_flux', 'gfx942', merge_variants=True,
                     stage_members=True, stage_group=4)
    assert 'threadIdx.y % 4' in src
    assert 'threadIdx.y % 8' not in src


def test_without_it_members_are_read_from_global():
    src = _kernel_at('local_flux', 'gfx942', merge_variants=True,
                     stage_members=False)
    assert not SHARD.search(src)
    assert '__syncthreads();' not in _variant_loop(src)


def _case(name):
    """The module of case `name` under `tests/cases`."""
    path = next((Path(__file__).resolve().parent / 'cases').rglob(
        f'{name}.py'))
    spec = importlib.util.spec_from_file_location('tf_stage__' + name, path)
    case = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(case)
    return case


def test_the_group_stays_within_the_named_barriers():
    """On CUDA a multiplication wider than a warp meets on a named barrier
    of its own, `threadIdx.y + 1`, and a CTA has fifteen of them besides the
    block's: a group of sixteen would have its last multiplication ask for
    barrier 16, which does not exist.  So the group is held to fifteen."""
    case = _case('local_flux')
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        gen = Generator(case.descr_list(), Context(
            arch='sm_86', backend='cuda', fp_type=case.DTYPE,
            options=Options(merge_variants=True, stage_members=True,
                            launch_control=False, stage_group=16,
                            lanes_per_mult=64)))
        gen.generate()
    src = gen.get_kernel()
    assert SHARD.search(src), 'no member streamed; nothing to test'
    launch = json.loads(re.search(r'tensorforge-meta: (\{.*\})',
                                  src).group(1))['launch']
    assert launch['threads_per_mult'] == 64 and 'barrier.sync' in src
    assert launch['block'][1] == 15, launch


def _report(name, arch, **options):
    """What the shards held of the merged run of case `name`."""
    case = _case(name)
    out = io.StringIO()
    with warnings.catch_warnings(), contextlib.redirect_stdout(out):
        warnings.simplefilter('ignore')
        gen = Generator(case.descr_list(), Context(
            arch=arch, backend='hip', fp_type=case.DTYPE,
            options=Options(merge_variants=True, preload_shard_budget=65536,
                            ir_debug='report', **options)))
        gen.generate()
    return [line[len('shards: '):] for line in out.getvalue().splitlines()
            if line.startswith('shards: ')]


def test_the_element_s_own_members_stay_where_they_are():
    """`local_flux` merges its faces over the operator `A_f` and the flux
    solver `C_f`, which is the element's own: only the operator is a member
    every element reads alike, and only it is copied."""
    held = [line for line in _report('local_flux', 'gfx942')
            if line.startswith('+ ')]
    assert len(held) == 1 and ' a trip:' in held[0], held


def _four_products(**options):
    """`C += A_d B` four times, `A_d` an operator, merged into one run, on
    sm_86 with blocks that loop over the batch -- which the host oracle runs
    whole, every multiplication of the block and its barriers."""
    def t(shape, alias, addressing=Addressing.STRIDED):
        return SubTensor(Tensor(list(shape), addressing,
                                BoundingBox([0, 0], list(shape)),
                                alias=alias, datatype=Datatype.F32))
    b, c = t((32, 8), 'B'), t((32, 8), 'C')
    descrs = [GemmDescr(False, False, t((32, 32), f'A{d}', Addressing.NONE),
                        b, c, alpha=1.0, beta=1.0) for d in range(4)]
    out = io.StringIO()
    with warnings.catch_warnings(), contextlib.redirect_stdout(out):
        warnings.simplefilter('ignore')
        gen = Generator(descrs, Context(
            arch='sm_86', backend='cuda', fp_type=Datatype.F32,
            options=Options(merge_variants=True, launch_control=False,
                            preload_globals=False, preload_shards=False,
                            ir_debug='report', **options)))
        gen.generate()
    report = [line for line in out.getvalue().splitlines()
              if line.startswith('shards: + ')]
    return gen.get_kernel(), ke.launch_geometry(gen.get_launcher()), report


def _written(src, geometry, elements=24):
    """What the kernel leaves in the operands it writes over `elements`
    elements -- several trips of the batch loop for either block."""
    lanes, mults = geometry
    signature = re.search(r'\bkernel_\w+\(([^)]*)\)', src).group(1)
    outputs = {p.split()[-1] for p in signature.split(',')
               if '*' in p and not p.strip().startswith('const')}
    mem = ke.evaluate_wave(src, lanes, seed=17, globals_only=True,
                           mults=mults, block=True, elements=elements)
    return {k: v for k, v in mem.items() if k[0] in outputs}


@pytest.mark.parametrize('budget', [1024, 65536], ids=['part', 'whole'])
def test_a_streamed_member_computes_what_the_global_one_does(budget):
    """Held in part or whole, the member read from the copy gives the very
    numbers the member read from global memory does."""
    plain, plain_geo, _ = _four_products(stage_members=False)
    streamed, geo, report = _four_products(stage_members=True,
                                           preload_shard_budget=budget)
    assert report and ' a trip:' in report[0]
    assert streamed.count('__syncthreads();') == 2
    want, got = _written(plain, plain_geo), _written(streamed, geo)
    assert want and set(want) == set(got)
    for key in want:
        assert np.array_equal(np.asarray(got[key]), np.asarray(want[key]))


def _two_operators_a_face(size):
    """Four faces of `R += D_f (A_f B)` in double precision, with `A_f` and
    `D_f` operators of `size` x `size`: a merged run with two holes,
    `size * size * 8` bytes each, on gfx942 -- what the shards held of it,
    and the launch."""
    def t(shape, alias, addressing=Addressing.STRIDED):
        return SubTensor(Tensor(list(shape), addressing,
                                BoundingBox([0, 0], list(shape)),
                                alias=alias, datatype=Datatype.F64))
    b, r = t((size, 9), 'B'), t((size, 9), 'R')
    descrs = []
    for face in range(4):
        a = t((size, size), f'A{face}', Addressing.NONE)
        d = t((size, size), f'D{face}', Addressing.NONE)
        x = SubTensor(generate_tmp_matrix(a, b))
        descrs.append(GemmDescr(False, False, a, b, x))
        descrs.append(GemmDescr(False, False, d, x, r, alpha=1.0, beta=1.0))
    out = io.StringIO()
    with warnings.catch_warnings(), contextlib.redirect_stdout(out):
        warnings.simplefilter('ignore')
        gen = Generator(descrs, Context(
            arch='gfx942', backend='hip', fp_type=Datatype.F64,
            options=Options(merge_variants=True, preload_shard_budget=65536,
                            ir_debug='report')))
        gen.generate()
    src = gen.get_kernel()
    launch = json.loads(re.search(r'tensorforge-meta: (\{.*\})',
                                  src).group(1))['launch']
    report = [line for line in out.getvalue().splitlines()
              if line.startswith('shards: + ')]
    return report, launch


def test_members_too_large_for_the_block_are_streamed_in_part():
    """Two 40.5 kB operators a trip are 81 kB, and a gfx942 block has 64:
    the shards the trip reads soonest are copied in, the rest read from
    global memory, and the launch asks for no more than the block has."""
    report, launch = _two_operators_a_face(72)
    held = [re.search(r'(\d+) of (\d+) element', line).groups()
            for line in report if ' a trip:' in line]
    assert held
    assert sum(int(h) for h, _ in held) < sum(int(t) for _, t in held)
    assert launch['shared_bytes'] <= 65536, launch
