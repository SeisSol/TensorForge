# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The multi-element layout's operator steps on the asynchronous matrix
engines (`element_group.ASYNC_LOWERINGS`): wgmma on sm_90a, a warpgroup's,
A from registers by ldmatrix and D in them; tcgen05 on sm_100a, one
thread's, A and D in tensor memory.  F32 as 3xTF32, B staged once a block
in the canonical K-major order a descriptor reads (`Stage.order`), beside
the matrix track's mma_sync tiling, which stays as it was.

What is checked here: which engine a step takes and how large its group
is (`engine_elements`), the stage's order, the instructions the kernel
issues, the host oracle (`reference.kernel_eval`) on the sections of
`test_multi_element` and on two trips of a group over masked elements,
the build for sm_90a and sm_100a with the SASS read back, and on a GH200 or
a GB200 the sections against NumPy.
"""

from __future__ import annotations

import dataclasses
import re
import shutil
import subprocess
import types
import warnings
from pathlib import Path

import pytest

import test_multi_element as me
from tensorforge import toolchain
from tensorforge.common.basic_types import Datatype
from tensorforge.generators import element_group as eg

INCLUDE = Path(__file__).resolve().parents[1] / 'src' / 'tensorforge' / 'include'

#: The asynchronous engines, asked for: the option's default is mma_sync
#: until a measurement shows them ahead (`Options.matrix_engine`).
AUTO = {'matrix_engine': 'auto'}


def _operator_step(group):
    (step,) = [s for s in group.section.steps
               if s.kind is eg.Kind.OPERATOR]
    return step


def _group(shape='volume', order=6, precision='s', arch='sm_90a', **options):
    context = me._context(arch, precision, multi_element='force',
                          **{**AUTO, **options})
    section, why = eg.classify(me._descrs(shape, order, precision))
    assert section is not None, why
    group, why = eg.fill(section, context)
    assert group is not None, why
    return group


# -- the engine and the group ------------------------------------------------- #

@pytest.mark.parametrize('arch,order,precision,options,lowering,elements', [
    ('sm_90a', 6, 's', {}, 'wgmma', 16),
    ('sm_90a', 8, 's', {}, 'wgmma', 12),
    ('sm_100a', 6, 's', {}, 'tcgen05', 14),
    ('sm_100a', 8, 's', {}, 'tcgen05', 7),
    ('sm_90a', 6, 'd', {}, 'dmma', 16),
    ('sm_100a', 6, 'd', {}, 'dmma', 16),
    ('sm_90a', 6, 's', {'matrix_engine': 'mma_sync'}, 'mma_sync', 16),
    ('sm_100a', 6, 's', {'matrix_engine': 'mma_sync'}, 'mma_sync', 16),
    ('sm_90', 6, 's', {}, 'mma_sync', 16),
    ('sm_100', 6, 's', {}, 'mma_sync', 16),
    ('sm_120', 6, 's', {}, 'mma_sync', 8)])
def test_the_engine_is_the_targets_asynchronous_one(arch, order, precision,
                                                   options, lowering,
                                                   elements):
    """wgmma on sm_90a and tcgen05 on sm_100a where F32 is tiled, mma_sync
    where `matrix_engine` holds it and on the plain targets; F64 DMMA's,
    its group 16 at o6 (B's tiles of zeros left out of the room).
    wgmma's group is whole warpgroups (16, and o8's 12 of 14); tcgen05's
    rows, 9 an element, its 128 lanes (14 of o6's 16).  At o8 B goes
    through the ring (`test_o8_streams_b_through_the_ring`)."""
    group = _group('volume', order, precision, arch, **options)
    assert _operator_step(group).lowering == lowering
    assert group.elements == elements


@pytest.mark.parametrize('arch', ['sm_90a', 'sm_100a'])
def test_the_engine_is_mma_sync_unless_asked(arch):
    """`matrix_engine`'s default holds F32 to mma_sync on the targets with
    an asynchronous engine: the GH200's and GB200's measurements put them
    behind it so far (o6 10.52 against 10.75 ns on wgmma, 11.06 against
    9.68 on tcgen05).  `auto` asks for them."""
    context = me._context(arch, 's', multi_element='force')
    section, _ = eg.classify(me._descrs('volume', 6, 's'))
    group, _ = eg.fill(section, context)
    assert _operator_step(group).lowering == 'mma_sync'


def test_engine_elements_rounds_to_what_the_engines_take():
    group = _group(arch='sm_90a')
    assert eg.engine_elements(group.section, 16) == 16
    assert eg.engine_elements(group.section, 15) == 12
    assert eg.engine_elements(group.section, 3) is None
    group = _group(arch='sm_100a')
    assert eg.engine_elements(group.section, 16) == 14
    assert eg.engine_elements(group.section, 8) == 8
    assert eg.engine_elements(group.section, 3) is None
    synchronous = _group(arch='sm_90')
    assert eg.engine_elements(synchronous.section, 7) == 7


def test_a_step_wider_than_the_instruction_stays_on_mma_sync(monkeypatch):
    monkeypatch.setattr(eg, 'ASYNC_N', 32)
    assert _operator_step(_group(arch='sm_90a')).lowering == 'mma_sync'
    assert _operator_step(_group(arch='sm_100a')).lowering == 'mma_sync'


def test_tcgen05_takes_no_step_whose_columns_four_elements_overfill(
        monkeypatch):
    """The lowering's own refusal, before the fill rule's group sizes
    (`engine_elements`) would fall back to mma_sync's tiling anyway."""
    monkeypatch.setattr(eg, 'TCGEN05_ROWS', 32)
    context = me._context('sm_100a', 's', multi_element='force', **AUTO)
    section, _ = eg.classify(me._descrs('volume', 6, 's'))
    lowered = eg.lower(section, context, split=True)
    assert _operator_step(types.SimpleNamespace(
        section=lowered)).lowering == 'mma_sync'
    assert _operator_step(_group(arch='sm_100a')).lowering == 'mma_sync'


@pytest.mark.parametrize('arch,form', [('sm_90a', 'wgmma'),
                                       ('sm_100a', 'tcgen05')])
def test_b_is_staged_split_in_the_canonical_order(arch, form):
    """The operator's window in TF32 halves, laid out as core matrices of
    eight rows of N by four of K, 128 bytes along K and the depth's 32
    along N."""
    group = _group(arch=arch)
    step = _operator_step(group)
    stages = [group.section.stage(p.first.tensor, (p.rows, p.depth))
              for p in step.products]
    for stage in stages:
        assert stage.order == 'canonical' and stage.split
        assert stage.depth % 8 == 0 and stage.width % (
            8 if form == 'wgmma' else 16) == 0
        assert stage.leading == 128 and stage.stride == stage.depth * 32
        assert stage.canonical_offset(0, 0) == 0
        assert stage.canonical_offset(1, 0) == 4
        assert stage.canonical_offset(0, 4) == 32
        assert stage.canonical_offset(8, 0) == stage.depth * 8
        assert stage.volume == 2 * stage.depth * stage.width
        offsets = {stage.canonical_offset(n, k)
                   for n in range(stage.width) for k in range(stage.depth)}
        assert offsets == set(range(stage.depth * stage.width))


@pytest.mark.parametrize('arch', ['sm_90a', 'sm_100a'])
def test_a_is_read_from_pitched_buffers(arch):
    """The group operands an asynchronous step reads are pitched as
    mma_sync's: ldmatrix's 16-byte rows on four floats more than a multiple
    of eight, and the contraction's padding to the instruction's eight
    there, zero."""
    group = _group(arch=arch)
    step = _operator_step(group)
    for p in step.products:
        buffer = group.section.buffer(p.second.tensor)
        assert buffer.stride % 8 == 4
        assert buffer.stride >= (p.depth[0] - buffer.rows[0]) + (
            -(-(p.depth[1] - p.depth[0]) // 8) * 8)


def test_a_group_the_engine_does_not_take_falls_back_to_mma_sync(
        monkeypatch):
    """Where no group fits the asynchronous engine's tiling, mma_sync's
    takes the section, before the FMA forms."""
    real = eg.engine_elements
    monkeypatch.setattr(eg, 'engine_elements', lambda section, elements: (
        None if eg.asynchronous(section) else real(section, elements)))
    assert _operator_step(_group(arch='sm_90a')).lowering == 'mma_sync'
    assert _operator_step(_group(arch='sm_100a')).lowering == 'mma_sync'


@pytest.mark.parametrize('arch', ['sm_90a', 'sm_100a'])
def test_a_plan_held_to_mma_sync_is_built_on_it(arch, monkeypatch):
    """The generator lowers the section again for its body as the plan
    has it (`Generator._emit_element_group`): a plan the fill rule held to
    mma_sync is built on mma_sync, and not on the target's engine."""
    real = eg.engine_elements
    monkeypatch.setattr(eg, 'engine_elements', lambda section, elements: (
        None if eg.asynchronous(section) else real(section, elements)))
    src = _kernel(arch).get_kernel()
    assert 'mma.sync' in src
    assert 'wgmma' not in src and 'tcgen05' not in src


def test_mma_sync_keeps_its_fragments():
    group = _group(arch='sm_90')
    assert all(st.order == 'fragments' for st in group.section.stages
               if st.window is not None)


# -- the kernel ---------------------------------------------------------------- #

def _kernel(arch, shape='volume', order=6, precision='s', **options):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        gen = me._generate(me._descrs(shape, order, precision),
                           me._context(arch, precision,
                                       multi_element='force',
                                       **{**AUTO, **options}))
    return gen


def test_held_to_mma_sync_the_kernel_is_the_plain_targets():
    """sm_90a with the engine held to mma_sync writes sm_90's kernel, its
    name and the option's note apart."""
    def body(gen):
        text = gen.get_kernel()
        text = re.sub(r'kernel_kernel_[0-9a-f]+', 'kernel', text)
        return re.sub(r'// options: .*', '', text)
    assert body(_kernel('sm_90a', matrix_engine='mma_sync')) == \
        body(_kernel('sm_90'))


def test_the_wgmma_kernel():
    """o6's operator step: three products of five steps of K each, three
    products of 3xTF32 a step -- 45 wgmma -- each step A by one ldmatrix,
    a fence, a commit and a wait; no mma.sync."""
    src = _kernel('sm_90a').get_kernel()
    assert src.count('wgmma.mma_async') == 45
    assert src.count('tensorforge::ldmatrixX4') == 15
    assert src.count('tensorforge::wgmmaFence()') == 15
    assert src.count('tensorforge::wgmmaCommit()') == 15
    assert src.count('tensorforge::wgmmaWait<0>()') == 15
    assert src.count('tensorforge::fenceProxyAsync()') == 1
    assert 'mma.sync' not in src


@pytest.mark.parametrize('pending', [1, 2])
def test_the_pipelined_wgmma_kernel(pending):
    """`wgmma_pending`: the same 45 products, a fence and a commit a step
    of K, but each step's wait leaves the `pending` latest groups in
    flight -- 15 steps, so 15 - pending waits of their own and the last
    for all of them, ahead of D."""
    src = _kernel('sm_90a', wgmma_pending=pending).get_kernel()
    assert src.count('wgmma.mma_async') == 45
    assert src.count('tensorforge::ldmatrixX4') == 15
    assert src.count('tensorforge::wgmmaFence()') == 15
    assert src.count('tensorforge::wgmmaCommit()') == 15
    assert src.count(f'tensorforge::wgmmaWait<{pending}>()') == 15 - pending
    assert src.count('tensorforge::wgmmaWait<0>()') == 1
    assert 'mma.sync' not in src


def test_nothing_pending_is_the_kernel_as_it_was():
    """`wgmma_pending` 0, the default, writes the kernel of a wait a step,
    to the byte."""
    def body(gen):
        text = re.sub(r'kernel_kernel_[0-9a-f]+', 'kernel', gen.get_kernel())
        return re.sub(r'// options: .*', '', text)
    assert body(_kernel('sm_90a', wgmma_pending=0)) == body(_kernel('sm_90a'))


def test_the_ring_waits_for_each_chunk_whatever_is_pending(monkeypatch):
    """The block's next copy writes the slot the chunk before read, which
    every warpgroup has to have retired: a chunk waits for itself."""
    _streaming(monkeypatch, 'sm_90a')
    src = _kernel('sm_90a', wgmma_pending=1).get_kernel()
    assert src.count('tensorforge::wgmmaWait<0>()') == 15
    assert 'tensorforge::wgmmaWait<1>()' not in src


def test_wgmma_pending_takes_what_a_thread_has_registers_for():
    from tensorforge.common.options import parse_wgmma_pending
    assert [parse_wgmma_pending(t) for t in ('0', '1', '2')] == [0, 1, 2]
    for text in ('3', '-1'):
        with pytest.raises(ValueError):
            parse_wgmma_pending(text)


def test_the_tcgen05_kernel():
    """45 products in one round -- every product's A in columns of its
    own, 512 of them, so one commit and one wait for the step -- the
    tensor memory allocated and freed, A stored in it, 30 stores, both
    halves of each step of K, and D loaded: 55 rows to the next eight,
    32 + 16 + 8; B fenced for the async proxy once."""
    src = _kernel('sm_100a').get_kernel()
    assert src.count('tcgen05.mma.cta_group::1.kind::tf32') == 45
    assert src.count('tensorforge::tcgen05Commit(') == 1
    assert src.count('tensorforge::tmemAlloc<512>') == 1
    assert src.count('tensorforge::tmemDealloc<') == 1
    assert len(re.findall(r'tcgen05\.st\.sync\.aligned\.32x32b\.x8', src)) == 30
    loads = re.findall(r'tcgen05\.ld\.sync\.aligned\.32x32b\.x(\d+)', src)
    assert sorted(map(int, loads)) == [8, 16, 32]
    assert src.count('tensorforge::fenceProxyAsync()') == 1
    assert 'mma.sync' not in src


def test_tcgen05s_a_is_stored_by_every_band_of_warps():
    """o6's group of 14 holds three bands of four warps whole: a chunk of
    eight columns of A each in turn, warp w the lanes 32 (w % 4) on --
    and the band the chunk is not the band's skips it."""
    src = _kernel('sm_100a').get_kernel()
    bands = re.findall(r'\w+_mine = \w+_band == (\d+);', src)
    assert sorted(set(bands)) == ['0', '1', '2']


def test_tcgen05s_d_goes_out_through_every_element_warp():
    """D staged in shared memory, a row a lane-holding thread, an odd
    number of floats apart; every element warp's threads write a cell
    each a hop."""
    from tensorforge.backend.instructions.compute.multi_element import \
        d_stage_stride
    assert d_stage_stride(55) == 55 and d_stage_stride(56) == 57
    src = _kernel('sm_100a').get_kernel()
    assert 'dstage' in src
    assert re.search(r'for \(int32_t \w+_c = \w+_tid; \w+_c < 6930; '
                     r'\w+_c \+= 448\)', src)


@pytest.mark.parametrize('arch', ['sm_90a', 'sm_100a'])
@pytest.mark.parametrize('shape', ['volume', 'chain', 'reuse', 'assign',
                                   'window'])
def test_the_block_holds_what_the_plan_counted(shape, arch):
    """The kernel's block takes the shared memory the fill rule counted
    (`element_group.shared_bytes`), within the room it leaves the arena
    (`SHARED_RESERVE`): tcgen05's D is staged once a trip for all of its
    steps (`_d_stage`), the widest's rows.  A stage a step, the chain's two
    steps took 27 kB past the plan -- with the producer's ring, more than a
    GB200's block has."""
    group = _group(shape, 6, 's', arch)
    src = _kernel(arch, shape).get_kernel()
    shared = int(re.search(r'(\d+) B shared', src).group(1))
    assert group.shared_bytes <= shared <= (group.shared_bytes
                                            + eg.SHARED_RESERVE)
    assert src.count('_dstage = ') == (arch == 'sm_100a')


def test_a_ring_keeps_a_product_a_round():
    """With B streamed, a product's A shares D's end with the others':
    a product a round, a chunk a commit."""
    from tensorforge.backend.instructions.compute.multi_element import \
        _tcgen05_columns
    group = _group('volume', 8, 's', 'sm_100a')
    step = _operator_step(group)
    columns, at, apart = _tcgen05_columns(group.section, step)
    assert len(set(at)) == 1 and columns <= 512 and not apart


def test_a_ring_of_one_product_commits_a_chunk(monkeypatch):
    """A step of one product has columns of its own either way, but with B
    streamed its pieces are the ring's chunks: each its own round -- a
    commit, the next chunk converted, a wait -- not one round of all of
    them, which would leave the next chunks unconverted.  `assign`'s depth
    of 35 is five chunks of eight."""
    from tensorforge.backend.instructions.compute.multi_element import \
        _tcgen05_columns
    resident, group = _streaming(monkeypatch, 'sm_100a', 'assign')
    step = _operator_step(group)
    assert len(step.products) == 1
    assert _tcgen05_columns(group.section, step)[2] is False
    assert _tcgen05_columns(resident.section,
                            _operator_step(resident))[2] is True
    src = _kernel('sm_100a', 'assign').get_kernel()
    assert src.count('tcgen05.mma.cta') == 15
    assert src.count('tensorforge::tcgen05Commit(') == 5


def test_the_tensor_memory_and_the_mbarrier_are_the_blocks():
    """Allocated once, ahead of the batch loop, and freed behind it -- the
    allocation gives up the block's permit to allocate again -- and its
    slot and the mbarrier the products complete on in the block's arena,
    which every warp reads: a warp's window of the multiplications' arena
    is its own (`MultiElementOperators.tmem`)."""
    src = _kernel('sm_100a').get_kernel()
    lines = [line.strip() for line in src.splitlines()]
    alloc = next(i for i, line in enumerate(lines) if 'tmemAlloc<' in line)
    loop = next(i for i, line in enumerate(lines)
                if re.match(r'for \(size_t \w*batchIdGroup', line))
    free = next(i for i, line in enumerate(lines) if 'tmemDealloc<' in line)
    assert alloc < loop < free
    assert lines[free + 1:free + 3] == ['}', '}']
    slot = re.search(r'&(\w+)\[0\]', lines[alloc]).group(1)
    for name in (slot, 'tcmma_barrier'):
        (decl,) = [line for line in lines
                   if re.search(rf'\b{name} = ', line)]
        assert '&totalShrMem[' in decl, decl


# -- the host oracle ------------------------------------------------------------ #

@pytest.mark.parametrize('arch', ['sm_90a', 'sm_100a'])
@pytest.mark.parametrize('shape', ['volume', 'chain', 'reuse', 'assign',
                                   'window'])
def test_the_section_computes_on_the_oracle(shape, arch):
    """The kernel on the host interpreter against NumPy, normwise, F32 as
    for mma_sync's tiling (`test_multi_element`)."""
    for alias, error in me._oracle_errors(shape, 's', arch, AUTO).items():
        assert error <= 1e-6, (alias, error)


@pytest.mark.parametrize('arch', ['sm_90a', 'sm_100a'])
@pytest.mark.parametrize('shape', ['volume', 'reuse', 'assign'])
def test_every_element_of_two_trips_on_the_oracle(shape, arch, monkeypatch):
    """A group of four over seven elements: two trips, the second with a
    warp masked -- every element against NumPy, and no access to shared
    memory a barrier does not order (`kernel_eval.Races`)."""
    _two_trips(shape, arch, monkeypatch)


@pytest.mark.parametrize('arch,streamed', [
    ('sm_90a', False), ('sm_90a', True), ('sm_100a', False),
    ('sm_100a', True)], ids=['wgmma', 'wgmma-ring', 'tcgen05',
                             'tcgen05-ring'])
def test_every_element_of_two_trips_with_own_barriers_on_the_oracle(
        arch, streamed, monkeypatch):
    """As above, built without the barrier placement
    (`Options.enable_sync_block_opt` off): the lowering's own barriers are
    then all there is -- tcgen05's between D's stage and every element
    warp's share of it, wgmma's behind each chunk of the ring -- where the
    placement would otherwise put back one the lowering forgot."""
    _two_trips('volume', arch, monkeypatch, streamed=streamed,
               enable_sync_block_opt=False)


@pytest.mark.parametrize('pending', [1, 2])
@pytest.mark.parametrize('shape', ['volume', 'chain', 'reuse', 'assign',
                                   'window'])
def test_the_pipelined_steps_compute_on_the_oracle(shape, pending):
    """With groups left in flight, the oracle checks every product's A
    registers and B slots unchanged where the group retires, and D read
    only behind the last wait."""
    for alias, error in me._oracle_errors(
            shape, 's', 'sm_90a', {**AUTO, 'wgmma_pending': pending}).items():
        assert error <= 1e-6, (alias, error)


@pytest.mark.parametrize('shape', ['volume', 'reuse'])
def test_the_pipelined_steps_over_two_trips_on_the_oracle(shape,
                                                          monkeypatch):
    _two_trips(shape, 'sm_90a', monkeypatch, wgmma_pending=1)


def test_d_past_the_second_warps_lanes_on_the_oracle(monkeypatch):
    """A group of eight, 72 rows of D: tcgen05's tensor memory holds row
    `m` in lane `m` of the four warps a warp's 32 lanes, so the rows past
    64 are the third warp's to load -- every element against NumPy, the
    eighth's rows 63..71 among them."""
    _two_trips('volume', 'sm_100a', monkeypatch, group=8, count=8)


def _two_trips(shape, arch, monkeypatch, streamed=False, group=4, count=7,
               **options):
    import numpy as np
    from tensorforge.common.basic_types import Addressing
    from tensorforge.reference import kernel_eval
    monkeypatch.setattr(eg, 'ELEMENTS', {key: group for key in eg.ELEMENTS})
    monkeypatch.setattr(eg, 'DEFAULT_ELEMENTS', group)
    if streamed:
        _streaming(monkeypatch, arch, shape, 4)
    gen = _kernel(arch, shape, 4, 's', **options)
    built = gen.built._element_group
    assert built.elements == group
    assert all(s.lowering in eg.ASYNC_LOWERINGS for s in built.section.steps
               if s.kind is eg.Kind.OPERATOR)
    assert eg.streamed(built.section) is streamed
    tensors = {}
    for outer in gen.descr_list:
        for d in outer.operations():
            for m in d.matrix_list():
                t = getattr(m, 'tensor', m)
                tensors.setdefault(t.alias, t)

    def box(t):
        return [(int(lo), int(hi)) for lo, hi in zip(t.bbox.lower(),
                                                      t.bbox.upper())]

    def memory(t, read):
        (r0, r1), (c0, c1) = box(t)
        out = np.zeros((count, r1 - r0, c1 - c0))
        for e in range(count):
            for c in range(c1 - c0):
                for r in range(r1 - r0):
                    out[e, r, c] = read(e, r + c * (r1 - r0))
        return out

    fill = kernel_eval.Slot(me.SEED)
    preset, inputs = {}, {}
    for alias, t in tensors.items():
        if t.is_tmp:
            continue
        if t.addressing.kind is Addressing.NONE:
            preset[t.name] = [float(v) for v in t.storage_values()]
            (r0, r1), (c0, c1) = box(t)
            inputs[alias] = np.asarray(t.get_values())[None, r0:r1, c0:c1]
        else:
            inputs[alias] = memory(t, lambda e, s, t=t: fill.read(
                f'{t.name}@{e}', s))
    output = next(t for t in tensors.values() if not t.is_tmp
                  and t.addressing.kind is not Addressing.NONE
                  and any(d.writes().tensor is t for outer in gen.descr_list
                          for d in outer.operations()))
    lanes, mults = kernel_eval.launch_geometry(gen.get_launcher())
    races = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mem = kernel_eval.evaluate_wave(
            gen.get_kernel(), lanes, seed=me.SEED, globals_only=True,
            mults=mults, block=True, preset=preset, elements=count,
            races=races)
    assert not races, races[:3]
    got = memory(output, lambda e, s: mem.get((f'{output.name}@{e}', s),
                                              fill.read(f'{output.name}@{e}',
                                                        s)))
    want = me._reference(me._descrs(shape, 4, 's'))(
        inputs, inputs[output.alias])
    for e in range(count):
        scale = np.abs(want[e]).max() or 1.0
        error = np.abs(got[e] - want[e]).max() / scale
        assert error <= 1e-6, (e, error)


# -- the build ------------------------------------------------------------------ #

#: The first CUDA release that builds each arch-specific target.
ARCH_RELEASE = {'sm_90a': (12, 0), 'sm_100a': (12, 8)}


def _nvcc_builds(arch):
    found = toolchain.COMPILERS['cuda'].find()
    if not found or not shutil.which(found):
        pytest.skip('nvcc not found')
    listed = subprocess.run([found, '--list-gpu-code'], capture_output=True,
                            text=True, timeout=60).stdout.split()
    version = re.search(r'release (\d+)\.(\d+)', subprocess.run(
        [found, '--version'], capture_output=True, text=True,
        timeout=60).stdout)
    release = (int(version.group(1)), int(version.group(2))) if version \
        else (0, 0)
    if (arch.rstrip('af') not in listed
            or release < ARCH_RELEASE.get(arch, (0, 0))):
        pytest.skip(f'nvcc {release[0]}.{release[1]} builds no {arch}')
    return found


def _sass(arch, tmp_path, shape='volume', order=6, **options):
    nvcc = _nvcc_builds(arch)
    gen = _kernel(arch, shape, order, 's', **options)
    context = me._context(arch, 's', multi_element='force',
                          **{**AUTO, **options})
    headers = list(context.target.headers()) + list(gen.get_helper_headers())
    src = tmp_path / 'kernel.cu'
    src.write_text(''.join(f'#include "{h}"\n' for h in headers)
                   + '#include "tensorforge_aux.h"\n' + gen.get_kernel()
                   + '\n' + gen.get_launcher())
    entry = toolchain.COMPILERS['cuda']
    obj = tmp_path / 'kernel.o'
    run = subprocess.run([nvcc, *entry.language_flags(),
                          *entry.target_flags(arch), '-I', str(INCLUDE),
                          '-c', str(src), '-o', str(obj)],
                         capture_output=True, text=True, timeout=1200)
    assert run.returncode == 0, run.stderr[-4000:]
    tool = shutil.which('cuobjdump')
    if tool is None:
        pytest.skip('cuobjdump not found')
    return subprocess.run([tool, '-sass', str(obj)], capture_output=True,
                          text=True, timeout=600).stdout


def _count(sass, op):
    return len(re.findall(rf'\b{re.escape(op)}\b', sass))


def test_compiled_for_sm90a_each_product_is_one_hgmma(tmp_path):
    sass = _sass('sm_90a', tmp_path)
    assert _count(sass, 'HGMMA') == 45 and _count(sass, 'LDSM') == 15
    assert _count(sass, 'HMMA') == 0


def test_compiled_for_sm100a_each_product_is_one_utchmma(tmp_path):
    sass = _sass('sm_100a', tmp_path)
    assert _count(sass, 'UTCHMMA') == 45 and _count(sass, 'UTCBAR') == 1
    assert _count(sass, 'STTM') == 30 and _count(sass, 'LDTM') == 3


# -- on the hardware ----------------------------------------------------------- #

GPU_CASES = [('volume', 6), ('volume', 4), ('chain', 6), ('reuse', 6),
             ('assign', 6)]


@pytest.mark.parametrize('shape,order', GPU_CASES,
                         ids=[f'{s}-o{o}' for s, o in GPU_CASES])
def test_the_section_computes_on_the_gpu(shape, order, target, cache_root,
                                         tensorforge_include):
    """Built for the arch-specific target of the GPU present -- sm_90a on a
    GH200, wgmma; sm_100a on a GB200, tcgen05 -- and run against NumPy over
    a batch that ends in a partial group and gives a block several trips,
    as `test_multi_element`'s cases run."""
    if target.backend != 'cuda' or target.arch not in ('sm_90', 'sm_100'):
        pytest.skip('no GPU with an asynchronous matrix engine')
    from harness.runner import run_case
    special = dataclasses.replace(target, arch=target.arch + 'a')
    _nvcc_builds(special.arch)
    group = eg.plan(me._descrs(shape, order, 's'), me._context(
        special.arch, 's', multi_element='force', **AUTO))
    assert group is not None
    # The chain and the reuse have an operator step a product each.
    assert all(s.lowering in eg.ASYNC_LOWERINGS for s in group.section.steps
               if s.kind is eg.Kind.OPERATOR)
    case = types.SimpleNamespace(
        NAME=f'multi_element_async_{shape}_o{order}s',
        DTYPE=Datatype.F32, BATCH=4099, TOL=(1e-4, 1e-4),
        OPTIONS={'multi_element': 'force', **AUTO},
        INPUT_TRANSFORM=me._known_zeros(me._descrs(shape, order, 's')),
        descr_list=lambda: me._descrs(shape, order, 's'),
        reference=me._reference(me._descrs(shape, order, 's')))
    result = run_case(case, special, cache_root, tensorforge_include)
    assert result.passed, (result.max_abs_err, result.max_rel_err,
                           result.detail)


@pytest.mark.parametrize('shape', ['volume', 'chain'])
def test_the_pipelined_steps_compute_on_the_gpu(shape, target, cache_root,
                                                tensorforge_include):
    """`wgmma_pending` 1 on a GH200: the o6 volume and the chain against
    NumPy, as `test_the_section_computes_on_the_gpu` runs them, each
    element within 1e-5 of its largest cell (`test_multi_element.
    _normwise`: 3xTF32 at o6 is ~2e-6 of it) -- and over the same elements
    bit for bit what the steps waited for one at a time give
    (`wgmma_pending` 0): the same products in the same order, only the
    waits moved."""
    if target.backend != 'cuda' or target.arch != 'sm_90':
        pytest.skip('no GPU with wgmma')
    import numpy as np
    from harness import driver_emit
    from harness.runner import execute, run_case
    special = dataclasses.replace(target, arch='sm_90a')
    _nvcc_builds(special.arch)
    case = types.SimpleNamespace(
        NAME=f'multi_element_pipelined_{shape}_o6s',
        DTYPE=Datatype.F32, BATCH=4099, TOL=me._normwise(),
        OPTIONS={'multi_element': 'force', 'wgmma_pending': 1, **AUTO},
        INPUT_TRANSFORM=me._known_zeros(me._descrs(shape, 6, 's')),
        descr_list=lambda: me._descrs(shape, 6, 's'),
        reference=me._reference(me._descrs(shape, 6, 's')))
    result = run_case(case, special, cache_root, tensorforge_include)
    assert result.passed, (result.max_abs_err, result.max_rel_err,
                           result.detail)
    builds = [me._generate(me._descrs(shape, 6, 's'), me._context(
        special.arch, 's', multi_element='force', wgmma_pending=pending,
        **AUTO)) for pending in (1, 0)]
    batch = 4099
    rng = np.random.default_rng(13)
    inputs = {}
    for op in driver_emit.collect_operands(builds[0]):
        if op.is_scalar or not op.is_source:
            continue
        key = op.alias or op.kernel_name
        if op.values is not None:
            inputs[key] = np.asarray(op.values, dtype=np.float32).reshape(
                op.shape, order='F')[None]
            continue
        rows = 1 if op.addressing == 'none' else batch
        inputs[key] = rng.standard_normal((rows, *op.shape)).astype(
            np.float32)
    got, want = (execute(gen, special, cache_root, tensorforge_include,
                         batch, inputs) for gen in builds)
    assert set(got) == set(want)
    for name in want:
        assert np.array_equal(got[name], want[name]), name


# -- B streamed through a ring ------------------------------------------------- #

def _streaming(monkeypatch, arch, shape='volume', order=6):
    """The section's group with its B streamed (`Stage.streamed`): chunks
    of eight rows of K, so that the ring is smaller than the smallest B and
    a product takes several, and the block's room cut to one element less
    than the resident stage needs, so that the fill rule takes the ring at
    the same group size."""
    monkeypatch.setattr(eg, 'RING_CHUNK', 8)
    resident = _group(shape, order, 's', arch)
    hw = me._context(arch, 's').target.hw
    monkeypatch.setattr(eg, 'SHARED_RESERVE', hw.max_local_mem_size_per_block
                        - (resident.shared_bytes - 1))
    group = _group(shape, order, 's', arch)
    assert eg.streamed(group.section)
    return resident, group


@pytest.mark.parametrize('arch', ['sm_90a', 'sm_100a'])
def test_b_too_large_for_the_block_is_streamed_at_the_same_group(arch,
                                                                monkeypatch):
    """Where B's halves leave no room for the group, the ring takes them --
    `RING_SLOTS` chunks of `RING_CHUNK` rows of K, both halves -- ahead of
    B whole: the group keeps its size, the stages hold nothing between the
    trips, and the ring is counted in the block's room."""
    resident, group = _streaming(monkeypatch, arch)
    assert group.elements == resident.elements
    stages = [st for st in group.section.stages if st.window is not None]
    assert all(st.streamed and st.split and st.volume == 0 for st in stages)
    width = max(st.width for st in stages)
    assert eg.ring_volume(group.section) == \
        eg.RING_SLOTS * eg.RING_CHUNK * width * 2
    saved = sum(st.depth * st.width * 2 for st in stages) - \
        eg.ring_volume(group.section)
    assert resident.shared_bytes - group.shared_bytes == 4 * saved


@pytest.mark.parametrize('arch', ['sm_90a', 'sm_100a'])
def test_o8_streams_b_through_the_ring(arch):
    """o8's B, 3 x 88 x 120 in halves, fits no block: the ring takes it,
    and the group is the engine's (12 for wgmma's warpgroups, 7 tcgen05's
    halving reaches) rather than B whole's four."""
    group = _group('volume', 8, 's', arch)
    assert eg.streamed(group.section)
    assert group.elements == (12 if arch == 'sm_90a' else 7)


@pytest.mark.parametrize('arch,instruction', [('sm_90a', 'wgmma.mma_async'),
                                              ('sm_100a', 'tcgen05.mma.cta')])
def test_the_ring_kernel(arch, instruction, monkeypatch):
    """o6's depth of 40 in chunks of eight: fifteen chunks of the three
    products, each fenced for the async proxy behind its copy -- and the
    first ahead of them -- each one commit and one wait; the 45 products
    as without the ring."""
    _streaming(monkeypatch, arch)
    src = _kernel(arch).get_kernel()
    assert src.count(instruction) == 45
    assert src.count('tensorforge::fenceProxyAsync()') == 16
    if arch == 'sm_90a':
        assert src.count('tensorforge::wgmmaWait<0>()') == 15
    else:
        assert src.count('tensorforge::tcgen05Commit(') == 15


@pytest.mark.parametrize('arch,instruction', [('sm_90a', 'wgmma.mma_async'),
                                              ('sm_100a', 'tcgen05.mma.cta')])
def test_the_ring_reads_a_chunk_ahead(arch, instruction, monkeypatch):
    """A chunk's operator is read into registers a chunk before it is put
    into its slot: the loads of the first two stand ahead of the first
    product, and each chunk's are read once -- fifteen chunks of the three
    products.  Read where it is put, every chunk was a latency the block sat
    out (p5d's ncu: 46 % of tcgen05's samples at o8)."""
    _streaming(monkeypatch, arch)
    src = _kernel(arch).get_kernel()
    loads = r'float \w+_k = glb_m\d+\['
    total = len(re.findall(loads, src))
    ahead = len(re.findall(loads, src[:src.index(instruction)]))
    assert total and total % 15 == 0
    assert ahead == 2 * total // 15


@pytest.mark.parametrize('arch', ['sm_90a', 'sm_100a'])
@pytest.mark.parametrize('shape', ['volume', 'reuse', 'assign', 'window'])
def test_the_ring_computes_on_the_oracle(shape, arch, monkeypatch):
    _streaming(monkeypatch, arch, shape)
    for alias, error in me._oracle_errors(shape, 's', arch, AUTO).items():
        assert error <= 1e-6, (alias, error)


@pytest.mark.parametrize('arch', ['sm_90a', 'sm_100a'])
@pytest.mark.parametrize('shape', ['volume'])
def test_the_ring_over_two_trips_on_the_oracle(shape, arch, monkeypatch):
    """Two trips of a group of four over seven elements with the ring: a
    slot rewritten while the engine still reads it is what the oracle's
    check at the retirement refuses, and an access a barrier does not
    order a race."""
    _two_trips(shape, arch, monkeypatch, streamed=True)


def test_compiled_pipelined_each_product_is_one_hgmma_and_the_wait_keeps_one(
        tmp_path):
    """ptxas keeps the pipeline: 45 HGMMA, the steps' waits leave one
    group in flight (`DEPBAR.LE gsb0, 0x1`) and the last none."""
    sass = _sass('sm_90a', tmp_path, wgmma_pending=1)
    assert _count(sass, 'HGMMA') == 45
    waits = re.findall(r'WARPGROUP\.DEPBAR\.LE gsb0, (0x\d)', sass)
    assert sorted(waits) == ['0x0'] + ['0x1'] * 14


@pytest.mark.parametrize('arch,instruction', [('sm_90a', 'HGMMA'),
                                              ('sm_100a', 'UTCHMMA')])
def test_compiled_o8_streams_b_and_issues_each_product_once(arch, instruction,
                                                            tmp_path):
    """o8's volume with its B streamed: 99 products -- three of depth 88,
    eleven steps of K each, three products a step -- one instruction each."""
    sass = _sass(arch, tmp_path, order=8)
    assert _count(sass, instruction) == 99


@pytest.mark.parametrize('shape', ['volume', 'window'])
def test_the_ring_computes_on_the_gpu(shape, target, cache_root,
                                      tensorforge_include, monkeypatch):
    """The o6 sections with B streamed through the ring (`_streaming`:
    chunks of eight, the resident stage refused), built for the
    arch-specific target of a GH200 or a GB200 and run against NumPy as
    `test_the_section_computes_on_the_gpu` runs them."""
    if target.backend != 'cuda' or target.arch not in ('sm_90', 'sm_100'):
        pytest.skip('no GPU with an asynchronous matrix engine')
    from harness.runner import run_case
    special = dataclasses.replace(target, arch=target.arch + 'a')
    _nvcc_builds(special.arch)
    _streaming(monkeypatch, special.arch, shape)
    group = eg.plan(me._descrs(shape, 6, 's'), me._context(
        special.arch, 's', multi_element='force', **AUTO))
    assert group is not None and eg.streamed(group.section)
    case = types.SimpleNamespace(
        NAME=f'multi_element_ring_{shape}_o6s',
        DTYPE=Datatype.F32, BATCH=4099, TOL=(1e-4, 1e-4),
        OPTIONS={'multi_element': 'force', **AUTO},
        INPUT_TRANSFORM=me._known_zeros(me._descrs(shape, 6, 's')),
        descr_list=lambda: me._descrs(shape, 6, 's'),
        reference=me._reference(me._descrs(shape, 6, 's')))
    result = run_case(case, special, cache_root, tensorforge_include)
    assert result.passed, (result.max_abs_err, result.max_rel_err,
                           result.detail)
