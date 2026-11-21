# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The multi-element layout's inputs through a producer warp
(`Options.element_producer`, `GroupSection.ring`): the block's element warps
and a producer's warp behind them, in two roles (`IRBuilder.roles`), which
hand the inputs over through a ring of slots (`IRBuilder.mbarrier_ring`).
The producer runs the group's trips ahead of the element warps: it waits
for a slot's round on its empty side, copies the group's inputs into it --
a bulk copy a column where the copy takes the box
(`element_group.bulk_input`), its lanes' loads otherwise -- and announces
the bytes on the slot's full side; the element warps wait for the slot,
run the section's steps on it, and give it back.

What is checked here: what the plan takes and refuses, the kernel's
statements, the host oracle (`reference.kernel_eval`) on two trips of a
group over masked elements for every engine the steps take -- values,
races and the ring's protocol -- and on a GPU of sm_90 or newer the
sections against NumPy.
"""

from __future__ import annotations

import re
import types
import warnings

import pytest

import test_multi_element as me
from tensorforge.common.basic_types import Datatype
from tensorforge.generators import element_group as eg

#: The asynchronous engines, asked for where the target has one: the
#: option's default holds the tiling to mma_sync (`Options.matrix_engine`).
AUTO = {'matrix_engine': 'auto'}


def _plan(arch='sm_90', shape='volume', order=6, **options):
    context = me._context(arch, 's', multi_element='force',
                          element_producer=True, **{**AUTO, **options})
    section, why = eg.classify(me._descrs(shape, order, 's'))
    assert section is not None, why
    group, why = eg.fill(section, context)
    assert group is not None, why
    return group


def _kernel(arch='sm_90', shape='volume', order=6, **options):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return me._generate(me._descrs(shape, order, 's'),
                            me._context(arch, 's', multi_element='force',
                                        element_producer=True,
                                        **{**AUTO, **options}))


# -- the plan ------------------------------------------------------------------- #

@pytest.mark.parametrize('arch,elements,slots,lowering', [
    ('sm_90', 16, 3, 'mma_sync'), ('sm_90a', 16, 2, 'wgmma'),
    ('sm_100a', 14, 3, 'tcgen05')])
def test_the_inputs_go_through_a_ring(arch, elements, slots, lowering):
    """o6's volume: the group as without the producer, a warp more, and the
    deepest ring the block has room for -- three slots on sm_90, 221.4 kB,
    mma_sync's B staged without its tiles of zeros; 202.5 kB at two on
    sm_90a, wgmma's B whole; three on sm_100a's smaller group."""
    group = _plan(arch)
    assert (group.elements, group.section.ring) == (elements, slots)
    assert group.producers == 1 and group.warps == elements + 1
    assert all(s.lowering == lowering for s in group.section.steps
               if s.kind is eg.Kind.OPERATOR)


def test_a_bulk_copy_takes_the_inputs_it_can():
    """I -- 64 rows a column, 64-byte aligned -- by a bulk copy a column,
    its 35 rows rounded to 36 and the buffer pitched to hold them; the 9 x 9
    star matrices, unaligned, by the producer's lanes."""
    group = _plan()
    found = {b.tensor.alias: (eg.bulk_input(b), b.pitch)
             for b in group.section.inputs}
    assert found['I'] == (True, 36)
    assert all(not bulk for alias, (bulk, _) in found.items()
               if alias != 'I')


@pytest.mark.parametrize('arch,options,why', [
    ('sm_80', {}, 'no bulk copy'),
    ('sm_90', {'element_producer': False}, 'option is off')])
def test_the_ring_is_refused_where_the_producer_cannot_be(arch, options,
                                                          why):
    context = me._context(arch, 's', multi_element='force',
                          **{'element_producer': True, **options})
    section, _ = eg.classify(me._descrs('volume', 6, 's'))
    lowered = eg.lower(section, context, split=True, ring=2)
    assert lowered.ring == 0
    assert why in eg.producer_refusal(lowered, context)
    group, _ = eg.fill(section, context)
    assert group.section.ring == 0 and group.producers == 0


def test_no_bulk_input_no_producer():
    """Every input unaligned: a producer would load what each warp loads
    for itself, and the plan takes none."""
    context = me._context('sm_90', 's', multi_element='force',
                          element_producer=True)
    section, _ = eg.classify(me._descrs('volume', 6, 's'))
    lowered = eg.lower(section, context, split=True, ring=2)
    assert eg.producer_refusal(lowered, context) is None
    star_only = eg.dataclasses.replace(lowered, inputs=tuple(
        b for b in lowered.inputs if b.tensor.alias != 'I'))
    assert 'no input a bulk copy takes' in eg.producer_refusal(star_only,
                                                               context)


@pytest.mark.parametrize('arch', ['sm_90', 'sm_90a', 'sm_100a'])
def test_a_star_stored_sparse_keeps_the_warps_staging(arch):
    """SeisSol's GPU memory layout stores `star` sparse, 24 of its 81 cells
    (`test_multi_element`'s `sparse-star`): its cells are read through
    their places, and the producer copies boxes stored dense -- refused,
    with the reason, and the plan stages as without the option."""
    context = me._context(arch, 's', multi_element='force',
                          element_producer=True, **AUTO)
    section, why = eg.classify(me._descrs('sparse-star', 6, 's'))
    assert section is not None, why
    lowered = eg.lower(section, context, split=True, ring=2)
    assert lowered.ring == 0
    assert 'star(0) is stored sparse' in eg.producer_refusal(lowered,
                                                             context)
    group, why = eg.fill(section, context)
    assert group is not None, why
    assert group.section.ring == 0 and group.producers == 0


def test_the_ring_is_counted_in_the_blocks_room():
    group = _plan()
    section = group.section
    single = eg.dataclasses.replace(section, ring=0)
    inputs = sum(b.per_element for b in section.inputs)
    assert (group.shared_bytes - eg.shared_bytes(single, group.elements)
            == 4 * ((section.ring - 1) * inputs * group.elements
                    + 2 * section.ring * 2))


# -- the kernel ----------------------------------------------------------------- #

def test_the_kernel():
    """A block of 17 warps over groups of 16; the ring of three slots set
    up for one arrival a phase on the full side and 16 on the empty one;
    the producer
    nine bulk copies a trip, a column of I each, 144 bytes, and the bytes of
    the group announced -- 16 x 9 x 144; the element warps meeting on their
    own barrier, 512 threads, and each giving the slot back."""
    gen = _kernel()
    src = gen.get_kernel()
    assert 'block 32x17x1' in src
    assert re.search(r'batchIdGroup0 = 16 \* \(blockIdx\.x\); \w+ < '
                     r'numElements0; \w+ \+= \(gridDim\.x \* 16\)', src)
    assert 'tensorforge::mbarrierInit(&me_ring[r], 1);' in src
    assert 'tensorforge::mbarrierInit(&me_ring[3 + r], 16);' in src
    copies = re.findall(r'tensorforge::bulkCopyOn\(&me_ring\[\w+\], '
                        r'&\w+\[[^\]]*\], &eI_\d+\[\d+\], (\d+), true\)', src)
    assert copies == ['144'] * 9
    assert re.search(r'tensorforge::mbarrierExpect\(&me_ring\[\w+\], '
                     r'20736u, ', src)
    assert ('asm volatile("barrier.sync %0, %1;" :: "r"((14 + (threadIdx.y '
            '>= 16))), "r"(512) : "memory");') in src
    assert len(re.findall(r'tensorforge::mbarrierArrive\(&me_ring\[3 \+ ',
                          src)) == 1
    assert '__syncthreads' not in src[src.index('for (size_t'):]


@pytest.mark.parametrize('arch,elements,budgets', [
    ('sm_90a', 16, (96, 40, 112)), ('sm_100a', 12, (128, 40, 152))])
def test_the_producers_registers_go_to_the_element_warps(arch, elements,
                                                          budgets):
    """`element_registers`: the producer a warpgroup, the element warps
    whole warpgroups; the kernel's entry fixed at the SM's registers over
    the block's threads, the producer keeping 40 and the element warps
    taking the rest."""
    group = _plan(arch, element_registers=40)
    assert group.elements == elements and group.producers == 4
    assert eg.register_budgets(group) == budgets


def test_registers_move_where_the_target_moves_them():
    """sm_90 has no setmaxnreg: the producer stays one warp."""
    group = _plan('sm_90', element_registers=40)
    assert group.producers == 1 and eg.register_budgets(group) is None


def test_the_kernel_moves_the_registers_once():
    """`__maxnreg__` in place of the launch bounds; the producer's
    warpgroup lowers its registers, the block meets, the element warps
    raise theirs -- ahead of the loop, once."""
    src = _kernel('sm_90a', element_registers=40).get_kernel()
    assert '__maxnreg__(96)' in src and '__launch_bounds__' not in src
    dec = src.index('setmaxnreg.dec.sync.aligned.u32 40;')
    inc = src.index('setmaxnreg.inc.sync.aligned.u32 112;')
    loop = src.index('for (size_t')
    assert dec < src.index('__syncthreads();', dec) < inc < loop
    assert src.count('setmaxnreg') == 2


# -- the host oracle -------------------------------------------------------------- #

def _two_trips(shape, arch, monkeypatch, **options):
    """A group of four over seven elements -- two trips, the second with
    warps masked -- through a ring of three: every element against NumPy,
    no access a barrier or the ring does not order (`kernel_eval.Races`),
    no wait for a phase that never completes."""
    import numpy as np
    from tensorforge.common.basic_types import Addressing
    from tensorforge.reference import kernel_eval
    group, count = 4, 7
    monkeypatch.setattr(eg, 'ELEMENTS', {key: group for key in eg.ELEMENTS})
    monkeypatch.setattr(eg, 'DEFAULT_ELEMENTS', group)
    gen = _kernel(arch, shape, 4, **options)
    built = gen.built._element_group
    assert built.elements == group and built.section.ring
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
    assert mults == group + built.producers
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


@pytest.mark.parametrize('arch,options', [
    ('sm_90', {}), ('sm_90a', {}), ('sm_100a', {}),
    ('sm_90', {'tensor_cores': False}),
    ('sm_90a', {'element_registers': 40}),
    ('sm_90', {'enable_sync_block_opt': False})],
    ids=['mma_sync', 'wgmma', 'tcgen05', 'fma', 'wgmma-registers',
         'own-barriers'])
@pytest.mark.parametrize('shape', ['volume', 'reuse', 'assign'])
def test_every_element_of_two_trips_on_the_oracle(shape, arch, options,
                                                  monkeypatch):
    _two_trips(shape, arch, monkeypatch, **options)


# -- on the hardware ----------------------------------------------------------- #

GPU_CASES = [('volume', 6), ('volume', 4), ('chain', 6), ('reuse', 6),
             ('assign', 6)]


@pytest.mark.parametrize('shape,order', GPU_CASES,
                         ids=[f'{s}-o{o}' for s, o in GPU_CASES])
def test_the_section_computes_on_the_gpu(shape, order, target, cache_root,
                                         tensorforge_include):
    """Built for the GPU present, of sm_90 or newer, with the producer --
    on the arch-specific target where there is one, whose engine the steps
    then take -- and run against NumPy over a batch that ends in a partial
    group and gives a block several trips; a hang is the harness's
    timeout."""
    if target.backend != 'cuda':
        pytest.skip('the multi-element layout is CUDA\'s')
    level = int(re.sub(r'\D', '', target.arch) or 0)
    if level < 90:
        pytest.skip('no bulk copy below sm_90')
    from harness.runner import run_case
    import dataclasses
    special = (dataclasses.replace(target, arch=target.arch + 'a')
               if target.arch in ('sm_90', 'sm_100') else target)
    group = eg.plan(me._descrs(shape, order, 's'), me._context(
        special.arch, 's', multi_element='force', element_producer=True,
        **AUTO))
    assert group is not None
    if not group.section.ring:
        # The RTX 5070's 100 kB hold no ring at o6 but for the assignment.
        pytest.skip(f'no room for the ring at {special.arch}')
    case = types.SimpleNamespace(
        NAME=f'multi_element_producer_{shape}_o{order}s',
        DTYPE=Datatype.F32, BATCH=4099, TOL=(1e-4, 1e-4),
        OPTIONS={'multi_element': 'force', 'element_producer': True,
                 **AUTO},
        INPUT_TRANSFORM=me._known_zeros(me._descrs(shape, order, 's')),
        descr_list=lambda: me._descrs(shape, order, 's'),
        reference=me._reference(me._descrs(shape, order, 's')))
    result = run_case(case, special, cache_root, tensorforge_include)
    assert result.passed, (result.max_abs_err, result.max_rel_err,
                           result.detail)
