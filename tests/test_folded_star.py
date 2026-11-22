# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A for the asynchronous engines, written once by the star step
(`Options.matrix_star`): `halves` -- the star step splits A into its TF32
halves in the group's buffer (`element_group.Buffer.halves`), and wgmma's
ldmatrix and tcgen05's store into tensor memory read both halves, splitting
nothing; `engine` -- the star step folded into the operator step that reads
its intermediates as A (`element_group.fuse`): a block of eight rows of K of
every product at a time, A's halves written straight where the engine reads
them, tcgen05's tensor memory through a ring of slots handed over on
mbarriers, wgmma's shared memory in the canonical layout its descriptors
read, never in shared memory as values -- B resident, or streamed prepared
(`Options.matrix_ring` `prepared`) in the same blocks.

What is checked here: the plans -- the halves where they fit, the folded
star's group (o8 tcgen05 fourteen elements, against the seven of the
buffers), its refusals and the kernel's note of them; the kernel's text;
the host oracle over two trips of a group with a masked warp -- both
engines, B resident and streamed, the halves bit for bit as the buffers'
-- against NumPy and without a race; the builds for sm_90a and sm_100a;
and on a GH200 or a GB200, the volume against NumPy.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
import re
import types

import pytest

import test_multi_element as me
import test_multi_element_async as tma
from tensorforge.common.basic_types import Datatype
from tensorforge.generators import element_group as eg

#: The asynchronous engines, B prepared.
AUTO = {'matrix_engine': 'auto', 'matrix_ring': 'prepared'}


def _group(order=8, arch='sm_100a', shape='volume', **options):
    context = me._context(arch, 's', multi_element='force',
                          **{**AUTO, **options})
    section, why = eg.classify(me._descrs(shape, order, 's'))
    assert section is not None, why
    group, why = eg.fill(section, context)
    assert group is not None, why
    return group, context


def _operator(group):
    (step,) = [s for s in group.section.steps if s.kind is eg.Kind.OPERATOR]
    return step


# -- the plans ------------------------------------------------------------------ #

@pytest.mark.parametrize('arch', ['sm_90a', 'sm_100a'])
def test_the_halves_reach_the_asynchronous_engines_where_they_fit(arch):
    """`halves`: at o6 the star step writes both halves of as many of the
    three intermediates as fit; at o8 none fits beside the ring, and the
    group keeps its size."""
    group, _ = _group(6, arch, matrix_star='halves')
    assert eg.halves(group.section) > 0
    assert all(b.halves for b in eg.halvable(group.section)[
        :eg.halves(group.section)])
    buffers, _ = _group(8, arch)
    group, _ = _group(8, arch, matrix_star='halves')
    assert group.elements == buffers.elements


@pytest.mark.parametrize('arch,elements', [('sm_90a', 12), ('sm_100a', 14)])
def test_the_folded_star_frees_the_buffers_of_a(arch, elements):
    """`engine` at o8: the star steps folded into the operator step, their
    intermediates held by the engine and none of the group's room -- on
    tcgen05 the group doubles to fourteen, 126 of the 128 lanes; B streamed
    prepared in blocks of eight rows of every product."""
    group, context = _group(8, arch, matrix_star='engine')
    section = group.section
    assert group.elements == elements and section.star == 'engine'
    assert [s.lowering for s in section.steps][:3] == ['fused'] * 3
    assert all(b.held and b.room == 0 for b in section.intermediates)
    step = _operator(group)
    assert eg.folded(section, step)
    chunks = eg.ring_chunks(section, step)
    assert len(chunks) == 11
    assert chunks[0] == ((0, 0, 8), (1, 0, 8), (2, 0, 8))
    (note,) = eg.star_notes(section, context)
    assert note.startswith('the star step folded into Q: A\'s halves into ')
    if arch == 'sm_100a':
        from tensorforge.backend.instructions.compute import multi_element
        assert multi_element.tmem_columns(section) == 512
        assert multi_element._tcgen05_columns(section, step)[0] == \
            128 + eg.STAR_SLOTS['tcgen05'] * 48


def test_the_folded_star_needs_b_resident_or_prepared():
    """B streamed through the converting ring, a product's chunks at a time,
    cannot follow the folded star's blocks: refused, with the reason in the
    kernel's note, and A in halves where they fit instead -- never a silent
    switch."""
    group, context = _group(8, 'sm_100a', matrix_star='engine',
                            matrix_ring='convert')
    assert group.section.star == ''
    assert [s.lowering for s in group.section.steps][:3] == ['mix'] * 3
    (note,) = eg.star_notes(group.section, context)
    assert 'matrix_star=engine declined for Q: B streams through the ' \
        'converting ring' in note
    assert note.endswith('A in TF32 halves where they fit')
    # As `halves` would: the buffer of A that fits in halves beside the
    # converting ring, B's halves kept -- not the folded star over a B read
    # whole in TF32, which fits the group too.
    halves, _ = _group(8, 'sm_100a', matrix_star='halves',
                       matrix_ring='convert')
    assert eg.halves(group.section) == eg.halves(halves.section) == 1
    assert group.elements == halves.elements == 7
    assert eg.split(group.section)
    resident, _ = _group(6, 'sm_100a', matrix_star='engine',
                         matrix_ring='convert')
    assert resident.section.star == 'engine'


def test_b_keeps_its_halves_ahead_of_a():
    """On the asynchronous engines B whole is B in TF32
    (`multi_element._terms`), while A's halves spare work: `halves` takes
    them only behind B's -- localFluxAll at o6 on sm_90a, B's halves and one
    buffer of A's at a group of eight, where B whole held all three -- and
    the folded star never reads B whole (`fuse_refusal`): at o4 the fold
    does not leave a group of sixteen room for B's halves, and the unfolded
    tiling takes A's halves there instead."""
    group, _ = _group(6, 'sm_90a', shape='localFluxAll', matrix_star='halves')
    assert eg.split(group.section) and group.elements == 8
    assert eg.halves(group.section) == 1
    group, context = _group(4, 'sm_90a', shape='localFluxAll',
                            matrix_star='engine')
    assert eg.split(group.section) and group.elements == 16
    assert group.section.star == '' and eg.halves(group.section) == 3
    assert all(note.endswith('the fold does not fit the group the fill rule '
                             'chose, which the unfolded tiling does; A in '
                             'TF32 halves where they fit')
               for note in eg.star_notes(group.section, context)
               if 'declined for _tmp' in note)
    whole = eg.lower(group.section, context, star='engine')
    assert not eg.split(whole) and whole.star == ''
    folding = [s for s in whole.steps if s.lowering in eg.ASYNC_LOWERINGS
               and eg.star_mixes(whole, s) is not None]
    assert folding and all(eg.fuse_refusal(whole, s) == (
        'B is read whole, in TF32, where its halves leave the group no '
        'room: the folded star takes B\'s halves') for s in folding)


def test_a_step_whose_a_is_an_input_is_not_folded():
    """The chain's operator steps read I, an input, as A: there is no star
    step to fold."""
    group, context = _group(6, 'sm_100a', shape='chain', matrix_star='engine')
    assert group.section.star == ''
    notes = eg.star_notes(group.section, context)
    assert notes and all('declined' in n for n in notes)


# -- the kernel ----------------------------------------------------------------- #

@pytest.mark.parametrize('arch', ['sm_90a', 'sm_100a'])
def test_the_folded_kernel_stores_a_where_the_engine_reads_it(arch):
    """No buffer of the intermediates; tcgen05: the bands' stores into
    tensor memory, a commit a block onto A's ring and onto B's; wgmma: A by
    descriptors, no ldmatrix, its stores fenced for the async proxy."""
    src = tma._kernel(arch, 'volume', 8, 's', matrix_star='engine',
                      **AUTO).get_kernel()
    assert 'gtmp' not in src
    assert '// star step: the star step folded into Q' in src
    if arch == 'sm_100a':
        assert src.count('tensorforge::tcgen05CommitOn(') == 2 * 11
        assert 'tcgen05.st.sync.aligned.32x32b.x8' in src
    else:
        assert 'ldmatrix' not in src
        # Each block's stores of A fenced for the engine's reads: the only
        # proxy fences of the kernel, B being copied (`matrix_ring`).
        assert src.count('tensorforge::fenceProxyAsync();') == 11


# -- the oracle ------------------------------------------------------------------ #

def _two_trips_at(order, shape, arch, monkeypatch, streamed, **options):
    """`test_multi_element_async._two_trips` at `order`: a group of four over
    seven elements, B streamed where `streamed` (o4's B is not larger than
    the folded star's ring, so the ring cases run at o6)."""
    src = inspect.getsource(tma._two_trips)
    for old, new in (
            ('def _two_trips(shape, arch, monkeypatch, streamed=False, '
             'group=4, count=7,', 'def two(shape, arch, monkeypatch, '
             'streamed=False, group=4, count=7, order=4,'),
            ('_streaming(monkeypatch, arch, shape, 4)',
             '_streaming(monkeypatch, arch, shape, order)'),
            ("gen = _kernel(arch, shape, 4, 's', **options)",
             "gen = _kernel(arch, shape, order, 's', **options)"),
            ("me._descrs(shape, 4, 's')", "me._descrs(shape, order, 's')")):
        assert old in src
        src = src.replace(old, new)
    space = dict(vars(tma))
    exec(src, space)
    monkeypatch.setattr(tma, 'AUTO', {**tma.AUTO, **options})
    space['two'](shape, arch, monkeypatch, streamed=streamed, order=order,
                 **options)


@pytest.mark.parametrize('arch', ['sm_90a', 'sm_100a'])
@pytest.mark.parametrize('order,streamed', [(4, False), (6, False),
                                            (6, True)],
                         ids=['o4-resident', 'o6-resident', 'o6-ring'])
def test_the_folded_star_over_two_trips_on_the_oracle(order, streamed, arch,
                                                      monkeypatch):
    """Every element of two trips against NumPy, and no access a barrier,
    ring or wait does not order: the bands' A read only behind its slot's
    full side, a slot written again only behind the products that read it
    (o6 goes round tcgen05's ring of four within a trip, five blocks; with
    B resident, as nothing but A's empty side lands the products there --
    with B streamed, the producer's waits for B's slots land them in the
    oracle's order before the bands come round)."""
    _two_trips_at(order, 'volume', arch, monkeypatch, streamed,
                  matrix_star='engine', **AUTO)
    src = tma._kernel(arch, 'volume', order, 's', matrix_star='engine',
                      **AUTO).get_kernel()
    assert '// star step: the star step folded into Q' in src


@pytest.mark.parametrize('arch', ['sm_90a', 'sm_100a'])
def test_the_folded_star_with_own_barriers_on_the_oracle(arch, monkeypatch):
    _two_trips_at(4, 'volume', arch, monkeypatch, False,
                  matrix_star='engine', enable_sync_block_opt=False, **AUTO)


@pytest.mark.parametrize('arch', ['sm_90a', 'sm_100a'])
def test_the_folded_star_reads_only_what_was_written(arch, monkeypatch):
    """Shared memory nothing wrote reads NaN, tensor memory too: the folded
    star still gives NumPy's numbers."""
    from tensorforge.reference import kernel_eval
    read = kernel_eval.Slot.read

    def poisoned(self, base, index):
        if (base in (kernel_eval.Races.SHARED, 'tmem')
                and (base, int(index)) not in self.data):
            return math.nan
        return read(self, base, index)
    monkeypatch.setattr(kernel_eval.Slot, 'read', poisoned)
    _two_trips_at(4, 'volume', arch, monkeypatch, False,
                  matrix_star='engine', **AUTO)


@pytest.mark.parametrize('arch', ['sm_90a', 'sm_100a'])
@pytest.mark.parametrize('streamed', [False, True], ids=['resident', 'ring'])
def test_the_halves_over_two_trips_on_the_oracle(streamed, arch,
                                                 monkeypatch):
    """`halves` on the asynchronous engines: every element of two trips
    against NumPy, B resident and streamed prepared."""
    _two_trips_at(4, 'volume', arch, monkeypatch, streamed,
                  matrix_star='halves', matrix_ring_rows=8, **AUTO)
    # The plan the run took, under the same group and room: A in halves.
    built = tma._kernel(arch, 'volume', 4, 's', matrix_star='halves',
                        matrix_ring_rows=8, **AUTO).built._element_group
    assert eg.halves(built.section) > 0
    assert eg.streamed(built.section) is streamed


@pytest.mark.parametrize('arch', ['sm_90a', 'sm_100a'])
def test_the_halves_give_d_bit_for_bit(arch, monkeypatch):
    """The star step's split is the one the engine's reader would make: with
    A in halves the outputs are the buffers' to the last bit."""
    got, want = me._oracle_outputs('volume', 's', arch,
                                   {**AUTO, 'matrix_star': 'halves'})
    plain, _ = me._oracle_outputs('volume', 's', arch, AUTO)
    assert [key for key in want if got[key] != plain[key]] == []
    scale = max(abs(v) for v in want.values())
    assert max(abs(got[key] - want[key]) for key in want) <= 1e-6 * scale


# -- the build ------------------------------------------------------------------- #

@pytest.mark.parametrize('arch,engine', [('sm_90a', 'HGMMA'),
                                         ('sm_100a', 'UTCHMMA')])
def test_the_folded_build(arch, engine, tmp_path):
    """Built for the arch-specific target at o8: the products as before, a
    bulk copy a block, no ldmatrix."""
    sass = tma._sass(arch, tmp_path, 'volume', 8, matrix_star='engine',
                     matrix_ring='prepared')
    assert tma._count(sass, engine) == 99
    assert tma._count(sass, 'UBLKCP') == 11
    assert tma._count(sass, 'LDSM') == 0


@pytest.mark.parametrize('ring', ['prepared', 'convert'])
@pytest.mark.parametrize('order', [4, 6])
def test_several_folded_steps_build(order, ring, tmp_path):
    """localFluxAll folds a star step into each of its four operator steps on
    tcgen05: each sets up A's ring of its own, named after its step
    (`ta{n}`) as B's prepared ring is -- one name for all would be declared
    four times in one scope, which nvcc refuses.  Built for sm_100a."""
    src = tma._kernel('sm_100a', 'localFluxAll', order, 's',
                      matrix_star='engine', matrix_ring=ring).get_kernel()
    assert src.count('// star step: the star step folded into') == 4
    rings = re.findall(r'\b(ta\d+)_ring\s*=', src)
    assert len(rings) == len(set(rings)) == 4, rings
    sass = tma._sass('sm_100a', tmp_path, 'localFluxAll', order,
                     matrix_star='engine', matrix_ring=ring)
    assert tma._count(sass, 'UTCHMMA') > 0


@pytest.mark.parametrize('order', [4, 6])
def test_several_folded_steps_on_the_oracle(order, monkeypatch):
    """The four folded steps of localFluxAll over two trips of a group with a
    masked warp, each over A's ring of its own: against NumPy and without a
    race."""
    _two_trips_at(order, 'localFluxAll', 'sm_100a', monkeypatch, False,
                  matrix_star='engine', **AUTO)


# -- on the hardware ------------------------------------------------------------- #

GPU_CASES = [('volume', 8, 'engine'), ('volume', 6, 'engine'),
             ('volume', 6, 'halves')]


@pytest.mark.parametrize('shape,order,star', GPU_CASES,
                         ids=[f'{s}-o{o}-{t}' for s, o, t in GPU_CASES])
def test_the_star_computes_on_the_gpu(shape, order, star, target, cache_root,
                                      tensorforge_include):
    """On a GH200 (wgmma) or a GB200 (tcgen05), B prepared: against NumPy
    over a batch ending in a partial group, several trips a block -- the
    operators their description's numbers, which the prepared kernel
    multiplies whatever memory holds (`test_multi_element._described`), each
    element within 1e-5 of its largest cell (`_normwise`)."""
    if target.backend != 'cuda' or target.arch not in ('sm_90', 'sm_100'):
        pytest.skip('no GPU with an asynchronous matrix engine')
    from harness.runner import run_case
    special = dataclasses.replace(target, arch=target.arch + 'a')
    tma._nvcc_builds(special.arch)
    options = {**AUTO, 'matrix_star': star}
    group = eg.plan(me._descrs(shape, order, 's'), me._context(
        special.arch, 's', multi_element='force', **options))
    assert group is not None
    assert (group.section.star == 'engine') is (star == 'engine')
    case = types.SimpleNamespace(
        NAME=f'folded_star_{shape}_o{order}s_{star}',
        DTYPE=Datatype.F32, BATCH=4099, TOL=me._normwise(),
        OPTIONS={'multi_element': 'force', **options},
        INPUT_TRANSFORM=me._described(me._descrs(shape, order, 's')),
        descr_list=lambda: me._descrs(shape, order, 's'),
        reference=me._reference(me._descrs(shape, order, 's')))
    result = run_case(case, special, cache_root, tensorforge_include)
    assert result.passed, (result.max_abs_err, result.max_rel_err,
                           result.detail)
