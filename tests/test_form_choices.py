# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""`Options.multi_element` `auto`'s table (`element_group.FORM_CHOICES`,
`form_choice`, `auto_plan`) and the high-order tile asked for without
`force` (`Options.matrix_tile` `high`, `plan`).

What is checked here: which form each kind of section takes on which
target -- by its signature, its element type and the architecture -- the
row's fallback where its form refuses the section, `auto`'s rule where no
row names it, the table on NVIDIA alone; the tile asked for under `off`,
`auto` and `force` alike, on its engine whatever `matrix_engine` says, and
what each builds where the tile declines; the generator's gates for the
request; and what the kernel's comment says.  CPU only: the plans, and the
text of a few builds.
"""

from __future__ import annotations

import dataclasses
import re

import pytest

import hiorder_fixture as hf
import test_multi_element as me
import test_slot_mapping as sm
from tensorforge.common.basic_types import Datatype
from tensorforge.frontend.yateto import DescriptionReader
from tensorforge.generators import element_group as eg

AUTO = {'multi_element': 'auto'}
TILE = {'matrix_tile': 'high'}
#: The tile as `force` builds it with every option it implies asked for:
#: the kernels the jobs measured (`hi`).
FORCED = {'multi_element': 'force', 'matrix_engine': 'auto',
          'matrix_ring': 'prepared', 'matrix_star': 'engine', **TILE}


def _kernel(kernel, order, precision='s'):
    """SeisSol's kernel at `order` as `auto` sees it: the store's to o8
    (`/sparse`, its stars stored sparse; `test_slot_mapping._seissol`), the
    fixture's volume at o10 and o12, and the premultiplied fluxes
    (`hiorder_fixture.flux_description`)."""
    if kernel in hf.FLUXES:
        return DescriptionReader(None, {}).read(
            hf.flux_description(kernel, order, precision))[0]
    if order in hf.ORDERS:
        return DescriptionReader(None, {}).read(
            hf.description(order, precision))[0]
    return sm._seissol(kernel, order, precision)


def _form(group):
    """The form a plan takes: the tile, the slot form, or none -- the
    per-element kernel."""
    if group is None:
        return 'element'
    if group.section.tile is not None:
        return 'tile'
    if group.lanes is not None and group.lanes.mapping is not None:
        return 'slots'
    return 'other'


def test_every_row_is_a_form_on_its_targets():
    """Each row names one of the forms, a fallback only behind the tile --
    the slot form or the per-element kernel -- the tile at F32 alone, on
    the architectures it runs on, and no row hides behind an earlier one it
    shares a kind of section, an element type and an architecture with."""
    for i, row in enumerate(eg.FORM_CHOICES):
        assert row.form in eg.FORMS
        assert row.fallback in (None, 'slots', 'element')
        assert row.fallback is None or row.form == 'tile'
        if row.form == 'tile':
            assert row.dtypes == (Datatype.F32,)
            assert row.archs and set(row.archs) <= set(eg.TILE_ARCHS)
        for earlier in eg.FORM_CHOICES[:i]:
            assert not (earlier.signature == row.signature
                        and set(earlier.dtypes) & set(row.dtypes)
                        and (earlier.archs is None or row.archs is None
                             or set(earlier.archs) & set(row.archs))), \
                (earlier.why, row.why)


#: The kernels the jobs measured, at the orders they measured them.
MEASURED = ([(k, o) for k in ('gpu_volume', 'gpu_volume/sparse',
                              'gpu_localFlux[0]', 'gpu_localFluxAll',
                              'gpu_derivative', 'gpu_derivative/sparse')
             for o in (4, 6, 8)]
            + [('gpu_volume', o) for o in hf.ORDERS]
            + [(k, o) for k in hf.FLUXES for o in (6, 8, 10, 12)])


def test_every_row_names_a_seissol_kernel_and_its_jobs():
    """Each row of the table is the signature of a SeisSol kernel at an
    order the jobs measured, and says what was measured where: a job's
    number, or the local run's date."""
    signatures = {eg.slot_signature(eg.classify(_kernel(k, o))[0])
                  for k, o in MEASURED}
    for row in eg.FORM_CHOICES:
        assert row.signature in signatures, row.why
        assert re.search(r'(Vista|Horizon) \d+|local run, \d{4}-\d\d-\d\d',
                         row.why), row.why


#: (kernel, order, precision, arch, form): what `auto` takes, by the table
#: and, where no row names the section, by its rule -- the slot form on the
#: NVIDIA targets the jobs did not measure at o8.
AUTO_FORMS = [
    ('gpu_volume', 6, 's', 'sm_90a', 'slots'),
    ('gpu_volume', 6, 's', 'sm_100a', 'slots'),
    ('gpu_volume', 8, 's', 'sm_90a', 'tile'),
    ('gpu_volume', 8, 's', 'sm_100a', 'tile'),
    ('gpu_volume', 8, 's', 'sm_120', 'slots'),
    ('gpu_volume', 8, 's', 'sm_80', 'slots'),
    ('gpu_volume', 10, 's', 'sm_90a', 'tile'),
    ('gpu_volume', 10, 's', 'sm_100a', 'tile'),
    ('gpu_volume/sparse', 4, 's', 'sm_100a', 'slots'),
    ('gpu_volume/sparse', 8, 's', 'sm_90a', 'tile'),
    ('gpu_volume', 8, 'd', 'sm_90a', 'element'),
    ('gpu_volume', 8, 'd', 'sm_120', 'element'),
    ('gpu_localFlux[0]', 8, 's', 'sm_100a', 'element'),
    ('gpu_localFlux[0]', 8, 'd', 'sm_90a', 'element'),
    ('gpu_derivative', 8, 's', 'sm_90a', 'element'),
    ('gpu_derivative', 8, 'd', 'sm_100a', 'element'),
    ('gpu_derivative', 8, 'd', 'sm_90a', 'slots'),
    ('gpu_localFluxAllPre', 6, 's', 'sm_90a', 'tile'),
    ('gpu_localFluxAllPre', 8, 's', 'sm_100a', 'tile'),
    ('gpu_neighboringFluxPre[0]', 6, 's', 'sm_100a', 'element'),
    ('gpu_neighboringFluxPre[0]', 8, 's', 'sm_90a', 'element'),
    ('gpu_neighboringFluxPre[0]', 8, 's', 'sm_100a', 'slots'),
    ('gpu_neighboringFluxPre[0]', 10, 's', 'sm_90a', 'element'),
    ('gpu_neighboringFluxPre[0]', 10, 's', 'sm_100a', 'tile')]


@pytest.mark.parametrize('kernel,order,precision,arch,form', AUTO_FORMS,
                         ids=[f'{k.replace("/", "-")}-o{o}{p}-{a}'
                              for k, o, p, a, _ in AUTO_FORMS])
def test_auto_takes_the_form_its_table_names(kernel, order, precision, arch,
                                              form):
    """`multi_element=auto` on CUDA: the tile where a row names it and the
    section takes it, the group `force` takes with the tile asked for, its
    row in the kernel's comment; the per-element kernel where a row says
    so; the slot form where a row names it or none names the section."""
    descrs = _kernel(kernel, order, precision)
    group = eg.plan(descrs, me._context(arch, precision, **AUTO))
    assert _form(group) == form
    if form == 'tile':
        row = eg.form_choice(eg.classify(descrs)[0],
                             me._context(arch, precision))
        assert group.form_note == f'the high-order tile (FORM_CHOICES: ' \
                                  f'{row.why})'
        forced = eg.plan(descrs, me._context(arch, precision, **FORCED))
        assert (group.elements, group.section.tile) == (
            forced.elements, forced.section.tile)
    elif group is not None:
        assert group.form_note == ''


#: (kernel, order, arch, form): where the tile refuses the section, the
#: row's fallback -- the slot form, measured second behind the tile, or the
#: per-element kernel where the slot form refuses or lost.
FALLBACKS = [('gpu_volume', 8, 'sm_100a', 'slots'),
             ('gpu_volume/sparse', 8, 'sm_90a', 'slots'),
             ('gpu_volume', 10, 'sm_100a', 'element'),
             ('gpu_localFluxAllPre', 6, 'sm_90a', 'element'),
             ('gpu_neighboringFluxPre[0]', 10, 'sm_100a', 'element')]


@pytest.mark.parametrize('kernel,order,arch,form', FALLBACKS,
                         ids=[f'{k.replace("/", "-")}-o{o}-{a}'
                              for k, o, a, _ in FALLBACKS])
def test_where_the_tile_refuses_auto_takes_the_rows_fallback(
        monkeypatch, kernel, order, arch, form):
    """A row's tile that refuses the section by its own rule
    (`tile_refusal`, here made to) leaves it to the row's fallback, and the
    kernel's comment, or the plan's reason, says so."""
    monkeypatch.setattr(eg, 'tile_plan',
                        lambda *args, **kwargs: (None, 'refused here'))
    descrs = _kernel(kernel, order)
    context = me._context(arch, 's', **AUTO)
    section = eg.classify(descrs)[0]
    row = eg.form_choice(section, context)
    assert row.form == 'tile' and row.fallback == form
    group, why = eg.auto_plan(section, context)
    assert _form(group) == form
    if group is not None:
        assert group.form_note == (
            f'the high-order tile declined: refused here; the slot form, the '
            f'row\'s fallback (FORM_CHOICES: {row.why})')
    else:
        assert why == ('the high-order tile declined: refused here; the '
                       'per-element kernel, the row\'s fallback')
    assert _form(eg.plan(descrs, context)) == form


def test_a_row_without_a_fallback_leaves_the_section_to_autos_rule(
        monkeypatch):
    """A tile's row with no fallback, its tile refused: `auto`'s rule -- the
    slot form where its plan takes the section, and the comment says whose
    form it is."""
    descrs = _kernel('gpu_volume', 8)
    context = me._context('sm_100a', 's', **AUTO)
    row = eg.form_choice(eg.classify(descrs)[0], context)
    monkeypatch.setattr(eg, 'FORM_CHOICES',
                        (dataclasses.replace(row, fallback=None),))
    monkeypatch.setattr(eg, 'tile_plan',
                        lambda *args, **kwargs: (None, 'refused here'))
    group = eg.plan(descrs, context)
    assert _form(group) == 'slots'
    assert group.form_note == (f'the high-order tile declined: refused here; '
                               f'`auto`\'s rule (FORM_CHOICES: {row.why})')


def test_the_merged_faces_fall_back_to_the_per_element_kernel():
    """localFluxAll premultiplied, its four faces merged into one loop as
    `merge_variants=auto` merges them (`rolling.roll`): a section the tile
    refuses -- four steps on the asynchronous engines -- so `auto` builds
    it per element, the row's fallback, where its rule took the slot form
    (1.29x and 1.54x d's time on GB200 and GH200).  Written out, the tile."""
    from tensorforge.generators.rolling import ForDescr, roll
    descrs = _kernel('gpu_localFluxAllPre', 6)
    for arch in ('sm_90a', 'sm_100a'):
        context = me._context(arch, 's', **AUTO)
        options = context.get_user_options()
        merged = roll(list(descrs), min_count=options.merge_min_count,
                      max_arity=options.merge_max_arity)
        assert any(isinstance(d, ForDescr) for d in merged)
        flat = [op for d in merged for op in d.operations()]
        section = eg.classify(flat)[0]
        assert eg.form_choice(section, context).fallback == 'element'
        assert eg.plan(flat, context) is None
        assert eg.slot_decline(section, context) is not None
        assert _form(eg.plan(descrs, context)) == 'tile'


def test_the_table_is_nvidias():
    """Off CUDA no row names a section: on Intel under SPMD `auto` takes the
    lanes form where it measured to pay -- localFlux at o6, which the table
    builds per element on NVIDIA, included."""
    descrs = _kernel('gpu_localFlux[0]', 6)
    section = eg.classify(descrs)[0]
    pvc = me._context('pvc', 's', backend='oneapi', **AUTO)
    assert eg.form_choice(section, pvc) is None
    group = eg.plan(descrs, pvc)
    assert group is not None and group.lanes is not None
    assert group.lanes.mapping is None
    cuda = me._context('sm_90a', 's', **AUTO)
    assert eg.form_choice(section, cuda).form == 'element'
    assert eg.plan(descrs, cuda) is None


@pytest.mark.parametrize('mode', ['off', 'auto', 'force'])
def test_the_tile_asked_for_takes_the_section_under_every_mode(mode):
    """`matrix_tile=high` takes the tile wherever the section takes it, with
    `multi_element` `off`, `auto` and `force` alike -- the o6 volume too,
    which `auto`'s table builds in the slot form -- on the target's
    asynchronous engine though `matrix_engine` is left `mma_sync`."""
    descrs = _kernel('gpu_volume', 6)
    for arch, engine in (('sm_90a', 'wgmma'), ('sm_100a', 'tcgen05')):
        context = me._context(arch, 's', multi_element=mode, **TILE)
        assert context.get_user_options().matrix_engine == 'mma_sync'
        group = eg.plan(descrs, context)
        assert _form(group) == 'tile'
        assert group.section.tile.engine == engine
        assert group.tile_note == '' and group.form_note == ''


def test_where_the_tile_asked_for_declines_off_builds_per_element():
    """Where the section does not take the tile -- F64, a chained section, a
    target without the engines, the matrix engines turned off -- `off`
    builds per element, as without the request; `auto` follows its table,
    and its group says why the tile was not taken."""
    for descrs, precision, arch in (
            (_kernel('gpu_volume', 6, 'd'), 'd', 'sm_100a'),
            (_kernel('gpu_derivative', 6), 's', 'sm_90a'),
            (_kernel('gpu_volume', 6), 's', 'sm_120')):
        assert eg.plan(descrs, me._context(arch, precision, **TILE)) is None
    assert eg.plan(_kernel('gpu_volume', 6), me._context(
        'sm_100a', 's', tensor_cores=False, **TILE)) is None
    context = me._context('sm_100a', 'd', **AUTO, **TILE)
    group = eg.plan(_kernel('gpu_volume', 6, 'd'), context)
    assert _form(group) == 'slots'
    assert group.tile_note.startswith('F64 sections have no asynchronous '
                                      'engine')
    assert eg.tile_notes(group, context) == [
        f'matrix_tile=high declined: {group.tile_note}']
    assert eg.plan(_kernel('gpu_volume', 8, 'd'), context) is None


def _text(descrs, context):
    """A build's kernel, launcher and header, its name and the options line
    taken out: what two builds of one kernel under other options share."""
    gen = me._generate(descrs, context)
    text = '\n'.join([gen.get_kernel(), gen.get_launcher() or '',
                      gen.get_header() or ''])
    lines = [line.replace(gen.get_base_name(), 'KERNEL')
             for line in text.splitlines()
             if not line.strip().startswith('// options:')]
    return lines


@pytest.mark.parametrize('arch', ['sm_90a', 'sm_100a'])
def test_the_tile_without_force_is_the_kernel_force_builds(arch):
    """Built, the tile asked for under `off` and the tile `auto`'s table
    names are the kernel `force` builds with the tile and every option it
    implies asked for -- the folded star and B prepared in the comment as
    there -- but for the name, the options line and `auto`'s line on why."""
    descrs = _kernel('gpu_volume', 8)
    forced = _text(descrs, me._context(arch, 's', **FORCED))
    assert any(line.strip().startswith('// star step: the star step folded')
               for line in forced)
    assert any(line.strip().startswith('// operator ring: B of kDivM(0)')
               for line in forced)
    assert _text(descrs, me._context(arch, 's', **TILE)) == forced
    auto = _text(descrs, me._context(arch, 's', **AUTO))
    note = [line for line in auto
            if line.strip().startswith('// multi_element=auto: ')]
    assert len(note) == 1 and 'the high-order tile (FORM_CHOICES: the ' \
                              'volume at o8' in note[0]
    assert [line for line in auto if line not in note] == forced


def test_a_kernel_the_asked_tile_takes_is_neither_cut_nor_reassociated(
        monkeypatch):
    """The generator's gates for the request as for `auto` and `force`:
    `fission=auto` builds no probe of the list to decide its cuts by
    (`Generator._decide_by_fetch`, here made to fail) -- the tile plans the
    section whole, where the o8 volume per element is cut in two on
    sm_100a -- and the description is kept from `reassociate_products`, the
    order the tile folds its star step in (`Generator._lanes_take`), as it
    is not without the request."""
    from tensorforge.generators.generator import Generator

    def probed(self, *args, **kwargs):
        raise AssertionError('a fission probe of a kernel the tile takes '
                             'whole')
    monkeypatch.setattr(Generator, '_decide_by_fetch', probed)
    descrs = _kernel('gpu_volume', 8)
    context = me._context('sm_100a', 's', fission='auto',
                          reassociate_products=True, **TILE)
    gen = me._generate(descrs, context)
    assert gen._fission_pieces == 0
    built = gen.built if getattr(gen, 'built', None) is not None else gen
    assert built._element_group is not None
    assert built._element_group.section.tile is not None
    assert len(built._sections) == 1
    assert gen._lanes_take(context)
    assert not gen._lanes_take(me._context('sm_100a', 's',
                                           reassociate_products=True))
