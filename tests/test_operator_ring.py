# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The asynchronous engines' operator B prepared at generation
(`Options.matrix_ring` `prepared`, `element_group.Stage.prepared`): its TF32
halves made from the description's numbers, in the order and the canonical
layout the engine's descriptors read, as a table of the generated file
(`element_group.prepared_table`) -- copied once a block where B is resident,
streamed through a ring of `matrix_ring_slots` slots where it is not
(`multi_element._OperatorRing`): the block's first thread fills the slots by
bulk copies, tcgen05's products commit onto a slot's empty side, wgmma's
warps give a slot back behind their wait.

What is checked here: the halves (`prepare.tf32_halves`, the device's split
bit for bit); the plan -- the option off changes nothing, on it takes the
converting ring's room, an operator without numbers keeps the converting
ring and the kernel's comment says why; the table -- what the converting
ring would put, piece by piece, named after its numbers; the kernel's text;
the PIR's commit onto a ring's slot; the host oracle over two trips of a
group with masked elements -- tcgen05 and wgmma, streamed and resident,
with and without the barrier placement, with unwritten shared memory read
as NaN -- against NumPy and without a race; the contract of the
description's numbers -- other numbers in the operators' memory, the
prepared kernel's result unchanged, the converting one's not; the builds
for sm_90a and sm_100a read back; and on a GH200 or a GB200, the sections
against NumPy, the harness filling the operators with their description's
numbers.
"""

from __future__ import annotations

import dataclasses
import math
import re
import struct
import types

import numpy as np
import pytest

import test_multi_element as me
import test_multi_element_async as tma
from tensorforge.common.basic_types import Datatype
from tensorforge.common.matrix.prepare import tf32_halves
from tensorforge.generators import element_group as eg

#: The prepared ring asked for, on the asynchronous engines.
PREPARED = {'matrix_engine': 'auto', 'matrix_ring': 'prepared'}


def _group(shape='volume', order=8, arch='sm_100a', **options):
    context = me._context(arch, 's', multi_element='force',
                          **{**PREPARED, **options})
    section, why = eg.classify(me._descrs(shape, order, 's'))
    assert section is not None, why
    group, why = eg.fill(section, context)
    assert group is not None, why
    return group, context


def _step(group):
    (step,) = [s for s in group.section.steps if s.kind is eg.Kind.OPERATOR]
    return step


def _stages(group):
    return [group.section.stage(p.first.tensor, (p.rows, p.depth))
            for p in _step(group).products]


# -- the halves ---------------------------------------------------------------- #

def _rn_tf32(value: float) -> float:
    """`cvt.rn.tf32.f32` by the book: ten mantissa bits, ties to even."""
    bits = struct.unpack('<I', struct.pack('<f', value))[0]
    keep, rest = bits >> 13, bits & 0x1FFF
    if rest > 0x1000 or (rest == 0x1000 and keep & 1):
        keep += 1
    return struct.unpack('<f', struct.pack('<I', (keep << 13) & 0xFFFFFFFF))[0]


def test_the_halves_are_the_devices_split():
    """The upper half rounded to nearest even at ten mantissa bits -- ties
    included, both signs, subnormals -- the lower the exact remainder, the
    two summing to the value: what `splitFloatTF32` gives on sm_90 on."""
    rng = np.random.default_rng(7)
    values = np.concatenate([
        (rng.standard_normal(4000) * 10.0 ** rng.integers(
            -30, 30, 4000)).astype(np.float32),
        np.array([0.0, -0.0, 1.0, -1.0, 1.0 + 2.0 ** -11, 1.0 + 3 * 2.0 ** -11,
                  -(1.0 + 2.0 ** -11), 2.0 ** -140, 1e-45, 20.0, 21.0, 0.1],
                 dtype=np.float32)])
    upper, lower = tf32_halves(values)
    for v, u, lo in zip(values, upper, lower):
        assert float(u) == _rn_tf32(float(v)), v
        assert struct.unpack('<I', struct.pack('<f', float(u)))[0] & 0x1FFF \
            == 0
        assert float(np.float32(u) + np.float32(lo)) == float(v), v
    away, _ = tf32_halves(np.array([1.0 + 2.0 ** -11], dtype=np.float32),
                          ties='away')
    assert float(away[0]) == 1.0 + 2.0 ** -10


# -- the plan ------------------------------------------------------------------ #

@pytest.mark.parametrize('arch', ['sm_90a', 'sm_100a'])
@pytest.mark.parametrize('order', [6, 8])
def test_the_option_off_prepares_nothing(arch, order):
    """The default converts B as before: no stage prepared, no ring of the
    option's, and no note in the kernel's comment."""
    context = me._context(arch, 's', multi_element='force',
                          matrix_engine='auto')
    section, _ = eg.classify(me._descrs('volume', order, 's'))
    group, _ = eg.fill(section, context)
    assert not eg.prepared(group.section)
    assert group.section.ring_slots == 0
    assert eg.ring_notes(group.section, context) == []


@pytest.mark.parametrize('arch,elements,width', [('sm_90a', 12, 120),
                                                 ('sm_100a', 7, 128)])
def test_o8_streams_b_prepared_in_the_converting_rings_room(arch, elements,
                                                            width):
    """At o8 B's halves are streamed: prepared, through four slots of
    sixteen rows of K -- the room of the converting ring's two of
    thirty-two -- so that the group keeps its size."""
    group, _ = _group('volume', 8, arch)
    assert group.elements == elements
    stages = _stages(group)
    assert all(st.prepared and st.streamed and st.split for st in stages)
    section = group.section
    assert (section.ring_slots, section.ring_rows) == (4, 16)
    chunks = eg.ring_chunks(section, _step(group))
    assert len(chunks) == 3 * 6
    assert chunks[:7] == tuple(((0, k0, min(16, 88 - k0)),)
                               for k0 in range(0, 88, 16)) + (((1, 0, 16),),)
    assert eg.ring_volume(section) == 4 * 16 * width * 2 + 2 * 4 * 2
    converting, _ = _group('volume', 8, arch, matrix_ring='convert')
    assert eg.ring_volume(converting.section) + 16 == eg.ring_volume(section)


@pytest.mark.parametrize('arch', ['sm_90a', 'sm_100a'])
def test_o6_copies_b_prepared_once(arch):
    """At o6 B is resident: prepared, and copied once a block -- no ring."""
    group, _ = _group('volume', 6, arch)
    stages = _stages(group)
    assert all(st.prepared and not st.streamed for st in stages)
    assert group.section.ring_slots == 0 and eg.ring_volume(group.section) == 0


def test_the_ring_takes_the_options_slots_and_rows():
    group, _ = _group('volume', 8, 'sm_100a', matrix_ring_slots=3,
                      matrix_ring_rows=32)
    section = group.section
    assert (section.ring_slots, section.ring_rows) == (3, 32)
    assert len(eg.ring_chunks(section, _step(group))) == 3 * 3
    assert eg.ring_volume(section) == 3 * 32 * 128 * 2 + 2 * 3 * 2


def test_an_operator_without_numbers_keeps_the_converting_ring(monkeypatch):
    """The halves are made from the description's numbers: a step one of
    whose operators carries none keeps B converted by the element warps --
    and the kernel's comment says so, and why."""
    section, _ = eg.classify(me._descrs('volume', 8, 's'))
    (step,) = [s for s in section.steps if s.kind is eg.Kind.OPERATOR]
    tensor = step.products[1].first.tensor
    monkeypatch.setattr(tensor, 'has_values', lambda: False)
    assert eg.step_refusal(step) == f'{tensor.alias} carries no numbers'
    context = me._context('sm_100a', 's', multi_element='force', **PREPARED)
    group, why = eg.fill(section, context)
    assert group is not None, why
    assert not eg.prepared(group.section)
    assert all(st.streamed for st in _stages(group))
    (note,) = eg.ring_notes(group.section, context)
    assert note.startswith('matrix_ring=prepared declined for kDivM(0), '
                           'kDivM(1), kDivM(2) into Q: kDivM(1) carries no '
                           'numbers')


# -- the table ----------------------------------------------------------------- #

def _expected_piece(stage, k0, rows):
    """A piece of the converting ring as `_ring_put` lays it: the window's
    numbers, zero outside it, split, the upper halves' canonical layout over
    the piece's rows and then the lower halves'."""
    values = np.asarray(stage.tensor.get_values(), dtype=np.float64)
    (r0, r1), (d0, d1) = stage.window
    out = np.zeros(2 * rows * stage.width, dtype=np.float32)
    for k in range(rows):
        for n in range(stage.width):
            row, col = stage.n0 + n, d0 + k0 + k
            value = (values[row, col] if r0 <= row < r1 and col < d1
                     else 0.0)
            upper, lower = tf32_halves(np.array([value], dtype=np.float32))
            place = (n // 8) * rows * 8 + (k // 4) * 32 + (n % 8) * 4 + k % 4
            out[place] = upper[0]
            out[rows * stage.width + place] = lower[0]
    return out


@pytest.mark.parametrize('arch', ['sm_90a', 'sm_100a'])
def test_the_table_holds_what_the_converting_ring_puts(arch):
    """Every piece of the table, chunk after chunk, is the slot the element
    warps would have put for it -- its window's numbers split, zeros past
    the window's rows and contraction, the canonical layout over its rows."""
    group, _ = _group('volume', 8, arch, matrix_ring_slots=2,
                      matrix_ring_rows=32)
    step = _step(group)
    stages = _stages(group)
    name, numbers, offsets = eg.prepared_table(group.section, step, 90)
    assert re.fullmatch(r'tf_bk_[0-9a-f]{16}', name)
    table = np.asarray(numbers, dtype=np.float32)
    at = 0
    for (p, k0, rows), in eg.ring_chunks(group.section, step):
        assert offsets[(p, k0)] == at
        want = _expected_piece(stages[p], k0, rows)
        assert np.array_equal(table[at:at + want.size], want), (p, k0)
        at += want.size
    assert at == len(numbers)


def test_the_table_is_zero_outside_the_windows(monkeypatch):
    """An operator with numbers everywhere -- kDivM's own are zero past its
    contraction, so they cannot tell -- gives a table that is zero past each
    product's window, its rows and its contraction both: the tilings'
    padded steps of K multiply an input's padding by it (`Stage`)."""
    group, _ = _group('volume', 8, 'sm_100a', matrix_ring_slots=2,
                      matrix_ring_rows=32)
    step = _step(group)
    stages = _stages(group)
    rng = np.random.default_rng(3)
    for stage in stages:
        dense = rng.uniform(1.0, 2.0, np.asarray(
            stage.tensor.get_values()).shape)
        monkeypatch.setattr(stage.tensor, 'get_values',
                            lambda dense=dense: dense)
    _, numbers, offsets = eg.prepared_table(group.section, step, 100)
    table = np.asarray(numbers, dtype=np.float32)
    for (p, k0, rows), in eg.ring_chunks(group.section, step):
        want = _expected_piece(stages[p], k0, rows)
        assert np.array_equal(table[offsets[(p, k0)]:offsets[(p, k0)]
                                    + want.size], want), (p, k0)
    (_, _), (_, d1) = stages[0].window
    assert d1 < stages[0].depth and np.count_nonzero(table) < table.size


def test_the_oracle_refuses_a_copy_past_a_table():
    """A bulk copy out of a table of the generated file stays inside it: the
    host oracle refuses one that reads past its numbers, which no kernel
    built here does -- a read past the definition on the device."""
    from tensorforge.reference import kernel_eval
    kernel_eval._table_reach([1.0] * 8, 4, 4, 'copy')
    with pytest.raises(kernel_eval.Abort, match='out of a table of 8'):
        kernel_eval._table_reach([1.0] * 8, 6, 4, 'copy')
    with pytest.raises(kernel_eval.Abort):
        kernel_eval._table_reach([1.0] * 8, -4, 4, 'copy')


def test_a_resident_table_is_each_stage_in_its_canonical_layout():
    """Resident, the table is each product's stage whole, in the layout
    `MultiElementOperators._canonical` stores (`Stage.canonical_offset`)."""
    group, _ = _group('volume', 6, 'sm_100a')
    step = _step(group)
    stages = _stages(group)
    _, numbers, offsets = eg.prepared_table(group.section, step, 100)
    for p, stage in enumerate(stages):
        piece = np.asarray(numbers[offsets[(p, 0)]:offsets[(p, 0)]
                                   + stage.volume], dtype=np.float32)
        assert np.array_equal(piece, _expected_piece(stage, 0, stage.depth))
        values = np.asarray(stage.tensor.get_values())
        (r0, _), (d0, _) = stage.window
        assert piece[stage.canonical_offset(3, 5)] == tf32_halves(
            np.array([values[stage.n0 + 3, d0 + 5]], dtype=np.float32))[0][0]


def test_the_table_is_named_after_its_numbers():
    """Two plans with the same B share the table's name -- a file's kernels
    define it once -- and another cut of the ring is another table."""
    first, _ = _group('volume', 8, 'sm_100a')
    again, _ = _group('volume', 8, 'sm_100a')
    other, _ = _group('volume', 8, 'sm_100a', matrix_ring_slots=2,
                      matrix_ring_rows=32)
    name = eg.prepared_table(first.section, _step(first), 100)[0]
    assert name == eg.prepared_table(again.section, _step(again), 100)[0]
    assert name != eg.prepared_table(other.section, _step(other), 100)[0]


# -- the kernel ----------------------------------------------------------------- #

@pytest.mark.parametrize('arch,commits', [('sm_90a', 0), ('sm_100a', 18)])
def test_the_kernel_copies_b_and_converts_none_of_it(arch, commits):
    """At o8: a bulk copy a chunk from the table -- the first three at the
    trip's head -- tcgen05's products of a chunk committed onto its slot,
    the table defined on 128 bytes, the kernel's comment naming it, and no
    load of the operator in the trip."""
    gen = tma._kernel(arch, 'volume', 8, 's', **PREPARED)
    src = gen.get_kernel()
    assert src.count('tensorforge::bulkCopyOn(') == 18
    assert src.count('tensorforge::tcgen05CommitOn(') == commits
    (name,) = set(re.findall(r'\btf_bk_[0-9a-f]{16}\b', src))
    assert re.search(rf'__device__ __align__\(128\) const float {name}'
                     rf'\[{3 * 88 * (120 if arch == "sm_90a" else 128) * 2}'
                     rf'\]', src)
    assert re.search(rf'// operator ring: B of kDivM\(0\), kDivM\(1\), '
                     rf'kDivM\(2\) into Q prepared from the description\'s '
                     rf'numbers in {name}', src)
    converting = tma._kernel(arch, 'volume', 8, 's', matrix_engine='auto')
    text = converting.get_kernel()
    # The converting ring splits B where the prepared one copies it.
    assert src.count('splitFloatTF32') < text.count('splitFloatTF32')
    assert 'bulkCopyOn' not in text and 'tf_bk_' not in text


def test_a_resident_stage_is_one_bulk_copy_a_product():
    src = tma._kernel('sm_100a', 'volume', 6, 's', **PREPARED).get_kernel()
    assert src.count('tensorforge::bulk_copy(') == 3
    assert 'bulkCopyOn' not in src


# -- the PIR ------------------------------------------------------------------- #

def test_the_commit_onto_a_rings_slot_is_its_empty_side():
    """tcgen05's commit onto a ring's slot: the slot and the issuing thread,
    on the ring's empty barriers, lowered to `tcgen05CommitOn`; refused
    without its side, and onto no ring."""
    from tensorforge.backend.pir import engines
    from tensorforge.backend.pir.build import IRBuilder
    from tensorforge.backend.pir.core import BOOL, INDEX, IRError
    from tensorforge.common.context import Context
    from tensorforge.common.options import Options
    ctx = Context(arch='sm_100a', backend='cuda', fp_type=Datatype.F32,
                  options=Options())
    b = IRBuilder(Datatype.F32, ctx, arena='shr')
    ring = b.mbarrier_ring('bk', 4, full=1, empty=1,
                           leader='threadIdx.x == 0')
    slot = b.extern_value('slot', INDEX)
    issue = b.extern_value('issue', BOOL)
    s = b.commit_group('tcgen05', object=ring, issue=issue, slot=slot)
    assert s.attr('side') == 'empty' and engines.on_ring(s)
    assert engines.verify_engine(s) == []
    assert engines.lower(s, ctx.target, name=str).lines == (
        f'tensorforge::tcgen05CommitOn(&{ring.object}[4 + ({slot})], '
        f'{issue});',)
    sideless = dataclasses.replace(s, attrs=tuple(
        (k, v) for k, v in s.attrs if k != 'side'))
    assert any('empty side' in d for d in engines.verify_engine(sideless))
    with pytest.raises(IRError):
        b.commit_group('tcgen05', object=ring.object, issue=issue,
                       slot=slot)


# -- the oracle ------------------------------------------------------------------ #

def _prepared_two_trips(shape, arch, monkeypatch, streamed, **options):
    """`test_multi_element_async._two_trips` with B prepared: o4, a group of
    four over seven elements, streamed through a ring of eight-row pieces
    where `streamed`."""
    asked = {'matrix_ring': 'prepared', 'matrix_ring_rows': 8, **options}
    monkeypatch.setattr(tma, 'AUTO', {**tma.AUTO, **asked})
    tma._two_trips(shape, arch, monkeypatch, streamed=streamed, **asked)
    # The kernel that ran is the prepared one: B copied, a chunk a copy
    # into the ring, or once a stage.
    src = tma._kernel(arch, shape, 4, 's', **asked).get_kernel()
    assert 'tf_bk_' in src
    assert ('tensorforge::bulkCopyOn(' in src) is streamed
    assert ('tensorforge::bulk_copy(' in src) is not streamed


@pytest.mark.parametrize('arch', ['sm_90a', 'sm_100a'])
@pytest.mark.parametrize('shape,streamed', [
    ('volume', True), ('volume', False), ('chain', False)],
    ids=['volume-ring', 'volume-resident', 'chain-resident'])
def test_every_element_of_two_trips_on_the_oracle(shape, streamed, arch,
                                                  monkeypatch):
    """Against NumPy, every element of two trips, and no access to shared
    memory that no barrier, ring or wait orders (`kernel_eval.Races`): the
    products read each chunk as it landed and before the next round wrote
    its slot -- a write ahead of that is a race, or a product's operand
    changed in flight -- and a wait for a phase nobody completes is refused
    as the hang it would be."""
    _prepared_two_trips(shape, arch, monkeypatch, streamed)


@pytest.mark.parametrize('arch', ['sm_90a', 'sm_100a'])
def test_every_element_with_own_barriers_on_the_oracle(arch, monkeypatch):
    """Without the barrier placement: the ring's barriers and the
    lowering's own are all there is."""
    _prepared_two_trips('volume', arch, monkeypatch, True,
                        enable_sync_block_opt=False)


@pytest.mark.parametrize('arch', ['sm_90a', 'sm_100a'])
def test_a_slot_is_read_only_where_its_chunk_landed(arch, monkeypatch):
    """Shared memory nothing wrote reads NaN: the products read nothing of
    a slot before its chunk landed, and the elements still agree with
    NumPy."""
    from tensorforge.reference import kernel_eval
    read = kernel_eval.Slot.read
    unwritten = set()

    def poisoned(self, base, index):
        if (base == kernel_eval.Races.SHARED
                and (base, int(index)) not in self.data):
            unwritten.add(int(index))
            return math.nan
        return read(self, base, index)
    monkeypatch.setattr(kernel_eval.Slot, 'read', poisoned)
    _prepared_two_trips('volume', arch, monkeypatch, True)


@pytest.mark.parametrize('arch', ['sm_90a', 'sm_100a'])
def test_the_ring_over_several_rounds_of_trips_on_the_oracle(arch,
                                                             monkeypatch):
    """Two slots, so that every trip's chunks go round the ring several
    times and the second trip starts on a round of the first's."""
    _prepared_two_trips('volume', arch, monkeypatch, True,
                        matrix_ring_slots=2)


def _operators_in_memory_otherwise(monkeypatch):
    """The operators' memory -- every batch-constant tensor the oracle fills
    from its description (`test_multi_element_async._two_trips`) -- other
    numbers than the description's, at every cell, its known zeros too; the
    reference still the description's matrix: the names of the tensors
    changed."""
    from tensorforge.reference import kernel_eval
    evaluate = kernel_eval.evaluate_wave
    changed = []

    def evaluate_otherwise(*args, preset=None, **kwargs):
        preset = {name: [1.0 + 2.0 * v + 0.125 * (i % 5)
                         for i, v in enumerate(values)]
                  for name, values in (preset or {}).items()}
        changed.extend(preset)
        return evaluate(*args, preset=preset, **kwargs)
    monkeypatch.setattr(kernel_eval, 'evaluate_wave', evaluate_otherwise)
    return changed


@pytest.mark.parametrize('arch', ['sm_90a', 'sm_100a'])
@pytest.mark.parametrize('shape,streamed', [
    ('volume', True), ('volume', False), ('chain', False)],
    ids=['volume-ring', 'volume-resident', 'chain-resident'])
def test_the_prepared_kernel_multiplies_the_descriptions_numbers(
        shape, streamed, arch, monkeypatch):
    """The contract of the description's numbers: B prepared from them at
    generation, the operator's memory does not change the result -- other
    numbers there, and every element of two trips still agrees with NumPy
    over the description's matrix.  A caller passing other numbers gets the
    description's product (the GPU cases fill the operators with their
    numbers, `test_multi_element._described`)."""
    changed = _operators_in_memory_otherwise(monkeypatch)
    _prepared_two_trips(shape, arch, monkeypatch, streamed)
    assert changed


@pytest.mark.parametrize('arch', ['sm_90a', 'sm_100a'])
def test_the_converting_kernel_multiplies_the_operators_memory(arch,
                                                               monkeypatch):
    """The same other numbers under the converting ring, which reads B from
    memory: the elements no longer agree with the description's matrix --
    the contract test above would see a kernel that reads them."""
    _operators_in_memory_otherwise(monkeypatch)
    with pytest.raises(AssertionError, match='^\\(0, '):
        tma._two_trips('volume', arch, monkeypatch, streamed=False)


# -- the build ------------------------------------------------------------------- #

@pytest.mark.parametrize('arch,engine', [('sm_90a', 'HGMMA'),
                                         ('sm_100a', 'UTCHMMA')])
def test_the_build_copies_b_by_bulk_copies(arch, engine, tmp_path):
    """Built for the arch-specific target at o8: a bulk copy a chunk
    (UBLKCP), the products as before, and no load of the operator."""
    sass = tma._sass(arch, tmp_path, 'volume', 8, matrix_ring='prepared')
    assert tma._count(sass, 'UBLKCP') == 18
    assert tma._count(sass, engine) == 99
    (tmp_path / 'plain').mkdir()
    plain = tma._sass(arch, tmp_path / 'plain', 'volume', 8)
    assert tma._count(sass, 'LDG') < tma._count(plain, 'LDG')


# -- on the hardware ------------------------------------------------------------- #

GPU_CASES = [('volume', 8), ('volume', 6), ('chain', 6)]


@pytest.mark.parametrize('shape,order', GPU_CASES,
                         ids=[f'{s}-o{o}' for s, o in GPU_CASES])
def test_the_prepared_ring_computes_on_the_gpu(shape, order, target,
                                               cache_root,
                                               tensorforge_include):
    """On a GH200 (wgmma) or a GB200 (tcgen05), B prepared: streamed at o8,
    copied once at o6 -- against NumPy over a batch ending in a partial
    group, several trips a block.  The operators are their description's
    numbers in memory too (`test_multi_element._described`): the prepared
    kernel multiplies those whatever memory holds, the reference what the
    harness hands it -- each element within 1e-5 of its largest cell
    (`test_multi_element._normwise`: the real operators' sums cancel)."""
    if target.backend != 'cuda' or target.arch not in ('sm_90', 'sm_100'):
        pytest.skip('no GPU with an asynchronous matrix engine')
    from harness.runner import run_case
    special = dataclasses.replace(target, arch=target.arch + 'a')
    tma._nvcc_builds(special.arch)
    group = eg.plan(me._descrs(shape, order, 's'), me._context(
        special.arch, 's', multi_element='force', **PREPARED))
    assert group is not None and eg.prepared(group.section)
    case = types.SimpleNamespace(
        NAME=f'operator_ring_{shape}_o{order}s',
        DTYPE=Datatype.F32, BATCH=4099, TOL=me._normwise(),
        OPTIONS={'multi_element': 'force', **PREPARED},
        INPUT_TRANSFORM=me._described(me._descrs(shape, order, 's')),
        descr_list=lambda: me._descrs(shape, order, 's'),
        reference=me._reference(me._descrs(shape, order, 's')))
    result = run_case(case, special, cache_root, tensorforge_include)
    assert result.passed, (result.max_abs_err, result.max_rel_err,
                           result.detail)
