# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Derived forms of an operator (`common.matrix.forms`): the parts an
emulated product multiplies, made where the numbers are known.

* Each form is the arithmetic its path had, bit for bit: `split_tf32`'s
  pair, `tf32_halves`' pair, `split_bf16`'s triple.
* Each is what the device makes at run time, as the host models hold it:
  the oracle's TF32 rounding (`kernel_eval`), the stacked scheme's
  simulator's BF16 split (`test_amd_stacked`).
* The tables: `Tensor.storage_values` and the asynchronous engines'
  prepared B (`element_group.prepared_table`) through the forms are the
  numbers they were; a table of another form -- a BF16 triple through the
  canonical layout -- gives every cell back.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorforge.backend.pir import engines
from tensorforge.common.basic_types import Datatype
from tensorforge.common.matrix import forms
from tensorforge.common.matrix.prepare import (split_bf16, split_tf32,
                                               tf32_halves)


def _values(n=4096, seed=7):
    """Values of every magnitude a SeisSol operator has and then some,
    signs and exact ties included."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n) * 2.0 ** rng.integers(-30, 30, n)
    # Halfway between two TF32 neighbours, and between two BF16 ones: the
    # ties the two roundings tell apart.
    ties = np.array([1.0 + 2.0 ** -11, 1.0 + 3 * 2.0 ** -11, -(1.0 + 2.0 ** -11),
                     1.0 + 2.0 ** -8, 1.0 + 3 * 2.0 ** -8, 0.0, -0.0])
    return np.concatenate([x, ties]).astype(np.float32)


# -- each form is the arithmetic it was -------------------------------------- #

def test_the_stored_tf32_pair_is_split_tf32():
    x = _values()
    assert np.array_equal(forms.TF32_STORED.derive(x),
                          split_tf32(x).reshape(-1, 2).T)


@pytest.mark.parametrize('ties', ['even', 'away'])
def test_the_device_tf32_pair_is_tf32_halves(ties):
    x = _values()
    upper, lower = tf32_halves(x, ties)
    got = forms.tf32_device(ties).derive(x)
    assert np.array_equal(got[0], upper) and np.array_equal(got[1], lower)


def test_the_bf16_triple_is_split_bf16():
    x = _values()
    assert np.array_equal(forms.BF16_TRIPLE.derive(x),
                          split_bf16(x).reshape(-1, 3).T)


@pytest.mark.parametrize('form', [forms.TF32_STORED, forms.BF16_TRIPLE],
                         ids=['tf32', 'bf16'])
@pytest.mark.parametrize('planar', [0, 4, 64])
def test_stored_adjacent_or_planar_as_storage_values_stored_them(form,
                                                                 planar):
    x = _values(n=4096 - 7)
    old = split_tf32 if form is forms.TF32_STORED else split_bf16
    assert np.array_equal(form.store(x, planar=planar),
                          old(x, planar=planar))


def test_the_parts_count_names_the_form():
    assert forms.stored(2) is forms.TF32_STORED
    assert forms.stored(3) is forms.BF16_TRIPLE
    assert forms.stored(1) is None and forms.stored(4) is None
    assert forms.tf32_device_for(90) == forms.tf32_device('even')
    assert forms.tf32_device_for(80) == forms.tf32_device('away')


# -- what the parts are ------------------------------------------------------ #

def _exact_in(values, narrow):
    """Whether every value has no more significand bits than `narrow`."""
    bits = {Datatype.TF32: 13, Datatype.BF16: 16}[narrow]
    u = np.asarray(values, dtype=np.float32).view(np.uint32)
    return bool(((u & ((1 << bits) - 1)) == 0).all())


def test_the_bf16_terms_are_bf16_and_sum_to_the_value():
    x = _values()
    parts = forms.BF16_TRIPLE.derive(x)
    for p in parts:
        assert _exact_in(p, Datatype.BF16)
    normal = np.abs(x) > 2.0 ** -100
    assert np.array_equal(forms.BF16_TRIPLE.combine(parts)[normal],
                          x.astype(np.float64)[normal])


@pytest.mark.parametrize('ties', ['even', 'away'])
def test_the_device_pair_is_the_value_exactly(ties):
    """The upper half TF32, the lower the exact remainder."""
    x = _values()
    upper, lower = forms.tf32_device(ties).derive(x)
    assert _exact_in(upper, Datatype.TF32)
    assert np.array_equal(upper.astype(np.float64) + lower, x)


def test_the_stored_pair_is_both_halves_tf32_and_as_accurate():
    x = _values()
    hi, lo = forms.TF32_STORED.derive(x)
    assert _exact_in(hi, Datatype.TF32) and _exact_in(lo, Datatype.TF32)
    err = np.abs(forms.TF32_STORED.combine([hi, lo]) - x)
    assert (err <= 2.0 ** -21 * np.abs(x)).all()


# -- each is what the device makes, as the host models hold it -------------- #

def test_the_oracle_rounds_the_upper_half_as_the_form_does():
    """`kernel_eval._round_tf32` is `cvt.rna`; `_tf32` reads the upper
    nineteen bits of an operand, which the exact remainder keeps whole
    where the instruction reads it."""
    from tensorforge.reference import kernel_eval as ke
    x = _values(n=512)
    upper, lower = forms.tf32_device('away').derive(x)
    assert [ke._round_tf32(float(v)) for v in x] == [float(u) for u in upper]
    hi, _ = forms.TF32_STORED.derive(x)
    assert [ke._round_tf32(float(v)) for v in x] == [float(h) for h in hi]


def test_the_stacked_simulator_splits_as_the_form_does():
    """The simulator of the stacked scheme (`test_amd_stacked`) models
    `splitFloatx4BF16`, and the host-split operator it reads is
    `prepare.split_bf16`'s: the same three terms."""
    import test_amd_stacked as st
    x = _values(n=512)
    got = forms.BF16_TRIPLE.derive(x)
    for want, have in zip(st._split_bf16(x), got):
        assert np.array_equal(np.asarray(want, dtype=np.float32), have)


# -- tables --------------------------------------------------------------------- #

@pytest.mark.parametrize('rows,width', [(4, 8), (32, 16), (88, 120), (8, 64)])
def test_the_canonical_layout_is_the_descriptors(rows, width):
    """A cell's slot is where a descriptor of the piece reads it: the
    leading byte offset 128 between core matrices along K, the stride
    `rows` x 32 bytes along N (`_Piece.descriptors`)."""
    slots = forms.canonical_k_major(rows, width)
    assert sorted(slots) == list(range(rows * width))
    desc = engines.MatrixDescType(Datatype.TF32, lbo=128, sbo=rows * 32)
    for k in range(rows):
        for n in range(width):
            assert 4 * slots[k * width + n] == engines.descriptor_offset(
                desc, n, k)


@pytest.mark.parametrize('form', [forms.tf32_device('even'),
                                  forms.BF16_TRIPLE, forms.TF32_STORED],
                         ids=['tf32', 'bf16', 'stored'])
def test_a_table_of_any_form_gives_every_cell_back(form):
    rows, width = 16, 24
    rng = np.random.default_rng(3)
    block = rng.uniform(-2, 2, (rows, width)).astype(np.float32)
    slots = forms.canonical_k_major(rows, width)
    laid = forms.lay(form, block, slots)
    assert laid.size == form.parts * rows * width
    size = rows * width
    parts = [laid[p * size + slots] for p in range(form.parts)]
    assert np.array_equal(np.stack(parts), form.derive(block))
    if form is not forms.TF32_STORED:
        assert np.array_equal(form.combine(parts), block.ravel())


def _old_prepared_table(section, step, sm):
    """`element_group.prepared_table` as the engines' item 1 wrote it, before
    the forms: the reference the table is held to."""
    from tensorforge.backend.instructions.compute.multilinear import \
        _module_table_name
    from tensorforge.generators import element_group as eg
    stages = eg._step_stages(section, step)
    if stages[0].streamed:
        pieces = [piece for chunk in eg.ring_chunks(section, step)
                  for piece in chunk]
    else:
        pieces = [(p, 0, st.depth) for p, st in enumerate(stages)]
    parts, offsets, at = [], {}, 0
    for p, k0, rows in pieces:
        stage = stages[p]
        values = np.asarray(stage.tensor.get_values(), dtype=np.float64)
        (r0, r1), (d0, d1) = stage.window
        block = np.zeros((rows, stage.width))
        for k in range(rows):
            col = d0 + k0 + k
            if col >= d1:
                continue
            for n in range(stage.width):
                row = stage.n0 + n
                if r0 <= row < r1:
                    block[k, n] = values[row, col]
        upper, lower = tf32_halves(block.ravel(),
                                   'even' if sm >= 90 else 'away')
        laid = np.zeros(2 * rows * stage.width, dtype=np.float32)
        for k in range(rows):
            for n in range(stage.width):
                slot = (n // 8) * rows * 8 + (k // 4) * 32 + (n % 8) * 4 + k % 4
                laid[slot] = upper[k * stage.width + n]
                laid[rows * stage.width + slot] = lower[k * stage.width + n]
        offsets[(p, k0)] = at
        at += laid.size
        parts.append(laid)
    numbers = tuple(float(v) for v in np.concatenate(parts))
    return _module_table_name('bk', Datatype.F32, numbers), numbers, offsets


@pytest.mark.parametrize('arch,order,shape', [
    ('sm_100a', 6, 'volume'), ('sm_100a', 8, 'volume'),
    ('sm_90a', 6, 'volume'), ('sm_90a', 8, 'volume'),
    ('sm_100a', 4, 'volume')])
def test_the_prepared_tables_are_the_numbers_they_were(arch, order, shape):
    """The asynchronous engines' prepared B (`matrix_ring=prepared`), its
    pieces streamed or resident, through the forms: the table the engines'
    item 1 built, name, numbers and offsets."""
    import test_multi_element as me
    from tensorforge.backend.instructions.compute.primitives import nvidia
    from tensorforge.generators import element_group as eg
    context = me._context(arch, 's', multi_element='force',
                          matrix_engine='auto', matrix_ring='prepared')
    section, why = eg.classify(me._descrs(shape, order, 's'))
    assert section is not None, why
    group, why = eg.fill(section, context)
    assert group is not None, why
    sm = nvidia.sm_of(context)
    tables = 0
    for step in group.section.steps:
        if step.lowering not in eg.ASYNC_LOWERINGS:
            continue
        if not eg._step_stages(group.section, step)[0].prepared:
            continue
        assert eg.prepared_table(group.section, step, sm) == \
            _old_prepared_table(group.section, step, sm)
        tables += 1
    assert tables


def test_storage_values_are_the_numbers_they_were():
    """An operator stored in parts, adjacent and planar, as the offering
    and the operator copies store it (`Tensor.storage_values`)."""
    from tensorforge.common.basic_types import Addressing
    from tensorforge.common.matrix.tensor import Tensor
    rng = np.random.default_rng(5)
    data = rng.uniform(-1, 1, (12, 10)).astype(np.float64)
    for parts in (2, 3):
        for planar in (False, True):
            t = Tensor([12, 10], Addressing.NONE, alias='op', data=data,
                       datatype=Datatype.F32)
            t.storage_parts = parts
            t.storage_planar = planar
            flat = data.ravel(order='F').astype(np.float32)
            old = (split_tf32 if parts == 2 else split_bf16)(
                flat, planar=flat.size if planar else 0)
            assert t.storage_values() == tuple(float(v) for v in old)
