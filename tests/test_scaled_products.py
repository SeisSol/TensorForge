# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Scaled products in the multi-element layout (`element_group.Product.
scalar`): yateto's alpha, a product times a scalar -- `dest (+)= s first x
second` -- as the frontend hands it on, one more operand over no axis.  The
lanes form and its slot form scale a product by its factor
(`compute.multi_element._LaneTrip`); the warp form, the matrix tilings and
the high-order tile refuse it so far.

Checked numerically on the host oracle against NumPy in double precision,
with factors 0.37, 2.5 and -1.3 a product, never 1 and never the same for
two products in a row, so that a factor dropped, doubled or given to
another product shows: once as numbers the description holds, once as
scalars the kernel is handed -- the case that matters to SeisSol's Taylor
scaling, whose dt / (j + 1) differs from one time cluster to another -- the
derivative's every order scaled, the chain's middle orders included, whose
outputs the later orders read back."""

from __future__ import annotations

import json
import types
import warnings

import numpy as np
import pytest

import test_multi_element as me
from tensorforge.common.basic_types import Addressing
from tensorforge.generators import element_group as eg

#: The factors the products take in turn.
FACTORS = (0.37, 2.5, -1.3)

#: The slot form asked for, and the lanes form.
SLOTS = {'multi_element': 'force', 'multi_element_fma': 'slots'}
LANES = {'multi_element': 'force', 'multi_element_fma': 'lanes'}


def _scaled_description(shape, order, precision, by_value):
    """The description of `shape` with every product's alpha a factor of
    `FACTORS` in turn -- yateto names one for every operation, 1 where it
    sends none -- a number the description holds, or (`by_value`) a scalar
    the kernel is handed, as the derivative's `power(j)` are.  The Taylor
    sum's scaled copies keep their `power(j)`."""
    d = json.loads(json.dumps(me.DESCRIPTIONS[shape](order, precision)))
    named = {t['name']: t for t in d['tensors']}
    n = 0
    for op in d['operations']:
        if op.get('type') != 'multilinear' or len(op['args']) != 2:
            continue
        alpha = op['linear']['alpha']['name']
        if not alpha.startswith('_scalar'):
            continue
        tensor = named[alpha]
        tensor['values'] = (None if by_value else
                            {'kind': 'entries',
                             'data': [[[], FACTORS[n % len(FACTORS)]]]})
        n += 1
    return d


def _descrs(shape, order, precision, by_value):
    from tensorforge.frontend.yateto import DescriptionReader
    return DescriptionReader(None, {}).read(
        _scaled_description(shape, order, precision, by_value))[0]


def _factors(section):
    """The scaled products' factors passed by value, by alias: `FACTORS`
    in the products' order."""
    out = {}
    for p in section.products:
        if p.scalar is not None and not p.scalar.tensor.has_values():
            out.setdefault(p.scalar.tensor.alias,
                           FACTORS[len(out) % len(FACTORS)])
    return out


# -- the plan ------------------------------------------------------------------- #

@pytest.mark.parametrize('by_value', [False, True],
                         ids=['numbers', 'by-value'])
def test_a_scaled_product_classifies(by_value):
    """The derivative at o4 with every product scaled: each product carries
    its factor -- the description's number, or a scalar the kernel is
    handed, which the section's scalars name besides the Taylor sum's
    `power(j)` -- and the section is the chained one it was."""
    section, why = eg.classify(_descrs('derivative', 4, 's', by_value))
    assert section is not None, why
    products = [p for p in section.products if p.kind is not eg.Kind.SCALE]
    assert len(products) == 18
    assert all(p.scalar is not None for p in products)
    numbers = [float(np.asarray(p.scalar.tensor.get_values()).reshape(-1)[0])
               for p in products if p.scalar.tensor.has_values()]
    # In the section's order of its steps, which is not the description's.
    assert sorted(numbers) == ([] if by_value else
                               sorted(FACTORS[n % 3] for n in range(18)))
    handed = [t.alias for t in section.scalars if not t.has_values()]
    assert {f'power({j})' for j in range(4)} <= set(handed)
    assert (len(handed) == 4 + 18) is by_value
    assert all(p.scalar.tensor.addressing is Addressing.SCALAR
               for p in products if not p.scalar.tensor.has_values())
    assert section.chained
    assert eg.scaled_product(section).startswith('_tmp')


def test_a_product_of_the_two_alone_has_no_factor():
    """yateto's alpha of one is no operand: the derivative as recorded, its
    products of the two alone, the scaled copies' factors where they
    were."""
    section, _ = eg.classify(me._descrs('derivative', 4, 's'))
    assert all(p.scalar is None for p in section.products)
    assert eg.scaled_product(section) is None


@pytest.mark.parametrize('by_value', [False, True],
                         ids=['numbers', 'by-value'])
def test_the_other_forms_refuse_a_scaled_product(by_value):
    """The volume at o4 scaled: the warp form and the matrix tilings refuse
    it (`fill`), and so does the high-order tile, the kernel's comment
    saying why (`tile_refusal`); the lanes form and its slot form take
    it."""
    descrs = _descrs('volume', 4, 's', by_value)
    section, _ = eg.classify(descrs)
    context = me._context('sm_100a', 's', multi_element='force')
    group, why = eg.fill(section, context)
    assert group is None
    assert why.endswith('a scaled product, is the lanes form\'s and its '
                        'slot form\'s alone so far')
    tile = me._context('sm_100a', 's', multi_element='force',
                       matrix_engine='auto', matrix_tile='high')
    assert eg.tile_refusal(section, tile).endswith(
        'a scaled product, which the tile does not take yet')
    assert eg.plan(descrs, tile) is None
    for options in (SLOTS, LANES):
        group = eg.plan(descrs, me._context('sm_100a', 's', **options))
        assert group is not None and group.lanes is not None


# -- on the oracle -------------------------------------------------------------- #

#: The sections and forms on the oracle: the derivative, a chained section
#: whose every order is scaled, at o4 in both precisions and at o6; the
#: volume at o4 and o6 in both; localFlux, whose second operator product
#: reads the first's output from registers (`_LaneTrip.walk`'s held
#: column); and the lanes form's volume.  On sm_90, and the derivative at
#: o4 on sm_100 too: the slot form's code is the same, the oracle runs the
#: CUDA kernel.
ORACLE = [('derivative', 4, 's', 'sm_90', SLOTS),
          ('derivative', 4, 'd', 'sm_90', SLOTS),
          ('derivative', 6, 's', 'sm_90', SLOTS),
          ('derivative', 4, 's', 'sm_100', SLOTS),
          ('volume', 4, 's', 'sm_90', SLOTS),
          ('volume', 6, 's', 'sm_90', SLOTS),
          ('volume', 4, 'd', 'sm_90', SLOTS),
          ('volume', 6, 'd', 'sm_90', SLOTS),
          ('localFlux', 4, 's', 'sm_90', SLOTS),
          ('volume', 4, 's', 'sm_90', LANES)]


@pytest.mark.parametrize('by_value', [False, True],
                         ids=['numbers', 'by-value'])
@pytest.mark.parametrize(
    'shape,order,precision,arch,options', ORACLE,
    ids=[f'{s}-o{o}{p}-{a}-{o2["multi_element_fma"]}'
         for s, o, p, a, o2 in ORACLE])
def test_scaled_products_on_the_oracle(shape, order, precision, arch,
                                       options, by_value):
    """Every product scaled, on the host oracle over a group and seven
    elements more against NumPy: each output -- a chained section's every
    one -- within 1e-12 of its largest entry (the oracle's arithmetic is
    the host's double), the factors the description's numbers or handed to
    the kernel, 0.37, 2.5 and -1.3 in turn."""
    descrs = _descrs(shape, order, precision, by_value)
    section, why = eg.classify(descrs)
    assert section is not None, why
    gen = me._generate(descrs, me._context(arch, precision, **options))
    group = gen._element_group
    assert group is not None and group.lanes is not None
    assert (group.lanes.mapping is not None) is (options is SLOTS)
    count = group.elements + 7
    values = _factors(section)
    assert bool(values) is by_value
    tensors, inputs, scalars, after = me._on_the_oracle(
        gen, count, scalar_values=values)
    assert all(scalars[alias] == value for alias, value in values.items())
    if section.chained:
        want = me._reference_all(descrs)(inputs, scalars)
    else:
        output = next(t for t in tensors.values() if not t.is_tmp
                      and t.addressing.kind is not Addressing.NONE
                      and any(d.writes().tensor is t
                              for outer in gen.descr_list
                              for d in outer.operations()))
        want = {output.alias: me._reference(descrs)(
            inputs, inputs[output.alias], scalars)}
    for alias, expected in want.items():
        got = after(tensors[alias])
        for e in range(count):
            scale = np.abs(expected[e]).max() or 1.0
            error = np.abs(got[e] - expected[e]).max() / scale
            assert error <= 1e-12, (alias, e, error)


def test_the_factors_are_parameters_where_the_kernel_is_handed_them():
    """Handed to the kernel, the derivative's factors are its parameters by
    value: the scaled products' eighteen beside the Taylor sum's four
    `power(j)`."""
    def handed(by_value):
        descrs = _descrs('derivative', 4, 's', by_value)
        gen = me._generate(descrs, me._context('sm_90', 's', **SLOTS))
        src = gen.get_kernel()
        head = src[:src.index('{')]
        return len([n for n in head.split(',')
                    if 'const float m' in n and '*' not in n])
    assert handed(True) == handed(False) + 18


# -- on the GPU ----------------------------------------------------------------- #

def _derivative_case(order, precision, output, by_value):
    """The harness's case of the scaled derivative, compared at `output`:
    the products' factors the description's numbers, or handed to the
    kernel as `SCALARS` names them; the Taylor sum's `power(j)` the
    driver's 0.5 (`test_multi_element._derivative_case`)."""
    descrs = _descrs('derivative', order, precision, by_value)
    section, _ = eg.classify(descrs)
    values = _factors(section)
    outputs, scalars = {}, dict(values)
    for descr in descrs:
        for op in descr.operations():
            scaled = eg._scaled(op)
            if scaled is not None and scaled[1] is not None:
                scalars.setdefault(scaled[1].tensor.alias, 0.5)
            if not op.dest.tensor.is_tmp:
                outputs.setdefault(op.dest.tensor.alias, op.dest.tensor)
    every = me._reference_all(descrs)

    def reference(inputs, dest_in):
        given = dict(inputs)
        for alias, t in outputs.items():
            if alias not in given:
                given[alias] = np.zeros((len(dest_in),) + tuple(
                    int(hi) - int(lo) for lo, hi in zip(t.bbox.lower(),
                                                        t.bbox.upper())))
        return every(given, scalars)[output]

    def tolerance(expected):
        rows = np.abs(expected).max(axis=(0, 2), keepdims=True)
        return 0.0, rows * (1e-13 if precision == 'd' else 1e-5)

    return types.SimpleNamespace(
        NAME=(f'scaled_products_derivative_o{order}{precision}_'
              f'{"values" if by_value else "numbers"}'),
        DTYPE=me._dtype(precision), BATCH=4099, OUTPUT=output, TOL=tolerance,
        OPTIONS=dict(SLOTS), INPUT_TRANSFORM=me._described(descrs),
        SCALARS=values,
        descr_list=lambda: _descrs('derivative', order, precision, by_value),
        reference=reference)


@pytest.mark.parametrize('by_value', [False, True],
                         ids=['numbers', 'by-value'])
@pytest.mark.parametrize('order,precision', [(4, 's'), (4, 'd')])
def test_the_scaled_derivative_computes_on_the_gpu(
        order, precision, by_value, target, cache_root,
        tensorforge_include):
    """The scaled derivative in the slot form built and run on the local
    GPU against NumPy over a batch whose last group is partial: the time
    integral, which sums every order, and the last order's derivative --
    the factors handed to the kernel at the values `SCALARS` names, or the
    description's numbers."""
    from harness.runner import run_case
    if (target.backend, target.vendor) not in eg.SLOT_TARGETS:
        pytest.skip(f'the slot form is not written for {target.backend} on '
                    f'{target.vendor}')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for output in ('I', f'dQ({order - 1})'):
            result = run_case(_derivative_case(order, precision, output,
                                               by_value),
                              target, cache_root, tensorforge_include)
            assert result.passed, (output, result.max_abs_err,
                                   result.max_rel_err, result.detail)
