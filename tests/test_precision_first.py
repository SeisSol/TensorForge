# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""F32's accuracy first in the multi-element layout's plans: an F32 plan
never has an asynchronous engine (wgmma, tcgen05) read an operand in TF32
alone -- B staged whole, A's two halves by B's one, two of 3xTF32's three
products -- to keep a larger group.  The fill rule takes B's halves at each
size of the group (resident, the prepared ring where asked, the converting
ring), then mma_sync, which splits B where it reads it, then the FMA forms;
mma_sync over the asynchronous engine where it holds the larger group, up to
the engine's lane limit (`element_group.prefer_mma_sync`, the cost model's to
replace).  A plan-level check (`element_group.tf32_refusal`) refuses any
other plan, in the fill and again in the generator, unless
`Options.matrix_tf32_operands` lets it trade -- and the kernel's comment
says which engine the plan took and why.

What is checked here: the four plans that read B in TF32 alone before
(sm_100a chain o7 and o8, localFluxAll o6; sm_90a neighboringFlux o8) now
keep both halves, on mma_sync; localFluxAll o8 was precise already; the
rule's other takes (mma_sync where it holds the larger group); the check
refusing a TF32-only plan made by hand, in the fill's terms and in the
generator; the option restoring the trade, with its note; the comparison
itself; and the volume's plans as measured, unchanged.
"""

from __future__ import annotations

import dataclasses

import pytest

import test_multi_element as me
from tensorforge.backend.instructions.compute.primitives import nvidia
from tensorforge.common.exceptions import InternalError
from tensorforge.generators import element_group as eg


def _plan(arch, shape, order, **options):
    context = me._context(arch, 's', multi_element='force',
                          matrix_engine='auto', **options)
    section, why = eg.classify(me._descrs(shape, order, 's'))
    assert section is not None, why
    group, why = eg.fill(section, context)
    assert group is not None, why
    return group, context, section


def _engines(section):
    return sorted({s.lowering for s in section.steps
                   if s.lowering in eg.MATRIX_LOWERINGS})


# -- the plans that traded -------------------------------------------------------- #

@pytest.mark.parametrize('arch,shape,order,elements,split,before', [
    ('sm_100a', 'chain', 7, 14, True, 'tcgen05 at 14'),
    ('sm_100a', 'chain', 8, 14, False, 'tcgen05 at 7'),
    ('sm_100a', 'localFluxAll', 6, 8, True, 'tcgen05 at 8'),
    ('sm_90a', 'neighboringFlux', 8, 14, True, 'wgmma at 12')])
def test_the_plans_that_read_b_in_tf32_keep_both_halves(arch, shape, order,
                                                        elements, split,
                                                        before):
    """They read B whole on the asynchronous engine (`before`); with B's
    halves it holds half the group or less, mma_sync -- B in its halves, or
    whole and split where it reads it -- at least the same: mma_sync, and
    the check passes.  The kernel's comment says why."""
    group, context, _ = _plan(arch, shape, order)
    section = group.section
    assert _engines(section) == ['mma_sync'], before
    assert group.elements == elements and eg.split(section) is split
    assert eg.tf32_refusal(section, context) is None
    (note,) = eg.precision_notes(section, context)
    assert ' declined: with every operand in both TF32 halves it holds a ' \
        'group of ' in note
    assert note.endswith('(Options.matrix_tf32_operands)')
    src = me._generate(me._descrs(shape, order, 's'), context).get_kernel()
    assert '// matrix engine: ' + note in src


@pytest.mark.parametrize('ring', ['convert', 'prepared'])
@pytest.mark.parametrize('star', ['buffer', 'halves', 'engine'])
@pytest.mark.parametrize('arch,shape,order', [
    ('sm_90a', 'neighboringFlux', 8), ('sm_100a', 'localFluxAll', 6)])
def test_the_options_of_the_ring_and_the_star_trade_nothing(arch, shape,
                                                            order, star, ring):
    """The same under the options the operator ring and A's halves add
    (`matrix_ring`, `matrix_star`): before the check, `buffer` and `halves`
    read B whole in TF32 there, with the converting ring or the prepared one
    (a GH200's neighboringFlux o8, `prepared` and `halves`: 2.4e-4 against
    d), and only the fold kept B's halves (`fuse_refusal`).  No plan reads
    an operand in TF32 alone now."""
    group, context, _ = _plan(arch, shape, order, tensor_cores=True,
                              matrix_star=star, matrix_ring=ring)
    section = group.section
    assert eg.tf32_refusal(section, context) is None
    assert all(section.stage(p.first.tensor, (p.rows, p.depth)).split
               for step in section.steps
               if step.lowering in eg.ASYNC_LOWERINGS
               for p in step.products)


def test_neighboringflux_o8_with_the_ring_and_a_halves_on_the_oracle():
    """The GH200's case on the host oracle, which multiplies TF32 operands
    as the engines do: every element of a trip against NumPy within
    1e-6 of its largest cell -- B read whole in TF32 gave 6e-4 to 9e-4."""
    import numpy as np
    from tensorforge.common.basic_types import Addressing
    descrs = me._descrs('neighboringFlux', 8, 's')
    context = me._context('sm_90a', 's', multi_element='force',
                          tensor_cores=True, matrix_engine='auto',
                          matrix_ring='prepared', matrix_star='halves')
    gen = me._generate(descrs, context)
    tensors, inputs, _, after = me._on_the_oracle(gen, 2)
    output = next(t for t in tensors.values() if not t.is_tmp
                  and t.addressing.kind is not Addressing.NONE
                  and any(d.writes().tensor is t for outer in gen.descr_list
                          for d in outer.operations()))
    got = after(output)
    want = me._reference(descrs)(inputs, inputs[output.alias])
    for e in range(2):
        scale = np.abs(want[e]).max() or 1.0
        assert np.abs(got[e] - want[e]).max() <= 1e-6 * scale, e


@pytest.mark.parametrize('arch', ['sm_90a', 'sm_100a'])
def test_localfluxall_o8_was_precise_already(arch):
    """mma_sync with B whole, split where it reads it: unchanged, no note."""
    group, context, _ = _plan(arch, 'localFluxAll', 8)
    assert _engines(group.section) == ['mma_sync'] and group.elements == 4
    assert eg.precision_notes(group.section, context) == []


@pytest.mark.parametrize('arch,shape,order,before,elements', [
    ('sm_100a', 'localFlux', 8, 7, 14),
    ('sm_100a', 'localFluxAll', 7, 4, 7),
    ('sm_100a', 'neighboringFlux', 8, 7, 14),
    ('sm_100a', 'reuse', 8, 7, 14),
    ('sm_90a', 'chain', 8, 4, 14),
    ('sm_90a', 'localFluxAll', 7, 4, 7)])
def test_mma_sync_takes_the_larger_group(arch, shape, order, before,
                                         elements):
    """The comparison's other takes: the asynchronous engine kept both
    halves but held a smaller group (`before`) than mma_sync does -- a
    performance choice of the rule (`prefer_mma_sync`), not of accuracy."""
    group, context, _ = _plan(arch, shape, order)
    assert _engines(group.section) == ['mma_sync']
    assert group.elements == elements
    (note,) = eg.precision_notes(group.section, context)
    assert f'holds a group of {before}, mma_sync' in note


# -- the check ------------------------------------------------------------------- #

def test_the_check_refuses_a_tf32_plan_made_by_hand():
    """B whole on tcgen05 -- the plan the fill rule took before -- refused
    with its reason, and let through by the option; the terms are the
    reason: 3xTF32's three with B in halves, two without."""
    group, context, section = _plan('sm_100a', 'chain', 7)
    whole = eg.lower(section, context)
    assert _engines(whole) == ['tcgen05'] and not eg.split(whole)
    why = eg.tf32_refusal(whole, context)
    assert why.startswith('tcgen05 would read kDivM(')
    assert why.endswith('in TF32 alone, two of 3xTF32\'s three products '
                        '(about 2^-11 relative a product against 2^-22): it '
                        'needs Options.matrix_tf32_operands')
    allowed = me._context('sm_100a', 's', multi_element='force',
                          matrix_engine='auto', matrix_tf32_operands=True)
    assert eg.tf32_refusal(whole, allowed) is None
    stage = whole.stages[0]
    assert eg.async_terms(stage) == ((0, 0), (1, 0))
    assert eg.async_terms(dataclasses.replace(stage, split=True)) == \
        nvidia.TF32_TERMS


def test_the_generator_refuses_a_tf32_plan(monkeypatch):
    """A fill rule that hands the generator B whole -- as a change of it
    might -- stops the generator instead of building the kernel."""
    group, context, section = _plan('sm_100a', 'chain', 7)
    whole = eg.lower(section, context)
    original = eg.lower

    def lowered_whole(*args, **kwargs):
        kwargs = {**kwargs, 'split': False, 'engine': None}
        return original(*args, **kwargs)

    monkeypatch.setattr(eg, 'fill', lambda s, c, form=None: (
        eg.ElementGroup(group.elements, whole,
                        eg.shared_bytes(whole, group.elements)), ''))
    monkeypatch.setattr(eg, 'lower', lowered_whole)
    with pytest.raises(InternalError, match='TF32 alone'):
        me._generate(me._descrs('chain', 7, 's'), context)


def test_the_option_lets_the_plan_trade():
    """`matrix_tf32_operands`: the plan that read B in TF32 before, and the
    kernel's comment saying so."""
    group, context, _ = _plan('sm_100a', 'chain', 7, matrix_tf32_operands=True)
    assert _engines(group.section) == ['tcgen05'] and group.elements == 14
    assert not eg.split(group.section)
    (note,) = eg.precision_notes(group.section, context)
    assert note.endswith('read in TF32 alone, two of 3xTF32\'s three '
                         'products (about 2^-11 relative a product): '
                         'Options.matrix_tf32_operands')
    src = me._generate(me._descrs('chain', 7, 's'), context).get_kernel()
    assert '// matrix engine: ' + note in src


@pytest.mark.parametrize('asynchronous,mma,cap,taken', [
    (7, 14, 14, True), (14, 16, 14, False), (12, 14, 12, False),
    (4, 7, 14, True), (7, 7, 14, False)])
def test_the_one_comparison(asynchronous, mma, cap, taken):
    """mma_sync where it holds the larger group, counted to the
    asynchronous engine's own lane limit; the asynchronous engine on a
    tie."""
    assert eg.prefer_mma_sync(asynchronous, mma, cap) is taken


# -- the volume, as measured ----------------------------------------------------- #

VOLUME = [
    ('sm_90a', 6, 'buffer', 'convert', 16, 'wgmma', True, False, 0, ''),
    ('sm_90a', 6, 'buffer', 'prepared', 16, 'wgmma', True, False, 0, ''),
    ('sm_90a', 6, 'halves', 'convert', 16, 'wgmma', True, True, 3, ''),
    ('sm_90a', 6, 'halves', 'prepared', 16, 'wgmma', True, True, 3, ''),
    ('sm_90a', 6, 'engine', 'convert', 16, 'wgmma', True, False, 0, 'engine'),
    ('sm_90a', 6, 'engine', 'prepared', 16, 'wgmma', True, False, 0, 'engine'),
    ('sm_90a', 8, 'buffer', 'convert', 12, 'wgmma', True, True, 0, ''),
    ('sm_90a', 8, 'buffer', 'prepared', 12, 'wgmma', True, True, 0, ''),
    ('sm_90a', 8, 'halves', 'convert', 12, 'wgmma', True, True, 0, ''),
    ('sm_90a', 8, 'halves', 'prepared', 12, 'wgmma', True, True, 0, ''),
    ('sm_90a', 8, 'engine', 'convert', 12, 'wgmma', True, True, 0, ''),
    ('sm_90a', 8, 'engine', 'prepared', 12, 'wgmma', True, True, 0, 'engine'),
    ('sm_100a', 6, 'buffer', 'convert', 14, 'tcgen05', True, False, 0, ''),
    ('sm_100a', 6, 'buffer', 'prepared', 14, 'tcgen05', True, False, 0, ''),
    ('sm_100a', 6, 'halves', 'convert', 14, 'tcgen05', True, True, 3, ''),
    ('sm_100a', 6, 'halves', 'prepared', 14, 'tcgen05', True, True, 3, ''),
    ('sm_100a', 6, 'engine', 'convert', 14, 'tcgen05', True, False, 0,
     'engine'),
    ('sm_100a', 6, 'engine', 'prepared', 14, 'tcgen05', True, False, 0,
     'engine'),
    ('sm_100a', 8, 'buffer', 'convert', 7, 'tcgen05', True, True, 0, ''),
    ('sm_100a', 8, 'buffer', 'prepared', 7, 'tcgen05', True, True, 0, ''),
    ('sm_100a', 8, 'halves', 'convert', 7, 'tcgen05', True, True, 1, ''),
    ('sm_100a', 8, 'halves', 'prepared', 7, 'tcgen05', True, True, 1, ''),
    ('sm_100a', 8, 'engine', 'convert', 7, 'tcgen05', True, True, 1, ''),
    ('sm_100a', 8, 'engine', 'prepared', 14, 'tcgen05', True, True, 0,
     'engine')]


@pytest.mark.parametrize('arch,order,star,ring,elements,engine,split,'
                         'streamed,halves,folded', VOLUME)
def test_the_volume_plans_as_measured(arch, order, star, ring, elements,
                                      engine, split, streamed, halves,
                                      folded):
    """The volume's plans of the GH200/GB200 jobs (o6 and o8, every
    `matrix_star` and `matrix_ring`): the asynchronous engine at the group
    it had, B's halves as they were -- precise before, and unchanged."""
    group, context, _ = _plan(arch, 'volume', order, matrix_star=star,
                              matrix_ring=ring)
    section = group.section
    assert (group.elements, _engines(section), eg.split(section),
            eg.streamed(section), eg.halves(section), section.star) == (
        elements, [engine], split, streamed, halves, folded)
    assert eg.precision_notes(section, context) == []
