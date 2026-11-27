# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""How accurate the arithmetic TensorForge chooses has to be
(`Options.accuracy`).

What is checked here: each term plan's class by the vendor whose engine adds
its products up (`pir.terms.ACCURACY`, `product_accuracy`), a kernel's from
its bodies (`body_accuracy`), a form's before it is lowered
(`element_group.form_accuracy`); `multi_element=auto` under `ffma`, which
takes the fallbacks of the table's tile rows and builds nothing of 3xTF32's
class on NVIDIA; the tuner's matrix path at F32 kept to the class
(`tuning._relaxed_refused`); what the options ask for themselves built as
asked, with the kernel's comment saying where it is below the class asked
for; and the default, `relaxed`, changing nothing.  CPU only.
"""

from __future__ import annotations

import pytest

import test_form_choices as fc
import test_multi_element as me
from test_lanes_rule import _local_flux
from test_matrix_twin import _offered
from tensorforge.backend.instructions.compute.primitives import (intel,
                                                                 intel_spmd,
                                                                 nvidia)
from tensorforge.backend.instructions.compute.primitives.amd import stacked
from tensorforge.backend.pir import terms
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context, Options
from tensorforge.common.options import registry
from tensorforge.generators import element_group as eg
from tensorforge.generators import tuning

FFMA = {'accuracy': 'ffma'}


def test_the_option_is_relaxed_by_default():
    """`relaxed` by default, from `TF_ACCURACY` in the environment, `ffma`
    the other value; asked for as the default, no option at all."""
    entry = registry()['accuracy']
    assert entry.default == 'relaxed' and entry.env == 'TF_ACCURACY'
    assert entry.parse(' FFMA ') == 'ffma'
    with pytest.raises(ValueError):
        entry.parse('better')
    context = me._context('sm_90a', 's', accuracy='relaxed')
    assert context.get_user_options().describe() == 'default'
    assert eg.accuracy_asked(me._context('sm_90a', 's')) == 'relaxed'


def test_each_plan_has_its_engines_class():
    """3xTF32 and A's halves by B whole on NVIDIA's engines, which
    accumulate with RZ, 3xTF32's class; Intel's 3xTF32 in two chains
    rounded to nearest and AMD's stacked BF16X6 FFMA's; a product in full
    FFMA's; a plan on a vendor nobody measured it on, 3xTF32's; one an
    accumulator a product as its whole."""
    assert terms.product_accuracy(nvidia.TF32_PLAN, 'nvidia') == 'relaxed'
    assert terms.product_accuracy(eg.TF32_HALVES_BY_WHOLE, 'nvidia') == \
        'relaxed'
    assert terms.product_accuracy(intel.TF32_PLAN, 'intel') == 'ffma'
    assert terms.product_accuracy(intel_spmd.PLAN, 'intel') == 'ffma'
    assert terms.product_accuracy(stacked.BF16X6, 'amd') == 'ffma'
    assert terms.product_accuracy(None, 'nvidia') == 'ffma'
    assert terms.product_accuracy(nvidia.TF32_PLAN, 'amd') == 'relaxed'
    assert nvidia.TF32_PLAN.each().name == '3xtf32/each'
    assert terms.product_accuracy(nvidia.TF32_PLAN.each(), 'nvidia') == \
        'relaxed'
    assert terms.product_accuracy(intel.TF32_PLAN.each(), 'intel') == 'ffma'
    assert terms.admits('relaxed', 'relaxed') and terms.admits('ffma', 'ffma')
    assert terms.admits('ffma', 'better') and not terms.admits('ffma',
                                                               'relaxed')


def test_a_forms_class_before_it_is_lowered():
    """The tile 3xTF32's class on NVIDIA; the FMA forms and the per-element
    kernel FFMA's."""
    section = eg.classify(fc._kernel('gpu_volume', 8))[0]
    for arch in ('sm_90a', 'sm_100a'):
        context = me._context(arch, 's')
        assert eg.form_accuracy('tile', section, context) == 'relaxed'
        for form in ('slots', 'lanes', 'warp', 'element'):
            assert eg.form_accuracy(form, section, context) == 'ffma'


#: (kernel, order, arch, form): under `ffma` the fallback of each tile row
#: the jobs measured.
FFMA_FORMS = [('gpu_volume', 8, 'sm_90a', 'slots'),
              ('gpu_volume', 8, 'sm_100a', 'slots'),
              ('gpu_volume/sparse', 8, 'sm_100a', 'slots'),
              ('gpu_volume', 10, 'sm_90a', 'element'),
              ('gpu_localFluxAllPre', 6, 'sm_100a', 'element'),
              ('gpu_neighboringFluxPre[0]', 10, 'sm_100a', 'element')]


@pytest.mark.parametrize('kernel,order,arch,form', FFMA_FORMS,
                         ids=[f'{k.replace("/", "-")}-o{o}-{a}'
                              for k, o, a, _ in FFMA_FORMS])
def test_under_ffma_auto_takes_the_tile_rows_fallbacks(kernel, order, arch,
                                                        form):
    """`auto` with `accuracy=ffma`: no tile where a row names it -- its
    3xTF32 is below the class -- but the row's fallback, and the comment,
    or the plan's reason, says why; the tile under `relaxed`."""
    descrs = fc._kernel(kernel, order)
    context = me._context(arch, 's', **fc.AUTO, **FFMA)
    section = eg.classify(descrs)[0]
    row = eg.form_choice(section, context)
    group, why = eg.auto_plan(section, context)
    assert fc._form(group) == form
    declined = ('the high-order tile declined: its 3xTF32 on the '
                'asynchronous engine is relaxed, below accuracy=ffma')
    if group is not None:
        assert group.form_note == (f'{declined}; the slot form, the row\'s '
                                   f'fallback (FORM_CHOICES: {row.why})')
    else:
        assert why == f'{declined}; the per-element kernel, the row\'s ' \
                      f'fallback'
    assert fc._form(eg.plan(descrs, me._context(arch, 's', **fc.AUTO))) == \
        'tile'


def test_under_ffma_auto_builds_nothing_below_its_class():
    """Built under `auto` and `ffma`, the o8 volume is the slot form's FMA
    kernel: no product of 3xTF32's class, and no line on accuracy."""
    descrs = fc._kernel('gpu_volume', 8)
    gen = me._generate(descrs, me._context('sm_100a', 's', **fc.AUTO,
                                           **FFMA))
    built = gen.built if getattr(gen, 'built', None) is not None else gen
    assert built._element_group.lanes.mapping is not None
    bodies = [s.body for s in built._sections]
    assert terms.body_accuracy(bodies, 'nvidia') == ('ffma', None)
    assert '// accuracy:' not in gen.get_kernel()


def test_the_tuner_keeps_to_the_class():
    """Under `ffma` the lanes rule's matrix twin is offered where every
    product is F64 (DMMA's, FFMA's class) and not at F32 (3xTF32), and the
    tuner's `tensor_cores` knob keeps the path off at F32; under `relaxed`
    both as before."""
    assert _offered('sm_90', Datatype.F32)
    assert _offered('sm_90', Datatype.F32, **FFMA) == []
    assert _offered('sm_100', Datatype.F64, **FFMA)

    def paths(dtype, **opts):
        descrs = _local_flux(datatype=dtype)()
        context = Context(arch='sm_90', backend='cuda', fp_type=dtype,
                          options=Options(**opts))
        origin = tuning.start(descrs, context)
        knob = {k.name: k for k in tuning.space(descrs, context)}
        return tuple(knob['tensor_cores'].values(origin))
    assert paths(Datatype.F32) == (False, True)
    assert paths(Datatype.F32, **FFMA) == (False,)
    assert paths(Datatype.F64, **FFMA) == (False, True)


def _notes(descrs, arch, backend, dtype, **opts):
    """A build's lines on accuracy, and its kernel's class."""
    import contextlib
    import io
    import warnings
    from tensorforge.generators.generator import Generator
    context = Context(arch=arch, backend=backend, fp_type=dtype,
                      options=Options(**opts))
    gen = Generator(descrs, context)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        gen.generate()
    built = gen.built if getattr(gen, 'built', None) is not None else gen
    level, _ = terms.body_accuracy([s.body for s in built._sections],
                                   context.target.hw.vendor)
    return level, [line.strip() for line in gen.get_kernel().splitlines()
                   if line.strip().startswith('// accuracy:')]


#: What asks for a matrix path of 3xTF32's class itself: the description,
#: the target and the options.
REQUESTS = [
    ('matrix_tile', lambda: fc._kernel('gpu_volume', 8), 'sm_100a',
     {'matrix_tile': 'high'}, 'matrix_tile=high'),
    ('tensor_cores', lambda: _local_flux()(), 'sm_90a',
     {'tensor_cores': True}, 'tensor_cores=1'),
    ('force', lambda: me._descrs('volume', 6, 's'), 'sm_100a',
     {'multi_element': 'force'}, 'multi_element=force')]


@pytest.mark.parametrize('name,descrs,arch,opts,by', REQUESTS,
                         ids=[r[0] for r in REQUESTS])
def test_what_the_options_ask_for_is_built_as_asked_and_said(name, descrs,
                                                             arch, opts, by):
    """The high-order tile, the per-element kernel's mma.sync and the
    layout's tilings, asked for, are built as asked under `ffma` too, of
    3xTF32's class -- and the kernel's comment says so below the options
    line, naming what asked for it; under `relaxed` nothing is said."""
    level, notes = _notes(descrs(), arch, 'cuda', Datatype.F32, **opts, **FFMA)
    assert level == 'relaxed'
    assert notes == [f'// accuracy: 3xTF32 on NVIDIA\'s matrix engines ({by}),'
                     f' relaxed, below accuracy=ffma: they accumulate with RZ, '
                     f'4x to 9x FFMA\'s normwise error on SeisSol\'s volume '
                     f'from o4 to o14, growing with the contraction']
    level, notes = _notes(descrs(), arch, 'cuda', Datatype.F32, **opts)
    assert level == 'relaxed' and notes == []


@pytest.mark.parametrize('arch,backend,dtype,opts', [
    ('sm_90a', 'cuda', Datatype.F64, {'tensor_cores': True}),
    ('pvc', 'oneapi', Datatype.F32, {'sycl_dpas': True}),
    ('gfx942', 'hip', Datatype.F32, {'amd_matrix_k4': True,
                                     'amd_matrix_k4_emulate': True})],
    ids=['dmma', 'dpas', 'bf16x6'])
def test_a_matrix_path_of_ffmas_class_is_not_said(arch, backend, dtype, opts):
    """DMMA at F64, Intel's DPAS and AMD's stacked scheme under `ffma`: of
    FFMA's class, no line on accuracy."""
    level, notes = _notes(_local_flux(datatype=dtype)(), arch, backend, dtype,
                          **opts, **FFMA)
    assert level == 'ffma' and notes == []
