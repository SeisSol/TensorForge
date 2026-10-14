# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Several consecutive elements of the batch in one multiplication
(`Options.stack_elements`).

What a stack computes is what its elements compute one at a time.  These
tests hold the generated kernel to that in the host oracle: the stacked build
against the plain one, over three elements -- an odd count, so the last stack
is half empty -- with the middle one masked where the kernel takes a mask, so
that an element of a stack is past the batch in one trip and masked in the
other.  A cell neither build touches holds what memory held before, which is
what makes "nothing was written there" a comparison like any other.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import re
from pathlib import Path

import pytest

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.context import Context
from tensorforge.common.exceptions import GenerationError
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.common.options import Options
from tensorforge.generators import stacking
from tensorforge.common.operation import Operation
from tensorforge.generators.descriptions import ElementwiseDescr, GemmDescr
from tensorforge.generators.generator import Generator
from tensorforge.reference import kernel_eval

CASES = Path(__file__).parent / 'cases'


def _module(name):
    for path in sorted(CASES.rglob('*.py')):
        if path.name.startswith('_'):
            continue
        spec = importlib.util.spec_from_file_location(f'tf_stack_{path.stem}', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if getattr(mod, 'NAME', None) == name:
            return mod
    raise LookupError(name)


def _build(name, arch='sm_86', backend='cuda', **options):
    """The case's kernel and its launcher."""
    mod = _module(name)
    return _generate(mod.descr_list(), getattr(mod, 'DTYPE', None), arch,
                     backend, **options)


def _generate(descrs, fp, arch='sm_86', backend='cuda', **options):
    gen = Generator(descrs, Context(arch=arch, backend=backend, fp_type=fp,
                                    options=Options(**options)))
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    return gen.get_kernel(), gen.get_launcher()


#: How the runs mask the elements: the middle one of three, so that the
#: first stack writes its first element only and the second -- the batch ends
#: inside it -- its first only too; the first of four, so that a stack is
#: visited for its second element alone; and none of four.
MASKS = ([1, 0, 1], [0, 1, 1, 1], [1, 1, 1, 1])


def _run(kernel, launcher, flags=MASKS[0], races=None):
    lanes, _ = kernel_eval.launch_geometry(launcher)
    preset = {'flags0': list(flags)} if 'flags0' in kernel else None
    return kernel_eval.evaluate_wave(kernel, lanes, seed=5, globals_only=True,
                                     races=races, elements=len(flags),
                                     preset=preset)


def _by_alias(kernel, launcher, flags, races=None):
    """`_run`, its memory keyed by the operands' aliases: two builds that name
    the kernel's parameters in another order -- one reads an operand the
    batch shares through a copy the other does not make -- are handed the
    same numbers for the same operand, the ones the oracle's memory would
    hold for it under its alias."""
    meta = json.loads(re.search(r'tensorforge-meta: (\{.*\})', kernel).group(1))
    names = {o['alias']: o['name'] for o in meta['operands']}
    initial = kernel_eval.Slot(5)
    preset = {name: [initial.read(alias, i) for i in range(4 * 256)]
              for alias, name in names.items()}
    if 'flags0' in kernel:
        preset['flags0'] = list(flags)
    lanes, _ = kernel_eval.launch_geometry(launcher)
    memory = kernel_eval.evaluate_wave(kernel, lanes, seed=5, globals_only=True,
                                       races=races, elements=len(flags),
                                       preset=preset)
    alias = {name: a for a, name in names.items()}
    return {(alias.get(name, name), i): v for (name, i), v in memory.items()}


def _differ(got, want):
    """The cells the two runs leave different, a cell a run never touched
    holding the memory's initial value."""
    initial = kernel_eval.Slot(5)
    out = []
    for key in set(got) | set(want):
        a = got[key] if key in got else initial.read(*key)
        b = want[key] if key in want else initial.read(*key)
        if abs(a - b) > 1e-4 * max(1.0, abs(b)):
            out.append(key)
    return sorted(out)


@pytest.mark.parametrize('name', [
    # through pointers of the elements' own -- with merged runs over tables
    # below, `local_flux` with two stages
    'gemm_addressing_ptr_based',
    # strided, a product of two operands the element owns
    'gemm_56x18_x_18x18',
    # strided accumulations each over fewer rows, and a narrow write back
    'accumulate_narrowing_chain',
    'narrow_write_after_wide_read',
    # pointwise operations and reductions, beside contractions
    'elementwise_sliced_max',
    'mixed_ml_then_red',
    'reduction_sum_axis1',
])
def test_a_stack_computes_what_its_elements_compute(name):
    stacked, stacked_launcher = _build(name, stack_elements=2)
    plain, plain_launcher = _build(name)
    for flags in MASKS:
        races = []
        got = _run(stacked, stacked_launcher, flags, races=races)
        want = _run(plain, plain_launcher, flags)
        assert races == [], (flags, races)
        assert _differ(got, want) == [], flags


def test_an_operand_the_batch_shares_is_read_alike_by_a_stack():
    """A pointwise operation reads an operand the batch shares on fewer axes
    than its stacked destination has: the same for each element of the
    stack."""
    def t(alias, addressing):
        return SubTensor(Tensor([16, 16], addressing,
                                BoundingBox([0, 0], [16, 16]), alias=alias,
                                datatype=Datatype.F32))

    def descrs():
        return [ElementwiseDescr(Operation.ADD, t('C', Addressing.STRIDED),
                                 [t('A', Addressing.STRIDED),
                                  t('K', Addressing.NONE)])]
    stacked = _generate(descrs(), Datatype.F32, stack_elements=2)
    plain = _generate(descrs(), Datatype.F32)
    for flags in MASKS:
        races = []
        got = _by_alias(*stacked, flags, races=races)
        want = _by_alias(*plain, flags)
        assert races == [], (flags, races)
        assert [k for k in got if k[0] == 'C'], 'the result is written'
        assert _differ(got, want) == [], flags


def test_a_stack_writes_nothing_for_an_element_it_lacks():
    """The batch ends inside the second stack, and the first stack's second
    element is masked: neither is written, though both are computed -- from
    the stack's first element, read in their place."""
    kernel, launcher = _build('gemm_56x18_x_18x18', stack_elements=2)
    head = kernel[:kernel.index('{')]
    written = re.findall(r'[(,]\s*float\s*\*\s*(m\d+)', head)
    assert len(written) == 1, head
    got = _run(kernel, launcher, MASKS[0])
    volume = 56 * 18
    cells = {i for (name, i) in got if name == written[0]}
    assert cells & set(range(0, volume)), 'the first element is written'
    assert not cells & set(range(volume, 2 * volume)), 'the masked one is not'
    assert not cells & set(range(3 * volume, 4 * volume)), (
        'nor the one past the batch')


def test_the_traversal_visits_stacks():
    kernel, launcher = _build('local_flux', stack_elements=2)
    assert '< ((numElements0 + 1) / 2);' in kernel, kernel
    # each element of the stack is masked on its own, its flag read only
    # where the element is in the batch
    assert re.search(r'const bool \w*batchIdStack0_1 = 2 \* \w+ \+ 1 < '
                     r'numElements0 && \(flags0 == nullptr \|\| '
                     r'static_cast<bool>\(flags0\[2 \* \w+ \+ 1\]\)\);',
                     kernel), kernel
    # the first stack a thread visits is clamped to the stacks, as the loop
    # clamps its successor
    assert re.search(r'blockIdx\.x\)\) < \(\(numElements0 \+ 1\) / 2\)\) \? ',
                     kernel), kernel
    # and the grid covers the stacks
    assert '((numElements0 + 1) / 2)' in launcher, launcher
    # a tool reading the kernel back learns which operands are stacked
    meta = json.loads(re.search(r'tensorforge-meta: (\{.*\})', kernel).group(1))
    stacks = {o['name']: o.get('stack') for o in meta['operands']}
    assert stacks['m0'] is None, 'the operator the batch shares is no stack'
    assert set(stacks.values()) == {None, 2}, stacks


def test_with_two_stages_a_stack_computes_what_its_elements_compute():
    """The copies for the next stack issued a whole trip ahead into a second
    stage, each element's through its own pointer for that stack."""
    stacked, stacked_launcher = _build('local_flux', stack_elements=2,
                                       enable_multibuffer=True)
    plain, plain_launcher = _build('local_flux', enable_multibuffer=True)
    for flags in MASKS:
        races = []
        got = _run(stacked, stacked_launcher, flags, races=races)
        want = _run(plain, plain_launcher, flags)
        assert races == [], (flags, races)
        assert _differ(got, want) == [], flags


def test_every_write_of_a_stack_runs_under_its_elements_mask():
    """An element the stack lacks reads its stack's first element in its
    place and computes that element's numbers, so a write of them would put
    the same numbers where they already are -- invisible, until the write is
    an accumulation into memory, an atomic one on gfx942: it would count the
    first element twice.  So each write is held to its own element's mask by
    its form: the element's mask where the element is a constant, the
    choice among the masks by the element where a loop walks the stack."""
    for arch, backend, name in (('sm_86', 'cuda', 'gemm_56x18_x_18x18'),
                                ('gfx942', 'hip', 'gemm_add_accumulate')):
        kernel, _ = _build(name, arch=arch, backend=backend, stack_elements=2)
        lines = kernel.splitlines()
        writes = [i for i, line in enumerate(lines)
                  if re.search(r'\bglb_m\d+_s(e?)(\d+)\[[^\]]*\] = '
                               r'|atomic\w*\(&glb_m\d+_s(e?)\d+\[', line)]
        assert writes, (name, kernel)
        for i in writes:
            guards = [line for line in lines[max(0, i - 3):i] if 'if (' in line]
            assert guards, (name, lines[i])
            pointer = re.search(r'glb_m\d+_s(e?)(\d+)', lines[i])
            if pointer.group(1):
                assert re.search(r'== 0 \? batchIdStack0_0 : batchIdStack0_1',
                                 guards[-1]), (name, guards[-1], lines[i])
            else:
                assert f'batchIdStack0_{pointer.group(2)}' in guards[-1], (
                    name, guards[-1], lines[i])


def test_on_gfx942_the_rows_of_a_group_each_mask_their_stack():
    """A group of rows traverses the batch together on gfx942, its rows
    stacks; a row's elements are masked by the row's own index, so a row past
    the end -- held at its leader's stack for what it reads -- writes
    nothing."""
    kernel, _ = _build('local_flux', arch='gfx942', backend='hip',
                       stack_elements=2)
    assert re.search(r'const bool batchIdStack0_0 = 2 \* (\w+_row) \+ 0 < '
                     r'numElements0', kernel), kernel
    assert 'const bool batchIdActive0 = batchIdStack0_0 || batchIdStack0_1;' \
        in kernel, kernel
    from harness import syntax
    if syntax.compiler() is None:
        pytest.skip('no host C++ compiler')
    result = syntax.check_source(kernel, shim=syntax.shim_for('hip'))
    assert result.ok, result.errors()


def _gemm(addressing_a=Addressing.NONE):
    def t(shape, alias, addressing=Addressing.STRIDED):
        return SubTensor(Tensor(shape, addressing, BoundingBox([0, 0], shape),
                                alias=alias, datatype=Datatype.F32))
    return [GemmDescr(trans_a=False, trans_b=False,
                      a=t([56, 56], 'A', addressing_a), b=t([56, 9], 'B'),
                      c=t([56, 9], 'C'), alpha=1.0, beta=1.0)]


def test_the_stack_is_the_last_axis_of_what_an_element_owns():
    descrs = _gemm()
    stacked = stacking.stack(descrs, 2)
    (d,) = stacked
    a, b = d.ops
    assert a is descrs[0].ops[0], 'the operator the batch shares stays as it is'
    assert b.tensor.shape == (56, 9, 2) and stacking.factor(b.tensor) == 2
    assert d.dest.tensor.shape == (56, 9, 2)
    assert d.target == [[0, -1], [-1, 1, 2]]
    assert d.add_dims is None and d.add
    # a list already stacked is the list
    assert stacking.stack(stacked, 2) is stacked
    assert stacking.stack(descrs, 1) is descrs


def test_what_does_not_stack_is_refused_by_name():
    with pytest.raises(GenerationError, match='stack_elements = 3'):
        stacking.stack(_gemm(), 3)
    bad = _gemm()
    bad[0].dest = bad[0].ops[0]
    with pytest.raises(GenerationError, match='which the batch shares'):
        stacking.stack(bad, 2)
    rank0 = [GemmDescr(trans_a=False, trans_b=False, a=_gemm()[0].ops[0],
                       b=_gemm()[0].ops[1], c=_gemm()[0].dest, alpha=2.0,
                       beta=0.0)]
    # a factor passed by value is the batch's, and stays as it is
    assert stacking.stack(rank0, 2)[0].ops[2] is rank0[0].ops[2]


def _built(descrs, arch='gfx942', backend='hip', **options):
    gen = Generator(descrs, Context(arch=arch, backend=backend,
                                    fp_type=Datatype.F32,
                                    options=Options(**options)))
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    return gen


def test_the_bound_is_per_element_of_the_batch():
    """A stacked build counts its figures for the whole stack, and
    `pipeline.of` divides them by it -- so that the tuner weighs a stack
    against a plain build in one unit, and a margin read off the bound means
    the same for both (`tuning._worth_it`)."""
    from tensorforge.analysis import pipeline
    gen = _built(_gemm(), stack_elements=2)
    assert gen.stack == 2
    built = gen.built
    whole = pipeline.bound(built.metrics.issue_mix, built.metrics.memory_bytes,
                           built._context.target.hw,
                           max(1, built._num_threads or 1),
                           built._num_active_threads or None)
    per = pipeline.of(gen)
    assert per.cycles == pytest.approx(whole.cycles / 2)
    assert per.binding == whole.binding
    assert per.per_resource == pytest.approx(
        {name: v / 2 for name, v in whole.per_resource.items()})
    plain = _built(_gemm())
    assert plain.stack == 1
    assert pipeline.of(plain).cycles == pytest.approx(pipeline.bound(
        plain.metrics.issue_mix, plain.metrics.memory_bytes,
        plain.built._context.target.hw, max(1, plain.built._num_threads or 1),
        plain.built._num_active_threads or None).cycles)


def test_a_persistent_grid_counts_its_rounds_in_stacks():
    """The least time a measurement could take (`MeasuredScore.least`): the
    grid walks the batch a multiplication -- a stack -- per slot, so a batch
    of two rounds of elements and two more is two rounds of stacks, not
    three."""
    from tensorforge.analysis import pipeline
    from tensorforge.generators import tuning
    gen = _built(_gemm(), stack_elements=2)
    slots = gen.launch_config().mults_per_block * gen.resident_blocks
    batch = 2 * slots + 2
    score = tuning.MeasuredScore(lambda result: None, sms=1, clock_ghz=1.0,
                                 batch=batch)
    result = tuning.Build(tuning.Candidate(), generator=gen,
                          context=gen.built._context)
    bound = pipeline.of(gen, conservative=True)
    efficiency = (slots + 1) / (2 * slots)
    assert score.least(result) == pytest.approx(
        pipeline.least_seconds(bound, batch, 1, 1.0, efficiency))
