# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A batch-constant `B` whose numbers the description carries, as literals.

`inline_constants`: where every reduction reading it unrolls whole and it has
at most that many non-zero entries, the kernel reads it as a `Data` operand --
an FMA takes the number as an immediate, and a zero drops its product -- and
the launcher keeps the pointer the caller passes without reading it.  Large by
default on NVIDIA, where the immediate fits the instruction; small on AMD and
Intel.  What it leaves goes by value (`argument_constants`) or stays in memory.
"""

from __future__ import annotations

import contextlib
import io
import json
import re

import numpy as np
import pytest

import kernel_eval
from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.context import Context, Options
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import MultilinearDescr
from tensorforge.generators.generator import Generator
from tensorforge.generators.tuning import space

M = 8


def tensor(shape, alias, data=None, addressing=None):
    return Tensor(list(shape), addressing or (Addressing.NONE if data is not None
                                              else Addressing.STRIDED),
                  BoundingBox([0] * len(shape), list(shape)), alias=alias,
                  datatype=Datatype.F32, data=data)


def numbers(shape):
    return np.arange(1, int(np.prod(shape)) + 1,
                     dtype=np.float64).reshape(shape, order='F') / 8.0


def product(k=M, n=M, backend='cuda', arch='sm_86', role='B', **options):
    """`C[i,j] = X[i,l] K[l,j]` (or `K[i,l] X[l,j]`), K carrying numbers."""
    if role == 'B':
        x, c = tensor((M, k), 'X'), tensor((M, n), 'C')
        kk = tensor((k, n), 'K', numbers((k, n)))
        descr = MultilinearDescr(SubTensor(c), [SubTensor(x), SubTensor(kk)],
                                 [[0, -1], [-1, 1]], [[0, 1], [0, 1]])
    else:
        kk = tensor((M, k), 'K', numbers((M, k)))
        x, c = tensor((k, n), 'X'), tensor((M, n), 'C')
        descr = MultilinearDescr(SubTensor(c), [SubTensor(kk), SubTensor(x)],
                                 [[0, -1], [-1, 1]], [[0, 1], [0, 1]])
    context = Context(arch=arch, backend=backend, fp_type=Datatype.F32,
                      options=Options(**options) if options else None)
    generator = Generator([descr], context)
    with contextlib.redirect_stdout(io.StringIO()):
        generator.generate()
    return generator, (x, kk, c)


def signature(generator):
    return generator.get_kernel().split('{')[0]


def rows(generator):
    line = generator.get_kernel().split('tensorforge-meta: ')[1].split('\n')[0]
    return {row['alias']: row for row in json.loads(line)['operands']}


def inlined(generator):
    return rows(generator)['K'].get('inlined') is True


class TestNvidia:
    def test_it_is_literals_by_default(self):
        generator, (_, k, _) = product()
        assert inlined(generator)
        assert not re.search(rf'\b{k.name}\b', signature(generator))
        assert '1.5f' in generator.get_kernel()

    def test_the_launcher_keeps_the_pointer_and_does_not_read_it(self):
        generator, (_, k, _) = product()
        assert re.search(rf'const float \*\s*{k.name}\b', generator.get_header())
        launcher = generator.get_launcher()
        assert f'(void){k.name};' in launcher
        assert not re.search(rf'\b{k.name}\b', launcher.split('<<<')[1])

    def test_the_numbers_are_the_right_ones(self):
        generator, (x, k, c) = product()
        lanes, mults = kernel_eval.launch_geometry(generator.get_launcher())
        mem = kernel_eval.evaluate_wave(generator.get_kernel(), lanes, seed=5,
                                        globals_only=True, mults=mults)
        seed = kernel_eval.Slot(5)
        xx = np.array([[seed.read(x.name, i + j * M) for j in range(M)]
                       for i in range(M)])
        want = xx @ k.data
        got = np.array([[mem.get((c.name, i + j * M), np.nan)
                         for j in range(M)] for i in range(M)])
        assert np.max(np.abs(got - want)) < 1e-4 * np.max(np.abs(want))

    def test_below_the_limit_it_goes_by_value(self):
        generator, _ = product(inline_constants=M * M - 1)
        assert not inlined(generator)
        assert rows(generator)['K'].get('embedded') is True

    def test_zero_takes_none(self):
        generator, _ = product(inline_constants=0, argument_constants=False)
        assert not inlined(generator)
        assert 'embedded' not in rows(generator)['K']

    def test_a_reduction_that_rolls_takes_none(self):
        """80 steps, over `k_unroll_max`: rolled, so no literal to index."""
        generator, _ = product(k=80, n=4)
        assert not inlined(generator)

    def test_nor_where_a_roll_is_asked_for(self):
        generator, _ = product(k_roll=4)
        assert not inlined(generator)

    def test_a_lead_operand_stays_in_memory(self):
        generator, (_, k, _) = product(role='A')
        assert not inlined(generator)
        assert re.search(rf'\b{k.name}\b', signature(generator))


class TestAmd:
    def test_a_small_operator_is_literals(self):
        """16 entries: under AMD's default limit."""
        generator, (_, k, _) = product(k=4, n=4, backend='hip', arch='gfx942')
        assert inlined(generator)

    def test_a_larger_one_stays_in_the_constant_space(self):
        """81 entries: over it -- read out of device memory by scalar loads."""
        generator, (_, k, _) = product(k=9, n=9, backend='hip', arch='gfx942')
        assert not inlined(generator)
        assert f'ConstantMemspace> {k.name}' in signature(generator)


def test_the_tuning_space_turns_the_limit_at_the_operands_counts():
    x, c = tensor((M, M), 'X'), tensor((M, M), 'C')
    k = tensor((M, M), 'K', numbers((M, M)))
    descrs = [MultilinearDescr(SubTensor(c), [SubTensor(x), SubTensor(k)],
                               [[0, -1], [-1, 1]], [[0, 1], [0, 1]])]
    context = Context(arch='sm_86', backend='cuda', fp_type=Datatype.F32)
    knob = next(k for k in space(descrs, context) if k.name == 'inline_constants')
    assert tuple(knob.values({})) == (0, M * M)


def test_the_harness_still_passes_the_buffer():
    from harness.driver_emit import collect_operands
    generator, (_, k, _) = product()
    assert k.name in [op.kernel_name for op in collect_operands(generator)]


@pytest.mark.parametrize('options', [{}, {'inline_constants': 0},
                                     {'inline_constants': 0,
                                      'argument_constants': False}],
                         ids=['literals', 'by-value', 'memory'])
def test_the_call_yateto_is_given_matches_the_launcher(options):
    """The yateto frontend writes the call through `generate_call_site`,
    which skipped every `Data` symbol -- an inlined one included, whose
    pointer the launcher still takes: one argument short of the signature."""
    generator, _ = product(**options)
    generator.register()
    names = {s.obj.alias: s.obj.alias
             for s in generator._scopes.get_global_scope().values()}
    call = generator.generate_call_site(names, {n: '0' for n in names})
    arguments = call.split('(', 1)[1].rsplit(')', 1)[0].split(', ')
    prototype = generator.get_header().split(
        f'launcher_{generator.get_base_name()}(')[1].split(')')[0]
    assert len(arguments) == len(prototype.split(', '))
