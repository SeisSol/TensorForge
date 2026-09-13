# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""An operand whose data is in the code rather than in memory.

`Residence` is the axis apart from `Addressing`: the latter says how an
address is formed from a parameter, and a code-resident operand has neither.
Its numbers arrive with the description, so the kernel reads them where it
would otherwise have issued a load, nothing is passed for it, and the host
allocates nothing.
"""

from __future__ import annotations

import contextlib
import io
import json
import re

import numpy as np
import pytest

import kernel_eval

from tensorforge.common.basic_types import Addressing, Datatype, Residence
from tensorforge.common.context import Context
from tensorforge.common.exceptions import GenerationError
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr
from tensorforge.generators.generator import Generator

N = 8


def data():
    values = np.zeros((N, N), dtype=np.float64)
    values[0, 0] = 0.5
    values[1, 2] = -1.0
    return values


def operator(residence):
    return SubTensor(Tensor([N, N], Addressing.NONE,
                            BoundingBox([0, 0], [N, N]), alias='A',
                            datatype=Datatype.F32, data=data(),
                            residence=residence))


def generated(residence):
    """The generated surface for `C[b] = A @ B[b]`, A as given."""
    perBatch = [SubTensor(Tensor([N, N], Addressing.STRIDED,
                                 BoundingBox([0, 0], [N, N]), alias=alias,
                                 datatype=Datatype.F32))
                for alias in ('B', 'C')]
    context = Context(arch='sm_86', backend='cuda', fp_type=Datatype.F32)
    generator = Generator([GemmDescr(False, False, operator(residence),
                                     *perBatch, alpha=1.0, beta=0.0)],
                          context)
    generator.generate()
    return generator


def signature(generator):
    return generator.get_kernel().split('{')[0]


class TestResidenceIsStated:
    def test_a_tensor_resides_in_memory_unless_it_says_otherwise(self):
        assert Tensor([N, N], Addressing.NONE).residence is Residence.MEMORY

    def test_the_spellings_the_description_uses_are_read(self):
        assert Residence.str2residence('code') is Residence.CODE
        assert Residence.str2residence('argument') is Residence.ARGUMENT
        assert Residence.str2residence('memory') is Residence.MEMORY

    def test_an_unknown_spelling_is_refused(self):
        with pytest.raises(ValueError, match='residence must be one of'):
            Residence.str2residence('register')

    def test_code_residence_without_data_is_refused(self):
        """Nothing else could supply the numbers."""
        with pytest.raises(GenerationError, match='carries no data'):
            Tensor([N, N], Addressing.NONE, residence=Residence.CODE)


class TestGeneratedKernel:
    def test_the_numbers_are_in_the_kernel(self):
        assert '0.5' in generated(Residence.CODE).get_kernel()

    def test_nothing_is_passed_for_it(self):
        """Three operands, and only the two in memory are parameters."""
        inCode = signature(generated(Residence.CODE))
        inMemory = signature(generated(Residence.MEMORY))
        assert inCode.count('float') == inMemory.count('float') - 1

    def test_the_same_operator_in_memory_is_a_parameter_and_not_a_literal(self):
        kernel = generated(Residence.MEMORY).get_kernel()
        assert '0.5' not in kernel

    def test_the_two_kernels_are_not_the_same_kernel(self):
        """The name is the hash of the source, and the source differs."""
        assert generated(Residence.CODE).get_base_name() \
            != generated(Residence.MEMORY).get_base_name()


def product(code_first, backend='cuda', arch='sm_86'):
    """`C[b] = A @ B[b]` (or `B[b] @ A`), A in the code; the generator and the
    two per-batch tensors."""
    perBatch = [Tensor([N, N], Addressing.STRIDED, BoundingBox([0, 0], [N, N]),
                       alias=alias, datatype=Datatype.F32)
                for alias in ('B', 'C')]
    b, c = (SubTensor(t) for t in perBatch)
    first, second = ((operator(Residence.CODE), b) if code_first
                     else (b, operator(Residence.CODE)))
    context = Context(arch=arch, backend=backend, fp_type=Datatype.F32)
    generator = Generator([GemmDescr(False, False, first, second, c,
                                     alpha=1.0, beta=0.0)], context)
    with contextlib.redirect_stdout(io.StringIO()):
        generator.generate()
    return generator, perBatch


@pytest.mark.parametrize('code_first', [True, False], ids=['A', 'B'])
def test_the_numbers_are_the_right_ones(code_first):
    """Against numpy, through the host oracle.

    As the first operand the numbers run along the lead index, which is the
    lane's: reading them as the other indices are read gave every lane row
    0's numbers, and `C` came out as `0.5 B[0, :]` in every row.
    """
    generator, (b, c) = product(code_first)
    lanes, mults = kernel_eval.launch_geometry(generator.get_launcher())
    mem = kernel_eval.evaluate_wave(generator.get_kernel(), lanes, seed=3,
                                    globals_only=True, mults=mults)
    seed = kernel_eval.Slot(3)
    batch = np.array([[seed.read(b.name, i + j * N) for j in range(N)]
                      for i in range(N)])
    want = data() @ batch if code_first else batch @ data()
    got = np.array([[mem.get((c.name, i + j * N), np.nan) for j in range(N)]
                    for i in range(N)])
    assert np.max(np.abs(got - want)) < 1e-4 * np.max(np.abs(want))


@pytest.mark.parametrize('backend,arch', [('hip', 'gfx942'),
                                          ('hip', 'gfx1150'),
                                          ('oneapi', 'pvc')])
@pytest.mark.parametrize('code_first', [True, False], ids=['A', 'B'])
def test_nothing_reads_it_by_a_name(code_first, backend, arch):
    """The AMD broadcast path loaded a second operand in the code by the
    parameter name it does not have (`m15[threadIdx.x]`), and hipcc stopped
    at the undeclared name."""
    generator, _ = product(code_first, backend, arch)
    meta = generator.get_kernel().split('tensorforge-meta: ')[1].split('\n')[0]
    name = next(o['name'] for o in json.loads(meta)['operands']
                if o['alias'] == 'A')
    # the banner spells the operation in index notation, which is no read
    code = re.sub(r'//[^\n]*', '', generator.get_kernel())
    assert not re.search(rf'\b{name}\s*\[', code)
