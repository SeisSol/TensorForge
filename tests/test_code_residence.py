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

import numpy as np
import pytest

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
