# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A batch-constant first operand with few nonzeros per row, gathered.

`ell_gather`: the lanes run along the rows of `S` in `C = S X`, and the
usual nest issues every step of the reduction on every lane.  Gathered
(`MultilinearInstruction._ell_plan`), a lane takes its own row's nonzeros --
sliced ELL, a slot of lanes as many steps as its fullest row -- and reads
`X` at their columns: at a per-lane address where `X` is in memory, across
the lanes where it is a register image.  Columns and weights the lanes do
not share come from tables at namespace scope, named by their numbers.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.context import Context, Options
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr
from tensorforge.generators.generator import Generator
from tensorforge.reference import kernel_eval

K, N = 40, 9


def structured(kind, seed=5):
    """`S` with one nonzero per row at a permutation, unweighted or not, and
    a second and third in some rows."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(K)
    s = np.zeros((K, K))
    for row in range(K):
        s[row, perm[row]] = 1.0 if kind == 'perm' else rng.uniform(0.5, 1.5)
        if kind in ('two', 'three') and row % 2 == 0:
            s[row, (perm[row] + 7) % K] = rng.uniform(-1.0, -0.25)
        if kind == 'three' and row % 3 == 0:
            s[row, (perm[row] + 13) % K] = rng.uniform(0.25, 1.0)
    return s


def product(values, arch='sm_86', backend='cuda', **options):
    s = Tensor([K, K], Addressing.NONE, BoundingBox([0, 0], [K, K]),
               alias='S', datatype=Datatype.F32, data=values)
    x = Tensor([K, N], Addressing.STRIDED, BoundingBox([0, 0], [K, N]),
               alias='X', datatype=Datatype.F32)
    c = Tensor([K, N], Addressing.STRIDED, BoundingBox([0, 0], [K, N]),
               alias='C', datatype=Datatype.F32)
    generator = Generator(
        [GemmDescr(False, False, SubTensor(s), SubTensor(x), SubTensor(c),
                   alpha=1.0, beta=0.0)],
        Context(arch=arch, backend=backend, fp_type=Datatype.F32,
                options=Options(**options)))
    generator.generate()
    return generator, (s, x, c)


def tables(generator):
    return sorted(set(re.findall(r'\btf_ell[cw]_[0-9a-f]+\b',
                                 generator.get_kernel())))


def computed(generator, x, c):
    lanes, mults = kernel_eval.launch_geometry(generator.get_launcher())
    mem = kernel_eval.evaluate_wave(generator.get_kernel(), lanes, seed=5,
                                    globals_only=True, mults=mults)
    seed = kernel_eval.Slot(5)
    xx = np.array([[seed.read(x.name, i + j * K) for j in range(N)]
                   for i in range(K)])
    got = np.array([[mem.get((c.name, i + j * K), np.nan) for j in range(N)]
                    for i in range(K)])
    return xx, got


@pytest.mark.parametrize('kind', ['perm', 'weighted', 'two', 'three'])
def test_the_numbers_are_the_right_ones(kind):
    values = structured(kind)
    generator, (_, x, c) = product(values)
    assert tables(generator)
    xx, got = computed(generator, x, c)
    want = values @ xx
    assert np.max(np.abs(got - want)) < 1e-5 * np.max(np.abs(want))


def test_a_permutation_reads_its_columns_and_multiplies_by_nothing():
    generator, _ = product(structured('perm'))
    names = tables(generator)
    assert [n[:7] for n in names] == ['tf_ellc']
    body = generator.get_kernel().split('__global__')[1]
    assert not re.search(r'\*\s*v\d+_data', body)


def test_the_tables_are_defined_once_per_translation_unit():
    """yateto appends every kernel of a file to one translation unit: a
    table is guarded by its name, and its name says which numbers it holds,
    so a second kernel with the same operator defines nothing twice."""
    first, _ = product(structured('two'))
    second, _ = product(structured('two'))
    other, _ = product(structured('two', seed=6))
    assert tables(first) == tables(second)
    assert not set(tables(first)) & set(tables(other))
    for name in tables(first):
        guard = f'TENSORFORGE_TABLE_{name.upper()}'
        assert f'#ifndef {guard}\n#define {guard}\n' in first.get_kernel()


def test_zero_turns_it_off():
    generator, _ = product(structured('two'), ell_gather=0)
    assert not tables(generator)


def test_a_row_fuller_than_the_limit_is_not_gathered():
    generator, _ = product(structured('three'), ell_gather=2)
    assert not tables(generator)


def test_a_dense_operator_is_not_gathered():
    """Every step has a nonzero in every row: gathering is no fewer steps."""
    values = np.random.default_rng(3).uniform(0.5, 1.5, (K, K))
    generator, _ = product(values, ell_gather=K)
    assert not tables(generator)


def test_the_identity_takes_precedence():
    """It needs neither a table nor a read across the lanes."""
    generator, _ = product(np.diag(np.linspace(0.5, 1.5, K)))
    assert not tables(generator)


def test_a_register_image_is_gathered_across_the_lanes():
    """On gfx1150 `X` is a register image over two slots of 32 lanes: each
    slot of it is gathered on its own, through `ds_bpermute`."""
    generator, _ = product(structured('two'), arch='gfx1150', backend='hip')
    kernel = generator.get_kernel()
    assert tables(generator)
    assert re.search(r'tensorforge::gather<32>\(\w+, \w+\)', kernel)


def test_the_operator_stays_a_parameter():
    """The gather takes its numbers into tables, and the kernel is handed
    the operator all the same: the interface does not move with the path."""
    generator, (s, _, _) = product(structured('two'))
    kernel = generator.get_kernel()
    signature = kernel[kernel.index('__global__'):].split('{')[0]
    assert re.search(rf'\b{s.name}\b', signature)
