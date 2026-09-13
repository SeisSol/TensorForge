# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""An operand passed by value: `Residence.ARGUMENT`.

The caller hands over the numbers once per launch, and they live where kernel
arguments live -- the constant bank on NVIDIA, the kernarg segment on AMD --
so a batch-constant `B` is an operand the multiply reads as it is rather than
a load.  The kernel indexes it like the pointer it would otherwise be, and the
launcher copies it out of host memory.
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
from tensorforge.common.context import Context, Options
from tensorforge.common.exceptions import GenerationError
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.spp import ListSPP
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr
from tensorforge.generators.generator import Generator

N = 8
TARGETS = [('cuda', 'sm_86'), ('hip', 'gfx942'), ('hip', 'gfx1150'),
           ('oneapi', 'pvc'), ('acpp', 'sm_86')]


def tensor(alias, addressing=Addressing.STRIDED, residence=Residence.MEMORY,
           size=N):
    return Tensor([size, size], addressing, BoundingBox([0, 0], [size, size]),
                  alias=alias, datatype=Datatype.F32, residence=residence)


def generated(descrs, backend='cuda', arch='sm_86', **options):
    context = Context(arch=arch, backend=backend, fp_type=Datatype.F32,
                      options=Options(**options) if options else None)
    generator = Generator(descrs, context)
    with contextlib.redirect_stdout(io.StringIO()):
        generator.generate()
    return generator


def product(passed_first, backend='cuda', arch='sm_86', size=N):
    """`C[b] = A B[b]` (or `B[b] A`), A passed by value."""
    a = tensor('A', Addressing.NONE, Residence.ARGUMENT, size)
    b, c = tensor('B', size=size), tensor('C', size=size)
    first, second = (a, b) if passed_first else (b, a)
    return generated([GemmDescr(False, False, SubTensor(first),
                                SubTensor(second), SubTensor(c),
                                alpha=1.0, beta=0.0)], backend, arch), (a, b, c)


def code(generator):
    """The kernel without its comments: the banner names every operand."""
    return re.sub(r'//[^\n]*', '', generator.get_kernel())


def staged(text, name):
    """Whether the kernel reads `name` out of shared memory: the binding the
    body reads through then points into the arena."""
    return re.search(rf'\bglb_{name} = &totalShrMem\[', text) is not None


def read_in_place(text, name):
    """Whether the body reads `name` where the kernel was handed it."""
    return re.search(rf'\bglb_{name} = [^;]*&{name}\[', text) is not None


def meta(generator):
    line = generator.get_kernel().split('tensorforge-meta: ')[1].split('\n')[0]
    return json.loads(line)


def test_only_a_batch_constant_is_passed_by_value():
    with pytest.raises(GenerationError, match='passed as an argument'):
        tensor('A', Addressing.STRIDED, Residence.ARGUMENT)


@pytest.mark.parametrize('backend,arch', TARGETS,
                         ids=[f'{b}-{a}' for b, a in TARGETS])
@pytest.mark.parametrize('passed_first', [True, False], ids=['A', 'B'])
def test_the_kernel_takes_the_numbers(passed_first, backend, arch):
    """A struct of them, and nothing to offset into."""
    generator, (a, _, _) = product(passed_first, backend, arch)
    signature = generator.get_kernel().split('{')[0]
    assert f'tensorforge::ValueArray<float, {N * N}> {a.name}' in signature
    assert f'{a.name}_extraOffset' not in signature


def test_nvidia_keeps_them_in_the_constant_bank():
    generator, (a, _, _) = product(False)
    assert (f'__grid_constant__ const tensorforge::ValueArray<float, {N * N}> '
            f'{a.name}') in generator.get_kernel()


@pytest.mark.parametrize('arch', ['gfx942', 'gfx1150'])
def test_amd_reads_them_at_one_index_for_every_lane(arch):
    """A scalar to every product, not spread over the lanes by the DPP or
    MFMA paths: spread, the index is the lane's, and the struct is copied into
    private memory to be indexed at all.  Which space that reaches is left to
    the compiler -- cast into the constant space, the private copy faulted on
    gfx1150."""
    generator, (a, _, _) = product(False, 'hip', arch)
    text = code(generator)
    assert re.search(rf'\bglb_{a.name} = &{a.name}\[0\]', text)
    assert not re.search(rf'\bglb_{a.name}\[[^]]*_lead\b', text)


def test_amd_reads_a_batch_constant_in_memory_through_the_constant_space():
    """Where a uniform read is a scalar load: the constant path out of device
    memory, with no argument to pass."""
    m = tensor('A', Addressing.NONE)
    generator = generated([GemmDescr(False, False, SubTensor(tensor('B')),
                                     SubTensor(m), SubTensor(tensor('C')),
                                     alpha=1.0, beta=0.0)], 'hip', 'gfx1150',
                          preload_globals=False)
    text = code(generator)
    assert f'tensorforge::SpacePtr<const float, tensorforge::ConstantMemspace> {m.name}' \
        in generator.get_kernel().split('{')[0]
    assert re.search(rf'ConstantMemspace> const glb_{m.name} =', text)


def test_the_launcher_copies_them_out_of_host_memory():
    generator, (a, _, _) = product(False)
    assert f'const float *{a.name}' in generator.get_header()
    launcher = generator.get_launcher()
    assert (f'const auto {a.name}Arg = tensorforge::ValueArray<float, '
            f'{N * N}>::from({a.name});') in launcher
    assert re.search(rf'\b{a.name}Arg\b', launcher.split('<<<')[1])


def test_the_meta_says_so():
    generator, (a, b, _) = product(False)
    rows = {row['alias']: row for row in meta(generator)['operands']}
    assert rows['A']['residence'] == 'argument'
    assert 'residence' not in rows['B']


@pytest.mark.parametrize('passed_first', [True, False], ids=['A', 'B'])
def test_the_numbers_are_the_right_ones(passed_first):
    """Against numpy, through the host oracle, which fills the struct the
    kernel indexes as it fills any operand."""
    generator, (a, b, c) = product(passed_first)
    lanes, mults = kernel_eval.launch_geometry(generator.get_launcher())
    mem = kernel_eval.evaluate_wave(generator.get_kernel(), lanes, seed=3,
                                    globals_only=True, mults=mults)
    seed = kernel_eval.Slot(3)
    matrix = {t: np.array([[seed.read(t.name, i + j * N) for j in range(N)]
                           for i in range(N)]) for t in (a, b)}
    want = (matrix[a] @ matrix[b] if passed_first
            else matrix[b] @ matrix[a])
    got = np.array([[mem.get((c.name, i + j * N), np.nan) for j in range(N)]
                    for i in range(N)])
    assert np.max(np.abs(got - want)) < 1e-4 * np.max(np.abs(want))


def test_it_is_not_staged():
    """Already where arguments live, so `preload_globals` (AMD's default)
    leaves it alone -- and it still stages the same operand in memory."""
    passed, (a, _, _) = product(False, 'hip', 'gfx942')
    assert read_in_place(code(passed), a.name)
    assert not staged(code(passed), a.name)
    m = tensor('A', Addressing.NONE)
    inMemory = generated([GemmDescr(False, False, SubTensor(tensor('B')),
                                    SubTensor(m), SubTensor(tensor('C')),
                                    alpha=1.0, beta=0.0)], 'hip', 'gfx942')
    assert staged(code(inMemory), m.name)


def test_explicit_simd_refuses_it():
    with pytest.raises(GenerationError, match='explicit-SIMD'):
        product(False, 'esimd', 'pvc')


def test_more_than_the_arguments_hold_is_refused():
    with pytest.raises(GenerationError, match='more than'):
        product(False, size=32)


def test_the_limit_is_the_targets():
    """16 kB of numbers: CUDA 12.1 took the limit to 32 kB from Volta on, and
    sm_120 needs a toolchain that new; sm_86 is held to the older 4 kB."""
    product(False, 'cuda', 'sm_120', size=64)
    with pytest.raises(GenerationError, match='sm_86'):
        product(False, 'cuda', 'sm_86', size=64)


class TestPreloadRoles:
    """`preload_roles=broadcast` stages the operand no operation reads along
    the lead index and leaves the other one in global memory."""

    def kernel(self, roles):
        lead = tensor('L', Addressing.NONE)
        broadcast = tensor('R', Addressing.NONE)
        generator = generated(
            [GemmDescr(False, False, SubTensor(lead), SubTensor(broadcast),
                       SubTensor(tensor('C')), alpha=1.0, beta=0.0)],
            'hip', 'gfx942', preload_globals=True, preload_roles=roles)
        return code(generator), lead, broadcast

    def test_all_stages_both(self):
        text, lead, broadcast = self.kernel('all')
        assert staged(text, lead.name) and staged(text, broadcast.name)

    def test_broadcast_stages_only_the_broadcast_operand(self):
        text, lead, broadcast = self.kernel('broadcast')
        assert read_in_place(text, lead.name)
        assert not staged(text, lead.name)
        assert staged(text, broadcast.name)

    def test_lead_stages_only_the_lead_operand(self):
        text, lead, broadcast = self.kernel('lead')
        assert staged(text, lead.name)
        assert read_in_place(text, broadcast.name)
        assert not staged(text, broadcast.name)

    def test_an_unknown_role_is_refused(self):
        with pytest.raises(GenerationError, match='preload_roles'):
            self.kernel('b')


def constant(alias, size=N, spp=None):
    """A batch-constant operand in memory that carries its numbers."""
    values = np.arange(1, size * size + 1, dtype=np.float64).reshape(size, size)
    if spp is not None:
        values = np.where(np.vectorize(lambda i, j: spp.is_nz((i, j)))(
            *np.indices((size, size))), values, 0.0)
    return Tensor([size, size], Addressing.NONE,
                  BoundingBox([0, 0], [size, size]), alias=alias,
                  datatype=Datatype.F32, spp=spp, data=values)


class TestEmbedded:
    """`argument_constants`: a broadcast operand whose numbers the description
    carries is passed by value -- implicitly, the launcher keeps taking the
    pointer and passes the numbers instead."""

    def gemm(self, backend='cuda', arch='sm_86', role='B', size=N, spp=None,
             **options):
        k = constant('K', size, spp)
        x, c = tensor('X', size=size), tensor('C', size=size)
        first, second = (x, k) if role == 'B' else (k, x)
        generator = generated([GemmDescr(False, False, SubTensor(first),
                                         SubTensor(second), SubTensor(c),
                                         alpha=1.0, beta=0.0)],
                              backend, arch, **options)
        return generator, k

    @staticmethod
    def embedded(generator, k):
        return (f'ValueArray<float, {k.storage_volume()}> {k.name}'
                in generator.get_kernel().split('{')[0])

    @staticmethod
    def numbers(generator, k):
        found = re.search(rf'static const tensorforge::ValueArray<float, \d+> '
                          rf'{k.name}Arg\{{\{{([^}}]*)\}}\}}; \(void\){k.name};',
                          generator.get_launcher())
        return found.group(1).split(', ') if found else None

    def test_nvidia_passes_the_numbers(self):
        generator, k = self.gemm()
        assert self.embedded(generator, k)
        assert re.search(rf'const float \*{k.name}\b', generator.get_header())

    def test_the_numbers_are_in_storage_order(self):
        generator, k = self.gemm()
        want = [Datatype.F32.literal(v) for v in k.data.flatten(order='F')]
        assert self.numbers(generator, k) == want

    def test_a_sparse_operand_passes_its_entries_in_their_order(self):
        entries = [(i, j) for j in range(N) for i in range(N)
                   if abs(i - j) <= 1][::-1]
        generator, k = self.gemm(spp=ListSPP(entries, [N, N]))
        assert self.numbers(generator, k) == [
            Datatype.F32.literal(k.data[e]) for e in entries]

    def test_a_lead_operand_stays_in_memory(self):
        generator, k = self.gemm(role='A')
        assert not self.embedded(generator, k)

    def test_amd_reads_it_out_of_memory_through_the_constant_space(self):
        generator, k = self.gemm('hip', 'gfx942')
        assert not self.embedded(generator, k)
        assert f'ConstantMemspace> {k.name}' in generator.get_kernel()

    def test_switched_off_it_stays_in_memory(self):
        generator, k = self.gemm(argument_constants=False)
        assert not self.embedded(generator, k)

    def test_what_does_not_fit_stays_in_memory(self):
        """6.4 kB: over sm_86's 4 kB, so memory and no error; sm_120 takes it."""
        generator, k = self.gemm(size=40)
        assert not self.embedded(generator, k)
        generator, k = self.gemm(arch='sm_120', size=40)
        assert self.embedded(generator, k)

    def test_the_meta_says_so(self):
        generator, k = self.gemm()
        rows = {row['alias']: row for row in meta(generator)['operands']}
        assert rows['K'].get('embedded') is True
        assert 'embedded' not in rows['X']
