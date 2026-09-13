# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""yateto's ternary, `where(condition, yes, no)`: yes, no, condition.

A condition with no axes is one value per batch element, so the choice is the
same for every entry: it is hoisted into the guards yateto already sends --
`result = yes` under `cond`, `result = no` under not `cond` -- and the branch
not taken is not computed at all.  A condition with axes chooses per entry and
is an elementwise `SELECT`.
"""

from __future__ import annotations

import contextlib
import io
import json
import pathlib
import re

import numpy as np
import pytest

import kernel_eval
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context
from tensorforge.common.operation import Operation
from tensorforge.frontend.yateto import DescriptionReader
from tensorforge.generators.descriptions import (ElementwiseDescr,
                                                 MultilinearDescr)
from tensorforge.generators.generator import Generator

N = 8


def tensor(name, shape, datatype='f32'):
    return {'name': name, 'addressing': 'n*N+o&', 'residence': 'memory',
            'datatype': datatype,
            'storage': {'shape': list(shape), 'type': 'bbox',
                        'start': [0] * len(shape), 'sizes': list(shape)},
            'values': None, 'alignment': 0,
            'flags': {'temporary': False, 'constant': False}}


def ref(name, indices='', shape=()):
    return {'name': name, 'indices': list(indices),
            'bbox': [[0] * len(shape), list(shape)] if shape else None,
            'offset': [0] * len(shape), 'offset_from': None, 'sliced': False}


def ternary(result='A', yes='B', no='C', cond=('c', '', ()), condition=(),
            add=False):
    name, indices, shape = cond
    return {'type': 'elementwise', 'result': ref(result, 'ij', (N, N)),
            'args': [ref(yes, 'ij', (N, N)), ref(no, 'ij', (N, N)),
                     ref(name, indices, shape)],
            'condition': list(condition),
            'linear': {'alpha': None, 'add': add}, 'optype': 'Ternary'}


def read(operation, extra=()):
    tensors = [tensor(n, (N, N)) for n in 'ABC'] + [
        tensor('c', (), 'bool'), tensor('m', (N, N), 'bool')] + list(extra)
    descrs, _ = DescriptionReader(None, {}).read(
        {'version': 7, 'tensors': tensors, 'operations': [operation]})
    return descrs


def generated(descrs):
    generator = Generator(descrs, Context(arch='sm_86', backend='cuda',
                                          fp_type=Datatype.F32))
    with contextlib.redirect_stdout(io.StringIO()):
        generator.generate()
    return generator


def matrix(seed, name):
    return np.array([[seed.read(name, i + j * N) for j in range(N)]
                     for i in range(N)])


class TestHoisted:
    def test_a_rank0_condition_becomes_two_guarded_copies(self):
        yes, no = read(ternary())
        assert isinstance(yes, MultilinearDescr)
        assert isinstance(no, MultilinearDescr)
        (lit_yes,), (lit_no,) = yes.condition, no.condition
        assert lit_yes.tensor.tensor is lit_no.tensor.tensor
        assert (lit_yes.negated, lit_no.negated) == (False, True)
        assert lit_yes.version == lit_no.version < 0

    def test_an_outer_guard_covers_both_halves(self):
        outer = {'tensor': ref('d'), 'version': 0, 'negated': False}
        yes, no = read(ternary(condition=[outer]),
                       extra=[tensor('d', (), 'bool')])
        assert len(yes.condition) == len(no.condition) == 2
        assert yes.condition[0].tensor.tensor is no.condition[0].tensor.tensor

    def test_two_hoists_do_not_share_a_version(self):
        first = DescriptionReader(None, {})
        tensors = [tensor(n, (N, N)) for n in 'ABC'] + [tensor('c', (), 'bool')]
        descrs, _ = first.read({'version': 7, 'tensors': tensors,
                                'operations': [ternary(), ternary()]})
        assert descrs[0].condition[0].version != descrs[2].condition[0].version

    def test_accumulating_accumulates_in_both(self):
        yes, no = read(ternary(add=[0, 1]))
        assert yes.add and no.add

    def test_a_condition_the_ternary_writes_is_not_hoisted(self):
        """The second half would read it after the first had written it."""
        op = ternary(result='A', cond=('A', '', ()))
        op['result'] = ref('c')
        op['args'][2] = ref('c')
        for arg in op['args'][:2]:
            arg.update(indices=[], bbox=None, offset=[])
        descrs = DescriptionReader(None, {}).read(
            {'version': 7,
             'tensors': [tensor('B', ()), tensor('C', ()),
                         tensor('c', (), 'bool')],
             'operations': [op]})[0]
        assert [d.op for d in descrs if isinstance(d, ElementwiseDescr)] \
            == [Operation.SELECT]

    @pytest.mark.parametrize('flag', [0.0, 1.0])
    def test_the_branch_taken_is_the_one_computed(self, flag):
        descrs = read(ternary())
        generator = generated(descrs)
        a, b, c = (descrs[0].dest.tensor, descrs[0].ops[0].tensor,
                   descrs[1].ops[0].tensor)
        cond = descrs[0].condition[0].tensor.tensor
        lanes, mults = kernel_eval.launch_geometry(generator.get_launcher())
        mem = kernel_eval.evaluate_wave(generator.get_kernel(), lanes, seed=4,
                                        globals_only=True, mults=mults,
                                        preset={cond.name: flag})
        seed = kernel_eval.Slot(4)
        want = matrix(seed, b.name if flag else c.name)
        got = np.array([[mem.get((a.name, i + j * N), np.nan)
                         for j in range(N)] for i in range(N)])
        assert np.allclose(got, want)


class TestSelect:
    def test_a_condition_with_axes_selects_per_entry(self):
        (descr,) = read(ternary(cond=('m', 'ij', (N, N))))
        assert isinstance(descr, ElementwiseDescr)
        assert descr.op is Operation.SELECT
        assert len(descr.srcs) == 3

    def test_it_is_emitted_as_a_select(self):
        generator = generated(read(ternary(cond=('m', 'ij', (N, N)))))
        assert ' ? ' in generator.get_kernel()


FIXTURE = pathlib.Path(__file__).parent / 'fixtures' / 'kernels' / 'ternary.json'
RECORDED = ['ternary_rank0', 'ternary_tensor', 'ternary_written',
            'ternary_literal', 'ternary_rank0_t', 'ternary_tensor_t',
            'ternary_written_t', 'ternary_literal_t', 'ternary_rank0_s',
            'ternary_written_s', 'ternary_written_any']


def recorded(name, strided=False):
    """The descriptors TensorForge reads from what yateto sent.  `strided`
    addresses the batch operands by a stride instead of through an array of
    pointers, as yateto sends them -- the host oracle follows no pointer
    arrays; nothing about the operations or their guards changes."""
    description = json.loads(FIXTURE.read_text())['descriptions'][name]
    if strided:
        for tensor in description['tensors']:
            if tensor['addressing'] == 'n&+o&':
                tensor['addressing'] = 'n*N+o&'
    descrs, _ = DescriptionReader(None, {}).read(description)
    return descrs


class TestRecorded:
    """What yateto sent for `where` (seissol/yateto 50200d3, interface 7)."""

    def test_a_rank0_condition_is_hoisted(self):
        descrs = recorded('ternary_rank0')
        assert [type(d).__name__ for d in descrs] == ['MultilinearDescr'] * 2
        assert [d.condition[-1].negated for d in descrs] == [False, True]

    def test_a_tensor_condition_selects(self):
        assert [d.op for d in recorded('ternary_tensor')] \
            == [Operation.GE, Operation.SELECT]

    def test_a_condition_the_kernel_computes_guards_what_follows(self):
        hoisted = [d for d in recorded('ternary_written') if d.condition]
        assert len(hoisted) == 2
        assert hoisted[0].condition[0].tensor.tensor \
            is hoisted[1].condition[0].tensor.tensor

    def test_a_branch_that_stores_nothing_is_the_number_zero(self):
        select = recorded('ternary_literal')[-1]
        assert select.op is Operation.SELECT
        assert select.srcs[1] == 0

    @pytest.mark.parametrize('name', RECORDED)
    @pytest.mark.parametrize('backend,arch', [('cuda', 'sm_86'),
                                              ('hip', 'gfx1150'),
                                              ('hip', 'gfx942')])
    def test_it_is_built(self, name, backend, arch):
        descrs = recorded(name)
        generator = Generator(descrs, Context(
            arch=arch, backend=backend,
            fp_type=descrs[-1].dest.tensor.datatype))
        with contextlib.redirect_stdout(io.StringIO()):
            generator.generate()
        assert generator.get_kernel()

    @pytest.mark.parametrize('flag', [0.0, 1.0])
    def test_the_hoisted_branches_compute_the_right_numbers(self, flag):
        """`A = X ? B : C^T`, through the host oracle, either way."""
        descrs = recorded('ternary_rank0_t', strided=True)
        generator = generated(descrs)
        a, b = descrs[0].dest.tensor, descrs[0].ops[0].tensor
        c = descrs[1].ops[0].tensor
        cond = descrs[0].condition[-1].tensor.tensor
        lanes, mults = kernel_eval.launch_geometry(generator.get_launcher())
        mem = kernel_eval.evaluate_wave(generator.get_kernel(), lanes, seed=6,
                                        globals_only=True, mults=mults,
                                        preset={cond.name: flag})
        seed = kernel_eval.Slot(6)
        want = matrix(seed, b.name) if flag else matrix(seed, c.name).T
        got = np.array([[mem.get((a.name, i + j * N), np.nan)
                         for j in range(N)] for i in range(N)])
        assert np.allclose(got, want)


@pytest.mark.parametrize('backend,arch,fence', [
    ('cuda', 'sm_86', '__syncwarp'),
    ('hip', 'gfx1150', '__builtin_amdgcn_fence'),
    ('hip', 'gfx942', '__builtin_amdgcn_fence')])
def test_the_guard_waits_for_the_owner_lane_to_store_its_condition(
        backend, arch, fence):
    """`X1 = all(B >= C^T)` is one number: the owner lane stores it, and the
    guard over it is read by every lane.  With nothing in between, sm_120
    took both branches in one element, and so did gfx1150 -- where the
    rendezvous of a wave is no instruction at all, and LLVM hoisted the
    other lanes' load above the owner's store until a fence stood there."""
    descrs = recorded('ternary_written_t')
    generator = Generator(descrs, Context(arch=arch, backend=backend,
                                          fp_type=Datatype.F32))
    with contextlib.redirect_stdout(io.StringIO()):
        generator.generate()
    lines = [line.strip() for line in generator.get_kernel().splitlines()]
    store = next(i for i, line in enumerate(lines)
                 if re.match(r'glb_\w+\[0\] = v\w+_red;', line))
    load = next(i for i, line in enumerate(lines)
                if i > store and re.match(r'bool v\w+ = glb_\w+\[0\];', line))
    assert any(fence in line for line in lines[store:load])


@pytest.mark.parametrize('backend,arch', [('cuda', 'sm_86'), ('hip', 'gfx1150')])
def test_a_boolean_staged_in_shared_memory_is_a_window_of_its_own_type(
        backend, arch):
    """The comparison `B >= C^T` is staged in the arena, which is an array of
    the kernel's float: `bool *s = &arena[0]` did not compile anywhere; the
    window is a reinterpret of the arena's address."""
    descrs = recorded('ternary_tensor_t')
    generator = Generator(descrs, Context(arch=arch, backend=backend,
                                          fp_type=Datatype.F32))
    with contextlib.redirect_stdout(io.StringIO()):
        generator.generate()
    kernel = generator.get_kernel()
    bools = [line for line in kernel.splitlines()
             if line.strip().startswith('bool') and 'ShrMem' in line]
    assert bools
    assert all('reinterpret_cast<bool*>' in line for line in bools)
