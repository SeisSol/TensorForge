# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The matrix op (`Op.MATRIX_MAC`): D = A * B + C on the fragments of a matrix
engine, one entry of the engines registry (`engines.MATRIX_ENGINES`).

The op states what the text used to: which engine, the shape, the element
types, the register counts of the fragments, whether C is written in place,
the fragment layouts the vendor has, the term of a split product and the
collective that executes it.  The metrics read it -- the issue category, the
work and the instructions an FP64 `mma.sync` is on sm_100 -- and the
registry writes it, byte for byte what the inline assembly it replaces
wrote.

NVIDIA's `mma.sync` and DMMA are the first engines on it.
"""

from __future__ import annotations

import pytest

from tensorforge.backend.instructions.compute.primitives import nvidia
from tensorforge.backend.pir import engines
from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import (Effect, IRError, MemSpace, Op,
                                          Participants, walk)
from tensorforge.backend.pir.emit import Emitter
from tensorforge.backend.pir.passes import verify
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context, Options
from tensorforge.common.exceptions import InternalError

TF32 = next(i for i in nvidia.INSTRS if i.name.startswith(
    'mma.sync.aligned.m16n8k8.row.col.f32'))
DMMA = next(i for i in nvidia.INSTRS if i.name.startswith(
    'mma.sync.aligned.m8n8k4.row.col.f64'))
DMMA16 = next(i for i in nvidia.INSTRS if i.name.startswith(
    'mma.sync.aligned.m16n8k16.row.col.f64'))


def _regs(atom):
    """The registers a lane holds of A, B and C: a tile over the warp."""
    return (atom.m * atom.k // 32, atom.k * atom.n // 32,
            atom.m * atom.n // 32)


def _fragments(atom):
    """A builder and the fragments of `atom` declared in it: the registers a
    lane holds of A, B and C."""
    b = IRBuilder(fptype=atom.d)
    na, nb, nc = _regs(atom)
    a = [b.declare(hint='a') for _ in range(na)]
    bb = [b.declare(hint='b') for _ in range(nb)]
    c = [b.declare(hint='c') for _ in range(nc)]
    return b, a, bb, c


def _issue(atom, term=None):
    """One instruction of `atom` on declared fragments, as the NVIDIA path
    issues it (`MMAInstr.asmcall`)."""
    b, a, bb, c = _fragments(atom)
    atom.asmcall(b, c, a, bb, c, term=term)
    return b, a, bb, c


def _stmt(body):
    return next(s for s, _ in walk(body) if s.op == Op.MATRIX_MAC)


def _emitted(body, context=None, metrics=None):
    lines = []
    Emitter(lines.append, context, metrics).run(body)
    return '\n'.join(lines)


# -- the registry -------------------------------------------------------------- #

def test_the_op_is_an_entry_of_the_registry():
    spec = engines.spec_of(Op.MATRIX_MAC)
    assert spec is not None and spec.results == ('d',)
    assert spec.completes is engines.Completion.SYNC
    assert not spec.movable and spec.retires is None and not spec.settles


@pytest.mark.parametrize('name', ['mma_sync', 'dmma'])
def test_nvidia_engines_are_warp_collectives_counted_as_matrix_work(name):
    engine = engines.MATRIX_ENGINES[name]
    assert engine.collective is Participants.WAVE
    assert engine.work


@pytest.mark.parametrize('atom', nvidia.INSTRS, ids=lambda i: i.name)
def test_every_instruction_is_named_by_its_shape_and_types(atom):
    """The PTX name the op is written with is derived, and is the table's."""
    assert engines.mma_sync_name((atom.m, atom.n, atom.k),
                                 atom.types) == atom.name


@pytest.mark.parametrize('atom,engine', [(TF32, 'mma_sync'), (DMMA, 'dmma')],
                         ids=['tf32', 'f64'])
def test_an_instruction_is_its_engine(atom, engine):
    assert atom.engine == engine


# -- the statement ------------------------------------------------------------- #

def test_mma_sync_is_a_matrix_op_in_place():
    """C read and written in place: no value allocated, a read and a write of
    each of C's registers declared, and nothing else."""
    b, a, bb, c = _issue(TF32, term=(0, 1))
    s = _stmt(b.finish())
    assert s.target == ()
    assert s.args == tuple(a + bb + c)
    assert s.attr('engine') == 'mma_sync'
    assert s.attr('shape') == (16, 8, 8)
    assert s.attr('types') == (Datatype.TF32, Datatype.TF32, Datatype.F32,
                               Datatype.F32)
    assert s.attr('fragments') == (4, 2, 4)
    assert s.attr('in_place') is True
    assert s.attr('term') == (0, 1)
    assert s.attr('collective') is Participants.WAVE
    assert s.effect == Effect.WRITE and not s.pure and not s.movable
    assert [(x.base, x.space, x.kind) for x in s.accesses] == [
        (v, MemSpace.REGISTER, Effect.READ | Effect.WRITE) for v in c]


def test_the_accumulators_are_pinned():
    """C is written through its name, so its producer escapes: no pass may
    take the name away underneath the instruction (`IRBuilder.pin`).  A and
    B are only read, and theirs do not."""
    from tensorforge.backend.pir.core import ScalarType
    b = IRBuilder(fptype=Datatype.F64)
    f64 = ScalarType(Datatype.F64)
    x, y, z = (b.declare(hint=h) for h in 'xyz')
    # Four distinct sums: the builder shares a pure operation on equal
    # operands.
    a, bb, c0, c1 = (b.op('add', f64, p, q, hint=h) for p, q, h in (
        (x, y, 'a'), (x, z, 'b'), (y, z, 'c'), (x, x, 'c')))
    DMMA.asmcall(b, [c0, c1], [a], [bb], [c0, c1])
    producers = {s.target[0].id: s for s, _ in walk(b.finish()) if s.target}
    assert producers[c0.id].attr('escapes') and producers[c1.id].attr(
        'escapes')
    assert not producers[a.id].attr('escapes')
    assert not producers[bb.id].attr('escapes')


def test_the_layouts_are_the_ptx_ones():
    """A and D as PTX lays them out; B is implicit in the NVIDIA path."""
    b, *_ = _issue(DMMA16)
    s = _stmt(b.finish())
    assert s.attr('layouts') == (nvidia.a_fragment_bits(DMMA16), None,
                                 nvidia.d_fragment_bits(DMMA16))


def test_a_full_product_has_no_term():
    b, *_ = _issue(DMMA)
    assert _stmt(b.finish()).attr('term') is None


def test_the_terms_are_the_products_in_their_order():
    """3xTF32: the upper halves, the upper by B's lower, A's lower by B's
    upper -- in the order the products are issued."""
    assert nvidia.TF32_TERMS == ((0, 0), (0, 1), (1, 0))
    assert [TF32.term(i) for i in range(3)] == list(nvidia.TF32_TERMS)
    assert DMMA.term(0) is None


def test_the_accumulator_has_to_be_written_in_place():
    b = IRBuilder(fptype=Datatype.F64)
    a, bb, c, d = (b.declare(hint=h) for h in 'abcd')
    with pytest.raises(InternalError, match='in place'):
        DMMA.asmcall(b, [d, d], [a], [bb], [c, c])


def test_a_written_register_is_a_value():
    b = IRBuilder(fptype=Datatype.F64)
    a, bb = b.declare(hint='a'), b.declare(hint='b')
    with pytest.raises(IRError, match='not a value'):
        b.matrix_mac('dmma', (8, 8, 4), (Datatype.F64,) * 4, [a], [bb],
                     ['acc[0]', 'acc[1]'])


def test_an_engine_it_does_not_know_is_refused():
    b = IRBuilder(fptype=Datatype.F32)
    c = b.declare(hint='c')
    with pytest.raises(IRError, match='engine'):
        b.matrix_mac('wgmma', (64, 8, 8), (Datatype.TF32,) * 4, [], [], [c])


def test_the_op_verifies():
    b, *_ = _issue(TF32, term=(0, 0))
    assert not [d for d in verify(b.finish(), strict=False)
                if 'matrix' in d]


def test_wrong_fragment_counts_are_reported():
    b, *_ = _issue(DMMA)
    s = _stmt(b.finish())
    bad = s.with_attr('fragments', (1, 1, 1))
    assert any('fragments' in d for d in engines.verify_engine(bad))


def test_c_written_without_its_accesses_is_reported():
    import dataclasses
    b, *_ = _issue(DMMA)
    s = _stmt(b.finish())
    bad = dataclasses.replace(s, accesses=s.accesses[:1])
    assert any('read and a write' in d for d in engines.verify_engine(bad))


def test_in_place_it_writes_and_otherwise_it_does_not():
    assert engines.effect_of(Op.MATRIX_MAC, (('in_place', True),)) \
        == Effect.WRITE
    assert engines.effect_of(Op.MATRIX_MAC, ()) == Effect.NONE


# -- the text ------------------------------------------------------------------ #

def _asm_reference(atom):
    """The same instruction as inline assembly (`IRBuilder.asm_stmt`), the
    form the NVIDIA path wrote before the op: the text the op has to write."""
    b, a, bb, c = _fragments(atom)
    f, r = ('d', 'd') if atom.d == Datatype.F64 else ('f', 'r')
    group = lambda n, first: '{' + ','.join(f'%{i + first}'
                                            for i in range(n)) + '}'
    dgrp = group(len(c), 0)
    template = (f'"{atom.name} "\n'
                f'"{dgrp}, {group(len(a), len(c))}, '
                f'{group(len(bb), len(c) + len(a))}, {dgrp};"')
    b.asm_stmt(template, [(f'+{f}', v) for v in c] + [(r, v) for v in a]
               + [(r, v) for v in bb])
    return b.finish()


@pytest.mark.parametrize('atom', [i for i in nvidia.INSTRS
                                  if i.mode in nvidia.EMITTED_MODES],
                         ids=lambda i: i.name)
def test_it_writes_what_the_inline_assembly_wrote(atom):
    b, *_ = _issue(atom)
    assert _emitted(b.finish()) == _emitted(_asm_reference(atom))


def test_the_text_of_an_fp64_instruction():
    b, a, bb, c = _issue(DMMA)
    text = _emitted(b.finish())
    assert text[text.index('asm('):] == (
        f'asm("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "\n'
        f'"{{%0,%1}}, {{%2}}, {{%3}}, {{%0,%1}};"\n'
        f': "+d"({c[0]}), "+d"({c[1]})\n: "d"({a[0]}), "d"({bb[0]})\n);')


def test_an_accumulator_of_another_type_is_not_written():
    b = IRBuilder(fptype=Datatype.F32)
    a, bb, c = b.declare(hint='a'), b.declare(hint='b'), b.declare(hint='c')
    b.matrix_mac('mma_sync', (8, 8, 16), (Datatype.I8, Datatype.I8,
                                          Datatype.I32, Datatype.I32),
                 [a], [bb], [c, c])
    with pytest.raises(IRError, match='not written'):
        _emitted(b.finish())


# -- the figures --------------------------------------------------------------- #

class _Metrics:
    def __init__(self):
        self.mix, self.code, self.work = [], 0, []

    def record_mix(self, category, issued, laid):
        self.mix.append((category, issued, laid))

    def record_code(self, units):
        self.code += units

    def record_work(self, work, flops):
        self.work.append((work, flops))


@pytest.mark.parametrize('arch,atom,issues', [
    ('sm_90', DMMA16, 1), ('sm_100', DMMA16, 8), ('sm_100', DMMA, 1),
    ('sm_120', DMMA16, 8), ('sm_100', TF32, 1)],
    ids=['dmma-sm90', 'dmma-sm100', 'dmma-m8n8k4-sm100', 'dmma-sm120',
         'tf32-sm100'])
def test_an_instruction_is_counted_as_the_ones_it_issues(arch, atom, issues):
    """FP64 `mma.sync` is `DMMA.8x8x4` whatever its shape from sm_100 on:
    the issue mix and the code count it as so many; everything else is one."""
    b, *_ = _issue(atom)
    metrics = _Metrics()
    context = Context(arch=arch, backend='cuda', fp_type=atom.d,
                      options=Options())
    _emitted(b.finish(), context, metrics)
    assert metrics.mix == [('matrix', issues, issues)]
    assert metrics.code == issues
    assert metrics.work == [(1, None)]
