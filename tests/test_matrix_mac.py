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


# -- AMD's MFMA ---------------------------------------------------------------- #

from tensorforge.backend.instructions.compute.primitives.amd import (  # noqa: E402
    catalog, dpp_block, stacked)
from tensorforge.backend.pir.core import ScalarType, Uniformity  # noqa: E402

MFMA4 = next(o for o in catalog.MATRIX_OPS if o.builtin == 'mfma_f32_4x4x1f32')
MFMA64 = next(o for o in catalog.MATRIX_OPS
              if o.builtin == 'mfma_f64_16x16x4f64')


def _mfma(op, how='op', modifiers=(2, 1, 4)):
    """One MFMA on declared fragments: as the matrix op (`MatrixOp.mac`), or
    as the builtin call the AMD path wrote before it (`how='call'`)."""
    b = IRBuilder(fptype=op.d.dtype)
    acc = ScalarType(op.d.dtype, op.d.per_lane)
    a = b.declare(ScalarType(op.a.dtype, op.a.per_lane) if op.a.per_lane > 1
                  else ScalarType(op.a.dtype), hint='a')
    bb = b.declare(ScalarType(op.b.dtype, op.b.per_lane) if op.b.per_lane > 1
                   else ScalarType(op.b.dtype), hint='b')
    c = b.declare(acc, hint='acc')
    if how == 'op':
        d = op.mac(b, acc, a, bb, c, *modifiers)
    else:
        d = b.call(op.callee, acc, a, bb, c, *modifiers, hint='acc',
                   movable=False, materialize=True)
    return b, (a, bb, c), d


def _hip(arch='gfx942', dtype=Datatype.F32):
    return Context(arch=arch, backend='hip', fp_type=dtype, options=Options())


def test_an_mfma_is_a_matrix_op_with_a_result():
    """D is a new value, the statement pure and pinned, kept in a variable of
    its own; the instruction, its modifiers and the vendor's layouts are on
    it."""
    b, (a, bb, c), d = _mfma(MFMA4)
    s = _stmt(b.finish())
    assert s.target == (d,) and s.args == (a, bb, c)
    assert s.pure and not s.movable and s.effect == Effect.NONE
    assert not s.accesses and s.attr('no_inline')
    assert s.attr('engine') == 'mfma' and s.attr('instruction') == MFMA4.builtin
    assert s.attr('shape') == (4, 4, 1) and s.attr('fragments') == (1, 1, 1)
    assert s.attr('types') == (Datatype.F32,) * 4
    assert s.attr('modifiers') == (('cbsz', 2), ('abid', 1), ('blgp', 4))
    assert s.attr('collective') is Participants.WAVE
    assert s.attr('in_place') is None
    assert s.attr('layouts') == catalog.vendor_layouts(MFMA4)
    assert d.type == ScalarType(Datatype.F32, 4) and d.layout is None


def test_d_is_as_uniform_as_the_fragments():
    """D is a lane's own where a fragment is, as the call's result was."""
    b = IRBuilder(fptype=Datatype.F64)
    acc = ScalarType(Datatype.F64, MFMA64.d.per_lane)
    a = b.declare(ScalarType(Datatype.F64), hint='a', uniform=Uniformity.LANE)
    bb = b.declare(ScalarType(Datatype.F64), hint='b')
    c = b.declare(acc, hint='acc')
    d = MFMA64.mac(b, acc, a, bb, c)
    assert d.uniformity == Uniformity.LANE


@pytest.mark.parametrize('op', [MFMA4, MFMA64] + [
    next(o for o in catalog.MATRIX_OPS if o.builtin == name)
    for name in ('mfma_f32_16x16x1f32', 'mfma_f32_16x16x4f32',
                 stacked.EMULATED[Datatype.F32])], ids=lambda o: o.builtin)
def test_it_writes_what_the_builtin_call_wrote(op):
    """The declaration of D, its type spelled by the target's lexic, and the
    builtin with the fragments and the modifiers: the text of the call the
    AMD path issued before the op."""
    context = _hip(dtype=op.d.dtype)
    got = _emitted(_mfma(op)[0].finish(), context)
    want = _emitted(_mfma(op, how='call')[0].finish(), context)
    assert got == want
    assert f'= {op.callee}(' in got and '2, 1, 4);' in got


def test_a_lowering_with_a_result_is_declared_by_the_emitter():
    """`Lowered.result`: the expression the emitter declares D with."""
    b, _, d = _mfma(MFMA4)
    s = _stmt(b.finish())
    lowered = engines.lower(s, None, name=str)
    assert lowered.lines == () and lowered.result.startswith(
        '__builtin_amdgcn_mfma_f32_4x4x1f32(')


def test_an_mfma_without_its_result_is_not_written():
    b, (a, bb, c), d = _mfma(MFMA4)
    s = _stmt(b.finish())
    import dataclasses
    with pytest.raises(IRError, match='returns D'):
        engines.lower(dataclasses.replace(s, target=()), None, name=str)


def test_an_mfma_counts_under_its_shape_as_matrix_work():
    b, *_ = _mfma(MFMA4)
    metrics = _Metrics()
    _emitted(b.finish(), _hip(), metrics)
    assert metrics.mix == [('mfma.f32_4x4x1f32', 1, 1)]
    assert metrics.work == [(1, None)]


def test_the_category_is_the_instructions():
    assert engines.mfma_category('mfma_f64_16x16x4f64') == 'mfma.f64_16x16x4f64'
    assert engines.mfma_category('wmma_f32_16x16x16_f16') == 'mfma'


def test_every_mfma_the_paths_issue_is_priced():
    """Each instruction the lane-batched tiles, the stacked scheme and its
    emulation issue has its own clocks in the pipeline model, rather than the
    4x4x1 default an unknown shape takes."""
    from tensorforge.analysis.pipeline import MFMA_CLOCKS
    issued = ({t.op.builtin for t in catalog.MFMA_TILES
               if t.block in catalog.LANE_BATCHED_BLOCKS}
              | set(stacked.INSTRUCTIONS.values())
              | set(stacked.EMULATED.values()))
    for name in issued:
        assert engines.mfma_category(name)[len('mfma.'):] in MFMA_CLOCKS, name


def test_an_mfma_verifies():
    b, *_ = _mfma(MFMA4)
    assert not [d for d in verify(b.finish(), strict=False) if 'matrix' in d]


# -- the wait a DPP block owes an MFMA ------------------------------------------ #

def test_the_wait_behind_a_block_is_the_hazard_an_mfma_has():
    """A VALU write read by an MFMA as A or B: two wait states, `s_nop 1`."""
    assert dpp_block.HAZARD == engines.Hazard(then='mfma', states=2)
    assert dpp_block.WAIT_AFTER == 's_nop 1'


def test_a_kernel_with_an_mfma_waits_behind_its_blocks():
    b, *_ = _mfma(MFMA4)
    assert dpp_block.waits_behind([b.finish()])


def test_a_kernel_without_one_does_not():
    """Not for NVIDIA's matrix op either: its engine owes no such wait."""
    b, *_ = _issue(TF32)
    assert not dpp_block.waits_behind([b.finish()])
    plain = IRBuilder(fptype=Datatype.F32)
    plain.declare(hint='x')
    assert not dpp_block.waits_behind([plain.finish()])


def test_raw_text_that_spells_one_counts_too():
    b = IRBuilder(fptype=Datatype.F32)
    b('__builtin_amdgcn_mfma_f32_4x4x1f32(x, y, z, 0, 0, 0);')
    assert dpp_block.waits_behind([b.finish()])


def test_a_layout_derived_here_is_not_the_vendors():
    """gfx950's `mfma_f32_16x16x32_bf16` has layouts TensorForge derives
    from a precedent: none of them goes on the op as the vendor's."""
    op = next(o for o in catalog.MATRIX_OPS
              if o.builtin == 'mfma_f32_16x16x32_bf16')
    assert catalog.vendor_layouts(op) == (None, None, None)


def test_an_emulated_product_states_its_bf16_operands():
    op = next(o for o in catalog.MATRIX_OPS
              if o.builtin == stacked.EMULATED[Datatype.F32])
    b, *_ = _mfma(op, modifiers=(0, 0, 0))
    assert _stmt(b.finish()).attr('types') == (
        Datatype.BF16, Datatype.BF16, Datatype.F32, Datatype.F32)


# -- Intel's DPAS ---------------------------------------------------------------- #

from tensorforge.backend.instructions.compute.primitives import (  # noqa: E402
    intel, intel_spmd)
from tensorforge.backend.pir.core import SCALAR_LAYOUT, join_layout  # noqa: E402
from tensorforge.backend.pir.emit_esimd import EsimdEmitter  # noqa: E402


def _sycl(dpas=True):
    context = Context(arch='pvc', backend='oneapi', fp_type=Datatype.F32,
                      options=Options())
    context.target.lexic.dpas = dpas
    return context


def _spmd_dpas(how='op'):
    """One SPMD DPAS on declared fragments: as the matrix op
    (`intel_spmd.mac`), or as the call the SPMD path issued before it."""
    b = IRBuilder(fptype=Datatype.F32)
    a = b.declare(ScalarType(Datatype.F32, 4), hint='a')
    bb = b.declare(ScalarType(Datatype.F32, 8), hint='b')
    c = b.declare(ScalarType(Datatype.F32, 8), hint='dacc')
    v8 = ScalarType(Datatype.F32, 8)
    if how == 'op':
        d = intel_spmd.mac(b, v8, a, bb, c, (1, 0))
    else:
        d = b.call('tensorforge::dpasTF32', v8, 'item.get_sub_group()', a, bb,
                   c, hint='dacc', movable=False, materialize=True)
    return b, (a, bb, c), d


def test_spmd_dpas_is_a_matrix_op_with_a_result():
    """The sub-group of 16 issues it together; D a new value, pinned and in
    a variable of its own; the layouts are the measured ones."""
    b, (a, bb, c), d = _spmd_dpas()
    s = _stmt(b.finish())
    assert s.target == (d,) and s.args == (a, bb, c)
    assert s.attr('engine') == 'dpas_spmd'
    assert s.attr('shape') == (8, 16, 8)
    assert s.attr('types') == (Datatype.TF32, Datatype.TF32, Datatype.F32,
                               Datatype.F32)
    assert s.attr('term') == (1, 0) and s.attr('no_inline')
    assert s.attr('collective') is Participants.WAVE
    assert s.attr('layouts') == (intel_spmd.a_fragment_bits(),
                                 intel_spmd.b_fragment_bits(),
                                 intel_spmd.d_fragment_bits())
    assert s.pure and not s.movable and d.layout is None


def test_spmd_dpas_writes_what_the_call_wrote():
    """`NativeFloat8 d = tensorforge::dpasTF32(item.get_sub_group(), a, b,
    c);`, the sub-group as the lexic spells it."""
    got = _emitted(_spmd_dpas()[0].finish(), _sycl())
    want = _emitted(_spmd_dpas('call')[0].finish(), _sycl())
    assert got == want
    assert 'tensorforge::dpasTF32(item.get_sub_group(), ' in got


def test_spmd_dpas_needs_the_targets_sub_group():
    b, *_ = _spmd_dpas()
    with pytest.raises(IRError, match='sub-group'):
        engines.lower(_stmt(b.finish()), None, name=str)


def test_spmd_dpas_takes_the_products_in_their_order():
    """Smallest first (`split.products`): A's lower half by B's upper, A's
    upper by B's lower, then the upper halves -- `term` is (A's half, B's)."""
    assert [(q, p) for p, q in intel_spmd.PRODUCTS] == [(1, 0), (0, 1),
                                                       (0, 0)]


def _esimd_dpas(how='op'):
    """One ESIMD DPAS into an accumulator, as `dpas_matmul` issues it: the
    matrix op written into the assignment, or the raw expression it was."""
    atom = intel.ATOMS['tf32']
    b = IRBuilder(fptype=Datatype.F32)
    acc = b.declare(ScalarType(Datatype.F32, atom.c_elems), hint='dacc',
                    layout=SCALAR_LAYOUT)
    af = b.declare(ScalarType(Datatype.TF32, atom.a_elems), hint='ahi',
                   layout=SCALAR_LAYOUT)
    bf = b.declare(ScalarType(Datatype.TF32, atom.b_elems), hint='bhi',
                   layout=SCALAR_LAYOUT)
    if how == 'op':
        d = b.matrix_mac(
            'dpas_esimd', (atom.m, atom.n, atom.k),
            (Datatype.TF32, Datatype.TF32, Datatype.F32, Datatype.F32),
            (af,), (bf,), (acc,), result=acc.type, hint='dp',
            layout=join_layout((acc, bf, af)), materialize=False,
            term=(0, 0), modifiers=(('depth', atom.depth),))
    else:
        d = b.rawexpr(f'tensorforge::intel_xmx::dpas<{atom.depth}, '
                      f'{atom.repeat}, float>({{0}}, {{1}}, {{2}})',
                      acc, bf, af, type_=acc.type, hint='dp', pure=True)
    b.assign(acc, d)
    return b, d


def _esimd_emitted(body):
    lines = []
    context = Context(arch='pvc', backend='esimd', fp_type=Datatype.F32)
    EsimdEmitter(writer=lines.append, context=context, strict=False).run(body)
    return '\n'.join(lines)


def test_esimd_dpas_is_written_into_the_assignment():
    """`acc = dpas<8, 8, float>(acc, B, A)`: D is written into the one
    statement that reads it, as the raw expression was, under the explicit
    vector's declaration."""
    got = _esimd_emitted(_esimd_dpas()[0].finish())
    assert got == _esimd_emitted(_esimd_dpas('raw')[0].finish())
    assert 'tensorforge::intel_xmx::dpas<8, 8, float>(' in got


def test_esimd_dpas_is_the_work_items_alone():
    """The explicit vector's work-item is the sub-group: there is no
    collective to state, and D is free to be written into its reader."""
    b, d = _esimd_dpas()
    s = _stmt(b.finish())
    assert s.attr('engine') == 'dpas_esimd'
    assert s.attr('collective') is None and not s.attr('no_inline')
    assert s.attr('modifiers') == (('depth', 8),)
    assert d.layout == SCALAR_LAYOUT


@pytest.mark.parametrize('how', ['spmd', 'esimd'])
def test_dpas_counts_as_other_code_and_no_work(how):
    """The figures DPAS was counted under, as a call and as raw text: one
    unit of other code, no work."""
    metrics = _Metrics()
    if how == 'spmd':
        _emitted(_spmd_dpas()[0].finish(), _sycl(), metrics)
    else:
        b, _ = _esimd_dpas()
        context = Context(arch='pvc', backend='esimd', fp_type=Datatype.F32)
        EsimdEmitter(writer=lambda line: None, context=context,
                     metrics=metrics, strict=False).run(b.finish())
    assert ('other', 1, 1) in metrics.mix
    assert not [m for m in metrics.mix if m[0] == 'matrix']
    assert metrics.work == []


def test_a_matrix_op_ends_a_run_and_a_window():
    """As the call and the raw text it was: a block read does not reach
    across it."""
    from tensorforge.backend.pir import block_reads
    b, *_ = _spmd_dpas()
    s = _stmt(b.finish())
    assert block_reads._run_ends(s, MemSpace.GLOBAL)


def test_esimd_dpas_in_a_kernel_updates_its_accumulator():
    """`square_notrans` with `tensor_cores` under the explicit vector:
    every DPAS is the right-hand side of the assignment to the accumulator
    it reads as C, never a declaration of its own -- and three products of
    the halves for each of two column blocks and two depths."""
    import contextlib
    import importlib.util
    import io
    import re
    from pathlib import Path
    from tensorforge.generators.generator import Generator
    path = Path(__file__).parent / 'cases' / 'square_notrans.py'
    spec = importlib.util.spec_from_file_location('square_notrans', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with contextlib.redirect_stdout(io.StringIO()):
        gen = Generator(mod.descr_list(), Context(
            arch='pvc', backend='esimd', fp_type=Datatype.F32,
            options=Options(tensor_cores=True)))
        gen.generate()
    lines = [line.strip() for line in gen.get_kernel().splitlines()
             if 'intel_xmx::dpas<' in line]
    assert len(lines) == 2 * intel.TF32_TERMS * 2
    assignment = re.compile(r'(\w+) = \(tensorforge::intel_xmx::dpas<8, 8, '
                            r'float>\(\1, \w+, \w+\)\);$')
    assert all(assignment.match(line) for line in lines), lines
