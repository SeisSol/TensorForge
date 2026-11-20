# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""wgmma on the matrix op (`Op.MATRIX_MAC`, engine `wgmma`): NVIDIA's
asynchronous product of a warpgroup, sm_90a.

A and B are read from shared memory through descriptors (`Op.MATRIX_DESC`,
`engines.MatrixDescType`), A also from registers; D stays in registers,
written in place.  The product completes by group: a fence of the registers
ahead of it (`FenceKind.REGISTERS`), a commit behind it (`Op.COMMIT_GROUP`),
and a wait on the engine's counter (`engines.matrix_wait`) that retires its
group, behind which D is valid.  What is checked here:

* the vocabulary: the engine, the warpgroup, the commit, the fences;
* the descriptor's bits and the canonical layouts it states;
* the statement the builder makes, its effects, and what the verifier
  holds it to;
* the text each op is written as, and what it is counted as;
* the host oracle (`reference.kernel_eval`), whose layouts are PTX's as the
  ISA states them: the products the bodies here compute, D landing at the
  wait, a read of D ahead of it and a write of an operand in flight;
* the arch-specific targets and the build for them: `sm_90a`, and the SASS
  of a body compiled for it (local `nvcc`);
* on an sm_90 GPU, the same bodies run, against NumPy.
"""

from __future__ import annotations

import contextlib
import dataclasses
import re
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from tensorforge import toolchain
from tensorforge.backend.instructions.compute.primitives import nvidia
from tensorforge.backend.pir import engines
from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import (BOOL, INDEX, Effect, IRError,
                                          MemSpace, Op, Participants,
                                          ScalarType, Uniformity, walk)
from tensorforge.backend.pir.emit import Emitter
from tensorforge.backend.pir.engines import (Completion, FenceKind,
                                             MatrixDescType, Source)
from tensorforge.backend.pir.passes import verify
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context
from tensorforge.reference import kernel_eval

INCLUDE = Path(__file__).resolve().parents[1] / 'src' / 'tensorforge' / 'include'

#: The no-swizzle K-major layout the bodies here stage A and B in: the K
#: halves of a row 128 bytes apart (the leading byte offset), the groups of
#: eight rows 256 apart (the stride byte offset) -- GH200's reading
#: (`~/tf-probe/tcgen05/gh`, `wg_selftest`).
PLAIN = MatrixDescType(Datatype.TF32, lbo=128, sbo=256)
#: The 128-byte swizzle: rows of 128 bytes, 32 TF32 of K, atoms 1024 apart.
SW128 = MatrixDescType(Datatype.TF32, swizzle=128, lbo=16, sbo=1024)


def _ctx(arch='sm_90a', backend='cuda'):
    return Context(arch=arch, backend=backend, fp_type=Datatype.F32)


def _stmts(body, op):
    return [s for s, _ in walk(body) if s.op == op]


class _Sink:
    """The emitter's writer, keeping the lines and the blocks."""

    def __init__(self):
        self.lines = []

    def __call__(self, text, *args, **kwargs):
        self.lines.append(text)

    def __getattr__(self, _):
        return lambda *a, **k: None

    @contextlib.contextmanager
    def _block(self, head):
        self.lines.append(f'{head} {{')
        yield
        self.lines.append('}')

    def If(self, cond):
        return self._block(f'if ({cond})')

    def Else(self):
        return self._block('else')

    def Block(self, head=''):
        return self._block(head)


class _Metrics:
    def __init__(self):
        self.mix, self.code, self.work = [], 0, []

    def record_mix(self, category, issued, laid):
        self.mix.append((category, issued, laid))

    def record_code(self, units):
        self.code += units

    def record_work(self, work, flops):
        self.work.append((work, flops))


def _emit(body, ctx=None, metrics=None):
    sink = _Sink()
    Emitter(sink, ctx if ctx is not None else _ctx(), metrics).run(body)
    return sink.lines


# -- the bodies ------------------------------------------------------------- #

@dataclasses.dataclass
class Case:
    """A body of one warpgroup -- 128 threads, one multiplication -- that
    stages its operands from global memory into shared memory, multiplies
    them on wgmma and stores D back: the shape, where A is, the layout in
    shared memory, how many products and groups.

    `products` K steps of 8, each one product: A is 64 x 8K and B is n x 8K
    in global memory, row-major; D = A B^T, 64 x n.  `groups` > 1 commits
    each product as a group of its own and waits for all but the newest
    behind each, the last wait for none.  `scale` a value: scale-d is
    computed from the lane rather than written as a literal."""
    n: int = 8
    rs: bool = False
    desc: MatrixDescType = PLAIN
    products: int = 1
    groups: int = 1
    scale: bool = False

    @property
    def k(self):
        return 8 * self.products

    @property
    def a_elems(self):
        return 64 * self.k

    @property
    def b_elems(self):
        return self.n * self.k

    def step_offset(self, rows, step):
        """Where step `step` of an operand of `rows` rows starts, in
        elements: the next 32 bytes of the atom's rows where swizzled, the
        next block of rows x 8 where not."""
        if self.desc.swizzle:
            return 8 * step
        return step * 8 * rows


def _stage(b, case, dst, src, rows, lane):
    """Every lane copies its elements of the `rows` x `case.k` operand
    `src` (row-major) to where the layout puts them in `dst`."""
    total = rows * case.k
    for i in range((total + 127) // 128):
        g = b.op('add', INDEX, lane, 128 * i, hint='g') if i else lane
        r = b.op('div', INDEX, g, case.k, hint='r')
        k = b.op('rem', INDEX, g, case.k, hint='k')
        if case.desc.swizzle:
            # (r / 8) sbo + (r % 8) 128 + 4 k, its chunk XORed with r % 8.
            byte = b.op('add', INDEX, b.op('mul', INDEX, b.op('rem', INDEX, r, 8),
                                           case.desc.swizzle),
                        b.op('mul', INDEX, k, 4))
            chunk = b.op('bitxor', INDEX, b.op('div', INDEX, byte, 16),
                         b.op('rem', INDEX, r, 8))
            within = b.op('add', INDEX, b.op('mul', INDEX, chunk, 16),
                          b.op('rem', INDEX, byte, 16))
            at = b.op('add', INDEX,
                      b.op('mul', INDEX, b.op('div', INDEX, r, 8),
                           case.desc.sbo // 4),
                      b.op('div', INDEX, within, 4), hint='at')
        else:
            step = b.op('div', INDEX, k, 8)
            kk = b.op('rem', INDEX, k, 8)
            core = b.op('add', INDEX,
                        b.op('add', INDEX,
                             b.op('mul', INDEX, b.op('div', INDEX, r, 8),
                                  case.desc.sbo // 4),
                             b.op('mul', INDEX, b.op('div', INDEX, kk, 4),
                                  case.desc.lbo // 4)),
                        b.op('add', INDEX,
                             b.op('mul', INDEX, b.op('rem', INDEX, r, 8), 4),
                             b.op('rem', INDEX, kk, 4)))
            at = b.op('add', INDEX, b.op('mul', INDEX, step, 8 * rows), core,
                      hint='at')
        if total % 128 and i == total // 128:
            with b.if_(b.op('lt', BOOL, g, total, hint='in')):
                b.store(dst, b.load(src, g, hint='x'), at)
        else:
            b.store(dst, b.load(src, g, hint='x'), at)


def _wgmma_body(case: Case):
    """`case` built: (context, body, the shared arena in floats)."""
    ctx = _ctx()
    b = IRBuilder(fptype=Datatype.F32, context=ctx)
    align = engines.swizzle_alignment(case.desc.swizzle) // 4
    a_at = 0
    b_at = -(-case.a_elems // align) * align
    arena = b_at + case.b_elems
    sa = b.alloc(Datatype.F32, (case.a_elems,), MemSpace.SHARED, hint='sa',
                 extern='sa', arena='blk', offset=a_at,
                 place_align=align if case.desc.swizzle else None)
    sb = b.alloc(Datatype.F32, (case.b_elems,), MemSpace.SHARED, hint='sb',
                 extern='sb', arena='blk', offset=b_at,
                 place_align=align if case.desc.swizzle else None)
    ga = b.alloc(Datatype.F32, (case.a_elems,), MemSpace.GLOBAL, hint='ga',
                 extern='m1')
    gb = b.alloc(Datatype.F32, (case.b_elems,), MemSpace.GLOBAL, hint='gb',
                 extern='m2')
    out = b.alloc(Datatype.F32, (64 * case.n,), MemSpace.GLOBAL, hint='o',
                  extern='m0')
    lane = b.thread_id('x')
    if not case.rs:
        _stage(b, case, sa, ga, 64, lane)
    _stage(b, case, sb, gb, case.n, lane)
    b.fence(FenceKind.PROXY)
    b.barrier('block')
    d = [b.declare(hint='d', uniform=Uniformity.LANE)
         for _ in range(case.n // 2)]
    w = b.op('div', INDEX, lane, 32, hint='w')
    gq = b.op('div', INDEX, b.op('rem', INDEX, lane, 32), 4, hint='gq')
    tq = b.op('rem', INDEX, lane, 4, hint='tq')
    reads = []
    for step in range(case.products):
        bd = b.matrix_desc(sb, case.step_offset(case.n, step), desc=case.desc,
                           hint='bd')
        if case.rs:
            a = []
            for i in range(4):
                row = b.op('add', INDEX, b.op('add', INDEX,
                                              b.op('mul', INDEX, w, 16), gq),
                           8 * (i % 2))
                col = b.op('add', INDEX, tq, 4 * (i // 2) + 8 * step)
                x = b.load(ga, b.op('add', INDEX,
                                    b.op('mul', INDEX, row, case.k), col),
                           hint='ax')
                a.append(b.rawexpr('__float_as_uint({0})', x,
                                   type_=ScalarType(Datatype.TF32), hint='a'))
        else:
            a = [b.matrix_desc(sa, case.step_offset(64, step), desc=case.desc,
                               hint='ad')]
        if step == 0 or case.groups > 1:
            b.fence(FenceKind.REGISTERS, engine='wgmma',
                    registers=d + (a if case.rs else []))
        if case.scale:
            # 0 for the first product, 1 behind it: a value the builder
            # cannot fold into the literal, and one the multiplication's
            # lanes agree on (`wgmma_refusal`).
            scale = b.op('lt', BOOL, b.thread_id('y'), step, hint='acc')
        else:
            scale = 0 if step == 0 else 1
        b.wgmma(case.n, a, bd, d, scale_d=scale, layouts=(
            nvidia.wgmma_a_fragment_bits() if case.rs else None, None,
            nvidia.wgmma_d_fragment_bits(case.n)))
        reads += [bd] + ([] if case.rs else a)
        if case.groups > 1 or step == case.products - 1:
            b.commit_group('wgmma')
            last = step == case.products - 1
            b.matrix_wait('wgmma', 0 if last else 1,
                          registers=d + (a if case.rs else []),
                          reads=reads if last else reads[:-2 if not case.rs
                                                         else -1])
    for i in range(case.n // 2):
        row = b.op('add', INDEX, b.op('add', INDEX, b.op('mul', INDEX, w, 16),
                                      gq), 8 * ((i % 4) // 2))
        col = b.op('add', INDEX, b.op('mul', INDEX, tq, 2),
                   8 * (i // 4) + i % 2)
        b.store(out, d[i], b.op('add', INDEX, b.op('mul', INDEX, row, case.n),
                                col))
    return ctx, b.finish(), arena


def _oracle_source(lines):
    keep = [line for line in lines if not re.match(r'^float m\d+\[', line)]
    return ('void k() {\n'
            '  auto* blk = reinterpret_cast<float*>(totalShrMemPtr);\n'
            + ''.join(f'  {line}\n' for line in keep) + '}')


def _expected(case, a, b):
    """D = A B^T over the TF32 operands, summed in double: A as `case.k`
    columns of 64 rows, B as `case.n` rows of `case.k`."""
    tf = kernel_eval._tf32
    out = []
    for r in range(64):
        for c in range(case.n):
            out.append(sum(tf(a[r * case.k + x]) * tf(b[c * case.k + x])
                           for x in range(case.k)))
    return out


CASES = {
    'ss8': Case(8),
    'ss16': Case(16),
    'ss128': Case(128),
    'rs8': Case(8, rs=True),
    'rs24': Case(24, rs=True),
    'k2': Case(16, products=2),
    'k2-groups': Case(16, products=2, groups=2),
    'k3-scale': Case(8, products=3, scale=True),
    'rs-k2-groups': Case(8, rs=True, products=2, groups=2),
    'sw128-k4': Case(16, desc=SW128, products=4),
}


# -- the vocabulary ------------------------------------------------------------ #

def test_the_engine_is_a_warpgroup_product_that_completes_by_group():
    engine = engines.MATRIX_ENGINES['wgmma']
    assert engine.collective is Participants.WARPGROUP
    assert engine.completes is Completion.GROUP and engine.work
    assert engine.sources == ((Source.SHARED, Source.REGISTER),
                              (Source.SHARED,), (Source.REGISTER,))
    assert engines.GROUP_ENGINES == {'wgmma': 'wgmma'}
    # Every synchronous engine is unchanged: registers, on issue.
    for name in ('mma_sync', 'dmma', 'mfma', 'dpas_spmd', 'dpas_esimd'):
        assert engines.MATRIX_ENGINES[name].completes is Completion.SYNC
        assert engines.MATRIX_ENGINES[name].sources == ((Source.REGISTER,),) * 3


@pytest.mark.parametrize('threads,wave,level', [
    (128, 32, Uniformity.MULT), (256, 32, Uniformity.MULT),
    (64, 32, Uniformity.BLOCK), (96, 32, Uniformity.BLOCK),
    (192, 32, Uniformity.BLOCK)])
def test_a_warpgroup_holds_one_multiplication_only_where_it_is_whole(
        threads, wave, level):
    """Where the multiplication is a whole number of warpgroups, each of
    them holds lanes of that multiplication alone; anywhere else a
    warpgroup holds lanes of several, and only the block agrees for all."""
    assert Participants.WARPGROUP.arrival(threads, wave) is level


def test_the_commit_is_an_entry_of_the_registry():
    spec = engines.spec_of(Op.COMMIT_GROUP)
    assert spec is not None and spec.results == ()
    assert spec.effect == Effect.ASYNC and not spec.movable
    assert spec.completes is Completion.GROUP and spec.pipe == 'sync'
    s = engines.commit_group('wgmma')
    assert s.attrs == (('engine', 'wgmma'),)
    assert engines.verify_engine(s) == []
    with pytest.raises(IRError, match='commits no groups'):
        engines.commit_group('mma_sync')
    bad = s.with_attr('engine', 'tdm')
    assert any('commits no groups' in d for d in engines.verify_engine(bad))


def test_the_wait_is_the_counter_s_and_names_what_it_makes_valid():
    b = IRBuilder(fptype=Datatype.F32, context=_ctx())
    d = [b.declare(hint='d') for _ in range(4)]
    s = engines.matrix_wait('wgmma', 1, d, ['buf'])
    assert s.op == Op.WAIT_COUNTER
    assert s.attrs == (('counter', 'wgmma'), ('count', 1))
    assert s.args == tuple(d)
    assert [(a.base, a.space, a.kind) for a in s.accesses] == [
        (v, MemSpace.REGISTER, Effect.READ | Effect.WRITE) for v in d] + [
        ('buf', MemSpace.SHARED, Effect.READ)]
    # Pinned, and no wait on copies: nothing moves it, `asyncmem` leaves
    # its count alone.
    assert not s.movable and s.effect == Effect.READ | Effect.WRITE | Effect.ASYNC
    from tensorforge.backend.pir.core import retires
    assert not retires(s)
    assert engines.verify_engine(s) == []
    with pytest.raises(IRError, match='commits no groups'):
        engines.matrix_wait('tcgen05', 0)


def test_a_fence_of_registers_names_them_and_no_other_fence_names_any():
    b = IRBuilder(fptype=Datatype.F32, context=_ctx())
    d = [b.declare(hint='d') for _ in range(2)]
    s = engines.fence_stmt(FenceKind.REGISTERS, engine='wgmma', registers=d)
    assert s.args == tuple(d) and s.attr('engine') == 'wgmma'
    assert [(a.base, a.kind) for a in s.accesses] == [
        (v, Effect.READ | Effect.WRITE) for v in d]
    assert s.effect == Effect.UNKNOWN and engines.verify_engine(s) == []
    with pytest.raises(IRError, match='names no registers'):
        engines.fence_stmt(FenceKind.MEMORY, registers=d)
    bare = dataclasses.replace(s, accesses=())
    assert any('read and a write' in x for x in engines.verify_engine(bare))
    other = s.with_attr('engine', 'mfma')
    assert any('registers of' in x for x in engines.verify_engine(other))


# -- the descriptor ---------------------------------------------------------- #

def test_the_bits_of_a_descriptor():
    """PTX's fields: the offsets in 16-byte units from bit 16 and bit 32,
    the swizzle at the top -- wgmma's two bits from 62, tcgen05's three
    from 61 beside its fixed 0b001 at 46.  The start address is the
    device's to fill in."""
    assert engines.matrix_descriptor_bits(PLAIN) == (8 << 16) | (16 << 32)
    assert engines.matrix_descriptor_bits(SW128) == (
        (1 << 16) | (64 << 32) | (1 << 62))
    wg64 = MatrixDescType(Datatype.TF32, swizzle=64, lbo=16, sbo=512)
    wg32 = MatrixDescType(Datatype.TF32, swizzle=32, lbo=16, sbo=256)
    assert engines.matrix_descriptor_bits(wg64) >> 62 == 2
    assert engines.matrix_descriptor_bits(wg32) >> 62 == 3
    # The tcgen05 prototype's SW128 operand (`volume7`): LBO 1, SBO 1024,
    # layout 2 -- and its plain one (`gemm56`) the version bit alone.
    tc = MatrixDescType(Datatype.TF32, swizzle=128, lbo=16, sbo=1024,
                        form='tcgen05')
    assert engines.matrix_descriptor_bits(tc) == (
        (1 << 16) | (64 << 32) | (1 << 46) | (2 << 61))
    plain_tc = dataclasses.replace(PLAIN, form='tcgen05')
    assert engines.matrix_descriptor_bits(plain_tc) == (
        (8 << 16) | (16 << 32) | (1 << 46))


@pytest.mark.parametrize('field', [
    dict(swizzle=48), dict(lbo=8), dict(sbo=16 << 14), dict(form='mma')])
def test_a_descriptor_it_cannot_state_is_refused(field):
    with pytest.raises(IRError):
        MatrixDescType(Datatype.TF32, **{'lbo': 128, 'sbo': 256, **field})


def test_the_canonical_layouts():
    """A core matrix is eight rows of 16 contiguous bytes; the K halves of a
    row of TF32 are the leading byte offset apart, the groups of eight rows
    the stride byte offset.  Swizzled, a row is `swizzle` bytes of K, and
    an address's 16-byte chunk is XORed with its row in the atom."""
    off = engines.descriptor_offset
    assert [off(PLAIN, 0, k) for k in range(8)] == [0, 4, 8, 12, 128, 132,
                                                     136, 140]
    assert [off(PLAIN, r, 0) for r in (1, 7, 8, 9, 17)] == [16, 112, 256, 272,
                                                             528]
    assert [off(SW128, r, 5) for r in (0, 1, 7, 8)] == [20, 148, 916, 1044]
    swz = engines.swizzled
    assert [swz(a, 128) for a in (0, 16, 128, 144, 1023, 1024 + 128)] == [
        0, 16, 144, 128, 1023 ^ (7 << 4), 1024 + 144]
    assert swz(128 + 32, 64) == 128 + 32 ^ (1 << 4)
    assert swz(256 + 16, 32) == 256 + 16 ^ (0 << 4)
    assert swz(128, 32) == 128 ^ 16 and swz(48, 0) == 48
    assert [engines.swizzle_alignment(s) for s in (0, 32, 64, 128)] == [
        16, 256, 512, 1024]


def test_a_descriptor_is_a_pure_value_that_names_its_buffer():
    ctx = _ctx()
    b = IRBuilder(fptype=Datatype.F32, context=ctx)
    sa = b.alloc(Datatype.F32, (512,), MemSpace.SHARED, hint='sa',
                 extern='sa', arena='blk', offset=0)
    desc = b.matrix_desc(sa, 8, desc=PLAIN)
    s = _stmts(b.finish(), Op.MATRIX_DESC)[0]
    assert s.target == (desc,) and s.args == (sa, 8)
    assert s.pure and s.movable and s.effect == Effect.NONE
    assert not s.accesses and s.attr('no_inline')
    assert desc.type is PLAIN


def test_a_descriptor_of_what_it_cannot_describe_is_refused():
    b = IRBuilder(fptype=Datatype.F32, context=_ctx())
    g = b.alloc(Datatype.F32, (512,), MemSpace.GLOBAL, hint='g', extern='m1')
    with pytest.raises(IRError, match='no buffer of shared memory'):
        b.matrix_desc(g, 0, desc=PLAIN)
    loose = b.alloc(Datatype.F32, (512,), MemSpace.SHARED, hint='s',
                    extern='s', arena='blk', offset=0)
    with pytest.raises(IRError, match='not placed on 256 elements'):
        b.matrix_desc(loose, 0, desc=SW128)
    wide = b.alloc(Datatype.F64, (512,), MemSpace.SHARED, hint='w',
                   extern='w', arena='blk', offset=0)
    with pytest.raises(IRError, match='over a buffer of'):
        b.matrix_desc(wide, 0, desc=PLAIN)
    placed = b.alloc(Datatype.F32, (512,), MemSpace.SHARED, hint='p',
                     extern='p', arena='blk', offset=0, place_align=256)
    assert b.matrix_desc(placed, 0, desc=SW128).type is SW128


def test_a_descriptor_is_written_with_its_layout_bits():
    """`uint64_t d = tensorforge::matrixDescriptor(&sa[i], bits)`: the
    address goes in on the device."""
    lines = _emit(_wgmma_body(CASES['ss8'])[1])
    descs = [line for line in lines if 'matrixDescriptor' in line]
    assert descs == [
        'uint64_t v{}_bd = tensorforge::matrixDescriptor(&sb[0], '
        '0x0000001000080000ull);'.format(_id(lines, 'bd')),
        'uint64_t v{}_ad = tensorforge::matrixDescriptor(&sa[0], '
        '0x0000001000080000ull);'.format(_id(lines, 'ad'))]


def _id(lines, hint):
    for line in lines:
        m = re.search(rf'\bv(\d+)_{hint}\b', line)
        if m:
            return m.group(1)
    raise AssertionError(hint)


def test_without_a_spelling_a_descriptor_is_refused():
    """Only CUDA spells one; elsewhere the type and the value are refused
    where they would be written."""
    ctx = _ctx('gfx942', 'hip')
    lex = ctx.target.lexic
    assert lex.matrix_descriptor_type() is None
    assert lex.matrix_descriptor('&s[0]', 0) is None
    with pytest.raises(IRError, match='matrix descriptor'):
        Emitter(_Sink(), ctx).ctype(PLAIN)
    assert Emitter(_Sink(), _ctx()).ctype(PLAIN) == 'uint64_t'


# -- the statement ------------------------------------------------------------- #

def _product(case):
    body = _wgmma_body(case)[1]
    return body, _stmts(body, Op.MATRIX_MAC)


def test_a_product_from_shared_memory_is_a_matrix_op_in_place():
    """D read and written in place, A and B one descriptor each whose
    buffers it reads, scale-d behind the fragments; a collective of the
    warpgroup, asynchronous, pinned."""
    body, (s,) = _product(CASES['ss16'])
    sa = next(x.target[0] for x in _stmts(body, Op.ALLOC)
              if str(x.target[0]).endswith('_sa'))
    sb = next(x.target[0] for x in _stmts(body, Op.ALLOC)
              if str(x.target[0]).endswith('_sb'))
    assert s.attr('engine') == 'wgmma' and s.attr('shape') == (64, 16, 8)
    assert s.attr('types') == (Datatype.TF32, Datatype.TF32, Datatype.F32,
                               Datatype.F32)
    assert s.attr('fragments') == (1, 1, 8)
    assert s.attr('sources') == (Source.SHARED, Source.SHARED,
                                 Source.REGISTER)
    assert s.attr('scalars') == ('scale_d',) and s.args[-1] == 0
    assert s.attr('collective') is Participants.WARPGROUP
    assert s.attr('in_place') is True and s.target == ()
    assert s.effect == Effect.WRITE | Effect.READ | Effect.ASYNC
    assert not s.pure and not s.movable
    a, bb, c = engines.fragments(s)
    assert [x.space for x in s.accesses] == [MemSpace.REGISTER] * 8 + [
        MemSpace.SHARED] * 2
    assert [x.base for x in s.accesses] == list(c) + [sa, sb]
    assert [x.kind for x in s.accesses[8:]] == [Effect.READ] * 2
    assert engines.verify_engine(s) == []


@pytest.mark.parametrize('n', [8, 24, 64, 128, 256])
def test_the_fragment_layouts_are_ptx_s(n):
    """D's and A's registers as PTX lays them out over the warpgroup (the
    oracle's maps, `Interp._wgmma`), said in the bits a value is read in:
    every element at the lane and the register the formulas give."""
    d = nvidia.wgmma_d_fragment_bits(n)
    a = nvidia.wgmma_a_fragment_bits()
    for lane in range(128):
        w, g, t = lane // 32, (lane % 32) // 4, lane % 4
        for i in range(n // 2):
            row, col = 16 * w + g + 8 * ((i % 4) // 2), 8 * (i // 4) + 2 * t + i % 2
            got = d.locate(row, col)
            assert (got.slot, got.lane) == (i, lane), (row, col)
        for i in range(4):
            got = a.locate(16 * w + g + 8 * (i % 2), t + 4 * (i // 2))
            assert (got.slot, got.lane) == (i, lane)


def test_the_op_states_the_layouts():
    _, (s,) = _product(CASES['rs24'])
    assert s.attr('layouts') == (nvidia.wgmma_a_fragment_bits(), None,
                                 nvidia.wgmma_d_fragment_bits(24))
    _, (s,) = _product(CASES['ss8'])
    assert s.attr('layouts') == (None, None, nvidia.wgmma_d_fragment_bits(8))


def test_a_from_registers_reads_shared_memory_for_b_alone():
    _, (s,) = _product(CASES['rs8'])
    a, _, c = engines.fragments(s)
    assert len(a) == 4 and all(v.type.base == Datatype.TF32 for v in a)
    assert s.attr('sources')[0] is Source.REGISTER
    assert len(s.accesses) == len(c) + 1
    assert engines.verify_engine(s) == []


def test_the_effect_follows_the_engine_and_the_sources():
    wg = (('engine', 'wgmma'), ('in_place', True),
          ('sources', (Source.SHARED, Source.SHARED, Source.REGISTER)))
    assert engines.effect_of(Op.MATRIX_MAC, wg) == (
        Effect.WRITE | Effect.READ | Effect.ASYNC)
    rs = (('engine', 'wgmma'), ('in_place', True),
          ('sources', (Source.REGISTER, Source.SHARED, Source.REGISTER)))
    assert engines.effect_of(Op.MATRIX_MAC, rs) == (
        Effect.WRITE | Effect.READ | Effect.ASYNC)
    # The synchronous engines as they were.
    assert engines.effect_of(Op.MATRIX_MAC, (('engine', 'mma_sync'),
                                             ('in_place', True))) == Effect.WRITE
    assert engines.effect_of(Op.MATRIX_MAC, (('engine', 'mfma'),)) == \
        Effect.NONE


def test_the_body_verifies():
    for case in CASES.values():
        body = _wgmma_body(case)[1]
        assert not [d for d in verify(body, strict=False)
                    if 'matrix' in d or 'fence' in d or 'commit' in d], case


@pytest.mark.parametrize('name', ['ss8', 'rs24', 'k2-groups', 'k3-scale',
                                  'sw128-k4'])
def test_a_body_comes_through_the_pipeline_with_its_products_in_place(name):
    """The standard passes move the address arithmetic and leave the
    products, the fences, the commits and the waits where they stand -- and
    the body still computes its product."""
    from tensorforge.backend import pir
    case = CASES[name]
    ctx, body, _ = _wgmma_body(case)
    out = pir.optimize(body)
    kept = (Op.MATRIX_MAC, Op.COMMIT_GROUP, Op.WAIT_COUNTER, Op.FENCE,
            Op.BARRIER)
    assert [s.op for s, _ in walk(out) if s.op in kept] == [
        s.op for s, _ in walk(body) if s.op in kept]
    got, want = _run_oracle(case, _oracle_source(_emit(out, ctx)))
    assert max(abs(g - w) / max(1.0, abs(w)) for g, w in zip(got, want)) \
        < 1e-12


def _broken(s, **change):
    return dataclasses.replace(s, **change)


def test_what_the_verifier_holds_a_product_to():
    _, (s,) = _product(CASES['ss8'])
    a, bb, c = engines.fragments(s)
    check = engines.verify_engine
    assert any('fragments' in d for d in check(s.with_attr('fragments',
                                                           (1, 1, 3))))
    assert any('shared memory' in d for d in check(_broken(
        s, accesses=s.accesses[:-1])))
    assert any('read and a write' in d for d in check(_broken(
        s, accesses=s.accesses[1:])))
    assert any('through a descriptor' in d for d in check(s.with_attr(
        'sources', (Source.REGISTER, Source.SHARED, Source.REGISTER))))
    assert any('takes no C from' in d for d in check(s.with_attr(
        'sources', (Source.SHARED, Source.SHARED, Source.SHARED))))
    assert any('writes D in place' in d for d in check(_broken(
        s, attrs=tuple(x for x in s.attrs if x[0] != 'in_place'),
        target=())))


@pytest.mark.parametrize('change,why', [
    (dict(shape=(64, 12, 8)), 'shape'),
    (dict(shape=(64, 8, 16)), 'shape'),
    (dict(types=(Datatype.F64,) * 4), 'TF32 operands'),
])
def test_a_product_this_lowering_does_not_write_is_refused(change, why):
    _, (s,) = _product(CASES['ss8'])
    for k, v in change.items():
        s = s.with_attr(k, v)
    with pytest.raises(IRError, match=why):
        engines.lower(s, _ctx().target, name=str)


def test_a_scale_that_is_neither_bit_is_refused():
    _, (s,) = _product(CASES['ss8'])
    s = _broken(s, args=s.args[:-1] + (2,))
    with pytest.raises(IRError, match='scale-d'):
        engines.lower(s, _ctx().target, name=str)


def test_what_the_lanes_do_not_agree_on_is_refused():
    """The descriptors and scale-d are the warpgroup's, one value of each
    (uniform registers in the SASS): a value that differs between the lanes
    is refused, and the oracle refuses lanes that disagree."""
    b = IRBuilder(fptype=Datatype.F32, context=_ctx())
    sa = b.alloc(Datatype.F32, (512,), MemSpace.SHARED, hint='sa',
                 extern='sa', arena='blk', offset=0)
    lane = b.thread_id('x')
    d = [b.declare(hint='d', uniform=Uniformity.LANE) for _ in range(4)]
    mine = b.matrix_desc(sa, b.op('mul', INDEX, b.op('div', INDEX, lane, 32),
                                  64), desc=PLAIN)
    ours = b.matrix_desc(sa, 0, desc=PLAIN)
    with pytest.raises(IRError, match='differs between the lanes'):
        b.wgmma(8, [mine], ours, d)
    with pytest.raises(IRError, match='differs between the lanes'):
        b.wgmma(8, [ours], ours, d, scale_d=b.op('lt', BOOL, lane, 1))
    # The lowering asks the same of a statement built around the builder.
    (s,) = _product(CASES['ss8'])[1]
    lane_scale = b.op('add', INDEX, lane, 0, hint='s')
    with pytest.raises(IRError, match='differs between the lanes'):
        engines.lower(_broken(s, args=s.args[:-1] + (lane_scale,)),
                      _ctx().target, name=str)


def test_d_declared_the_same_in_every_lane_is_refused():
    """Every lane holds other elements of D: an accumulator declared as a
    value the lanes agree on is refused, the bug class a pass that takes
    it for one produced on `mma.sync` once."""
    b = IRBuilder(fptype=Datatype.F32, context=_ctx())
    sa = b.alloc(Datatype.F32, (512,), MemSpace.SHARED, hint='sa',
                 extern='sa', arena='blk', offset=0)
    d = [b.declare(hint='d') for _ in range(4)]
    desc = b.matrix_desc(sa, 0, desc=PLAIN)
    with pytest.raises(IRError, match='register of D'):
        b.wgmma(8, [desc], desc, d)


@pytest.mark.parametrize('arch,backend', [('sm_90', 'cuda'),
                                          ('sm_100a', 'cuda'),
                                          ('gfx942', 'hip')])
def test_only_sm90a_writes_one(arch, backend):
    """Plain sm_90 has no wgmma in its PTX, sm_100a has tcgen05 in its
    place, and no other backend spells it -- refused, not written."""
    body, (s,) = _product(CASES['ss8'])
    target = _ctx(arch, backend).target
    with pytest.raises(IRError, match='wgmma'):
        engines.lower(s, target, name=str)
    for op in (Op.COMMIT_GROUP, Op.WAIT_COUNTER, Op.FENCE):
        stmt = next(x for x in _stmts(body, op)
                    if op != Op.FENCE or x.attr('kind') is FenceKind.REGISTERS)
        with pytest.raises(IRError):
            engines.lower(stmt, target, name=str)


# -- the text ------------------------------------------------------------------ #

def test_the_text_of_a_product_from_shared_memory():
    """Inside the guard of the one target that has it, volatile, D at %0 on
    under "+f", the descriptors under "l", scale-d the literal."""
    _, (s,) = _product(CASES['ss8'])
    a, bb, c = engines.fragments(s)
    lowered = engines.lower(s, _ctx().target, name=str)
    assert lowered.lines == (
        '#if defined(__CUDA_ARCH_FEAT_SM90_ALL)',
        'asm volatile("wgmma.mma_async.sync.aligned.m64n8k8.f32.tf32.tf32 "\n'
        '"{%0, %1, %2, %3}, %4, %5, 0, 1, 1;"\n'
        f': "+f"({c[0]}), "+f"({c[1]}), "+f"({c[2]}), "+f"({c[3]})\n'
        f': "l"({a[0]}), "l"({bb[0]})\n);',
        '#endif')


def test_the_text_of_a_product_from_registers_with_a_value_scale():
    body = _wgmma_body(Case(8, rs=True, scale=True))[1]
    (s,) = _stmts(body, Op.MATRIX_MAC)
    a, bb, c = engines.fragments(s)
    scale = s.args[-1]
    text = engines.lower(s, _ctx().target, name=str).lines[1]
    assert text == (
        'asm volatile("{\\n.reg .pred p;\\nsetp.ne.b32 p, %9, 0;\\n"\n'
        '"wgmma.mma_async.sync.aligned.m64n8k8.f32.tf32.tf32 "\n'
        '"{%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, p, 1, 1;\\n}"\n'
        f': "+f"({c[0]}), "+f"({c[1]}), "+f"({c[2]}), "+f"({c[3]})\n'
        f': "r"({a[0]}), "r"({a[1]}), "r"({a[2]}), "r"({a[3]}), '
        f'"l"({bb[0]}), "r"(static_cast<uint32_t>({scale}))\n);')


def test_the_fence_the_commit_and_the_wait():
    """The operand fence keeps the compiler's accesses of the registers on
    their side, then the hardware's fence; the commit; the wait and the
    operand fence behind it."""
    body = _wgmma_body(CASES['ss8'])[1]
    target = _ctx().target
    fence = next(x for x in _stmts(body, Op.FENCE)
                 if x.attr('kind') is FenceKind.REGISTERS)
    names = ', '.join(f'"+f"({v})' for v in fence.args)
    assert engines.lower(fence, target, name=str).lines == (
        f'asm volatile("" : {names} :: "memory");',
        'tensorforge::wgmmaFence();')
    (commit,) = _stmts(body, Op.COMMIT_GROUP)
    assert engines.lower(commit, target, name=str).lines == (
        'tensorforge::wgmmaCommit();',)
    (wait,) = _stmts(body, Op.WAIT_COUNTER)
    assert engines.lower(wait, target, name=str).lines == (
        'tensorforge::wgmmaWait<0>();',
        f'asm volatile("" : {names} :: "memory");')
    proxy = next(x for x in _stmts(body, Op.FENCE)
                 if x.attr('kind') is FenceKind.PROXY)
    assert engines.lower(proxy, target, name=str).lines == (
        'tensorforge::fenceProxyAsync();',)


def test_an_operand_fence_names_each_register_by_its_class():
    b = IRBuilder(fptype=Datatype.F32)
    regs = [b.declare(ScalarType(t), hint='r') for t in (
        Datatype.F32, Datatype.F64, Datatype.TF32)]
    assert engines.operand_fence(regs, str) == (
        f'asm volatile("" : "+f"({regs[0]}), "+d"({regs[1]}), '
        f'"+r"({regs[2]}) :: "memory");')
    with pytest.raises(IRError, match='register class'):
        engines.operand_fence([b.declare(ScalarType(Datatype.BOOL))], str)


def test_the_proxy_fence_is_sm90s_and_cuda_s():
    fence = engines.fence_stmt(FenceKind.PROXY)
    assert engines.lower(fence, _ctx('sm_90').target).lines == (
        'tensorforge::fenceProxyAsync();',)
    for arch, backend in (('sm_80', 'cuda'), ('gfx942', 'hip')):
        with pytest.raises(IRError, match='async proxy'):
            engines.lower(fence, _ctx(arch, backend).target)


def test_the_emitted_body_keeps_its_wait():
    """A body with no asynchronous copy writes the wait of its products:
    the emitter drops a counter's wait only where the copies it would retire
    were written as loads and stores."""
    lines = _emit(_wgmma_body(CASES['ss8'])[1])
    order = [i for i, line in enumerate(lines) if line in (
        'tensorforge::fenceProxyAsync();', '__syncthreads();',
        'tensorforge::wgmmaFence();', 'tensorforge::wgmmaCommit();',
        'tensorforge::wgmmaWait<0>();')]
    assert [lines[i] for i in order] == [
        'tensorforge::fenceProxyAsync();', '__syncthreads();',
        'tensorforge::wgmmaFence();', 'tensorforge::wgmmaCommit();',
        'tensorforge::wgmmaWait<0>();']
    product = next(i for i, line in enumerate(lines) if 'wgmma.mma_async' in line)
    assert lines[product - 1] == '#if defined(__CUDA_ARCH_FEAT_SM90_ALL)'
    assert lines[product + 1] == '#endif'
    assert order[2] < product < order[3]


def test_a_product_counts_as_one_matrix_instruction_of_work():
    """Priced as `mma.sync` is: one issue of the matrix category and inexact
    work; the commit and the wait count as synchronization."""
    _, body, _ = _wgmma_body(CASES['ss16'])
    metrics = _Metrics()
    _emit(body, metrics=metrics)
    assert ('matrix', 1, 1) in metrics.mix
    assert len([m for m in metrics.mix if m[0] == 'matrix']) == 1
    assert metrics.work.count((1, None)) == 1
    assert len([m for m in metrics.mix if m[0] == 'sync']) == 2


# -- the host oracle ------------------------------------------------------------ #

def _run_oracle(case, src=None, seed=3):
    ctx, body, _ = _wgmma_body(case)
    if src is None:
        src = _oracle_source(_emit(body, ctx))
    mem = kernel_eval.evaluate_wave(src, 128, seed=seed, globals_only=True)
    slot = kernel_eval.Slot(seed)
    a = [slot.read('m1', i) for i in range(case.a_elems)]
    b = [slot.read('m2', i) for i in range(case.b_elems)]
    got = [mem[('m0', i)] for i in range(64 * case.n)]
    return got, _expected(case, a, b)


@pytest.mark.parametrize('name', sorted(CASES))
def test_the_oracle_multiplies_as_the_ptx_lays_the_operands_out(name):
    """The oracle reads the operands and writes D as PTX lays them out,
    stated on its own (`Interp._wgmma`); the bodies stage them by the
    layouts the descriptors state (`engines.descriptor_offset`), and read D
    by the fragment layout.  The two agree on every product."""
    got, want = _run_oracle(CASES[name])
    worst = max(abs(g - w) / max(1.0, abs(w)) for g, w in zip(got, want))
    assert worst < 1e-12, worst


def test_d_read_ahead_of_its_wait_is_refused():
    """D holds nothing readable until the wait that retires its group."""
    case = CASES['ss8']
    lines = _emit(_wgmma_body(case)[1])
    wait = lines.index('tensorforge::wgmmaWait<0>();')
    early = lines[:wait] + lines[wait + 2:] + lines[wait:wait + 2]
    with pytest.raises(kernel_eval.Abort, match='in flight'):
        _run_oracle(case, _oracle_source(early))


def test_an_operand_written_while_the_product_is_in_flight_is_a_race():
    case = CASES['ss8']
    lines = _emit(_wgmma_body(case)[1])
    wait = lines.index('tensorforge::wgmmaWait<0>();')
    racy = lines[:wait] + ['sa[0] = 0.5f;'] + lines[wait:]
    with pytest.raises(kernel_eval.Abort, match='in flight'):
        _run_oracle(case, _oracle_source(racy))


def test_a_wait_that_leaves_the_group_in_flight_retires_nothing():
    """`wgmmaWait<1>` with one group committed: D is still in flight behind
    it, and reading it is refused."""
    case = CASES['ss8']
    lines = _emit(_wgmma_body(case)[1])
    lax = [line.replace('wgmmaWait<0>', 'wgmmaWait<1>') for line in lines]
    with pytest.raises(kernel_eval.Abort, match='in flight'):
        _run_oracle(case, _oracle_source(lax))


def test_the_oracle_parses_the_statement_it_models():
    _, (s,) = _product(CASES['rs8'])
    text = engines.lower(s, _ctx().target, name=str).lines[1]
    wg = kernel_eval._wgmma_statement(text.replace('\n', ' ').rstrip(';'))
    a, bb, c = engines.fragments(s)
    assert (wg.m, wg.n, wg.k, wg.atype) == (64, 8, 8, 'tf32')
    assert wg.d == tuple(str(v) for v in c) and not wg.a_desc
    assert wg.a == tuple(str(v) for v in a) and wg.b == str(bb[0])
    assert wg.scale == '0'


# -- the arch-specific targets ----------------------------------------------- #

def test_sm90a_is_sm90_with_wgmma():
    """The same part as sm_90, its parameters from sm_90's row; the one
    target that writes wgmma."""
    a, plain = _ctx('sm_90a').target, _ctx('sm_90').target
    assert a.hw.model == 'sm_90a' and a.hw.sm_level() == 90
    for field in ('vec_unit_length', 'max_local_mem_size_per_block',
                  'max_reg_per_block', 'max_threads_per_block',
                  'icache_miss_factor', 'max_argument_size', 'shmem_banks'):
        assert getattr(a.hw, field) == getattr(plain.hw, field), field
    assert a.arch_specific() and a.wgmma()
    assert not plain.arch_specific() and not plain.wgmma()
    b = _ctx('sm_100a').target
    assert b.arch_specific() and not b.wgmma() and b.hw.sm_level() == 100


def test_an_arch_specific_target_is_built_for_itself_alone():
    """`-arch=sm_90a` builds `compute_90` PTX besides, where ptxas refuses
    wgmma; the arch-specific target alone is `-gencode`.  A plain target
    keeps its flags, and the report build its `-arch=sm_90a`."""
    nvcc = toolchain.COMPILERS['cuda']
    assert nvcc.target_flags('sm_90a') == [
        '-gencode=arch=compute_90a,code=sm_90a']
    assert nvcc.target_flags('sm_100a') == [
        '-gencode=arch=compute_100a,code=sm_100a']
    assert nvcc.target_flags('sm_90') == ['-arch=sm_90']
    cmd, _ = nvcc.report_build('nvcc', 'k.cu', '/tmp', 'sm_90', '/i')
    assert '-arch=sm_90a' in cmd
    cmd, _ = nvcc.report_build('nvcc', 'k.cu', '/tmp', 'sm_90a', '/i')
    assert '-gencode=arch=compute_90a,code=sm_90a' in cmd


# -- compiled for sm_90a ------------------------------------------------------- #

def _cuda_source(case, lines, arena):
    """`lines` as a kernel of one warpgroup and a host driver: the arena on
    1024 bytes (`engines.swizzle_alignment`), the inputs from the file named
    first, D to the second."""
    keep = [line for line in lines if not re.match(r'^float m\d+\[', line)]
    body = ''.join(f'  {line}\n' for line in keep)
    return f'''#include <cstdio>
#include <cstdlib>
#include <vector>
#include "tensorforge_device/cuda.h"

__global__ void __launch_bounds__(128) wgmma_case(float *__restrict__ m0,
    const float *__restrict__ m1, const float *__restrict__ m2) {{
  extern __shared__ __align__(1024) unsigned char tfRaw[];
  float *blk = reinterpret_cast<float *>(tfRaw + ((1024u -
      (static_cast<unsigned>(__cvta_generic_to_shared(tfRaw)) & 1023u)) & 1023u));
{body}}}

#define CHECK(x) do {{ cudaError_t e_ = (x); if (e_ != cudaSuccess) {{ \\
  std::fprintf(stderr, "%s: %s\\n", #x, cudaGetErrorString(e_)); return 1; }} }} while (0)

int main(int argc, char **argv) {{
  const size_t na = {case.a_elems}, nb = {case.b_elems}, nd = {64 * case.n};
  std::vector<float> h(na + nb), d(nd);
  FILE *in = std::fopen(argv[1], "rb");
  if (!in || std::fread(h.data(), 4, na + nb, in) != na + nb) return 2;
  std::fclose(in);
  float *m0, *m1, *m2;
  CHECK(cudaMalloc(&m0, nd * 4));
  CHECK(cudaMalloc(&m1, na * 4));
  CHECK(cudaMalloc(&m2, nb * 4));
  CHECK(cudaMemcpy(m1, h.data(), na * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(m2, h.data() + na, nb * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMemset(m0, 0, nd * 4));
  const int smem = {arena * 4} + 1024;
  CHECK(cudaFuncSetAttribute(wgmma_case,
      cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  wgmma_case<<<1, 128, smem>>>(m0, m1, m2);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(d.data(), m0, nd * 4, cudaMemcpyDeviceToHost));
  FILE *out = std::fopen(argv[2], "wb");
  if (!out || std::fwrite(d.data(), 4, nd, out) != nd) return 3;
  std::fclose(out);
  return 0;
}}
'''


def _nvcc():
    found = toolchain.COMPILERS['cuda'].find()
    if not found or not shutil.which(found):
        pytest.skip('nvcc not found')
    return found


def _build(case, tmp_path, flags, product='cubin'):
    ctx, body, arena = _wgmma_body(case)
    src = tmp_path / 'wgmma.cu'
    src.write_text(_cuda_source(case, _emit(body, ctx), arena))
    nvcc = toolchain.COMPILERS['cuda']
    out = tmp_path / ('k.cubin' if product == 'cubin' else 'wgmma')
    cmd = [_nvcc(), *nvcc.language_flags(), *flags, '-I', str(INCLUDE),
           '-o', str(out), str(src)]
    if product == 'cubin':
        cmd.insert(1, '-cubin')
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stderr[-4000:]
    return out


def _sass(cubin):
    tool = shutil.which('cuobjdump')
    if tool is None:
        pytest.skip('cuobjdump not found')
    run = subprocess.run([tool, '-sass', str(cubin)], capture_output=True,
                         text=True, timeout=600)
    assert run.returncode == 0, run.stderr
    return run.stdout


@pytest.mark.parametrize('name', sorted(CASES))
def test_compiled_for_sm90a_each_product_is_one_hgmma(name, tmp_path):
    """Built as the toolchain builds an sm_90a target -- every body the GPU
    cases run: as many HGMMA of the shape as the body issues products, the
    fence an arrive of the warpgroup, the wait a dependency barrier on its
    group."""
    case = CASES[name]
    form = f'HGMMA.64x{case.n}x8.F32.TF32'
    flags = toolchain.COMPILERS['cuda'].target_flags('sm_90a')
    sass = _sass(_build(case, tmp_path, flags))
    hgmma = re.findall(r'HGMMA\.\S+', sass)
    assert hgmma and all(h.startswith(form) for h in hgmma), hgmma
    assert len(hgmma) == case.products
    assert 'WARPGROUP.ARRIVE' in sass
    assert re.search(r'WARPGROUP\.DEPBAR\.LE gsb0, 0x0', sass)
    assert 'BPT.TRAP' not in sass


#: The instructions of the fence, the commit, the wait and the proxy fence
#: in PTX, as the helpers write them (`cuda.h`).
_PTX_SEQUENCE = re.compile(
    r'(wgmma\.fence\.sync\.aligned|wgmma\.commit_group\.sync\.aligned'
    r'|wgmma\.wait_group\.sync\.aligned\s+\d+|fence\.proxy\.async'
    r'\.shared::cta|wgmma\.mma_async)')


def _ptx_expected(body):
    """What the body's fences, products, commits and waits are in PTX, in
    the order the body issues them."""
    out = []
    for s, _ in walk(body):
        if s.op == Op.FENCE and s.attr('kind') is FenceKind.PROXY:
            out.append('fence.proxy.async.shared::cta')
        elif s.op == Op.FENCE and s.attr('kind') is FenceKind.REGISTERS:
            out.append('wgmma.fence.sync.aligned')
        elif s.op == Op.MATRIX_MAC:
            out.append('wgmma.mma_async')
        elif s.op == Op.COMMIT_GROUP:
            out.append('wgmma.commit_group.sync.aligned')
        elif s.op == Op.WAIT_COUNTER:
            out.append(f'wgmma.wait_group.sync.aligned {s.attr("count")}')
    return out


@pytest.mark.parametrize('name', sorted(CASES))
def test_compiled_for_sm90a_the_ptx_issues_what_the_body_states(name,
                                                                tmp_path):
    """The SASS cannot say it -- ptxas puts an arrive of the warpgroup ahead
    of a product and a wait for every group ahead of a read of D where the
    PTX has none -- so the PTX: the proxy fence, every fence of the
    registers, product, commit and wait the body states, in its order, the
    waits with their counts."""
    case = CASES[name]
    ctx, body, arena = _wgmma_body(case)
    src = tmp_path / 'wgmma.cu'
    src.write_text(_cuda_source(case, _emit(body, ctx), arena))
    nvcc = toolchain.COMPILERS['cuda']
    out = tmp_path / 'k.ptx'
    run = subprocess.run([_nvcc(), '-ptx', *nvcc.language_flags(),
                          *nvcc.target_flags('sm_90a'), '-I', str(INCLUDE),
                          '-o', str(out), str(src)],
                         capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stderr[-4000:]
    got = [' '.join(m.split()) for m in _PTX_SEQUENCE.findall(out.read_text())]
    assert got == _ptx_expected(body)


def test_built_with_arch_sm90a_the_generic_ptx_traps_and_compiles(tmp_path):
    """`-arch=sm_90a` compiles the source for `compute_90` too: the guard
    leaves the products out there and the fence traps, so that build
    succeeds -- and the sm_90a code in it is the same."""
    _build(CASES['ss8'], tmp_path, ['-arch=sm_90a'], product='object')


# -- on the hardware ---------------------------------------------------------- #

def _gpu():
    """The compute capability of the first visible GPU, as `sm_XY`."""
    tool = shutil.which('nvidia-smi')
    if tool is None:
        return None
    try:
        run = subprocess.run([tool, '--query-gpu=compute_cap',
                              '--format=csv,noheader'], capture_output=True,
                             text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    first = run.stdout.strip().splitlines()[:1]
    if run.returncode or not first:
        return None
    return 'sm_' + first[0].strip().replace('.', '')


def _inputs(case, kind):
    """A and B, row-major: small integers, whose products and sums are
    exact in FP32 and so must match to the bit; or values spread over a few
    binades, for the TF32 reading of an operand."""
    out = []
    for i in range(case.a_elems + case.b_elems):
        h = (i * 2654435761) & 0xFFFFFFFF
        if kind == 'int':
            out.append(float((h >> 7) % 7 - 3))
        else:
            out.append(((h >> 8) % 4096 - 2048) / 977.0)
    return out


@pytest.mark.parametrize('kind', ['int', 'real'])
@pytest.mark.parametrize('name', sorted(CASES))
def test_gpu_on_sm90(name, kind, tmp_path):
    """The bodies run on an sm_90 part, built for sm_90a, against the same
    products computed here: to the bit with integer operands, within FP32's
    rounding of the sums with the others (TF32 operands as the instruction
    reads them, `kernel_eval._tf32`)."""
    if _gpu() != 'sm_90':
        pytest.skip('no sm_90 GPU')
    case = CASES[name]
    exe = _build(case, tmp_path,
                 toolchain.COMPILERS['cuda'].target_flags('sm_90a'),
                 product='binary')
    values = _inputs(case, kind)
    (tmp_path / 'in.bin').write_bytes(struct.pack(f'<{len(values)}f', *values))
    run = subprocess.run([str(exe), str(tmp_path / 'in.bin'),
                          str(tmp_path / 'out.bin')], capture_output=True,
                         text=True, timeout=600)
    assert run.returncode == 0, run.stderr[-2000:]
    raw = (tmp_path / 'out.bin').read_bytes()
    got = struct.unpack(f'<{64 * case.n}f', raw)
    want = _expected(case, values[:case.a_elems], values[case.a_elems:])
    # FP32 sums of up to 32 products round by a few ulps of the largest; a
    # TF32 operand rounded rather than cut, or one element in the wrong
    # place, is off by orders of magnitude more.  Integer operands are
    # exact.
    scale = max(abs(w) for w in want) or 1.0
    bound = 0.0 if kind == 'int' else 1e-5 * scale
    bad = [(i // case.n, i % case.n, g, w)
           for i, (g, w) in enumerate(zip(got, want)) if abs(g - w) > bound]
    assert not bad, (f'{len(bad)} of {len(want)} elements of D differ '
                     f'(row, column, got, want): {bad[:6]}')
