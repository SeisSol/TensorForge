# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The DPAS table is derived, and this is where that is checked.

`primitives/intel.py` states operand shapes that come out of the arithmetic in
`sycl/ext/intel/esimd/xmx/dpas.hpp`.  Restating a fact from someone else's
header is a copy, and copies drift -- so the formulas are written out here a
second time, independently, and the two are compared.  The same arrangement
`test_amd_caps.py` uses against `hip.h`.

What this cannot check is the *fragment layout*: which lane holds which
element of A, B and C.  That is not in the header either, which is why
`intel.ENABLED` is off; see the module docstring.
"""

from __future__ import annotations

import pytest

from tensorforge.backend.instructions.compute.primitives import intel
from tensorforge.common.basic_types import Datatype
from tensorforge.backend.instructions.compute.matmul import MatmulOperands
from tensorforge.backend.instructions.compute import split
from tensorforge.backend.instructions.compute.strategy import (
    ComputeShape, Strategy)


# --------------------------------------------------------------------------
# The header's arithmetic, written out again
# --------------------------------------------------------------------------

def _ops_per_channel(a_bits, b_bits):
    """`MaxElemsInDword > 8 ? 8 : (MaxElemsInDword < 1 ? 1 : MaxElemsInDword)`"""
    max_elems_in_dword = 32 // max(a_bits, b_bits)
    return 8 if max_elems_in_dword > 8 else max(max_elems_in_dword, 1)


def _shape(depth, repeat, elem_bits, exec_size):
    m = repeat
    k = depth * _ops_per_channel(elem_bits, elem_bits)
    n = exec_size
    return m, n, k


@pytest.mark.parametrize('name', sorted(intel.ATOMS))
def test_the_shape_matches_the_header_arithmetic(name):
    atom = intel.ATOMS[name]
    m, n, k = _shape(atom.depth, atom.repeat, atom.elem_bits, atom.exec_size)
    assert (atom.m, atom.n, atom.k) == (m, n, k)


@pytest.mark.parametrize('name', sorted(intel.ATOMS))
def test_the_operand_sizes_are_the_ones_the_static_asserts_want(name):
    """`_M*_K*A_bits == AN*sizeof(AT)*8`, and the same for B.

    Element counts, not bytes: `dpas` is called with `simd<AT, AN>` where `AT`
    is the precision's own type, so the two sides of the header's assertion
    reduce to `AN == M*K`.
    """
    atom = intel.ATOMS[name]
    assert atom.a_elems == atom.m * atom.k
    assert atom.b_elems == atom.k * atom.n
    assert atom.c_elems == atom.m * atom.n


def test_tf32_is_one_op_per_channel():
    """32-bit elements do not pack into a dword, so K is the systolic depth."""
    atom = intel.ATOMS['tf32']
    assert atom.ops_per_channel == 1
    assert atom.k == intel.SYSTOLIC_DEPTH


def test_sixteen_bit_types_pack_two_per_channel():
    for name in ('bf16', 'fp16'):
        assert intel.ATOMS[name].ops_per_channel == 2
        assert intel.ATOMS[name].k == 2 * intel.SYSTOLIC_DEPTH


@pytest.mark.parametrize('repeat', [1, 2, 4, 8])
def test_repeat_is_the_only_free_parameter(repeat):
    """`verify_repeat_count` admits 1, 2, 4 and 8; everything else follows."""
    atom = intel.ATOMS['tf32'].with_repeat(repeat)
    assert atom.m == repeat and atom.k == 8 and atom.n == 16
    assert atom.a_elems == repeat * 8 and atom.c_elems == repeat * 16
    assert atom.b_elems == 128, 'B does not depend on the repeat count'


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------

def test_the_path_is_parked():
    """Off, and the reason is not the same as NVIDIA's.

    There the open question is a register-allocation constraint no front end
    can see.  Here it is the fragment layout, which is in no header at all --
    a wrong one compiles and computes wrong numbers.
    """
    assert intel.ENABLED is False


def test_only_a_sixteen_wide_wave():
    """`ExecutionSize` is 16 for every type in the table, and under ESIMD the
    vector width *is* the thread count -- so another width is a different
    instruction, not a narrower use of this one."""
    assert intel.supports(16, Datatype.F32, False)
    assert not intel.supports(8, Datatype.F32, False)
    assert not intel.supports(32, Datatype.F32, False)


def test_fp64_has_no_dpas():
    """Not an omission.  XMX has no FP64, and emulating it from TF32 loses to
    PVC's own vector units -- ~419 TF of TF32 against ~52 TF of native FP64,
    with 53 mantissa bits needing about fifteen products."""
    assert not intel.supports(16, Datatype.F64, False)
    assert intel.atom_for(Datatype.F64) is None


def test_a_sparse_operand_is_still_servable():
    """By the target: `supports` is about the wave and the type.  Which
    arrangement serves a packed `B` is `strategies`' question -- and the
    answer there is neither (see below)."""
    assert intel.supports(16, Datatype.F32, True)


def test_fp32_is_emulated_through_tf32():
    assert intel.atom_for(Datatype.F32).name == 'tf32'
    assert intel.TF32_TERMS == 3
    # And it is that because of the split, not because a constant says so.
    assert intel.TF32_SPLIT_TERMS == 2
    assert intel.TF32_TERMS == len(split.products(intel.TF32_SPLIT_TERMS))


def test_dpas_is_not_offered_for_a_packed_operand():
    """The strategies drop out, not the target.

    DPAS has no fragment to read out of a packed `B`.  Nor has the broadcast
    chain a column to read: it asks `B(j, k0 // threads)`, the lane vector of
    depths `k0..` of column `j`, and a packed operand answers that with its
    storage slots from `k0 // threads` on -- slot 0 for every column of
    `damageCellIntegral`, which is a wrong product where it compiles and an
    IRError under ESIMD, where the read has no distribution to declare.  The
    nest reads a packed operand by its pattern, so it goes there."""
    from tensorforge.backend.instructions.compute.strategy import ComputeShape
    from tensorforge.backend.instructions.compute.strategy import Strategy
    shape = ComputeShape(threads=16, accumulator=Datatype.F32, sparse=True,
                         explicit_simd=True)
    assert Strategy.MATRIX not in intel.strategies(shape, None)
    assert Strategy.BROADCAST not in intel.strategies(shape, None)


# --------------------------------------------------------------------------
# `Options.tensor_cores` asks for the path per build
# --------------------------------------------------------------------------

def _ctx(backend='esimd', **options):
    from tensorforge.common.context import Context, Options
    return Context(arch='pvc', backend=backend, fp_type=Datatype.F32,
                   options=Options(**options))


def _dense(explicit_simd=True):
    return ComputeShape(threads=16, accumulator=Datatype.F32, sparse=False,
                        explicit_simd=explicit_simd)


def test_the_default_build_does_not_take_dpas(monkeypatch):
    """Unset, the option defers to `ENABLED` -- off -- so a default build is
    the one it was before the option reached this module."""
    monkeypatch.delenv('TF_TENSOR_CORES', raising=False)
    ctx = _ctx()
    assert ctx.get_user_options().tensor_cores is None
    assert not intel.enabled(ctx)
    assert Strategy.MATRIX not in intel.strategies(_dense(), ctx)
    assert Strategy.BROADCAST in intel.strategies(_dense(), ctx)


def test_the_option_asks_for_dpas():
    ctx = _ctx(tensor_cores=True)
    assert intel.enabled(ctx)
    assert Strategy.MATRIX in intel.strategies(_dense(), ctx)


def test_the_option_wins_over_the_module_default(monkeypatch):
    """Both ways: set, it is the answer whatever the constant says."""
    monkeypatch.setattr(intel, 'ENABLED', True)
    assert Strategy.MATRIX in intel.strategies(_dense(), None)
    assert Strategy.MATRIX not in intel.strategies(
        _dense(), _ctx(tensor_cores=False))


def test_spmd_never_takes_dpas():
    """`esimd::xmx::dpas` is ESIMD's, and an SPMD kernel cannot call it --
    yet the SPMD lowering reaches the gate with 16-wide multiplications too
    (20 cases of the corpus under `oneapi`, had the old gate been on)."""
    ctx = _ctx('oneapi', tensor_cores=True)
    assert Strategy.MATRIX not in intel.strategies(_dense(False), ctx)


def _kernel(case, backend, **options):
    import contextlib
    import importlib.util
    import io
    from pathlib import Path
    from tensorforge.generators.generator import Generator
    path = Path(__file__).parent / 'cases' / f'{case}.py'
    spec = importlib.util.spec_from_file_location(case, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with contextlib.redirect_stdout(io.StringIO()):
        gen = Generator(mod.descr_list(), _ctx(backend, **options))
        gen.generate()
    return gen.get_kernel()


def test_the_option_reaches_the_emitter(monkeypatch):
    """`gemm_square_16`: a 16x16 GEMM, one 16-wide multiplication.  Asked for,
    two column blocks of eight at three TF32 products each; not asked for,
    the same kernel as before; SPMD, never."""
    monkeypatch.delenv('TF_TENSOR_CORES', raising=False)
    on = _kernel('square_notrans', 'esimd', tensor_cores=True)
    assert on.count('intel_xmx::dpas<') == 2 * intel.TF32_TERMS * 2
    assert 'intel_xmx::dpas<' not in _kernel('square_notrans', 'esimd')
    assert 'intel_xmx::dpas<' not in _kernel('square_notrans', 'oneapi',
                                             tensor_cores=True)


# -- which lanes of B each block reads ------------------------------------- #

_SELECT = r'\w+\.template select<(\d+), 1>\((\d+)\)'
#: A Src2 fill: the run of B arrives inline, `(x.template select<W, 1>(S))`,
#: or through a name the emitter declared -- whether it materializes one is
#: its business, and either way it is a `select`.
_SRC2 = (r'splitFloatTF32<\d+>\(\(\w+_ahi\.template select<\d+, 1>\(\d+\)\), '
         r'\(\w+_alo\.template select<\d+, 1>\(\d+\)\), '
         r'(?:\(' + _SELECT + r'\)|(\w+))\);')
_DECLARED = r'simd<\w+, (\d+)> (\w+) = ' + _SELECT + ';'


def _src2_runs(case):
    """`(width, B lane)` of every Src2 fill, as emitted."""
    import re
    src = _kernel(case, 'esimd', tensor_cores=True)
    named = {name: (int(w), int(lane))
             for _, name, w, lane in re.findall(_DECLARED, src)}
    return [(int(w), int(lane)) if w else named.get(name)
            for w, lane, name in re.findall(_SRC2, src)]


def test_the_second_block_of_depths_reads_the_upper_lanes():
    """`gemm_square_16`: K = 16, so two blocks of eight out of one 16-lane
    vector of B.  The second starts at lane 8.  It started at lane 0 -- k = 0..7
    twice and k = 8..15 never -- and nothing on the host could tell."""
    runs = _src2_runs('square_notrans')
    assert runs, 'the Src2 fills are no longer where this test looks'
    assert set(runs) == {(8, 0), (8, 8)}


def test_a_ragged_block_reads_only_the_depths_that_exist():
    """`gemm_alpha_9x9`: K = 9, so the second block has one depth.  One lane of B,
    not eight: the other seven are the next row, or past the operand."""
    runs = _src2_runs('csa_alpha')
    assert set(runs) == {(8, 0), (1, 8)}


def test_every_lead_slot_is_multiplied():
    """`local_flux` at 16 lanes: a lead of 56 is four slots of sixteen rows.

    Per face, `X = A_f @ B` is two column blocks of eight over seven depth
    blocks, and `X @ C_f` two over two -- eighteen, at three TF32 products
    each, in every slot.  Only slot 0 used to be computed: a quarter of the
    products, the other 40 rows never stored.  And each operator is read in
    every slot: column 0 at rows 0, 16, 32 and 48.
    """
    import re
    src = _kernel('local_flux', 'esimd', tensor_cores=True, lanes_per_mult=16)
    slots, faces = 4, 4
    assert src.count('intel_xmx::dpas<') == (
        faces * slots * (2 * 7 + 2 * 2) * intel.TF32_TERMS)
    operators = {m for m in re.findall(r'glb_m\d+', src)
                 if re.search(m + r' \+ \(16(?:_i32)?\)', src)}
    assert len(operators) == faces
    for m in operators:
        rows = {int(x) for x in re.findall(m + r' \+ \((\d+)(?:_i32)?\)', src)}
        assert {0, 16, 32, 48} <= rows, m


# -- the prepared order: slot-major, and for DPAS the halves --------------- #

def _prepared(case, **options):
    """`(source, batch-constant operands)` of `case` under ESIMD on pvc."""
    import contextlib
    import importlib.util
    import io
    from pathlib import Path
    from tensorforge.common.basic_types import Addressing
    from tensorforge.generators.generator import Generator
    path = Path(__file__).parent / 'cases' / f'{case}.py'
    spec = importlib.util.spec_from_file_location(f'{case}_prepared', path)
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
        gen = Generator(mod.descr_list(), _ctx('esimd', **options))
        gen.generate()
    operands = [s.obj for s in gen._scopes.get_global_scope().values()
                if getattr(s.obj, 'addressing', None) is Addressing.NONE]
    return gen.get_kernel(), operands


def test_the_slot_major_order_is_a_padded_permutation():
    """Row `s*16 + l` of column `k` at `(s*depth + k)*16 + l`: every cell
    once, and the rows past 20 and the columns past 9 padding."""
    order = intel.slot_major_order((20, 9), 16, 16)
    assert len(order) == 2 * 16 * 16
    assert sorted(c for c in order if c != -1) == list(range(20 * 9))
    assert order[(1 * 16 + 3) * 16 + 1] == 17 + 20 * 3
    assert order[(1 * 16 + 3) * 16 + 4] == -1     # row 20
    assert order[(0 * 16 + 9) * 16 + 0] == -1     # column 9


def test_the_order_is_offered_where_an_arrangement_reads_it():
    """Under ESIMD at sixteen lanes and F32, which is where the broadcast
    chain and DPAS are; with DPAS the depths come in whole fragments and the
    operand in the two halves the three products multiply."""
    esimd = _ctx('esimd')
    fma = intel.prepared_order((56, 56), Datatype.F32, esimd, lead=56,
                               depth=56, threads=16)
    assert fma.slot_major == (16, 56) and fma.parts == 1
    dpas = intel.prepared_order((56, 35), Datatype.F32,
                                _ctx('esimd', tensor_cores=True), columns=9,
                                lead=56, depth=35, threads=16)
    assert dpas.slot_major == (16, 40) and dpas.parts == intel.TF32_SPLIT_TERMS
    assert len(dpas) == 4 * 40 * 16
    for shape, dtype, ctx, threads in (
            ((56, 56), Datatype.F32, _ctx('oneapi'), 16),
            ((56, 56), Datatype.F32, esimd, 32),
            ((56, 56), Datatype.F64, esimd, 16),
            ((4, 4, 4), Datatype.F32, esimd, 16)):
        assert intel.prepared_order(shape, dtype, ctx, threads=threads) is None


def test_a_prepared_operator_is_read_in_runs():
    """`local_flux` at 16 lanes: each 56x56 operator stored slot-major, and
    its 224 lane vectors read in 56 block messages of four -- where the
    broadcast chain read them a row stride apart, one message each."""
    import re
    src, operands = _prepared('local_flux', lanes_per_mult=16,
                              prepare_operands=True)
    assert len(operands) == 4
    for op in operands:
        assert op.slot_major == (16, 56) and op.storage_parts == 1
        assert len(op.storage_order) == 4 * 56 * 16
    runs = [len(re.findall(m + r'_run\d+;', src))
            for m in set(re.findall(r'(glb_m\d+)_run\d+;', src))]
    assert sorted(runs) == [56] * 4
    plain, _ = _prepared('local_flux', lanes_per_mult=16)
    assert 3 * src.count('copy_from') < plain.count('copy_from')


def test_dpas_reads_the_halves_it_was_stored_as():
    """With DPAS the operators are stored as their TF32 halves, planar.  Src1
    is converted from what it reads (`castTF32`) rather than split, and the
    lower halves are read a plane -- the order's length -- further on.  The
    view asks the operand for its part count: copied when the view was made,
    before the order split it, the lower half was read one float on."""
    import re
    src, operands = _prepared('local_flux', lanes_per_mult=16,
                              prepare_operands=True, tensor_cores=True)
    plane = 4 * 56 * 16
    for op in operands:
        assert op.storage_parts == 2 and op.storage_planar
        assert len(op.storage_order) == plane
    faces, blocks, slots, depth = 4, 2, 4, 56
    assert src.count('castTF32<16>') == faces * blocks * slots * depth * 2
    for m in set(re.findall(r'(glb_m\d+)_run\d+;', src)):
        offsets = {int(x) for x in
                   re.findall(m + r' \+ \((\d+)(?:_i32)?\)', src)}
        assert 0 in offsets and plane in offsets, m
        assert not offsets & {1, 2, 3}, m


def test_every_reader_of_an_operator_shares_its_order():
    """SeisSol's order-6 `derivative`: each `kDivMT` is read five times, once
    per derivative order, over boxes that shrink with it, and is stored from
    column 1 (the constant's derivative is zero).  The order is the tensor's
    -- `Symbol.load` rewrites every read -- so all five readers take it, and
    the box's column offset folds into the address."""
    import contextlib
    import io
    import seissol_suite as fx
    from tensorforge.common.basic_types import Addressing
    from tensorforge.frontend.yateto import DescriptionReader
    from tensorforge.generators.generator import Generator

    def generate(**options):
        system, config = 'elastic-linearck', 'elastic-linearck-o6-s'
        descrs = DescriptionReader(None, {}).read(
            fx.description(system, config, 'gpu_derivative'))[0]
        with contextlib.redirect_stdout(io.StringIO()):
            gen = Generator(descrs, _ctx('esimd', lanes_per_mult=16,
                                         **options))
            gen.generate()
        return gen.get_kernel(), [
            s.obj for s in gen._scopes.get_global_scope().values()
            if getattr(s.obj, 'addressing', None) is Addressing.NONE]

    src, operators = generate(prepare_operands=True)
    assert len(operators) == 3
    for op in operators:
        assert op.slot_major is not None, op.alias
        assert list(op.get_bbox().lower()) == [0, 1]
    plain, _ = generate()
    assert 3 * src.count('copy_from') < plain.count('copy_from')


@pytest.mark.parametrize('case_file', ['square_notrans', 'csa_alpha'])
def test_a_run_is_as_wide_as_its_select(case_file):
    """A run of B's lane vector was declared `simd<float, 16 * 8>` around an
    8-wide `select`: it inherited B's distribution over the lanes."""
    import re
    src = _kernel(case_file, 'esimd', tensor_cores=True)
    wrong = [(w, name, sel) for w, name, sel, _ in re.findall(_DECLARED, src)
             if w != sel]
    assert not wrong


# --------------------------------------------------------------------------
# the fragment layout, from vISA rather than from the SYCL header
# --------------------------------------------------------------------------

def _src1_operands_per_chan(ops_per_chan, precision_bits):
    """`SRC1_OPERANDS_PER_CHAN = 32 / (OPS_PER_CHAN * Src1PrecisionInBits)`"""
    return 32 // (ops_per_chan * precision_bits)


def test_tf32_leaves_nothing_for_src1_to_pack():
    """B's "special" layout is a packing, and a 32-bit element does not pack.

    `documentation/visa/instructions/DPAS.md` lays Src1 out over a 2-D view of
    the GRFs -- row = depth, DW column = n -- with several `k` sharing one DW
    for sub-dword types.  At one operand per channel the GRF-row index equals
    the depth and B comes out `B[k * N + n]`: ordinary row-major.
    """
    atom = intel.ATOMS['tf32']
    assert _src1_operands_per_chan(atom.ops_per_channel, atom.elem_bits) == 1


def test_sixteen_bit_types_do_pack():
    """Two `k` per DW, which is what makes Src1's layout worth describing at
    all -- and what this table would have to encode before bf16 is usable."""
    for name in ('bf16', 'fp16'):
        atom = intel.ATOMS[name]
        assert _src1_operands_per_chan(atom.ops_per_channel, atom.elem_bits) == 1
        assert atom.ops_per_channel == 2


def test_the_two_paths_are_flagged_separately():
    """They wait on different things.

    DPAS waits on a machine -- nothing here can check a systolic arrangement.
    The register-only path uses an element read and an FMA, which a front end
    does see, so it is on.
    """
    assert intel.ENABLED is False
    assert intel.BROADCAST_ENABLED is True


# --------------------------------------------------------------------------
# the register-only contraction, checked structurally
# --------------------------------------------------------------------------

def _esimd_kernel(name):
    import importlib.util, io, contextlib
    from pathlib import Path
    from tensorforge.common.context import Context
    from tensorforge.generators.generator import Generator
    p = Path(__file__).parent / 'cases' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, p)
    case = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(case)
    with contextlib.redirect_stdout(io.StringIO()):
        gen = Generator(case.descr_list(),
                        Context(arch='pvc', backend='esimd',
                                fp_type=getattr(case, 'DTYPE', Datatype.F32)))
        gen.generate()
    return gen.get_kernel()


def _products(src):
    """`acc += (B[lane] * A)` triples, by accumulator."""
    import re
    out = {}
    for m in re.finditer(r'(\w+_acc) \+= \(\((\w+)\[(\d+)\]\) \* (\w+)\)', src):
        out.setdefault(m.group(1), []).append(
            (m.group(2), int(m.group(3)), m.group(4)))
    return out


def test_the_contraction_is_complete_and_has_no_repeats():
    """A 16x16 GEMM: sixteen accumulators, each sweeping the whole K.

    This is the check that would have caught the `Mx` for `M` mix-up, which
    made every accumulator receive the same product -- an error nowhere, and
    wrong everywhere.
    """
    per = _products(_esimd_kernel('square_notrans'))
    assert len(per) == 16, 'one accumulator per output column'
    assert {len(v) for v in per.values()} == {16}, 'each sweeps the full K'
    for acc, prods in per.items():
        assert len(prods) == len(set(prods)), f'{acc} receives a product twice'


def test_every_accumulator_sweeps_the_same_contraction():
    per = _products(_esimd_kernel('square_notrans'))
    lanes = {tuple(sorted(l for _, l, _ in v)) for v in per.values()}
    avals = {tuple(sorted(a for _, _, a in v)) for v in per.values()}
    assert len(lanes) == 1 and len(avals) == 1


def test_one_b_vector_per_output_column():
    """B's contraction index lives in the lanes, so a column of the output is
    one vector broadcast lane by lane -- which is the whole arrangement."""
    per = _products(_esimd_kernel('square_notrans'))
    assert len({b for v in per.values() for b, _, _ in v}) == len(per)


# --------------------------------------------------------------------------
# the fragment layout, checked against what the instruction is supposed to do
# --------------------------------------------------------------------------

def _flatten(atom, A, B, C):
    a = [0.0] * atom.a_elems
    b = [0.0] * atom.b_elems
    c = [0.0] * atom.c_elems
    for m in range(atom.m):
        for k in range(atom.k):
            a[intel.a_offset(atom, m, k)] = A[m][k]
    for k in range(atom.k):
        for n in range(atom.n):
            b[intel.b_offset(atom, k, n)] = B[k][n]
    for m in range(atom.m):
        for n in range(atom.n):
            c[intel.c_offset(atom, m, n)] = C[m][n]
    return a, b, c


@pytest.mark.parametrize('name', sorted(intel.ATOMS))
def test_the_transcribed_instruction_is_a_matrix_product(name):
    """The check that makes the layout a fact rather than a reading.

    `reference` is the vISA pseudo-code transcribed, and the offset functions
    are how an (m, k) or (k, n) lands in a fragment.  Neither is verifiable on
    its own -- but if placing a matrix through the offsets and running the
    transcription gives `C + A @ B`, then both are right together, and that is
    exactly the thing no C++ front end can check.
    """
    import random
    atom = intel.ATOMS[name]
    random.seed(7)
    A = [[random.uniform(-1, 1) for _ in range(atom.k)] for _ in range(atom.m)]
    B = [[random.uniform(-1, 1) for _ in range(atom.n)] for _ in range(atom.k)]
    C = [[random.uniform(-1, 1) for _ in range(atom.n)] for _ in range(atom.m)]
    a, b, c = _flatten(atom, A, B, C)
    out = intel.reference(atom, c, b, a)
    for m in range(atom.m):
        for n in range(atom.n):
            want = C[m][n] + sum(A[m][k] * B[k][n] for k in range(atom.k))
            assert abs(out[intel.c_offset(atom, m, n)] - want) < 1e-12


def test_a_thirty_two_bit_operand_leaves_src1_row_major():
    """The packing is what makes Src1 "neither row-major nor column major",
    and a 32-bit element leaves nothing to pack."""
    atom = intel.ATOMS['tf32']
    for k in range(atom.k):
        for n in range(atom.n):
            assert intel.b_offset(atom, k, n) == k * atom.n + n


def test_a_sixteen_bit_operand_packs_two_depths_into_one_dword():
    """And then it is *not* row-major -- `B[0][n]` and `B[1][n]` are adjacent,
    which is the whole reason the spec spends a paragraph on Src1."""
    atom = intel.ATOMS['bf16']
    assert intel.b_offset(atom, 1, 0) - intel.b_offset(atom, 0, 0) == 1
    assert intel.b_offset(atom, 0, 1) - intel.b_offset(atom, 0, 0) == 2


def test_every_fragment_slot_is_used_exactly_once():
    """A permutation, not merely a map into the right range: a collision would
    silently drop an element and a gap would read an uninitialized one."""
    for name, atom in intel.ATOMS.items():
        for off, n_slots, dims in (
                (intel.a_offset, atom.a_elems, (atom.m, atom.k)),
                (intel.b_offset, atom.b_elems, (atom.k, atom.n)),
                (intel.c_offset, atom.c_elems, (atom.m, atom.n))):
            seen = sorted(off(atom, i, j)
                          for i in range(dims[0]) for j in range(dims[1]))
            assert seen == list(range(n_slots)), name


def test_tf32_is_its_own_type_and_four_bytes_wide():
    """A storage format, not an arithmetic one: values are converted into it
    and handed to an instruction, and nothing in the generator computes with
    it.  Spelling it `F32` is a lie no front end catches -- `simd<float, 128>`
    and `simd<tf32, 128>` are both well-formed and only one is the operand."""
    assert Datatype.TF32.size() == 4
    assert Datatype.TF32 is not Datatype.F32
    assert Datatype.TF32.ctype() == 'tensorforge::tf32'


def test_both_matrix_paths_name_the_same_type():
    """The NVIDIA halves used to be `U32` -- "four bytes of something", chosen
    because `splitFloatTF32` took `uint32_t&`.  It still does on CUDA, where
    the PTX constraint letter forces a typedef; what changed is that the
    *generator* now knows what those four bytes are."""
    from tensorforge.backend.instructions.compute.primitives import nvidia
    assert nvidia.TF32_HALF.base is Datatype.TF32


# -- the repeat count ------------------------------------------------------ #

def test_the_default_repeat_is_what_the_table_held():
    """The ranking has to be inert where nothing bounds it, or the change is
    not a refactor."""
    assert intel.atom_for(Datatype.F32).repeat == 8
    assert intel.atom_for(Datatype.F32, columns=9, lead=56, depth=56).repeat == 8


def test_issues_alone_always_return_the_widest_repeat():
    """Which is the point of saying so: nothing else in the count changes
    with `repeat`, so without a register bound the ranking is a constant."""
    for columns in (1, 5, 9, 16, 33):
        assert intel.atom_for(Datatype.F32, columns=columns).repeat == 8


def test_a_register_bound_is_what_makes_it_a_choice():
    """Eight columns of output per issue against one, and eight times the
    accumulator and Src2 to hold them."""
    sizes = {a.repeat: intel.fragment_bytes(a)
             for a in intel.atoms_for(Datatype.F32)}
    assert sizes[8] > sizes[4] > sizes[2] > sizes[1]
    assert intel.atom_for(Datatype.F32, columns=9,
                          budget=sizes[8]).repeat == 8
    assert intel.atom_for(Datatype.F32, columns=9,
                          budget=sizes[8] - 1).repeat == 4
    assert intel.atom_for(Datatype.F32, columns=9,
                          budget=sizes[1] - 1) is None


def test_only_the_repeats_the_header_admits():
    """`verify_repeat_count` takes 1, 2, 4 and 8."""
    assert set(intel.REPEATS) == {1, 2, 4, 8}
    assert {a.repeat for a in intel.atoms_for(Datatype.F32)} == set(intel.REPEATS)


def test_a_type_with_no_atom_has_no_candidates():
    assert intel.atoms_for(Datatype.F64) == ()
    assert intel.atom_for(Datatype.F64) is None
