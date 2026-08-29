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


def test_a_sparse_operand_is_not_a_fragment():
    assert not intel.supports(16, Datatype.F32, True)


def test_fp32_is_emulated_through_tf32():
    assert intel.atom_for(Datatype.F32) is intel.ATOMS['tf32']
    assert intel.TF32_TERMS == 3


def test_a_sparse_operand_falls_through():
    """The register path contracts over B's lanes, and a sparse operand is
    loaded by linear index rather than as a lane-distributed vector.  Declining
    sends the caller to the generic path, which handles it."""
    ops = MatmulOperands(A=None, B=None, C=None,
                         sparse=lambda k, j: True,
                         lead_slots=1, lead_elements=16, n=1, k=1, kx=0,
                         threads=16, dtype=Datatype.F32)
    assert intel.matmul(None, ops, None) is False


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
    silently drop an element and a gap would read an uninitialised one."""
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
