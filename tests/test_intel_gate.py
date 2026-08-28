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


def test_matmul_declines_while_parked():
    """`_is_matmul` asks `ENABLED` first, so this is unreachable -- but a path
    that is not deployed must decline rather than emit if something does reach
    it, so that the caller falls through to the generic path."""
    assert intel.matmul(None, None, None, None, 1, 1, 1, 0, 16,
                        Datatype.F32, None, None) is False
