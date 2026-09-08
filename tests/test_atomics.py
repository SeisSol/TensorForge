# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""An atomic accumulation is offered where the instruction exists.

`atomic_accumulation` was a field of the vendor table, so every AMD target
answered as gfx90a does.  Four of the thirteen architectures in
`hw_descr_db.yml` have no floating-point atomic add of any kind and were
handed `__builtin_amdgcn_global_atomic_fadd_f32` regardless -- not a slow
kernel, a kernel that does not compile.  A fifth, gfx908, has the instruction
in its non-returning form only, which is the form an accumulation wants and
the one the returning builtin cannot reach.

The other half of what this checks is that the path exists at all on the other
two vendors.  It did not: the single call site passes four arguments,
`CudaLexic.atomic_store` took three and `SyclLexic` had none, so NVIDIA raised
`TypeError` and Intel `AttributeError` the moment either was asked.  Tests
that assert a spelling would have passed while that was true -- what catches
it is going through the interface the builder uses, which is why the lexic
tests below call `has_atomic_store` first and then `atomic_store`, in that
order and with the same arguments the builder passes.
"""

from __future__ import annotations

import pytest

from tensorforge.backend import atomics
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context


AMD_ARCHS = ["gfx900", "gfx906", "gfx908", "gfx90a", "gfx940", "gfx942",
             "gfx950", "gfx1010", "gfx1030", "gfx1100", "gfx1200", "gfx1250",
             "gfx1251"]

#: The AMD targets with a global FP32 atomic add, in either form.  gfx900 and
#: gfx906 are GCN, gfx1010 and gfx1030 are RDNA1 and RDNA2 -- the add arrives
#: with RDNA3, although `atomic-fmin-fmax-global-f32` reaches back further,
#: which is why a "has atomics" predicate written per family gets it wrong.
AMD_F32 = {"gfx908", "gfx90a", "gfx940", "gfx942", "gfx950", "gfx1100",
           "gfx1200", "gfx1250", "gfx1251"}

#: And with a global FP64 one.  CDNA2 up, plus gfx125x; no RDNA generation.
AMD_F64 = {"gfx90a", "gfx940", "gfx942", "gfx950", "gfx1250", "gfx1251"}


def _ctx(arch, backend="hip", dtype=Datatype.F32):
    return Context(arch=arch, backend=backend, fp_type=dtype)


# --------------------------------------------------------------------------- #
# What the hardware has
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("arch", AMD_ARCHS)
def test_amd_f32_follows_the_subtarget_feature(arch):
    assert atomics.native_add(_ctx(arch), Datatype.F32) is (arch in AMD_F32)


@pytest.mark.parametrize("arch", AMD_ARCHS)
def test_amd_f64_follows_the_subtarget_feature(arch):
    ctx = _ctx(arch, dtype=Datatype.F64)
    assert atomics.native_add(ctx, Datatype.F64) is (arch in AMD_F64)


def test_gfx908_has_the_instruction_and_not_the_builtin():
    """The returning and non-returning forms are separate features.

    An accumulation throws the old value away, so the non-returning form is
    the one it wants -- and `__builtin_amdgcn_global_atomic_fadd_f32` returns,
    so on the one target carrying only that form the builtin is a compile
    error while the instruction is right there.  `__hip_atomic_fetch_add`
    reaches it.
    """
    ctx = _ctx("gfx908")
    assert atomics.native_add(ctx, Datatype.F32)
    assert atomics.amd_add_builtin(ctx, Datatype.F32) is None


@pytest.mark.parametrize("arch", ["gfx1250", "gfx1251"])
def test_gfx125x_has_the_f64_instruction_and_not_the_builtin(arch):
    """The same split, from the other side.

    `flat-buffer-global-fadd-f64-inst` covers these two; the builtin is gated
    on `gfx90a-insts`, which does not.  Reading the builtin's availability off
    the instruction's feature would emit an undeclared name.
    """
    ctx = _ctx(arch, dtype=Datatype.F64)
    assert atomics.native_add(ctx, Datatype.F64)
    assert atomics.amd_add_builtin(ctx, Datatype.F64) is None


@pytest.mark.parametrize("arch", sorted(AMD_F32))
def test_a_builtin_is_only_named_where_its_feature_is_present(arch):
    """No spelling is offered that the target's feature set does not gate."""
    from tensorforge.backend.instructions.compute.primitives.amd import (
        has_feature)
    ctx = _ctx(arch)
    named = atomics.amd_add_builtin(ctx, Datatype.F32) is not None
    assert named is has_feature(ctx, "atomic-fadd-rtn-insts")


# --------------------------------------------------------------------------- #
# What the caller still has to promise
# --------------------------------------------------------------------------- #

def test_the_assurance_is_asked_for_where_the_fallback_needs_it():
    """gfx908 reaches its instruction only through the fallback spelling.

    Which the compiler lowers to a compare-and-swap loop unless it is told the
    pointer is not fine-grained -- and gfx908 does not carry
    `agent-scope-fine-grained-remote-memory-atomics`, so the promise has to
    come from the build.
    """
    assert atomics.unsafe_fp_atomics_required(_ctx("gfx908"), Datatype.F32)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950", "gfx1200", "gfx1250"])
def test_agent_scope_fine_grained_targets_need_no_assurance(arch):
    assert not atomics.unsafe_fp_atomics_required(_ctx(arch), Datatype.F32)


def test_a_target_with_a_builtin_needs_no_assurance():
    """gfx90a lacks the fine-grained feature and still answers False.

    Two different reasons for the same answer, and this is the other one: the
    builtin *is* the instruction, so nothing about the lowering is left for a
    flag to decide.
    """
    from tensorforge.backend.instructions.compute.primitives.amd import (
        has_feature)
    ctx = _ctx("gfx90a")
    assert not has_feature(ctx, "agent-scope-fine-grained-remote-memory-atomics")
    assert atomics.amd_add_builtin(ctx, Datatype.F32) is not None
    assert not atomics.unsafe_fp_atomics_required(ctx, Datatype.F32)


@pytest.mark.parametrize("arch", ["gfx900", "gfx1030"])
def test_a_target_without_the_instruction_is_not_asked_for_one(arch):
    """Nothing to promise where nothing will be emitted."""
    assert not atomics.unsafe_fp_atomics_required(_ctx(arch), Datatype.F32)


# --------------------------------------------------------------------------- #
# The other two vendors
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("arch", ["sm_60", "sm_80", "sm_90", "sm_100"])
@pytest.mark.parametrize("dtype", [Datatype.F32, Datatype.F64])
def test_nvidia_adds_both_float_types(arch, dtype):
    assert atomics.native_add(_ctx(arch, "cuda", dtype), dtype)


def test_nvidia_declines_a_half_precision_scalar():
    """`atomicAdd(__half *)` does not exist.

    The half forms are `__half2` and `__nv_bfloat162`, which are the width
    axis and not a scalar the text store path can hand over -- so the answer
    is no until that path carries a width, rather than a spelling that
    silently is not the type asked for.

    Asked of an F32 context, because the kernel's `fp_type` and the type of
    the destination being stored are two things and only the second one is the
    question here.
    """
    assert not atomics.native_add(_ctx("sm_90", "cuda"), Datatype.F16)


def test_intel_has_a_native_f64_add_on_pvc_only():
    """Xe-HPC has it; Xe-HPG emulates, and an emulated one is what the gate
    exists to refuse."""
    assert atomics.native_add(_ctx("pvc", "oneapi", Datatype.F64), Datatype.F64)
    assert not atomics.native_add(_ctx("dg1", "oneapi", Datatype.F64),
                                  Datatype.F64)
    assert atomics.native_add(_ctx("dg1", "oneapi"), Datatype.F32)


# --------------------------------------------------------------------------- #
# The lexics, through the interface the builder uses
# --------------------------------------------------------------------------- #

LEXIC_CASES = [
    ("sm_90", "cuda", Datatype.F32, "atomicAdd"),
    ("sm_90", "cuda", Datatype.F64, "atomicAdd"),
    ("gfx90a", "hip", Datatype.F32, "__builtin_amdgcn_global_atomic_fadd_f32"),
    ("gfx90a", "hip", Datatype.F64, "__builtin_amdgcn_global_atomic_fadd_f64"),
    ("gfx908", "hip", Datatype.F32, "__hip_atomic_fetch_add"),
    ("gfx1250", "hip", Datatype.F64, "__hip_atomic_fetch_add"),
    ("pvc", "oneapi", Datatype.F32, "sycl::atomic_ref"),
]


@pytest.mark.parametrize("arch,backend,dtype,expected", LEXIC_CASES,
                         ids=[f"{a}-{d.name.lower()}"
                              for a, _, d, _ in LEXIC_CASES])
def test_the_offered_spelling_is_the_one_for_this_target(arch, backend, dtype,
                                                         expected):
    ctx = _ctx(arch, backend, dtype)
    lexic = ctx.get_vm().get_lexic()
    assert lexic.has_atomic_store(ctx, None, dtype)
    stmt = lexic.atomic_store(ctx, "glb[i]", "value", None, dtype)
    assert expected in stmt
    assert stmt.endswith(';'), 'a store is a statement, not an expression'


@pytest.mark.parametrize("arch", ["gfx900", "gfx1030"])
def test_a_hip_target_without_the_instruction_is_not_offered_one(arch):
    ctx = _ctx(arch)
    assert not ctx.get_vm().get_lexic().has_atomic_store(ctx, None,
                                                         Datatype.F32)


def test_the_result_of_the_cuda_add_is_dropped():
    """Which is what makes ptxas emit `RED` rather than `ATOM`.

    Binding the old value costs the reduction form at the same address and for
    nothing, so the spelling has to be a statement with no assignment in it.
    """
    ctx = _ctx("sm_90", "cuda")
    stmt = ctx.get_vm().get_lexic().atomic_store(ctx, "glb[i]", "value", None,
                                                 Datatype.F32)
    assert '=' not in stmt


def test_the_hip_fallback_is_agent_scoped_and_relaxed():
    """System scope is what makes the backend give up on the instruction."""
    ctx = _ctx("gfx908")
    stmt = ctx.get_vm().get_lexic().atomic_store(ctx, "glb[i]", "value", None,
                                                 Datatype.F32)
    assert '__HIP_MEMORY_SCOPE_AGENT' in stmt
    assert '__ATOMIC_RELAXED' in stmt


def test_hip_targeting_nvidia_does_not_emit_amd_builtins():
    """HIP compiles for CUDA too, where every `__builtin_amdgcn_*` is
    undeclared.  `glb_store` next door has always had this condition."""
    ctx = _ctx("sm_80", "hip")
    stmt = ctx.get_vm().get_lexic().atomic_store(ctx, "glb[i]", "value", None,
                                                 Datatype.F32)
    assert 'amdgcn' not in stmt
    assert 'atomicAdd' in stmt


def test_explicit_simd_declines_whatever_the_hardware_can_do():
    """`atomic_ref` binds one reference to one element and an ESIMD store
    carries a `simd<T, N>` with a mask beside it.  The instruction for that is
    `esimd::atomic_update`, which is a different emitter; until it exists,
    refusing is what keeps the kernel compiling."""
    spmd = _ctx("pvc", "oneapi")
    esimd = _ctx("pvc", "esimd")
    assert spmd.get_vm().get_lexic().has_atomic_store(spmd, None, Datatype.F32)
    assert not esimd.get_vm().get_lexic().has_atomic_store(esimd, None,
                                                           Datatype.F32)
