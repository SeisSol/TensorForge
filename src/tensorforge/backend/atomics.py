# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Which atomic updates the hardware performs, and which it emulates.

An atomic accumulation is chosen for speed: it lets a store go out on its own
instead of waiting for the tensor's other writers, which is what
`placement.atomic_accumulation` is about.  That reasoning holds only where the
update is one instruction.  Where the target has no instruction for it the
compiler emits a compare-and-swap loop, which is slower than the
read-modify-write the atomic was chosen to avoid -- so "does this target have
one" is not a detail of the spelling, it is the premise of the decision, and
it belongs where the decision is made rather than inside the lexic that writes
the call down.

The question is asked per `(vendor, architecture, datatype)`.  Not per vendor:
`atomic_accumulation=True` was a row of the vendor table, and the vendor is the
one thing that does not decide it.  Every AMD target took the AMD row, and the
row emitted `__builtin_amdgcn_global_atomic_fadd_f32` -- a builtin gated on
`atomic-fadd-rtn-insts`, which gfx900, gfx906, gfx1010 and gfx1030 do not have
and gfx908 has only in its non-returning form.  Four of the thirteen
architectures in `hw_descr_db.yml` could not compile the kernel they were
handed, and a fifth compiled a different instruction than the one named.

What is *not* modelled here, and why:

* **Shared memory.**  Nothing reaches an atomic on a non-global symbol today,
  and `Symbol.store` now says so rather than silently writing a plain
  assignment.  When it does, the AMD gates are `lds-atomic-add-f64` for f64
  and nothing at all for f32 -- `ds_add_f32` predates every target here.
* **Operations other than add.**  `Symbol.store` passes `op=None` and every
  lexic writes an add.  Min and max exist on both vendors and on different
  targets again (`atomic-fmin-fmax-global-f32` reaches RDNA1, which has no
  add at all), so the parameter is threaded through to keep the question
  askable, and answered only for addition.

**Width** is modelled, and answering it honestly is what lets
`placement.atomic_write_is_exact` stop standing in for it.  That condition
refused every widened lead for two reasons at once -- a peeled tail element no
lane owns, and a wide value handed to a scalar instruction -- and only the
first is a fact about the nest.  The second is this table's question.

The answer still refuses everything actually reached, and by its own content
rather than by a switch: AMD is the one vendor whose policy asks for atomics,
and AMD has a packed add for f16 and bf16 and none for f32 or f64.  There is
no `global_atomic_pk_add_f32` -- no builtin, and no subtarget feature to gate
one on.  So lifting this is hardware arriving, not an edit here.

The fine-grained memory question is a separate one and is *not* answered by
this module, because it is not a property of the architecture.  AMD's hardware
FP atomics are processed in the L2 and therefore work on coarse-grained
memory; on fine-grained memory they are forwarded to the Infinity Fabric,
which has no FP32 add, and the operation becomes a no-op that silently returns
the wrong result.  The compiler will not emit the instruction without an
assurance that the pointer is not fine-grained -- `-fatomic-fine-grained-memory`
and `-fatomic-ignore-denormal-mode` (the old `-munsafe-fp-atomics`), or
`[[clang::atomic]]` per statement -- except on the targets carrying
`agent-scope-fine-grained-remote-memory-atomics`.  Which allocator the output
buffer came from is the caller's fact, not ours; `unsafe_fp_atomics_required`
states where the assurance is what stands between the fallback spelling and a
compare-and-swap loop.

gfx1250 and gfx1251 are the exception to the silent part, and only to that
part: they carry `emulated-system-scope-atomics`, which LLVM describes as
system-scope atomics the PCI-e cannot do being emulated in hardware by a CAS
loop and remaining functional.  So the failure mode there is slowness rather
than a wrong answer -- which is a reason to keep asking for agent scope, not a
reason to stop.  What has *not* changed on those two, re-checked against LLVM
main: `__builtin_amdgcn_global_atomic_fadd_f64` is still gated on
`gfx90a-insts`, and `gfx90a-insts` is still gfx90a, gfx942 and gfx950 alone.
They have the f64 instruction under `flat-buffer-global-fadd-f64-inst` and no
builtin that reaches it, so the fallback spelling is the only way in.
"""

from tensorforge.common.basic_types import Datatype


def _model(ctx) -> str:
    return ctx.get_vm().get_hw_descr().model


def _vendor(ctx) -> str:
    return ctx.get_vm().get_hw_descr().vendor


# --------------------------------------------------------------------------- #
# NVIDIA
# --------------------------------------------------------------------------- #

def _sm(ctx) -> int:
    """`sm_80` -> 80.  Three digits from sm_100 on, so this is not a slice."""
    return int(_model(ctx)[3:])


#: `(datatype, length)` -> the compute capability that has it.
#:
#: The half formats are *only* packed: `atomicAdd(__half *)` does not exist,
#: `atomicAdd(__half2 *)` does, and the same for bf16.  So the width axis is
#: not a widening of the scalar table here -- it is where two of the types
#: live at all, which is why the two are one table rather than a table and a
#: multiplier.
#:
#: `float2` and `float4` arrived with sm_90 and are global-memory only, with
#: atomicity guaranteed per component.  Per component is all an accumulation
#: needs: it adds to each element independently and nothing reads the pair
#: back as a unit.
_NVIDIA_ADD = {
    (Datatype.F32, 1): 20,
    (Datatype.F64, 1): 60,
    (Datatype.F16, 2): 60,
    (Datatype.BF16, 2): 80,
    (Datatype.F32, 2): 90,
    (Datatype.F32, 4): 90,
}


def _nvidia_add(ctx, datatype, length) -> bool:
    need = _NVIDIA_ADD.get((datatype, length))
    return need is not None and _sm(ctx) >= need


# --------------------------------------------------------------------------- #
# AMD
# --------------------------------------------------------------------------- #

#: What an *instruction* needs, which is not what the *builtin* needs.  The
#: non-returning form is the one to ask for: an accumulation throws the old
#: value away, and gfx908 has only that form.
#: Keyed by `(datatype, length)` for the reason the NVIDIA table is: the
#: packed 16-bit adds are the only form those types have, and there is no
#: packed f32 or f64 entry to widen to -- `global_atomic_pk_add_f32` is not a
#: builtin and no subtarget feature gates one.
_AMD_ADD_FEATURE = {
    (Datatype.F32, 1): 'atomic-fadd-no-rtn-insts',
    (Datatype.F64, 1): 'flat-buffer-global-fadd-f64-inst',
    (Datatype.F16, 2): 'atomic-buffer-global-pk-add-f16-insts',
    (Datatype.BF16, 2): 'atomic-global-pk-add-bf16-inst',
}

#: The builtin that *is* the instruction, and the feature clang gates it on.
#: Narrower than the table above in both directions: `..._f32` is gated on the
#: returning feature, so gfx908 has the instruction and not the builtin; and
#: `..._f64` is gated on `gfx90a-insts`, which gfx1250 and gfx1251 are not in
#: although they carry `flat-buffer-global-fadd-f64-inst`.  Where a target has
#: the instruction and no builtin, `__hip_atomic_fetch_add` reaches it -- given
#: the fine-grained assurance, which is the subject of
#: `unsafe_fp_atomics_required`.
_AMD_ADD_BUILTIN = {
    Datatype.F32: ('__builtin_amdgcn_global_atomic_fadd_f32',
                   'atomic-fadd-rtn-insts'),
    Datatype.F64: ('__builtin_amdgcn_global_atomic_fadd_f64',
                   'gfx90a-insts'),
}


def _amd_add(ctx, datatype, length) -> bool:
    from tensorforge.backend.instructions.compute.primitives.amd import (
        has_feature)
    feature = _AMD_ADD_FEATURE.get((datatype, length))
    return feature is not None and has_feature(ctx, feature)


def amd_add_builtin(ctx, datatype):
    """The intrinsic spelling, or None where this target has no builtin.

    None is not "no atomic": it means the guaranteed spelling is unavailable
    and the caller falls back to `__hip_atomic_fetch_add`, which the backend
    lowers to the same instruction under the conditions the module docstring
    sets out.
    """
    from tensorforge.backend.instructions.compute.primitives.amd import (
        has_feature)
    entry = _AMD_ADD_BUILTIN.get(datatype)
    if entry is None:
        return None
    builtin, feature = entry
    return builtin if has_feature(ctx, feature) else None


def unsafe_fp_atomics_required(ctx, datatype) -> bool:
    """Whether the fallback spelling needs the fine-grained assurance here.

    False on two counts and they are different counts: because the builtin is
    available and the assurance is not what decides, or because the target
    carries `agent-scope-fine-grained-remote-memory-atomics` and the compiler
    emits the instruction without being told anything.
    """
    from tensorforge.backend.instructions.compute.primitives.amd import (
        has_feature)
    if _vendor(ctx) != 'amd' or not _amd_add(ctx, datatype, 1):
        return False
    if amd_add_builtin(ctx, datatype) is not None:
        return False
    return not has_feature(ctx, 'agent-scope-fine-grained-remote-memory-atomics')


# --------------------------------------------------------------------------- #
# Intel
# --------------------------------------------------------------------------- #

#: Xe-HPC is the one target here with a native FP64 atomic add.  Everything
#: else emulates it, and an emulated one is what this gate exists to refuse.
_INTEL_NATIVE_F64 = ('pvc',)


def _intel_add(ctx, datatype, length) -> bool:
    # Scalar only, and that is the SPMD spelling rather than the hardware:
    # `sycl::atomic_ref` binds one reference to one element and has no packed
    # form.  A wide update under ESIMD is `atomic_update` over a `simd<T, N>`,
    # which is a different emitter and answers for itself in `SyclLexic`.
    if length != 1:
        return False
    if datatype is Datatype.F32:
        return True
    if datatype is Datatype.F64:
        return _model(ctx) in _INTEL_NATIVE_F64
    return False


# --------------------------------------------------------------------------- #

_VENDORS = {'nvidia': _nvidia_add, 'amd': _amd_add, 'intel': _intel_add}


def native_add(ctx, datatype, length: int = 1) -> bool:
    """Does this target add `length` adjacent elements in one instruction?

    A vendor with no entry answers False, which costs a preference and never
    correctness: the accumulation goes out as an ordinary read-modify-write,
    which is what every target did before atomics existed here.

    `length` defaults to 1 so a caller that has no width to offer keeps
    asking the question it was asking.  It is *not* a hint: a target with a
    scalar add and no packed one answers False for 2, rather than yes with a
    silent fallback to two scalar updates.  Splitting a wide value is the
    store path's decision and it has the information to make it; making it
    here would hide a doubled instruction count behind a capability query.
    """
    return _VENDORS.get(_vendor(ctx), lambda *_: False)(ctx, datatype, length)
