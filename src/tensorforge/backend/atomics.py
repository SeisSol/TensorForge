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

* **Width.**  `red.global.add.v2.f32` and `.v4.f32` exist from sm_90 and cut
  the number of atomic transactions by two or four.  They are the natural
  extension of the packed-FMA work and want the same `ScalarType.length` the
  store path already carries; the text store path this reaches does not carry
  it yet, so a width parameter here would have no caller.
* **Shared memory.**  Nothing reaches an atomic on a non-global symbol today,
  and `Symbol.store` now says so rather than silently writing a plain
  assignment.  When it does, the AMD gates are `lds-atomic-add-f64` for f64
  and nothing at all for f32 -- `ds_add_f32` predates every target here.
* **Operations other than add.**  `Symbol.store` passes `op=None` and every
  lexic writes an add.  Min and max exist on both vendors and on different
  targets again (`atomic-fmin-fmax-global-f32` reaches RDNA1, which has no
  add at all), so the parameter is threaded through to keep the question
  askable, and answered only for addition.

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


def _nvidia_add(ctx, datatype) -> bool:
    if datatype is Datatype.F32:
        return True                 # sm_20 and up; every row of the table
    if datatype is Datatype.F64:
        return _sm(ctx) >= 60
    # `atomicAdd(__half *)` does not exist -- the half-precision forms are
    # `__half2` and `__nv_bfloat162`, which are the width axis above and not a
    # scalar the store path can hand over.
    return False


# --------------------------------------------------------------------------- #
# AMD
# --------------------------------------------------------------------------- #

#: What an *instruction* needs, which is not what the *builtin* needs.  The
#: non-returning form is the one to ask for: an accumulation throws the old
#: value away, and gfx908 has only that form.
_AMD_ADD_FEATURE = {
    Datatype.F32: 'atomic-fadd-no-rtn-insts',
    Datatype.F64: 'flat-buffer-global-fadd-f64-inst',
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


def _amd_add(ctx, datatype) -> bool:
    from tensorforge.backend.instructions.compute.primitives.amd import (
        has_feature)
    feature = _AMD_ADD_FEATURE.get(datatype)
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
    if _vendor(ctx) != 'amd' or not _amd_add(ctx, datatype):
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


def _intel_add(ctx, datatype) -> bool:
    if datatype is Datatype.F32:
        return True
    if datatype is Datatype.F64:
        return _model(ctx) in _INTEL_NATIVE_F64
    return False


# --------------------------------------------------------------------------- #

_VENDORS = {'nvidia': _nvidia_add, 'amd': _amd_add, 'intel': _intel_add}


def native_add(ctx, datatype) -> bool:
    """Does this target add to global memory in one instruction?

    A vendor with no entry answers False, which costs a preference and never
    correctness: the accumulation goes out as an ordinary read-modify-write,
    which is what every target did before atomics existed here.
    """
    return _VENDORS.get(_vendor(ctx), lambda *_: False)(ctx, datatype)
