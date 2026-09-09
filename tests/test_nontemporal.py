# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The nontemporal hint, against the overload set that has to accept it.

`__ldcg` and `__stcg` are an overload set, so a hint on a type outside it is
not a slower access but a compile error --- and one the host check cannot
see, because `g++` never reads the intrinsic's declaration.  The device front
end does see it, which is how `gemm_square_16_f128` was found, but it needs
`nvcc` present; these run anywhere.

What they pin is the *decision*, not the spelling: which types get a hint and
which are emitted plainly.  Whether the hint should be `.cg` or `.cs` is a
policy question above this, and changing the answer to it should not have to
rewrite these.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tensorforge.common.basic_types import Datatype
from tensorforge.common.vm.lexic.cuda_lexic import CudaLexic
from tensorforge.common.vm.lexic.hip_lexic import HipLexic

CASES = Path(__file__).parent / "cases"


def _cuda():
    return CudaLexic("cuda", "nvidia")


def _hip(hardware="amd"):
    return HipLexic("hip", hardware)


# --- the gate -------------------------------------------------------------

@pytest.mark.parametrize("datatype", [Datatype.F32, Datatype.F64,
                                      Datatype.I32, Datatype.I64,
                                      Datatype.U32, Datatype.TF32])
def test_cuda_hints_the_types_the_intrinsic_declares(datatype):
    lex = _cuda()
    assert lex.has_nontemporal(datatype)
    assert lex.glb_load("g[i]", datatype=datatype,
                        nontemporal=True) == "__ldcg(&g[i])"
    assert lex.glb_store("g[i]", "v", datatype=datatype,
                         nontemporal=True) == "__stcg(&g[i], v);"


@pytest.mark.parametrize("datatype", [Datatype.F128, Datatype.BOOL,
                                      Datatype.F16, Datatype.BF16])
def test_cuda_emits_a_plain_access_for_the_rest(datatype):
    """No overload, so the hint is dropped rather than spelled.

    `__float128` is the one that reached a front end: 16 bytes, no `__ldcg`
    declared over it at any architecture, and the error names the argument
    list rather than the type, which is what made it look like an
    architecture fact next to the other reason that case is refused.
    """
    lex = _cuda()
    assert not lex.has_nontemporal(datatype)
    assert lex.glb_load("g[i]", datatype=datatype,
                        nontemporal=True) == "g[i]"
    assert lex.glb_store("g[i]", "v", datatype=datatype,
                         nontemporal=True) == "g[i] = v;"


@pytest.mark.parametrize("length", [2, 4])
def test_cuda_declines_a_wide_access(length):
    """A wide value is a `VectorT`, which the overloads are not declared over.

    Not hypothetical and not only about the overloads: `_write_hop` spells
    both sides of a staged transfer as `*(VectorT<T, N>*)&...` above a width
    of one, and a GNU vector value does not survive the device front end at
    all.  Reaching a hint here needs a different value type, not a different
    intrinsic.
    """
    lex = _cuda()
    assert not lex.has_nontemporal(Datatype.F32, length)
    assert lex.glb_load("*(tensorforge::VectorT<float, 4>*)&g[i]",
                        datatype=Datatype.F32, length=length,
                        nontemporal=True) == \
        "*(tensorforge::VectorT<float, 4>*)&g[i]"


def test_hip_hints_every_type_on_amd():
    """The builtins are generic, so the type is not the question there."""
    lex = _hip()
    for datatype in (Datatype.F32, Datatype.F128):
        assert lex.has_nontemporal(datatype)
        assert lex.glb_load("g[i]", datatype=datatype, nontemporal=True) == \
            "__builtin_nontemporal_load(&g[i])"
    assert lex.has_nontemporal(Datatype.F32, 4)


def test_hip_on_nvidia_hardware_has_neither():
    """HIP compiles for NVIDIA, where the builtins are not declared."""
    lex = _hip("nvidia")
    assert not lex.has_nontemporal(Datatype.F32)
    assert lex.glb_load("g[i]", datatype=Datatype.F32,
                        nontemporal=True) == "g[i]"


def test_the_type_cannot_be_left_out():
    """Keyword-only and no default: a call site that forgets is a TypeError.

    The defect this replaces was a `glb_load` that never asked, so every
    caller was silently right until one of them loaded a type the intrinsic
    does not take.  A default would restore exactly that.
    """
    with pytest.raises(TypeError):
        _cuda().glb_load("g[i]", True)          # noqa: FBT003


# --- end to end -----------------------------------------------------------

def _kernel(case_name, backend, arch):
    from tensorforge.common.context import Context
    from tensorforge.generators.generator import Generator
    spec = importlib.util.spec_from_file_location(
        case_name, CASES / f"{case_name}.py")
    case = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(case)
    ctx = Context(arch=arch, backend=backend,
                  fp_type=getattr(case, "DTYPE", None))
    gen = Generator(case.descr_list(), ctx, attrs=getattr(case, "ATTRS", None))
    gen.generate()
    return gen.get_kernel()


def test_f128_kernel_carries_no_cache_hint():
    assert "__ldcg" not in _kernel("f128", "cuda", "sm_86")


def test_f128_keeps_its_hint_on_amd():
    """The gate is CUDA's, and the AMD path is not collateral."""
    assert "__builtin_nontemporal_load" in _kernel("f128", "hip", "gfx90a")


def test_an_f32_kernel_still_gets_one():
    """The other half of the claim: the gate turned away one type, not all.

    Without this, dropping every hint would pass the test above.
    """
    assert "__ldcg" in _kernel("aligned_operands", "cuda", "sm_86")
