# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""An element's address with parameters of its own, and through an index list.

`Addressing.INDEXED` reaches element `n` of a launch at `base + index[n] *
stride + offset`, `INDEXED_POINTER` at `vars[index[n] * slots + slot] +
offset`, both through one index list per launch (`elementIndices`).  A
strided operand may take a stride and an offset of its own, and a pointer
based one an offset (`ElementAddressing`); a stride or a slot may be bound
at run time (`RUNTIME`).

Checked here: the parameters and their refusals; that the defaults render
byte for byte as before; the addresses each target spells; the signature,
the launcher and the call site yateto's struct is bound with; the interface
version and the binding a description states; the numbers, on the host
oracle against NumPy -- over a permuted and a repeating index list -- and,
where a GPU is there, a SeisSol kernel built both ways on the device.
"""

from __future__ import annotations

import contextlib
import copy
import io
import re

import numpy as np
import pytest

from tensorforge.common.basic_types import (RUNTIME, Addressing, Datatype,
                                            ElementAddressing,
                                            element_addressing, indexed,
                                            indexed_pointer, pointer_based,
                                            strided)
from tensorforge.common.context import Context
from tensorforge.common.exceptions import GenerationError
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.common.options import Options
from tensorforge.generators.descriptions import (GemmDescr, GridBarrierDescr)
from tensorforge.generators.generator import Generator
from tensorforge.reference import kernel_eval

DTYPE = Datatype.F32
M = 8

#: The targets a binding is spelled for: CUDA, HIP, SYCL's SPMD lowering on
#: both of its compilers, and the explicitly vectorized one.
TARGETS = [("cuda", "sm_86"), ("cuda", "sm_120"), ("hip", "gfx942"),
           ("hip", "gfx1150"), ("oneapi", "pvc"), ("acpp", "pvc"),
           ("esimd", "pvc")]


def _gemm(addressings, shape=(M, M), names="ABC", beta=0.0):
    tensors = [Tensor(list(shape), addressing,
                      BoundingBox([0, 0], list(shape)), alias=name,
                      datatype=DTYPE)
               for name, addressing in zip(names, addressings)]
    a, b, c = (SubTensor(t) for t in tensors)
    return [GemmDescr(False, False, a, b, c, alpha=1.0, beta=beta)]


def _generate(descrs, backend="cuda", arch="sm_86", **options):
    ctx = Context(arch=arch, backend=backend, fp_type=DTYPE,
                  options=Options(**options))
    gen = Generator(descrs, ctx)
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    return gen


def _surface(gen):
    return "\n".join([gen.get_header() or "", gen.get_launcher() or "",
                      gen.get_kernel() or ""])


#: A runtime stride, an indexed pointer with a runtime slot and an offset,
#: and a strided destination with a stride and an offset of its own.
MIXED = (indexed(stride=RUNTIME), indexed_pointer(slots=4, slot=RUNTIME,
                                                  offset=4),
         strided(stride=300, offset=8))


# --------------------------------------------------------------------------- #
# The parameters
# --------------------------------------------------------------------------- #

def test_the_defaults_are_the_bare_kind():
    """A description stating the defaults is the one that states none."""
    assert strided() is Addressing.STRIDED
    assert indexed() is Addressing.INDEXED
    assert indexed_pointer() is Addressing.INDEXED_POINTER
    assert pointer_based() is Addressing.PTR_BASED
    assert strided(stride=None, offset=0) is Addressing.STRIDED
    t = Tensor([4], ElementAddressing(Addressing.STRIDED), alias="t",
               datatype=DTYPE)
    assert t.addressing is Addressing.STRIDED


def test_a_parameterized_kind_never_equals_a_bare_one():
    """A site comparing with `Addressing.STRIDED` assumes the default stride;
    one with a stride of its own must not match it, and falls through to
    what the site does with a kind it does not know."""
    own = strided(stride=96)
    assert own != Addressing.STRIDED and Addressing.STRIDED != own
    assert own not in (Addressing.STRIDED, Addressing.PTR_BASED)
    assert own.kind is Addressing.STRIDED
    assert own == strided(stride=96) and own != strided(stride=97)
    assert hash(own) == hash(strided(stride=96))


def test_the_parameters_print():
    assert str(Addressing.INDEXED) == "indexed"
    assert str(Addressing.INDEXED_POINTER) == "indexed_pointer"
    assert str(indexed(stride=RUNTIME)) == "indexed(stride=runtime)"
    assert (str(indexed_pointer(slots=4, slot=RUNTIME, offset=4))
            == "indexed_pointer(offset=4, slots=4, slot=runtime)")
    assert str(strided(stride=300, offset=8)) == "strided(stride=300, offset=8)"


@pytest.mark.parametrize("addressing", [
    Addressing.NONE, Addressing.STRIDED, Addressing.INDEXED,
    Addressing.INDEXED_POINTER, indexed(stride=RUNTIME),
    indexed_pointer(slots=4, slot=RUNTIME, offset=4),
    strided(stride=300, offset=8), pointer_based(offset=2)])
def test_what_prints_reads_back(addressing):
    """The metadata line and a descriptor dump carry `str(addressing)`; the
    host tools read it back (`parse_addressing`)."""
    from tensorforge.common.basic_types import parse_addressing
    assert parse_addressing(str(addressing)) == addressing


def test_runtime_survives_a_deep_copy():
    """The tuner copies a description and asks `is RUNTIME` of the copy."""
    copied = copy.deepcopy(indexed(stride=RUNTIME))
    assert copied.stride is RUNTIME
    assert copy.deepcopy(RUNTIME) is RUNTIME


def test_pointer_types():
    assert Addressing.INDEXED.to_pointer() == "*"
    assert Addressing.INDEXED_POINTER.to_pointer() == "**"
    assert indexed(stride=7).to_pointer() == "*"
    assert indexed_pointer(slots=2).to_pointer() == "**"
    assert Addressing.addr2ptr_type(indexed_pointer(slots=2)) == "**"


@pytest.mark.parametrize("make,why", [
    (lambda: pointer_based(offset=-1), "at least 0"),
    (lambda: strided(stride=0), "at least 1"),
    (lambda: strided(stride=2.5), "integer"),
    (lambda: strided(offset=RUNTIME), "not bound at run time"),
    (lambda: ElementAddressing(Addressing.PTR_BASED, stride=4), "stride means"),
    (lambda: ElementAddressing(Addressing.INDEXED_POINTER, stride=4),
     "stride means"),
    (lambda: ElementAddressing(Addressing.INDEXED, slots=2), "slots mean"),
    (lambda: indexed_pointer(slots=2, slot=2), "slot 2 of 2"),
    (lambda: indexed_pointer(slots=0), "at least 1"),
    (lambda: ElementAddressing(Addressing.NONE), "no address parameters"),
    (lambda: ElementAddressing(Addressing.SCALAR), "no address parameters"),
])
def test_a_parameter_a_kind_does_not_take_is_refused(make, why):
    with pytest.raises(ValueError, match=why):
        make()


def test_element_addressing_fills_in_the_defaults():
    assert element_addressing(Addressing.NONE) is None
    assert element_addressing(Addressing.SCALAR) is None
    params = element_addressing(Addressing.INDEXED_POINTER)
    assert (params.stride, params.offset, params.slots, params.slot) == (
        None, 0, 1, 0)
    assert element_addressing(indexed(stride=3)).stride == 3


def test_a_tensor_with_its_own_stride_is_not_similar_to_one_without():
    box = BoundingBox([0], [4])
    one = Tensor([4], Addressing.STRIDED, box, datatype=DTYPE)
    two = Tensor([4], strided(stride=8), box, datatype=DTYPE)
    assert not one.is_similar(two)
    assert two.is_similar(Tensor([4], strided(stride=8), box, datatype=DTYPE))


def test_an_indexed_operand_is_not_passed_by_value():
    from tensorforge.common.basic_types import Residence
    with pytest.raises(GenerationError, match="passed as an argument"):
        Tensor([4], Addressing.INDEXED, alias="x", datatype=DTYPE,
               residence=Residence.ARGUMENT)


# --------------------------------------------------------------------------- #
# What the kernel and the launcher say
# --------------------------------------------------------------------------- #

def test_the_defaults_render_byte_for_byte_as_before():
    """A tensor given the default parameters is the bare kind, down to the
    generated text and the kernel's name."""
    plain = _generate(_gemm([Addressing.STRIDED, Addressing.PTR_BASED,
                             Addressing.STRIDED]))
    stated = _generate(_gemm([ElementAddressing(Addressing.STRIDED),
                              pointer_based(offset=0),
                              strided(stride=None, offset=0)]))
    assert _surface(plain) == _surface(stated)
    assert "elementIndices" not in _surface(plain)


@pytest.mark.parametrize("backend,arch", TARGETS,
                         ids=[f"{b}-{a}" for b, a in TARGETS])
def test_every_target_spells_the_addresses(backend, arch):
    """The entry of the list is read once, into a value the addresses take
    (`carry_element_index`; `tests/test_carry_element_index.py` has the
    spelling without)."""
    gen = _generate(_gemm(MIXED), backend, arch)
    kernel = gen.get_kernel()
    # The entry of the list, widened before a stride multiplies it.
    assert re.search(r"const size_t \w*elementIndex = static_cast<size_t>"
                     r"\(elementIndices\[[^\]]+\]\);", kernel)
    # A: the entry times the stride, then the offset.
    assert re.search(r"&m1\[\w*elementIndex \* m1_stride \+ 0 \+ "
                     r"m1_extraOffset\]", kernel)
    # B: the cell's pointer, its slot, then the offset into what it reaches.
    assert re.search(r"&m2\[\w*elementIndex \* 4 \+ m2_slot\]\[0 \+ 4 \+ "
                     r"m2_extraOffset\]", kernel)
    # C: a stride and an offset of its own, numbers in the code.
    assert re.search(r"&m0\[[^\]]* \* 300 \+ 0 \+ 8 \+ m0_extraOffset\]",
                     kernel)


@pytest.mark.parametrize("backend,arch", TARGETS,
                         ids=[f"{b}-{a}" for b, a in TARGETS])
def test_the_signature_takes_the_list_and_the_runtime_parameters(backend,
                                                                  arch):
    gen = _generate(_gemm(MIXED), backend, arch)
    header = gen.get_header()
    proto = header[header.index("void launcher_"):]
    proto = proto[:proto.index(")")]
    names = [p.split()[-1] for p in proto[proto.index("(") + 1:].split(",")]
    assert names[:8] == ["m0", "m0_extraOffset", "m1", "m1_extraOffset",
                         "m1_stride", "m2", "m2_extraOffset", "m2_slot"]
    # The list beside the count it is as long as, ahead of the flags.
    at = names.index("numElements0")
    assert names[at + 1] == "elementIndices"
    assert "const unsigned * elementIndices" in proto


def test_the_call_site_binds_yatetos_members():
    gen = _generate(_gemm(MIXED))
    call = gen.generate_call_site(
        {"A": "A", "B": "B", "C": "C"},
        {name: f"extraOffset_{name}" for name in "ABC"},
        {"A": {"stride": "stride_A"}, "B": {"slot": "slot_B"}})
    assert call.endswith("(C, extraOffset_C, A, extraOffset_A, stride_A, B, "
                         "extraOffset_B, slot_B, numElements, elementIndices, "
                         "flags, streamPtr);")


def test_a_kernel_without_the_list_takes_none():
    gen = _generate(_gemm([strided(stride=100), pointer_based(offset=4),
                           Addressing.STRIDED]))
    assert "elementIndices" not in _surface(gen)
    assert "&m2[" in gen.get_kernel() and "[0 + 4 + m2_extraOffset]" in (
        gen.get_kernel())


def test_the_pointer_array_of_an_indexed_pointer_is_promised():
    """Not written, not aliased by what is written: `restrict_batch_arrays`
    holds of `vars` as it does of a pointer-based operand's array."""
    gen = _generate(_gemm(MIXED))
    kernel = gen.get_kernel()
    head = kernel[kernel.index("__global__"):kernel.index("{")]
    assert "const float *const * __restrict__ m2" in head
    assert "const unsigned * __restrict__ elementIndices" in head


def test_an_indexed_address_is_followed_under_the_flag():
    """An entry of the list for a masked element is held to nothing, as a
    pointer-based element's pointer is: the peeled and wrapped copies of the
    binding read it only behind the element's flag."""
    gen = _generate(_gemm(MIXED), enable_wrap_loads=True)
    kernel = gen.get_kernel()
    for line in kernel.splitlines():
        if "peel_glb_m1[" in line or "wrap_glb_m1[" in line:
            assert "allowed" in line, line


def test_the_alignment_holds_only_as_far_as_the_offset_keeps_it():
    """A promise about every element's base, which an offset of three floats
    keeps to four bytes and a stride of 96 to 128."""
    from tensorforge.backend.symbol import Symbol, SymbolType
    box = BoundingBox([0, 0], [M, M])
    for addressing, want in ((Addressing.INDEXED, 64),
                             (indexed(offset=3), 4),
                             (indexed(stride=96), 64),
                             (indexed(stride=40, offset=16), 32),
                             (indexed(stride=RUNTIME), 64)):
        t = Tensor([M, M], addressing, box, alias="x", datatype=DTYPE,
                   alignment=64)
        symbol = Symbol("glb_x", SymbolType.Global, t)
        assert symbol.linear_align_bytes() == want, addressing


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("backend", ["omptarget", "targetdart"])
@pytest.mark.parametrize("addressing", [Addressing.INDEXED,
                                        indexed_pointer(slots=2),
                                        strided(stride=40)])
def test_a_lowering_without_the_list_refuses(backend, addressing):
    """The OpenMP lowerings declare their operands from lists of their own
    (`TargetLexic.kernel_definition`), which know neither."""
    descrs = _gemm([Addressing.STRIDED, addressing, Addressing.STRIDED])
    with pytest.raises(GenerationError, match="no index list"):
        _generate(descrs, backend, "sm_86")


def test_two_element_counts_and_one_list_are_refused():
    """A grid barrier gives the second section a count of its own; the list
    is one per launch."""
    first = _gemm([Addressing.INDEXED, Addressing.STRIDED, Addressing.STRIDED])
    second = _gemm([Addressing.STRIDED, Addressing.STRIDED, Addressing.STRIDED],
                   names="DEF")
    with pytest.raises(GenerationError, match="index list"):
        _generate(first + [GridBarrierDescr()] + second)


# --------------------------------------------------------------------------- #
# The interface
# --------------------------------------------------------------------------- #

def _tensor(name, formula, binding="default", shape=(M, M)):
    out = dict(name=name, addressing=formula, residence="memory",
               datatype="f32",
               storage=dict(shape=list(shape), type="bbox",
                            start=[0] * len(shape), sizes=list(shape)),
               values=None, alignment=0,
               flags=dict(temporary=False, constant=False))
    if binding == "default":
        binding = dict(stride=None, offset=0, slots=1, slot=0)
    if binding is not None:
        out["binding"] = binding
    return out


def _ref(name, indices):
    return dict(name=name, indices=list(indices), bbox=[[0, 0], [M, M]],
                offset=[0, 0], sliced=False)


def _description(version, tensors):
    scalar = dict(name="_scalar0", addressing="", residence="argument",
                  datatype="f32", storage=dict(shape=[], type="full"),
                  alignment=0, values=dict(kind="entries", data=[[[], 1.0]]),
                  flags=dict(temporary=False, constant=True))
    if version >= 8:
        scalar["binding"] = None
    return dict(version=version, tensors=list(tensors) + [scalar],
                operations=[dict(
                    type="multilinear", result=_ref("C", "ij"),
                    args=[_ref("A", "ik"), _ref("B", "kj")], condition=[],
                    permute=[[0, 1], [0, 1]], target=[[0, -1], [-1, 1]],
                    linear=dict(alpha=dict(name="_scalar0", indices=[],
                                           bbox=None, offset=None,
                                           sliced=False),
                                add=False))])


def _read(description):
    from tensorforge.frontend.yateto import DescriptionReader
    return DescriptionReader(None, {}).read(description)[1]


def test_the_interface_is_version_8_and_reads_7():
    from tensorforge.frontend.yateto import YatetoFrontend
    assert YatetoFrontend.INTERFACE_VERSION == 8
    assert YatetoFrontend.READABLE_VERSIONS == (7, 8)


def test_a_version_7_description_reads_with_the_default_bindings():
    cache = _read(_description(7, [_tensor(n, "n&+o&", None) for n in "ABC"]))
    assert all(cache[n].addressing is Addressing.PTR_BASED for n in "ABC")


def test_a_version_8_description_states_its_bindings():
    cache = _read(_description(8, [
        _tensor("A", "i*N+o&", dict(stride="runtime", offset=0, slots=1,
                                    slot=0)),
        _tensor("B", "i*K+k&+o&", dict(stride=None, offset=4, slots=4,
                                       slot="runtime")),
        _tensor("C", "n*N+o&", dict(stride=300, offset=8, slots=1, slot=0)),
    ]))
    assert cache["A"].addressing == indexed(stride=RUNTIME)
    assert cache["B"].addressing == indexed_pointer(slots=4, slot=RUNTIME,
                                                    offset=4)
    assert cache["C"].addressing == strided(stride=300, offset=8)
    assert cache["_scalar0"].addressing is Addressing.SCALAR


def test_the_bare_formulas_of_version_8():
    cache = _read(_description(8, [_tensor("A", "i*N+o&"),
                                   _tensor("B", "i*K+k&+o&"),
                                   _tensor("C", "n&+o&")]))
    assert cache["A"].addressing is Addressing.INDEXED
    assert cache["B"].addressing is Addressing.INDEXED_POINTER
    assert cache["C"].addressing is Addressing.PTR_BASED


@pytest.mark.parametrize("tensor,why", [
    (_tensor("A", "n*N+o&", None), "binding for every tensor"),
    (_tensor("A", "n&+o&", dict(stride=4, offset=0, slots=1, slot=0)),
     "stride means nothing"),
    (_tensor("A", "n*N+o&", dict(stride=4, offset=0)), "states exactly"),
    (_tensor("A", "i*N+o&", dict(stride="later", offset=0, slots=1, slot=0)),
     "integer or RUNTIME"),
    (_tensor("A", "&", dict(stride=None, offset=0, slots=1, slot=0)),
     "takes no parameters"),
])
def test_a_binding_the_reader_cannot_take_is_refused(tensor, why):
    with pytest.raises(NotImplementedError, match=why):
        _read(_description(8, [tensor, _tensor("B", "n&+o&"),
                               _tensor("C", "n&+o&")]))


def test_the_emitted_call_site_names_the_struct_members():
    """What yateto's kernel struct has: `stride_<name>`, `slot_<name>` beside
    `extraOffset_<name>`, and one `elementIndices`."""
    from tensorforge.frontend.yateto import DescriptionReader, KernelEmitter
    description = _description(8, [
        _tensor("A", "i*N+o&", dict(stride="runtime", offset=0, slots=1,
                                    slot=0)),
        _tensor("B", "i*K+k&+o&", dict(stride=None, offset=0, slots=4,
                                       slot="runtime")),
        _tensor("C", "n&+o&")])
    descrs, cache = DescriptionReader(None, {}).read(description)
    gen = Generator(descrs, Context(arch="sm_86", backend="cuda",
                                    fp_type=DTYPE))
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    call = KernelEmitter(None, {}, descrs, cache)._gen_call_site(gen)
    assert "stride_A" in call and "slot_B" in call
    assert "extraOffset_A, stride_A" in call
    assert call.endswith("numElements, elementIndices, flags, streamPtr);")


# --------------------------------------------------------------------------- #
# The numbers, on the host oracle
# --------------------------------------------------------------------------- #

def _oracle(addressings, indices, scalars, elements, backend="cuda",
            arch="sm_86", **options):
    """`C = A B` for `elements` elements, with the list `indices`; the
    worst relative error against NumPy over the inputs the oracle seeds,
    read where each operand's address says its element is."""
    gen = _generate(_gemm(addressings), backend, arch, **options)
    lanes, mults = kernel_eval.launch_geometry(gen.get_launcher())
    mem = kernel_eval.evaluate_wave(
        gen.get_kernel(), lanes, seed=3, globals_only=True, mults=mults,
        block=True, elements=elements,
        preset={"elementIndices": [int(i) for i in indices]},
        scalars=scalars)
    tensors = {s.obj.alias: s.obj
               for s in gen.built._scopes.get_global_scope().values()}
    seed = kernel_eval.Slot(3)

    def where(t, e):
        params = t.element_addressing()
        stride = params.stride
        if stride is RUNTIME:
            stride = scalars[f"{t.name}_stride"]
        if stride is None:
            stride = M * M
        slot = (scalars.get(f"{t.name}_slot") if params.slot is RUNTIME
                else params.slot)
        kind = params.kind
        if kind is Addressing.STRIDED:
            return t.name, e * stride + params.offset
        if kind is Addressing.INDEXED:
            return t.name, indices[e] * stride + params.offset
        if kind is Addressing.PTR_BASED:
            return f"{t.name}@{e}", params.offset
        return f"{t.name}@{indices[e] * params.slots + slot}", params.offset

    def matrix(base, at, source):
        return np.array([[source(base, at + i + j * M) for j in range(M)]
                         for i in range(M)])

    worst = 0.0
    for e in range(elements):
        want = (matrix(*where(tensors["A"], e), seed.read)
                @ matrix(*where(tensors["B"], e), seed.read))
        got = matrix(*where(tensors["C"], e),
                     lambda base, at: mem.get((base, at), np.nan))
        worst = max(worst, float(np.max(np.abs(got - want))
                                 / np.max(np.abs(want))))
    return worst


#: A permutation, and a list that repeats cells and skips some.
PERMUTED = [2, 0, 3, 1, 5, 4]
REPEATED = [1, 1, 0, 4, 1, 0]


@pytest.mark.parametrize("indices", [PERMUTED, REPEATED],
                         ids=["permuted", "repeated"])
def test_inputs_through_the_list_against_numpy(indices):
    """Inputs may name a cell twice and in any order: each element reads its
    cell, whatever else names it."""
    assert _oracle(MIXED, indices, {"m1_stride": 70, "m2_slot": 3},
                   len(indices)) < 1e-6


def test_a_destination_through_the_list_against_numpy():
    """Written through a permutation: every cell once."""
    assert _oracle([Addressing.INDEXED, pointer_based(offset=2),
                    indexed(offset=5)], PERMUTED, {}, len(PERMUTED)) < 1e-6


def test_a_destination_through_an_indexed_pointer_against_numpy():
    assert _oracle([indexed(stride=80), Addressing.STRIDED,
                    indexed_pointer(slots=3, slot=RUNTIME)], PERMUTED,
                   {"m0_slot": 1}, len(PERMUTED)) < 1e-6


@pytest.mark.parametrize("indices", [PERMUTED, PERMUTED[:5]],
                         ids=["even", "odd"])
def test_a_stack_of_two_reads_each_elements_own_entry(indices):
    """`stack_elements = 2`: each element of a stack its own entry, and a
    stack the batch ends inside its first element's."""
    assert _oracle([indexed(stride=RUNTIME), indexed_pointer(slots=4, slot=2),
                    indexed(stride=100, offset=3)], indices,
                   {"m1_stride": 70}, len(indices), stack_elements=2) < 1e-6


@pytest.mark.parametrize("options", [
    dict(enable_multibuffer=True), dict(enable_prefetch=True),
    dict(enable_wrap_loads=False), dict(enable_move_loads=True),
    dict(trip_reads=True, enable_multibuffer=True),
    dict(pointer_stage=False, enable_multibuffer=True), dict(pin_loads=8),
    dict(wave_uniform_group_rows=True)],
    ids=lambda o: ",".join(f"{k}={v}" for k, v in o.items()))
def test_the_passes_that_move_a_binding_keep_its_entry(options):
    """The passes that clone a binding for another element -- the peel, the
    wrap, the pointer stage, a hint -- read the entry of the list at that
    element's index, which they substitute like any operand."""
    assert _oracle(MIXED, REPEATED, {"m1_stride": 70, "m2_slot": 3},
                   len(REPEATED), **options) < 1e-6
    assert _oracle([Addressing.INDEXED, indexed_pointer(slots=2, slot=1),
                    indexed(stride=90, offset=2)], PERMUTED, {},
                   len(PERMUTED), **options) < 1e-6


@pytest.mark.parametrize("lanes", [16, 32])
def test_a_wave_reads_its_rows_entries_as_the_waves(lanes, monkeypatch):
    """gfx942 with the rows of a group each a multiplication's, and each
    row's address read as the wave's (`wave_uniform_group_rows`): an
    indexed address goes through the same first-lane read a pointer-based
    one does, and every row still reads its own entry of the list."""
    from tensorforge.backend.instructions.compute.primitives import amd
    from tensorforge.backend.instructions.compute.strategy import Strategy
    monkeypatch.setattr(amd, "ORDER", (Strategy.GENERIC,))
    options = dict(lanes_per_mult=lanes, wave_group_traversal=True,
                   wave_uniform_group_rows=True)
    kernel = _generate(_gemm(MIXED), "hip", "gfx942", **options).get_kernel()
    assert "tensorforge::pick<" in kernel
    assert _oracle(MIXED, REPEATED * 3, {"m1_stride": 70, "m2_slot": 3},
                   3 * len(REPEATED), "hip", "gfx942", **options) < 1e-6


def test_a_pointer_hint_is_for_a_pointer_array_only():
    """`enable_prefetch` asks for the successor's entry of a pointer array;
    an indexed binding reads the index list, and has none to ask for."""
    gen = _generate(_gemm([Addressing.PTR_BASED, Addressing.INDEXED,
                           Addressing.INDEXED_POINTER]), enable_prefetch=True)
    hints = [line.strip() for line in gen.get_kernel().splitlines()
             if "prefetchL2(" in line]
    assert len(hints) == 1 and hints[0].startswith("tensorforge::prefetchL2(&m1[")


def test_the_oracle_reads_the_identity_without_a_list():
    """Without a preset list element `n` is cell `n`: an indexed kernel then
    reads what the strided one does."""
    assert kernel_eval.Indices()[7] == 7


# --------------------------------------------------------------------------- #
# A SeisSol kernel, both ways, on the device
# --------------------------------------------------------------------------- #

#: An order in single precision and one in double.
SEISSOL = [("elastic-linearck", "elastic-linearck-o4-s", Datatype.F32),
           ("elastic-linearck", "elastic-linearck-o6-d", Datatype.F64)]


def _seissol_plan(kernel):
    """How SeisSol would reach each per-element operand of `kernel` through
    its storage (`binding-design.md`, step 2), as `(formula, binding)`: its
    own dofs indexed; the integrated dofs through the cell's pointer -- the
    neighbor's, one per face, for the neighboring flux; the star matrices
    and the flux solvers members of a struct, a stride bound at run time or
    a number; the derivatives through a pointer per order."""
    def bind(stride=None, offset=0, slots=1, slot=0):
        return dict(stride=stride, offset=offset, slots=slots, slot=slot)

    def plan(name):
        if name == "Q":
            return "i*N+o&", bind()
        if name == "I":
            if kernel.startswith("gpu_neighboringFlux"):
                return "i*K+k&+o&", bind(slots=4, slot="runtime")
            return "i*K+k&+o&", bind(offset=16)
        if name.startswith("star("):
            order = int(name[5:-1])
            return "i*N+o&", bind(stride="runtime", offset=3 * order + 1)
        if name.startswith("dQ("):
            # A pointer per order, up to order 8.
            order = int(name[3:-1])
            return "i*K+k&+o&", bind(slots=8, slot=order)
        return "n*N+o&", bind(stride=100, offset=5)
    return plan


def _converted(description, plan):
    """`description`, version 8, with each per-element operand reached as
    `plan` says; the rest as recorded, with its default binding."""
    out = copy.deepcopy(description)
    out["version"] = 8
    for tensor in out["tensors"]:
        formula = tensor["addressing"]
        own = formula in ("n*N+o&", "n&+o&")
        tensor["binding"] = (dict(stride=None, offset=0, slots=1, slot=0)
                             if own else None)
        if formula == "n&+o&" and not tensor["flags"]["temporary"]:
            tensor["addressing"], tensor["binding"] = plan(tensor["name"])
    return out


@pytest.mark.parametrize("system,config,dtype", SEISSOL,
                         ids=[c for _, c, _ in SEISSOL])
@pytest.mark.parametrize("kernel", ["gpu_volume", "gpu_localFlux[0]",
                                    "gpu_neighboringFlux[0]",
                                    "gpu_derivative"])
def test_a_seissol_kernel_reads_the_same_both_ways(kernel, system, config,
                                                   dtype, target, cache_root,
                                                   tensorforge_include):
    """The kernel as recorded -- pointer-based, the path SeisSol runs today --
    and the same kernel reaching its operands through the index list, over a
    permutation of the cells, built for the device and run on the same
    inputs: the results agree."""
    import seissol_suite as suite
    from harness import driver_emit
    from harness.runner import execute
    from tensorforge.frontend.yateto import DescriptionReader

    description = suite.description(system, config, kernel)
    real = np.float32 if dtype is Datatype.F32 else np.float64
    batch = 41
    indices = np.random.default_rng(7).permutation(batch)
    builds = []
    for described in (description,
                      _converted(description, _seissol_plan(kernel))):
        descrs = DescriptionReader(None, {}).read(described)[0]
        gen = Generator(descrs, Context(arch=target.arch,
                                        backend=target.backend,
                                        fp_type=dtype))
        with contextlib.redirect_stdout(io.StringIO()):
            gen.generate()
        builds.append(gen)
    rng = np.random.default_rng(11)
    inputs = {}
    for op in driver_emit.collect_operands(builds[0]):
        if op.is_scalar or not op.is_source:
            continue
        rows = 1 if op.addressing == "none" else batch
        if op.values is not None:
            # An operator the description gives the numbers of: those, as
            # SeisSol's buffers hold them, and as the kernel may take them.
            inputs[op.alias or op.kernel_name] = np.asarray(
                op.values, dtype=real).reshape(op.shape, order="F")[None]
            continue
        inputs[op.alias or op.kernel_name] = rng.standard_normal(
            (rows, *op.shape)).astype(real)
    want = execute(builds[0], target, cache_root, tensorforge_include, batch,
                   inputs)
    got = execute(builds[1], target, cache_root, tensorforge_include, batch,
                  inputs, indices=indices)
    assert set(want) == set(got)
    for name in want:
        scale = max(1.0, float(np.max(np.abs(want[name]))))
        tol = 1e-5 if dtype is Datatype.F32 else 1e-12
        np.testing.assert_allclose(got[name], want[name], rtol=tol,
                                   atol=tol * scale, err_msg=name)
