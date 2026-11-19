# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""An element's entry of the index list, read once and carried across the
batch loop (`Options.carry_element_index`).

An operand addressed through the launch's index list (`Addressing.INDEXED`,
`INDEXED_POINTER`) reads `elementIndices[n]` before its own address.  Read
in each address, at the top of the element, that is a load every address of
the element waits for first -- index, pointer and data, where a pointer
table has two.  The loop reads the entry once, as a value every address
takes (`BatchLoop._bind_entries`), and where the wrap reads the next
element's for the transfers it issues ahead, the loop carries that one into
the next iteration (`pir.wrap`, `_Rewrite._carry`).

Checked here: the text -- one read an element, the next element's carried,
the first element's in the peel, nothing carried that nothing reads, and
the old text where the switch is off or nothing reads the list; and the
numbers on the host oracle against NumPy, over index lists that repeat and
skip cells and over permutations, with a mask holding elements back, every
row of the block taking several trips, under the passes that clone an
address for another element.
"""

from __future__ import annotations

import contextlib
import io
import re

import numpy as np
import pytest

from tensorforge.common.basic_types import (RUNTIME, Addressing, Datatype,
                                            indexed, indexed_pointer,
                                            pointer_based, strided)
from tensorforge.common.context import Context
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.common.options import Options, registry
from tensorforge.generators.descriptions import GemmDescr
from tensorforge.generators.generator import Generator
from tensorforge.reference import kernel_eval

DTYPE = Datatype.F32
M = 8

#: CUDA, HIP, SYCL's SPMD lowering on both of its compilers, and the
#: explicitly vectorized one.
TARGETS = [("cuda", "sm_86"), ("cuda", "sm_120"), ("hip", "gfx942"),
           ("hip", "gfx1150"), ("oneapi", "pvc"), ("acpp", "pvc"),
           ("esimd", "pvc")]
IDS = [f"{b}-{a}" for b, a in TARGETS]

#: `C = A B`, the inputs through the list -- a stride bound at run time, a
#: pointer per cell with its slot bound at run time -- and the output
#: strided: what the element computes reads the list only for the
#: transfers the wrap moves.
INPUTS = (indexed(stride=RUNTIME), indexed_pointer(slots=4, slot=RUNTIME,
                                                   offset=4),
          strided(stride=300, offset=8))
#: Everything through the list, the output too: the element's own address
#: stays in the body whatever moves.
OWN = (Addressing.INDEXED, indexed_pointer(slots=2, slot=1),
       indexed(stride=90, offset=2))
#: The output through a pointer per cell.
OWN_POINTER = (indexed(stride=80), Addressing.STRIDED,
               indexed_pointer(slots=3, slot=RUNTIME))
#: `C = A B + A D` with `A` the batch's and both `B` and `D` through the
#: list: two copies into shared memory, which `enable_multibuffer` issues at
#: the head following pointers the loop carries (`pointer_stage`).
STAGED = (Addressing.NONE, indexed(stride=RUNTIME),
          strided(stride=300, offset=8), indexed_pointer(slots=2, slot=1))
SCALARS = {"m1_stride": 70, "m2_slot": 3, "m0_slot": 1, "m2_stride": 70}


def _gemm(addressings):
    """`C = A B`, or `C = A B + A D` for four addressings."""
    names = "ABC" if len(addressings) == 3 else "ABCD"
    tensors = {name: SubTensor(Tensor([M, M], addressing,
                                      BoundingBox([0, 0], [M, M]),
                                      alias=name, datatype=DTYPE))
               for name, addressing in zip(names, addressings)}
    a, b, c = tensors["A"], tensors["B"], tensors["C"]
    out = [GemmDescr(False, False, a, b, c, alpha=1.0, beta=0.0)]
    if "D" in tensors:
        out.append(GemmDescr(False, False, a, tensors["D"], c, alpha=1.0,
                             beta=1.0))
    return out


def _generate(addressings, backend="cuda", arch="sm_86", **options):
    ctx = Context(arch=arch, backend=backend, fp_type=DTYPE,
                  options=Options(**options))
    gen = Generator(_gemm(addressings), ctx)
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    return gen


def _loop(kernel: str):
    """The kernel's lines ahead of the batch loop, and those of the loop."""
    lines = kernel.splitlines()
    at = next(i for i, line in enumerate(lines)
              if re.search(r"\bfor \(size_t v\d+_batchId(?:Group)?0\b", line))
    return lines[:at], lines[at:]


def _reads(lines):
    """The lines that read the index list."""
    return [line.strip() for line in lines if "elementIndices[" in line]


def _names(reads):
    """What the reads of the index list declare."""
    return [re.match(r"const size_t (\w+) =", line).group(1) for line in reads]


def _same_but_for_the_name(one: str, other: str) -> bool:
    """Whether two kernels differ in nothing but the name the options hash
    into and the line that states them."""
    def body(kernel):
        kernel = re.sub(r"kernel_[0-9a-f]+", "kernel", kernel)
        return re.sub(r"// options:.*", "", kernel)
    return body(one) == body(other)


# --------------------------------------------------------------------------- #
# The switch
# --------------------------------------------------------------------------- #

def test_on_by_default():
    assert registry()["carry_element_index"].default is True


# --------------------------------------------------------------------------- #
# The text
# --------------------------------------------------------------------------- #

def test_the_entry_is_read_once_an_element_and_carried():
    """The element's addresses take the carried entry; the head reads the
    next element's once, for the copies there and for the next iteration;
    the peel reads the first element's, which the loop starts with."""
    kernel = _generate(OWN).get_kernel()
    ahead, loop = _loop(kernel)
    assert _reads(ahead) == [
        next(line for line in _reads(ahead)
             if line.startswith("const size_t peel_elementIndex = "))]
    assert _reads(loop) == [
        next(line for line in _reads(loop)
             if line.startswith("const size_t wrap_elementIndex = "))]
    # One read an element, and no address reads the list itself.
    assert not any("elementIndices" in line and "=" in line
                   and "glb_" in line for line in kernel.splitlines())
    start = re.search(r"size_t (v\d+_elementIndex) = peel_elementIndex;",
                      kernel)
    assert start, "the loop starts with the peel's entry"
    held = start.group(1)
    assert any(line.strip() == f"{held} = wrap_elementIndex;"
               for line in loop), "the loop carries the head's read"
    # The element's own addresses take the carried entry, every one of them.
    own = [line for line in loop if re.search(r"\bglb_m\d = &m\d\[", line)]
    assert own and all(held in line for line in own), own
    # The next element's copies take the one read at the head.
    nxt = [line for line in loop if re.search(r"\bwrap_glb_m\d = ", line)]
    assert nxt and all("wrap_elementIndex" in line for line in nxt), nxt


@pytest.mark.parametrize("backend,arch", TARGETS, ids=IDS)
def test_every_target_reads_an_elements_entry_once(backend, arch):
    """On every lowering, whether it wraps (all but DPC++'s SPMD one) and
    however its batch loop is spelled -- per row, or a group of rows a
    sub-group takes together: the list is read once per element and loop
    position, and every address of an operand through it takes the value."""
    kernel = _generate(OWN, backend, arch).get_kernel()
    ahead, loop = _loop(kernel)
    reads = _reads(ahead) + _reads(loop)
    assert reads and all(re.match(r"const size_t \w*elementIndex = static_cast"
                                  r"<size_t>\(elementIndices\[", line)
                         for line in reads), reads
    names = _names(reads)
    assert len(names) == len(set(names)), names
    addresses = [line for line in kernel.splitlines()
                 if re.search(r"glb_m\d(?:_s\d)? = .*&m\d\[", line)]
    assert addresses and all(re.search(r"&m\d\[\w*elementIndex \* ", line)
                             for line in addresses), addresses
    if "wrap_elementIndex" in kernel:
        # Carried: the element's addresses take what the loop holds.
        assert re.search(r"(v\d+_elementIndex) = wrap_elementIndex;", kernel)
    else:
        # Read once at the top of the element, inside its guard.
        assert [line for line in reads if "peel_" not in line] == [
            next(line for line in reads
                 if line.startswith("const size_t elementIndex = "))]


def test_without_the_wrap_the_entry_is_read_at_the_top_of_the_element():
    """Nothing ahead of the loop and nothing carried: one read inside the
    element's guard, which every address takes."""
    kernel = _generate(OWN, enable_wrap_loads=False).get_kernel()
    ahead, loop = _loop(kernel)
    assert not _reads(ahead)
    assert _reads(loop) == ["const size_t elementIndex = static_cast<size_t>"
                            "(elementIndices[" + re.search(
                                r"elementIndices\[(\w+)\]", kernel).group(1)
                            + "]);"]
    guard = next(i for i, line in enumerate(loop) if "if (allowed)" in line)
    read = next(i for i, line in enumerate(loop) if "elementIndices[" in line)
    first = next(i for i, line in enumerate(loop) if "&m" in line)
    assert guard < read < first
    addresses = [line for line in loop if re.search(r"glb_m\d = &m\d\[", line)]
    assert len(addresses) == 3 and all("[elementIndex * " in line
                                       for line in addresses), addresses


def test_nothing_is_carried_where_nothing_left_reads_the_entry():
    """Every address through the list is for a transfer the wrap moved: the
    element computes with what was fetched for it, and its own copy of the
    address goes -- so does the read, and nothing is carried."""
    kernel = _generate(INPUTS).get_kernel()
    assert "peel_elementIndex" in kernel and "wrap_elementIndex" in kernel
    assert not re.search(r"\bv\d+_elementIndex\b", kernel)
    assert not re.search(r"\bconst size_t elementIndex\b", kernel)
    # One read at the head, which both copies take.
    _, loop = _loop(kernel)
    assert len(_reads(loop)) == 1


@pytest.mark.parametrize("backend,arch", TARGETS, ids=IDS)
def test_off_every_address_reads_the_list(backend, arch):
    """Off, the text the addresses always had: each reads its entry of the
    list, widened before the stride multiplies it."""
    kernel = _generate(INPUTS, backend, arch,
                       carry_element_index=False).get_kernel()
    assert "elementIndex " not in kernel and "elementIndex;" not in kernel
    assert re.search(r"&m1\[static_cast<size_t>\(elementIndices\[[^\]]+\]\)"
                     r" \* m1_stride \+ 0 \+ m1_extraOffset\]", kernel)
    assert re.search(r"&m2\[static_cast<size_t>\(elementIndices\[[^\]]+\]\)"
                     r" \* 4 \+ m2_slot\]\[0 \+ 4 \+ m2_extraOffset\]", kernel)
    assert not re.search(r"\w*elementIndex\b", kernel)


@pytest.mark.parametrize("addressings", [
    (Addressing.STRIDED, Addressing.PTR_BASED, Addressing.STRIDED),
    (pointer_based(offset=2), Addressing.NONE, strided(stride=80)),
    (Addressing.PTR_BASED, Addressing.PTR_BASED, Addressing.PTR_BASED)],
    ids=["strided-ptr", "parameters", "ptr"])
@pytest.mark.parametrize("backend,arch", TARGETS, ids=IDS)
def test_a_kernel_without_the_list_is_the_same_either_way(addressings,
                                                          backend, arch):
    """Nothing reads the list: the switch changes nothing but the name."""
    on = _generate(addressings, backend, arch, enable_multibuffer=True)
    off = _generate(addressings, backend, arch, enable_multibuffer=True,
                    carry_element_index=False)
    for part in ("get_kernel", "get_launcher", "get_header"):
        assert _same_but_for_the_name(getattr(on, part)(),
                                      getattr(off, part)()), part


def test_a_stack_reads_each_of_its_elements_entries_once():
    """`stack_elements = 2`: each element of the stack its own entry, read
    once and carried, picked as its addresses pick their element."""
    kernel = _generate(OWN, stack_elements=2).get_kernel()
    ahead, loop = _loop(kernel)
    assert sorted(_names(_reads(ahead))) == ["peel_elementIndex_s0",
                                             "peel_elementIndex_s1"]
    assert sorted(_names(_reads(loop))) == ["wrap_elementIndex_s0",
                                            "wrap_elementIndex_s1"]
    for e in range(2):
        held = re.search(rf"size_t (v\d+_elementIndex_s{e}) = "
                         rf"peel_elementIndex_s{e};", kernel)
        assert held, e
        assert f"{held.group(1)} = wrap_elementIndex_s{e};" in kernel
        own = [line for line in loop
               if re.search(rf"\bglb_m\d_s{e} = .*&m\d\[", line)]
        assert own and all(held.group(1) in line for line in own), own


def test_the_pointer_stage_reads_the_entry_after_next_once():
    """Under `enable_multibuffer` a copy at the head follows a pointer the
    loop carries, read with the element after next: the entry of that
    element is read once for every pointer staged, and the peel reads the
    second element's once.  Nothing reads the next element's entry, whose
    pointers come in carried, and it is not read."""
    kernel = _generate(STAGED, enable_multibuffer=True,
                       move_distance=2).get_kernel()
    ahead, loop = _loop(kernel)
    assert sorted(_names(_reads(ahead))) == ["peel_elementIndex",
                                             "second_elementIndex"]
    assert _names(_reads(loop)) == ["ahead_elementIndex"]
    staged = [line for line in loop if re.search(r"\bahead_glb_m\d = ", line)]
    assert len(staged) == 2 and all("ahead_elementIndex" in line
                                    for line in staged), staged
    assert "wrap_elementIndex" not in kernel
    # Without the stage the copies follow the next element's pointers, read
    # from its entry at the head, once.
    plain = _generate(STAGED, enable_multibuffer=True, move_distance=2,
                      pointer_stage=False).get_kernel()
    assert "ahead_elementIndex" not in plain
    assert len(_reads(_loop(plain)[1])) == 1


def test_a_group_of_rows_carries_each_rows_entry():
    """SYCL's SPMD lowering drives the rows of a sub-group together: each
    row's entry is read once, from the row's element the group's index
    gives, and the copy for the next group is what the next trip reads."""
    kernel = _generate(OWN, "acpp", "pvc").get_kernel()
    _, loop = _loop(kernel)
    assert "batchIdGroup0" in loop[0]
    assert len(_reads(loop)) == 1 and "wrap_elementIndex" in _reads(loop)[0]
    assert re.search(r"(v\d+_elementIndex) = wrap_elementIndex;", kernel)


def test_rows_each_multiplication_reads_as_the_waves_keep_their_reads(
        monkeypatch):
    """Where each multiplication of a group reads its row as the wave's,
    `pir.rows` gives each a copy of what it reads through its row -- the
    entry with it -- and the addresses read the list as they did."""
    from tensorforge.backend.instructions.compute.primitives import amd
    from tensorforge.backend.instructions.compute.strategy import Strategy
    monkeypatch.setattr(amd, "ORDER", (Strategy.GENERIC,))
    options = dict(lanes_per_mult=16, wave_group_traversal=True,
                   wave_uniform_group_rows=True)
    on = _generate(OWN, "hip", "gfx942", **options).get_kernel()
    off = _generate(OWN, "hip", "gfx942", carry_element_index=False,
                    **options).get_kernel()
    assert "tensorforge::pick<" in on
    assert _same_but_for_the_name(on, off)


def test_the_queue_reads_the_entry_once_and_carries_nothing():
    """Blackwell's work queue: the copies for the element the queue answers
    next read its entry at the head, and the loop, a `while` that carries
    its element alone, carries nothing -- the element's own entry is read
    once, in its guard."""
    ctx = Context(arch="sm_120", backend="cuda", fp_type=DTYPE,
                  options=Options(launch_control=True,
                                  launch_control_depth=2))
    gen = Generator(_gemm(OWN), ctx)
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    kernel = gen.get_kernel()
    assert "launchCursor" in kernel
    assert not re.search(r"\bv\d+_elementIndex\b", kernel)
    reads = _reads(kernel.splitlines())
    names = _names(reads)
    assert "elementIndex" in names and len(names) == len(set(names))


# --------------------------------------------------------------------------- #
# The numbers, on the host oracle
# --------------------------------------------------------------------------- #

def _where(t, e, indices, scalars):
    """Where element `e` of operand `t` is: `(base, offset)`."""
    params = t.element_addressing()
    if params is None:
        # The batch's: one copy for every element.
        return t.name, 0
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
        return t.name, int(indices[e]) * stride + params.offset
    if kind is Addressing.PTR_BASED:
        return f"{t.name}@{e}", params.offset
    return f"{t.name}@{int(indices[e]) * params.slots + slot}", params.offset


def _worst(addressings, indices, backend="cuda", arch="sm_86", **options):
    """`C = A B` -- `+ A D` -- over as many elements as `indices` has, which
    takes every row of the block several trips, with a mask holding three of
    them back; the worst error against NumPy over the others.  A held-back
    element's output is not written."""
    gen = _generate(addressings, backend, arch, **options)
    lanes, mults = kernel_eval.launch_geometry(gen.get_launcher())
    elements = len(indices)
    assert elements > 2 * mults, "every row takes several trips"
    held = {1, mults + 2, elements - 1}
    preset = {"elementIndices": [int(i) for i in indices],
              "flags0": [0 if e in held else 1 for e in range(elements)]}
    races = []
    mem = kernel_eval.evaluate_wave(
        gen.get_kernel(), lanes, seed=3, globals_only=True, mults=mults,
        block=True, elements=elements, preset=preset, scalars=SCALARS,
        races=races)
    assert not races, races[:3]
    tensors = {s.obj.alias: s.obj
               for s in gen.built._scopes.get_global_scope().values()}
    seed = kernel_eval.Slot(3)

    def matrix(base, at, source):
        return np.array([[source(base, at + i + j * M) for j in range(M)]
                         for i in range(M)])

    worst = 0.0
    for e in range(elements):
        where = _where(tensors["C"], e, indices, SCALARS)
        got = matrix(*where, lambda base, at: mem.get((base, at), np.nan))
        if e in held:
            assert np.all(np.isnan(got)), f"element {e} is held back"
            continue
        a = matrix(*_where(tensors["A"], e, indices, SCALARS), seed.read)
        want = sum(a @ matrix(*_where(tensors[name], e, indices, SCALARS),
                              seed.read)
                   for name in "BD" if name in tensors)
        worst = max(worst, float(np.max(np.abs(got - want))
                                 / np.max(np.abs(want))))
    return worst


def _lists(elements):
    """An index list that repeats cells and skips some, for inputs; one that
    names each cell once, out of order and with gaps, for any list an output
    goes through."""
    rng = np.random.default_rng(elements)
    repeats = rng.integers(0, elements + 3, size=elements)
    assert len(set(repeats.tolist())) < elements
    once = rng.permutation(elements + 5)[:elements]
    return repeats, once


#: The passes that compute an address again for another element, and what
#: the switch is set to; on every one of them the kernel reads what the list
#: names.
OPTIONS = [
    dict(),
    dict(enable_wrap_loads=False),
    dict(enable_multibuffer=True),
    dict(enable_multibuffer=True, pointer_stage=False),
    dict(stack_elements=2),
    dict(stack_elements=2, enable_multibuffer=True),
    dict(enable_prefetch=True, prefetch_data=True),
    # The rows of a wave driven as one group, as SYCL's SPMD lowering
    # drives a sub-group's: each row's element is bound from the group's
    # index, and the copy for the next group computes it again.
    dict(wave_group_traversal=True),
    dict(wave_group_traversal=True, stack_elements=2),
    dict(carry_element_index=False),
    dict(carry_element_index=False, enable_multibuffer=True),
]
OPTION_IDS = [",".join(f"{k}={v}" for k, v in o.items()) or "default"
              for o in OPTIONS]


@pytest.fixture(params=[("cuda", "sm_86"), ("cuda", "sm_120"),
                        ("hip", "gfx942"), ("hip", "gfx1150")],
                ids=["cuda-sm_86", "cuda-sm_120", "hip-gfx942",
                     "hip-gfx1150"])
def oracle_target(request, monkeypatch):
    """A target the host oracle runs: CUDA, and HIP with the products the
    oracle evaluates (the generic strategy, not the DPP and matrix ones)."""
    backend, arch = request.param
    if backend == "hip":
        from tensorforge.backend.instructions.compute.primitives import amd
        from tensorforge.backend.instructions.compute.strategy import Strategy
        monkeypatch.setattr(amd, "ORDER", (Strategy.GENERIC,))
    return backend, arch


@pytest.mark.parametrize("options", OPTIONS, ids=OPTION_IDS)
def test_inputs_through_a_list_that_repeats_and_skips(options, oracle_target):
    """The inputs through a list naming some cells twice and others never;
    each element reads its own cells, whichever trip it is on."""
    backend, arch = oracle_target
    lanes, mults = kernel_eval.launch_geometry(
        _generate(INPUTS, backend, arch, **options).get_launcher())
    repeats, _ = _lists(3 * mults + 1)
    assert _worst(INPUTS, repeats, backend, arch, **options) < 1e-6


@pytest.mark.parametrize("options", [
    dict(enable_multibuffer=True, move_distance=2),
    dict(enable_multibuffer=True, move_distance=2, pointer_stage=False),
    dict(move_distance=2),
    dict(enable_multibuffer=True, move_distance=2,
         carry_element_index=False)],
    ids=lambda o: ",".join(f"{k}={v}" for k, v in o.items()))
def test_two_staged_pointers_through_a_list_that_repeats(options,
                                                         oracle_target):
    """Both copies at the head follow pointers read with the element after
    next, from its entry read once; each element still reads its own."""
    backend, arch = oracle_target
    lanes, mults = kernel_eval.launch_geometry(
        _generate(STAGED, backend, arch, **options).get_launcher())
    repeats, _ = _lists(3 * mults + 1)
    assert _worst(STAGED, repeats, backend, arch, **options) < 1e-6


@pytest.mark.parametrize("options", OPTIONS, ids=OPTION_IDS)
@pytest.mark.parametrize("addressings", [OWN, OWN_POINTER],
                         ids=["own", "own-pointer"])
def test_everything_through_a_permutation(addressings, options,
                                          oracle_target):
    """The output through the list as well -- a cell each, out of order and
    with gaps -- so the element's own address stays in the body and takes
    the carried entry."""
    backend, arch = oracle_target
    lanes, mults = kernel_eval.launch_geometry(
        _generate(addressings, backend, arch, **options).get_launcher())
    _, once = _lists(3 * mults + 1)
    assert _worst(addressings, once, backend, arch, **options) < 1e-6


def test_the_queue_reads_what_the_list_names():
    """Blackwell's work queue, handing the block its CTAs out of order: the
    element after this one is not a stride on, and the entry read for it is
    that element's."""
    ctx = Context(arch="sm_120", backend="cuda", fp_type=DTYPE,
                  options=Options(launch_control=True,
                                  launch_control_depth=2))
    gen = Generator(_gemm(OWN), ctx)
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    lanes, mults = kernel_eval.launch_geometry(gen.get_launcher())
    elements = 3 * mults - 1
    _, once = _lists(elements)
    flags = [1] * elements
    flags[1] = 0
    mem = kernel_eval.evaluate_wave(
        gen.get_kernel(), lanes, seed=3, globals_only=True, mults=mults,
        block=True, elements=elements, launches=(2, 1),
        preset={"elementIndices": [int(i) for i in once], "flags0": flags})
    tensors = {s.obj.alias: s.obj
               for s in gen.built._scopes.get_global_scope().values()}
    seed = kernel_eval.Slot(3)

    def matrix(base, at, source):
        return np.array([[source(base, at + i + j * M) for j in range(M)]
                         for i in range(M)])
    for e in range(elements):
        got = matrix(*_where(tensors["C"], e, once, SCALARS),
                     lambda base, at: mem.get((base, at), np.nan))
        if e == 1:
            assert np.all(np.isnan(got))
            continue
        want = (matrix(*_where(tensors["A"], e, once, SCALARS), seed.read)
                @ matrix(*_where(tensors["B"], e, once, SCALARS), seed.read))
        assert np.max(np.abs(got - want)) / np.max(np.abs(want)) < 1e-6, e
