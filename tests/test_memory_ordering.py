# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Ordering invariants the stores of an operation have to hold in the
generated source.

``mixed_ml_glb_then_ew`` is the shape that needs one.  In the wrong order it
still generates:

    // glb_m3 = abs(glb_m0)      <- reads whatever the caller left in M
    // glb_m0 = store{r>g}(r2);  <- writes M only then

A snapshot of that would be green, so the order needs stating as an invariant
instead.  That is this file.  It holds because an operation stores what it
computes behind itself, before the next one runs.

The section-boundary test is about a different lifetime: a section's stores
have to land before the next section runs rather than at the end of the
kernel, since a store emitted after the barrier that was meant to publish it
is wrong in a way no snapshot would flag.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

from harness import UNSUPPORTED
from tensorforge.backend.instructions.memory.store import StoreRegToShr
from tensorforge.backend.pir.core import Effect, Op
from tensorforge.common.context import Context, Options
from tensorforge.generators.generator import Generator

CASES = Path(__file__).parent / "cases"

#: The four codegen targets ``test_snapshots`` covers.
ALL_TARGETS = [("cuda", "sm_86"), ("hip", "gfx90a"),
               ("acpp", "pvc"), ("esimd", "pvc")]


def _load(stem: str):
    path = CASES / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(f"tf_case__{stem.replace('/', '__')}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _generate(stem: str, backend: str, arch: str, **options):
    """The generator and the descriptor list, whose tensors it has named."""
    mod = _load(stem)
    descrs = mod.descr_list()
    ctx = Context(arch=arch, backend=backend, fp_type=mod.DTYPE,
                  options=Options(**options) if options else None)
    gen = Generator(descrs, ctx)
    gen.generate()
    return gen, descrs


def _accesses(src: str, array: str):
    """``(kind, line)`` for every access to ``array``, in emitted order.

    Four spellings reach a global array and they have to be told apart by
    shape, not by guessing: a subscripted store, a wide store through a cast,
    and the ESIMD pair, where the direction is in the method name and the
    address is pointer arithmetic rather than a subscript.

    Writes are tested first, since every write also contains the array name in
    a position a read test would match.
    """
    name = re.escape(array)
    subscript_store = re.compile(rf"(?:^|[^\w])({name})\s*\[[^\]]*\]\s*=(?!=)")
    wide_store = re.compile(rf"\*\s*\([^)]*\)\s*&\s*{name}\s*\[[^\]]*\]\s*=(?!=)")
    copy_to = re.compile(rf"\.copy_to\(\s*{name}\b")
    copy_from = re.compile(rf"\.copy_from\(\s*{name}\b")
    subscript = re.compile(rf"(?:^|[^\w]){name}\s*\[")

    out = []
    for i, raw in enumerate(src.splitlines()):
        line = raw.strip()
        if line.startswith("//"):
            continue
        if wide_store.search(line) or subscript_store.search(line) \
                or copy_to.search(line):
            out.append(("w", i, line))
        elif subscript.search(line) or copy_from.search(line):
            out.append(("r", i, line))
    return out


def _first(accesses, kind):
    return next((a for a in accesses if a[0] == kind), None)


def _flatten(stream):
    """The section's instructions in emitted order, loop bodies included.

    A section's stream is two or three entries deep: the shared-memory
    allocation and a `BatchLoop` holding everything else, since a wide body
    puts the whole section inside one loop.
    """
    out = []
    for instr in stream:
        out.append(instr)
        out.extend(_flatten(getattr(instr, "_region", ()) or ()))
    return out


@pytest.mark.parametrize("backend,arch", ALL_TARGETS,
                         ids=[b for b, _ in ALL_TARGETS])
def test_pointwise_read_follows_the_contraction_store(backend, arch):
    """``C = abs(M)`` may not read ``M`` before ``M = A @ B`` has written it."""
    # Without the forwarding, which hands each lane the value it stored and
    # leaves no read of `M` at all (`pir.forward`): what is asked here is the
    # order of the read and the write.
    gen, descrs = _generate("mixed/ml_glb_then_ew", backend, arch,
                            forward_memory=False)
    array = f"glb_{descrs[0].dest.tensor.name}"
    accesses = _accesses(gen.get_kernel(), array)

    read = _first(accesses, "r")
    write = _first(accesses, "w")
    assert read is not None and write is not None, (
        f"expected both a read and a write of {array}, got "
        f"{[(k, ln) for k, ln, _ in accesses]}")
    assert write[1] < read[1], (
        f"{array} is read at line {read[1]} before it is written at line "
        f"{write[1]}, so the pointwise operation sees the destination's "
        f"previous contents:\n"
        f"  {read[1]}: {read[2]}\n"
        f"  {write[1]}: {write[2]}")


@pytest.mark.parametrize("backend,arch", ALL_TARGETS,
                         ids=[b for b, _ in ALL_TARGETS])
def test_temporary_assembled_from_slices_is_read_after_both_writes(backend,
                                                                   arch):
    """Two contractions write half a temporary each, and
    ``SectionPlan.written_in_slices`` lays its buffer out for both: the
    pointwise read comes behind both stores, of the one buffer they share.
    Stated as a test and not only a snapshot, since a consumer that cannot
    address a shared temporary at all fails it.
    """
    gen, _ = _generate("mixed/ml_slices_then_ew", backend, arch)
    lines = gen.get_kernel().splitlines()

    # The emitted operation comments, which name symbols (`s1`), not the
    # metadata header at the top of the kernel, which names tensors (`TMP`).
    # Both spell the operation the same way, so the symbol shape is what tells
    # them apart -- and matching on it also makes the test state that the
    # pointwise read and the two stores refer to one and the same buffer.
    stores = [(i, m.group(1)) for i, line in enumerate(lines)
              if (m := re.search(r"//\s*(s\d+) = store\{r>s\}", line))]
    reads = [(i, m.group(1)) for i, line in enumerate(lines)
             if (m := re.search(r"//\s*glb_\w+ = abs\((s\d+)\)", line))]
    assert len(stores) == 2 and len(reads) == 1, (
        f"expected two shared stores and one pointwise read, found "
        f"{[s for _, s in stores]} and {[r for _, r in reads]}")

    (read_line, buffer), = reads
    assert {s for _, s in stores} == {buffer}, (
        f"the pointwise operation reads {buffer} but the stores write "
        f"{sorted({s for _, s in stores})}")
    assert max(i for i, _ in stores) < read_line, (
        f"the pointwise read at line {read_line} precedes the store at line "
        f"{max(i for i, _ in stores)}, so it sees only part of the temporary")


@pytest.mark.parametrize("case_stem", ["barrier/barrier_two_gemms",
                                  "barrier/fence_two_gemms"],
                         ids=["barrier", "fence"])
@pytest.mark.parametrize("backend,arch", ALL_TARGETS,
                         ids=[b for b, _ in ALL_TARGETS])
def test_a_sections_stores_land_before_the_next_section(case_stem, backend,
                                                        arch):
    """The first section's store lands before the second section runs.

    Nothing in the second section knows about the first section's registers,
    so the store has to have happened by the time the second section starts
    -- and "by the end of the kernel" is not by then.

    Stated as an access order rather than against the barrier instruction,
    which each backend spells differently.

    The two cases carry different amounts of evidence for that, and the
    difference is deliberate rather than incidental:

    * ``barrier`` keeps a cross-section dependency -- section 1 reads back the
      tensor section 0 wrote -- so the order can be asserted on that one array,
      which is the sharpest form the invariant has.
    * ``fence`` has none, and must not: a fence does not order its sections,
      so a case that read section 0's output in section 1 would be racing by
      construction (see ``cases/barrier/fence_two_gemms.py``).  What is left to
      assert there is the boundary itself -- section 0's destination is written
      before section 1's destination is first touched, rather than both being
      stored together at the end of the kernel.  That is the part of the
      invariant that does not need a dependency to be true.
    """
    try:
        gen, descrs = _generate(case_stem, backend, arch)
    except UNSUPPORTED as exc:
        # Two separate gaps on the Intel targets, neither about ordering: a
        # group barrier under a simd-uniform trip count is refused, and
        # `SyclLexic.sync_grid` is an unimplemented stub.  Either way there is
        # no source to state an order about, and that the case does not
        # generate there is the snapshots' business.
        pytest.skip(f"{case_stem} does not generate on {backend}: "
                    f"{type(exc).__name__}: {exc}")
    produced = f"glb_{descrs[0].dest.tensor.name}"
    src = gen.get_kernel()
    accesses = _accesses(src, produced)

    write = _first(accesses, "w")
    assert write is not None, (
        f"expected {produced} to be written by the first section, got "
        f"{[(k, ln) for k, ln, _ in accesses]}")

    read = _first(accesses, "r")
    if read is not None:
        # The cross-section dependency the barrier case carries: the readback
        # in section 1 must not overtake the store in section 0.
        assert write[1] < read[1], (
            f"{produced} is read at line {read[1]} by the section after the "
            f"barrier, before the section before it wrote at line "
            f"{write[1]}:\n"
            f"  {read[1]}: {read[2]}\n"
            f"  {write[1]}: {write[2]}")

    # Holds with or without that dependency: the first section's store
    # precedes anything the second section does with its own destination.
    consumed = f"glb_{descrs[-1].dest.tensor.name}"
    assert consumed != produced, (
        f"the two sections of this case share destination {consumed}; the "
        "boundary assertion below cannot tell them apart")
    later = _accesses(src, consumed)
    assert later, f"expected the second section to touch {consumed}"
    assert write[1] < later[0][1], (
        f"{produced} is written at line {write[1]}, after the second section "
        f"first touches {consumed} at line {later[0][1]}, so the first "
        f"section's store outlived the section boundary:\n"
        f"  {later[0][1]}: {later[0][2]}\n"
        f"  {write[1]}: {write[2]}")


@pytest.mark.parametrize("backend,arch", ALL_TARGETS,
                         ids=[b for b, _ in ALL_TARGETS])
def test_a_stored_temporary_is_published_before_it_is_read(backend, arch):
    """The store behind a product gets a barrier, without one being emitted.

    `tmp = A @ B` stores its result into its shared buffer, and the pointwise
    operation reads it there.  Each lane writes its own part, and nothing says
    the pointwise operation hands the same elements back to the same lanes, so
    the store has to be published before the read.

    No barrier is emitted next to the store, deliberately: the barrier
    placement reads the section's body in its final order and puts one where
    a write of shared memory meets a read of it (`pir.barriers`).  This is
    the assertion that the pair is recognized.

    Asked of the body rather than of the source, because a SIMD-scope barrier
    lowers to nothing on the targets where a wave runs in lockstep --
    `HipLexic.sync_simd` returns None, and so does `SyclLexic` in SIMD mode.
    Only CUDA emits text (`__syncwarp`, for the independent thread
    scheduling), so a textual assertion would test the architecture rather
    than the pass.
    """
    # Without the forwarding: here the pointwise operation reads each element
    # in the lane that stored it, which the forwarding finds, and then
    # nothing is read through shared memory (`pir.forward`).  And for the
    # same reason without `keep_own`: a buffer no lane hands on to another
    # stays in registers where the vendor table says so, and then there is
    # no window to publish (`pir.carry.keep_own`).
    gen, _ = _generate("mixed/ml_then_ew", backend, arch,
                       forward_memory=False, keep_own=False)
    stream = _flatten(gen._sections[0].stream)
    stores = [i for i in stream if isinstance(i, StoreRegToShr)]
    assert len(stores) == 1, f"expected exactly one shared store, got {stores}"
    dest = stores[0].get_dest()

    def stmts(body):
        for s in body:
            yield s
            for r in s.regions:
                yield from stmts(r.body)
    body = list(stmts(gen._sections[0].body))
    windows = {s.target[0].id for s in body
               if s.op == Op.ALLOC and s.attr('identity') is dest}
    assert windows, f"no window of {dest.name} in the body"

    def touches(s, kind):
        return s.op not in (Op.ALLOC, Op.MARK) and any(
            a.kind & kind and getattr(a.base, 'id', None) in windows
            for a in s.accesses)
    read = next(k for k, s in enumerate(body) if touches(s, Effect.READ))
    write = max(k for k, s in enumerate(body[:read]) if touches(s, Effect.WRITE))
    assert any(s.op == Op.BARRIER for s in body[write + 1:read]), (
        "nothing publishes the temporary between the store and the read:\n"
        + "\n".join(f"  {k}: {s.op}" for k, s in
                    enumerate(body[write:read + 1], write)))


#: The targets the forward of memory runs on: not the explicit vector
#: (`pir.Forward`).
SPMD_TARGETS = [t for t in ALL_TARGETS if t[0] != "esimd"]


def _shared_writes(gen):
    from tensorforge.backend.pir.core import MemSpace, walk_stmts
    return sum(1 for s in walk_stmts(gen._sections[0].body)
               for a in s.accesses
               if a.space is MemSpace.SHARED and a.kind & Effect.WRITE)


@pytest.mark.parametrize("case_stem", ["mixed/ew_then_ml", "mixed/red_then_ml"],
                         ids=["elementwise", "reduction"])
@pytest.mark.parametrize("backend,arch", SPMD_TARGETS,
                         ids=[b for b, _ in SPMD_TARGETS])
def test_a_contraction_reads_a_produced_temporary_out_of_registers(
        case_stem, backend, arch):
    """No memory round trip where every lane reads what it stored.

    A temporary produced pointwise or by a reduction is stored into its
    shared buffer behind the operation, and the contraction consuming it
    stages it from there into registers.  Each lane loads the elements it
    stored, so the pseudo-IR reads the registers the store took them from,
    and the store, which nothing reads then, goes (`pir.forward`): the value
    never reaches shared memory at all.  `keep_own`, which keeps such a
    buffer in registers of every lane's own as well, is off in both builds.
    """
    try:
        gen, _ = _generate(case_stem, backend, arch, keep_own=False)
        kept, _ = _generate(case_stem, backend, arch, forward_memory=False,
                            keep_own=False)
    except UNSUPPORTED as exc:
        pytest.skip(f"{case_stem} does not generate on {backend}: {exc}")
    assert _shared_writes(gen) < _shared_writes(kept), (
        "the temporary still goes through shared memory")
