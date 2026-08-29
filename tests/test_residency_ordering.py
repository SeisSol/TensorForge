# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Ordering invariants the residency has to hold in the generated source.

Seven of the eight cases under ``cases/mixed/`` refuse to generate, and their
snapshots record the refusal, so nothing further is needed to pin them: when
the residency stops being private to ``MultilinearBuilder`` those snapshots
turn from a message into source, and the diff is the evidence.

``mixed_ml_glb_then_ew`` is the exception.  It generates, and it is wrong:

    // glb_m3 = abs(glb_m0)      <- reads whatever the caller left in M
    // glb_m0 = store{r>g}(r2);  <- writes M only now

A snapshot of that is green, so the defect needs stating as an invariant
instead.  That is this file.

Which targets it used to fail on is worth keeping in mind.  With a global
destination and register residency enabled the store was deferred to the
section boundary, so the pointwise read overtook it; where register residency
is off -- every vendor the placement flags in ``MultilinearBuilder.__init__``
do not name -- the store was eager and the order was already right.  The same
code, sorted by a placement decision.

It holds on all four now, because a descriptor that cannot consult the
residency has the tensors it touches settled back into memory before it runs.

The last test here is about a different lifetime and is green: a section's
residency has to empty at the section boundary rather than at the end of the
kernel, since a writeback emitted after the barrier that was meant to publish
it is wrong in a way no snapshot would flag.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

from tensorforge.backend.instructions.compute.elementwise import (
    ElementwiseInstruction)
from tensorforge.backend.instructions.memory.store import StoreRegToShr
from tensorforge.backend.instructions.sync_block import SyncThreads
from tensorforge.common.context import Context
from tensorforge.common.exceptions import GenerationError
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


def _generate(stem: str, backend: str, arch: str):
    """The generator and the descriptor list, whose tensors it has named."""
    mod = _load(stem)
    descrs = mod.descr_list()
    ctx = Context(arch=arch, backend=backend, fp_type=mod.DTYPE)
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
    gen, descrs = _generate("mixed/ml_glb_then_ew", backend, arch)
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
    """The one mixed shape that is already correct, pinned as a control.

    Two contractions write half a temporary each, so ``_written_in_slices``
    forces both into shared memory as they are produced and the residency is
    empty by the time the pointwise read happens.  That makes this the case
    which separates "the consumer cannot see the residency" from "the consumer
    cannot address a shared temporary at all": it is only ever the former.

    It also has to survive the change that fixes the others, which is the
    reason it is a test and not only a snapshot.
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
def test_a_sections_residency_empties_at_the_section_boundary(case_stem, backend,
                                                              arch):
    """The second section reads what the first one produced.

    The first contraction leaves its result in registers and the second
    section, past the barrier, reads it back from global memory.  Nothing in
    the second section knows about the first section's registers, so the
    writeback has to have happened by then -- and "by the end of the kernel"
    is not by then.

    Stated as an access order rather than against the barrier instruction,
    which each backend spells differently.
    """
    try:
        gen, descrs = _generate(case_stem, backend, arch)
    except (GenerationError, NotImplementedError) as exc:
        # Two separate gaps on the Intel targets, neither about ordering: a
        # group barrier under a simd-uniform trip count is refused, and
        # `SyclLexic.sync_grid` is an unimplemented stub.  Either way there is
        # no source to state an order about, and that the case does not
        # generate there is the snapshots' business.
        pytest.skip(f"{case_stem} does not generate on {backend}: "
                    f"{type(exc).__name__}: {exc}")
    array = f"glb_{descrs[0].dest.tensor.name}"
    accesses = _accesses(gen.get_kernel(), array)

    write = _first(accesses, "w")
    read = _first(accesses, "r")
    assert write is not None and read is not None, (
        f"expected {array} to be both written by the first section and read "
        f"by the second, got {[(k, ln) for k, ln, _ in accesses]}")
    assert write[1] < read[1], (
        f"{array} is read at line {read[1]} by the section after the barrier, "
        f"before the section before it wrote at line {write[1]}:\n"
        f"  {read[1]}: {read[2]}\n"
        f"  {write[1]}: {write[2]}")


@pytest.mark.parametrize("backend,arch", ALL_TARGETS,
                         ids=[b for b, _ in ALL_TARGETS])
def test_a_settled_temporary_is_published_before_it_is_read(backend, arch):
    """The store the flush emits gets a barrier, without one being emitted.

    `tmp = A @ B` leaves the result in an accumulator; the pointwise read
    cannot see an accumulator, so the tensor is settled into its shared buffer
    first.  Each lane writes its own part, and nothing says the pointwise
    operation hands the same elements back to the same lanes, so the store has
    to be published before the read.

    No barrier is emitted next to the flush, deliberately: `SyncThreadsOpt`
    discards every sync in the section and reinserts them from the
    shared-memory write/use pairs.  One placed by hand would be removed again.
    This is the assertion that the pair is recognised.

    Asked of the instruction stream rather than of the source, because a
    SIMD-scope barrier lowers to nothing on the targets where a wave runs in
    lockstep -- `HipLexic.sync_simd` returns None, and so does `SyclLexic` in
    SIMD mode.  Only CUDA emits text (`__syncwarp`, for the independent thread
    scheduling), so a textual assertion would test the architecture rather than
    the pass.
    """
    gen, _ = _generate("mixed/ml_then_ew", backend, arch)
    stream = _flatten(gen._sections[0].stream)

    def index_of(predicate, what):
        found = [i for i, instr in enumerate(stream) if predicate(instr)]
        assert len(found) == 1, f"expected exactly one {what}, found {found}"
        return found[0]

    store = index_of(lambda i: isinstance(i, StoreRegToShr), "shared store")
    read = index_of(lambda i: isinstance(i, ElementwiseInstruction),
                    "pointwise operation")
    assert store < read

    assert any(isinstance(instr, SyncThreads)
               for instr in stream[store + 1:read]), (
        "nothing publishes the temporary between the store and the read:\n"
        + "\n".join(f"  {i}: {type(instr).__name__}"
                    for i, instr in enumerate(stream[store:read + 1], store)))


@pytest.mark.parametrize("case_stem,expect_shared", [
    ("mixed/ew_then_ml", False),
    ("mixed/red_then_ml", False),
], ids=["elementwise", "reduction"])
@pytest.mark.parametrize("backend,arch", ALL_TARGETS,
                         ids=[b for b, _ in ALL_TARGETS])
def test_a_contraction_reads_a_produced_temporary_out_of_registers(
        case_stem, expect_shared, backend, arch):
    """No memory round trip when the consumer can read the image in place.

    A temporary produced pointwise or by a reduction lands in a register array
    with a pending writeback.  A contraction consuming it consults the
    residency and takes the array, so the value never reaches shared memory at
    all -- the writeback is dropped unflushed when nothing else wants it.

    This is the difference between settling a value because a consumer cannot
    see it and settling it on principle, and it is the reason `materialise`
    goes through registers rather than writing shared memory directly.
    """
    try:
        gen, _ = _generate(case_stem, backend, arch)
    except GenerationError as exc:
        pytest.skip(f"{case_stem} does not generate on {backend}: {exc}")
    except Exception as exc:            # ESIMD has no reduction lowering yet
        pytest.skip(f"{case_stem} does not generate on {backend}: {exc}")

    stream = _flatten(gen._sections[0].stream)
    shared_stores = [i for i in stream if isinstance(i, StoreRegToShr)]
    assert bool(shared_stores) is expect_shared, (
        f"expected {'a' if expect_shared else 'no'} shared store, found "
        f"{len(shared_stores)}; the temporary should stay in registers")
