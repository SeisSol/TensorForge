# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The NVIDIA matmul path is asked whether it can emit, not told to try.

`primitives/nvidia.py` was unreachable: `_is_matmul` asked
`vendor in ['amd']`, so the `elif vendor == 'nvidia'` branch under it could
never run.  Turning it on is one word, and one word is exactly the wrong size
for this change -- the emitter's preconditions were `assert` statements, which
were harmless only while nothing reached them.

With the path live an assertion is not a rejection, it is an abort: a case
with a 16-wide wave would stop generating altogether, when the generic path
handles it perfectly well.  So the preconditions became `nvidia.supports()`, a
question the caller asks first, and `_is_matmul` consults it.  This file
checks both halves -- that the gate turns the right cases away, and that a
case it turns away still comes out of the generator.

Measured while enabling it: 9 of the corpus's CUDA cases take the path, and
the same 9 snapshots changed.  No HIP snapshot moved.

The path is parked (`nvidia.ENABLED`) pending a run on real hardware: `"+f"`
versus `"=f"`/`"f"` on the accumulator is a register-allocation difference no
front end can see.  The tests below that need the emitter's output turn it on
for themselves.  A parked path whose tests skip is a path that quietly rots --
that is how `nvidia.py` accumulated 23 unreachable definitions in the first
place -- so what is checked here is the emitter, which is worth checking
whether or not it is deployed.  `test_the_switch_is_off` is the separate,
one-line statement of the deployment decision.
"""

from __future__ import annotations

import pytest

from tensorforge.backend.instructions.compute.primitives import nvidia
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context

ATOM_TYPE = Datatype.F32


# --------------------------------------------------------------------------- #
# What the gate admits
# --------------------------------------------------------------------------- #

def test_a_warp_wide_dense_case_of_the_atoms_type_is_admitted():
    assert nvidia.supports(32, ATOM_TYPE, sparse=None)


@pytest.mark.parametrize("threads", [1, 2, 4, 8, 16, 64])
def test_any_other_wave_width_is_turned_away(threads):
    """The emitter is warp-level throughout -- it stages operands through
    `__syncwarp` and indexes shared memory by `threadIdx.x` modulo the atom's
    `k`.  Narrower waves need a warp-level broadcast and a way back; wider
    ones are a different instruction."""
    assert not nvidia.supports(threads, ATOM_TYPE, sparse=None)


def test_a_different_operand_type_is_turned_away():
    """`ATOM` is a TF32 instruction and nothing downstream compares the
    operand type against it, so an f64 case would emit
    `mma.sync...f32.tf32.tf32.f32` over doubles.  Quietly wrong is worse than
    loudly unsupported."""
    other = Datatype.I64
    assert not nvidia.supports(32, other, sparse=None)


def test_a_sparse_operand_is_turned_away():
    """`matmul` already declines these by returning `False`.  The gate has to
    agree, because `temp_shmem` reserves shared memory off the same
    predicate -- disagreement means a reservation for a kernel that never
    uses it."""
    assert not nvidia.supports(32, ATOM_TYPE, sparse=lambda k, j: True)


# --------------------------------------------------------------------------- #
# A rejected case still generates
# --------------------------------------------------------------------------- #

def _generate(case, backend="cuda", arch="sm_86"):
    import importlib.util
    from pathlib import Path

    from tensorforge.common.context import Context
    from tensorforge.generators.generator import Generator

    path = Path(__file__).parent / "cases" / case
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ctx = Context(arch=arch, backend=backend,
                  fp_type=getattr(mod, "DTYPE", None) or Datatype.F32)
    gen = Generator(mod.descr_list(), ctx)
    gen.generate()
    return gen.get_kernel()


#: `NAME` is `gemm_56x18_x_18x18`; the snapshot goes by that.
CASE_THAT_TAKES_THE_PATH = "rectangular.py"


@pytest.fixture
def enabled(monkeypatch):
    """The emitter, independent of whether the path is deployed."""
    monkeypatch.setattr(nvidia, "ENABLED", True)


def test_the_switch_is_off():
    """Not an opinion about whether it should be -- a place where the
    deployment decision is written down once, so flipping it is a diff."""
    assert nvidia.ENABLED is False, (
        "the path is live now; drop this test and re-record the CUDA "
        "snapshots, which move for 9 cases")


def test_a_case_that_takes_the_path_emits_the_instruction(enabled):
    source = _generate(CASE_THAT_TAKES_THE_PATH)
    assert nvidia.INSTRS[1].name in source, (
        "the case no longer reaches the MMA path; pick another one")


def test_the_same_case_still_generates_when_the_gate_says_no(enabled,
                                                            monkeypatch):
    monkeypatch.setattr(nvidia, "supports", lambda *a, **k: False)
    source = _generate(CASE_THAT_TAKES_THE_PATH)
    assert source, "generation produced nothing"
    assert nvidia.INSTRS[1].name not in source, "the gate was not consulted"


# --------------------------------------------------------------------------- #
# The inline asm the path emits
# --------------------------------------------------------------------------- #

def test_the_accumulator_is_one_read_write_operand(enabled):
    """`D` and `C` are the same accumulator at every call site.

    Listing it as `"=f"` under outputs and again as `"f"` under inputs states
    two unrelated operands that happen to name one C++ lvalue, and nothing
    then requires the compiler to give them the same register: it may read the
    accumulator into one and write the result into another, dropping the
    accumulation.  `"+f"` is the constraint that says read-and-write.
    """
    source = _generate(CASE_THAT_TAKES_THE_PATH)
    assert '"+f"' in source, "the accumulator is not a read-write operand"
    assert '"=f"' not in source, (
        "an output-only constraint on an operand that is also read")


def test_the_operand_numbering_survives_the_fold(enabled):
    """PTX numbers outputs and inputs in one sequence, so folding C into D
    shifts A and B down by `len(C)`.  Getting that wrong reads the wrong
    registers and still compiles."""
    import re

    source = _generate(CASE_THAT_TAKES_THE_PATH)
    m = re.search(r'"\{([%\d,]+)\}, \{([%\d,]+)\}, \{([%\d,]+)\}, '
                  r'\{([%\d,]+)\};"', source)
    assert m, "no mma operand groups found"
    d, a, b, c = (g.split(",") for g in m.groups())
    assert d == c, "D and C must name the same operands once folded"
    numbers = [int(x.lstrip("%")) for x in d + a + b]
    assert numbers == list(range(len(numbers))), (
        f"operand numbering is not contiguous from 0: {numbers}")


def test_the_matmul_emits_no_raw_statements(enabled):
    """The end of the conversion, asserted rather than remembered.

    Every operand in this path is a value now: the accessors hand back the
    value instead of a name to write into, the accumulator slots and the
    padding fragments are `declare`, the warp syncs are `barrier`, the staging
    store is `store` over a `pack`, and `mma.sync` is `asm_stmt`.

    A count, not a list, because the list would need re-recording on every
    unrelated change.  Zero is the only number here that means anything: one
    raw statement is a place where a pass cannot see what the code does, and
    the whole point of the conversion was that there is no such place left.
    """
    import traceback

    from tensorforge.backend.pir import build as pir_build

    seen = []
    original = pir_build.IRBuilder.__call__

    def call(self, code, *args, **kwargs):
        frame = next((f for f in reversed(traceback.extract_stack())
                      if f.filename.endswith('primitives/nvidia.py')), None)
        if frame is not None:
            seen.append(f'nvidia.py:{frame.lineno}: {code.strip()[:60]}')
        return original(self, code, *args, **kwargs)

    pir_build.IRBuilder.__call__ = call
    try:
        _generate(CASE_THAT_TAKES_THE_PATH)
    finally:
        pir_build.IRBuilder.__call__ = original

    assert not seen, "raw statements left in the MMA path:\n" + "\n".join(seen)


def test_the_matmul_emits_no_raw_index_expressions(enabled):
    """The addresses are operations, not text.

    Raw statements went first; the addresses stayed as `rawexpr` for a while
    after, in six shapes over 5908 instances, all of them `threadIdx.x` and
    constants.  Text is where an address stops being analysable: `cse` cannot
    merge two identical `rawexpr` nodes because they are not pure, the bank
    census has to parse the generated source to answer a question the IR could
    answer directly, and a pass wanting to reason about the access pattern had
    nothing to reason over.

    A count, not a list, for the same reason as the statement test beside it.
    """
    import traceback

    from tensorforge.backend.pir import build as pir_build

    seen = []
    original = pir_build.IRBuilder.rawexpr

    def rawexpr(self, text, *args, **kwargs):
        frame = next((f for f in reversed(traceback.extract_stack())
                      if f.filename.endswith('primitives/nvidia.py')), None)
        if frame is not None:
            seen.append(f'nvidia.py:{frame.lineno}: {text.strip()[:60]}')
        return original(self, text, *args, **kwargs)

    pir_build.IRBuilder.rawexpr = rawexpr
    try:
        _generate(CASE_THAT_TAKES_THE_PATH)
    finally:
        pir_build.IRBuilder.rawexpr = original

    assert not seen, "raw index expressions left:\n" + "\n".join(seen)


def test_the_repeated_thread_reads_collapse(enabled):
    """What the conversion bought beyond the opacity count.

    Every one of those addresses started with `threadIdx.x`, and a `rawexpr`
    naming it is opaque and impure, so each was its own read.  As operations
    they are one value: 660 reads in this kernel became 192, and the kernel
    lost 234 lines.
    """
    source = _generate(CASE_THAT_TAKES_THE_PATH)
    assert source.count('threadIdx.x') < 300, (
        f"{source.count('threadIdx.x')} thread-index reads; they are supposed "
        f"to be hash-consed")


# -- which instruction, and why that one ----------------------------------- #

def test_the_ranking_reproduces_the_indices_it_replaced():
    """Two hardcoded dicts indexed the same list by hand, in `shmsize` and in
    `matmul`.  A size computed for one entry and an issue of another is a
    buffer nobody fills; the point of one function is that they cannot
    differ.  That it lands on the same entries is what makes the change
    inert."""
    assert nvidia.instr_for(Datatype.F32, 9, 56, 56, sm=80) is nvidia.INSTRS[1]
    assert nvidia.instr_for(Datatype.F64, 9, 56, 56, sm=80) is nvidia.INSTRS[2]


def test_the_i8_entries_are_not_candidates():
    """`generate`'s I8 branch is a `pass`, so an I8 entry would be selected
    and then emit nothing.

    Asked at the widest capability in the table rather than at the baseline:
    the baseline is a floor that excludes, so `instrs_for` returns nothing
    there and the loop below would pass by never running -- which is the
    trivially-true property `tools/mutation_check.py` exists to catch.
    """
    for dtype in (Datatype.F32, Datatype.F64):
        candidates = nvidia.instrs_for(dtype, sm=90)
        assert candidates, dtype
        for op in candidates:
            assert op.mode in nvidia.EMITTED_MODES


@pytest.mark.parametrize("sm", [80, 90, 120])
def test_the_reservation_covers_whichever_entry_is_issued(sm):
    """`shmsize` is asked without the shape the ranking reads, so it cannot
    reproduce the choice -- it bounds it instead.

    Now asked per capability, because the capability is what makes the bound
    load-bearing: the sm_90 F64 entries are wider than the sm_80 one, so a
    reservation sized against one table and an issue out of another is the
    overrun this bounds -- reached through the arch rather than through the
    shape.  `scratch` passes the context's, which is why it takes one.
    """
    from tensorforge.backend.instructions.compute.primitives import nvidia as n
    for dtype in (Datatype.F32, Datatype.F64):
        budget = n.shmsize(1, dtype, sm=sm)
        for op in n.instrs_for(dtype, sm=sm):
            aregs = (op.m * op.k) // 32
            bregs = (op.n * op.k) // 32
            cregs = (op.m * op.n) // 32
            assert budget >= 32 * max(aregs + bregs, cregs), op.name


def test_the_baseline_excludes_rather_than_guesses():
    """The floor for a caller with no target, and it is set to exclude.

    This test used to assert the opposite fact: that the baseline was 80 and
    that the SM_90 F64 entries stayed out of reach "until something plumbs the
    target's compute capability".  Something does now -- `sm_of` reads it off
    the context and `matmul`, `strategies` and `scratch` all pass it -- so what
    is left for the baseline is the case where there is no context at all.

    75 rather than 80 for that case, because a floor should refuse.  Every F32
    entry in the table is sm_80, so a caller with no target now selects nothing
    and falls through to the generic nest, instead of being handed sm_80 PTX
    for a target that may not run it.
    """
    assert nvidia.BASELINE_SM == 75
    assert nvidia.instrs_for(Datatype.F32) == ()
    assert nvidia.instrs_for(Datatype.F64) == ()


def test_the_context_is_what_brings_an_entry_into_reach():
    """`sm_of` is the plumbing, and this is what it buys.

    The SM_90 F64 entries are wider in both m and k, so a ranking prefers them
    -- and before the context reached this far they were unreachable on every
    target, sm_120 included.
    """
    wide = [op for op in nvidia.INSTRS if op.d is Datatype.F64 and op.sm > 80]
    assert wide, 'the table carries SM_90 F64 entries'

    assert not set(wide) & set(nvidia.instrs_for(Datatype.F64, sm=80))
    assert set(wide) & set(nvidia.instrs_for(Datatype.F64, sm=90))
    assert nvidia.instr_for(Datatype.F64, 9, 56, 56, sm=90) in wide

    ctx = Context(arch='sm_120', backend='cuda', fp_type=Datatype.F32)
    assert nvidia.sm_of(ctx) == 120
    assert nvidia.instr_for(Datatype.F64, 9, 56, 56,
                            sm=nvidia.sm_of(ctx)) in wide


def test_an_unreadable_target_falls_to_the_floor():
    """A model this cannot parse yields the floor, not a guess.

    The failure mode it forecloses: crediting an unrecognised target with
    instructions it may not have.  `None` reaches here from the tests that
    call `strategies` without a context.
    """
    assert nvidia.sm_of(None) == nvidia.BASELINE_SM

    class _NoModel:
        def get_vm(self):
            raise AttributeError
    assert nvidia.sm_of(_NoModel()) == nvidia.BASELINE_SM


@pytest.mark.parametrize(
    "atom", [op for op in nvidia.INSTRS if op.mode in nvidia.EMITTED_MODES],
    ids=lambda op: op.name.split('.aligned.')[1])
def test_the_accumulator_epilogue_lands_every_slot_exactly_once(atom):
    """The epilogue's store and its read-back have to be inverses.

    They are written apart -- 32 lanes each store `cregs` registers into a
    shared tile, a barrier, then each lane reads a row of it back -- so nothing
    in the source makes them agree.  What has to hold is that the stores land
    on the `m x n` tile exactly once and that the read-back names the same
    cells; a store that misses one leaves the caller's previous value there,
    and a store that leaves the tile corrupts whatever follows it.

    This is the check the F64 path did not have.  `m8n8k4.f64` stored one of
    its two slots at `2 * t + 64` in a tile of 64 elements and never stored the
    other: out of bounds and short by half, and the only symptom anywhere was
    NaN out of a device run with the deployment switch flipped.
    """
    threads = 32
    cregs = (atom.m * atom.n) // threads
    slots = nvidia.accumulator_slots(atom)

    assert len(slots) == cregs, (
        f"{len(slots)} slots for {cregs} accumulator registers")
    assert sorted(s for s, _ in slots) == list(range(cregs)), (
        "the slots have to be the operand list's own indices")

    written = {}
    for lane in range(threads):
        for slot, off in slots:
            idx = 2 * lane + off
            assert 0 <= idx < cregs * threads, (
                f"lane {lane} slot {slot} writes {idx} into a tile of "
                f"{cregs * threads}")
            assert idx not in written, (
                f"lane {lane} slot {slot} overwrites {written[idx]} at {idx}")
            written[idx] = (lane, slot)

    # The read-back: `(t % m) * n + jj`, guarded to the lanes of one m-tile.
    read = {(lane % atom.m) * atom.n + jj
            for lane in range(threads) for jj in range(atom.n)}
    assert read == set(written), (
        "the epilogue reads cells the stores never wrote, or the other way "
        f"round: {sorted(read ^ set(written))[:8]}")


def test_every_entry_states_the_capability_it_needs():
    """It was a comment on each row, which a selection cannot read."""
    for op in nvidia.INSTRS:
        assert op.sm in (75, 80, 90), op.name
