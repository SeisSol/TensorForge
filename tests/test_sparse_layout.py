# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What the sparse loader says it produces, checked against what it emits.

A value from `load_linear` without a layout is exempt from every operand
check -- `emitters.fmadpp` lets `None` through -- and the broadcast operand of
an `fmacdpp16` has a DPP pattern that assumes a particular distribution.

The distribution is not recoverable from the read.  A linearized register
operand is read as ``r[i / threads]``, which has no lane term at all; every
lane names the same slot and holds a different element, and which element is
decided entirely by the write.  So the layout is recorded by the write --
`Symbol._record_linear_layout`, called from `store_linear` -- and this file
checks that recording against the code the fill emits, rather than against the
prose next to it.

That is the same arrangement as `test_amd_relayout.py`, and for the same
reason: a layout claim can read correctly and still be wrong.
`LaneAxis.holders` is the executable form of the map, and agreement with a
*parsed* fill means something that agreement with a docstring does not.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

from tensorforge.backend.pir.core import LaneAxis, RegisterLayout
from tensorforge.backend.symbol import Symbol, SymbolType
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context
from tensorforge.generators.generator import Generator

CASES = Path(__file__).resolve().parent / "cases"

#: A case whose B operand is sparse, so the linearized register path is taken.
#: (`NAME` is `gemm_sparse_band_B`; the snapshot goes by that.)
SPARSE_CASE = "sparsity_band.py"

#: `float v23_lin = glb_m2[0 + threadIdx.x * 1];`, or, where the run does not
#: fill a whole slot, the clamped tail
#: `float v33_lin = glb_m2[32 + ((threadIdx.x * 1) < 14 ? (threadIdx.x * 1) : 14 - 1)];`
#: -- `span` is how many elements are left, and the lanes above them read the
#: last one instead of the one past it (`Symbol._linear_addr`).
#:
#: The temporary is a PIR value, so its name comes from the shared allocator
#: and carries the builder's hint: `v23_lin`.
_READ = re.compile(
    r"^\s*\w+\s+(?P<tmp>v\d+\w*)\s*=\s*(?P<src>\w+)\[(?P<base>\d+)\s*\+\s*"
    r"(?:threadIdx\.x\s*\*\s*(?P<vec>\d+)"
    r"|\(\(threadIdx\.x\s*\*\s*(?P<cvec>\d+)\)\s*<\s*(?P<span>\d+)"
    r"\s*\?\s*\(threadIdx\.x\s*\*\s*\d+\)\s*:\s*\d+\s*-\s*\d+\))"
    r"\]\s*;", re.M)
#: `r1[0] = v23_lin;`
_WRITE = re.compile(r"^\s*(?P<reg>r\d+)\[(?P<slot>\d+)\]\s*=\s*(?P<tmp>v\d+\w*)\s*;",
                    re.M)


def _generate(case: str, arch: str = "gfx90a"):
    path = CASES / case
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ctx = Context(arch=arch, backend="hip",
                  fp_type=getattr(mod, "DTYPE", None) or Datatype.F32)
    gen = Generator(mod.descr_list(), ctx)
    gen.generate()
    return gen.get_kernel()


def _fills(source: str):
    """`{register: {slot: (source_base, vec, span)}}`, read out of the emitted
    code.  `span` is `None` for a run that fills the slot in every lane, and
    the number of lanes that hold an element for a clamped tail."""
    tmp_of = {m.group("tmp"): (int(m.group("base")),
                               int(m.group("vec") or m.group("cvec")),
                               int(m.group("span")) if m.group("span") else None)
              for m in _READ.finditer(source)}
    out = {}
    for m in _WRITE.finditer(source):
        src = tmp_of.get(m.group("tmp"))
        if src is not None:
            out.setdefault(m.group("reg"), {})[int(m.group("slot"))] = src
    return out


# --------------------------------------------------------------------------- #
# The claim, against the emitted fill
# --------------------------------------------------------------------------- #

# `case` is auto-parametrized across the whole corpus by conftest, so the
# name is taken; this test wants one specific sparse case.
@pytest.mark.parametrize("case_file,threads", [
    ("sparsity_band.py", 16),
])
def test_the_fill_puts_element_slot_times_threads_plus_lane_in_each_slot(
        case_file, threads):
    """`LaneAxis(threads, 1)`, re-derived from the generated statements.

    For every slot the loader fills, the read it came from is
    ``glb[base + threadIdx.x * vec]``, so lane `t` puts element `base + t*vec`
    into that slot.  The recorded layout claims lane `t` holds element
    `slot * threads + t`; the two agree exactly when `base == slot * threads`
    and `vec == 1`, and that is what is checked --- through `holders`, so the
    test asks the type rather than restating it.
    """
    fills = _fills(_generate(case_file))
    assert fills, "no linearized register fill in the generated source"

    axis = LaneAxis(threads, 1)
    for reg, slots in fills.items():
        for slot, (base, vec, span) in sorted(slots.items()):
            assert vec == 1, (
                f"{reg}[{slot}] is filled at vector width {vec}; the recorded "
                f"layout describes scalar granularity")
            # Up to `span` where the tail is clamped: the lanes above it hold
            # a duplicate of the last element, which the map does not describe
            # and no broadcast reads.
            for lane in range(threads if span is None else span):
                element = base + lane * vec
                assert axis.holders(element, threads) == (lane,), (
                    f"{reg}[{slot}]: lane {lane} loads element {element}, "
                    f"which {axis!r} says is held by "
                    f"{axis.holders(element, threads)}")
                assert axis.slot(element) == slot, (
                    f"{reg}[{slot}]: element {element} belongs in slot "
                    f"{axis.slot(element)} according to {axis!r}")


def _nonzeros(case: str) -> int:
    """How many elements the case's compressed operand has."""
    path = CASES / case
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    counts = {t.spp.count_nz()
              for d in mod.descr_list() for m in d.matrix_list()
              for t in [getattr(m, "tensor", m)]
              if t is not None and not t.is_dense()}
    assert len(counts) == 1, (
        f"{case} has {len(counts)} compressed shapes {sorted(counts)}; the "
        f"bound below names one operand")
    return counts.pop()


def test_no_linear_read_runs_past_the_operand():
    """The read that faults, as against the read that is merely wrong.

    A flat fill covers the operand in whole hops of `threads` elements and a
    tail of fewer, and the lanes above the tail have nothing to read.  Left
    unguarded they read into the next matrix of the batch -- 2 of 16 lanes
    here, 23 of 32 for a 9-element operand -- and at the *last* matrix there
    is no next one: the allocation ends and the read faults, which is what it
    did on AMD.  So the tail clamps its index and the lanes above it reread
    the last element (`Symbol._linear_addr`).

    Stated as the bound rather than as the clamp, so that covering the tail
    some other way -- predicating it, or narrowing the hop -- still passes.
    That the case *has* a partial tail is asked of the arithmetic and not of
    the emitted form, which would make the test pass for the one reason it
    exists to rule out; and the reads have to account for every slot, so a
    tail written in a form this does not parse fails here instead of
    vanishing from it.
    """
    source = _generate(SPARSE_CASE)
    total = _nonzeros(SPARSE_CASE)
    threads = 16
    assert total % threads, (
        f"{SPARSE_CASE} covers {total} elements in whole hops of {threads}, "
        f"so it has no tail and this says nothing")
    reads = [(m.group("src"), int(m.group("base")),
              int(m.group("vec") or m.group("cvec")),
              int(m.group("span")) if m.group("span") else None)
             for m in _READ.finditer(source)]
    assert len({(src, base) for src, base, _, _ in reads}) == \
        -(-total // threads), (
            f"{len(reads)} linearized reads for {total} elements over "
            f"{threads} lanes; the fill is written in a form this test does "
            f"not parse, so the bound below would not be checked")
    for src, base, vec, span in reads:
        lanes = threads if span is None else span
        top = base + (lanes - 1) * vec + vec - 1
        assert top < total, (
            f"{src}[{base} + lane * {vec}] reaches element {top} of an "
            f"operand with {total}; the lanes past the end read into the "
            f"next matrix of the batch, and the last one has no next")


def test_a_sparse_operand_reaches_the_intrinsic_with_a_layout():
    """The point of all of it: `fmacdpp` sees a distribution, not a `None`.

    `emitters.fmadpp` compares what arrives against
    `fmadpp_operand_layout(step)` and lets `None` through unchecked, so a
    sparse operand without a layout would take that exemption.
    """
    source = _generate(SPARSE_CASE)
    # as a call, or as a block of them on gfx9 (`amd.dpp_block`)
    assert re.search(r"fmacdpp|v_fmac_f(32|64)_dpp", source), \
        "case no longer reaches the DPP path"

    from tensorforge.backend.pir import build as pirbuild
    from tensorforge.backend.pir.core import Value

    seen = []
    original = pirbuild.IRBuilder.call_stmt

    def call_stmt(self, callee, *args, **kwargs):
        if callee.startswith("tensorforge::fmacdpp") and len(args) > 1:
            operand = args[1]
            if isinstance(operand, Value) and operand.hint == "lin":
                seen.append(operand.layout)
        return original(self, callee, *args, **kwargs)

    pirbuild.IRBuilder.call_stmt = call_stmt
    try:
        _generate(SPARSE_CASE)
    finally:
        pirbuild.IRBuilder.call_stmt = original

    assert seen, "no sparse operand reached fmacdpp"
    assert all(layout is not None for layout in seen), (
        f"{sum(l is None for l in seen)} of {len(seen)} sparse operands still "
        f"arrive untracked")


# --------------------------------------------------------------------------- #
# When the recorder declines to say
# --------------------------------------------------------------------------- #

def _register(threads=16):
    sym = Symbol("r0", SymbolType.Register, obj=None)
    sym.num_threads = threads
    return sym


def test_a_slot_aligned_fill_is_recorded():
    sym = _register(16)
    for index in (0, 16, 32):
        sym._record_linear_layout(index, 1)
    assert sym.layout == RegisterLayout((LaneAxis(16, 1),))


def test_a_fill_that_does_not_start_on_a_slot_boundary_says_nothing():
    """Not conservatism for its own sake: `r[i / threads]` floor-divides, so a
    fill starting mid-slot writes somewhere the map does not describe, and a
    layout claimed anyway would be a wrong one rather than a missing one."""
    sym = _register(16)
    sym._record_linear_layout(8, 1)
    assert sym.layout is None


def test_a_vector_fill_is_measured_in_granules():
    sym = _register(16)
    sym._record_linear_layout(64, 4)      # 64 == 16 * 4, one whole slot in
    assert sym.layout == RegisterLayout((LaneAxis(16, 1),))
    other = _register(16)
    other._record_linear_layout(32, 4)    # not a multiple of 16 * 4
    assert other.layout is None


def test_two_fills_that_disagree_leave_it_unknown():
    """A second fill with a different distribution would otherwise hand its
    claim to consumers of the first."""
    sym = _register(16)
    sym._record_linear_layout(0, 1)
    assert sym.layout is not None
    sym.num_threads = 32
    sym._record_linear_layout(0, 1)
    assert sym.layout is None


def test_a_symbol_without_a_thread_count_says_nothing():
    sym = Symbol("s0", SymbolType.Register, obj=None)
    assert sym.num_threads is None
    sym._record_linear_layout(0, 1)
    assert sym.layout is None


def test_a_shared_image_records_the_same_distribution():
    """A shared staging image is filled by the same linearized run.

    Under SPMD an unknown distribution costs only precision, but under ESIMD
    a declaration cannot be written without one -- so the explicit-vector
    lowering needs the shared case answered too.

    The thread count is the difference.  A register image knows its own; a
    shared one does not, because a shared buffer is not owned by one
    multiplication, so the loader that writes the run passes it.
    """
    sym = Symbol("s0", SymbolType.SharedMem, obj=None)
    sym.num_threads = 16
    sym._record_linear_layout(0, 1)
    assert sym.layout == RegisterLayout((LaneAxis(16, 1),))


def test_a_shared_image_with_no_thread_count_still_declines():
    """`None` means unknown, and a shared symbol carries no thread count of
    its own -- so without one from the fill there is nothing to claim."""
    sym = Symbol("s0", SymbolType.SharedMem, obj=None)
    assert sym.num_threads is None
    sym._record_linear_layout(0, 1)
    assert sym.layout is None


def test_a_global_symbol_is_still_left_alone():
    """The two staging kinds are described; a global image is addressed
    differently and is not one of them."""
    sym = Symbol("g0", SymbolType.Global, obj=None)
    sym.num_threads = 16
    sym._record_linear_layout(0, 1)
    assert sym.layout is None


def test_clone_carries_the_layout():
    sym = _register(16)
    sym._record_linear_layout(0, 1)
    assert sym.clone().layout == sym.layout
