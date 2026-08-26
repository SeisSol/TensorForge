# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What the sparse loader now says it produces, checked against what it emits.

`load_linear` used to hand back a value with no layout, and it was the largest
untracked population in the corpus: 3053 operand checks exempted because the
annotation was absent, 1841 of them the broadcast operand of an `fmacdpp16`
whose DPP pattern assumes a particular distribution.  `emitters.fmadpp` says
so in as many words -- `None` is allowed through because *the sparse loader
does not yet say what it produces*.

The distribution is not recoverable from the read.  A linearized register
operand is read as ``r[i / threads]``, which has no lane term at all; every
lane names the same slot and holds a different element, and which element is
decided entirely by the write.  So the layout is recorded by the write --
`Symbol._record_linear_layout`, called from `store_linear` -- and this file
checks that recording against the code the fill emits, rather than against the
prose next to it.

That is the same arrangement as `test_amd_relayout.py`, and for the same
reason: the two previous layout claims in this codebase were both wrong while
reading correctly.  `LaneAxis.holders` is the executable form of the map, and
agreement with a *parsed* fill means something that agreement with a docstring
does not.
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
SPARSE_CASE = "slicing/sparsity_band.py"

#: `float v0 = glb_m2[0 + threadIdx.x * 1];`
_READ = re.compile(
    r"^\s*\w+\s+(?P<tmp>v\d+)\s*=\s*(?P<src>\w+)\[(?P<base>\d+)\s*\+\s*"
    r"threadIdx\.x\s*\*\s*(?P<vec>\d+)\]\s*;", re.M)
#: `r1[0] = v0;`
_WRITE = re.compile(r"^\s*(?P<reg>r\d+)\[(?P<slot>\d+)\]\s*=\s*(?P<tmp>v\d+)\s*;",
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
    """`{register: {slot: (source_base, vec)}}`, read out of the emitted code."""
    tmp_of = {m.group("tmp"): (int(m.group("base")), int(m.group("vec")))
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
    ("slicing/sparsity_band.py", 16),
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
        for slot, (base, vec) in sorted(slots.items()):
            assert vec == 1, (
                f"{reg}[{slot}] is filled at vector width {vec}; the recorded "
                f"layout describes scalar granularity")
            for lane in range(threads):
                element = base + lane * vec
                assert axis.holders(element, threads) == (lane,), (
                    f"{reg}[{slot}]: lane {lane} loads element {element}, "
                    f"which {axis!r} says is held by "
                    f"{axis.holders(element, threads)}")
                assert axis.slot(element) == slot, (
                    f"{reg}[{slot}]: element {element} belongs in slot "
                    f"{axis.slot(element)} according to {axis!r}")


def test_a_sparse_operand_reaches_the_intrinsic_with_a_layout():
    """The point of all of it: `fmacdpp` sees a distribution, not a `None`.

    `emitters.fmadpp` compares what arrives against
    `fmadpp_operand_layout(step)` and lets `None` through unchecked.  Before
    this, every sparse operand took that exemption.
    """
    source = _generate(SPARSE_CASE)
    assert "fmacdpp" in source, "case no longer reaches the DPP path"

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


def test_a_non_register_symbol_is_left_alone():
    """Shared and global images are addressed as `name[i + threadIdx.x * vec]`,
    which is a different map; `_record_linear_layout` describes the register
    one and declines to speak about the others."""
    sym = Symbol("s0", SymbolType.SharedMem, obj=None)
    sym.num_threads = 16
    sym._record_linear_layout(0, 1)
    assert sym.layout is None


def test_clone_carries_the_layout():
    sym = _register(16)
    sym._record_linear_layout(0, 1)
    assert sym.clone().layout == sym.layout
