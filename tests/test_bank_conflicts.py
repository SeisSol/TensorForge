# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The bank-conflict model, against cases whose answer is known by hand.

A diagnostic that reports a number nobody can check is worse than none, and
this one got three things wrong before it agreed with arithmetic:

* it read the vector width out of `*(float4*)&tile[i]` with a pattern that
  only matched a single identifier, so every wide access was measured as four
  bytes;
* it decided store-versus-load from whether the line begins with the buffer
  name, which a cast store never does;
* it folded the element size and the access span into one number, so a
  `VectorT<float,4>` access was taken to have a lane stride of 64 bytes rather
  than 16 -- turning a conflict-free store into a reported 4-way.

Each of those made the tool *over*-report, which is the direction that gets a
check ignored.  So the model is pinned here against hand-computed answers, and
the tool reads the same function.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools"


def _tool():
    spec = importlib.util.spec_from_file_location(
        "bank_conflicts", TOOLS / "bank_conflicts.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bank_conflicts"] = mod
    spec.loader.exec_module(mod)
    return mod


bc = _tool()


@pytest.mark.parametrize("expr,base,width,want,why", [
    ("0", 4, 1, 1,
     "every lane reads one address: a broadcast, and free"),
    ("threadIdx.x", 4, 1, 1,
     "32 consecutive floats are 32 banks"),
    ("threadIdx.x * 32", 4, 1, 32,
     "the classic column read of a 32-wide row-major tile: every lane in "
     "bank 0 with a different address"),
    ("threadIdx.x * 33", 4, 1, 1,
     "padding the row by one is what fixes a stride-32 column read"),
    ("threadIdx.x * 2", 4, 1, 2,
     "an even stride halves the banks reached"),
    ("threadIdx.x * 4", 4, 4, 1,
     "consecutive float4: eight lanes of sixteen bytes cover the bank width, "
     "and the hardware serves them in exactly that phase"),
    ("threadIdx.x", 8, 1, 1,
     "consecutive doubles: two banks each, still a bijection"),
    ("threadIdx.x * 2", 8, 1, 2,
     "stride-2 doubles collide, which is what double2 exists to fix"),
])
def test_the_model_agrees_with_arithmetic(expr, base, width, want, why):
    assert bc.ways(expr, base, width) == want, why


def test_the_mma_fragment_before_and_after_the_swizzle():
    """The measurement the swizzle was built from, kept where it can fail."""
    plain = "(threadIdx.x % 4) + (threadIdx.x / 4) * 8"
    swizzled = f"({plain}) ^ (((({plain}) >> 3)) & 7)"
    assert bc.ways(plain, 4, 1) == 2
    assert bc.ways(swizzled, 4, 1) == 1


def test_a_guard_narrows_the_lanes():
    """Counting inactive lanes into a bank is how a diagnostic cries wolf.

    The staging steps are guarded to a quarter or a half of the wave, so
    without this every one of them reports a conflict it does not have.
    """
    half = bc._lanes_under(["threadIdx.x < 16"])
    assert half == list(range(16))
    band = bc._lanes_under(["threadIdx.x >= 8 && threadIdx.x < 16"])
    assert band == list(range(8, 16))


def test_an_unreadable_guard_leaves_the_lanes_alone():
    """Over-reporting is recoverable; a silently narrowed lane set is not."""
    assert bc._lanes_under(["someRuntimeFlag"]) == list(range(bc.LANES))


def test_a_vector_cast_is_recognised():
    """The spelling the emitter actually produces, not the one a `float4`
    would have."""
    line = "*(tensorforge::VectorT<float, 4>*)&v59_atile[v225_a] = v224_q;"
    m = bc._CAST.search(line)
    assert m and bc._VECTOR_WIDTH.get(m.group(1).strip()) == 4


def test_a_cast_store_is_not_counted_as_a_load():
    windows = {"tile": 4}
    line = "*(tensorforge::VectorT<float, 4>*)&tile[0] = q;"
    kinds = [k for _n, _i, _b, _w, k, _d, _l in bc.accesses(
        "float* tile = &arena[0];\n" + line)]
    assert kinds == ["store"], kinds


def test_an_address_that_is_not_static_is_reported_not_guessed():
    with pytest.raises(bc.Unresolved):
        bc.ways("someRuntimeValue + threadIdx.x", 4, 1)


# --------------------------------------------------------------------------- #
# What the census could not see
# --------------------------------------------------------------------------- #

def test_a_loop_variable_is_resolved_at_its_initialiser():
    """An unresolved access is one the census does not count, which is a blind
    spot and not a caveat.  77 of them were loop variables.

    Substituting the initialiser is sound because these are bounds over tensor
    dimensions: every lane in the wave is on the same iteration, so the value
    shifts every lane's address by the same amount and the bank pattern is
    unchanged.  Anything lane-dependent reaches the address through
    `threadIdx`, which is substituted separately.
    """
    source = ("float* tile = &arena[0];\n"
              "for (int32_t i = 0; i < 6; i += 1) {\n"
              "  float x = tile[threadIdx.x + i * 16];\n"
              "}\n")
    got = [bc.ways(bc._resolve(index, defs), base, width, lanes)
           for _n, index, base, width, _k, defs, lanes in bc.accesses(source)]
    assert got == [1], got


def test_an_identifier_that_is_not_a_generator_name_is_still_substituted():
    """`_resolve` matched only `v{n}` names, so a definition sitting in the
    table went unused because the variable was called `i`."""
    assert bc._resolve('i * 16', {'i': '0'}) == '(0) * 16'


@pytest.mark.parametrize("expr", [
    "threadIdx.x + threadIdx.y * blockDim.x",
    "threadIdx.x + blockIdx.x * 32",
])
def test_the_warp_uniform_builtins_do_not_stop_the_census(expr):
    """They shift every lane's address equally, so they cannot create or
    remove a conflict -- and refusing the access left it uncounted."""
    assert bc.ways(expr, 4, 1) == 1


def test_a_genuinely_unknown_symbol_is_still_refused():
    """The point is to resolve what is knowable, not to guess."""
    with pytest.raises(bc.Unresolved):
        bc.ways("someRuntimeValue + threadIdx.x", 4, 1)


def test_a_member_access_is_not_a_substitutable_name():
    """`threadIdx.x` ends in an identifier that a word boundary matches, so
    widening the resolver rewrote the thread index into whatever `x` was.  The
    result was a self-referential expression 25 levels deep."""
    assert bc._resolve('threadIdx.x + i', {'x': '99', 'i': '0'}) \
        == 'threadIdx.x + (0)'


def test_a_loaded_value_is_not_an_address_expression():
    """`float x = tile[...]` defines `x` from memory, not from the lane.
    Substituting it into an address produces something that is neither."""
    source = ("float* tile = &arena[0];\n"
              "float x = tile[threadIdx.x];\n"
              "float y = tile[threadIdx.x + 1];\n")
    for _n, index, base, width, _k, defs, lanes in bc.accesses(source):
        assert 'x' not in defs and 'y' not in defs
        assert bc.ways(bc._resolve(index, defs), base, width, lanes) == 1
