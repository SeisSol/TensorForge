# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The widened path, checked against the scalar one by evaluating both.

The host oracle models vector values, which is what puts numbers behind the
vectorized path without a GPU.  What these compare is the generated code
against *itself*: the same case with the vectorization off and on has to write
the same numbers to the same places.  That is the check that catches a
lane-mapping disagreement, which is the failure mode the whole arrangement is
prone to -- the compute instruction blocks the register image by the width and
every other loop over it has to agree, and when one does not the code still
compiles and the snapshot still looks plausible.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

import kernel_eval
from tensorforge.common.context import Context, Options
from tensorforge.generators.generator import Generator

VEC_CASES = ['aligned_operands']


def _build(name, widen, blocking=1):
    """The kernel and the geometry its launcher starts it with."""
    path = pathlib.Path(__file__).parent / 'cases' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    gen = Generator(mod.descr_list(),
                    Context(arch='sm_86', backend='cuda',
                            fp_type=mod.DTYPE,
                            options=Options(lead_vectorize=widen,
                                            lead_blocking=blocking)))
    gen.generate()
    return gen.get_kernel(), kernel_eval.launch_geometry(gen.get_launcher())


def _destination(name, widen, blocking=1):
    """The destination, run as one block over one memory.

    A staged operand arrives cooperatively --- lane `t` copies its own stripe
    and no other --- so a lane run on its own memory would compute from a
    window that is one stripe of operand and seed fill everywhere else, and
    two configurations would agree because they were both reading the same
    fill.  And no fixed count is this kernel's width: the widened build here
    is launched with four lanes, so of 64 tids, sixty would be a second
    block's threads addressing one block's memory, copying past the end of
    the operand as they went.
    """
    src, (lanes, mults) = _build(name, widen, blocking)
    mem = kernel_eval.evaluate_wave(src, lanes, seed=11, globals_only=True,
                                    mults=mults)
    return src, {k: v for k, v in mem.items() if k[0] == 'm0'}


@pytest.mark.parametrize('blocking', [1, 2, 4])
@pytest.mark.parametrize('vcase', VEC_CASES)
def test_blocking_does_not_move_the_destination(vcase, blocking):
    """More than one vector per lane, which is where the two readings of a
    slot number can disagree.

    `build` answers in elements and `build_nonlead` in register floats, and
    the width separates them; taking the scaled one in both would apply it
    twice.  At one slot per lane -- every arrangement the width alone
    produces -- the slot is 0 and the two readings agree, so only a lane
    holding two can show it.
    """
    _, base = _destination(vcase, widen=False)
    _, wide = _destination(vcase, widen=True, blocking=blocking)
    assert set(base) == set(wide)
    for key in sorted(base):
        assert base[key] == pytest.approx(wide[key], abs=1e-4), key


@pytest.mark.parametrize('vcase', VEC_CASES)
def test_the_widened_kernel_writes_the_same_numbers(vcase):
    """The check that catches a store-side lane mismatch.

    The compute instruction writes the register image blocked by the width;
    the store, the loader and the linear pass all read it back, and a cyclic
    reader of a blocked image puts fourteen of sixteen entries in the wrong
    place for `w = 2` without any diagnostic at all.
    """
    _, base = _destination(vcase, widen=False)
    src, wide = _destination(vcase, widen=True)
    assert 'VectorT' in src, 'the case did not actually vectorize'
    assert set(base) == set(wide), 'the widened kernel wrote elsewhere'
    for key in sorted(base):
        assert base[key] == pytest.approx(wide[key], abs=1e-4), key


@pytest.mark.parametrize('vcase', VEC_CASES)
def test_the_widened_kernel_is_not_trivially_empty(vcase):
    """Guards the guard.

    A destination of all zeros compares equal to nothing and would make the
    test above vacuous -- which is exactly what a `'::'` catch-all broad
    enough to swallow the computation produces.
    """
    _, wide = _destination(vcase, widen=True)
    assert wide
    assert any(abs(v) > 1e-9 for v in wide.values())


# --------------------------------------------------------------------------- #
# Cross-lane traffic, which needs the lanes run together
# --------------------------------------------------------------------------- #

def _wave(name, widen, blocking=1):
    src, (lanes, mults) = _build(name, widen, blocking)
    return src, kernel_eval.evaluate_wave(src, lanes, seed=11,
                                          globals_only=True, mults=mults)


@pytest.mark.parametrize('vcase', VEC_CASES)
def test_a_wave_run_agrees_with_the_scalar_kernel(vcase):
    """The same comparison, with the lanes advanced together.

    Everything a per-lane run could check, this checks too; what it adds is
    the cross-lane traffic. A `readlane` cannot be answered lane by lane --
    by the time the argument is evaluated it already holds the *reading*
    lane's copy, which is the one value the call is not asking for.
    """
    _, base = _wave(vcase, widen=False)
    src, wide = _wave(vcase, widen=True)
    assert 'VectorT' in src
    assert set(base) == set(wide)
    for key in sorted(base):
        assert base[key] == pytest.approx(wide[key], abs=1e-4), key


def test_a_cross_lane_read_outside_a_wave_run_refuses():
    """Rather than returning the local copy, which is how a broadcast
    disappears from the model without anything noticing."""
    interp = kernel_eval.Interp(kernel_eval.Slot(0), {})
    with pytest.raises(kernel_eval.Abort):
        interp.env['READLANE']('v1', 3)


def test_a_lane_masked_off_at_the_definition_refuses():
    """On the hardware the register holds whatever it held before, which is
    not something to invent a number for."""
    mem = kernel_eval.Slot(0)
    lanes = [kernel_eval.Interp(mem, {}) for _ in range(4)]
    kernel_eval.Lockstep(lanes)
    lanes[1].env['v9'] = 2.5
    assert lanes[0].env['READLANE']('v9', 1) == 2.5
    with pytest.raises(kernel_eval.Abort):
        lanes[0].env['READLANE']('v9', 2)
