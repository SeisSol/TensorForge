# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The widened path, checked against the scalar one by evaluating both.

Until the host oracle could model vector values there was no numerical
coverage of the vectorised path at all -- and worse than none for a while,
because a catch-all for `cuda::pipeline` swallowed every statement containing
`::`, so a widened kernel evaluated to a destination of all zeros and every
oracle test passed without touching the arithmetic it was there to check.

What these compare is the generated code against *itself*: the same case with
the vectorisation off and on has to write the same numbers to the same places.
That is the check that catches a lane-mapping disagreement, which is the
failure mode the whole arrangement is prone to -- the compute instruction
blocks the register image by the width and every other loop over it has to
agree, and when one does not the code still compiles and the snapshot still
looks plausible.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

import kernel_eval
from tensorforge.backend.instructions.memory import vectorize
from tensorforge.common.context import Context
from tensorforge.generators.generator import Generator

VEC_CASES = ['aligned_operands']


def _destination(name, widen, blocking=1, threads=64):
    old = (vectorize.LEAD_VECTORIZE, vectorize.LEAD_BLOCKING)
    vectorize.LEAD_VECTORIZE, vectorize.LEAD_BLOCKING = widen, blocking
    try:
        path = pathlib.Path(__file__).parent / 'cases' / f'{name}.py'
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        gen = Generator(mod.descr_list(),
                        Context(arch='sm_86', backend='cuda',
                                fp_type=mod.DTYPE))
        gen.generate()
        src = gen.get_kernel()
    finally:
        vectorize.LEAD_VECTORIZE, vectorize.LEAD_BLOCKING = old
    out = {}
    for tid in range(threads):
        mem = kernel_eval.evaluate(src, tid=tid, seed=11, globals_only=True)
        out.update({k: v for k, v in mem.items() if k[0] == 'm0'})
    return src, out


@pytest.mark.parametrize('blocking', [1, 2, 4])
@pytest.mark.parametrize('vcase', VEC_CASES)
def test_blocking_does_not_move_the_destination(vcase, blocking):
    """More than one vector per lane, which is where the two readings of a
    slot number stopped agreeing.

    `build` answers in elements and `build_nonlead` in register floats, and
    the width separates them; taking the scaled one in both applied it twice.
    At one slot per lane -- every arrangement the width alone produces -- the
    slot is 0 and the two readings agree, so nothing showed it until a lane
    held two.
    """
    _, base = _destination(vcase, widen=False)
    _, wide = _destination(vcase, widen=True, blocking=blocking)
    assert set(base) == set(wide)
    for key in sorted(base):
        assert base[key] == pytest.approx(wide[key], abs=1e-4), key


@pytest.mark.parametrize('vcase', VEC_CASES)
def test_the_widened_kernel_writes_the_same_numbers(vcase):
    """The check that would have caught the store-side lane mismatch.

    The compute instruction writes the register image blocked by the width;
    the store, the loader and the linear pass all read it back, and a cyclic
    reader of a blocked image puts fourteen of sixteen entries in the wrong
    place for `w = 2` without any diagnostic at all.
    """
    _, base = _destination(vcase, widen=False)
    src, wide = _destination(vcase, widen=True)
    assert 'VectorT' in src, 'the case did not actually vectorise'
    assert set(base) == set(wide), 'the widened kernel wrote elsewhere'
    for key in sorted(base):
        assert base[key] == pytest.approx(wide[key], abs=1e-4), key


@pytest.mark.parametrize('vcase', VEC_CASES)
def test_the_widened_kernel_is_not_trivially_empty(vcase):
    """Guards the guard.

    A destination of all zeros compares equal to nothing and would have made
    the test above vacuous -- which is exactly what the `'::'` catch-all
    produced before it was narrowed.
    """
    _, wide = _destination(vcase, widen=True)
    assert wide
    assert any(abs(v) > 1e-9 for v in wide.values())


# --------------------------------------------------------------------------- #
# Cross-lane traffic, which needs the lanes run together
# --------------------------------------------------------------------------- #

def _wave(name, widen, blocking=1, lanes=32):
    old = (vectorize.LEAD_VECTORIZE, vectorize.LEAD_BLOCKING)
    vectorize.LEAD_VECTORIZE, vectorize.LEAD_BLOCKING = widen, blocking
    try:
        path = pathlib.Path(__file__).parent / 'cases' / f'{name}.py'
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        gen = Generator(mod.descr_list(),
                        Context(arch='sm_86', backend='cuda',
                                fp_type=mod.DTYPE))
        gen.generate()
        src = gen.get_kernel()
    finally:
        vectorize.LEAD_VECTORIZE, vectorize.LEAD_BLOCKING = old
    return src, kernel_eval.evaluate_wave(src, lanes, seed=11,
                                          globals_only=True)


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
