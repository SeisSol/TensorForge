# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""An operand loaded once is handed to a later operation only where it fits.

`pir.load_cse` gives a value read from memory to every later read of the same
location.  Within an operation that saves loads; from one operation to the
next it keeps the value live through everything in between, and where that
does not fit the register file the compiler spills it.  SeisSol's elastic
time derivative held its three star matrices through every step: 2 KB of
wave-uniform values a lane on sm_90, 1.8 to 2.3 KB of stack from order 4 in
double precision.  So a section's body that does not fit is built again with
each operation reading its operands itself, and kept where that needs fewer
registers (`Options.reread_over_budget`).
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import re
from pathlib import Path
from types import SimpleNamespace

from tensorforge.backend.instructions import abstract_instruction as ai
from tensorforge.common.context import Context, Options
from tensorforge.generators.generator import Generator

CASE = (Path(__file__).parent / "cases" / "slicing"
        / "register_operand_control.py")


def _build(budget=None, **options):
    spec = importlib.util.spec_from_file_location("reread_case", CASE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ctx = Context(arch="sm_86", backend="cuda", fp_type=mod.DTYPE,
                  options=Options(**options))
    if budget is not None:
        ctx.target.hw.max_reg_per_thread = budget
    gen = Generator(mod.descr_list(), ctx, measure_pressure=True)
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    return gen, ctx


def _normalized(src):
    """The kernel without what names it: value numbers, the digest, and the
    banner line that lists the options asked for."""
    src = re.sub(r"\bv\d+_?", "vN", src)
    src = re.sub(r"(?m)^\s*// options:.*$", "", src)
    return re.sub(r"kernel_kernel_[0-9a-f]+", "kernel_K", src)


def test_an_operand_held_across_operations_is_read_again_where_it_does_not_fit():
    """The chain's operands, held from one product to the next, are 2 KB of
    wave-uniform values against 1020 B of registers; read by each product
    itself they fit with room to spare."""
    held, ctx = _build(reread_over_budget=False)
    budget = ctx.target.hw.max_reg_per_thread
    assert held.metrics.peak_pressure > budget
    reread, _ = _build()
    assert reread.metrics.peak_pressure <= ai.FIT_FRACTION * budget


def test_where_the_body_fits_the_operands_are_still_handed_on():
    """With room to spare nothing is built again: the loads stay shared."""
    roomy, _ = _build(budget=1 << 30)
    held, _ = _build(budget=1 << 30, reread_over_budget=False)
    assert _normalized(roomy.get_kernel()) == _normalized(held.get_kernel())


def _context(budget, scalar=None):
    hw = SimpleNamespace(max_reg_per_thread=budget,
                         max_scalar_reg_per_wave=scalar,
                         vec_unit_length=32)
    target = SimpleNamespace(hw=hw, explicit_simd=False)
    return SimpleNamespace(target=target)


def test_one_file_holds_both_where_the_target_has_one(monkeypatch):
    """NVIDIA's uniform datapath holds no floats: a value every lane agrees
    on takes a vector register like any other, so the sum is what fits."""
    monkeypatch.setattr(ai, "body_pressure",
                        lambda ctx, body, simd, lanes: (900, [300, 700]))
    assert ai._register_excess(_context(1000), ()) == 900 - 850
    assert ai._register_excess(_context(2000), ()) == 0


def test_two_files_are_held_apart_where_the_target_has_two(monkeypatch):
    """AMD keeps a wave-uniform value in an SGPR: the lane-varying peak goes
    against the vector file and the uniform one against the scalar file."""
    monkeypatch.setattr(ai, "body_pressure",
                        lambda ctx, body, simd, lanes: (900, [300, 700]))
    assert ai._register_excess(_context(1000, scalar=1000), ()) == 0
    assert ai._register_excess(_context(1000, scalar=400), ()) == 700 - 340
    assert ai._register_excess(_context(300, scalar=1000), ()) == 300 - 255


def test_no_budget_stated_asks_nothing():
    assert ai._register_excess(_context(None), ()) is None
