# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A materialised broadcast is taken only where it fits.

On RDNA the broadcast of a reused `A` value can be one DPP move read by
several plain FMAs, which VOPD may pair, or a DPP modifier on each FMA.  The
move keeps every moved value in a register until its last product, and on
`local_flux` at 16 lanes on gfx1150 that was 5.6 KB of scratch and 46 times
the runtime; the fused form was 8 % faster than the default.  So the body is
built, measured against the target's register budget, and built again fused
if the materialised form does not fit.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import re
from pathlib import Path

from tensorforge.backend.instructions.compute.primitives.amd import select
from tensorforge.common.context import Context
from tensorforge.generators.generator import Generator
from tensorforge.generators.lanes import LaneConfig

CASE = Path(__file__).parent / "cases" / "local_flux.py"


def _kernel(budget=None):
    spec = importlib.util.spec_from_file_location("lf_fallback", CASE)
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    ctx = Context(arch="gfx1150", backend="hip", fp_type=mod.DTYPE)
    if budget is not None:
        ctx.get_vm().get_hw_descr().max_reg_per_thread = budget
    gen = Generator(mod.descr_list(), ctx, lanes=LaneConfig(16, 56, 1))
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    return gen.get_kernel()


def _normalised(src):
    src = re.sub(r"\bv\d+_", "vN_", src)
    return re.sub(r"kernel_kernel_[0-9a-f]+", "kernel_K", src)


def test_a_materialised_broadcast_that_does_not_fit_is_fused():
    src = _kernel()
    assert "movdpp16" not in src
    assert "fmacdpp16" in src


def test_a_materialised_broadcast_that_fits_stays():
    """With room to spare the move is kept: it is what lets the FMAs pair."""
    src = _kernel(budget=1 << 30)
    assert "movdpp16" in src


def test_the_second_build_is_the_fused_build(monkeypatch):
    """Nothing of the discarded build survives but the value numbers it used."""
    fallback = _kernel()
    monkeypatch.setattr(select, "MATERIALISE_FROM", 10 ** 6)
    assert _normalised(fallback) == _normalised(_kernel())
