# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Holding ptxas to the body's order (`Options.pin_loads`).

ptxas schedules a body as it likes, whatever order the text has; a
`__syncwarp()` is the one point it moves no memory access across, and a NOP in
converged code.  So where a body would spill, the emitter writes it in its own
order, nothing inlined, with a fence in front of every so many loads -- and
ptxas' demand follows the count `pir.pressure` takes of that order.
"""

from __future__ import annotations

import contextlib
import io
import re
from types import SimpleNamespace

import seissol_suite as suite
from tensorforge.backend.instructions import abstract_instruction as ai
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context, Options
from tensorforge.frontend.yateto import DescriptionReader
from tensorforge.generators.generator import Generator


def _build(config, kernel, arch='sm_90', backend='cuda', **options):
    system = config.rsplit('-o', 1)[0]
    dt = (Datatype.F64 if suite.config(system, config)['precision'] == 'd'
          else Datatype.F32)
    d = DescriptionReader(None, {}).read(
        suite.description(system, config, kernel))[0]
    ctx = Context(arch=arch, backend=backend, fp_type=dt,
                  options=Options(**options))
    gen = Generator(d, ctx)
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    return gen.get_kernel()


def _normalized(src):
    src = re.sub(r'(?m)^\s*// options:.*$', '', src)
    return re.sub(r'kernel_kernel_[0-9a-f]+', 'K', src)


def _context(vendor='nvidia', backend='cuda', budget=1020, pin=32):
    hw = SimpleNamespace(vendor=vendor, vec_unit_length=32,
                         max_reg_per_thread=budget)
    target = SimpleNamespace(hw=hw, backend=backend, explicit_simd=False)
    opts = SimpleNamespace(pin_loads=pin)
    return SimpleNamespace(target=target, get_user_options=lambda: opts)


def test_nothing_is_pinned_unless_asked():
    assert ai._pin_every(_context(pin=0), (), 32) == 0
    assert ai._pin_every(_context(pin=32), (), 32) == 32


def test_only_cuda_and_only_whole_warps():
    """The fence is `__syncwarp()`; a multiplication narrower than its warp
    shares it with others whose batch loop can end a trip earlier."""
    assert ai._pin_every(_context(vendor='amd', backend='hip'), (), 64) == 0
    assert ai._pin_every(_context(), (), 16) == 0
    assert ai._pin_every(_context(), (), 64) == 32


def test_auto_pins_where_ptxas_would_spill(monkeypatch):
    """`'auto'`: the count of the body, read through `PTXAS_NEED`, against
    the register file -- 255 registers -- less `PIN_MARGIN`."""
    a, b = ai.PTXAS_NEED
    edge = 255 - ai.PIN_MARGIN
    fits = int((edge - a) / b * 4) - 8
    spills = int((edge - a) / b * 4) + 8
    monkeypatch.setattr(ai, 'body_pressure',
                        lambda ctx, body, simd, lanes: (fits, [fits, 0]))
    assert ai._pin_every(_context(pin='auto'), (), 32) == 0
    monkeypatch.setattr(ai, 'body_pressure',
                        lambda ctx, body, simd, lanes: (spills, [spills, 0]))
    assert ai._pin_every(_context(pin='auto'), (), 32) == ai.PIN_EVERY


def test_a_spilling_body_is_fenced_and_written_in_its_order():
    """Viscoelastic order 5, single precision: the derivative ptxas spills
    1.3 KB of; pinned, a fence every 32 loads and no chain inlined."""
    config, kernel = 'viscoelastic-linearckanelastic-o5-s', 'gpu_derivative'
    base = _build(config, kernel, pin_loads=0)
    pinned = _build(config, kernel, pin_loads='auto')
    added = pinned.count('__syncwarp();') - base.count('__syncwarp();')
    assert added > 10
    # nothing inlined: every product a statement of its own, where the
    # inlined text spells a chain as one nested expression
    def chained(src):
        return len(re.findall(r'\(v\d+_\w+ \* v\d+_\w+\) \+', src))
    assert chained(base) > 0 and chained(pinned) == 0


def test_no_fence_under_a_lane_guard():
    """A full-mask `__syncwarp` under `if (vN_g)` would wait for lanes the
    guard keeps out."""
    src = _build('viscoelastic-linearckanelastic-o5-s', 'gpu_derivative',
                 pin_loads=1)
    depth, guarded = [], 0
    for line in src.split('\n'):
        code = line.split('//')[0]
        if '__syncwarp();' in code and any(re.search(r'if \(v\d+_g\)', o)
                                            for o in depth):
            guarded += 1
        for ch in code:
            if ch == '{':
                depth.append(code)
            elif ch == '}' and depth:
                depth.pop()
    assert guarded == 0


def test_off_and_elsewhere_the_source_is_unchanged():
    for arch, backend in (('sm_90', 'cuda'), ('gfx942', 'hip')):
        plain = _build('elastic-linearck-o4-d', 'gpu_neighboringFlux[0]',
                       arch=arch, backend=backend)
        off = _build('elastic-linearck-o4-d', 'gpu_neighboringFlux[0]',
                     arch=arch, backend=backend, pin_loads=0)
        assert _normalized(plain) == _normalized(off)
    amd = _build('elastic-linearck-o4-d', 'gpu_neighboringFlux[0]',
                 arch='gfx942', backend='hip', pin_loads=32)
    assert '__syncwarp();' not in amd
