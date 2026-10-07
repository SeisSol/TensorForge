# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Moving a transfer must not change the numbers.

`enable_wrap_loads` issues the first transfers of an iteration at the tail of
the one before it, for the element it is about to compute.  It is a
scheduling change; the results are supposed to be identical, bit for bit,
because the same values are read in the same order.

The snapshot tests record one configuration and the syntax tests only ask
whether it compiles, so this is where the numbers are compared, by running
both builds on the host oracle.  That needs the oracle to read a wrapped
kernel: the wrap carries the element flags as `uint32_t` words, and without
`uint32_t` in `_DECL` the first line ahead of the loop would abort the run.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import pathlib

import pytest

from tensorforge.common.context import Context, Options
from tensorforge.generators.generator import Generator
from tensorforge.reference import kernel_eval


def _load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _generate(mod, wrap, report=None, **extra):
    ctx = Context(arch='sm_86', backend='cuda',
                  fp_type=getattr(mod, 'DTYPE', None),
                  options=Options(enable_wrap_loads=wrap, **extra))
    gen = Generator(mod.descr_list(), ctx)
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    if report is not None:
        report.extend(gen.metrics.wrap_report or [])
    return gen.get_kernel(), kernel_eval.launch_geometry(gen.get_launcher())


def _run(src, geometry):
    """One block over one memory, at the width the launcher starts.

    Not lane by lane over a fixed set of tids.  A staged operand arrives
    cooperatively, so a lane running on its own memory fills one stripe of the
    window and reads seed fill for the rest --- and two builds then agree about
    a window neither of them wrote.  A fixed tid set can also reach past the
    end of the block, where the unguarded hops copy from beyond the operand.
    """
    lanes, mults = geometry
    return kernel_eval.evaluate_wave(src, lanes, seed=17, globals_only=True,
                                     mults=mults)


def _case(name):
    return _load(next(p for p in
                      (pathlib.Path(__file__).parent / 'cases').rglob('*.py')
                      if p.stem == name))


def _compare(name, report=None):
    """`(plain, wrapped)` destinations, or None when not evaluable."""
    mod = _case(name)
    plain, pg = _generate(mod, False)
    wrapped, wg = _generate(mod, True, report)
    if plain == wrapped:
        return None
    return _run(plain, pg), _run(wrapped, wg)


def _same(both):
    plain, wrapped = both
    for key in sorted(set(plain) | set(wrapped)):
        assert plain.get(key, 0.0) == pytest.approx(wrapped.get(key, 0.0),
                                                    abs=1e-4), key


def test_the_oracle_can_read_a_wrapped_kernel():
    """The gap that would hide everything else.

    Without `uint32_t` in the declaration pattern a wrapped kernel aborts
    ahead of its loop, where the flag words it carries are declared, and
    every case below silently becomes unevaluable.
    """
    assert kernel_eval._DECL.match(
        'const uint32_t flagWordFirst = flags0 == nullptr ? 1 : flags0[0];')
    src, geometry = _generate(_case('square_notrans'), True)
    assert 'flagWordFirst' in src
    assert _run(src, geometry)


@pytest.mark.parametrize('name', ['square_notrans', 'rectangular',
                                  'accumulate_chain', 'chain'])
def test_prefetch_does_not_change_the_numbers(name):
    both = _compare(name)
    if both is None:
        pytest.skip('prefetch changed nothing in this kernel')
    _same(both)


def test_a_moved_shared_transfer_does_not_change_the_numbers():
    """A shared window filled at the tail, for the next element.

    What keeps it right is not the pass alone: the barrier placement fences
    the write against the reads of the iteration it ends, and the allocator
    keeps the window live across the back edge.  Either one wrong is wrong
    numbers and not a crash.  `trans_a` moves a shared transfer.
    """
    report = []
    both = _compare('trans_a', report)
    assert any(line.endswith('[shr]') for line in report), report
    _same(both)


@pytest.mark.parametrize('name', ['addressing_none', 'known_zero_rows',
                                  'tw_split_load'])
def test_a_second_stage_does_not_change_the_numbers(name):
    """The copy for the next element issued at the head of the body, into
    the stage the iteration does not read.

    What keeps it right is the stage the loop carries, the windows into it,
    and the barriers between one stage's readers and the copy into it an
    iteration later -- any of them wrong is wrong numbers, not a crash.
    """
    report = []
    mod = _case(name)
    plain, pg = _generate(mod, False)
    staged, sg = _generate(mod, True, report, enable_multibuffer=True)
    assert any(line.endswith('[shr, 2 stages]') for line in report), report
    _same((_run(plain, pg), _run(staged, sg)))
