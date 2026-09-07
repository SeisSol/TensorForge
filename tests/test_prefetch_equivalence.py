# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Prefetch must not change the numbers.

`enable_wrap_loads` moves a register transfer ahead of its consumer, wrapping
to the previous iteration when that runs off the front of the body. It is a
scheduling change; the results are supposed to be identical, bit for bit,
because the same values are read in the same order.

Nothing checked that. The snapshot tests record one configuration, the syntax
tests only ask whether it compiles, and the host oracle could not read a
prefetched kernel at all: `wrap.py` emits `uint32_t pipeStage0` as its
bookkeeping and `_DECL` had no `uint32_t`, so the very first line of the loop
aborted the run. The one configuration that most needed an oracle was the one
it could not evaluate.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import pathlib

import pytest

import kernel_eval
from tensorforge.common.context import Context, Options
from tensorforge.generators.generator import Generator

#: Reads its own destination through a rotating buffer whose first stage
#: nothing fills; see `test_the_known_prefetch_divergence`.
KNOWN_BAD = {'trans_a'}

TIDS = (0, 1, 7, 15, 16, 31)


def _load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _generate(mod, wrap):
    ctx = Context(arch='sm_86', backend='cuda',
                  fp_type=getattr(mod, 'DTYPE', None),
                  options=Options(enable_wrap_loads=wrap))
    gen = Generator(mod.descr_list(), ctx)
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    return gen.get_kernel()


def _run(src):
    out = {}
    for tid in TIDS:
        out.update(kernel_eval.evaluate(src, tid=tid, seed=17,
                                        globals_only=True))
    return out


def _compare(name):
    """`(plain, prefetched)` destinations, or None when not evaluable."""
    path = next(p for p in
                (pathlib.Path(__file__).parent / 'cases').rglob('*.py')
                if p.stem == name)
    mod = _load(path)
    plain, wrapped = _generate(mod, False), _generate(mod, True)
    if plain == wrapped:
        return None
    return _run(plain), _run(wrapped)


def test_the_oracle_can_read_a_prefetched_kernel():
    """The gap that hid everything else.

    Without `uint32_t` in the declaration pattern this aborts on the loop's
    first statement, and every case below silently becomes unevaluable.
    """
    src = _generate(_load(next(
        p for p in (pathlib.Path(__file__).parent / 'cases').rglob('*.py')
        if p.stem == 'square_notrans')), True)
    assert 'uint32_t pipeStage0' in src
    assert kernel_eval.evaluate(src, tid=0, seed=17, globals_only=True)


@pytest.mark.parametrize('name', ['square_notrans', 'rectangular',
                                  'accumulate_chain', 'chain'])
def test_prefetch_does_not_change_the_numbers(name):
    both = _compare(name)
    if both is None:
        pytest.skip('prefetch changed nothing in this kernel')
    plain, wrapped = both
    for key in sorted(set(plain) | set(wrapped)):
        assert plain.get(key, 0.0) == pytest.approx(wrapped.get(key, 0.0),
                                                    abs=1e-4), key


@pytest.mark.xfail(strict=True, reason=(
    'prefetch turns s1 into a rotating buffer -- the compute reads '
    '`(pipeStage0 % 2) * 320` and the transfer fills `((pipeStage0 + 1) % 2) '
    '* 320` -- but nothing fills the stage the first iteration reads. There '
    'is no prologue transfer before the batch loop, so iteration 0 computes '
    'from whatever the shared arena held.'))
def test_the_known_prefetch_divergence():
    plain, wrapped = _compare('trans_a')
    for key in sorted(set(plain) | set(wrapped)):
        assert plain.get(key, 0.0) == pytest.approx(wrapped.get(key, 0.0),
                                                    abs=1e-4), key
