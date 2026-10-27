# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The host oracle folds over the lanes as the device does.

`tensorforge::reduction<ReductionOperation<T, Op>, Block, Subblock, T>(v)`
is an all-reduce: a butterfly over every `Subblock`-th lane of a run of
`Block` lanes, after which each of them holds the fold
(`include/tensorforge_device/cuda.h`).  A reduction over the lane axis
spells its fold that way, and so does a contraction to one value of a
register image.  The oracle runs it the same way, distance `Block / 2`
first, so that a sum comes out of it in the order the device adds in; the
lanes past a ragged extent fold the neutral element, `-INFINITY` for a
`max`.  Each of the cases below folds, and on the oracle computes what its
NumPy reference does.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import warnings
from pathlib import Path

import numpy as np
import pytest

from tensorforge.common.context import Context
from tensorforge.common.options import Options
from tensorforge.generators.generator import Generator
from tensorforge.reference import kernel_eval as ke

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / 'tests' / 'cases'
SEED = 11


def _host():
    path = ROOT / 'tools' / 'host' / 'corpus_check.py'
    spec = importlib.util.spec_from_file_location('fold_corpus_check', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HOST = _host()


def _case(stem: str):
    path = next(CASES.rglob(f'{stem}.py'))
    spec = importlib.util.spec_from_file_location('tf_fold__' + stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tensors(descrs):
    out = {}
    for d in descrs:
        for view in [d.writes()] + list(d.reads()):
            t = getattr(view, 'tensor', None)
            if t is not None and t.alias is not None:
                out.setdefault(t.alias, t)
    return out


@pytest.mark.parametrize('stem', [
    'red_scalar_then_ew', 'red_lead_then_ml', 'max_all', 'sum_axis0',
    'sum_axis0_40'])
def test_a_fold_over_the_lanes_computes_on_the_oracle(stem):
    case = _case(stem)
    descrs = case.descr_list()
    gen = Generator(descrs, Context(arch='sm_86', backend='cuda',
                                    fp_type=case.DTYPE,
                                    options=Options(**getattr(case, 'OPTIONS',
                                                              {}))))
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        gen.generate()
    src = gen.get_kernel()
    assert 'tensorforge::reduction' in src
    lanes, mults = ke.launch_geometry(gen.get_launcher())
    mem = ke.evaluate_wave(src, lanes, seed=SEED, globals_only=True,
                           mults=mults, block=True)

    fill = ke.Slot(SEED)
    tensors = _tensors(descrs)
    sinks = [d.writes().tensor.alias for d in descrs
             if not d.writes().tensor.is_tmp]
    out = getattr(case, 'OUTPUT', None) or sinks[-1]
    arrays = {}
    for alias, t in tensors.items():
        if t.is_tmp:
            continue
        arr = np.zeros((1,) + tuple(int(s) for s in t.shape))
        for slot, cell in HOST._slots(t).items():
            arr[(0,) + tuple(cell)] = fill.read(t.name, slot)
        arrays[alias] = arr
    want = np.asarray(case.reference(arrays, arrays[out].copy()))
    t = tensors[out]
    rtol, atol = getattr(case, 'TOL', (1e-5, 1e-5))
    for slot, cell in HOST._slots(t).items():
        got = mem.get((t.name, slot), fill.read(t.name, slot))
        expected = want.reshape((1,) + tuple(int(s) for s in t.shape))[
            (0,) + tuple(cell)]
        assert got == pytest.approx(expected, rel=rtol, abs=atol), (slot, cell)


def test_a_fold_reaches_every_lane_of_its_run():
    """Every lane of the run holds the fold, not the first alone: a
    reduction into a temporary without axes, which every lane reads
    (`red_scalar_then_ew`), reads one value in all of them."""
    case = _case('red_scalar_then_ew')
    gen = Generator(case.descr_list(), Context(arch='sm_86', backend='cuda',
                                               fp_type=case.DTYPE,
                                               options=Options()))
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        gen.generate()
    lanes, mults = ke.launch_geometry(gen.get_launcher())
    seen = []
    real = ke.Interp._reduce

    def spy(self, op, block, subblock, key):
        value = real(self, op, block, subblock, key)
        seen.append((self.env['threadIdx'].x, op, int(block), value))
        return value
    ke.Interp._reduce = spy
    try:
        ke.evaluate_wave(gen.get_kernel(), lanes, seed=SEED, globals_only=True,
                         mults=mults, block=True)
    finally:
        ke.Interp._reduce = real
    assert seen
    for op, block in {(op, block) for _, op, block, _ in seen}:
        values = [v for _, o, b, v in seen if (o, b) == (op, block)]
        lanes_seen = {x for x, o, b, _ in seen if (o, b) == (op, block)}
        assert lanes_seen == set(range(block))
        assert len(set(values)) == 1


def test_a_fold_combines_the_farthest_lane_first():
    """`Block = 4`: lane 0 adds lane 2 first, then lane 1's partial -- the
    order a floating-point sum depends on."""
    lane_values = [1e16, 1.0, -1e16, 1.0]
    lanes = []
    run = type('Run', (), {})()
    run.wave_cache = {}
    for x, v in enumerate(lane_values):
        it = ke.Interp(ke.Slot(), {'v': v})
        it.env['threadIdx'] = type('Idx', (), {'x': x, 'y': 0, 'z': 0})()
        it.lockstep = run
        lanes.append(it)
    run.current = lanes
    got = [it.ev('tensorforge::reduction<tensorforge::ReductionOperation<'
                 'double, tensorforge::Operation::Add>, 4, 1, double>(v)')
           for it in lanes]
    # (1e16 + -1e16) + (1.0 + 1.0) where the butterfly pairs 0 with 2 and
    # 1 with 3 first; left to right the ones would be lost.
    assert got == [2.0] * 4
