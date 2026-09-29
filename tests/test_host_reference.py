# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The host oracle states yateto's `=`, not "write where it was computed".

`tools/host/reference.py` assigned an operation's result over the box it
computed and kept whatever the destination held elsewhere.  That is exactly
what a kernel does that forgets the zeros an assignment owes, so the oracle
agreed with it: SeisSol's free-surface-gravity kernel re-assigned a temporary
from a product one row supports, kept the previous step's other rows, and the
host check passed where the GPU unit test failed.

Three rules, the ones `MultilinearBuilder._promised_box` makes the stores
keep: an assignment onto the tensor itself defines the whole tensor (zeros
where nothing was computed), one onto a slice defines that slice only, and an
accumulation defines nothing.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
HOST = ROOT / "tools" / "host"
# `tfpaths` finds `kernel_eval` through the package; say where it is instead
os.environ.setdefault("TF_TESTS", str(ROOT / "tests"))
if str(HOST) not in sys.path:
    sys.path.insert(0, str(HOST))


def _load(name):
    spec = importlib.util.spec_from_file_location(f"host_{name}",
                                                  HOST / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


reference = _load("reference")


def _view(name, bbox, tbbox=((0,), (4,)), offset=(0,), sliced=False,
          is_tmp=False, shape=(4,)):
    return dict(name=name, alias=name, shape=list(shape),
                ashape=[h - l for l, h in zip(*tbbox)],
                tbbox=[list(tbbox[0]), list(tbbox[1])],
                bbox=[list(bbox[0]), list(bbox[1])], offset=list(offset),
                addressing="strided", is_tmp=is_tmp, storage=4,
                sliced=sliced, pack=None, data=None)


def _copy(dest, src, add=False):
    """`dest[i] (+)= src[i]`, as a capture writes it."""
    return dict(kind="multilinear", dest=dest, ops=[src], target=[[0]],
                permute=[[0]], add=add)


def _run(row, before):
    arrays = {"M": np.array(before, dtype=float),
              "b": np.array([5.0, 6.0, 7.0, 8.0])}
    reference.apply(row, arrays)
    return arrays["M"]


# `b` supports cell 0 only, so every row below computes M over [0, 1)
NARROW = _view("b", ((0,), (1,)), tbbox=((0,), (1,)))


def test_an_assignment_to_the_tensor_defines_all_of_it():
    row = _copy(_view("M", ((0,), (4,))), NARROW)
    assert _run(row, [9, 9, 9, 9]).tolist() == [5, 0, 0, 0]


def test_an_assignment_to_a_slice_defines_the_slice_only():
    # M(2:4) = b: view cell 0 is M[2], view cell 1 M[3], nothing else is ours
    row = _copy(_view("M", ((0,), (2,)), offset=(2,), sliced=True), NARROW)
    assert _run(row, [9, 9, 9, 9]).tolist() == [9, 9, 5, 0]


def test_a_slice_at_offset_zero_is_still_a_slice():
    row = _copy(_view("M", ((0,), (2,)), sliced=True), NARROW)
    assert _run(row, [9, 9, 9, 9]).tolist() == [5, 0, 9, 9]


def test_an_accumulation_defines_nothing():
    row = _copy(_view("M", ((0,), (4,))), NARROW, add=True)
    assert _run(row, [9, 9, 9, 9]).tolist() == [14, 9, 9, 9]


def test_a_seed_lands_where_the_tensor_is_stored():
    rows = [_copy(_view("M", ((1,), (3,)), tbbox=((1,), (3,))), NARROW),
            _copy(_view("T", ((0,), (4,)), is_tmp=True), NARROW)]
    shapes = {"M": (4,), "T": (4,), "b": (4,)}
    storage = reference.storage_of(rows)
    arrays = {n: np.zeros(s) for n, s in shapes.items()}
    seeded = reference.seed_destinations(rows, arrays, shapes, storage, 3)
    assert seeded == {"M"}                  # a temporary has no entry value
    assert arrays["M"][0] == arrays["M"][3] == 0
    assert np.all(arrays["M"][1:3] != 0)


def _reassigned(read_between):
    """`M = b0; [out1 = M;] M = b1; out = M`, `M` a temporary and `b1` stored
    over one cell: yateto's `out` is `[b1[0], 0, 0, 0]`."""
    b1 = _view("b1", ((0,), (1,)), tbbox=((0,), (1,)))
    b0 = _view("b0", ((0,), (4,)))
    m = _view("M", ((0,), (4,)), is_tmp=True)
    rows = [_copy(m, b0)]
    if read_between:
        rows.append(_copy(_view("out1", ((0,), (4,))), m))
    return rows + [_copy(m, b1), _copy(_view("out", ((0,), (4,))), m)]


@pytest.mark.parametrize("read_between", [False, True])
def test_a_temporary_reassigned_from_a_narrower_product(read_between):
    # generated, run on every lane, and held to the reference: the kernel
    # that kept M's earlier cells agreed with the reference that kept them
    bisect = _load("prefix_bisect")
    rows = _reassigned(read_between)
    assert bisect.evaluate(rows, len(rows)) < 1e-6


def test_the_lane_count_is_the_kernels_own():
    lockstep = _load("lockstep")
    src = ('    // tensorforge-meta: {"launch":{"threads_per_mult":4},'
           '"version":"0.0.1\\n"}\n')
    assert lockstep.lanes_of(src) == 4
    assert lockstep.lanes_of("no meta line") == lockstep.THREADS
