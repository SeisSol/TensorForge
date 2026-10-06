# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What a peeled iteration is allowed to dereference.

Peeling runs a copy of the loop body *before* the loop, so it runs before the
size guard.  Every thread executes it, including the ones whose element index
is past the end --- and those are not rare.  The grid is sized in blocks:
``min(occupancy, numElements)`` blocks of ``blockDim.y`` rows, so at 100
elements and a 16-row block the starts run to 1599 and fifteen out of sixteen
threads are out of range before the loop has done anything.

Inside the loop this is invisible, because the size guard catches them.  A peel
has no guard, so the index it names has to be one that is safe to dereference
unconditionally.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

from tensorforge.common.context import Context, Options
from tensorforge.generators.generator import Generator

CASES = Path(__file__).parent / "cases"

# every case in the corpus whose body has more than one compute slot, so
# there is something for WrapLoads to peel
PEELING_CASES = ["accumulate_chain.py", "accumulate_then_read.py",
                 "chain_five.py", "narrow_write_after_wide_read.py"]


def _kernel(case_file: str, backend: str, arch: str, **opt_kwargs) -> str:
    path = CASES / case_file
    spec = importlib.util.spec_from_file_location("tf_peel__" + path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ctx = Context(arch=arch, backend=backend,
                  fp_type=getattr(mod, "DTYPE", None),
                  options=Options(**opt_kwargs))
    gen = Generator(mod.descr_list(), ctx)
    gen.generate()
    return gen.get_kernel()


def _before_the_loop(kernel: str) -> list:
    lines = kernel.splitlines()
    stop = next((i for i, l in enumerate(lines)
                 if re.search(r"for \(size_t v\d+_batchId0\b", l)), None)
    assert stop is not None, "no batch loop found"
    return lines[:stop]


def _peel_subscripts(head: list) -> list:
    """The element index each peeled pointer is computed from."""
    return [m.group(1) for l in head
            for m in [re.search(r"peel_glb_\w+ = (?:\([^)]*\))?&\w+\[([^\]]+)\]", l)]
            if m]


@pytest.mark.parametrize("case_file", PEELING_CASES)
@pytest.mark.parametrize("backend,arch", [("hip", "gfx90a"), ("cuda", "sm_86")])
@pytest.mark.parametrize("distance", [1, 2, 4])
def test_peeled_load_does_not_dereference_the_raw_block_id(case_file, backend,
                                                           arch, distance):
    kernel = _kernel(case_file, backend, arch,
                     enable_wrap_loads=True, move_distance=distance)
    head = _before_the_loop(kernel)

    # The start is the launch geometry's spelling, bound where it is read; a
    # subscript built from it would spell the geometry itself.
    offenders = [index for index in _peel_subscripts(head)
                 if re.search(r"threadIdx|blockIdx|get_local_id|get_group_id",
                              index)]
    assert not offenders, (
        "peeled iteration indexes with the unclamped block id:\n  "
        + "\n  ".join(offenders))


@pytest.mark.parametrize("case_file", PEELING_CASES)
def test_peeled_load_reads_a_clamped_element(case_file):
    """The peel should still read *something*, and it should be batchId1."""
    kernel = _kernel(case_file, "hip", "gfx90a",
                     enable_wrap_loads=True, move_distance=2)
    head = _before_the_loop(kernel)
    subscripts = _peel_subscripts(head)
    assert subscripts, "expected WrapLoads to peel at least one transfer"
    named = {v for index in subscripts
             for v in re.findall(r"\bv\d+_batchId1\b", index)}
    assert named, ("no peeled pointer uses the clamped index:\n  "
                   + "\n  ".join(subscripts))
    # ...and the value is the clamp: the start where it is inside the batch,
    # element 0 where it is not.
    for value in named:
        definition = next(l for l in head if re.search(rf"\b{value} = ", l))
        assert "< numElements0" in definition and re.search(r": 0\w*;", definition), \
            definition
