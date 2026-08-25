# SPDX-License-Identifier: MIT
"""Two defects in `backend.opt.pipeline`, pinned rather than described.

Both are dormant: `enable_pipeline` and `enable_multibuffer` default to False,
so nothing in the corpus generates them today.  They matter because the
wrap-around schedule builds on exactly this machinery -- the peeled prologue,
the rolling pointer, the loop's lookahead bindings -- and inheriting a
prefetch that is skipped for masked elements, or a stage index that does not
alternate, would surface as wrong values in the new pass rather than the old.

Both are marked `xfail(strict=True)`, so the day the fix lands these turn into
XPASS failures asking for the marker to be removed.  That is the point: the
tests should stop being expected failures at the same commit that fixes them.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tensorforge.common.context import Context, Options
from tensorforge.generators.generator import Generator

CASES = Path(__file__).parent / "cases"


def _kernel(case_file: str, **opt_kwargs) -> str:
    path = CASES / case_file
    spec = importlib.util.spec_from_file_location("tf_defect__" + path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ctx = Context(arch="sm_86", backend="cuda",
                  fp_type=getattr(mod, "DTYPE", None),
                  options=Options(**opt_kwargs))
    gen = Generator(mod.descr_list(), ctx)
    gen.generate()
    return gen.get_kernel()


def _loop_body(kernel: str) -> list:
    """Lines from the batch loop header to the end, with indentation kept."""
    lines = kernel.splitlines()
    start = next(i for i, l in enumerate(lines) if "for (size_t batchId0" in l)
    return lines[start:]

def test_stage_index_is_not_the_element_id():
    kernel = _kernel("chain_five.py", enable_pipeline=True,
                     enable_multibuffer=True, pipeline_depth=2)
    staged = [l for l in _loop_body(kernel) if "% 2" in l]
    assert staged, "expected the rotated buffers to appear in the loop body"
    offenders = [l.strip() for l in staged if "batchId0 %" in l
                 or "(batchId0 + 1) %" in l]
    assert not offenders, (
        "stage index derived from the element id:\n  "
        + "\n  ".join(offenders))

def test_rolling_pointer_advance_is_not_under_the_flag_guard():
    kernel = _kernel("chain_five.py", enable_pipeline=True,
                     enable_multibuffer=False, pipeline_depth=2)
    body = _loop_body(kernel)
    guard = next(i for i, l in enumerate(body) if l.strip() == "if (allowed) {")
    guard_indent = len(body[guard]) - len(body[guard].lstrip())

    inside = []
    for line in body[guard + 1:]:
        stripped = line.strip()
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip())
        if indent <= guard_indent and stripped.startswith("}"):
            break
        if stripped.startswith("pipe_") and "=" in stripped:
            inside.append(stripped)

    assert not inside, (
        "prefetch pointer advanced under the per-element guard:\n  "
        + "\n  ".join(inside))
