# SPDX-License-Identifier: MIT
"""End-to-end runner: one case + one target -> pass/fail."""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from tensorforge.common.basic_types import Addressing, DataFlowDirection
from tensorforge.common.context import Context
from tensorforge.generators.generator import Generator

from . import driver_emit, layout
from .reference import multilinear_reference
from .toolchain import BuildInputs, Target, build


@dataclass
class RunResult:
    passed: bool
    max_abs_err: float
    max_rel_err: float
    detail: str           # path to artifacts on failure, empty on success


# ----------------------------------------------------------------------
# Reference dispatch
# ----------------------------------------------------------------------

def _reference_for_case(case, inputs: Dict[str, np.ndarray],
                       dest_in: np.ndarray) -> np.ndarray:
    """If the case provides its own ``reference()``, use it; else derive from descr."""
    if hasattr(case, "reference"):
        return case.reference(inputs, dest_in)

    # Auto-reference via the first descr's Multilinear structure.
    from tensorforge.generators.descriptions import MultilinearDescr
    descr_list = case.descr_list()
    if len(descr_list) != 1 or not isinstance(descr_list[0], MultilinearDescr):
        raise RuntimeError(
            "auto-reference only supports a single MultilinearDescr; "
            "give the case an explicit reference(inputs, dest_in) function"
        )
    d = descr_list[0]
    # Pull operand arrays by alias. Synthetic scalar operands (e.g. the
    # alpha tensor that GemmDescr injects for alpha != 1) carry their
    # value in op.tensor.data, not in the user-supplied inputs dict.
    operand_arrays = []
    for op in d.ops:
        if op.tensor.addressing == Addressing.SCALAR and op.tensor.has_values():
            val = op.tensor.get_values()[0]
            operand_arrays.append(np.array(val, dtype=layout.np_dtype(dt)))
            continue
        key = op.tensor.alias
        if key is None or key not in inputs:
            raise RuntimeError(
                f"auto-reference cannot find input for operand alias={op.tensor.alias!r}; "
                "either set alias= on each Tensor or provide reference() yourself"
            )
        operand_arrays.append(inputs[key])
    return multilinear_reference(
        target=d.target, permute=d.permute, add=d.add,
        operands=operand_arrays, dest_in=dest_in,
    )


# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------

def run_case(case, target: Target, cache_root: Path,
             tensorforge_include: Path,
             runtime_timeout: float = 60.0) -> RunResult:
    """Generate, build (cached), execute, and compare.

    ``case`` is a loaded module object with the contract described in
    ``tests_new/cases/README.md`` (``NAME``, ``descr_list()``, ``DTYPE``,
    ``BATCH``, optional ``reference()`` and ``TOL``).
    """
    dt = case.DTYPE
    batch = int(getattr(case, "BATCH", 8))
    rtol, atol = getattr(case, "TOL", (1e-5, 1e-5))

    descr_list = case.descr_list()
    ctx = Context(arch=target.arch, backend=target.backend, fp_type=dt)
    gen = Generator(descr_list, ctx)
    gen.generate()

    # Build or fetch cached executable.
    driver_src = driver_emit.emit(gen, target.backend, default_batch=batch)

    includes_src = "\n".join(f'#include "{header}"' for header in ctx.get_vm().get_headers())
    includes_src += "\n".join(f'#include "{header}"' for header in gen.get_helper_headers())

    bi = BuildInputs(
        workdir=cache_root,
        includes_src=includes_src,
        kernel_src=gen.get_kernel(),
        header_src=gen.get_header(),
        launcher_src=gen.get_launcher(),
        driver_src=driver_src,
        target=target,
        tensorforge_include=tensorforge_include,
    )
    exe = build(bi, cache_root)

    # Prepare inputs and expected output.
    ops_meta = driver_emit.collect_operands(gen)
    rng = np.random.default_rng(abs(hash((case.NAME, target.arch))) & 0xFFFFFFFF)

    inputs_by_alias: Dict[str, np.ndarray] = {}
    flats: Dict[str, np.ndarray] = {}          # kernel_name -> flat buffer
    dest_alias = None
    dest_in_view = None
    for op in ops_meta:
        key = op.alias or op.kernel_name
        if op.is_sink and not op.is_source:
            # pure output: init to zero so reference can read a defined C_in
            view, flat = layout.zeros_batch(op.shape, batch, dt)
            dest_alias = key
            dest_in_view = np.array(view, copy=True)       # snapshot pre-kernel
        elif op.is_source and not op.is_sink:
            view, flat = layout.make_batch(rng, op.shape, batch, dt)
            inputs_by_alias[key] = np.array(view, copy=True)
        else:
            # SOURCESINK: acts as both input and initial accumulator (beta!=0).
            view, flat = layout.make_batch(rng, op.shape, batch, dt)
            inputs_by_alias[key] = np.array(view, copy=True)
            dest_alias = key
            dest_in_view = np.array(view, copy=True)
        flats[op.kernel_name] = flat

    if dest_in_view is None:
        return RunResult(False, float("inf"), float("inf"),
                         "case has no sink operand — nothing to compare")

    expected = _reference_for_case(case, inputs_by_alias, dest_in_view)

    # Lay out a per-run work directory (separate from compile cache).
    with tempfile.TemporaryDirectory(prefix=f"tf_run_{case.NAME}_") as tmp:
        tmp = Path(tmp)
        in_dir = tmp / "in"; in_dir.mkdir()
        out_dir = tmp / "out"; out_dir.mkdir()
        for op in ops_meta:
            if op.is_source:
                (in_dir / f"in_{op.kernel_name}.bin").write_bytes(
                    flats[op.kernel_name].tobytes()
                )
            # SINK that is also SOURCE (beta != 0): the kernel needs the
            # initial C on device, so dump it under the input name too.
            # Current GemmDescr with beta != 0 sets SOURCESINK direction,
            # which already takes the is_source branch above.

        env = os.environ.copy()
        # Pin visible device so the index we pass actually lines up.
        if target.vendor == "nvidia":
            env["CUDA_VISIBLE_DEVICES"] = str(target.device_index)
        elif target.vendor == "amd":
            env["HIP_VISIBLE_DEVICES"] = str(target.device_index)

        try:
            proc = subprocess.run(
                [str(exe), str(in_dir), str(out_dir), str(batch), "0"],
                capture_output=True, text=True, timeout=runtime_timeout,
                check=False, env=env,
            )
        except subprocess.TimeoutExpired:
            return RunResult(False, float("nan"), float("nan"),
                             f"timeout after {runtime_timeout}s")

        if proc.returncode != 0:
            detail_dir = cache_root / "_failures" / f"{case.NAME}-{target.id}"
            detail_dir.mkdir(parents=True, exist_ok=True)
            (detail_dir / "stderr.txt").write_text(proc.stderr)
            (detail_dir / "stdout.txt").write_text(proc.stdout)
            return RunResult(False, float("nan"), float("nan"),
                             f"driver exit {proc.returncode}; see {detail_dir}")

        # Read back the sink's buffer.
        sink_op = next(o for o in ops_meta if o.is_sink)
        got_bytes = (out_dir / f"out_{sink_op.kernel_name}.bin").read_bytes()
        got_flat = np.frombuffer(got_bytes, dtype=layout.np_dtype(dt))
        got_view = layout.view_of(got_flat, sink_op.shape, batch)

        abs_err = np.abs(np.asarray(got_view) - np.asarray(expected))
        max_abs = float(abs_err.max())
        denom = np.maximum(np.abs(np.asarray(expected)), 1e-30)
        max_rel = float((abs_err / denom).max())

        ok = np.allclose(np.asarray(got_view), np.asarray(expected),
                         rtol=rtol, atol=atol)

        if not ok:
            detail_dir = cache_root / "_failures" / f"{case.NAME}-{target.id}"
            detail_dir.mkdir(parents=True, exist_ok=True)
            np.save(detail_dir / "expected.npy", np.asarray(expected))
            np.save(detail_dir / "got.npy", np.asarray(got_view))
            (detail_dir / "kernel.cpp").write_text(gen.get_kernel())
            return RunResult(False, max_abs, max_rel, f"mismatch; see {detail_dir}")

        return RunResult(True, max_abs, max_rel, "")
