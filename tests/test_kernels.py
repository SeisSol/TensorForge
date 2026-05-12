# SPDX-License-Identifier: MIT
"""Per-case × per-target end-to-end numerical tests.

The actual GPU work happens in :func:`harness.runner.run_case`; this
module is just glue. Two host-only tests at the bottom keep the
non-GPU pieces (layout + reference) honest.
"""

from __future__ import annotations

import numpy as np

from harness.layout import make_batch, view_of, zeros_batch, np_dtype
from harness.reference import multilinear_reference
from harness.runner import run_case
from tensorforge.common.basic_types import Datatype


# ----------------------------------------------------------------------
# End-to-end (skipped automatically when no target is available)
# ----------------------------------------------------------------------

def test_kernel(case, target, cache_root, tensorforge_include):
    """Generate, build, run, and numerically validate one case on one target."""
    result = run_case(case, target, cache_root, tensorforge_include)
    assert result.passed, (
        f"{case.NAME} on {target.id} failed: "
        f"max_abs_err={result.max_abs_err:.3g} "
        f"max_rel_err={result.max_rel_err:.3g}\n{result.detail}"
    )


# ----------------------------------------------------------------------
# Host-only smoke tests; always run, no GPU required
# ----------------------------------------------------------------------

def test_layout_roundtrip():
    """Batch layout: per-element F-order over a batch-major flat buffer."""
    rng = np.random.default_rng(0)
    view, flat = make_batch(rng, (3, 4), batch=2, dt=Datatype.F32)
    # The kernel adresses [i, j] in element b at flat[b*12 + i + j*3]:
    assert view[0, 1, 0] == flat[1]
    assert view[0, 0, 1] == flat[3]
    assert view[1, 0, 0] == flat[12]

    # Round-trip via tobytes: writing to flat must be visible in view.
    flat[5] = 99.0
    assert view[0, 2, 1] == 99.0

    # zeros_batch yields a defined initial sink buffer.
    zv, zf = zeros_batch((3, 4), 2, Datatype.F64)
    assert zf.dtype == np.float64
    assert (zf == 0).all()


def test_reference_matches_einsum():
    """Auto-reference reproduces einsum for plain GEMM, alpha-scaled, and trans-A."""
    rng = np.random.default_rng(1)
    A = rng.standard_normal((2, 4, 5)).astype(np.float32)
    B = rng.standard_normal((2, 5, 6)).astype(np.float32)
    C0 = np.zeros((2, 4, 6), dtype=np.float32)

    plain = multilinear_reference(
        target=[[0, -1], [-1, 1]],
        permute=[[0, 1], [0, 1]],
        add=False,
        operands=[A, B],
        dest_in=C0,
    )
    assert np.allclose(plain, np.einsum("bik,bkj->bij", A, B))

    alpha = np.float32(1.5)
    scaled = multilinear_reference(
        target=[[0, -1], [-1, 1], []],
        permute=[[0, 1], [0, 1], []],
        add=False,
        operands=[A, B, np.array(alpha)],
        dest_in=C0,
    )
    assert np.allclose(scaled, 1.5 * np.einsum("bik,bkj->bij", A, B))

    A_raw = rng.standard_normal((2, 5, 4)).astype(np.float32)
    trans = multilinear_reference(
        target=[[0, -1], [-1, 1]],
        permute=[[1, 0], [0, 1]],
        add=False,
        operands=[A_raw, B],
        dest_in=C0,
    )
    assert np.allclose(trans, np.einsum("bki,bkj->bij", A_raw, B))
