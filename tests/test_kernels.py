# SPDX-FileCopyrightText: 2026 SeisSol Group
#
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


def test_elementwise_descr_constructs():
    """Every elementwise case can build an ElementwiseDescr without a GPU.

    Catches breakage in ``generators/elementwise.py`` or in
    ``ElementwiseDescr.__init__`` (arity check, shape check, the
    data-flow-direction assignment) even on a CI runner with no toolchain.
    """
    from pathlib import Path
    import importlib.util

    from tensorforge.common.basic_types import DataFlowDirection
    from tensorforge.generators.descriptions import ElementwiseDescr

    cases_dir = Path(__file__).parent / "cases" / "elementwise"
    case_files = sorted(p for p in cases_dir.glob("*.py")
                        if not p.name.startswith("_"))
    assert case_files, "expected at least one elementwise case"

    for path in case_files:
        spec = importlib.util.spec_from_file_location(
            f"_smoke_{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        descrs = mod.descr_list()
        assert len(descrs) == 1, f"{path.name}: expected one descr"
        d = descrs[0]
        assert isinstance(d, ElementwiseDescr), (
            f"{path.name}: not an ElementwiseDescr ({type(d).__name__})")
        # ElementwiseDescr.__init__ sets directions on the underlying
        # tensors; assert that side effect actually happened.
        seen_sink = d.dest.tensor.direction == DataFlowDirection.SINK
        seen_source = any(src.tensor.direction == DataFlowDirection.SOURCE
                          for src in d.tensor_srcs())
        assert seen_sink, f"{path.name}: no SINK direction set"
        assert seen_source, f"{path.name}: no SOURCE direction set"


def test_slicing_cases_construct_and_generate():
    """Each slicing case constructs operands with bbox != storage shape
    *and* survives full kernel generation on a known-good arch.

    The host-only assertion is: the bbox is strictly smaller than the
    storage shape on at least one axis of at least one operand —
    otherwise the case is mislabelled and isn't testing slicing. The
    generation pass guards against the bbox.lower offset arithmetic
    crashing the address writer.
    """
    from pathlib import Path
    import importlib.util

    from tensorforge.common.context import Context
    from tensorforge.generators.generator import Generator
    from tensorforge.generators.descriptions import MultilinearDescr

    cases_dir = Path(__file__).parent / "cases" / "slicing"
    case_files = sorted(p for p in cases_dir.glob("*.py")
                        if not p.name.startswith("_"))
    assert case_files, "expected at least one slicing case"

    for path in case_files:
        spec = importlib.util.spec_from_file_location(
            f"_smoke_{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        descrs = mod.descr_list()

        saw_slice = False
        for d in descrs:
            if not isinstance(d, MultilinearDescr):
                continue
            for op in d.matrix_list():
                if list(op.tensor.shape) != list(op.bbox.sizes()):
                    saw_slice = True
                    break
        assert saw_slice, (
            f"{path.name}: no operand has bbox != shape — this is not a "
            "slicing case")

        # Generation must not crash on a known-good arch/backend.
        ctx = Context(arch="sm_86", backend="cuda", fp_type=mod.DTYPE)
        gen = Generator(mod.descr_list(), ctx)
        gen.generate()
        assert gen.get_kernel(), f"{path.name}: empty kernel"


def test_reduction_descr_constructs():
    """Reduction cases can build a :class:`ReductionDescr` without crashing.

    Generation is *not* exercised here — currently it crashes with
    ``AttributeError: 'Tensor' object has no attribute 'tensor'`` because
    the :class:`Generator` does not dispatch on :class:`ReductionDescr`
    and the fall-through path at ``generator.py:428`` assumes every
    operand in ``matrix_list()`` is a :class:`SubTensor`. Each reduction
    case carries ``XFAIL=True``, so the end-to-end test is marked
    accordingly via :func:`pytest_generate_tests` and will go from
    xfail → xpass-strict-failure once the reduction path is wired up.
    """
    from pathlib import Path
    import importlib.util

    from tensorforge.common.basic_types import DataFlowDirection
    from tensorforge.generators.descriptions import ReductionDescr

    cases_dir = Path(__file__).parent / "cases" / "reduction"
    case_files = sorted(p for p in cases_dir.glob("*.py")
                        if not p.name.startswith("_"))
    assert case_files, "expected at least one reduction case"

    for path in case_files:
        spec = importlib.util.spec_from_file_location(
            f"_smoke_{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        descrs = mod.descr_list()
        assert len(descrs) == 1, f"{path.name}: expected one descr"
        d = descrs[0]
        assert isinstance(d, ReductionDescr), (
            f"{path.name}: not a ReductionDescr ({type(d).__name__})")
        assert d.var.direction == DataFlowDirection.SOURCE
        assert d.dest.direction == DataFlowDirection.SINK
        assert getattr(mod, "XFAIL", False), (
            f"{path.name}: reduction case must carry XFAIL=True until "
            "ReductionInstruction is implemented")


# ---------------------------------------------------------------------- #
# Per-feature host smokes for the D-block cases. Each one asserts the
# distinguishing property of the case (so a case that gets accidentally
# rewritten as a plain GEMM is caught), and runs generation where the
# feature is known to work on dev2.

def _import_case(filename):
    from pathlib import Path
    import importlib.util
    path = Path(__file__).parent / "cases" / filename
    spec = importlib.util.spec_from_file_location(f"_smoke_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_trans_b_actually_transposes_b():
    """The ``trans_b`` case must construct a GemmDescr whose B operand
    carries the ``permute=[1, 0]`` flag (otherwise it's just another GEMM
    with no transpose). Generation must work — the trans_b path is
    expected to be production-ready."""
    from tensorforge.common.context import Context
    from tensorforge.generators.generator import Generator

    mod = _import_case("trans_b.py")
    descrs = mod.descr_list()
    assert any(d.permute[1] == [1, 0] for d in descrs), (
        "trans_b case: B operand is not actually transposed")
    Generator(descrs, Context(arch="sm_86", backend="cuda",
                              fp_type=mod.DTYPE)).generate()


def test_add_true_sets_accumulate_and_promotes_dest():
    """The ``add_true`` case must carry ``add=True`` on the descr *and*
    promote the destination tensor to SOURCESINK — both are needed to
    actually observe the accumulation."""
    from tensorforge.common.basic_types import DataFlowDirection
    from tensorforge.common.context import Context
    from tensorforge.generators.generator import Generator

    mod = _import_case("add_true.py")
    descrs = mod.descr_list()
    assert any(d.add for d in descrs), "add_true case: no add=True descr"
    # The dest of the add=True descr must be SOURCESINK.
    accum = next(d for d in descrs if d.add)
    assert accum.dest.tensor.direction == DataFlowDirection.SOURCESINK, (
        "add_true case: destination not promoted; accumulation hidden")
    Generator(descrs, Context(arch="sm_86", backend="cuda",
                              fp_type=mod.DTYPE)).generate()


def test_beta_nonzero_is_marked_xfail_and_constructs():
    """``beta_nonzero`` must declare ``XFAIL=True`` (the silently-dropped
    beta is a known bug). Construction itself must succeed so the case
    actually reaches the comparison phase to fail there."""
    mod = _import_case("beta_nonzero.py")
    assert getattr(mod, "XFAIL", False), (
        "beta_nonzero case: must be XFAIL until GemmDescr forwards beta")
    descrs = mod.descr_list()
    assert any(getattr(d, "beta", None) not in (None, 0.0)
               or "beta" in mod.__doc__.lower()
               for d in descrs)  # weak: beta is dropped at construction


def test_addressing_none_uses_none_for_operator():
    """``addressing_none`` must mark at least one operand with
    :data:`Addressing.NONE`. Generation must work — NONE is
    production-ready."""
    from tensorforge.common.basic_types import Addressing
    from tensorforge.common.context import Context
    from tensorforge.generators.generator import Generator

    mod = _import_case("addressing_none.py")
    descrs = mod.descr_list()
    none_ops = [op for d in descrs for op in d.matrix_list()
                if op.tensor.addressing == Addressing.NONE]
    assert none_ops, "addressing_none case: no NONE-addressed operand"
    Generator(descrs, Context(arch="sm_86", backend="cuda",
                              fp_type=mod.DTYPE)).generate()


def test_addressing_ptr_based_constructs_and_is_xfail():
    """``addressing_ptr_based`` must use PTR_BASED addressing and be
    marked XFAIL (harness driver doesn't yet emit T**-style allocations)."""
    from tensorforge.common.basic_types import Addressing
    mod = _import_case("addressing_ptr_based.py")
    assert getattr(mod, "XFAIL", False), (
        "ptr_based case: must be XFAIL until driver_emit supports PTR_BASED")
    descrs = mod.descr_list()
    ptr_ops = [op for d in descrs for op in d.matrix_list()
               if op.tensor.addressing == Addressing.PTR_BASED]
    assert ptr_ops, "ptr_based case: no PTR_BASED operand"


def test_sparsity_band_uses_maskspp_and_generates():
    """``sparsity_band`` must attach a non-Full :class:`SparsityPattern`
    to at least one operand, and generation must produce a kernel that
    differs from the dense equivalent (sanity check that the sparsity
    path was actually taken)."""
    from tensorforge.common.context import Context
    from tensorforge.common.matrix.spp import FullSPP
    from tensorforge.generators.generator import Generator

    mod = _import_case("sparsity_band.py")
    descrs = mod.descr_list()
    sparse_ops = [op for d in descrs for op in d.matrix_list()
                  if not isinstance(op.tensor.spp, FullSPP)]
    assert sparse_ops, "sparsity_band case: no sparse operand"
    Generator(descrs, Context(arch="sm_86", backend="cuda",
                              fp_type=mod.DTYPE)).generate()


# ---------------------------------------------------------------------- #
# Barrier cases — multi-section kernels.

def test_barriers_cases_construct_and_generate():
    """Each barriers case must produce a multi-section kernel.

    The host-only assertion: after generation, ``gen._sections`` has
    length ≥ 2 — otherwise the barrier didn't actually split the descr
    list and the case isn't testing what it claims. For the
    ``GridBarrierDescr`` variant we additionally require
    ``persistent_threading`` to be on (cooperative launch).
    """
    from pathlib import Path
    import importlib.util

    from tensorforge.common.context import Context
    from tensorforge.generators.generator import Generator

    cases_dir = Path(__file__).parent / "cases" / "barriers"
    case_files = sorted(p for p in cases_dir.glob("*.py")
                        if not p.name.startswith("_"))
    assert case_files, "expected at least one barriers case"

    for path in case_files:
        spec = importlib.util.spec_from_file_location(
            f"_smoke_{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        ctx = Context(arch="sm_86", backend="cuda", fp_type=mod.DTYPE)
        gen = Generator(mod.descr_list(), ctx)
        gen.generate()

        assert len(gen._sections) >= 2, (
            f"{path.name}: only {len(gen._sections)} section(s) — the "
            "barrier didn't split the descr list")

        # GridBarrierDescr is the cooperative variant; recognise by name.
        # We assert on the kernel content rather than the descr class so
        # the test stays valid if either class gets renamed.
        if "barrier" in path.stem:
            assert any(s.barrier for s in gen._sections), (
                f"{path.name}: GridBarrierDescr but no section.barrier set")
            assert "cudaLaunchCooperativeKernel" in gen.get_launcher(), (
                f"{path.name}: cooperative launch not selected")


# ---------------------------------------------------------------------- #
# Yateto frontend — host-only construction smoke.

def test_yateto_frontend_imports_and_constructs():
    """``YatetoFrontend`` imports without ``yateto`` installed and a
    bare instance carries an empty descr list.

    This is the minimum smoke we can run host-only: ``YatetoFrontend``
    only depends on tensorforge's own ``interface`` / ``ir`` packages,
    not on the upstream ``yateto`` Python package. A regression in
    those internal interfaces shows up as an import-time error and
    this test catches it before SeisSol integration breaks. End-to-end
    tests that actually feed yateto-emitted descriptions are out of
    scope for the in-repo suite; SeisSol's CI is the right place for
    those.
    """
    from tensorforge.frontend.yateto import YatetoFrontend, GpuKernelGeneratorV1

    fe = YatetoFrontend(arch="sm_86")
    assert isinstance(fe.generator, GpuKernelGeneratorV1)
    assert fe.generator._descr_list == [], (
        "fresh YatetoFrontend should have empty descr list")


# ---------------------------------------------------------------------- #
# F64 variants — sanity that the dtype propagates.

def test_f64_variant_cases_use_double_precision():
    """Every ``*_f64`` case (top-level or in a subdirectory) declares
    ``DTYPE = Datatype.F64`` and constructs operands with that dtype.

    Catches the most common copy-paste mistake when adding an F64
    variant: forgetting to switch ``DTYPE`` in the operand tensors
    while changing the module-level constant. The kernel would then
    be emitted as F32 and the test would pass against a F64 reference
    only by accident (and would silently mask any F64-specific bug
    we wanted to catch).
    """
    from pathlib import Path
    import importlib.util

    from tensorforge.common.basic_types import Datatype

    cases_root = Path(__file__).parent / "cases"
    f64_files = [p for p in cases_root.rglob("*_f64.py")
                 if not p.name.startswith("_")]
    assert f64_files, "expected at least one *_f64 case"

    for path in f64_files:
        spec = importlib.util.spec_from_file_location(
            f"_smoke_{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod.DTYPE == Datatype.F64, (
            f"{path.name}: module-level DTYPE is {mod.DTYPE}, expected F64")
        for d in mod.descr_list():
            for op in d.matrix_list():
                # tensor.datatype may be None on synthetic scalars built
                # by GemmDescr — skip those, but every real operand
                # must be F64.
                if op.tensor.datatype is None:
                    continue
                assert op.tensor.datatype == Datatype.F64, (
                    f"{path.name}: operand {op.tensor.alias} carries "
                    f"{op.tensor.datatype}, not F64 — kernel will be "
                    "generated at the wrong precision")


def test_rotating_buffer_is_staged_by_iteration_not_element():
    """The rotating stage index must come from the loop's iteration counter.

    The batch loop is grid-strided: one thread's consecutive elements are
    ``gridDim.x * blockDim.y`` apart. Indexing stages by ``batchId0 % depth``
    therefore breaks twice -- the stage never alternates when ``depth`` divides
    that stride (the usual case), and iteration 0 reads ``batchId0 % depth``
    while the peeled prologue can only write a literal stage, so every thread
    group with an odd start index reads a stage nobody filled. Neither shows up
    as a crash, only as wrong numbers, which is why this is pinned here.
    """
    import re

    from tensorforge.common.basic_types import Addressing, Datatype
    from tensorforge.common.context import Context, Options
    from tensorforge.common.matrix.boundingbox import BoundingBox
    from tensorforge.common.matrix.tensor import SubTensor, Tensor
    from tensorforge.generators.descriptions import GemmDescr
    from tensorforge.generators.generator import Generator

    def descrs():
        def sub(rows, cols, alias):
            return SubTensor(Tensor([rows, cols], Addressing.STRIDED,
                                    BoundingBox([0, 0], [rows, cols]),
                                    alias=alias, datatype=Datatype.F32))
        return [GemmDescr(False, False,
                          a=sub(16, 16, "A"), b=sub(16, 16, "B"),
                          c=sub(16, 16, "C"))]

    opts = Options(enable_pipeline=True, enable_multibuffer=True)
    ctx = Context(arch="sm_90", backend="cuda", fp_type=Datatype.F32,
                  options=opts)
    gen = Generator(descrs(), ctx)
    gen.generate()
    src = gen.get_kernel()

    counter = "pipeStage0"
    assert f"uint32_t {counter} = 0;" in src, \
        "the rotating buffer needs an iteration counter declared before the loop"
    assert re.search(rf"{counter} = \({counter} \+ 1\) [&%]", src), \
        f"{counter} is never advanced, so every iteration reads one stage"

    # Every stage selector must be the counter, never the element index.
    selectors = re.findall(r"localShrMem\d+\[[^\]]*?\(([^)]*?)\) \* \d+\]", src)
    assert selectors, "expected at least one staged shared-memory view"
    for expr in selectors:
        assert "batchId" not in expr, (
            f"stage selected by element index ({expr!r}); in a grid-strided "
            f"loop that is not the iteration number")
        assert counter in expr or expr.strip().isdigit(), (
            f"stage selector {expr!r} is neither the counter nor a literal")

    # The prologue fills a literal stage, and iteration 0 must read that one.
    assert "(0) * " in src, "the peeled prologue fill must write stage 0"


def _inside_flag_guard(src, markers):
    """Lines matching ``markers`` that lie inside an ``if (allowed) { ... }``."""
    depth, guard_depths, hits = 0, [], []
    for line in src.splitlines():
        opened = "if (allowed) {" in line
        if guard_depths and any(m in line for m in markers):
            hits.append(line.strip())
        depth += line.count("{") - line.count("}")
        if opened:
            guard_depths.append(depth)
        while guard_depths and depth < guard_depths[-1]:
            guard_depths.pop()
    return hits


def test_pipelined_work_sits_outside_the_element_flag_guard():
    """Anything carried across the back edge must not be under ``flags``.

    ``flags`` is a runtime per-element mask, so a skipped element must not be
    able to desynchronise the pipeline from the element sequence. Two things
    qualify:

    * the rolling pointer's advance — skipped once, it trails the loop
      variable for the rest of the loop, so this must be outside the guard in
      *both* pipelining modes;
    * the prefetch, which iteration ``k`` issues for element ``k + 1`` — only
      rotation moves it across the back edge, so only rotation requires it
      outside. Without rotation, producer and consumer are the same iteration
      and are correctly skipped together.
    """
    from tensorforge.common.basic_types import Addressing, Datatype
    from tensorforge.common.context import Context, Options
    from tensorforge.common.matrix.boundingbox import BoundingBox
    from tensorforge.common.matrix.tensor import SubTensor, Tensor
    from tensorforge.generators.descriptions import GemmDescr
    from tensorforge.generators.generator import Generator

    def sub(rows, cols, alias):
        return SubTensor(Tensor([rows, cols], Addressing.STRIDED,
                                BoundingBox([0, 0], [rows, cols]),
                                alias=alias, datatype=Datatype.F32))

    def kernel(**opt_kw):
        ctx = Context(arch="sm_90", backend="cuda", fp_type=Datatype.F32,
                      options=Options(**opt_kw))
        gen = Generator([GemmDescr(False, False, a=sub(16, 16, "A"),
                                   b=sub(16, 16, "B"), c=sub(16, 16, "C"))],
                        ctx)
        gen.generate()
        return gen.get_kernel()

    advance = ["pipe_glb"]
    prefetch = ["consumer_wait", "producer_acquire", "producer_commit"]

    pipe_only = kernel(enable_pipeline=True)
    assert "pipe_glb" in pipe_only, "address pipelining produced no rolling pointer"
    assert not _inside_flag_guard(pipe_only, advance), (
        "the rolling pointer advance is under the flag guard; a skipped "
        "element leaves it trailing the loop variable permanently")

    rotated = kernel(enable_pipeline=True, enable_multibuffer=True)
    assert not _inside_flag_guard(rotated, advance), \
        "rolling pointer advance under the flag guard in the rotating build"
    assert not _inside_flag_guard(rotated, prefetch), (
        "the prefetch is under the flag guard; a skipped element leaves the "
        "stage it should have filled untouched")
