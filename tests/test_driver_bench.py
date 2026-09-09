# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The measurement driver still agrees with the generator.

The failure this is here for has a shape: a driver builds its call site from
its own idea of the launcher's parameter list, the launcher's parameter list
changes, and the driver keeps emitting -- it only stops at the compiler, on
whichever machine has the toolchain, in a message about argument count that
names neither the generator nor the driver. `driver_emit` and `driver_bench`
now share `launcher_call_expr` for exactly this reason, and what follows checks
that the shared answer is the right one across the corpus rather than that the
two agree with each other.

Host-only: generation and text, no compiler.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import re
from pathlib import Path

import pytest

from harness import driver_bench
from harness.driver_emit import collect_operands, launcher_call_expr
from tensorforge.common.context import Context
from tensorforge.generators.generator import Generator

HERE = Path(__file__).resolve().parent
CASES = HERE / "cases"

#: One arch per backend; the launcher signature does not vary with the arch,
#: and sweeping architectures here would buy repetitions rather than coverage.
BACKENDS = (("cuda", "sm_80"), ("hip", "gfx90a"), ("acpp", "sm_80"))


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(f"db_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


def _cases():
    for path in sorted(CASES.rglob("*.py")):
        if path.name.startswith("_"):
            continue
        mod = _load(path)
        if not hasattr(mod, "NAME") or not hasattr(mod, "descr_list"):
            continue
        try:
            mod.descr_list()
        except Exception:
            continue
        yield mod


def _generate(mod, backend, arch):
    ctx = Context(arch=arch, backend=backend, fp_type=mod.DTYPE)
    gen = Generator(mod.descr_list(), ctx, attrs=getattr(mod, "ATTRS", None))
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    return ctx, gen


def _split_top_level(text: str):
    """Comma-separated items, ignoring commas inside brackets.

    `static_cast<float>(2.0f)` and `dim3(a, b, c)` both appear in these lists,
    so a plain `split(",")` reports an argument count that is wrong in exactly
    the cases where the count is interesting.
    """
    items, depth, current = [], 0, ""
    for ch in text:
        if ch in "(<[":
            depth += 1
        elif ch in ")>]":
            depth -= 1
        if ch == "," and depth == 0:
            items.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        items.append(current.strip())
    return items


def _prototype_params(header: str):
    inner = header[header.index("(") + 1:header.rindex(")")]
    return _split_top_level(inner)


# ----------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("backend,arch", BACKENDS,
                         ids=[b for b, _ in BACKENDS])
def test_the_call_has_as_many_arguments_as_the_launcher_has_parameters(
        backend, arch):
    """The check the compiler would eventually make, made here instead.

    Run over the whole corpus, because the parameter list is not one list: a
    `NONE`-addressed operand drops its `extraOffset`, a scalar collapses to a
    literal, a barrier raises the section count and with it the number of
    `numElements` and `flags` parameters. A case exercising only the common
    shape would pass while any of those drifted.
    """
    checked, mismatched = 0, []
    for mod in _cases():
        try:
            _, gen = _generate(mod, backend, arch)
        except Exception:
            continue          # a case this backend refuses is not this test's
        params = _prototype_params(gen.get_header())
        args = _split_top_level(
            launcher_call_expr(gen, collect_operands(gen))
            .split("(", 1)[1].rsplit(")", 1)[0])
        checked += 1
        if len(params) != len(args):
            mismatched.append(f"{mod.NAME}: {len(params)} parameters, "
                              f"{len(args)} arguments")

    assert checked >= 40, f"only {checked} cases generated on {backend}"
    assert not mismatched, "\n".join(mismatched)


@pytest.mark.slow
def test_every_workload_exposes_the_three_entry_points():
    """The driver holds function pointers and never sees a launcher signature.
    A workload that emitted a body but no entry point links, runs, and measures
    the wrong thing -- there is no symbol clash to catch it."""
    for mod in _cases():
        try:
            ctx, gen = _generate(mod, "cuda", "sm_80")
            tu = driver_bench.emit_workload_tu(gen, "cuda", mod.NAME, "")
        except NotImplementedError:
            continue          # documented refusals, covered below
        except Exception:
            continue
        tag = driver_bench.slug(mod.NAME)
        for entry in ("setup", "launch", "teardown", "symbol"):
            assert f"tfb_{entry}_{tag}" in tu, f"{mod.NAME}: no {entry}"


def test_pointer_based_addressing_builds_an_identity_pointer_table():
    """It used to be refused, on the grounds that a timing run would be
    measuring the table as much as the kernel.

    Half of that is true and the half that is true is why the layout is
    pinned rather than left open.  The table is built once in `setup` and
    never touched by the timing loop, so its *construction* is not measured.
    What is measured is the permutation it encodes, and that is a first-order
    term: identity keeps the coalescing a STRIDED operand would get, so a
    number from it is the upper bound on what pointer indirection can reach,
    not what a mesh-ordered batch will see.

    Refusing left the one case the prefetch hint was written for --
    `local_flux`, whose operands are `PTR_BASED` -- outside every measurement
    the corpus can make.  An upper bound that says which bound it is beats no
    number at all.
    """
    mod = _load(CASES / "addressing_ptr_based.py")
    _, gen = _generate(mod, "cuda", "sm_80")
    tu = driver_bench.emit_workload_tu(gen, "cuda", mod.NAME, "")
    ops = [o for o in collect_operands(gen) if o.addressing == "pointer_based"]
    assert ops, "the case no longer has a pointer-based operand"
    for op in ops:
        assert f"d_p_{op.kernel_name}" in tu, f"no table for {op.kernel_name}"
        # identity, and by the element stride rather than the scalar one
        assert (f"h[i] = d_{op.kernel_name} + i * (size_t)" in tu), (
            f"{op.kernel_name}: the table is not the identity layout, or it "
            f"advances by one scalar where the kernel indexes by one element")
        # a source-only batch is read through `const T**`; `T**` does not
        # convert to it, so the declaration has to carry the constness
        elem = op.ctype if op.is_sink else f"const {op.ctype}"
        assert f"static {elem}** d_p_{op.kernel_name}" in tu


def test_a_sparse_operand_is_allocated_at_its_storage_volume():
    """`Tensor.storage_volume()` is the batch stride the kernel indexes with
    (`ptr_manip.py`), and for a compressed tensor it is smaller than the
    bounding box's volume. Allocating the box would be harmless; the two names
    being one word apart is why this is pinned rather than assumed."""
    mod = _load(CASES / "slicing" / "sparsity_band.py")
    _, gen = _generate(mod, "cuda", "sm_80")
    sparse = [o for o in collect_operands(gen)
              if not o.is_scalar and o.storage_volume < o.volume]
    assert sparse, "the banded case stopped being stored compressed"
    tu = driver_bench.emit_workload_tu(gen, "cuda", mod.NAME, "")
    for op in sparse:
        assert f"(size_t){op.storage_volume}u" in tu
        assert f"DEV_MALLOC(d_{op.kernel_name}, (size_t){op.volume}u" not in tu


def test_the_driver_declares_exactly_what_it_is_given():
    """A binary is linked from the workloads that compiled, so the driver is
    emitted twice -- once to know the names, once for the survivors. A stale
    declaration is an undefined symbol at link time and takes the whole binary
    with it."""
    names = ["gemm_square_16", "kernel_30948bd44e", "odd name/with-chars"]
    src = driver_bench.emit_driver(names, "cuda")
    for name in names:
        tag = driver_bench.slug(name)
        assert f"tfb_launch_{tag}" in src
    assert src.count("tfb_setup_") == 2 * len(names)      # decl and table
    assert "odd name/with-chars" in src, (
        "the table lost the human-readable name to sanitisation; the driver "
        "matches on it and the report keys on it")


def test_slugs_are_identifiers_and_stay_distinct():
    assert driver_bench.slug("a-b.c") == "a_b_c"
    assert driver_bench.slug("9lives").startswith("w_")
    assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", driver_bench.slug("x/y z"))


def test_sycl_reports_the_absence_of_a_device_clock():
    """A SYCL launcher submits internally and returns void, so there is no
    event to time. The driver has to say so rather than print a zero, which a
    caller would divide by."""
    src = driver_bench.emit_driver(["w"], "acpp")
    assert "#define DEV_HAS_EVENTS         0" in src
    assert '\\"event_ns\\": null' in src
