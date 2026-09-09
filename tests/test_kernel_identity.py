# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""One symbol, one program.

The name is the digest of the generated source, so a property that reaches
the code reaches the name by construction and there is no list to keep up to
date.  What is left to check is that the construction holds, which is what
these tests do: that the placeholder is gone from every surface, that a
difference in the body separates two names, and -- the one that does not need
to know what to look for -- that across the whole corpus no two different
sources share a name.

That last one is the point of the file.  Sparsity was the third property to
be missing from a name derived from a list of properties, after the flag mask
and the resolved options, and each was found by a link error rather than by a
test.  A test that compares sources instead of enumerating properties finds
the fourth without being told what it is.
"""
from __future__ import annotations

import collections

import pytest

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.context import Context
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.spp import MaskSPP
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr
from tensorforge.generators.generator import Generator
from tensorforge.generators.identity import (KernelNameCollision,
                                              KernelRegistry, registry)

BACKEND, ARCH = "cuda", "sm_86"


def _generate(case, backend: str = BACKEND, arch: str = ARCH):
    ctx = Context(arch=arch, backend=backend,
                  fp_type=getattr(case, "DTYPE", None))
    gen = Generator(case.descr_list(), ctx, attrs=getattr(case, "ATTRS", None))
    gen.generate()
    return gen


def _gemm(size: int = 16, spp=None, dtype: Datatype = Datatype.F32):
    """``C = A @ B``, with an optional sparsity pattern on B."""
    def operand(alias, **kwargs):
        return SubTensor(Tensor([size, size], Addressing.STRIDED,
                                BoundingBox([0, 0], [size, size]),
                                alias=alias, datatype=dtype, **kwargs))
    return [GemmDescr(False, False,
                      operand("A"), operand("B", spp=spp), operand("C"),
                      alpha=1.0, beta=0.0)]


def _band_mask(size: int = 16, width: int = 1):
    import numpy as np
    mask = np.zeros((size, size), dtype=bool, order="F")
    for i in range(size):
        for j in range(size):
            if abs(i - j) <= width:
                mask[i, j] = True
    return mask


class _Case:
    """A case built here rather than discovered, shaped like the ones in
    ``tests/cases`` so that `_generate` takes either."""

    def __init__(self, descrs, dtype: Datatype = Datatype.F32):
        self._descrs = descrs
        self.DTYPE = dtype
        self.NAME = "synthetic"

    def descr_list(self):
        return self._descrs


# ----------------------------------------------------------------------
# The construction
# ----------------------------------------------------------------------

def _generated_or_skip(case, backend: str = BACKEND, arch: str = ARCH):
    """The generator for ``case``, or a skip.

    Two cases in the corpus do not generate at all, and that is a statement
    the snapshots already make -- ``test_generated_source_matches_snapshot``
    records the exception and fails when a case stops or starts raising.
    Repeating it here would report one fault twice; a name has nothing to be
    said about it either way.
    """
    try:
        return _generate(case, backend, arch)
    except Exception as exc:                       # noqa: BLE001
        pytest.skip(f"{case.NAME} [{backend}] does not generate: "
                    f"{type(exc).__name__}")


def test_no_placeholder_survives(snapshot_case):
    """Every spelling of the name has to be the one the substitution knows.

    The source is written before the name exists, so anything emitting the
    name by a route other than ``_base_kernel_name`` leaves the placeholder in
    a file -- an identifier that does not resolve, found by the compiler
    rather than here.  All four targets, because the name is spelled by the
    lexic and the SYCL ones spell more of it than CUDA and HIP do.
    """
    checked = 0
    for backend, arch in (("cuda", "sm_86"), ("hip", "gfx90a"),
                          ("acpp", "pvc"), ("esimd", "pvc")):
        try:
            gen = _generate(snapshot_case, backend, arch)
        except Exception:                          # noqa: BLE001
            continue                               # the snapshots record it
        checked += 1
        for surface in (gen.get_kernel(), gen.get_launcher(), gen.get_header()):
            assert Generator.NAME_PLACEHOLDER not in (surface or ""), (
                f"{snapshot_case.NAME} [{backend}] emits the kernel name by a "
                f"route the substitution does not reach")
    if not checked:
        pytest.skip(f"{snapshot_case.NAME} generates on no target")


def test_name_does_not_change_the_source(snapshot_case):
    """The body is what names the kernel, so the name cannot shape the body.

    A generation with the name pinned to something short must produce the
    source of an unpinned one with the name swapped, character for character.
    Where that fails the digest is taken over text that its own result
    changes, and the name is not a function of the program.
    """
    free = _generated_or_skip(snapshot_case)

    ctx = Context(arch=ARCH, backend=BACKEND,
                  fp_type=getattr(snapshot_case, "DTYPE", None))
    pinned = Generator(snapshot_case.descr_list(), ctx,
                       attrs=getattr(snapshot_case, "ATTRS", None))
    pinned.set_kernel_name("K")
    pinned._announce_identity = False        # 'K' is not meant to be unique
    pinned.generate()

    def surfaces(gen):
        return (gen.get_kernel() or "") + (gen.get_launcher() or "")

    assert surfaces(free).replace(free.get_base_name(), "K") == surfaces(pinned)


# ----------------------------------------------------------------------
# What the construction buys
# ----------------------------------------------------------------------

def test_sparsity_separates_two_kernels():
    """A dense operand and a band-sparse one of the same shape.

    They print the same descriptor -- shape, bounding box and addressing are
    identical and ``gen_descr`` says nothing about the pattern -- and they
    generate different programs, the sparse one around a quarter the size.
    """
    dense = _generate(_Case(_gemm()))
    sparse = _generate(_Case(_gemm(spp=MaskSPP(_band_mask()))))

    assert dense.get_kernel() != sparse.get_kernel()
    assert dense.get_base_name() != sparse.get_base_name()


def test_two_sparsity_patterns_separate_two_kernels():
    """And not merely sparse from dense: one band from another."""
    narrow = _generate(_Case(_gemm(spp=MaskSPP(_band_mask(width=1)))))
    wide = _generate(_Case(_gemm(spp=MaskSPP(_band_mask(width=3)))))

    assert narrow.get_base_name() != wide.get_base_name()


def test_identical_descriptions_share_one_name():
    """The other half of it: one program, one symbol.

    Two generations that agree on every character are the same kernel, and a
    name that separated them would have the routine cache emit one body twice
    under two symbols.
    """
    first = _generate(_Case(_gemm()))
    second = _generate(_Case(_gemm()))

    assert first.get_base_name() == second.get_base_name()
    assert first.unnamed_source() == second.unnamed_source()


# ----------------------------------------------------------------------
# The one that does not need to know what to look for
# ----------------------------------------------------------------------

def test_no_two_corpus_kernels_share_a_name():
    """Across the corpus, a shared name has to mean a shared source.

    This is the test the three name collisions to date would each have failed
    on the commit that introduced them, without anyone having to think of the
    property that was missing.  It says nothing about *which* property: it
    generates everything, groups by symbol, and reports any group whose
    members are not the same program.
    """
    from conftest import _discover_cases      # same directory as this file

    by_name = collections.defaultdict(list)
    for case in _discover_cases():
        for backend, arch in (("cuda", "sm_86"), ("hip", "gfx90a")):
            try:
                gen = _generate(case, backend, arch)
            except Exception:                      # noqa: BLE001
                continue                           # covered by the snapshots
            by_name[gen.get_base_name()].append(
                (f"{case.NAME} [{backend}]", gen.unnamed_source()))

    collisions = {
        name: [label for label, _ in entries]
        for name, entries in by_name.items()
        if len({source for _, source in entries}) > 1
    }
    assert not collisions, (
        "one symbol, two programs:\n  "
        + "\n  ".join(f"{name}: {', '.join(labels)}"
                      for name, labels in sorted(collisions.items())))


# ----------------------------------------------------------------------
# The registry
# ----------------------------------------------------------------------

def test_registry_accepts_a_repeated_kernel():
    reg = KernelRegistry()
    reg.register("kernel_a", "void a() {}")
    reg.register("kernel_a", "void a() {}")
    assert len(reg) == 1


def test_registry_rejects_two_sources_under_one_name():
    reg = KernelRegistry()
    reg.register("kernel_a", "void a() {}")
    with pytest.raises(KernelNameCollision, match="kernel_a"):
        reg.register("kernel_a", "void a() { different(); }")


def test_registry_names_both_kernels():
    """The report has to say which two, or it only says to go looking."""
    reg = KernelRegistry()
    reg.register("kernel_a", "one", descriptors=["descriptor of the first"])
    with pytest.raises(KernelNameCollision) as excinfo:
        reg.register("kernel_a", "two", descriptors=["descriptor of the second"])
    message = str(excinfo.value)
    assert "descriptor of the first" in message
    assert "descriptor of the second" in message


def test_generation_registers_its_name():
    """Generation announces itself, or nothing above is load-bearing."""
    gen = _generate(_Case(_gemm()))
    assert gen.get_base_name() in registry().names()


def test_a_pinned_name_used_twice_is_refused():
    """The path that stays a real check once names are pinned by hand.

    ``set_kernel_name`` hands the uniqueness question to the caller.  Nothing
    in the tree calls it today, which is exactly why the check has to be here
    rather than in the caller that will.
    """
    ctx = Context(arch=ARCH, backend=BACKEND, fp_type=Datatype.F32)
    first = Generator(_gemm(), ctx)
    first.set_kernel_name("kernel_pinned_by_hand")
    first.generate()

    second = Generator(_gemm(spp=MaskSPP(_band_mask())), ctx)
    second.set_kernel_name("kernel_pinned_by_hand")
    with pytest.raises(KernelNameCollision, match="kernel_pinned_by_hand"):
        second.generate()
