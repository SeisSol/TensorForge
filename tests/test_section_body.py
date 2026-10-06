# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A section is one body: the arena, the preloaded operators, the traversal.

What one body buys is visibility across what used to be separate bodies: a
window bound in the prologue is a value the loop reads, a transfer issued in
the prologue and the wait that retires it are statements of one body, and
the indices the traversal starts from are values a pass can delete where
nothing reads them.  These pin the three.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

from tensorforge.backend import pir
from tensorforge.backend.instructions import abstract_instruction
from tensorforge.common.context import Context, Options
from tensorforge.generators.generator import Generator

CASES = Path(__file__).parent / "cases"

TARGETS = [("cuda", "sm_86"), ("hip", "gfx942"), ("acpp", "pvc"),
           ("esimd", "pvc")]


def _generator(case_file: str, backend: str, arch: str, **options):
    path = CASES / case_file
    spec = importlib.util.spec_from_file_location(
        "tf_section__" + path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ctx = Context(arch=arch, backend=backend,
                  fp_type=getattr(mod, "DTYPE", None),
                  options=Options(**options))
    return Generator(mod.descr_list(), ctx, attrs=getattr(mod, "ATTRS", None))


@pytest.mark.parametrize("backend,arch", TARGETS)
@pytest.mark.parametrize("case_file", ["chain_three.py",
                                       "barrier/barrier_two_gemms.py",
                                       "barrier/fence_two_gemms.py"])
def test_a_section_is_emitted_as_one_body(case_file, backend, arch,
                                          monkeypatch):
    if case_file == "barrier/barrier_two_gemms.py" and arch == "pvc":
        pytest.skip("no grid barrier on pvc")
    emitted = []
    original = pir.emit

    def counting(body, writer, context=None):
        emitted.append(body)
        return original(body, writer, context)

    monkeypatch.setattr(abstract_instruction.pir, "emit", counting)
    gen = _generator(case_file, backend, arch, merge_variants=False)
    gen.generate()
    assert len(emitted) == len(gen._sections), (
        f"{len(emitted)} bodies for {len(gen._sections)} section(s)")


@pytest.mark.parametrize("backend,arch", TARGETS)
def test_nothing_declares_an_index_nothing_reads(backend, arch):
    """Without a peel nothing reads where the traversal starts, and the values
    standing for it are gone."""
    gen = _generator("chain_three.py", backend, arch)
    gen.generate()
    kernel = gen.get_kernel()
    assert not re.search(r"\bbatchId(_start|1|2)\b", kernel), kernel


@pytest.mark.parametrize("backend,arch", [("cuda", "sm_86"), ("hip", "gfx90a")])
def test_the_peel_and_the_loop_name_different_first_elements(backend, arch):
    """Ahead of the loop `batchId1` is the clamped start, inside it the
    clamped successor: two values with one hint."""
    gen = _generator("chain_five.py", backend, arch, enable_wrap_loads=True)
    gen.generate()
    lines = gen.get_kernel().splitlines()
    loop = next(i for i, l in enumerate(lines)
                if re.search(r"for \(size_t v\d+_batchId0\b", l))
    before = {v for l in lines[:loop]
              for v in re.findall(r"\b(v\d+)_batchId1 = ", l)}
    inside = {v for l in lines[loop:]
              for v in re.findall(r"\b(v\d+)_batchId1 = ", l)}
    assert len(before) == 1 and len(inside) == 1 and before != inside, (
        before, inside)
