# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A batch-constant `A` stored so that a lane reads its rows in vectors.

The generic nest gives lane `l` the rows `l, l + T, l + 2T, ...`, which a
column-major `A` keeps `T` elements apart -- one scalar load per row.  With
`Options.prepare_operands` the operand is instead stored with each lane's rows
side by side in groups of one 16-byte vector (`Tensor.simt_interleave`), and
the kernel reads a group with one aligned load.  The host packs it from
`storage_map`, which the device tests exercise; these tests hold the
contract: when it is offered, what the order is, and that the kernel reads
nothing but the vectors.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import re
from pathlib import Path

from tensorforge.common.context import Context
from tensorforge.generators.generator import Generator

CASE = Path(__file__).parent / "cases" / "local_flux.py"


def _generate(monkeypatch, options):
    monkeypatch.setenv("TF_OPTIONS", options)
    spec = importlib.util.spec_from_file_location("lf_interleave", CASE)
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    gen = Generator(mod.descr_list(),
                    Context(arch="sm_120", backend="cuda", fp_type=mod.DTYPE))
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    operators = [s.obj for s in gen._scopes.get_global_scope().values()
                 if s.obj.alias and s.obj.alias.startswith("A")]
    return gen.get_kernel(), operators


def test_the_operators_are_interleaved_at_eight_lanes(monkeypatch):
    """56 rows over 8 lanes: 7 slots a lane, two groups of four, one slot
    of padding per lane and column."""
    _, operators = _generate(monkeypatch, "prepare_operands=1,lanes_per_mult=8")
    assert len(operators) == 4
    for a in operators:
        assert a.simt_interleave == (8, 4, 64)
        order = a.storage_map()
        cells = [c for c in order if c >= 0]
        assert len(order) == 64 * 56
        assert sorted(cells) == list(range(56 * 56))
        assert len(order) - len(cells) == 8 * 56


def test_the_slot_a_lane_reads_is_the_row_it_owns(monkeypatch):
    """Lane `l`, slot `s`, column `k` sits at `k*ld + (s//4)*32 + 4l + s%4`
    and holds row `l + 8s` -- the cyclic row the nest gives that lane."""
    _, operators = _generate(monkeypatch, "prepare_operands=1,lanes_per_mult=8")
    order = operators[0].storage_map()
    for k in (0, 17, 55):
        for lane in range(8):
            for s in range(7):
                pos = k * 64 + (s // 4) * 32 + 4 * lane + s % 4
                assert order[pos] == (lane + 8 * s) + 56 * k


def test_the_kernel_reads_the_operators_only_as_aligned_vectors(monkeypatch):
    src, _ = _generate(monkeypatch, "prepare_operands=1,lanes_per_mult=8")
    reads = re.findall(r"\*\((tensorforge::\w+<[^>]*>)\*\)&glb_m[0468]\[", src)
    assert reads and set(reads) == {"tensorforge::VectorT<float, 4>"}
    assert not re.search(r"= glb_m[0468]\[", src)


def test_nothing_is_prepared_unless_asked(monkeypatch):
    _, operators = _generate(monkeypatch, "lanes_per_mult=8")
    assert all(a.storage_map() is None for a in operators)
    assert all(a.simt_interleave is None for a in operators)


def test_rows_that_do_not_fill_the_lanes_are_left_alone(monkeypatch):
    """At 32 lanes 56 rows leave a guarded block, whose loads a guard would
    split into separate scopes; the order is not offered there."""
    _, operators = _generate(monkeypatch, "prepare_operands=1")
    assert all(a.simt_interleave is None for a in operators)


def test_a_merged_run_stores_every_member_alike(monkeypatch):
    """Faces 1-3 of `local_flux` merge into one body over a stand-in.  The
    stand-in is no buffer; its members are, and the one body reads whichever
    the counter selects, so all of them are interleaved and the stand-in
    carries the addressing."""
    src, operators = _generate(
        monkeypatch, "prepare_operands=1,lanes_per_mult=8,merge_variants=1")
    stored = [a for a in operators if not a.is_variant]
    assert len(stored) == 4
    assert all(a.simt_interleave == (8, 4, 64) for a in stored)
    assert all(a.storage_map() is not None for a in stored)
    reads = re.findall(r"\*\((tensorforge::\w+<[^>]*>)\*\)&glb_(?:m0|v0)\[",
                       src)
    assert reads and set(reads) == {"tensorforge::VectorT<float, 4>"}
    assert not re.search(r"= glb_(?:m0|v0)\[", src)
