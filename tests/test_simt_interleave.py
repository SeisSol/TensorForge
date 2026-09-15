# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A batch-constant `A` stored so that a lane reads its rows in vectors.

The generic nest gives lane `l` the rows `l, l + T, l + 2T, ...`, which a
column-major `A` keeps `T` elements apart -- one scalar load per row.  With
`Options.prepare_operands` the operand is instead stored with each lane's rows
side by side in groups of up to one 16-byte vector, as wide as a lane's rows
need (`Tensor.simt_interleave`), and the kernel reads a group with one aligned
load.  The host packs it from
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


def test_a_ragged_last_slot_is_padding(monkeypatch):
    """At 32 lanes 56 rows are two slots, the second 24 rows deep: one pair
    a lane and column, of which rows 56 and on are padding.  It was not
    offered at all, since the rows do not fill the lanes; the tail's lanes
    past the data read the padding, which is zero and stored."""
    src, operators = _generate(monkeypatch, "prepare_operands=1")
    assert len(operators) == 4
    for a in operators:
        assert a.simt_interleave == (32, 2, 64)
        order = a.storage_map()
        assert len(order) == 64 * 56
        assert sorted(c for c in order if c >= 0) == list(range(56 * 56))
    reads = re.findall(r"\*\((tensorforge::\w+<[^>]*>)\*\)&glb_m[0468]\[", src)
    assert reads and set(reads) == {"tensorforge::VectorT<float, 2>"}
    assert not re.search(r"= glb_m[0468]\[", src)


def test_one_row_a_lane_keeps_the_plain_layout():
    """At one row a lane there is nothing to put side by side: the plain
    layout already hands the lanes of a column one contiguous run, and a
    group would be padding in every load -- three floats in four for
    `localFluxAll`'s 32-row `fMrT` at 32 lanes.  Its 64-row `rDivM` are two
    rows a lane, and take a pair."""
    _, operators = _seissol("localFluxAll", options={"prepare_operands": True})
    rows = {a.name: int(a.get_actual_shape()[0]) for a in operators}
    assert {32, 64} <= set(rows.values())
    for a in operators:
        if rows[a.name] == 32:
            assert a.simt_interleave is None, a.name
            assert a.storage_map() is None, a.name
        elif rows[a.name] == 64:
            assert a.simt_interleave == (32, 2, 64), a.name


def _seissol(kernel, config="elastic-linearck-o6-s", options=None):
    import seissol_suite as fx
    from tensorforge.common.basic_types import Addressing, Datatype
    from tensorforge.common.context import Options
    from tensorforge.frontend.yateto import DescriptionReader
    system = config.rsplit("-o", 1)[0]
    descrs = DescriptionReader(None, {}).read(
        fx.description(system, config, f"gpu_{kernel}"))[0]
    ctx = Context(arch="sm_80", backend="cuda", fp_type=Datatype.F32,
                  options=Options(**(options or {})))
    gen = Generator(descrs, ctx)
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    operators = [s.obj for s in gen._scopes.get_global_scope().values()
                 if getattr(s.obj, "addressing", None) is Addressing.NONE]
    return gen.get_kernel(), operators


def test_a_box_that_starts_on_a_slot_takes_the_order():
    """SeisSol's order-6 `volume` reads rows 1 to 54 of each 64-row `kDivM`:
    whole slots of it from the first, which is all the address needs."""
    _, operators = _seissol("volume", options={"prepare_operands": True})
    assert len(operators) == 3
    assert all(a.simt_interleave is not None for a in operators)


def test_every_reader_of_an_operator_shares_its_interleave():
    """The order-6 `derivative` reads each `kDivMT` five times, over boxes
    that shrink with the order, and stores it from column 1.  Every reader
    is a nest of the same lanes, so all take the one order."""
    src, operators = _seissol("derivative", options={"prepare_operands": True})
    assert len(operators) == 3
    for a in operators:
        assert a.simt_interleave is not None, a.alias
        assert list(a.get_bbox().lower()) == [0, 1]
    plain, _ = _seissol("derivative")
    vectors = lambda text: len(re.findall(r"VectorT<float, [24]>", text))
    assert vectors(src) > vectors(plain)


def test_a_preloaded_operator_is_copied_as_it_is_stored():
    """With `preload_globals` the section prologue copies each operator into
    shared memory as it lies, and the nest reads the copy in the interleave.
    The copy has to hold the whole stored operator: sized for the dense one,
    the interleaved reads ran into the next operator's copy.  So each image
    is laid out at least as long as its storage, one after the other.  The
    derivative's 48-row operators are what tells the two sizes apart: at 32
    lanes a lane holds two of their rows or one and padding."""
    src, operators = _seissol("derivative", options={"prepare_operands": True,
                                                      "preload_globals": True})
    stored = {a.name: int(a.storage_volume()) for a in operators
              if a.simt_interleave is not None}
    assert len(stored) == 3
    assert all(int(a.storage_volume()) > int(a.get_actual_volume())
               for a in operators if a.simt_interleave is not None)
    at = {m.group(1)[len("glb_"):]: int(m.group(2)) for m in re.finditer(
        r"\* __restrict__ (glb_m\d+) = &totalShrMem\[(\d+)\];", src)}
    names = sorted(stored, key=lambda name: at[name])
    for first, second in zip(names, names[1:]):
        assert at[second] - at[first] >= stored[first]


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
