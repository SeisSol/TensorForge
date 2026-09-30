# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Shared memory is not overwritten under a value still to be read.

An independent check of what `LivenessAnalysis` and the region allocator
decide together.  It walks the emitted stream in program order -- a merged
run's body twice, for its back edge -- and keeps, per buffer, whether another
buffer placed on overlapping memory has been written since the buffer was last
written in full; a read of a buffer in that state is a clobber.  Coverage is
judged from the boxes each write names, not from `partial_defs`, so the check
does not reuse the reasoning it checks: a flaw there would leave the verifier
green while a slice written into a buffer a whole write defined
(`temp_slice_after_whole`), a pointwise write into a slice
(`elementwise_slice_after_whole`) or a slice written first inside a merged run
(below) comes out clobbered.

A merged run whose body reads the image it carries after re-computing it is
the other thing pinned here: closing the chain renames the image register,
and a reader holding it as a view that kept the old name would read a
register nothing writes.

And a store into a buffer a clearing store wrote waits for a barrier: the
zeros go out on the clearing nest's lanes, the later store writes some of the
same cells from others, and two lanes writing one cell unordered is a race.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import itertools
import warnings
from pathlib import Path

import pytest

from tensorforge.backend.instructions.allocate import RegisterAlloc
from tensorforge.backend.instructions.memory import AbstractShrMemWrite
from tensorforge.backend.instructions.memory.load import (GlbToShrLoader,
                                                          LoadWait)
from tensorforge.backend.instructions.memory.store import StoreRegToShr
from tensorforge.backend.instructions.sync_block import SyncThreads
from tensorforge.backend.instructions.ptr_manip import VariantLoop
from tensorforge.backend.symbol import SymbolType
from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.context import Context, Options
from tensorforge.common.helper import generate_tmp_matrix
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr
from tensorforge.generators.generator import Generator

CASES = Path(__file__).parent / "cases"
TARGETS = [("cuda", "sm_86"), ("hip", "gfx90a")]


# -- the check ----------------------------------------------------------- #

def _flat(instrs, out):
    for instr in instrs:
        out.append(instr)
        for region in instr.regions():
            body = _flat(region, [])
            out.extend(body)
            if isinstance(instr, VariantLoop):
                out.extend(body)
    return out


def _cells(lo, hi):
    return set(itertools.product(*[range(a, b) for a, b in zip(lo, hi)]))


def _written(instr, sym):
    """The cells of `sym`'s buffer `instr` writes, or None for all of them."""
    if isinstance(instr, (GlbToShrLoader, LoadWait)):
        return None
    if isinstance(instr, StoreRegToShr):
        src = instr._src.data_view.get_bbox()
        cells = _cells([l + o for l, o in zip(src.lower(), instr._dest_offset)],
                       [u + o for u, o in zip(src.upper(), instr._dest_offset)])
        if getattr(instr, '_clear', False):
            within = getattr(instr, '_clear_within', None)
            if within is None:
                return None
            cells |= _cells(within.lower(), within.upper())
        return cells
    view = getattr(instr, '_dest', None)
    if getattr(view, 'symbol', None) is sym and hasattr(view, 'bbox'):
        offset = view.offset or [0] * view.bbox.rank()
        return _cells([l + o for l, o in zip(view.bbox.lower(), offset)],
                      [u + o for u, o in zip(view.bbox.upper(), offset)])
    return set()


def _extents(flat):
    out = {}
    for instr in flat:
        offset = getattr(instr, '_shr_mem_offset', None)
        size = getattr(instr, 'compute_shared_mem_size', None)
        if (not isinstance(offset, int) or not callable(size)
                or getattr(instr, '_global_offset', False)):
            continue
        for sym in instr.defs():
            if (getattr(sym, 'stype', None) is SymbolType.SharedMem
                    and not getattr(sym, 'block_shared', False)):
                lo, hi = out.get(id(sym), (offset, offset))
                out[id(sym)] = (offset, max(hi, offset + int(size())))
    return out


def clobbers(gen):
    """`(buffer read, buffer written over it)` for every clobbered read."""
    found = []
    for section in gen._sections:
        flat = _flat(list(section.stream), [])
        extents = _extents(flat)
        held, over, covered = set(), {}, {}
        for instr in flat:
            if instr.regions():
                continue
            for sym in instr.uses():
                if over.get(id(sym)):
                    found.append((sym.name, over[id(sym)]))
                    over[id(sym)] = None
            for sym in instr.defs():
                if id(sym) not in extents:
                    continue
                lo, hi = extents[id(sym)]
                for other in held - {id(sym)}:
                    olo, ohi = extents[other]
                    if lo < ohi and olo < hi and not over.get(other):
                        over[other], covered[other] = sym.name, set()
                held.add(id(sym))
                if over.get(id(sym)):
                    cells = _written(instr, sym)
                    buf = sym.data_view.get_bbox()
                    if cells is None or _cells(buf.lower(), buf.upper()) <= (
                            covered[id(sym)] | cells):
                        over[id(sym)] = None
                    else:
                        covered[id(sym)] |= cells
    return found


def unwritten_registers(gen):
    """Registers read inside a merged run that no instruction writes."""
    found = []
    for section in gen._sections:
        flat = _flat(list(section.stream), [])
        written = {id(s) for i in flat
                   if not isinstance(i, RegisterAlloc) and not i.regions()
                   for s in i.defs()}
        for instr in flat:
            if not isinstance(instr, VariantLoop):
                continue
            for inner in _flat(list(instr.region), []):
                for sym in inner.uses():
                    if (getattr(sym, 'stype', None) is SymbolType.Register
                            and id(sym) not in written):
                        found.append((str(inner), sym.name))
    return found


def unfenced_rewrites(gen):
    """Shared writes into a buffer a clearing store wrote, no barrier between."""
    found = []
    for section in gen._sections:
        cleared = set()
        for instr in _flat(list(section.stream), []):
            if isinstance(instr, SyncThreads):
                cleared = set()
            elif isinstance(instr, AbstractShrMemWrite):
                dest = instr.get_dest()
                if id(dest) in cleared:
                    found.append((str(instr), dest.name))
                if getattr(instr, '_clear', False):
                    cleared.add(id(dest))
    return found


# -- shapes -------------------------------------------------------------- #

def _t(rows, alias, cols=12):
    return Tensor([32, 32], Addressing.STRIDED,
                  BoundingBox([0, 0], [rows, cols]), alias=alias,
                  datatype=Datatype.F32)


def _rows(tensor, lo, hi):
    return SubTensor(tensor, BoundingBox([0, 0], [hi - lo, 12]), [lo, 0],
                     sliced=True)


def slice_first_in_run(count=3):
    """`tmp` written whole before a merged run, whose body writes rows 0..4
    of it first and then reads all of it."""
    b, c, g, h = (SubTensor(_t(12, n)) for n in "BCGH")
    f1, f2 = SubTensor(_t(6, "F1")), SubTensor(_t(6, "F2"))
    d = SubTensor(_t(12, "D"))
    tmp, x = generate_tmp_matrix(b, c), generate_tmp_matrix(b, c)
    out = [GemmDescr(False, False, a=b, b=c, c=SubTensor(tmp)),
           GemmDescr(False, False, a=f1, b=g, c=_rows(x, 0, 6)),
           GemmDescr(False, False, a=f2, b=g, c=_rows(x, 6, 12)),
           GemmDescr(False, False, a=SubTensor(x), b=h, c=d)]
    for k in range(count):
        out += [GemmDescr(False, False, a=SubTensor(_t(4, f"N{k}")), b=c,
                          c=_rows(tmp, 0, 4)),
                GemmDescr(False, False, a=SubTensor(tmp),
                          b=SubTensor(_t(12, f"E{k}")), c=d,
                          alpha=1.0, beta=1.0)]
    return out


def carried_reader(count=4):
    """`t = N(k) C; t += u Q(k)` in a merged run, read again by `D += t E(k)`."""
    b, c, p = (SubTensor(_t(12, n)) for n in "BCP")
    d = SubTensor(_t(12, "D"))
    t, u = generate_tmp_matrix(b, c), generate_tmp_matrix(b, c)
    out = [GemmDescr(False, False, a=b, b=c, c=SubTensor(t)),
           GemmDescr(False, False, a=b, b=p, c=d)]
    for k in range(count):
        out += [GemmDescr(False, False, a=SubTensor(t), b=p, c=SubTensor(u)),
                GemmDescr(False, False, a=SubTensor(_t(12, f"N{k}")), b=c,
                          c=SubTensor(t)),
                GemmDescr(False, False, a=SubTensor(u),
                          b=SubTensor(_t(12, f"Q{k}")), c=SubTensor(t),
                          alpha=1.0, beta=1.0),
                GemmDescr(False, False, a=SubTensor(t),
                          b=SubTensor(_t(12, f"E{k}")), c=d,
                          alpha=1.0, beta=1.0)]
    return out


def _generate(descrs, backend, arch, fp=Datatype.F32, **options):
    ctx = Context(arch=arch, backend=backend, fp_type=fp,
                  options=Options(**options) if options else None)
    gen = Generator(descrs, ctx)
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        gen.generate()
    return gen


def _case(stem):
    path = CASES / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(
        f"tf_case__{stem.replace('/', '__')}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# -- tests --------------------------------------------------------------- #

@pytest.mark.parametrize("stem", ["slicing/temp_slice_after_whole",
                                  "slicing/temp_dead_slice_reassign",
                                  "slicing/two_assembled_reuse",
                                  "slicing/temp_two_writers",
                                  "elementwise/slice_after_whole",
                                  "mixed/ml_slices_then_ew"])
@pytest.mark.parametrize("backend,arch", TARGETS, ids=[b for b, _ in TARGETS])
def test_a_slice_keeps_what_it_does_not_write(stem, backend, arch):
    mod = _case(stem)
    gen = _generate(mod.descr_list(), backend, arch, fp=mod.DTYPE)
    assert clobbers(gen) == []


@pytest.mark.parametrize("backend,arch", TARGETS, ids=[b for b, _ in TARGETS])
def test_a_slice_first_in_a_merged_run_keeps_what_came_before(backend, arch):
    gen = _generate(slice_first_in_run(), backend, arch, merge_variants=True)
    assert any(isinstance(i, VariantLoop)
               for s in gen._sections for i in _flat(list(s.stream), [])), \
        "the repetition was not merged; the test no longer tests a loop"
    assert clobbers(gen) == []


@pytest.mark.parametrize("backend,arch", TARGETS, ids=[b for b, _ in TARGETS])
def test_a_merged_run_reads_the_image_it_carries(backend, arch):
    gen = _generate(carried_reader(), backend, arch, merge_variants=True)
    assert any(isinstance(i, VariantLoop)
               for s in gen._sections for i in _flat(list(s.stream), [])), \
        "the repetition was not merged; the test no longer tests a loop"
    assert unwritten_registers(gen) == []


@pytest.mark.parametrize("stem", ["slicing/temp_dead_slice_reassign",
                                  "slicing/temp_reassign_narrower_slice",
                                  "slicing/temp_two_writers",
                                  "mixed/ml_then_ew_temp_narrower"])
@pytest.mark.parametrize("backend,arch", TARGETS, ids=[b for b, _ in TARGETS])
def test_a_store_after_a_clear_waits_for_it(stem, backend, arch):
    mod = _case(stem)
    gen = _generate(mod.descr_list(), backend, arch, fp=mod.DTYPE)
    assert unfenced_rewrites(gen) == []
