# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Shared memory is not overwritten under a value still to be read.

An independent check of what the allocator decides (`pir.allocate`): the same
kernel built twice, once with its buffers laid out by their lifetimes and once
with every buffer on bytes of its own, has to compute the same numbers on the
host oracle.  Nothing in the comparison reuses the allocator's reasoning --
not the liveness, not what an instruction says it defines whole -- so a flaw
there shows up as a difference: a slice written into a buffer a whole write
defined (`temp_slice_after_whole`), a pointwise write into a slice
(`elementwise_slice_after_whole`) or a slice written first inside a merged run
(below) that comes out clobbered.

A merged run whose body reads the image it carries after re-computing it is
the other thing pinned here: closing the chain renames the image register,
and a reader holding it as a view that kept the old name would read a
register nothing writes.

And a store into a buffer a clearing store wrote waits for a barrier, as does
the clearing store for a store before it: the zeros go out on the clearing
nest's lanes, the other store writes some of the same cells from others, and
two lanes writing one cell unordered is a race -- which the oracle's race
check sees whichever order the lanes happened to take.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import warnings
from pathlib import Path

import pytest

from tensorforge.backend.instructions.allocate import RegisterAlloc
from tensorforge.backend.instructions.ptr_manip import VariantLoop
from tensorforge.backend.pir import allocate
from tensorforge.backend.pir.core import Effect, MemSpace, Op
from tensorforge.backend.symbol import SymbolType
from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.context import Context, Options
from tensorforge.common.helper import generate_tmp_matrix
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.generators.descriptions import GemmDescr
from tensorforge.generators.generator import Generator
from tensorforge.reference import kernel_eval

CASES = Path(__file__).parent / "cases"
TARGETS = [("cuda", "sm_86"), ("hip", "gfx90a")]
#: What the host oracle can run: AMD's matrix paths call intrinsics it does
#: not model.
ORACLE = [("cuda", "sm_86")]


# -- the checks ---------------------------------------------------------- #

def _flat(instrs, out):
    for instr in instrs:
        out.append(instr)
        for region in instr.regions():
            body = _flat(region, [])
            out.extend(body)
            if isinstance(instr, VariantLoop):
                out.extend(body)
    return out


@contextlib.contextmanager
def _no_sharing():
    """Every buffer on bytes of its own: as though each one were occupied
    together with every other."""
    original = allocate._Liveness.record

    def record(self, occupied):
        original(self, occupied)
        if self.recording and occupied:
            self.neighbors = [self.everything] * len(self.neighbors)
    allocate._Liveness.record = record
    try:
        yield
    finally:
        allocate._Liveness.record = original


def _numbers(gen, seed=11):
    lanes, mults = kernel_eval.launch_geometry(gen.get_launcher())
    return kernel_eval.evaluate_wave(gen.get_kernel(), lanes, seed=seed,
                                     globals_only=True, mults=mults)


def clobbers(descrs, backend, arch, fp=Datatype.F32, **options):
    """The outputs a lifetime layout gets wrong, against a layout that shares
    no bytes at all."""
    packed = _generate(descrs(), backend, arch, fp, **options)
    with _no_sharing():
        apart = _generate(descrs(), backend, arch, fp, **options)
    shared = lambda g: sum(s.shr_mem_obj.get_size_per_mult() or 0
                           for s in g._sections)
    assert shared(apart) >= shared(packed)
    want, got = _numbers(apart), _numbers(packed)
    return sorted(k for k in set(want) | set(got)
                  if abs(want.get(k, 0.0) - got.get(k, 0.0)) > 1e-4)


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


def _stmts(body):
    for s in body:
        yield s
        for r in s.regions:
            yield from _stmts(r.body)


def unfenced_clears(gen):
    """Stores into a buffer a clearing store wrote, and clearing stores into a
    buffer something wrote, with no barrier between -- in the body as it is
    emitted."""
    found = []
    for section in gen._sections:
        cleared, written = set(), set()
        for s in _stmts(section.body):
            if s.op == Op.BARRIER:
                cleared, written = set(), set()
                continue
            if s.op == Op.MARK:
                keys = {a.id for a in s.args}
                if s.attr('mark') == 'clears' and keys & written:
                    found.append(('clears', s.args[0].hint))
                if s.attr('mark') == 'cleared':
                    cleared |= keys
                continue
            if s.op == Op.ALLOC:
                continue
            for a in s.accesses:
                if (a.space == MemSpace.SHARED and a.kind & Effect.WRITE
                        and getattr(a.base, 'id', None) is not None):
                    if a.base.id in cleared:
                        found.append(('store', a.base.hint))
                    written.add(a.base.id)
    return found


def races(gen):
    lanes, mults = kernel_eval.launch_geometry(gen.get_launcher())
    found = []
    kernel_eval.evaluate_wave(gen.get_kernel(), lanes, seed=3, races=found,
                              elements=2)
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


def slice_after_gap():
    """`tmp` written whole, then a slice of it after another temporary has
    come and gone: rows 4..12 of `tmp` are still the first write's, and `y`,
    occupied only in between, must not be laid over them.  The slice defines
    nothing whole, so it does not say it does."""
    b, c, f, g, h, e = (SubTensor(_t(12, n)) for n in "BCFGHE")
    d = SubTensor(_t(12, "D"))
    tmp, y = generate_tmp_matrix(b, c), generate_tmp_matrix(b, c)
    return [GemmDescr(False, False, a=b, b=c, c=SubTensor(tmp)),
            GemmDescr(False, False, a=f, b=g, c=SubTensor(y)),
            GemmDescr(False, False, a=SubTensor(y), b=h, c=d),
            GemmDescr(False, False, a=SubTensor(_t(4, "N")),
                      b=SubTensor(_t(12, "C2")), c=_rows(tmp, 0, 4)),
            GemmDescr(False, False, a=SubTensor(tmp), b=e, c=d,
                      alpha=1.0, beta=1.0)]


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
@pytest.mark.parametrize("backend,arch", ORACLE, ids=[b for b, _ in ORACLE])
def test_a_slice_keeps_what_it_does_not_write(stem, backend, arch):
    mod = _case(stem)
    assert clobbers(mod.descr_list, backend, arch, fp=mod.DTYPE) == []


@pytest.mark.parametrize("backend,arch", ORACLE, ids=[b for b, _ in ORACLE])
def test_a_slice_after_another_temporary_keeps_the_rest(backend, arch):
    """The layout has to share bytes for this to say anything, and it does:
    `tmp` and `y` are each laid over a staged operand."""
    gen = _generate(slice_after_gap(), backend, arch)
    sizes = [s.shr_mem_obj.get_size_per_mult() for s in gen._sections]
    with _no_sharing():
        apart = _generate(slice_after_gap(), backend, arch)
    assert sizes != [s.shr_mem_obj.get_size_per_mult() for s in apart._sections]
    assert clobbers(slice_after_gap, backend, arch) == []


@pytest.mark.parametrize("backend,arch", ORACLE, ids=[b for b, _ in ORACLE])
def test_a_slice_first_in_a_merged_run_keeps_what_came_before(backend, arch):
    gen = _generate(slice_first_in_run(), backend, arch, merge_variants=True)
    assert any(isinstance(i, VariantLoop)
               for s in gen._sections for i in _flat(list(s.stream), [])), \
        "the repetition was not merged; the test no longer tests a loop"
    assert clobbers(slice_first_in_run, backend, arch,
                    merge_variants=True) == []


@pytest.mark.parametrize("backend,arch", TARGETS, ids=[b for b, _ in TARGETS])
def test_a_merged_run_reads_the_image_it_carries(backend, arch):
    gen = _generate(carried_reader(), backend, arch, merge_variants=True)
    assert any(isinstance(i, VariantLoop)
               for s in gen._sections for i in _flat(list(s.stream), [])), \
        "the repetition was not merged; the test no longer tests a loop"
    assert unwritten_registers(gen) == []


@pytest.mark.parametrize("stem", ["slicing/temp_dead_slice_reassign",
                                  "slicing/temp_reassign_narrower_slice",
                                  "mixed/ml_then_ew_temp_narrower"])
@pytest.mark.parametrize("backend,arch", TARGETS, ids=[b for b, _ in TARGETS])
def test_a_store_and_a_clear_wait_for_each_other(stem, backend, arch):
    mod = _case(stem)
    gen = _generate(mod.descr_list(), backend, arch, fp=mod.DTYPE)
    assert any(s.op == Op.MARK and s.attr('mark') == 'cleared'
               for section in gen._sections for s in _stmts(section.body)), \
        "nothing is cleared; the test no longer tests a clear"
    assert unfenced_clears(gen) == []
    if (backend, arch) in ORACLE:
        assert races(gen) == []
