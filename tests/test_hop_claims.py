# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""What a synchronous copy into shared memory states about its addresses.

A wide access is spelled as a reinterpret cast, and the cast is undefined
unless the address is as aligned as the access is wide.  The IR cannot work
that out from an index it holds as an expression, so the access carries it
(`align`), and the verifier refuses a wide one that carries nothing.  The
copy knows both ends: the image starts on sixteen bytes, and the operand on
what the frontend promised.  Where that covers the access it is a byte
count; where it does not, the access is spelled with the vector type that
needs no more than the element (`relaxed`), and keeps its width -- where the
lowering has such a type.  `sycl::vec` has none, and there the promise
bounds the width instead.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import re
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorforge.backend import pir
from tensorforge.backend.instructions.memory.load import GlbToShrLoader
from tensorforge.backend.pir import verify
from tensorforge.backend.pir.core import Op
from tensorforge.backend.pir.passes import walk_stmts
from tensorforge.backend.symbol import RELAXED, Symbol, SymbolType
from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.context import Context
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.common.options import Options
from tensorforge.generators.descriptions import GemmDescr
from tensorforge.generators.generator import Generator

CASES = Path(__file__).resolve().parent / 'cases'


# --------------------------------------------------------------------------- #
# One access
# --------------------------------------------------------------------------- #

def _symbol(stype, alignment=0, dtype=Datatype.F32):
    return Symbol('x', stype, SimpleNamespace(alignment=alignment,
                                              datatype=dtype))


@pytest.mark.parametrize('stype,alignment,dtype,offset,width,claim', [
    # One element is no vector, and asks for nothing.
    (SymbolType.Global, 16, Datatype.F32, 0, 1, None),
    # The promise, as far as the offset keeps it.
    (SymbolType.Global, 16, Datatype.F32, 0, 4, 16),
    (SymbolType.Global, 16, Datatype.F32, 4, 4, 16),
    (SymbolType.Global, 16, Datatype.F32, 2, 2, 8),
    (SymbolType.Global, 16, Datatype.F64, 0, 2, 16),
    # Less than the access: the element-aligned spelling.
    (SymbolType.Global, 16, Datatype.F32, 2, 4, RELAXED),
    (SymbolType.Global, 16, Datatype.F64, 1, 2, RELAXED),
    (SymbolType.Global, 8, Datatype.F64, 0, 2, RELAXED),
    (SymbolType.Global, 0, Datatype.F32, 0, 4, RELAXED),
    (SymbolType.Global, 16, Datatype.F32, 'k', 4, RELAXED),
    # The image starts on sixteen bytes.
    (SymbolType.SharedMem, 0, Datatype.F32, 0, 4, 16),
    (SymbolType.SharedMem, 0, Datatype.F32, 3, 4, RELAXED),
    # A private address names no contiguous bytes at all.
    (SymbolType.Register, 16, Datatype.F32, 0, 4, RELAXED),
])
def test_what_an_access_claims(stype, alignment, dtype, offset, width, claim):
    symbol = _symbol(stype, alignment, dtype)
    assert GlbToShrLoader._hop_claim(symbol, offset, width) == claim


# --------------------------------------------------------------------------- #
# The copies of a build
# --------------------------------------------------------------------------- #

def _generate(descr, arch, backend, monkeypatch, attrs=None, dtype=None,
              **options):
    """The kernel, and the bodies it was emitted from."""
    bodies = []
    emit = pir.emit

    def keep(body, writer, context=None, metrics=None):
        bodies.append(body)
        return emit(body, writer, context, metrics)

    monkeypatch.setattr(pir, 'emit', keep)
    ctx = Context(arch=arch, backend=backend, fp_type=dtype,
                  options=Options(**options))
    gen = Generator(descr, ctx, attrs=attrs)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        gen.generate()
    return gen.get_kernel(), bodies


def _case(stem, arch, backend, monkeypatch):
    path = next(CASES.rglob(f'{stem}.py'))
    spec = importlib.util.spec_from_file_location('tf_claims__' + stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return _generate(mod.descr_list(), arch, backend, monkeypatch,
                     attrs=getattr(mod, 'ATTRS', None),
                     dtype=getattr(mod, 'DTYPE', None))


def _wide(bodies):
    """The wide accesses of the bodies: a load's value, a store's."""
    found = []
    for body in bodies:
        for s in walk_stmts(body):
            carrier = (s.target[:1] if s.op == Op.LOAD
                       else s.args[1:2] if s.op == Op.STORE else ())
            if any(getattr(getattr(v, 'type', None), 'length', None)
                   for v in carrier):
                found.append(s)
    return found


@pytest.mark.parametrize('stem,arch,backend', [
    ('local_flux', 'gfx942', 'hip'), ('known_zero_tiles', 'gfx942', 'hip'),
    ('local_flux', 'pvc', 'esimd'), ('aligned_operands', 'pvc', 'esimd')])
def test_the_wide_accesses_of_a_copy_carry_their_claims(stem, arch, backend,
                                                        monkeypatch):
    """`preload_globals` on gfx942, and every staged operand under ESIMD,
    are copied with ordinary loads and stores."""
    _, bodies = _case(stem, arch, backend, monkeypatch)
    wide = _wide(bodies)
    assert wide, 'no wide access: nothing to check'
    assert all(s.attr('align') is not None for s in wide)
    found = [d for body in bodies for d in verify(body, strict=False)
             if 'alignment' in d]
    assert found == []


def _operator(alignment):
    """`C = A @ B` with `A` an operator every element multiplies, 64 x 64:
    copied into shared memory ahead of the loop on gfx942, four floats an
    access by the 256 threads of a block."""
    def tensor(shape, alias, addressing):
        return SubTensor(Tensor(shape, addressing,
                                BoundingBox([0, 0], list(shape)),
                                alias=alias, datatype=Datatype.F32,
                                alignment=alignment))
    return [GemmDescr(False, False, tensor([64, 64], 'A', Addressing.NONE),
                      tensor([64, 8], 'B', Addressing.STRIDED),
                      tensor([64, 8], 'C', Addressing.STRIDED),
                      alpha=1.0, beta=0.0)]


_CAST = re.compile(r'\(tensorforge::(Vector\w*T)<float, 4>\*\)&(\w+)\[')


@pytest.mark.parametrize('alignment,spelling', [(16, 'VectorT'),
                                                (0, 'VectorRelaxedT')])
def test_an_operand_is_read_as_its_promise_allows(alignment, spelling,
                                                  monkeypatch):
    """Sixteen bytes promised: the aligned vector type.  Nothing promised:
    the relaxed one, still four floats an access.  The image is written
    through the aligned type either way."""
    src, _ = _generate(_operator(alignment), 'gfx942', 'hip', monkeypatch,
                       dtype=Datatype.F32)
    casts = _CAST.findall(src)
    reads = {kind for kind, base in casts if base.startswith('ptr_glb')}
    writes = {kind for kind, base in casts if not base.startswith('ptr_glb')}
    assert reads == {spelling}, casts
    assert writes == {'VectorT'}, casts


@pytest.mark.parametrize('alignment', [16, 0])
def test_a_lowering_without_the_relaxed_type_claims_what_is_proved(alignment,
                                                                   monkeypatch):
    """`sycl::vec` has no twin at element alignment: its lexic spells the
    aligned type whatever it is asked for (`Target.relaxed_vectors`).  So on
    acpp/gfx942 the promise bounds the width -- four floats an access where
    sixteen bytes are promised, one where nothing is -- and no access claims
    `relaxed`."""
    # Unprepared: what bounds the width here is the promise the operand's
    # own layout makes, and a prepared one is read out of a buffer the host
    # writes for us instead -- aligned by agreement, whatever the layout it
    # replaces promised (`_offer_simt_order`).
    src, bodies = _generate(_operator(alignment), 'gfx942', 'acpp',
                            monkeypatch, dtype=Datatype.F32,
                            prepare_operands=False)
    assert all(s.attr('align') not in (None, RELAXED) for s in _wide(bodies))
    assert ('*(sycl::vec<float, 4>*)&' in src) == (alignment == 16)
