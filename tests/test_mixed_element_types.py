# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Values whose element is not the kernel's floating-point type.

A kernel has one floating-point type, the context's, which is what its arena
holds and what it reads by default.  Values of another element still occur:
the double temporary a widening cast writes in a kernel that reads floats --
SeisSol's `toCanonical` in single precision, the time integral of a neighbor
widened to the canonical form of its family and multiplied by a double
operator into a double result -- and the booleans a comparison writes.

What has to hold for them, each checked here on the body the generator emits:

* the double work is done in double: its accumulator, the operator's numbers
  taken out of their vectors, every result it stores into a double buffer
  (`work_type`, the register images of `Temporaries.register_array`);
* it generates on every target, where it failed on gfx90a and gfx942 (the
  matrix path stored the operator in an order chosen for float and declined
  to read it) and did not compile under the explicit vector;
* a wide access never reinterprets one element as another: the verifier
  says so (`passes.verify`), and a wide store of a float image into a
  boolean window is written an element at a time (`IRBuilder.store`).
"""

from __future__ import annotations

import contextlib
import io
from types import SimpleNamespace

import numpy as np
import pytest

from tensorforge.backend.pir.build import IRBuilder
from tensorforge.backend.pir.core import (INDEX, MemSpace, Op, ScalarType,
                                          base_elem, def_use, walk)
from tensorforge.backend.pir.passes import verify
from tensorforge.common.basic_types import Datatype, work_type
from tensorforge.common.context import Context
from tensorforge.common.options import Options
from tensorforge.frontend.yateto import DescriptionReader
from tensorforge.generators.generator import Generator
from tensorforge.generators.lanes import LaneConfig

#: The operator's columns (an order-3 basis), its rows (order 4) and the
#: quantities: `toCanonical` from order 3 to the canonical order 4.
BASES, CANONICAL, QUANTITIES = 10, 20, 9

TARGETS = [('sm_86', 'cuda'), ('sm_100', 'cuda'), ('gfx90a', 'hip'),
           ('gfx942', 'hip'), ('gfx1150', 'hip'), ('gfx1250', 'hip'),
           ('pvc', 'oneapi'), ('pvc', 'esimd')]


def _tensor(name, shape, datatype, temporary=False, values=None):
    addressing = '&' if values is not None else ('n*N+o&' if temporary
                                                 else 'n&+o&')
    return {'name': name, 'addressing': addressing, 'residence': 'memory',
            'datatype': datatype,
            'storage': {'shape': list(shape), 'type': 'bbox',
                        'start': [0] * len(shape), 'sizes': list(shape)},
            'values': values, 'alignment': 64,
            'flags': {'temporary': temporary,
                      'constant': values is not None}}


def _ref(name, indices, shape):
    return {'name': name, 'indices': list(indices),
            'bbox': [[0] * len(shape), list(shape)],
            'offset': [0] * len(shape), 'offset_from': None, 'sliced': False}


def _to_canonical(bases=BASES, canonical=CANONICAL):
    """SeisSol's `toCanonical` in single precision, padded: the integral
    read as floats, widened, and multiplied by a double operator whose
    numbers are not 0 and 1 -- SeisSol's embedding is, and multiplying by it
    is exact in any precision, which is how the float accumulator went
    unnoticed there."""
    rng = np.random.default_rng(7)
    embed = rng.standard_normal((canonical, bases)) / 3.0
    entries = [[[i, j], float(embed[i, j])] for i in range(canonical)
               for j in range(bases)]
    tensors = [
        _tensor('I', (bases, QUANTITIES), 'f32'),
        _tensor('wideI', (bases, QUANTITIES), 'f64', temporary=True),
        _tensor('embed', (canonical, bases), 'f64',
                values={'kind': 'entries', 'data': entries}),
        _tensor('canonicalI', (canonical, QUANTITIES), 'f64')]
    operations = [
        {'type': 'elementwise',
         'result': _ref('wideI', 'kp', (bases, QUANTITIES)),
         'args': [_ref('I', 'kp', (bases, QUANTITIES))],
         'condition': [], 'linear': {'alpha': None, 'add': False},
         'optype': 'Cast<f64>'},
        {'type': 'multilinear',
         'result': _ref('canonicalI', 'kq', (canonical, QUANTITIES)),
         'args': [_ref('embed', 'kl', (canonical, bases)),
                  _ref('wideI', 'lq', (bases, QUANTITIES))],
         'condition': [], 'permute': [[0, 1], [0, 1]],
         'target': [[0, -1], [-1, 1]],
         'linear': {'alpha': None, 'add': False}}]
    descrs, _ = DescriptionReader(None, {}).read(
        {'version': 7, 'tensors': tensors, 'operations': operations})
    return descrs


def _compare_and_select(rows=32, cols=4):
    """A comparison into a boolean temporary and a select over it, whose
    register image -- floats, 1 and 0, by convention -- is stored into the
    temporary's boolean window: SeisSol's `damageStep` in small."""
    shape = (rows, cols)
    tensors = [_tensor('A', shape, 'f32'), _tensor('B', shape, 'f32'),
               _tensor('m', shape, 'bool', temporary=True),
               _tensor('C', shape, 'f32')]
    operations = [
        {'type': 'elementwise', 'result': _ref('m', 'ij', shape),
         'args': [_ref('A', 'ij', shape), _ref('B', 'ij', shape)],
         'condition': [], 'linear': {'alpha': None, 'add': False},
         'optype': 'CmpGt'},
        {'type': 'elementwise', 'result': _ref('C', 'ij', shape),
         'args': [_ref('A', 'ij', shape), _ref('B', 'ij', shape),
                  _ref('m', 'ij', shape)],
         'condition': [], 'linear': {'alpha': None, 'add': False},
         'optype': 'Ternary'}]
    descrs, _ = DescriptionReader(None, {}).read(
        {'version': 7, 'tensors': tensors, 'operations': operations})
    return descrs


def _generated(descrs, arch, backend, fp_type=Datatype.F32, lanes=None,
               **options):
    gen = Generator(descrs, Context(arch=arch, backend=backend,
                                    fp_type=fp_type,
                                    options=Options(**options)),
                    lanes=lanes)
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    return gen


def _bodies(gen):
    return [section.body for section in gen.built._sections]


def _narrowed(body):
    """What of `body` takes a double and keeps a float of it: an element
    taken out of a double vector as a float, arithmetic over a double that
    produces a float, a float stored into a double buffer.  A literal is
    left out: a zero in the kernel's type is the same zero."""
    defs, _ = def_use(body)
    found = []
    for s, _ in walk(body):
        target = s.target[0].type if s.target else None
        doubles = [a for a in s.operands()
                   if isinstance(getattr(a, 'type', None), ScalarType)
                   and a.type.base == Datatype.F64]
        if (s.op == Op.STORE and base_elem(s.args[0]) == Datatype.F64
                and getattr(s.args[1], 'type', None) is not None
                and getattr(s.args[1].type, 'base', None) == Datatype.F32
                and getattr(defs.get(s.args[1].id), 'op', None) != Op.CONST):
            found.append(f'a float stored into {s.args[0]}')
        elif (isinstance(target, ScalarType) and target.base == Datatype.F32
              and doubles and s.op != Op.STORE):
            found.append(f'{s.op} of {doubles[0]!r} into a float')
    return found


@pytest.mark.parametrize('arch,backend', TARGETS)
def test_a_double_product_in_a_float_kernel_is_computed_in_double(arch,
                                                                   backend):
    """Generates, and the double work stays double: the accumulator of the
    product is a double array, and nothing narrows a double on its way to
    the double result.  Before, every target summed it in float, and the
    operator's numbers went in as floats."""
    gen = _generated(_to_canonical(), arch, backend)
    accumulators = []
    for body in _bodies(gen):
        assert not _narrowed(body), _narrowed(body)
        accumulators += [s.target[0].type.elem for s, _ in walk(body)
                         if s.op == Op.ALLOC and s.target
                         and s.target[0].type.space == MemSpace.REGISTER
                         and str(s.attr('extern', '')).startswith('ir')]
    assert accumulators and set(accumulators) == {Datatype.F64}, accumulators


def test_the_double_product_is_declared_double_in_the_source():
    """The same, as the text says it: a double accumulator, no float one,
    and no double operator element read into a float."""
    src = _generated(_to_canonical(), 'sm_86', 'cuda').get_kernel()
    assert 'double ir' in src
    assert 'float ir' not in src
    assert not [line for line in src.splitlines()
                if line.strip().startswith('float v') and '_p = ' in line]


def test_a_double_product_takes_the_double_matrix_instruction():
    """Asked for the matrix path, the product is a double one: `mma.sync` in
    double, not the TF32 halves of a float one that a double was split into
    when the instruction was chosen for the kernel's type.  At order 5 to 6,
    the shape the matrix path takes."""
    src = _generated(_to_canonical(35, 56), 'sm_100', 'cuda',
                     tensor_cores=True).get_kernel()
    assert '.f64.f64.f64.f64' in src
    assert 'tf32' not in src


def test_the_tuner_weighs_the_matrix_path_for_the_product_s_type():
    """The twin on the matrix path is offered where the part's matrix units
    are faster in the type the products are computed in: in single precision
    on sm_100, which a double product is not, and in double on sm_90."""
    from tensorforge.generators import tuning
    first = SimpleNamespace(ok=True)
    for arch, offered in (('sm_100', False), ('sm_90', True)):
        context = Context(arch=arch, backend='cuda', fp_type=Datatype.F32)
        descrs = _to_canonical()
        assert tuning._product_types(descrs, context) == {Datatype.F64}
        twins = tuning._matrix_candidates(
            descrs, first, context, tuning.start(descrs, context), None)
        assert bool(twins) == offered, (arch, twins)


def test_the_work_type_is_the_widest_floating_one():
    assert work_type((Datatype.F32, Datatype.F64), Datatype.F32) \
        == Datatype.F64
    assert work_type((Datatype.F32, Datatype.BOOL), Datatype.F64) \
        == Datatype.F32
    assert work_type((Datatype.BOOL, None), Datatype.F64) == Datatype.F64
    assert work_type((Datatype.F16, Datatype.BF16), Datatype.F64) \
        == Datatype.F32


# --------------------------------------------------------------------------- #
# A wide access through a buffer of another element
# --------------------------------------------------------------------------- #

def _reinterpreted(diagnostics):
    return [d for d in diagnostics if 'reinterprets' in d]


def test_the_verifier_refuses_a_wide_load_of_another_element():
    b = IRBuilder(fptype=Datatype.F32, arena='shrMem')
    buf = b.alloc(Datatype.F64, (8,), MemSpace.SHARED, hint='wide')
    b.store(buf, b.const(1.0, ScalarType(Datatype.F64)), b.const(0, INDEX))
    b.load(buf, b.const(0, INDEX), type_=ScalarType(Datatype.F32, 2),
           align='relaxed', hint='pair')
    assert _reinterpreted(verify(b.finish(), strict=False))


def test_the_verifier_refuses_a_wide_store_of_another_element():
    """Built as the store was before `IRBuilder.store` split it."""
    b = IRBuilder(fptype=Datatype.F32, arena='shrMem')
    flags = b.alloc(Datatype.BOOL, (8,), MemSpace.SHARED, hint='flags')
    pair = b.pack(ScalarType(Datatype.F32, 2), b.const(1.0), b.const(0.0))
    b._emit_op(Op.STORE, (), (flags, pair, b.const(0, INDEX)), pure=False,
               movable=True, attrs=(('align', 'relaxed'),))
    assert _reinterpreted(verify(b.finish(), strict=False))


def test_a_wide_store_into_a_buffer_of_another_element_is_split():
    """One converting store an element, at the next addresses: what the
    wide store meant, which its reinterpret cast did not write."""
    b = IRBuilder(fptype=Datatype.F32, arena='shrMem')
    flags = b.alloc(Datatype.BOOL, (8,), MemSpace.SHARED, hint='flags')
    pair = b.pack(ScalarType(Datatype.F32, 2), b.const(1.0), b.const(0.0))
    b.store(flags, pair, b.const(2, INDEX), align='relaxed')
    body = b.finish()
    stores = [s for s, _ in walk(body) if s.op == Op.STORE]
    assert len(stores) == 2
    assert all(s.args[1].type == ScalarType(Datatype.F32) for s in stores)
    assert not _reinterpreted(verify(body, strict=False))


@pytest.mark.parametrize('arch,backend', [('sm_100', 'cuda'),
                                          ('gfx942', 'hip')])
def test_a_boolean_window_takes_a_float_image_an_element_at_a_time(arch,
                                                                   backend):
    """At a lead width of two the comparison's image is stored a pair at a
    time, and into its boolean window that pair was eight bytes of float
    bits over two booleans -- 1.0f the byte 0x00 where the first one sits.
    SeisSol's `damageStep` at order 3 wrote ten such stores."""
    gen = _generated(_compare_and_select(), arch, backend,
                     lanes=LaneConfig(16, 32, 2))
    stores = []
    for body in _bodies(gen):
        assert not _reinterpreted(verify(body, strict=False))
        stores += [s for s, _ in walk(body) if s.op == Op.STORE
                   and base_elem(s.args[0]) == Datatype.BOOL]
    assert stores
    assert all(getattr(s.args[1].type, 'length', None) is None
               for s in stores)
