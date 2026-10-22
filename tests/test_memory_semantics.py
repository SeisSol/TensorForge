# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Results handed between the operations of a list through memory alone.

An operation stages every operand from memory -- one in shared memory into
registers as well, where its placement says registers -- and stores its
result where its symbol says; nothing it computes in registers reaches the
next.  Held here: that the build does so -- the second multiplication of a
chain stages its operand anew -- that the forward of memory runs ahead of
the cover's count, and that the bias goes out behind the product.  What it
computes is what NumPy does, on the host oracle (`test_canonical_lanes.py`).
"""

from __future__ import annotations

import contextlib
import io

import pytest

import test_canonical_lanes as lanes
from tensorforge.backend.instructions.builders import multilinear_builder
from tensorforge.common.context import Context, Options
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor
from tensorforge.generators.descriptions import MultilinearDescr
from tensorforge.generators.generator import Generator


def _built(descrs, backend='cuda', arch='sm_86', **options):
    gen = Generator(descrs, Context(arch=arch, backend=backend,
                                    fp_type=lanes.F64,
                                    options=Options(**options)))
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    return gen


def _stagings(monkeypatch, descrs, **options):
    """Per multiplication, the operands it staged into registers."""
    staged = []
    real = multilinear_builder.MultilinearBuilder._make_loader_and_symbol_reg

    def spy(self, opview, *args, **kwargs):
        staged[-1].append(opview.symbol.name)
        return real(self, opview, *args, **kwargs)

    real_build = multilinear_builder.MultilinearBuilder.build

    def build(self, descr):
        staged.append([])
        return real_build(self, descr)
    with monkeypatch.context() as m:
        m.setattr(multilinear_builder.MultilinearBuilder,
                  '_make_loader_and_symbol_reg', spy)
        m.setattr(multilinear_builder.MultilinearBuilder, 'build', build)
        _built(descrs, **options)
    return staged


def test_the_next_operation_stages_what_the_last_one_stored(monkeypatch):
    """`slice_of_a_temporary`: the second multiplication reads the first's
    result, and stages it from where the first stored it."""
    staged = _stagings(monkeypatch, lanes.slice_of_a_temporary())
    assert staged[1]


def accumulation_over_fewer_rows():
    """`T = X Y` into a temporary, `T += A Z` with `A` stored over its first
    ten rows only, then `D = T C`: the second product reaches ten of the
    temporary's twenty rows and adds onto what the first stored, and `D`
    reads all twenty."""
    t = lanes._t([lanes.M, lanes.K], 'T', is_tmp=True)
    a = lanes._t([lanes.M, lanes.K], 'A', ([0, 0], [10, lanes.K]))

    def gemm(dest, x, y, **extra):
        return MultilinearDescr(dest=SubTensor(dest),
                                ops=[SubTensor(x), SubTensor(y)],
                                **lanes.GEMM, **extra)
    return [
        gemm(t, lanes._t([lanes.M, lanes.K], 'X'),
             lanes._t([lanes.K, lanes.K], 'Y')),
        gemm(t, a, lanes._t([lanes.K, lanes.K], 'Z'), add=True),
        gemm(lanes._t([lanes.M, lanes.N], 'D'), t,
             lanes._t([lanes.K, lanes.N], 'C')),
    ]


def test_an_accumulation_over_fewer_rows_lands_in_the_buffer_as_laid_out():
    """The accumulation's result goes to the temporary's shared buffer as
    rows 0..9 of the twenty the first store laid it out over
    (`MultilinearBuilder._in_buffer`); laid out over its own ten, every
    access of the buffer would be re-strided to them and `D` would read rows
    of one column in another."""
    assert lanes._worst(accumulation_over_fewer_rows()) < 1e-9


def accumulation_onto_a_value_without_axes():
    """`r = v[2:5] p[2:5]` and `t = x[2:5] y[2:5]`, values without axes,
    `E = t B`, then `r += w[2:5] q[2:5]` and `D = r C`: the accumulation
    onto `r` reads what the first product stored there, and `t` lives in
    between -- SeisSol's damage step adds onto its values without axes this
    way."""
    def vec(alias):
        return lanes._t([lanes.M], alias)

    def val(alias):
        return lanes._t([], alias, is_tmp=True)

    def dot(dest, x, y, **extra):
        window = BoundingBox([2], [5])
        return MultilinearDescr(dest=SubTensor(dest),
                                ops=[SubTensor(x, window),
                                     SubTensor(y, window)],
                                target=[[-1], [-1]], permute=[[0], [0]],
                                **extra)

    def scale(dest, s, m):
        return MultilinearDescr(dest=SubTensor(dest),
                                ops=[SubTensor(s), SubTensor(m)],
                                target=[[], [0, 1]], permute=[[], [0, 1]])
    r, t = val('r'), val('t')
    return [
        dot(r, vec('v'), vec('p')),
        dot(t, vec('x'), vec('y')),
        scale(lanes._t([lanes.M, lanes.N], 'E'), t,
              lanes._t([lanes.M, lanes.N], 'B')),
        dot(r, vec('w'), vec('q'), add=True),
        scale(lanes._t([lanes.M, lanes.N], 'D'), r,
              lanes._t([lanes.M, lanes.N], 'C')),
    ]


def test_an_accumulation_reads_what_its_destination_held(monkeypatch):
    """The instruction that adds onto `r` reads it before it writes it, so
    nothing ahead of it says that `r` holds nothing anybody wants
    (`Generator._declare_buffers`, `mark defines`): told so, the allocator
    gives the stretch before it to `t`, which writes over `r` there.

    Without the forwarding: every lane stores `r` alike and would read it
    back from its own register (`pir.forward`), and a buffer nothing reads
    can be written over unnoticed."""
    monkeypatch.setattr(lanes, '_built',
                        lambda descrs, backend='cuda', arch='sm_86':
                        _built(descrs, backend, arch, forward_memory=False))
    assert lanes._worst(accumulation_onto_a_value_without_axes()) < 1e-9


@pytest.mark.parametrize('atomic', [True, False])
def test_the_cover_is_counted_behind_the_forward_of_memory(atomic,
                                                          monkeypatch):
    """Behind every operation the result goes to memory, and the next stages
    it back: round trips the forward of memory takes away (`pir.forward`).
    The cover a transfer is issued with is counted behind that
    (`pir.standard_pipeline`, ``cap``) -- in the work the emitter emits, and
    none of the round trips -- with the atomics, which ask the forward of
    memory as well, and without."""
    from tensorforge.backend.pir import pipeline
    orders = []
    real = pipeline.standard_pipeline

    def spy(*args, **kwargs):
        pm = real(*args, **kwargs)
        orders.append([p.name for p in pm.passes])
        return pm
    monkeypatch.setattr(pipeline, 'standard_pipeline', spy)
    _built(lanes.window_from_row_one(), backend='hip', arch='gfx942',
           atomic_accumulation=atomic)
    capped = [names for names in orders if 'cap' in names]
    assert capped
    for names in capped:
        assert 'forward-memory' in names, names
        assert names.index('forward-memory') < names.index('cap'), names


def test_the_bias_is_loaded_behind_the_product(monkeypatch):
    """An accumulation stages what it adds onto from memory, and its load
    goes out behind the product, which does not read it, right ahead of the
    epilogue, which does (`MultilinearBuilder.alloc_destination`)."""
    built = []
    real = multilinear_builder.MultilinearBuilder.get_instructions

    def spy(self):
        out = real(self)
        built.append([type(x).__name__ for x in out])
        return out
    monkeypatch.setattr(multilinear_builder.MultilinearBuilder,
                        'get_instructions', spy)
    _built(lanes.window_from_row_one(), backend='hip', arch='gfx942')
    accumulations = [names for names in built if 'MultilinearEpilogue' in names
                     and names.count('GlbToRegLoader') == 3]
    assert accumulations
    for names in accumulations:
        product = names.index('MultilinearInstruction')
        epilogue = names.index('MultilinearEpilogue')
        assert names[product + 1:epilogue] == ['RegisterAlloc',
                                               'GlbToRegLoader'], names
