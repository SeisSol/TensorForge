# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Results handed between the operations of a list through memory alone
(`memory_semantics`), and shared operands staged into registers
(`stage_shared`).

With both, an operation stages every operand from memory and stores its
result where its symbol says; nothing it leaves in registers reaches the
next.  Held here: that the build does so -- the second multiplication of a
chain stages its operand anew -- and that what it computes is what NumPy
does, on the host oracle, for the shapes the canonical lanes are checked on
(`test_canonical_lanes.py`).
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

OPTIONS = dict(memory_semantics=True, stage_shared=True)


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


def test_the_next_operation_stages_what_the_last_one_left_in_registers(
        monkeypatch):
    """`slice_of_a_temporary`: the second multiplication reads the first's
    result.  With the residency it reads the register image the first left;
    with memory semantics it stages the result from where it was stored."""
    held = _stagings(monkeypatch, lanes.slice_of_a_temporary())
    handed = _stagings(monkeypatch, lanes.slice_of_a_temporary(), **OPTIONS)
    assert not held[1] and handed[1]


@pytest.mark.parametrize('shape', lanes.SHAPES, ids=lambda f: f.__name__)
def test_what_memory_semantics_computes_is_what_numpy_does(shape, monkeypatch):
    monkeypatch.setattr(lanes, '_built',
                        lambda descrs, backend='cuda', arch='sm_86':
                        _built(descrs, backend, arch, **OPTIONS))
    assert lanes._worst(shape()) < 1e-9


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


@pytest.mark.parametrize('options', [OPTIONS, dict(memory_semantics=True)],
                         ids=['both', 'memory_semantics'])
def test_an_accumulation_over_fewer_rows_lands_in_the_buffer_as_laid_out(
        options, monkeypatch):
    """The accumulation's result goes to the temporary's shared buffer as
    rows 0..9 of the twenty the first store laid it out over
    (`ResidencyEntry.in_buffer`); laid out over its own ten, every access of
    the buffer would be re-strided to them and `D` would read rows of one
    column in another."""
    monkeypatch.setattr(lanes, '_built',
                        lambda descrs, backend='cuda', arch='sm_86':
                        _built(descrs, backend, arch, **options))
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


@pytest.mark.parametrize('options', [OPTIONS, dict(memory_semantics=True),
                                     {}],
                         ids=['both', 'memory_semantics', 'residency'])
def test_an_accumulation_reads_what_its_destination_held(options,
                                                         monkeypatch):
    """The instruction that adds onto `r` reads it before it writes it, so
    nothing ahead of it says that `r` holds nothing anybody wants
    (`Generator._declare_buffers`, `mark defines`): told so, the allocator
    gives the stretch before it to `t`, which writes over `r` there.

    Without the forwarding: every lane stores `r` alike and would read it
    back from its own register (`pir.forward`), and a buffer nothing reads
    can be written over unnoticed."""
    monkeypatch.setattr(lanes, '_built',
                        lambda descrs, backend='cuda', arch='sm_86':
                        _built(descrs, backend, arch, forward_memory=False,
                               **options))
    assert lanes._worst(accumulation_onto_a_value_without_axes()) < 1e-9


@pytest.mark.parametrize('atomic', [True, False])
def test_a_transfer_keeps_the_cover_the_round_trips_would_take(atomic):
    """Behind every operation the result goes to memory, and the next stages
    it back: round trips the forward of memory takes away (`pir.forward`).
    The cover a transfer is issued with is counted behind that
    (`pir.standard_pipeline`, ``cap``), so `chain_three`'s transfers stand at
    least as far ahead of their first reads as with the residency -- with
    the atomics, which ask the forward of memory as well, and without."""
    import test_issue_cover as cover
    held = cover._gaps(cover._generate('chain_three.py', 'hip', 'gfx942',
                                       atomic_accumulation=atomic))
    handed = cover._gaps(cover._generate('chain_three.py', 'hip', 'gfx942',
                                         atomic_accumulation=atomic,
                                         **OPTIONS))
    assert held and handed.keys() == held.keys()
    assert all(handed[k] >= held[k] for k in held), (held, handed)


def test_the_forward_of_memory_runs_ahead_of_the_cover_without_atomics(
        monkeypatch):
    """The forward of memory runs for the cover's count as much as for the
    atomics: a build that adds nothing atomically has it as well."""
    from tensorforge.backend.instructions import abstract_instruction
    seen = []
    real = abstract_instruction.pir.optimize

    def spy(body, **kwargs):
        seen.append(kwargs.get('early_forward'))
        return real(body, **kwargs)
    monkeypatch.setattr(abstract_instruction.pir, 'optimize', spy)
    _built(lanes.slice_of_a_temporary(), backend='hip', arch='gfx942',
           atomic_accumulation=False, **OPTIONS)
    assert seen and all(early is not None for early in seen)


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
    _built(lanes.window_from_row_one(), backend='hip', arch='gfx942',
           **OPTIONS)
    accumulations = [names for names in built if 'MultilinearEpilogue' in names
                     and names.count('GlbToRegLoader') == 3]
    assert accumulations
    for names in accumulations:
        product = names.index('MultilinearInstruction')
        epilogue = names.index('MultilinearEpilogue')
        assert names[product + 1:epilogue] == ['RegisterAlloc',
                                               'GlbToRegLoader'], names
