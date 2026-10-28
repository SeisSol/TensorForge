# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A kernel taken apart into one launch per operation
(`suite.split_description`).

The comparison it is for -- what fusing a chain is worth -- is only a
comparison if the pieces compute what the kernel computes.  So what is
checked is that nothing but the temporaries' home changes: every operation
once and in order, every operand declared, the arithmetic summing to the
kernel's own, the bytes growing by what the temporaries now move, and the
pieces building on the targets the paper measures.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / 'tools' / 'bench'
for extra in (str(BENCH), str(BENCH / 'suites')):
    if extra not in sys.path:
        sys.path.insert(0, extra)

import suite  # noqa: E402
from capture import Piece  # noqa: E402
from tensorforge.analysis.cost import list_cost  # noqa: E402
from tensorforge.common.basic_types import Datatype  # noqa: E402
from tensorforge.common.context import Context, Options  # noqa: E402
from tensorforge.generators.generator import Generator  # noqa: E402

EXPORT = BENCH / 'data' / 'elastic-4-d-s1.json'


def _description(kernel):
    return json.loads(EXPORT.read_text())['descriptions'][kernel]


def test_a_piece_names_its_kernel_and_its_place():
    piece = Piece('gpu_localFluxAll', 3, 12)
    assert piece.name == 'gpu_localFluxAll~03-of-12'
    assert Piece.of(piece.name) == piece
    assert Piece.of('gpu_neighboringFlux[0]') is None


def test_each_operation_once_and_in_order_with_its_operands():
    description = _description('gpu_localFluxAll')
    pieces = suite.split_description(description)
    assert [p['operations'] for p in pieces] == [
        [op] for op in description['operations']]
    for piece in pieces:
        declared = {t['name'] for t in piece['tensors']}
        assert set(suite._names_in(piece['operations'])) == declared
        assert not any(t['flags']['temporary'] for t in piece['tensors'])
        assert piece['version'] == description['version']
    # A temporary keeps its strided layout; it only leaves the registers.
    tmp = next(t for t in pieces[0]['tensors'] if t['name'] == '_tmp0')
    assert tmp['addressing'] == 'n*N+o&'
    # The original is not touched.
    assert any(t['flags']['temporary'] for t in description['tensors'])


def test_an_operand_nobody_declared_is_refused():
    description = _description('gpu_volume')
    broken = dict(description, tensors=description['tensors'][1:])
    with pytest.raises(ValueError, match='does not declare'):
        suite.split_description(broken)


@pytest.mark.parametrize('kernel', ['gpu_localFluxAll', 'gpu_derivative'])
def test_the_pieces_do_the_kernels_arithmetic_through_memory(kernel):
    workloads = suite.from_dump(EXPORT, kernel)
    pieces = [w for w in suite.from_dump(EXPORT, split=True)
              if Piece.of(w.name).kernel == kernel]
    assert all(w.capture == workloads[0].capture for w in pieces)
    fused = list_cost(workloads[0].descrs(), batch=1, datatype=Datatype.F64)
    costs = [list_cost(w.descrs(), batch=1, datatype=Datatype.F64)
             for w in pieces]
    assert sum(c.flops for c in costs) == fused.flops
    assert sum(c.nonzero_flops for c in costs) == fused.nonzero_flops
    assert sum(c.bytes for c in costs) > fused.bytes


@pytest.mark.parametrize('arch,backend', [('sm_90', 'cuda'),
                                          ('gfx90a', 'hip')])
def test_the_pieces_build(arch, backend):
    pieces = [w for w in suite.from_dump(EXPORT, 'gpu_volume', split=True)]
    assert [Piece.of(w.name).index for w in pieces] == list(range(6))
    for w in pieces:
        gen = Generator(w.descrs(), Context(arch=arch, backend=backend,
                                            fp_type=Datatype.F64,
                                            options=Options()))
        with contextlib.redirect_stdout(io.StringIO()), \
                warnings.catch_warnings():
            warnings.simplefilter('ignore')
            gen.generate()


def test_the_split_suite_has_each_kernel_whole_and_in_pieces(monkeypatch):
    """Whole and in pieces in one run, the whole under a name of its own so
    that no other reader sees a second measurement of the kernel."""
    monkeypatch.setenv('TF_BENCH_DUMP', str(EXPORT))
    spec = suite.load(BENCH / 'suites' / 'seissolsplit.py')
    names = [w.name for w in spec.workloads]
    import seissolbench
    assert not set(names) & set(seissolbench.KERNELS)
    pieces = [Piece.of(n) for n in names]
    assert all(pieces)
    kernels = sorted({p.kernel for p in pieces})
    exported = json.loads(EXPORT.read_text())['descriptions']
    assert kernels == sorted(set(seissolbench.KERNELS) & set(exported))
    for kernel in kernels:
        count = len(_description(kernel)['operations'])
        assert count > 1
        assert [n for n, p in zip(names, pieces) if p.kernel == kernel] == [
            Piece(kernel, 0, 1).name] + [Piece(kernel, i, count).name
                                         for i in range(count)]
    whole = spec.workloads[names.index(Piece('gpu_volume', 0, 1).name)]
    fused, = suite.from_dump(EXPORT, 'gpu_volume')
    assert list_cost(whole.descrs(), batch=1, datatype=Datatype.F64) == \
        list_cost(fused.descrs(), batch=1, datatype=Datatype.F64)
    assert set(spec.batches) <= set(seissolbench.BATCHES)


def test_only_an_export_splits(tmp_path):
    old = tmp_path / 'old.json'
    old.write_text(json.dumps({'all': {}}))
    with pytest.raises(ValueError, match='only an export'):
        suite.from_dump(old, split=True)
