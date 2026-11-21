# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""SeisSol's local step as one kernel (`tools/bench/suites/seissolfused.py`),
and the two fission experiments it brought: cuts at the boundaries between
operator sets (`Options.fission_cut='operators'`) and staging what each piece
reads (`Options.preload_by_section`).

The combined description is the three kernels' operations one after the
other, each kernel's temporaries and constants its own, the operands they
share one tensor; the partition `kernels` cuts it back into the launches
SeisSol makes.  A list cut flop-balanced puts a cut where the arithmetic
splits evenly, inside the time derivative of the local step; cut by operator
sets it puts them where the derivative's operators end and the volume's
begin.  Staged by section, a piece of a kernel cut in two stages its own
operators and not the other piece's, and the launch takes the larger of the
two rather than both.
"""

import sys
from pathlib import Path

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.context import Context
from tensorforge.common.helper import generate_tmp_matrix
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.common.options import Options
from tensorforge.generators.descriptions import BlockBarrierDescr, GemmDescr
from tensorforge.generators.generator import Generator, fission_cuts

BENCH = Path(__file__).resolve().parents[1] / 'tools' / 'bench'
for extra in (BENCH, BENCH / 'suites'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import seissolfused  # noqa: E402
import suite  # noqa: E402

SYSTEM, CONFIG = 'elastic-linearck', 'elastic-linearck-o4-s'


# --------------------------------------------------------------------------- #
# The combined description
# --------------------------------------------------------------------------- #

def test_the_combined_description_names_its_kernels():
    d = seissolfused.combine(SYSTEM, CONFIG)
    assert d['parts'] == [['gpu_derivative', 23], ['gpu_volume', 6],
                          ['gpu_localFluxAll', 12]]
    assert len(d['operations']) == 41
    names = [t['name'] for t in d['tensors']]
    assert len(names) == len(set(names))
    # Every kernel's temporaries and constants are its own ...
    own = [n for n in names if n.startswith('_')]
    assert own and all(n.startswith(('_k0_', '_k1_', '_k2_')) for n in own)
    # ... and what they share is one tensor.
    assert names.count('I') == names.count('Q') == names.count('star(0)') == 1


def test_without_the_derivatives_they_are_temporaries():
    d = seissolfused.combine(SYSTEM, CONFIG, keep_derivatives=False)
    for t in d['tensors']:
        if seissolfused.DERIVATIVE.match(t['name']):
            assert t['flags']['temporary'] and t['addressing'] == 'n*N+o&'
        elif t['name'] in ('I', 'Q'):
            assert not t['flags']['temporary']


def test_the_partition_cuts_it_into_the_three_launches():
    d = seissolfused.combine(SYSTEM, CONFIG)
    launches = suite.kernels_partition(d)
    assert [len(launch.operations) for launch in launches] == [23, 6, 12]
    pieces = suite.split_description(d, launches)
    # What one launch writes and the next reads is an operand, so nothing
    # becomes a global temporary: every piece keeps its own temporaries.
    for piece, kernel in zip(pieces, seissolfused.KERNELS):
        original = seissolfused.seissol_suite.description(SYSTEM, CONFIG,
                                                           kernel)
        assert len(piece['operations']) == len(original['operations'])
        assert (sum(t['flags']['temporary'] for t in piece['tensors'])
                == sum(t['flags']['temporary'] for t in original['tensors']))


def test_a_description_without_parts_is_one_launch():
    d = seissolfused.seissol_suite.description(SYSTEM, CONFIG, 'gpu_volume')
    assert len(suite.kernels_partition(d)) == 1


# --------------------------------------------------------------------------- #
# Cuts at operator boundaries, staging by section
# --------------------------------------------------------------------------- #

DTYPE = Datatype.F32


def _t(rows, cols, alias, addressing=Addressing.STRIDED):
    return SubTensor(Tensor([rows, cols], addressing,
                            BoundingBox([0, 0], [rows, cols]),
                            alias=alias, datatype=DTYPE))


def _two_operators(heavy: int = 3):
    """Two halves, each a chain of products with an operator of its own:
    `X = A (B C)` repeated `heavy` times, then `Z = D (B C)` once.  The
    first half holds most of the arithmetic, so the flop-balanced cut falls
    inside it; the boundary between the operators is after it."""
    a = _t(32, 32, 'A', Addressing.NONE)
    d = _t(32, 32, 'D', Addressing.NONE)
    b, c = _t(32, 9, 'B'), _t(9, 9, 'C')
    descrs = []
    for n in range(heavy):
        x = _t(32, 9, f'X{n}')
        t = SubTensor(generate_tmp_matrix(b, c))
        descrs += [GemmDescr(trans_a=False, trans_b=False, a=b, b=c, c=t),
                   GemmDescr(trans_a=False, trans_b=False, a=a, b=t, c=x,
                             alpha=1.0, beta=0.0)]
    z = _t(32, 9, 'Z')
    t = SubTensor(generate_tmp_matrix(b, c))
    descrs += [GemmDescr(trans_a=False, trans_b=False, a=b, b=c, c=t),
               GemmDescr(trans_a=False, trans_b=False, a=d, b=t, c=z,
                         alpha=1.0, beta=0.0)]
    return descrs


def _cut(arch, backend, **options):
    """The generator, built, and the entries of its list each cut stands
    ahead of."""
    gen = Generator(_two_operators(), Context(
        arch=arch, backend=backend, fp_type=DTYPE,
        options=Options(merge_variants=False, **options)))
    gen.generate()
    built = gen.built
    at, entries = [], 0
    for d in built._cut_for_fission(built._descr_list):
        if isinstance(d, BlockBarrierDescr):
            at.append(entries)
        else:
            entries += 1
    return gen, at


def test_flop_balanced_cuts_inside_the_heavy_half():
    descrs = _two_operators()
    assert fission_cuts(descrs) == [2, 4, 6]
    gen, at = _cut('sm_90', 'cuda', fission=2)
    assert len(gen.built._sections) == 2
    assert at == [4]          # after two of the three `A` products


def test_operator_cuts_fall_between_the_operators():
    gen, at = _cut('sm_90', 'cuda', fission=2, fission_cut='operators')
    assert len(gen.built._sections) == 2
    assert at == [6]          # after the last `A` product, ahead of `D`


def _launch_shared(**options):
    gen = Generator(_two_operators(), Context(
        arch='gfx942', backend='hip', fp_type=DTYPE,
        options=Options(fission=2, fission_cut='operators',
                        merge_variants=False, **options)))
    gen.generate()
    return gen.built._launch.shared_bytes


def test_each_piece_stages_its_own_operators():
    """On AMD, where the operators are staged by default: every piece of
    the kernel stages both `A` and `D` -- the first does not read `D`, the
    second not `A` -- unless each stages what it reads."""
    whole = _launch_shared()
    by_section = _launch_shared(preload_by_section=True)
    one = 32 * 32 * DTYPE.size()
    assert whole >= 2 * one
    assert by_section < whole
