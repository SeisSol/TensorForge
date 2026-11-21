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
    """On AMD, where the operators are staged by default: with the old
    staging (`preload_by_section=0`) every piece of the kernel stages both
    `A` and `D` -- the first does not read `D`, the second not `A`; by
    default each stages what it reads, and the launch takes the larger
    piece's shared memory rather than both operators'."""
    whole = _launch_shared(preload_by_section=False)
    by_section = _launch_shared()
    one = 32 * 32 * DTYPE.size()
    assert whole >= 2 * one
    assert by_section < whole


def test_staged_by_section_on_the_host_oracle():
    """The kernel cut at the operator boundary with each piece staging its
    own operator, built for CUDA with the operators staged
    (`preload_globals`, a zero-filling `cp.async`), on the host oracle
    against NumPy: one element, every output."""
    import numpy as np
    from harness.driver_emit import collect_operands
    from tensorforge.reference import kernel_eval as ke
    descrs = _two_operators()
    rng = np.random.default_rng(5)
    numbers = {'A': rng.standard_normal((32, 32)),
               'D': rng.standard_normal((32, 32)),
               'B': rng.standard_normal((32, 9)),
               'C': rng.standard_normal((9, 9))}
    bc = numbers['B'] @ numbers['C']
    want = {f'X{n}': numbers['A'] @ bc for n in range(3)}
    want['Z'] = numbers['D'] @ bc
    gen = Generator(descrs, Context(
        arch='sm_90', backend='cuda', fp_type=DTYPE,
        options=Options(fission=2, fission_cut='operators',
                        preload_globals=True, preload_by_section=True,
                        merge_variants=False)))
    gen.generate()
    assert len(gen.built._sections) == 2
    assert 'memcpy_async' in gen.get_kernel()
    lanes, mults = ke.launch_geometry(gen.get_launcher())
    preset, ops = {}, {}
    for op in collect_operands(gen):
        ops[op.alias] = op
        if op.alias in numbers:
            flat = numbers[op.alias].flatten(order='F')
            smap = op.pack_index
            preset[op.kernel_name] = (list(flat) if smap is None else
                                      [flat[i] if i >= 0 else 0.0
                                       for i in smap])
    mem = ke.evaluate_wave(gen.get_kernel(), lanes, seed=3, preset=preset,
                           races=[], mults=mults, block=True, elements=1)
    for alias, w in want.items():
        name = ops[alias].kernel_name
        got = np.array([mem[(name, i)] for i in range(w.size)]
                       ).reshape(w.shape, order='F')
        assert np.abs(got - w).max() < 1e-3 * np.abs(w).max(), alias


def _banded(n, alias, seed):
    """An `n` by `n` operator with its numbers known and a band of them
    nonzero: what the reading section packs (`pack_operands`)."""
    import numpy as np
    vals = np.zeros((n, n))
    rng = np.random.default_rng(seed)
    for c in range(n):
        hi = min(n - 1, c + 5)
        vals[c:hi + 1, c] = rng.uniform(0.5, 1.5, size=hi + 1 - c)
    return SubTensor(Tensor([n, n], Addressing.NONE, BoundingBox([0, 0], [n, n]),
                            alias=alias, datatype=DTYPE, data=vals))


def _staged_twice(**options):
    """The two-operator chain with banded operators on gfx1150, where the
    operators are staged by default and packed where they are read, cut
    between them: the kernel source, and the binding names of the two
    operators."""
    a, d = _banded(32, 'A', 1), _banded(32, 'D', 2)
    b, c = _t(32, 9, 'B'), _t(9, 9, 'C')
    descrs = []
    for n, op in enumerate((a, a, d)):
        t = SubTensor(generate_tmp_matrix(b, c))
        descrs += [GemmDescr(trans_a=False, trans_b=False, a=b, b=c, c=t),
                   GemmDescr(trans_a=False, trans_b=False, a=op, b=t,
                             c=_t(32, 9, f'X{n}'), alpha=1.0, beta=0.0)]
    gen = Generator(descrs, Context(
        arch='gfx1150', backend='hip', fp_type=DTYPE,
        options=Options(fission=2, fission_cut='operators',
                        merge_variants=False, **options)))
    gen.generate()
    scope = gen.built._scopes.get_global_scope().values()
    names = {s.obj.alias: s.name for s in scope
             if getattr(s.obj, 'alias', None) in ('A', 'D')}
    return gen, gen.get_kernel(), names, d.tensor


def test_a_piece_stages_no_operand_it_does_not_read():
    """The regression: staged in the piece that does not read it, `D` was
    copied as it was stored when that piece was built -- the box -- and the
    piece that reads it packed it after, which is how the host lays it out:
    the copy read past the end of the packed buffer (gfx1150, SeisSol's
    order-6 local step cut in three: a memory access fault).  By default
    each piece stages only what it reads."""
    gen, src, names, d = _staged_twice()
    assert len(gen.built._sections) == 2
    assert src.count(f'ptr_glb_{names["D"]} =') == 1
    assert src.count(f'ptr_glb_{names["A"]} =') == 1
    # The operator the second piece reads is packed: fewer cells stored
    # than its box holds -- which the first piece, built before, cannot know.
    assert d.storage_volume() < 32 * 32


def test_the_old_staging_copies_it_in_both_pieces():
    """The comparison, `preload_by_section=0`: both pieces stage both, and
    the first copies `D` at the cells it held when the first piece was
    built -- more than the packing the second piece then gave it leaves in
    the buffer."""
    from tensorforge.backend.instructions.builders import loader_builder
    copies = []
    real = loader_builder.GlobalLoaderBuilder.build

    def build(self, src):
        copies.append((getattr(src.obj, 'alias', None),
                       int(src.obj.storage_volume())))
        return real(self, src)
    loader_builder.GlobalLoaderBuilder.build = build
    try:
        gen, src, names, d = _staged_twice(preload_by_section=False)
    finally:
        loader_builder.GlobalLoaderBuilder.build = real
    assert src.count(f'ptr_glb_{names["D"]} =') == 2
    staged = [cells for alias, cells in copies if alias == 'D']
    assert max(staged) > d.storage_volume()
