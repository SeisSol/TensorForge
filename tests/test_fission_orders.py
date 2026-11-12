# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""An operand's order across the runs of a cut kernel (`Options.fission`).

The order a matrix product lays over a batch-constant operand is that
product's window and fragments (`MultilinearInstruction._offer_order`), so an
operand another operation reads is offered none.  Cut into runs, a kernel is
built a section at a time, and a reader in another section is not among the
users a section sees: SeisSol's order-8 derivative, cut in two, ordered
`kDivMT(1)` for the `dQ(1)` product of the first run, and the `dQ(2)`
product of the second took its own fragments out of that order -- every
value of `dQ(2)` wrong on sm_100 and sm_120.  Each section is now told which
of its operands another one reads (`Tensor.read_elsewhere`).
"""

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.context import Context
from tensorforge.common.helper import generate_tmp_matrix
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.common.options import Options
from tensorforge.generators.descriptions import GemmDescr
from tensorforge.generators.generator import Generator, sections_reading

DTYPE = Datatype.F64


def _t(rows, cols, alias, addressing=Addressing.STRIDED):
    return SubTensor(Tensor([rows, cols], addressing,
                            BoundingBox([0, 0], [rows, cols]),
                            alias=alias, datatype=DTYPE))


def _chain(second_operator: bool):
    """``X = A (B C)``, then ``D = A' (X C)`` with ``A'`` either ``A`` again
    or an operator of its own; the one cut is between the halves."""
    a = _t(56, 56, "A", Addressing.NONE)
    a2 = _t(56, 56, "A2", Addressing.NONE) if second_operator else a
    b, c = _t(56, 9, "B"), _t(9, 9, "C")
    x, d = _t(56, 9, "X"), _t(56, 9, "D")
    t1 = SubTensor(generate_tmp_matrix(b, c))
    t2 = SubTensor(generate_tmp_matrix(x, c))
    descrs = [
        GemmDescr(trans_a=False, trans_b=False, a=b, b=c, c=t1),
        GemmDescr(trans_a=False, trans_b=False, a=a, b=t1, c=x,
                  alpha=1.0, beta=0.0),
        GemmDescr(trans_a=False, trans_b=False, a=x, b=c, c=t2),
        GemmDescr(trans_a=False, trans_b=False, a=a2, b=t2, c=d,
                  alpha=1.0, beta=0.0),
    ]
    return descrs, a.tensor, a2.tensor


def _build(descrs, **options):
    gen = Generator(descrs, Context(arch="sm_90", backend="cuda",
                                    fp_type=DTYPE,
                                    options=Options(tensor_cores=True,
                                                    **options)))
    gen.generate()
    return gen


def test_an_operator_read_in_both_runs_is_offered_no_order():
    descrs, a, _ = _chain(second_operator=False)
    gen = _build(descrs, fission=2)
    assert len(gen.built._sections) == 2
    assert "mma.sync" in gen.get_kernel()
    assert a.storage_order is None


def test_an_operator_read_in_one_run_still_is():
    """The control: each product with an operator of its own, read in one
    run only, takes the order -- what the first test refuses is the second
    reader, not the cut."""
    descrs, a, a2 = _chain(second_operator=True)
    gen = _build(descrs, fission=2)
    assert len(gen.built._sections) == 2
    assert a.storage_order is not None and a2.storage_order is not None


def test_whole_the_operator_read_twice_takes_none_either():
    descrs, a, _ = _chain(second_operator=False)
    gen = _build(descrs)
    assert len(gen.built._sections) == 1
    assert a.storage_order is None


def test_the_sections_each_operand_is_read_in():
    descrs, a, _ = _chain(second_operator=False)
    reading = sections_reading([descrs[:2], descrs[2:]])
    assert reading[id(a)][1] == {0, 1}


# --------------------------------------------------------------------------- #
# The interleave and the packing across the runs, on the host oracle
# --------------------------------------------------------------------------- #

def _cross(form: str, y_first: bool):
    """``Y = E A`` and ``X = A (B C)``, cut between them in the order given:
    ``A`` the B operand of one run and the A operand of the other.  Dense
    for the interleave (`_offer_simt_order`); a band with its numbers known
    for the packing (`_offer_packing`), which only an operand with numbers
    is offered."""
    import numpy as np
    n = 56
    rng = np.random.default_rng(1)
    if form == 'packing':
        vals = np.zeros((n, n))
        band = np.random.default_rng(7)
        for c in range(n):
            vals[c:min(n - 1, c + 9) + 1, c] = band.uniform(
                0.5, 1.5, size=min(n - 1, c + 9) + 1 - c)
        a = SubTensor(Tensor([n, n], Addressing.NONE,
                             BoundingBox([0, 0], [n, n]),
                             alias="A", datatype=DTYPE, data=vals))
    else:
        vals = rng.standard_normal((n, n))
        a = _t(n, n, "A", Addressing.NONE)
    numbers = {'A': vals, 'B': rng.standard_normal((n, 9)),
               'C': rng.standard_normal((9, 9)),
               'E': rng.standard_normal((9, n))}
    b, c = _t(n, 9, "B"), _t(9, 9, "C")
    x, e, y = _t(n, 9, "X"), _t(9, n, "E"), _t(9, n, "Y")
    t1 = SubTensor(generate_tmp_matrix(b, c))
    xs = [GemmDescr(trans_a=False, trans_b=False, a=b, b=c, c=t1),
          GemmDescr(trans_a=False, trans_b=False, a=a, b=t1, c=x,
                    alpha=1.0, beta=0.0)]
    ys = [GemmDescr(trans_a=False, trans_b=False, a=e, b=a, c=y,
                    alpha=1.0, beta=0.0)]
    want = {'X': vals @ (numbers['B'] @ numbers['C']),
            'Y': numbers['E'] @ vals}
    return (ys + xs if y_first else xs + ys), numbers, want, a.tensor


def _oracle_errors(descrs, numbers, want):
    import numpy as np
    from harness.driver_emit import collect_operands
    from tensorforge.reference import kernel_eval as ke
    gen = Generator(descrs, Context(arch='sm_90', backend='cuda',
                                    fp_type=DTYPE,
                                    options=Options(tensor_cores=False,
                                                    fission=2)))
    gen.generate()
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
    errors = {}
    for alias, w in want.items():
        name = ops[alias].kernel_name
        got = np.array([mem[(name, i)] for i in range(w.size)]
                       ).reshape(w.shape, order='F')
        errors[alias] = float(np.abs(got - w).max())
    return errors


def test_a_run_built_first_reads_an_operand_plain_and_the_next_one_stores_it_so():
    """`Y = E A` built first reads `A` as it lies, so the run after it lays
    neither the interleave nor a packing over `A`: both were offered before,
    and left `Y` wrong by 31 and 63."""
    for form in ('interleave', 'packing'):
        descrs, numbers, want, _ = _cross(form, y_first=True)
        errors = _oracle_errors(descrs, numbers, want)
        assert max(errors.values()) < 1e-9, (form, errors)


def test_a_packing_the_first_run_lays_is_read_by_the_next_one():
    """The other way round the packing stays: `Y` built after `X` reads the
    packed `A` as a second reader in one run does."""
    descrs, numbers, want, a = _cross('packing', y_first=False)
    errors = _oracle_errors(descrs, numbers, want)
    assert max(errors.values()) < 1e-9, errors
    assert a.packing is not None


def test_no_interleave_over_an_operand_another_run_reads():
    """And the interleave stays off: `Y` after `X` could not address it
    (`interleaved rows need a lead index`)."""
    descrs, numbers, want, a = _cross('interleave', y_first=False)
    errors = _oracle_errors(descrs, numbers, want)
    assert max(errors.values()) < 1e-9, errors
    assert a.simt_interleave is None
