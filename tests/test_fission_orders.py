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
