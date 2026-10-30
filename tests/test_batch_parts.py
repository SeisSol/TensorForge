# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A resident grid over contiguous parts of the batch (`batch_parts`,
`resident_grid`).

On a Data Center GPU Max 1550 driven as one device, implicit scaling gives
the first half of the work-groups and the first half of every allocation to
one stack, the second halves to the other.  The grid stride over the whole
batch reads the other stack's memory every second round, 3.4 to 3.9x slower
than a grid that covers the batch in one round; cut into two parts, each
traversed by its own half of the grid, a looping grid reads its own stack's
memory only.  These tests pin that every element is visited exactly once
however the grid and the parts fall, on the host oracle against the grid
stride, and how the launcher sizes a resident grid.
"""
from __future__ import annotations

import contextlib
import io
import re

import pytest

from tensorforge.common.basic_types import Addressing, Datatype
from tensorforge.common.context import Context
from tensorforge.common.exceptions import GenerationError
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import SubTensor, Tensor
from tensorforge.common.options import Options
from tensorforge.generators.descriptions import GemmDescr
from tensorforge.generators.generator import Generator
from tensorforge.reference import kernel_eval as ke

DT = Datatype.F32


def _descr_list():
    """`C += A B`, every operand the element's own: an element visited twice
    adds its product twice, and one skipped keeps what memory held."""
    def tensor(rows, cols, alias):
        return SubTensor(Tensor([rows, cols], Addressing.PTR_BASED,
                                BoundingBox([0, 0], [rows, cols]),
                                alias=alias, datatype=DT))
    return [GemmDescr(False, False, a=tensor(24, 9, "A"), b=tensor(9, 9, "B"),
                      c=tensor(24, 9, "C"), alpha=1.0, beta=1.0)]


def _generate(arch="sm_120", backend="cuda", **options):
    ctx = Context(arch=arch, backend=backend, fp_type=DT,
                  options=Options(**options))
    gen = Generator(_descr_list(), ctx)
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    return gen


def _written(gen, grid, elements, masked=()):
    """What a launch over `grid` blocks leaves in the outputs, on the host
    oracle; `masked` elements have their flag cleared."""
    src = gen.get_kernel()
    lanes, mults = ke.launch_geometry(gen.get_launcher())
    flags = [0 if e in masked else 1 for e in range(elements)]
    mem = ke.evaluate_wave(src, lanes, seed=5, globals_only=True, mults=mults,
                           block=True, elements=elements, grid=grid,
                           preset={"flags0": flags})
    signature = re.search(r"\bkernel_\w+\(([^)]*)\)", src).group(1)
    outputs = {p.split()[-1] for p in signature.split(",")
               if "*" in p and not p.strip().startswith("const")
               and "<const " not in p}
    # An element's own operand is a buffer per element, `m0@3`.
    return {k: v for k, v in mem.items() if k[0].split("@")[0] in outputs}


def _rows(gen):
    return ke.launch_geometry(gen.get_launcher())[1]


@pytest.mark.parametrize("parts,grid", [(2, 3), (2, 2), (0, 3), (3, 4),
                                        (2, 1)])
def test_every_element_is_computed_once_whatever_the_parts(parts, grid):
    """Against the grid stride of one block, which visits the whole batch:
    a part that ends inside a round, a last part taking the rest of the
    grid, fewer work-groups than parts, and masked elements among them."""
    plain = _generate()
    parted = _generate(batch_parts=parts)
    assert "partEnd" in parted.get_kernel()
    rows = _rows(plain)
    assert _rows(parted) == rows
    elements = 3 * rows + 2
    masked = (1, rows + 2)
    expected = _written(plain, 1, elements, masked)
    assert expected
    assert _written(parted, grid, elements, masked) == expected


def test_the_grid_stride_is_what_one_part_still_gets():
    gen = _generate(batch_parts=1)
    assert gen.get_kernel() == _generate().get_kernel()
    assert "partEnd" not in gen.get_kernel()


def test_a_negative_count_of_parts_is_refused():
    with pytest.raises(GenerationError, match="batch_parts=-1"):
        _generate(batch_parts=-1)


def test_the_loop_strides_over_its_part_and_ends_with_it():
    """The traversal reads its stride and its bound off the part, its
    lookahead clamps to the part's end, and a part is a whole number of
    rounds of the rows."""
    src = _generate(batch_parts=2).get_kernel()
    loop = re.search(r"for \(size_t (\w+) = (\w+); \1 < (\w+); \1 \+= (\w+)\)",
                     src)
    assert loop, src
    assert loop.group(2).endswith("batchId_start")
    assert loop.group(3).endswith("partEnd")
    assert loop.group(4).endswith("partStride")
    assert re.search(r"_ahead1 < \w+_partEnd", src)
    assert re.search(r"_partSize = .*\* blockDim\.y;", src)


def test_a_resident_grid_is_sized_by_what_the_device_holds():
    """SYCL covers the batch in one round unless asked; asked, the grid is
    the option times what the device holds, and never more than the batch."""
    default = _generate(arch="pvc", backend="oneapi").get_launcher()
    assert "max_compute_units" not in default
    resident = _generate(arch="pvc", backend="oneapi",
                         resident_grid=2.0).get_launcher()
    assert "max_compute_units" in resident
    assert re.search(r"config\.grid\[0\] = std::min<std::size_t>\("
                     r"std::max<std::size_t>\(1, static_cast<std::size_t>\("
                     r"2\.0 \* gridsize\)\), numElements0\);", resident)


def test_a_resident_grid_scales_the_occupancy_grid_where_there_is_one():
    default = _generate().get_launcher()
    assert "config.grid[0] = std::min(gridsize, numElements0);" in default
    half = _generate(resident_grid=0.5).get_launcher()
    assert "static_cast<std::size_t>(0.5 * gridsize)" in half


def test_sycl_cuts_a_resident_grid_in_two_by_default():
    """Two parts are what a two-stack GPU needs, and cost nothing on one
    stack; a grid that covers the batch in one round is left as it is,
    since every row makes one trip whatever the parts."""
    one_round = _generate(arch="pvc", backend="oneapi").get_kernel()
    assert "partEnd" not in one_round
    resident = _generate(arch="pvc", backend="oneapi",
                         resident_grid=2.0).get_kernel()
    assert "partEnd" in resident
    assert re.search(r"\? 2u : 1u", resident)
    striding = _generate(arch="pvc", backend="oneapi", resident_grid=2.0,
                         batch_parts=1).get_kernel()
    assert "partEnd" not in striding


def test_a_grid_that_covers_the_batch_in_one_round_takes_no_parts():
    """Asked for parts on SYCL without a resident grid: one round, no parts."""
    src = _generate(arch="pvc", backend="oneapi", batch_parts=2).get_kernel()
    assert "partEnd" not in src


@pytest.mark.parametrize("arch,backend,expected", [
    ("pvc", "oneapi", False),
    ("pvc", "esimd", True),
    ("sm_120", "cuda", True),
    ("gfx942", "hip", True),
])
def test_moving_and_wrapping_are_off_under_intels_spmd_lowering(
        arch, backend, expected):
    """What they move spills under SPMD on PVC, and the grid there covers
    the batch in one round; the explicit vector is not measured and keeps
    both."""
    resolved = Options().resolve(Context(arch=arch, backend=backend,
                                         fp_type=DT).target)
    assert resolved.enable_move_loads is expected
    assert resolved.enable_wrap_loads is expected
    src = _generate(arch=arch, backend=backend).get_kernel()
    assert ("wrap_glb_" in src) is expected
