# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""One statement of the lane axis, and everyone reads it from there.

`Symbol.lead_dims` says which axis of a symbol is spread across the lanes.
`Symbol.load` and `Symbol.store` index through it for register and scratch
symbols, `GlbToShrLoader` writes a register image with the lane on it, and
`multilinear_builder` sets it to something other than 0 whenever a transposed
operand carries the destination's lead index elsewhere — `lead_index_off_dim0`
is that case, and `test_regressions.py` pins it — and whenever an operand
without the lead index contracts over another dimension than its first, as
`B` does in `trans_b`.

A compute instruction that kept its own copy -- `self._lead_dims = [0]` in its
constructor, or a local `lead_dim = [0]` deciding the destination image's
register slot count while the symbol's own attribute stays at the constructor
default -- would state the axis without reading it.

Copies that all say 0 and none of which reads the others are the arrangement
that produces a wrong answer with no shape check able to notice: an image
written with the lane on axis 0 while every reader addresses it on axis 1
hands each lane an element belonging to another. `load.py` carries a comment
about exactly that failure.

These tests do not check that the answer is 0. They check that the answer
comes from one place, which is the property that survives the day it stops
being 0.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tensorforge.backend.instructions.compute import ComputeInstruction
from tensorforge.backend.instructions.compute.elementwise import \
    ElementwiseInstruction
from tensorforge.backend.instructions.compute.multilinear import \
    MultilinearInstruction
from tensorforge.backend.instructions.compute.reduction import \
    ReductionInstruction
from tensorforge.backend.symbol import Symbol, SymbolType
from tensorforge.common.context import Context
from tensorforge.common.exceptions import InternalError

CASES = Path(__file__).parent / "cases"


def _load_case(rel: str):
    path = CASES / rel
    spec = importlib.util.spec_from_file_location(f"_lead_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _generate(rel: str, backend: str = "cuda", arch: str = "sm_86"):
    from tensorforge.generators.generator import Generator

    mod = _load_case(rel)
    ctx = Context(arch=arch, backend=backend,
                  fp_type=getattr(mod, "DTYPE", None))
    gen = Generator(mod.descr_list(), ctx)
    gen.generate()
    return gen


def _walk(instrs):
    """The same traversal `test_regressions.py` uses, for the same reason:
    instructions nest, and a section's stream is only the outer level."""
    for ins in instrs or []:
        yield ins
        for attr in ("_instructions", "instructions", "_region"):
            sub = getattr(ins, attr, None)
            if isinstance(sub, (list, tuple)):
                yield from _walk(sub)


def _stream(gen):
    section = gen._section
    return list(_walk(list(section.global_ir) + list(section.stream)))


# --- the accessor -------------------------------------------------------- #

class _View:
    def __init__(self, symbol):
        self.symbol = symbol


def _symbol(lead_dims):
    s = Symbol(name="s", stype=SymbolType.Register, obj=None)
    s.lead_dims = list(lead_dims)
    return s


def test_lead_dim_reads_the_symbol():
    assert ComputeInstruction.lead_dim(_View(_symbol([1]))) == 1


def test_lead_dim_takes_a_bare_symbol_too():
    """Multilinear holds the destination symbol; the others hold views."""
    assert ComputeInstruction.lead_dim(_symbol([1])) == 1


@pytest.mark.parametrize("lead_dims", [[], [0, 1]])
def test_lead_dim_refuses_anything_but_one_axis(lead_dims):
    """Two lane axes is not a configuration the loop nest can express.

    Silently taking `lead_dims[0]` would distribute one of them and address
    the other as if it were sequential.
    """
    with pytest.raises(InternalError, match="lead dimension"):
        ComputeInstruction.lead_dim(_View(_symbol(lead_dims)))


def test_operands_that_disagree_are_refused():
    """Elementwise iterates one space over all its operands.

    Iteration axis `i` is axis `i` of each of them, so a lane axis that
    differs between operands means whichever one the loop distributes, the
    others are read on an axis they do not spread.
    """
    class _Instr(ComputeInstruction):
        def get_operands(self):
            return []

        def gen_code_inner(self, writer):
            pass

    with pytest.raises(InternalError, match="disagree"):
        _Instr.shared_lead_dim(_Instr, [_View(_symbol([0])),
                                        _View(_symbol([1]))], "elementwise")


# --- nobody keeps a second copy ------------------------------------------ #

@pytest.mark.parametrize("cls", [ElementwiseInstruction, MultilinearInstruction,
                                 ReductionInstruction],
                         ids=lambda c: c.__name__)
def test_no_instruction_hardcodes_the_lane_axis(cls):
    """No `__init__` states the lane axis as a literal.

    Written against the source because that is where a duplicate would live:
    an instruction can agree with the symbol today and still be stating the
    fact itself, which is the thing this rules out.
    """
    import inspect

    source = inspect.getsource(cls)
    assert "_lead_dims = [0]" not in source, (
        f"{cls.__name__} states the lane axis itself instead of reading it "
        "from the symbol")


@pytest.mark.parametrize("lead,size", [
    # 8 x 40 over 32 lanes: along axis 0 one slot per column, along axis 1
    # two slots per row.
    (0, 1 * 40),
    (1, 8 * 2),
])
def test_a_register_array_states_its_lane_axis(lead, size):
    """Whoever counts the slots also tells the symbol.

    The count is taken from the lane axis, so the answer is known right there,
    and a reader taking `lead_dims` has to get the same axis rather than the
    constructor's default.  Asked of both axes of one box: the allocation and
    the symbol follow the argument together.
    """
    from tensorforge.backend.scopes import Scopes
    from tensorforge.backend.temporaries import Temporaries
    from tensorforge.common.context import Context
    from tensorforge.common.basic_types import Datatype
    from tensorforge.common.matrix.boundingbox import BoundingBox

    context = Context(arch="sm_86", backend="cuda", fp_type=Datatype.F32)
    temporaries = Temporaries(context, Scopes(), 32)
    registers, _ = temporaries.register_array(BoundingBox([0, 0], [8, 40]),
                                              lead)
    assert registers.lead_dims == [lead]
    assert registers.obj.size == size, (
        "the slot count keys on another axis than the one the symbol is "
        "given")


# --- a read the image cannot serve --------------------------------------- #

def test_a_read_one_broadcast_cannot_serve_is_refused():
    """Row 3 of the image is held by lane 3, all of it.

    Read along the lanes, every lane takes its own row; read as one element,
    the lane that holds it broadcasts it.  Read as row 3 spread over the lanes
    -- `B(k, j)` from lane `k`, out of an image with `j` on the lanes -- every
    lane wants a different element of lane 3's registers, and one broadcast
    would hand them all the same one.
    """
    from tensorforge.backend import pir
    from tensorforge.backend.scopes import Scopes
    from tensorforge.backend.symbol import DataView, LeadIndex
    from tensorforge.backend.temporaries import Temporaries
    from tensorforge.common.basic_types import Datatype
    from tensorforge.common.matrix.boundingbox import BoundingBox

    context = Context(arch="gfx90a", backend="hip", fp_type=Datatype.F32)
    box = BoundingBox([0, 0], [16, 16])
    image, _ = Temporaries(context, Scopes(), 16).register_array(box, 0)
    image.data_view = DataView(shape=[16, 16], permute=None, bbox=box)
    builder = pir.IRBuilder(fptype=context.fp_type, context=context)

    assert image.load(builder, context, None, [LeadIndex(0, 16, 1), 3], False)
    assert image.load(builder, context, None, [3, 5], False)
    with pytest.raises(InternalError, match="held by lane 3"):
        image.load(builder, context, None, [3, LeadIndex(0, 16, 1)], False)


# --- end to end ---------------------------------------------------------- #

@pytest.mark.parametrize("backend,arch", [("cuda", "sm_86"), ("hip", "gfx90a")])
def test_a_transposed_operand_still_spreads_dimension_one(backend, arch):
    """The case that makes the attribute matter, read through the accessor.

    `test_regressions.py` checks the loader sets it. This checks that reading
    it back through `ComputeInstruction.lead_dim` gives the same answer, so
    the accessor cannot quietly return 0 for everything and still pass.
    """
    gen = _generate("lead_index_off_dim0.py", backend, arch)

    symbols = []
    for ins in _stream(gen):
        for attr in ("_dest", "_src", "_op"):
            candidate = getattr(ins, attr, None)
            symbol = getattr(candidate, "symbol", candidate)
            if isinstance(symbol, Symbol):
                symbols.append(symbol)

    off_axis = [s for s in symbols if s.lead_dims == [1]]
    assert off_axis, (
        "no symbol spreads dimension 1 in this case any more; "
        + repr(sorted({(s.name, tuple(s.lead_dims)) for s in symbols})))
    for s in off_axis:
        assert ComputeInstruction.lead_dim(s) == 1


@pytest.mark.parametrize("backend,arch", [("hip", "gfx90a"), ("acpp", "pvc")])
def test_an_operand_without_the_lead_index_spreads_its_contraction(backend,
                                                                   arch):
    """`C = A @ B^T`, on targets that stage `B` in registers.

    `B` is stored `(j, k)`: it carries no lead index, and its contraction is
    dimension 1.  The matrix paths read `B(k, j)` from lane `k`, so that is
    the dimension on the lanes.  With dimension 0 there, lane `j` would hold
    all of column `j`, and the read would be the one the test above refuses.
    """
    gen = _generate("trans_b.py", backend, arch)
    images = [ins._dest for ins in _stream(gen)
              if type(ins).__name__ == "GlbToRegLoader"
              and getattr(ins._src.obj, "alias", None) == "B"]
    assert images, "B is not staged in registers on this target any more"
    assert all(image.lead_dims == [1] for image in images), \
        [image.lead_dims for image in images]
