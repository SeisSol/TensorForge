# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""One statement of the lane axis, and everyone reads it from there.

`Symbol.lead_dims` says which axis of a symbol is spread across the lanes.
`Symbol.load` and `Symbol.store` index through it for register and scratch
symbols, `GlbToShrLoader` writes a register image with the lane on it, and
`multilinear_builder` sets it to something other than 0 whenever a transposed
operand carries the destination's lead index elsewhere — `lead_index_off_dim0`
is that case, and `test_regressions.py` pins it.

Every compute instruction also kept its own copy: `self._lead_dims = [0]`, in
elementwise and in multilinear, plus a third local `lead_dim = [0]` inside
`_alloc_register_array` that decided the destination image's register slot
count while the symbol's own attribute stayed at the constructor default.

All four said 0 and none of them read the others, which is the arrangement
that produces a wrong answer with no shape check able to notice: an image
written with the lane on axis 0 while every reader addresses it on axis 1
hands each lane an element belonging to another. `load.py` carries a comment
about exactly that failure, having already had it once.

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
    """The literal that used to sit in each `__init__`.

    Written against the source because that is where the duplicate lived: an
    instruction can agree with the symbol today and still be stating the fact
    itself, which is the thing being removed.
    """
    import inspect

    source = inspect.getsource(cls)
    assert "_lead_dims = [0]" not in source, (
        f"{cls.__name__} states the lane axis itself instead of reading it "
        "from the symbol")


def test_a_register_array_states_its_lane_axis():
    """Whoever counts the slots also tells the symbol.

    The count is taken from the lane axis, so the answer is known right there;
    the symbol just did not get told, and a reader taking `lead_dims` got the
    constructor's guess instead. Both now come from the one argument
    `Temporaries.register_array` is given, which is what keeps them from
    parting company.
    """
    import inspect

    from tensorforge.backend import temporaries

    source = inspect.getsource(temporaries.Temporaries.register_array)
    assert "registers.lead_dims = [lead_pos]" in source
    assert "d != lead_pos" in source, (
        "the slot count no longer keys on the same lane axis the symbol is "
        "given")


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
