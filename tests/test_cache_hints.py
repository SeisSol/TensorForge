# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Which global transfers take a cache hint, and which one (`Options.cache_hints`,
`Options.hint_outputs`).

A hint is a cache policy and never a value, so these hold the spelling and
who takes it; whether a kind pays is a measurement on hardware.  SeisSol's
order-4 elastic kernels on sm_90: `volume` reads its element's DOFs once and
accumulates into `Q`, `localFluxAll` adds four faces into `Q`, and the Taylor
expansion writes `I` without reading it.
"""

from __future__ import annotations

import contextlib
import io
import re

import pytest

from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context, Options
from tensorforge.generators.generator import Generator


def _kernel(kernel, **options):
    import seissol_suite as fx
    from tensorforge.frontend.yateto import DescriptionReader
    system, config = "elastic-linearck", "elastic-linearck-o4-d"
    descrs = DescriptionReader(None, {}).read(
        fx.description(system, config, f"gpu_{kernel}"))[0]
    ctx = Context(arch="sm_90", backend="cuda", fp_type=Datatype.F64,
                  options=Options(**options))
    gen = Generator(descrs, ctx)
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    return gen.get_kernel()


def _hints(src):
    return {name: len(re.findall(rf"\b__{name}\(", src))
            for name in ("ldcg", "ldcs", "stcg", "stcs")}


@pytest.mark.parametrize("kernel", ["volume", "localFluxAll",
                                    "derivativeTaylorExpansion"])
def test_the_kind_is_the_spelling_and_the_rule_is_unchanged(kernel):
    """`cs` spells every hinted access `__ldcs`/`__stcs` where `cg` spells
    `__ldcg`/`__stcg`, and `none` drops them all: the same accesses, told
    differently what to keep."""
    cg = _hints(_kernel(kernel))
    cs = _hints(_kernel(kernel, cache_hints="cs"))
    none = _hints(_kernel(kernel, cache_hints="none"))
    assert cg["ldcg"] > 0 and cg["ldcs"] == 0
    assert cs == {"ldcg": 0, "ldcs": cg["ldcg"], "stcg": 0, "stcs": cg["stcg"]}
    assert not any(none.values())


def test_the_outputs_take_the_hint_only_when_asked():
    """An accumulated register image always has other users, so by default no
    output store takes the hint -- and `+=` reads its destination plainly.
    With `hint_outputs` both do, in the kind asked for."""
    plain = _hints(_kernel("localFluxAll"))
    out = _hints(_kernel("localFluxAll", hint_outputs=True))
    streamed = _hints(_kernel("localFluxAll", hint_outputs=True,
                              cache_hints="cs"))
    assert plain["stcg"] == 0
    assert out["stcg"] > 0 and out["ldcg"] > plain["ldcg"]
    assert streamed == {"ldcg": 0, "ldcs": out["ldcg"],
                        "stcg": 0, "stcs": out["stcg"]}


def test_an_unknown_kind_is_refused():
    with pytest.raises(ValueError, match="cache_hints"):
        _kernel("derivativeTaylorExpansion", cache_hints="evict_last")


def _read_twice(transposed, **options):
    """`D = A B` and `E = A C`, or `E = A^T C`: two products reading `A`."""
    from tensorforge.common.basic_types import Addressing
    from tensorforge.common.matrix.boundingbox import BoundingBox
    from tensorforge.common.matrix.tensor import SubTensor, Tensor
    from tensorforge.generators.descriptions import MultilinearDescr

    def t(shape, alias):
        return Tensor(shape, Addressing.STRIDED,
                      BoundingBox([0] * len(shape), list(shape)),
                      alias=alias, datatype=Datatype.F64)
    a = t([16, 16], 'A')
    target = [[0, -1], [-1, 1]]
    descrs = [MultilinearDescr(dest=SubTensor(t([16, 6], 'D')),
                               ops=[SubTensor(a), SubTensor(t([16, 6], 'B'))],
                               target=target, permute=[[0, 1], [0, 1]]),
              MultilinearDescr(dest=SubTensor(t([16, 6], 'E')),
                               ops=[SubTensor(a), SubTensor(t([16, 6], 'C'))],
                               target=target,
                               permute=[[1, 0] if transposed else [0, 1],
                                        [0, 1]])]
    gen = Generator(descrs, Context(arch="sm_90", backend="cuda",
                                    fp_type=Datatype.F64,
                                    options=Options(**options)))
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    return gen.get_kernel()


def test_copies_of_one_box_are_one_read(monkeypatch):
    """Each product stages the operand it reads (`memory_semantics`): two
    copies of one box of `A` are one read, and take the hint
    (`load._copies`).  Read the other
    way round by the second product, it is staged into shared memory there
    and into registers by the first -- two different copies, and neither
    takes it."""
    from tensorforge.backend.instructions.memory import load
    asked = []
    real = load._hint_allowed

    def spy(loader):
        out = real(loader)
        asked.append((loader._src, out))
        return out
    monkeypatch.setattr(load, '_hint_allowed', spy)

    def shared(transposed):
        asked.clear()
        _read_twice(transposed, memory_semantics=True)
        twice = {id(src) for src, _ in asked
                 if len(src.get_user_list()) > 1}
        assert twice, asked
        return {out for src, out in asked if id(src) in twice}
    assert shared(transposed=False) == {True}
    assert shared(transposed=True) == {False}
