# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""SeisSol's damage-model kernel, as yateto describes it.

`fixtures/kernels/seissol_damage.json` is the description yateto sent: thirteen
material parameters picked out into temporaries without axes, then per
direction a flux built in two terms -- the velocity part assigned, the stress
part accumulated onto the columns the first term leaves out -- and
`Q += kDivM(d) @ flux`.  It stopped at three places, one per target family,
and each is pinned here:

* `rhoInv[] * stressToFlux[c,p]` wrote row 0 of its destination: the axisless
  factor was given the destination's axis 0 (`MultilinearDescr` now gives it a
  contracted one of its own);
* the flux temporaries are read where no term defines them and accumulated
  onto cells the assignment left out -- zero in yateto's meaning, garbage in
  the buffer -- so their first store clears them (`SectionPlan.zero_first`);
* the sparse flux operators left values without a layout, which the ESIMD
  emitter cannot type.

The numbers were checked on hardware against a numpy evaluation of the
description, not here: the host oracle does not model pointer-based operands.
"""

from __future__ import annotations

import contextlib
import io
import json
import pathlib

import pytest

from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context
from tensorforge.frontend.yateto import DescriptionReader
from tensorforge.generators.descriptions import MultilinearDescr
from tensorforge.generators.generator import Generator

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "kernels" / "seissol_damage.json"
TARGETS = [("sm_86", "cuda"), ("gfx942", "hip"), ("gfx1150", "hip"),
           ("pvc", "oneapi"), ("pvc", "esimd")]


def _description():
    return json.loads(FIXTURE.read_text())["descriptions"]["seissol_damage"]


def _generate(arch, backend):
    descrs, _ = DescriptionReader(None, {}).read(_description())
    gen = Generator(descrs, Context(arch=arch, backend=backend,
                                    fp_type=Datatype.F32))
    with contextlib.redirect_stdout(io.StringIO()):
        gen.generate()
    return gen


@pytest.mark.parametrize("arch,backend", TARGETS,
                         ids=[f"{a}-{b}" for a, b in TARGETS])
def test_it_builds_on_every_target(arch, backend):
    assert _generate(arch, backend).get_kernel()


def test_the_axisless_factor_does_not_narrow_its_destination():
    """`_tmp14[c,p] = rhoInv[] * stressToFluxX[c,p]` runs over all of `c`.

    `rhoInv` takes no axis at all; the destination's are the matrix's.
    """
    descrs, _ = DescriptionReader(None, {}).read(_description())
    scaled = [d for d in descrs if isinstance(d, MultilinearDescr)
              and getattr(d.dest.tensor, "alias", None) == "_tmp14"]
    assert scaled and scaled[0].target == [[], [0, 1]]
    assert scaled[0].effective_boxes()[1].sizes() == [6, 3]


@pytest.mark.parametrize("arch,backend", TARGETS,
                         ids=[f"{a}-{b}" for a, b in TARGETS])
def test_each_flux_temporary_is_cleared_by_its_first_store(arch, backend):
    """`_tmp16`, `_tmp21` and `_tmp26`: one clearing store each, no more."""
    src = _generate(arch, backend).get_kernel()
    assert src.count("store{r>s, clear}") == 3
