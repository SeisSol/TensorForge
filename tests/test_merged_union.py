# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A merged run leaves out what none of its members has a number at.

One body runs for every member of a merged run, and reads them through a
stand-in that has no numbers of its own -- so it skipped no zero at all and
computed over the stand-in's whole box.  What it may skip is the union of
its members' patterns, which needs no table: the members are known when the
body is built (`MultilinearInstruction._zero_pattern`).  SeisSol's
`localFluxAll` at order 8 on sm_120: -9 % at 32 lanes, -13 % against +6 % at
16.
"""

from __future__ import annotations

import contextlib
import io
import re
import warnings

import numpy as np

import seissol_suite as suite
from tensorforge.backend.instructions.compute.multilinear import \
    MultilinearInstruction
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context
from tensorforge.frontend.yateto import DescriptionReader
from tensorforge.generators.generator import Generator

SYSTEM = 'elastic-linearck'


def _build(monkeypatch, options):
    monkeypatch.setenv("TF_OPTIONS", options)
    descrs = DescriptionReader(None, {}).read(suite.description(
        SYSTEM, f'{SYSTEM}-o6-s', 'gpu_localFluxAll'))[0]
    gen = Generator(descrs, Context(arch="sm_120", backend="cuda",
                                    fp_type=Datatype.F32))
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        gen.generate()
    return gen


def _stand_ins(gen):
    def walk(instrs):
        for i in instrs:
            yield i
            for r in i.regions():
                yield from walk(r)
    out = []
    for section in gen.built._sections:
        for ins in walk(section.stream):
            if isinstance(ins, MultilinearInstruction) and ins._ops:
                obj = ins._ops[0].symbol.obj
                if getattr(obj, 'is_variant', False) and obj not in out:
                    out.append(obj)
    return out


def test_a_stand_in_is_zero_where_every_member_is(monkeypatch):
    stand_ins = _stand_ins(_build(monkeypatch, "merge_variants=1"))
    assert stand_ins
    for obj in stand_ins:
        union = MultilinearInstruction._zero_pattern(obj)
        assert union is not None
        members = [np.asarray(m.get_values()) for m in obj.variant_members]
        assert ((union != 0) == np.any([m != 0 for m in members],
                                       axis=0)).all()
        # some cells no member has a number at, or there were nothing to skip
        assert not (union != 0).all()


def test_the_merged_body_multiplies_less_than_the_box(monkeypatch):
    """The same build with nothing skipped multiplies over every cell."""
    skipped = _build(monkeypatch, "merge_variants=1").get_kernel()
    whole = _build(monkeypatch,
                   "merge_variants=1,skip_known_zeros=0").get_kernel()
    products = re.compile(r"\(v\d+_\w+ \* v\d+_\w+\)")
    assert len(products.findall(skipped)) < len(products.findall(whole))
