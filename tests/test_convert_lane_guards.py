# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""A lane guard lowered to selects under SPMD as well
(`Options.convert_lane_guards`).

The pass is the explicit-vector lowering's own (`passes.if_convert`); what
is checked here is that the option reaches it -- from a target's
preferences through both callers of `pir.optimize` -- and that nothing else
changes: the guard of a statement that writes stays a branch, and a target
that does not ask keeps its guards whole.  Whether the Intel device compiler
then computes SeisSol's time derivative correctly is a question for the
device (`~/tf-probe/pvc/diffcfg.py` against a CUDA build).
"""

from __future__ import annotations

import pytest

from tensorforge.backend.pir import pipeline
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context
from tensorforge.common.options import Options


def _if_convert(convert_guards: bool, explicit_simd: bool = False) -> bool:
    pm = pipeline.standard_pipeline('', None, None, None, None, None, None,
                                    None, None, None, None, None, None)
    pc = pipeline.BodyContext((), explicit_simd=explicit_simd,
                              convert_guards=convert_guards)
    rewrite = next(p for p in pm.passes if getattr(p, 'name', '') == 'if_convert')
    return rewrite.enabled(pc)


def test_the_pass_runs_where_asked_and_under_the_explicit_vector():
    assert not _if_convert(False)
    assert _if_convert(True)
    assert _if_convert(False, explicit_simd=True)


@pytest.mark.parametrize('arch,backend,expected', [
    ('pvc', 'oneapi', True),
    ('sm_90', 'cuda', False),
    ('gfx942', 'hip', False),
])
def test_only_intel_asks(arch, backend, expected):
    ctx = Context(arch=arch, backend=backend, fp_type=Datatype.F32)
    assert ctx.get_user_options().convert_lane_guards is expected


def _derivative_source(convert: bool) -> str:
    """The order-4 time derivative of SeisSol's elastic system on PVC, where
    the miscompile was found."""
    import seissol_suite as fx
    from tensorforge.frontend.yateto import DescriptionReader
    from tensorforge.generators.generator import Generator
    descrs = DescriptionReader(None, {}).read(fx.description(
        'elastic-linearck', 'elastic-linearck-o4-d', 'gpu_derivative'))[0]
    ctx = Context(arch='pvc', backend='oneapi', fp_type=Datatype.F64,
                  options=Options(convert_lane_guards=convert))
    gen = Generator(descrs, ctx)
    gen.generate()
    return gen.get_kernel()


def test_the_derivative_keeps_branches_only_around_what_writes():
    """Converted, no guard encloses a loop any more: the guard moves into
    it, onto the one statement of each trip that writes, and the arithmetic
    and the reads in front of it run on every lane."""
    import re
    plain, converted = _derivative_source(False), _derivative_source(True)
    around_loop = re.compile(r'if \(v\d+_g\) \{\s*(?:int32_t [^;]+;\s*)?'
                             r'#pragma unroll\s*for \(')
    assert around_loop.search(plain)
    assert not around_loop.search(converted)
