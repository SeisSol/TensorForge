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


def _volume_source(convert: bool, backend: str = 'oneapi',
                   unguard=True) -> str:
    """The order-3 F32 volume kernel of SeisSol's elastic system on PVC, which
    IGC 2.41.5 computed wrong around a guarded register write.  The
    workaround stated, not left to the `ocloc` on PATH."""
    import seissol_suite as fx
    from tensorforge.frontend.yateto import DescriptionReader
    from tensorforge.generators.generator import Generator
    descrs = DescriptionReader(None, {}).read(fx.description(
        'elastic-linearck', 'elastic-linearck-o3-s', 'gpu_volume'))[0]
    ctx = Context(arch='pvc', backend=backend, fp_type=Datatype.F32,
                  options=Options(convert_lane_guards=convert,
                                  unguard_window_loads=unguard))
    gen = Generator(descrs, ctx)
    gen.generate()
    return gen.get_kernel()


def _load_block(src: str, operand: str) -> str:
    """The text of the register load of `operand`, up to the next one."""
    marker = f'= load{{g>r}}({operand});'
    start = src.index(marker)
    end = src.find('load{g>r}', start + len(marker))
    return src[start:end if end > 0 else len(src)]


def test_a_window_loads_into_its_own_image_without_a_guard():
    """`I`'s rows 0..3 and `Q`'s rows 1..7 into the products' own images:
    held to the window by the load, on both sides, and written on every
    lane, no branch around the write -- the form IGC 2.41.5 computes right.
    Without the conversion the guard is a branch around the whole load, as
    before."""
    import re
    converted = _load_block(_volume_source(True), 'glb_m0')
    assert not re.search(r'if \(v\d+_g\)', converted)
    assert re.search(r'r\d+\[v\d+_\w+\] = v\d+_data;', converted)
    assert re.search(r'v\d+_g \? \(v\d+_data_pre\) : \(0\.0f\)', converted)
    shifted = _load_block(_volume_source(True), 'glb_m2')
    assert not re.search(r'if \(v\d+_g\)', shifted.split('// r')[0])
    plain = _load_block(_volume_source(False), 'glb_m0')
    assert re.search(r'if \(v\d+_g\) \{', plain)


def test_without_the_workaround_a_window_keeps_its_guard():
    """Asked off -- an IGC other than 2.41.5 -- the guarded write stays, and
    the unguarded one costs no registers."""
    import re
    block = _load_block(_volume_source(True, unguard=False), 'glb_m0')
    assert re.search(r'if \(v\d+_g\) \{\s*r\d+\[', block)


@pytest.mark.parametrize('revision,expected', [
    ('39758', True), ('39758.10', True), ('40196.8', False),
    ('39395.13', False), (None, False)])
def test_auto_follows_the_runtime_of_the_ocloc_on_path(monkeypatch, revision,
                                                       expected):
    from tensorforge import toolchain
    from tensorforge.backend.instructions.memory import load
    monkeypatch.setattr(toolchain, 'intel_neo_revision', lambda *a: revision)
    assert load._unguard_window_loads(
        Options(unguard_window_loads='auto')) is expected
