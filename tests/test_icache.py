# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The instruction-cache footprint: counted, converted, weighed.

The emitter counts a unit per statement per copy the compiler lays down --
an unrolled loop multiplies its body, a rolled one does not -- and
`analysis.icache` turns the count into bytes against the target's cache.  Past
the cache the batch loop fetches its body again on every iteration, so the
tuning scorer ranks by the excess right after the register file.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import pathlib
import warnings
from types import SimpleNamespace

import pytest

from tensorforge.analysis.icache import (ICacheBudgetWarning, code_bytes,
                                         icache_excess)
from tensorforge.backend.pir.emit import _code_copies
from tensorforge.common.context import Context, Options
from tensorforge.generators.generator import Generator
from tensorforge.generators.tuning import _icache_over

CASES = pathlib.Path(__file__).parent / 'cases'


def generated(case='local_flux', arch='sm_86', backend='cuda', **options):
    spec = importlib.util.spec_from_file_location(case, CASES / f'{case}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    context = Context(arch=arch, backend=backend, fp_type=module.DTYPE,
                      options=Options(**options) if options else None)
    generator = Generator(module.descr_list(), context)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with contextlib.redirect_stdout(io.StringIO()):
            generator.generate()
    return generator, context, caught


class TestCopies:
    def test_the_bare_pragma_unrolls_a_known_count_whole(self):
        assert _code_copies(True, 8) == 8

    def test_a_counted_pragma_lays_down_that_many(self):
        assert _code_copies(4, 16) == 4
        assert _code_copies(32, 16) == 16

    def test_anything_else_stays_a_loop(self):
        assert _code_copies(None, 8) == 1
        assert _code_copies(False, 8) == 1

    def test_without_constant_bounds_even_the_pragma_leaves_a_loop(self):
        assert _code_copies(True, None) == 1


def test_a_kernel_counts_its_code():
    generator, _, _ = generated()
    assert generator.metrics.code_units and generator.metrics.code_units > 0


def test_the_figures_are_the_builds_and_not_a_probes():
    """`local_flux` on gfx942 is built merged, and the merge decision then
    builds the list written out to compare: the kernel is the merged build,
    and so are its figures -- not those of the build that came after it."""
    decided, _, _ = generated(arch='gfx942', backend='hip')
    merged, _, _ = generated(arch='gfx942', backend='hip', merge_variants=True)
    assert decided.metrics.code_units == merged.metrics.code_units
    assert decided.metrics.hot_profile == merged.metrics.hot_profile
    assert decided.metrics.load_slack == merged.metrics.load_slack


def test_rolling_the_reduction_makes_the_body_smaller():
    """What `k_roll` is for: a loop is written once however long it runs,
    while the arithmetic it does -- `emitted_work` -- stays the same."""
    unrolled, _, _ = generated()
    rolled, _, _ = generated(k_roll=4)
    assert rolled.metrics.code_units < unrolled.metrics.code_units
    assert rolled.metrics.emitted_work == unrolled.metrics.emitted_work


class TestWeighed:
    @staticmethod
    def hw(size, family='nvidia'):
        return SimpleNamespace(icache_size=size, family=family,
                               instruction_bytes=16)

    def test_what_fits_is_not_over(self):
        assert icache_excess(100, self.hw(1 << 20)) == 0

    def test_past_the_cache_the_excess_is_in_bytes(self):
        hw = self.hw(1024)
        assert icache_excess(1000, hw) == code_bytes(1000, hw) - 1024 > 0

    def test_a_target_without_a_stated_cache_judges_nothing(self):
        assert icache_excess(10 ** 9, self.hw(None)) == 0

    def test_the_scorer_counts_whole_kilobytes_past_it(self, monkeypatch):
        """Of the size the build it is handed laid down: `local_flux`
        against a cache made too small for it."""
        from tensorforge.common.vm import hw_descr
        from tensorforge.generators.tuning import Build, Candidate
        original = hw_descr.HwDecription.__init__

        def small(self, *args, **kwargs):
            original(self, *args, **kwargs)
            self.icache_size = 1024
        monkeypatch.setattr(hw_descr.HwDecription, '__init__', small)
        generator, context, _ = generated()
        over = _icache_over(Build(Candidate(), generator, context))
        hw = context.target.hw
        assert over == icache_excess(generator.metrics.code_units, hw) // 1024
        assert over > 0


def test_a_kernel_over_the_cache_says_so(monkeypatch):
    """With the cache made small enough that `local_flux` cannot fit."""
    from tensorforge.common.vm import hw_descr
    original = hw_descr.HwDecription.__init__

    def small(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.icache_size = 1024
    monkeypatch.setattr(hw_descr.HwDecription, '__init__', small)
    _, _, caught = generated()
    assert any(issubclass(w.category, ICacheBudgetWarning) for w in caught)


class TestTargets:
    @pytest.mark.parametrize('arch,backend,size', [
        ('sm_86', 'cuda', 128 * 1024), ('sm_120', 'cuda', 128 * 1024),
        ('gfx942', 'hip', 64 * 1024), ('gfx90a', 'hip', 64 * 1024),
        ('gfx1150', 'hip', 32 * 1024), ('gfx1250', 'hip', 32 * 1024)])
    def test_the_cache_is_stated(self, arch, backend, size):
        from tensorforge.common.basic_types import Datatype
        hw = Context(arch=arch, backend=backend,
                     fp_type=Datatype.F32).target.hw
        assert hw.icache_size == size

    def test_intel_states_none_yet(self):
        from tensorforge.common.basic_types import Datatype
        hw = Context(arch='pvc', backend='oneapi',
                     fp_type=Datatype.F32).target.hw
        assert hw.icache_size is None

    @pytest.mark.parametrize('arch,backend,size', [
        ('sm_60', 'cuda', 11), ('sm_120', 'cuda', 16), ('gfx942', 'hip', 6),
        ('pvc', 'oneapi', 12)])
    def test_an_instruction_takes_its_encoding(self, arch, backend, size):
        from tensorforge.common.basic_types import Datatype
        hw = Context(arch=arch, backend=backend,
                     fp_type=Datatype.F32).target.hw
        assert hw.instruction_bytes == size


def test_the_hot_set_is_the_code_the_executions_need():
    """`hot_set` answers what a size alone cannot: how much of a body a
    kernel actually runs through.

    Two statements written once, one run a hundred times and one run once:
    half the executions need one unit, all of them need both.  A kernel whose
    code does not fit is a problem only where the part it runs does not fit
    either -- `elastic-o6s:derivative` needs every counted unit of its 118 kB
    to cover nine tenths of its executions, and 35 kB of them once the
    contraction is rolled by two.
    """
    from tensorforge.analysis.icache import hot_set

    profile = {(100, 1): 1, (1, 1): 1}
    assert hot_set(profile, share=0.5) == (1, 2)
    assert hot_set(profile, share=0.999) == (2, 2)
    # Copies count as the code they are: one statement written four times is
    # four units, however often each copy runs.
    assert hot_set({(10, 4): 1}, share=0.9) == (4, 4)
    assert hot_set(None) is None
    assert hot_set({}) is None
