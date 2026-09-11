# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""`Options.autotune`: a generator that picks its own configuration.

Off by default, and switched on by one value.  Held here: that off changes
nothing, that on builds the pick and says so, that only safe knobs are turned,
and that a pick is remembered.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from tensorforge.common.context import Context, Options
from tensorforge.common.options import registry
from tensorforge.generators import tuning
from tensorforge.generators.generator import Generator

CASES = Path(__file__).resolve().parent / "cases"


def _case(name="local_flux.py"):
    path = CASES / name
    spec = importlib.util.spec_from_file_location("tf_autotune__" + path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _generate(arch="sm_100", **opts):
    mod = _case()
    gen = Generator(mod.descr_list(),
                    Context(arch=arch, backend="cuda", fp_type=mod.DTYPE,
                            options=Options(**opts)))
    gen.generate()
    return gen


@pytest.fixture(autouse=True)
def _forget_picks():
    tuning._PICKS.clear()
    tuning._LOADED.clear()
    yield
    tuning._PICKS.clear()
    tuning._LOADED.clear()


def test_autotune_is_off_by_default_and_off_changes_nothing():
    assert registry()['autotune'].default == 'off'
    assert not registry()['autotune'].codegen
    plain = _generate()
    off = _generate(autotune='off')
    assert plain.tuned is None and off.tuned is None
    assert plain.get_kernel() == off.get_kernel()


def test_static_autotune_builds_its_pick_and_says_so():
    gen = _generate(autotune='static', autotune_budget=6)
    assert gen.tuned is not None
    assert gen._num_threads == gen.tuned.lanes.num_threads
    assert gen._lead_width == gen.tuned.lanes.lead_width
    assert f'// tuned: {gen.tuned.label()}' in gen.get_kernel()


def test_the_budget_caps_the_builds(monkeypatch):
    built = []
    real = tuning.build

    def counting(factory, context, candidate):
        built.append(candidate)
        return real(factory, context, candidate)
    monkeypatch.setattr(tuning, 'build', counting)
    _generate(autotune='static', autotune_budget=4)
    # four for the walk, the default among them, and the cache key's build
    # is the default's -- so at most four distinct candidates
    assert len(set(built)) <= 4


def test_a_remembered_pick_costs_one_build(monkeypatch, tmp_path):
    path = tmp_path / "picks.json"
    first = _generate(autotune='static', autotune_budget=4,
                      autotune_cache=str(path))
    assert json.loads(path.read_text())
    tuning._PICKS.clear()
    tuning._LOADED.clear()
    built = []
    real = tuning.build

    def counting(factory, context, candidate):
        built.append(candidate)
        return real(factory, context, candidate)
    monkeypatch.setattr(tuning, 'build', counting)
    again = _generate(autotune='static', autotune_budget=4,
                      autotune_cache=str(path))
    assert len(built) == 1, 'the default build for the key, and nothing else'
    assert again.tuned == first.tuned
    assert again.get_kernel() == first.get_kernel()


def test_an_explicit_geometry_is_not_tuned():
    mod = _case()
    base = tuning.start(mod.descr_list(),
                        Context(arch="sm_100", backend="cuda", fp_type=mod.DTYPE))
    gen = Generator(mod.descr_list(),
                    Context(arch="sm_100", backend="cuda", fp_type=mod.DTYPE,
                            options=Options(autotune='static')),
                    lanes=base.lanes)
    gen.generate()
    assert gen.tuned is None


def _simple(arch, backend="cuda"):
    mod = _case()
    ctx = Context(arch=arch, backend=backend, fp_type=mod.DTYPE)
    descrs = mod.descr_list()
    origin = tuning.start(descrs, ctx)
    return {k.name: list(k.values(origin)) for k in tuning.simple_space(descrs, ctx)}


def test_the_simple_space_turns_only_what_is_safe_to_ship():
    knobs = _simple("sm_100")
    assert set(knobs) <= {'lanes', 'merge_variants', 'k_roll'}
    widths = {c.num_threads for c in knobs['lanes']}
    assert all(w & (w - 1) == 0 for w in widths), 'powers of two only'
    assert knobs['k_roll'] == [0, 28]


@pytest.mark.parametrize("arch,backend,paired", [
    ("sm_100", "cuda", True), ("sm_120", "cuda", False), ("sm_86", "cuda", False),
    ("gfx942", "hip", True), ("gfx1250", "hip", True), ("gfx1150", "hip", False)])
def test_width_two_only_where_one_instruction_does_two_fmas(arch, backend, paired):
    knobs = _simple(arch, backend)
    assert any(c.lead_width == 2 for c in knobs['lanes']) == paired
