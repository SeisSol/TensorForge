# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Measured preferences (`generators.preferences`) and `autotune=prefer`.

A preference is a measurement standing in for a ranking: which device, which
kernels, what to build them with, and on whose evidence.  Held here: the
device names a context answers to, the matching, the shipped GB200 entries,
and that `prefer` builds a preference and nothing else.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tensorforge.common.context import Context, Options
from tensorforge.generators import preferences, tuning
from tensorforge.generators.generator import Generator

CASES = Path(__file__).resolve().parent / "cases"


def _case(name="local_flux.py"):
    path = CASES / name
    spec = importlib.util.spec_from_file_location("tf_pref__" + path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _fresh():
    preferences._CACHE.clear()
    tuning._PICKS.clear()
    tuning._LOADED.clear()
    yield
    preferences._CACHE.clear()


def _ctx(arch="sm_100", backend="cuda", **opts):
    return Context(arch=arch, backend=backend, fp_type=_case().DTYPE,
                   options=Options(**opts))


def test_a_variant_is_asked_before_its_architecture_and_its_vendor():
    assert preferences.device_names(_ctx("gfx942", "hip", device="MI300A")) == [
        'gfx942:mi300a', 'gfx942', 'amd']
    assert preferences.device_names(_ctx()) == ['sm_100', 'nvidia']


def test_the_features_are_the_shape_of_the_kernel():
    feats = preferences.features(_case().descr_list(), _ctx())
    assert feats == {'dtype': 'f32', 'rows': 56, 'columns': 9, 'depth': 56}


def test_a_range_matches_inclusively_and_an_unnamed_feature_not_at_all():
    pref = preferences.Preference('x', match=(('rows', [41, 64]),))
    assert pref.matches({'rows': 41}) and pref.matches({'rows': 64})
    assert not pref.matches({'rows': 65})
    assert not preferences.Preference('x', match=(('depth', 8),)).matches({})


def _write(tmp_path, text):
    path = tmp_path / "prefs.yml"
    path.write_text(text)
    return str(path)


def test_the_most_specific_device_wins(tmp_path):
    path = _write(tmp_path, """
- device: gfx942
  prefer: {lanes: 16}
- device: gfx942:mi300x
  prefer: {lanes: 8}
""")
    descrs = _case().descr_list()
    x = preferences.lookup(descrs, _ctx("gfx942", "hip", device="mi300x",
                                        preferences=path))
    a = preferences.lookup(descrs, _ctx("gfx942", "hip", device="mi300a",
                                        preferences=path))
    assert dict(x.prefer)['lanes'] == 8
    assert dict(a.prefer)['lanes'] == 16


def test_the_callers_file_comes_before_the_shipped_one(tmp_path):
    path = _write(tmp_path, """
- device: sm_100
  prefer: {lanes: 16}
""")
    pref = preferences.lookup(_case().descr_list(), _ctx(preferences=path))
    assert pref.source == path


def test_the_shipped_gb200_entry_reaches_local_flux():
    descrs = _case().descr_list()
    ctx = _ctx()
    pref = preferences.lookup(descrs, ctx)
    assert pref is not None and pref.source == preferences.SHIPPED
    cand = preferences.candidate(pref, descrs, ctx)
    assert (cand.lanes.num_threads, cand.lanes.lead_width) == (8, 1)
    # `auto`: the largest divisor of the 56-step reduction up to 32
    assert cand.get('k_roll') == 28 and cand.get('merge_variants') is True
    assert cand.get('prepare_operands') is None, 'the host would have to pack'


def test_prefer_builds_the_preference_and_nothing_else(monkeypatch):
    built = []
    real = tuning.build

    def counting(factory, context, candidate):
        built.append(candidate)
        return real(factory, context, candidate)
    monkeypatch.setattr(tuning, 'build', counting)
    mod = _case()
    gen = Generator(mod.descr_list(), _ctx(autotune='prefer'))
    gen.generate()
    assert len(built) == 1
    assert gen.tuned.lanes.num_threads == 8
    assert '// tuned: 8w1 k_roll=28,merge_variants=1' in gen.get_kernel()


def test_prefer_without_a_preference_is_the_default():
    mod = _case()
    tuned = Generator(mod.descr_list(), _ctx("sm_120", autotune='prefer'))
    tuned.generate()
    plain = Generator(_case().descr_list(), _ctx("sm_120"))
    plain.generate()
    assert tuned.tuned is None
    assert tuned.get_kernel() == plain.get_kernel()


def test_igc_reports_a_retry_as_a_spill_and_silence_as_none():
    assert tuning.parse_igc('').spill_bytes == 0
    retry = ("[pvc] warning: in kernel 'k': [RetryManager] Start recompilation "
             "of the kernel")
    assert tuning.parse_igc(retry).spill_bytes == 1
    assert tuning.parse_igc('kernel k spilled 384 bytes').spill_bytes == 384
    assert tuning.parse_igc(retry).registers is None
