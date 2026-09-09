# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Options as declared entries: what each layer answers, and what identifies it.

Three properties carry the weight here.

A duplicate name is an error.  In a class body two fields of one name are an
assignment and the last one wins, silently; a registry that refuses the second
declaration is what makes the collision as loud as a duplicate argument was.

"Nothing was asked" is not "the default was asked".  They are the same value
and a different statement: only the first follows a rule that varies by vendor,
and only the second holds when the rule changes.

And the delta identifies a configuration.  A build that asked for nothing has
an empty one, which is what keeps its generated symbol names where they are;
anything else has one that spells out exactly which answers it took.
"""

from __future__ import annotations

import pytest

from tensorforge.common import options as opt
from tensorforge.common.basic_types import Datatype
from tensorforge.common.context import Context, Options


NVIDIA = dict(arch='sm_86', backend='cuda')
AMD = dict(arch='gfx90a', backend='hip')


def _ctx(target=None, **asked):
  target = NVIDIA if target is None else target
  return Context(fp_type=Datatype.F32, options=Options(**asked), **target)


def _resolved(target=None, **asked):
  return _ctx(target, **asked).get_user_options()


@pytest.fixture
def scratch_option():
  """A declaration that exists for one test and leaves no trace in the registry."""
  names = []

  def make(name, doc='scratch', **kwargs):
    names.append(name)
    return opt.declare(name, doc, **kwargs)

  yield make
  for name in names:
    entry = opt._REGISTRY.pop(name, None)
    if entry is not None and entry.env is not None:
      opt._BY_ENV.pop(entry.env, None)


# -- declaration ------------------------------------------------------------- #

def test_a_second_declaration_of_a_name_is_refused(scratch_option):
  scratch_option('scratch_flag', default=False)
  with pytest.raises(ValueError, match='declared twice'):
    scratch_option('scratch_flag', default=True)


def test_two_options_cannot_share_one_variable(scratch_option):
  scratch_option('scratch_a', default=False, env='TF_SCRATCH_SHARED')
  with pytest.raises(ValueError, match='TF_SCRATCH_SHARED'):
    scratch_option('scratch_b', default=False, env='TF_SCRATCH_SHARED')


def test_an_option_needs_a_default_or_a_rule(scratch_option):
  with pytest.raises(ValueError, match='default or a rule'):
    scratch_option('scratch_empty')


def test_an_unknown_option_is_refused_by_name():
  with pytest.raises(ValueError, match='enable_pipelien'):
    Options(enable_pipelien=True)


# -- layers ------------------------------------------------------------------ #

def test_nothing_asked_takes_the_declared_default():
  assert _resolved().wrap_distance == 1
  assert _resolved().wide_bodies is True


def test_the_caller_beats_the_default():
  assert _resolved(wrap_distance=3).wrap_distance == 3


def test_the_general_variable_reaches_every_option(monkeypatch):
  monkeypatch.setenv(opt.OPTIONS_ENV, 'enable_pipeline=1,wrap_distance=2')
  resolved = _resolved()
  assert resolved.enable_pipeline is True
  assert resolved.wrap_distance == 2


def test_a_bare_name_in_the_general_variable_means_on(monkeypatch):
  monkeypatch.setenv(opt.OPTIONS_ENV, 'enable_pipeline')
  assert _resolved().enable_pipeline is True


def test_the_general_variable_is_checked_against_the_registry(monkeypatch):
  monkeypatch.setenv(opt.OPTIONS_ENV, 'wrap_distanse=2')
  with pytest.raises(ValueError, match='unknown option'):
    _resolved()


def test_the_specific_variable_beats_the_general_one(monkeypatch):
  monkeypatch.setenv(opt.OPTIONS_ENV, 'wide_bodies=1')
  monkeypatch.setenv('TF_IR_WIDE', '0')
  assert _resolved().wide_bodies is False


def test_the_caller_beats_both(monkeypatch):
  monkeypatch.setenv('TF_IR_WIDE', '0')
  assert _resolved(wide_bodies=True).wide_bodies is True


def test_an_unparseable_variable_names_itself(monkeypatch):
  monkeypatch.setenv('TF_LEAD_BLOCK', 'wide')
  with pytest.raises(ValueError, match='TF_LEAD_BLOCK'):
    _resolved()


# -- the vendor rule --------------------------------------------------------- #

def test_the_rule_answers_where_nobody_asked():
  assert _resolved(AMD).preload_globals is True
  assert _resolved(NVIDIA).preload_globals is False


def test_asking_overrides_the_rule_in_both_directions():
  assert _resolved(AMD, preload_globals=False).preload_globals is False
  assert _resolved(NVIDIA, preload_globals=True).preload_globals is True


def test_none_is_not_how_a_rule_backed_option_is_left_alone():
  with pytest.raises(ValueError, match='omit it'):
    Options(preload_globals=None)


def test_none_stays_a_value_where_it_is_one():
  assert _resolved(merge_max_arity=None).merge_max_arity is None
  assert _resolved(merge_max_arity=4).merge_max_arity == 4


# -- identity ---------------------------------------------------------------- #

def test_two_equal_requests_are_one_key():
  assert Options(wrap_distance=2) == Options(wrap_distance=2)
  assert len({Options(wrap_distance=2), Options(wrap_distance=2)}) == 1


def test_the_order_of_the_keywords_does_not_change_the_request():
  assert (Options(wrap_distance=2, enable_pipeline=True)
          == Options(enable_pipeline=True, wrap_distance=2))


def test_a_resolved_set_cannot_be_written_to():
  with pytest.raises(AttributeError):
    _resolved().wrap_distance = 4


def test_asking_for_nothing_leaves_no_delta_and_no_digest():
  for target in (NVIDIA, AMD):
    resolved = _resolved(target)
    assert resolved.delta() == {}
    assert resolved.label() == ''
    assert resolved.digest() == ''


def test_asking_for_the_rules_own_answer_is_not_a_delta():
  """What the hardware would have said anyway does not identify anything."""
  assert _resolved(AMD, preload_globals=True).delta() == {}
  assert _resolved(NVIDIA, preload_globals=False).delta() == {}


def test_the_two_sides_of_a_question_get_different_digests():
  amd_yes = _resolved(AMD, preload_globals=True)
  amd_no = _resolved(AMD, preload_globals=False)
  assert amd_yes.digest() != amd_no.digest()
  assert amd_no.label() == 'preload_globals=0'


def test_the_digest_is_the_same_for_the_same_configuration():
  first = _resolved(enable_pipeline=True, wrap_distance=2)
  second = _resolved(wrap_distance=2, enable_pipeline=True)
  assert first.digest() == second.digest()
  assert first == second


def test_a_diagnostic_does_not_rename_anything(monkeypatch):
  """`ir_stats` prints; it does not generate.  Turning it on has to leave the
  identity where it was, or a debugging run compares against different names."""
  monkeypatch.setenv('TF_IR_STATS', '1')
  resolved = _resolved()
  assert resolved.ir_stats is True
  assert resolved.digest() == ''


def test_an_undeclared_name_is_an_error_and_not_a_None():
  with pytest.raises(AttributeError, match='no option'):
    _resolved().enable_pipelien


def test_the_context_keeps_what_was_asked_apart_from_what_it_became():
  context = _ctx(AMD)
  assert context.get_asked_options().asked() == {}
  assert context.get_user_options().preload_globals is True


def test_two_contexts_do_not_share_their_options():
  first = _ctx(wrap_distance=2)
  second = _ctx()
  assert first.get_user_options().wrap_distance == 2
  assert second.get_user_options().wrap_distance == 1


def test_two_contexts_do_not_share_their_pressure_state():
  first, second = _ctx(), _ctx()
  first.measure_pressure = True
  first.record_pressure(128)
  assert second.measure_pressure is False
  assert second.peak_pressure is None
