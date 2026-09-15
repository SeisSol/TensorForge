# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""Which global transfers take a cache hint, and which hint.

A hint is a cache policy and never a value: a load spelled `__ldcs` or a store
spelled `__stcg` moves the same numbers, and the only question is what the
caches keep afterwards.  So the rule for who takes one may be simple, and the
kind of hint is a measurement (`Options.cache_hints`, `Options.hint_outputs`).
"""

KINDS = ('cg', 'cs', 'none')


def readers(symbol):
  """The users of `symbol` that read it: every one but those that only write
  it."""
  return [u for u in symbol.get_user_list()
          if not (getattr(u, '_dest', None) is symbol
                  and getattr(u, '_src', None) is not symbol)]


def cache_hint(context, allowed):
  """`False`, or the kind of hint an access that may take one gets.

  `cg` caches at L2 only, `cs` streams -- evict first, at every level -- and
  `none` asks for nothing.  The kind rides on the access's `nontemporal`
  attribute to the lexic that spells it (`CudaLexic.glb_load`); a target with
  one kind of hint reads it as a yes.
  """
  kind = context.get_user_options().cache_hints
  if kind not in KINDS:
    raise ValueError(f'cache_hints is one of {KINDS}, not {kind!r}')
  return kind if allowed and kind != 'none' else False
