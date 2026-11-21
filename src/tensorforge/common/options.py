# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The generator's options: one place that declares them, one that resolves them.

An option is an *entry* and not a class attribute.  `declare` refuses a name it
already holds, so a second declaration of `move_distance` is an error at import
naming the option, rather than an assignment in a class body where the last one
silently wins.

Four layers answer for a value, nearest first:

1. what the caller passed to `Options`,
2. the option's own environment variable, where the declaration names one,
3. `TF_OPTIONS`, a comma-separated ``name=value`` list covering every option,
4. the rule the declaration carries -- the target's preference
   (`Target.prefs`) -- or its plain default.

Layer 3 is what a caller that cannot reach the constructor uses -- the yateto
frontend builds its own context -- and layer 2 is for the few switches whose
spelling is already written down in tools and scripts.

`resolve` runs the layers against one target and returns a frozen
`ResolvedOptions`: one value per declared name, hashable, and carrying the
*delta* to what the same target would have produced had nothing been asked at
all.  That delta is what identifies a configuration.  `label` spells it for a
report and `digest` for a symbol name, and it is empty for a caller that asked
for nothing, which is what keeps generated names stable for the default build.
"""
import hashlib
import os
from typing import Any, Callable, Dict, Mapping, Optional


class _Unset:
  """Absence of a value, distinct from `None`.

  `None` is a value some options take -- `merge_max_arity=None` is "no cap" --
  so it cannot also mean "nothing was said".
  """
  _instance = None

  def __new__(cls):
    if cls._instance is None:
      cls._instance = super().__new__(cls)
    return cls._instance

  def __repr__(self):
    return 'UNSET'


UNSET = _Unset()

#: A comma-separated ``name=value`` list setting any declared option, e.g.
#: ``TF_OPTIONS=enable_wrap_loads=1,move_distance=2``.  A bare ``name`` means
#: ``name=1``.
OPTIONS_ENV = 'TF_OPTIONS'


# -- value parsers ----------------------------------------------------------- #

def parse_bool(text: str) -> bool:
  low = text.strip().lower()
  if low in ('', '0', 'false', 'no', 'off'):
    return False
  if low in ('1', 'true', 'yes', 'on'):
    return True
  raise ValueError(f'expected a boolean, got {text!r}')


def parse_int(text: str) -> int:
  return int(text.strip())


def parse_optional_int(text: str) -> Optional[int]:
  low = text.strip().lower()
  if low in ('', 'none'):
    return None
  return int(low)


def parse_str(text: str) -> str:
  return text


def parse_optional_bool(text: str) -> Optional[bool]:
  if text.strip().lower() in ('', 'none'):
    return None
  return parse_bool(text)


def parse_bool_or_auto(text: str):
  """`auto`, or a boolean."""
  if text.strip().lower() == 'auto':
    return 'auto'
  return parse_bool(text)


def parse_float(text: str) -> float:
  return float(text.strip())


#: The routes `Options.amd_matrix_k4` names besides on and off.
MATRIX_K4_ROUTES = ('shard', 'transpose')


def parse_matrix_k4(text: str):
  """A route of `Options.amd_matrix_k4` (`MATRIX_K4_ROUTES`), or a boolean."""
  low = text.strip().lower()
  if low in MATRIX_K4_ROUTES:
    return low
  return parse_bool(text)


def parse_register_spill(text: str) -> str:
  """`off`, `remat` or `shared` (`Options.register_spill`)."""
  value = text.strip().lower()
  if value not in REGISTER_SPILL:
    raise ValueError(f'expected one of {REGISTER_SPILL}, got {text!r}')
  return value


#: What `Options.register_spill` takes.
REGISTER_SPILL = ('off', 'remat', 'shared')


def parse_crosslane_pins(text: str) -> str:
  """`use`, `definition` or `none` (`Options.crosslane_pins`)."""
  value = text.strip().lower()
  if value not in CROSSLANE_PINS:
    raise ValueError(f'expected one of {CROSSLANE_PINS}, got {text!r}')
  return value


#: What `Options.crosslane_pins` takes.
CROSSLANE_PINS = ('use', 'definition', 'none')


def parse_dpp_waits(text: str) -> str:
  """`needed` or `always` (`Options.amd_dpp_waits`)."""
  value = text.strip().lower()
  if value not in DPP_WAITS:
    raise ValueError(f'expected one of {DPP_WAITS}, got {text!r}')
  return value


#: What `Options.amd_dpp_waits` takes.
DPP_WAITS = ('needed', 'always')


#: What `Options.multi_element` takes.
MULTI_ELEMENT = ('off', 'auto', 'force')


def parse_multi_element(text: str) -> str:
  """`off`, `auto` or `force` (`Options.multi_element`)."""
  value = str(text).strip().lower()
  if value not in MULTI_ELEMENT:
    raise ValueError(f'expected one of {MULTI_ELEMENT}, got {text!r}')
  return value


#: What `Options.multi_element_fma` takes.
MULTI_ELEMENT_FMA = ('auto', 'warp', 'lanes')


def parse_multi_element_fma(text: str) -> str:
  """`auto`, `warp` or `lanes` (`Options.multi_element_fma`)."""
  value = str(text).strip().lower()
  if value not in MULTI_ELEMENT_FMA:
    raise ValueError(f'expected one of {MULTI_ELEMENT_FMA}, got {text!r}')
  return value


#: What `Options.carry_element_index` takes.
CARRY_ENTRY = ('auto', 'always', 'never')


def carry_mode(value) -> str:
  """`Options.carry_element_index` as one of `CARRY_ENTRY`: a boolean, as the
  switch was before it had a rule, reads as `always` or `never`."""
  if value is True:
    return 'always'
  if value is False:
    return 'never'
  mode = str(value).strip().lower()
  if mode not in CARRY_ENTRY:
    raise ValueError(f'expected one of {CARRY_ENTRY} or a boolean, '
                     f'got {value!r}')
  return mode


def parse_carry(text: str) -> str:
  """`auto`, `always` or `never`, or a boolean for the last two
  (`Options.carry_element_index`)."""
  low = text.strip().lower()
  if low in CARRY_ENTRY:
    return low
  return carry_mode(parse_bool(text))


_DEFAULT_PARSERS = {bool: parse_bool, int: parse_int, str: parse_str}


class Opt:
  """One declared option: its name, where its value comes from, and why."""
  __slots__ = ('name', 'doc', 'default', 'env', 'parse', 'rule', 'codegen')

  def __init__(self,
               name: str,
               doc: str,
               default: Any = UNSET,
               env: Optional[str] = None,
               parse: Optional[Callable[[str], Any]] = None,
               rule: Optional[Callable[[Any], Any]] = None,
               codegen: bool = True):
    if rule is None and default is UNSET:
      raise ValueError(f'option {name!r} needs either a default or a rule')
    if parse is None:
      parse = _DEFAULT_PARSERS.get(type(default))
    if env is not None and parse is None:
      raise ValueError(f'option {name!r} is settable from {env} but has no parser')
    self.name = name
    self.doc = doc
    self.default = default
    self.env = env
    self.parse = parse
    self.rule = rule
    #: Whether this option can change the generated text.  A diagnostic that
    #: only prints stays out of the identity, or switching it on would rename
    #: every kernel in the build.
    self.codegen = codegen

  def base(self, target) -> Any:
    """The value for `target` (`common.target.Target`) when nobody asked for
    one.

    The target and not the hardware, because a default is not always a fact
    about the hardware: the same Intel device runs both lowerings, and an
    operand staged once per block is worth 1.71x under the explicit vector and
    0.86x under SPMD.  A rule that sees only the vendor has to answer both
    with one number.
    """
    if self.rule is None:
      return self.default
    return self.rule(target)

  def check(self, value: Any) -> None:
    if value is UNSET:
      raise ValueError(f'option {self.name!r}: pass a value or omit the option')
    if value is None and self.default is not None:
      raise ValueError(
          f'option {self.name!r} has no value None; omit it to take the default')

  def read(self, text: str, source: str) -> Any:
    if self.parse is None:
      raise ValueError(f'{source}: option {self.name!r} cannot be set as text')
    try:
      return self.parse(text)
    except ValueError as exc:
      raise ValueError(f'{source}={text!r}: {exc}') from None


_REGISTRY: Dict[str, Opt] = {}
_BY_ENV: Dict[str, Opt] = {}


def declare(name: str, doc: str, **kwargs) -> Opt:
  """Register an option, refusing a name or a variable that is already taken."""
  if name in _REGISTRY:
    raise ValueError(f'option {name!r} is declared twice; a name is registered once '
                     f'so that a duplicate is an error and not an overwrite')
  opt = Opt(name, doc, **kwargs)
  if opt.env is not None:
    taken = _BY_ENV.get(opt.env)
    if taken is not None:
      raise ValueError(f'{opt.env} already sets option {taken.name!r}, '
                       f'so it cannot also set {name!r}')
    _BY_ENV[opt.env] = opt
  _REGISTRY[name] = opt
  return opt


def registry() -> Mapping[str, Opt]:
  """Every declared option, by name."""
  return dict(_REGISTRY)


def _spell(value: Any) -> str:
  if isinstance(value, bool):
    return '1' if value else '0'
  if value is None:
    return 'none'
  return str(value)


def _from_environment(env: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
  """Layers 2 and 3, flattened; the specific variable wins over the general one."""
  env = os.environ if env is None else env
  out: Dict[str, Any] = {}
  for item in env.get(OPTIONS_ENV, '').split(','):
    if not item.strip():
      continue
    name, sep, text = item.partition('=')
    name = name.strip()
    opt = _REGISTRY.get(name)
    if opt is None:
      raise ValueError(f'{OPTIONS_ENV} names an unknown option {name!r}; '
                       f'declared are {sorted(_REGISTRY)}')
    out[name] = opt.read(text if sep else '1', f'{OPTIONS_ENV}:{name}')
  for var, opt in _BY_ENV.items():
    text = env.get(var)
    if text is not None:
      out[opt.name] = opt.read(text, var)
  return out


class Options:
  """What a caller asked for, before any hardware is known.

  Holds only the options actually passed, so that "asked for nothing" is
  distinguishable from "asked for the default" -- the two are the same value
  and a different statement, and only the first follows a changing default.
  """
  __slots__ = ('_asked',)

  def __init__(self, **asked):
    unknown = sorted(set(asked) - set(_REGISTRY))
    if unknown:
      raise ValueError(f'unknown option(s) {unknown}; declared are {sorted(_REGISTRY)}')
    for name, value in asked.items():
      _REGISTRY[name].check(value)
    self._asked = tuple(sorted(asked.items()))

  def asked(self) -> Dict[str, Any]:
    return dict(self._asked)

  def resolve(self, target) -> 'ResolvedOptions':
    """Settle every declared option against `target` (`Opt.base`)."""
    from_env = _from_environment()
    asked = self.asked()
    values: Dict[str, Any] = {}
    delta: Dict[str, Any] = {}
    for name, opt in _REGISTRY.items():
      base = opt.base(target)
      if name in asked:
        value = asked[name]
      elif name in from_env:
        value = from_env[name]
      else:
        value = base
      values[name] = value
      if opt.codegen and value != base:
        delta[name] = value
    return ResolvedOptions(values, delta)

  def __eq__(self, other):
    return isinstance(other, Options) and self._asked == other._asked

  def __hash__(self):
    return hash(self._asked)

  def __repr__(self):
    inner = ', '.join(f'{n}={v!r}' for n, v in self._asked)
    return f'Options({inner})'


class ResolvedOptions:
  """One value per declared option, fixed, plus the delta that identifies it.

  Fixed because generation reads it from a dozen places and a pass that could
  answer a question differently the second time it is asked is not a
  configuration but a mood.
  """
  __slots__ = ('_values', '_delta')

  def __init__(self, values: Mapping[str, Any], delta: Mapping[str, Any]):
    object.__setattr__(self, '_values', dict(values))
    object.__setattr__(self, '_delta', dict(delta))

  def __setattr__(self, name, value):
    raise AttributeError(f'options are fixed once resolved; {name!r} cannot be set')

  def __getattr__(self, name):
    try:
      return self._values[name]
    except KeyError:
      raise AttributeError(f'no option {name!r}; declared are {sorted(self._values)}') from None

  def __contains__(self, name):
    return name in self._values

  def delta(self) -> Dict[str, Any]:
    """What this configuration says that the bare default for the same hardware
    does not.  Empty for the default build."""
    return dict(self._delta)

  def label(self) -> str:
    """The delta as text, canonical: same configuration, same spelling."""
    return ','.join(f'{n}={_spell(v)}' for n, v in sorted(self._delta.items()))

  def digest(self, length: int = 8) -> str:
    """A short hash of the delta, empty where there is none.

    Empty on purpose: a build that asked for nothing then contributes nothing
    to whatever names it, and its symbols do not move because this exists.
    """
    if not self._delta:
      return ''
    sha = hashlib.new('md5', usedforsecurity=False)
    sha.update(self.label().encode())
    return sha.hexdigest()[:length]

  def describe(self) -> str:
    """The delta for a generated file to carry, or a word saying there is none."""
    return self.label() or 'default'

  def __eq__(self, other):
    return isinstance(other, ResolvedOptions) and self._values == other._values

  def __hash__(self):
    return hash(tuple(sorted(self._values.items(), key=lambda kv: kv[0])))

  def __repr__(self):
    return f'ResolvedOptions({self.label() or "default"})'


# -- the options ------------------------------------------------------------- #

declare('align_shr_mem',
        default=True,
        doc='Round every shared-memory allocation up to the access alignment.')

declare('enable_sync_block_opt',
        default=True,
        doc='Drop barriers a data-flow argument shows to be redundant.')

declare('share_block_buffers',
        default=False,
        doc='Let the buffers of the block\'s shared arena share bytes where '
            'no statement occupies two of them, as the buffers of a '
            'multiplication\'s arena do (`pir.allocate`), instead of laying '
            'them out back to back.  For a block that runs phases one after '
            'another -- the multi-element layout\'s chained sections, an '
            'order\'s buffer dead once the next order has read it.  Taken '
            'only with `enable_sync_block_opt`: the barrier placement behind '
            'the allocator fences a write over the bytes another buffer was '
            'read from, which the builders\' own barriers, keyed on buffers, '
            'do not.  Off: every kernel as before.')

declare('enable_multibuffer',
        default=False,
        doc='Give a shared transfer `enable_wrap_loads` moves a second stage, '
            'and issue it at the head of the body instead of at its tail: a '
            'whole iteration ahead of its first read, into the stage the '
            'iteration does not read.\n'
            'For copies the hardware carries out asynchronously only -- a '
            'transfer written as loads and stores stalls on its loads where '
            'it is issued, at the head as at the tail.  The second stage '
            'doubles the buffer, and a launch pays for it in occupancy: over '
            'the corpus on sm_86, 17% more shared memory at `move_distance` '
            '1 and 84% at 2 (`tools/wrap_census.py`).  So it is asked for '
            'rather than given, off pending hardware numbers.  See '
            'backend/pir/wrap.py.')

declare('enable_move_loads',
        rule=lambda target: target.prefs.move_loads,
        parse=parse_bool,
        doc='Issue a transfer early and wait where the value is needed.\n'
            'The transfer moves up its statement list to hide its latency, '
            'and the wait stays at the first read; it stops where something '
            'between the two would write what it reads, and after '
            '`move_distance` transfers.  See backend/pir/move.py.\n'
            'On by default -- this switch exists so the default can be '
            '*priced*, not because it is in doubt. A pass with '
            'no way to be turned off is a pass whose contribution nobody has '
            'measured, and the same argument that put `preload_globals` behind '
            'a question applies to it.  Off under DPC++\'s SPMD lowering for '
            'Intel (`oneapi`), where what it moves spills '
            '(`Preferences.move_loads`); AdaptiveCpp\'s lowering of the same '
            'device is not measured and keeps it (`target.ONEAPI_ONLY`).')

declare('enable_wrap_loads',
        rule=lambda target: target.prefs.wrap_loads,
        parse=parse_bool,
        doc='Prefetch across the back edge: issue each transfer for the next '
            'element at the tail of the current iteration, after the last '
            'instruction that touches its buffer -- where '
            '`enable_move_loads` would put it in the loop unrolled once.  One '
            'buffer copy, register and shared destinations alike.  The same '
            'in any loop over constant bounds -- a merged run, a rolled '
            'reduction -- for its next trip; see backend/pir/wrap.py.  Off '
            'under DPC++\'s SPMD lowering for Intel (`oneapi`), where what it '
            'moves spills and a grid that covers the batch in one round has '
            'no next element (`Preferences.wrap_loads`); AdaptiveCpp\'s '
            'lowering of the same device is not measured and keeps it '
            '(`target.ONEAPI_ONLY`).')

declare('move_distance',
        default=1,
        doc='How many transfers a transfer is moved ahead by.  '
            '`enable_move_loads` lets one travel past this many earlier '
            'transfers of its pipeline before it stops (1: the one before '
            'it; `transfer_pipelines`); `enable_wrap_loads` wraps the '
            'transfers whose move runs across the back edge, which are the '
            'first this many of each pipeline in the body.  A dependence '
            'stops a transfer whatever the distance.')

declare('transfer_pipelines',
        default='path',
        parse=parse_str,
        doc='Which transfers form one pipeline for `move_distance`.  `path`, '
            'the default, gives every memory path its own: transfers into '
            'shared memory and transfers into registers move past each other '
            'without using up each other\'s distance, and '
            '`enable_wrap_loads` wraps the first `move_distance` of each.  '
            '`one` puts every transfer into one pipeline, as one counter would '
            'see them.  See backend/pir/transfers.py.')

declare('wrap_early',
        default=True,
        parse=parse_bool,
        doc='Issue a transfer `enable_wrap_loads` moves into the next '
            'iteration right behind the last access to its buffer in this one, '
            'rather than at the tail -- where its buffer is not shared with '
            'what the rest of the iteration holds, the allocator keeps it '
            'apart, so a buffer of its own buys the stretch behind that access '
            'as cover.  In a counted loop and in a batch loop without an '
            'element guard every moved transfer; in a guarded batch loop a '
            'transfer into registers, behind the last access inside the guard '
            'and, for a masked element, in the guard\'s other arm.  A copy into '
            'shared memory there keeps the tail, since a copy issued in one arm '
            'only would leave the count of groups in flight differing between '
            'the paths; `enable_multibuffer` moves it to the head instead.  No '
            'earlier than `issue_cover` ahead of its first read.')

declare('pointer_stage',
        default=True,
        parse=parse_bool,
        doc='Read the pointer an element owns -- its entry in a pointer '
            'array, `Addressing.PTR_BASED` -- a whole iteration before the '
            'transfer that follows it, where `enable_multibuffer` issues that '
            'transfer at the head of the body.  There the copy for the next '
            'element follows its pointer right away, and the read of the '
            'pointer array is what the copy waits for; carried in from the '
            'iteration before, the pointer is there.  It costs a register per '
            'pointer, and the head reads the entries of the element after '
            'next.  Elsewhere the read at the head has the body ahead of the '
            'transfer as cover, and nothing changes.  See backend/pir/wrap.py.')

declare('carry_element_index',
        default='auto',
        parse=parse_carry,
        doc='Where an element\'s entry of the launch\'s index list is read -- '
            'the operands of `Addressing.INDEXED` and `INDEXED_POINTER` reach '
            'their data through it.  It is read once, as one value every '
            'address through it takes, rather than once in each address; and '
            'where `enable_wrap_loads` reads the next element\'s entry for a '
            'transfer it issues ahead, the loop can carry that entry into the '
            'next iteration as the element\'s own, as the flag words ride the '
            'loop, with the first element\'s read in the peel.  Read at the '
            'top of the element, the entry is a load the addresses wait for '
            'before their own: through a pointer per cell, index, pointer and '
            'data are three dependent loads where a pointer table has two.  '
            'Carried, it is there -- and it holds a register from the top of '
            'the body to its first reader.  `auto` carries it where that '
            'stretch stays clear of the body\'s pressure peak -- through a '
            'pointer per cell, that takes SeisSol\'s volume and neighbor flux '
            'back under GB200\'s occupancy cliff -- and otherwise reads it at '
            'the top of the element, where the addresses through the list '
            'always stood (`pir.entries`); `always` carries it wherever the '
            'body still reads it; `never` reads it in each address.  Nothing '
            'changes for a kernel without such an operand.  See '
            'backend/pir/entries.py and backend/pir/wrap.py.')

declare('trip_reads',
        default=False,
        parse=parse_bool,
        doc='In a loop whose trips are counted -- a reduction `k_roll` rolls, '
            'typically -- read what a trip reads from global memory in place '
            'into a register array at the head of the trip instead, a slot '
            'per load, which `enable_wrap_loads` then fills a trip ahead.  '
            'Read in place, the operator\'s element is a load the products of '
            'the trip take at once; filled a trip ahead, it is there.  It '
            'costs a register per load of a trip.  See backend/pir/trips.py.')

declare('trip_shards',
        default=False,
        parse=parse_bool,
        doc='In a loop whose trips are counted -- a reduction `k_roll` rolls, '
            'typically -- copy what a trip reads from global memory in place '
            'into shared memory of the multiplication\'s own, the run of the '
            'operand the trip\'s lanes read, with the asynchronous copy, and '
            'read it there; `enable_wrap_loads` issues the copy a trip ahead, '
            'and with `enable_multibuffer` into a second stage at the head of '
            'the trip before.  No block barrier: the window is the '
            'multiplication\'s, and so is the barrier between the copy and the '
            'reads.  Where the target has the copy (`Target.async_staging`); '
            'ahead of `trip_reads`, which takes what is left.  See '
            'backend/pir/trips.py.')

declare('stack_elements',
        default=1,
        parse=parse_int,
        doc='How many consecutive elements of the batch one multiplication '
            'computes together.  Every operand an element owns gains a last '
            'axis of that extent and the ones the batch shares stay as they '
            'are, so a product against a shared operator comes out that many '
            'times wider: 56x9 by 9x9 becomes 56x9x2 by 9x9, a shape that '
            'fills a matrix instruction one element leaves half empty.  The '
            'batch loop visits a stack of elements per trip, each element '
            'reached through its own pointer; where the batch ends inside a '
            'stack, or a flag masks an element, nothing is written for it.  '
            'It costs registers and shared memory in proportion.  1 or 2; '
            'contractions, pointwise operations and reductions, without '
            'guards, and not under `launch_control`.  See '
            'generators/stacking.py.')

declare('issue_cover',
        default=512,
        parse=parse_int,
        doc='The most cover a transfer is issued with, in the units the '
            'emitter counts code in, a counted loop\'s body once per trip: one '
            '`move` takes further ahead of its first read goes back down '
            'behind `flatten`, and one issued early (`wrap_early`) goes no '
            'further ahead.  '
            'Cover beyond what hides a load\'s latency only holds its buffer: '
            'at some 600 to 800 cycles for a load from global memory and four '
            'warps on a scheduler, a warp issues 150 to 200 instructions in '
            'that time, and 512 leaves room.  0 is no limit.')

declare('bulk_copies',
        default=False,
        parse=parse_bool,
        doc='Copy a window into shared memory as one bulk copy by the tensor '
            'memory accelerator (`cp.async.bulk`, sm_90 and newer) instead of '
            'the sixteen bytes a thread of `cp.async`: issued by one thread of '
            'the multiplication, completing on an mbarrier every thread waits '
            'on, and counted in no group.  Only a window that is one run, of a '
            'multiple of sixteen bytes, on addresses the operand proves '
            'sixteen-byte aligned, without padded rows or a permutation, in a '
            'section\'s body; any other copy keeps `cp.async`.  One copy is '
            'in flight on a barrier, so a bulk copy keeps one stage under '
            '`enable_multibuffer`.  Off until measured against `cp.async`.')

declare('amd_async_copies',
        default=False,
        parse=parse_bool,
        doc='Stage into the LDS on AMD as NVIDIA stages into shared memory with '
            '`cp.async`: through the LDS-DMA (`global_load_lds`, gfx90a, gfx94x '
            'and gfx950), which moves a lane\'s four bytes from global memory '
            'into the LDS without a register.  Each wait for one is written '
            'out, `s_waitcnt vmcnt(N)` with N the copies issued behind it: the '
            'compiler places none in front of a read of what a copy wrote, '
            'and a read behind no block barrier reads the LDS before the copy '
            'lands.  Only where a multiplication is a whole number of waves '
            'and each lane moves one element of four bytes to its own place '
            'past a base the wave shares, which is the one destination the '
            'instruction writes; any other copy keeps the body to loads and '
            'stores.  On gfx1250 and gfx1251 through their copy into the LDS '
            '(`global_load_async_to_lds`) instead: each lane to an address of '
            'its own, 1, 4, 8 or 16 bytes, at any width of a multiplication, '
            'and each wait written on `asynccnt` -- which neither the compiler '
            'nor a block barrier waits for.  Off until measured: on gfx942 a '
            'copy a trip ahead did not beat an unrolled synchronous one, and '
            'no gfx1250 has run one.')

declare('amd_tensor_copies',
        default=False,
        parse=parse_bool,
        doc='On gfx1250 and gfx1251, with `amd_async_copies`, stage an '
            'operator the block preloads into the LDS as one tile of the '
            'tensor data mover (`tensor_load_to_lds`) rather than as a copy a '
            'lane: the block\'s first wave issues the whole run through a '
            'descriptor and waits for it on `tensorcnt`, and the barrier '
            'behind the wait shows it to the other waves.  Only where the '
            'block\'s lanes are a whole number of waves and the window is one '
            'run in order -- no rows of its own, no padded rows, no '
            'permutation; any other window, and a multiplication\'s own, keeps '
            'its copies a lane.  The descriptor\'s '
            'layout is the one LLVM\'s AMDGPU lowering, Triton and '
            'composable_kernel agree on (`hip.h`, `tensorDescriptorWords`).  '
            'Compiled only: no gfx1250 has run one.')

declare('enable_prefetch',
        default=False,
        parse=parse_bool,
        doc='Issue a cache hint for the next batch element at the top of the '
            'loop body, where the target has a data prefetch.\n'
            'Distinct from `enable_wrap_loads`, which moves the transfer '
            'itself: nothing is loaded here and nothing waits, so it costs no '
            'register and no buffer, and it is dropped outright on a target '
            'that cannot spell it. What it buys is the head of element k+1 '
            'being on its way while k is computed -- and under '
            '`Addressing.PTR_BASED` the pointer, which is the load every '
            'other address of that element depends on.\n'
            'Off pending numbers from hardware. A hint that arrives too early '
            'is evicted before its use and one that arrives too late is a '
            'wasted request, and which of the two a body does is a '
            'measurement.')

declare('prefetch_level',
        default='l2',
        parse=parse_str,
        doc='Which cache `enable_prefetch` asks for: `l1` or `l2`.\n'
            'A request rather than an instruction. NVIDIA spells both and is '
            'so far the only target that spells either; AMD takes a scope '
            'where this would be a level, and SYCL 2020 has no level at all, '
            'so the answer there is the same instruction both ways.\n'
            'L2 by default, because the distance this issues at is a whole '
            'loop body: L1 is small enough that the line is likely gone again '
            'before the iteration that wants it.')

declare('cache_hints',
        default='cg',
        parse=parse_str,
        doc='Which cache policy a global transfer that may take a hint gets: '
            '`cg` (L2 only: `__ldcg`/`__stcg`), `cs` '
            '(streaming, evict first at every level: `__ldcs`/`__stcs`) or '
            '`none`.\n'
            'Which transfers may take one is `hint_outputs`\'s question.  A '
            'hint is a cache policy and never a value, so the choice is a '
            'measurement: data a kernel reads once gains nothing from staying '
            'in a cache, and displaces the operators every element reuses.  '
            'NVIDIA spells both kinds, AMD one '
            '(`__builtin_nontemporal_*`), SYCL none.')

declare('hint_outputs',
        default=False,
        parse=parse_bool,
        doc='Let the outputs take the cache hint too.\n'
            'A load takes one where it is the only user of its source, and a '
            'store where its register source has no other user -- which an '
            'accumulated image never has.  So no output store of a SeisSol '
            'kernel takes one, and neither does the one read of a `+=` '
            'destination.  With this, a load takes the hint where it is the '
            'only reader of its source, and a store where nothing reads the '
            'destination but transfers (a `+=` destination\'s own preload).')

declare('atomic_accumulation',
        rule=lambda target: target.prefs.placement.atomic_accumulation,
        parse=parse_bool,
        doc='Let an accumulating result reach global memory as an atomic add '
            'rather than as a read, an add and a store.\n'
            'The vendor table decides by default -- AMD yes, NVIDIA no -- and '
            'the option exists so the two can be compared in one run.  What a '
            'comparison has to watch on NVIDIA: the atomic removes the '
            "destination *read*, which can free the load-moving pass to wrap "
            'an operand it left alone before, so the configurations differ in '
            'two things and not one.  `atomicAdd` with the result dropped is '
            'the non-returning `REDG`, and a lane mask costs nothing (it is '
            'predication); in a cache-resident microbenchmark the reduction '
            'beat the read-modify-write by 37 %, and at the bandwidth limit '
            'the two are equal because the write allocates the line either '
            'way.\n'
            'Only ever a preference: the body adds atomically only where '
            'nothing reads the sum -- not the element behind the store, not '
            'a register the sum passes through -- and the target has the '
            'instruction (`pir.atomics`).  Anything else goes out as the '
            'read, the add and the store.')

declare('preload_globals',
        rule=lambda target: target.prefs.preload_globals,
        parse=parse_bool,
        doc='Stage every `Addressing.NONE` operand into shared memory once per '
            'block, in the section prologue, instead of reading it from global '
            'inside the batch loop.\n'
            'A question and not a constant because the answer is a measurement '
            'nobody has taken on NVIDIA: the default is "AMD only", so the whole '
            'NVIDIA path -- including the tensor-core one, where a batch-constant '
            'operand would also carry a batch-constant *conversion* -- has never '
            'been compared against its own alternative.  A benchmark cannot ask '
            'a question the generator cannot be asked.\n'
            'On Intel it depends on the lowering rather than the vendor, which '
            'is why the default is the target\'s (`Target.prefs`).  Under the '
            'explicit vector every work-item holds its own copy of an '
            'operator, so staging it once per block is 1.71x over the twenty '
            'elastic kernels -- fifteen of them faster, up to 6.4x, and the '
            'five that lose give up 1 to 9 %.  '
            'Under SPMD the operators are read in place and the same switch is '
            '0.86x, ranging from 0.35x to 1.41x.  Both stay in the tuner\'s '
            'space, so the five are recoverable and the default is only where '
            'the walk starts.')

declare('split_predicated_load',
        rule=lambda target: target.prefs.split_predicated_load,
        parse=parse_bool,
        doc='Read a predicated load unconditionally at a clamped address and '
            'select the value afterwards, instead of predicating the load '
            'itself.\n'
            'For a miscompilation and not for the program; the two forms '
            'compute the same thing.  `p ? base[i] : 0` predicates the *load*, '
            'and the Intel device compiler carries that predicate backwards '
            'through the dependency chain: where a multiplication has fewer '
            'active rows than lanes, it concludes that the lanes above the '
            'active count produce nothing, drops their share of the register '
            'images staged for the same multiplication, and the broadcasts '
            'that read exactly those lanes read whatever the register held.  '
            'The loss is silent and exact -- the contraction steps whose '
            'source lane lies between the active count and the vector width; '
            '`tests/cases/tw_split_load` is the case, and `-cl-opt-disable` '
            'computes it.  Hoisting the load out of the conditional expression '
            'is what stops it: the load is then unconditional, nothing is '
            'predicated backwards, and the select on the value keeps the zero '
            'the false branch stands for.  Clamping the address is what makes '
            'the unconditional read legal.\n'
            'Both halves are needed.  Clamping alone, with the load still '
            'inside the ternary, still loses the steps; the barrier that looks '
            'like the obvious remedy makes it worse, from one wrong column to '
            'five.\n'
            'Intel only, and there only under SPMD: elsewhere a predicated '
            'load is one instruction and this would add a select to every one '
            'of them, and under the explicit-vector lowering the predicate is '
            'a mask over one work-item\'s own lanes -- it selects elements of '
            'a value, there is no other lane to read from, and an address '
            'clamped by a mask is not an address.')

declare('unguard_window_loads',
        default='auto',
        env='TF_UNGUARD_WINDOW_LOADS',
        parse=parse_bool_or_auto,
        doc='Write a window loaded into a register image of its own on every '
            'lane, the load held to the window by a select, instead of under '
            'the lane guard (`GlbToRegLoader._fresh_window`; F32, where lane '
            'guards are lowered to selects).  For a miscompilation: IGC 2.41.5 '
            '(compute-runtime 26.35) computes SeisSol\'s order-3 and order-5 '
            'F32 `volume` wrong around the guarded writes on the Max 1550, and '
            '2.40.13 and 2.42.6 compute them right.  Not free: the unguarded '
            'form costs IGC registers, F32 `volume` 22 to 29 % and '
            '`neighboringFlux` 10 to 15 % on PVC.  `auto` writes it where the '
            '`ocloc` on PATH is of the 26.35 runtime (`toolchain.'
            'intel_neo_revision`).')

declare('convert_lane_guards',
        rule=lambda target: target.prefs.convert_lane_guards,
        parse=parse_bool,
        doc='Lower a guard over a lane-varying condition the way the '
            'explicit-vector lowering always does (the PIR pass `if_convert`): '
            'arithmetic runs on every lane, reads become selects, and only a '
            'statement with a side effect keeps a branch of its own.\n'
            'For a miscompilation and not for the program, like '
            '`split_predicated_load`, and the same one met another way.  '
            'SeisSol\'s time derivative computes its last derivative on one '
            'lane, which reads by broadcast what the lanes above it computed, '
            'and assigns and stores the result under `if (lane < 1)`.  With '
            'its optimizer on, the Intel device compiler loses part of what '
            'those lanes contribute: on the Max 1550, 8 to 16 values an '
            'element of the last derivative and of the integral that adds it '
            'up, at orders 4 and 6 in both precisions, ahead of time and '
            'through the JIT alike, with every predicated load already split.  '
            '`-cl-opt-disable` computes it, and so does this, both ways; '
            'spelling the broadcast `select_from_group`, or turning off IGC\'s '
            'code sinking, does not.\n'
            'What it costs, over the twenty elastic kernels at orders 4 and 6 '
            'in both precisions: 0.3 % more time in geometric mean, from 13 % '
            'less (the order-4 derivative in double, which was wrong before) '
            'to 28 % more (`volume` at order 4 in double).\n'
            'Intel only.  Elsewhere a guard around a block of arithmetic is '
            'one branch, and this makes it one per statement that writes.')

declare('inline_pir_values',
        default=True,
        parse=parse_bool,
        doc='Write a pure single-use value straight into its consumer instead '
            'of naming it.\n'
            'Off, every structured operation leaves a named temporary and the '
            'source grows -- which is the reason it is on.  It is a question '
            'and not a constant because the Intel device compiler reads our '
            'nesting: hoisting one predicated load out of a conditional '
            'expression moved `o6d:volume` by 2.2x and the suite by 1.13x '
            '(`split_predicated_load`), so how deeply the rest nests is worth '
            'asking about rather than assuming.  The answer on pvc is no: '
            'naming everything grows `elastic-linearck` order 4 and 6 from '
            '1116 to 1736 lines and the twenty kernels to 0.987x of the '
            'default, so what the compiler minds is a *predicated* load and '
            'not how deeply an expression nests.')

declare('register_blocks',
        default=False,
        env='TF_REGISTER_BLOCKS',
        parse=parse_bool,
        doc='Declare a register image as blocks -- one buffer per run of its '
            'widest access, a column of the image -- where every access has a '
            'known offset inside one block, the short loops it is addressed '
            'through written out first so that the offsets are known '
            '(`pir.blocks`).  A whole image is one variable to the compiler: '
            'under the explicit vector one `simd` of its volume, which VC '
            'spills and fills in pieces around every `mad` once it is past '
            'the register file, and each column read is a copy out of it.  '
            'As blocks, the allocator can keep the live columns and spill the '
            'cold.  Under SPMD an access is one element of a lane\'s array, '
            'and the blocks are one-element arrays.\n'
            'Measured on PVC with the emitter-level prototype, under the '
            'explicit vector: `volume` spilled 60 to 95 % fewer scratch '
            'messages, and `localFluxAll` got worse.')

declare('register_values',
        default=False,
        env='TF_REGISTER_VALUES',
        parse=parse_bool,
        doc='Take the blocks of `register_blocks` to values -- which it '
            'implies: a store is what the block holds from there on, a load '
            'what it holds where the load stands, a narrower access a part '
            'inserted or sliced, a predicated one a select, and what a branch '
            'or a loop that stays stores a result of it or an argument of the '
            'loop (`pir.values`).  No block is memory then, and which column '
            'is live where is what the body says.  An option of its own, so '
            'that the arrays and the values can be measured apart, and the '
            'values switched off where they cost.')

declare('register_spill',
        default='off',
        env='TF_REGISTER_SPILL',
        parse=parse_register_spill,
        doc='What the body does with a value it cannot keep in registers where '
            'more is live than the budget (`register_budget`): `off`, nothing '
            '-- the compiler spills as it does; `remat`, a value that is a '
            'load of memory the kernel does not write, or a constant, is had '
            'again before its next read instead of kept across the gap, the '
            'one read again furthest ahead first (`pir.budget`); `shared`, '
            'that and then a value computed in the body is stored once into a '
            'slot of shared memory of its own lane and loaded from there '
            'before a read the gap holds it from (`pir.budget.spill_shared`, '
            '`register_shared`).  Implies '
            '`register_values`: '
            'the values a column holds are what is decided over.  Named for '
            'what the body does in place of the compiler\'s spill, and a '
            'choice of how, because the ways differ in what they cost: '
            'loading again costs a load and no memory.')

declare('register_budget',
        default=0,
        env='TF_REGISTER_BUDGET',
        parse=parse_int,
        doc='Bytes the register model (`pir.pressure`) may count live in a '
            'lane -- a work-item under the explicit vector -- before '
            '`register_spill` acts, or 0 for the target\'s own '
            '(`abstract_instruction.register_budget`).  In the model\'s own '
            'bytes, not the compiler\'s registers, so that a calibration '
            'moves it without a change of code.')

declare('register_shared',
        default=0,
        env='TF_REGISTER_SHARED',
        parse=parse_int,
        doc='Bytes of shared memory a multiplication may take for the values '
            '`register_spill=shared` spills there, or 0 for the target\'s own: '
            'what the multiplications the target keeps resident leave each '
            'without lowering their number (`abstract_instruction.'
            'shared_spill_bytes`).  More than that trades resident '
            'multiplications for registers.'),

declare('register_shared_volatile',
        default=True,
        env='TF_REGISTER_SHARED_VOLATILE',
        parse=parse_bool,
        doc='Access the slots `register_spill=shared` spills to as `volatile`: '
            'where the body stores and loads them, as often as it does.  Off, '
            'a compiler may take a load back from the store or from the load '
            'before -- IGC 2.41.5 kept 1665 of the order-8 dense `volume`\'s '
            '5245 loads back and 1284 of its 1928 stores -- and keep the '
            'register after all; on, it spilled 22720 B besides, against '
            '15232 off (8 KB of slots, listing).'),

declare('shared_operands_in_place',
        default=False,
        env='TF_SHARED_OPERANDS_IN_PLACE',
        parse=parse_bool,
        doc='Read a temporary that a shared buffer holds, where its '
            'multiplication reads it by broadcast -- it carries none of the '
            'destination\'s lead index -- from the buffer: one load hands '
            'every lane the element, and no register image of it is held to '
            'broadcast from (`MultilinearBuilder._make_load_op`).  SeisSol\'s '
            '`volume` holds `I x star(d)` there for the product with '
            '`kDivM(d)`.  Trades a load per element read for the registers of '
            'the image: on the Max 1550 a read of shared memory took 2.89 to '
            '2.99 ms against 2.60 for the split register broadcast at eight '
            'FMAs a broadcast (microbenchmark).'),

declare('register_left_over',
        default=-1.0,
        env='TF_REGISTER_LEFT_OVER',
        parse=parse_float,
        doc='The share of the excess over the register budget of a body as '
            'written -- under SPMD on Intel, of its values -- that may be left '
            'once its blocks are values and every load that pays is put back, '
            'for `register_spill` to put any back (`pir.budget.CLOSED_MIN`, '
            '`VALUES_CLOSED_MIN`); negative for the default, 1 or more to put '
            'back wherever a load pays.  For calibrating the stop against '
            'timings.')

declare('register_selects',
        default='auto',
        env='TF_REGISTER_SELECTS',
        parse=parse_bool_or_auto,
        doc='With `register_values`, whether a register block a store under a '
            'predicate writes becomes a value all the same, the store a select '
            'between what it stores and what the block held.  `auto`: '
            'everywhere but DPC++\'s SPMD lowering on Intel, where the block '
            'stays memory -- IGC 2.40.13 to 2.42.6 computed what feeds a '
            'broadcast only in the lanes such a select keeps -- although what '
            'feeds it is pinned there (`crosslane_pins`), select or branch '
            'alike.')

declare('register_unguard',
        default=False,
        env='TF_REGISTER_UNGUARD',
        parse=parse_bool,
        doc='With `register_values` under DPC++\'s SPMD lowering on Intel, let '
            'a store into a register block whose lane guard keeps nothing any '
            'read takes write every lane (`pir.values.unguard_register_'
            'stores`): fewer selects, and fewer values entering a guard to '
            'pin (`crosslane_pins`).  Off by default: on the Max 1550 (IGC '
            '2.41.5, 262144 elements, job 561193) it cost every load that pays '
            'put back with selects 3.6 % in geometric mean over SeisSol\'s '
            'elastic orders 6 and 8 and viscoelastic order 6 -- the order-8 '
            'dense `volume` +63 % against +2 % -- and the stop rule\'s order-6 '
            'dense `volume` +9 % against -13 %, though it took the order-6 '
            'sparse `derivative` as values from +23 to +12 %.'),

declare('crosslane_pins',
        default='use',
        env='TF_CROSSLANE_PINS',
        parse=parse_crosslane_pins,
        doc='Where, with `register_values` under DPC++\'s SPMD lowering on '
            'Intel, a value that comes of a cross-lane read and enters a lane '
            'guard is pinned by an empty volatile asm: right ahead of the '
            'first statement that reads it under the guard (`use`), right '
            'behind its definition (`definition`), or nowhere (`none`, which '
            'IGC 2.40.13 to 2.42.6 computes wrong: `pir.values.'
            'pin_crosslane_sources`).  For measuring what the pins cost.')

declare('preload_partial',
        default=False,
        parse=parse_bool,
        doc='With `preload_globals`, stage the `Addressing.NONE` operands that '
            'fit into shared memory rather than all of them or none.\n'
            'Taken first-fit in the order the kernel declares them, under the '
            'block\'s limit; the rest are read from global memory.  Where the '
            'staged ones leave no room for one multiplication, the last one '
            'taken is dropped and the section built again, one at a time, '
            'instead of dropping them all.  `local_flux` at b = 80 has four '
            '25.6 KB operators against 64 KB on gfx942, and stages none of '
            'them without this.')

declare('inline_constants',
        rule=lambda target: target.prefs.inline_constants,
        parse=parse_int,
        doc='Most non-zero entries of a batch-constant broadcast operand -- `B` '
            'in `C = A B` -- whose numbers the description carries, for the '
            'kernel to take them as literals; 0 takes none.\n'
            'An FMA then reads the number as an immediate, and a zero drops its '
            'product.  Only where every reduction reading the operand unrolls '
            'whole (no `k_roll`, no extent over `k_unroll_max`): a literal has '
            'no address a loop could index.  The interface keeps the pointer '
            'and does not read it, as with `argument_constants`, which takes '
            'what this leaves.\n'
            'Large on NVIDIA, where a 32-bit immediate fits the instruction: in '
            'a chain of four 9x9 operators on sm_120 the kernel had 768 '
            'instructions instead of 888 by value, and ran as fast as by value '
            '-- both about 1.5 % ahead of memory; the chain is memory-bound.  '
            'Small elsewhere: on AMD a literal is another dword per instruction '
            'or a scalar register, of which there are about a hundred -- the '
            'same chain with its 81-entry operators as literals was 6 % slower '
            'on gfx1150 than read from the constant space; Intel is unmeasured.')

declare('argument_constants',
        rule=lambda target: target.prefs.argument_constants,
        parse=parse_bool,
        doc='Pass a batch-constant operand every product reads as a scalar -- '
            '`B` in `C = A B` -- by value, with the numbers the description '
            'carries for it, instead of reading the buffer the caller passes.\n'
            'The numbers then live where kernel arguments do: the constant '
            'bank on NVIDIA, which an FFMA takes an operand from (sm_120: '
            'LDCU into a uniform register, and no LDG).  The interface does '
            'not change -- the launcher still takes the pointer and does not '
            'read it -- so this is only right where the description\'s numbers '
            'are the buffer\'s, as they are for a constant matrix.  What does '
            'not fit `max_argument_size` stays in memory, first fit in '
            'declaration order.\n'
            'NVIDIA only by default: on AMD the same read out of device memory '
            'through the constant space is a scalar load already and measured '
            'as fast or faster; Intel is unmeasured.')

declare('preload_roles',
        default='all',
        parse=parse_str,
        doc='Which `Addressing.NONE` operands `preload_globals` stages: `all`; '
            '`broadcast` -- only those no operation reads along the lead '
            'index, `B` in `C = A B`; or `lead`, the others.\n'
            '`lead` is the one for AMD\'s constant space: a broadcast operand '
            'left in memory is read at one address by every lane, which is a '
            'scalar load into the register a `v_fma` takes as it is.\n'
            'Those are a scalar to every product, and staged each element is '
            'one broadcast read from shared memory.  The operands read along '
            'the lead are the ones whose staging can cost the block its '
            'occupancy: on sm_120 staging all of them made `local_flux` 12 % '
            'faster and the damage kernel, with three 14 kB `kDivM`, 23 % '
            'slower.  The rest are read from global memory, as '
            '`preload_partial` leaves them.')

declare('preload_shards',
        rule=lambda target: target.prefs.preload_shards,
        parse=parse_bool,
        doc='Keep what fits of the batch-constant operands read from global '
            'memory inside the batch loop in the block\'s shared memory, a '
            'shard at a time, instead of all of an operand or none of it '
            '(`preload_globals`, `preload_partial`).\n'
            'A shard is what one load reads across the lanes of a '
            'multiplication -- a lane block of a column -- and the block '
            'copies the shards that fit in ahead of the loop, those whose '
            'loads stand nearest their readers first (`pir.shards`).  What '
            'fits: as much as a block can take more and an SM still hold as '
            'many blocks, by shared memory and threads '
            '(`Generator._shard_budget`), or `preload_shard_budget`.\n'
            'On by default on AMD (`Target.prefs`): at order 8 in double '
            'precision no operator fits a block whole, and on the MI300A the '
            'shards took SeisSol\'s `volume` from 42.5 to 33.1 ns an element and '
            'its `derivative` from 166 to 150, with nothing at order 6 slower.  '
            'Elsewhere unmeasured.')

declare('preload_shard_budget',
        default=0,
        parse=parse_int,
        doc='Bytes of shared memory a block\'s shards (`preload_shards`) may '
            'take, up to what the block can take at all, in place of what '
            'keeps the blocks an SM holds -- which counts no registers, so '
            'where they bind first, as on AMD at order 8, a block can take '
            'more than that at no cost.  0 is that default.')

declare('sycl_grf',
        default='default',
        parse=lambda text: text.strip(),
        doc='The register file a SYCL kernel on Intel asks IGC for, as a kernel '
            'property (`grf_size`, DPC++ only): `128` or `256` general registers '
            'a thread, `auto` for IGC to choose per kernel '
            '(`grf_size_automatic`), `default` for no property -- IGC\'s small '
            'file under SPMD, and 256 under the explicit vector, which is sized '
            'for it.  256 halves the threads in flight and spills less: in '
            'SeisSol on PVC (oneAPI 2025, elastic order 6) projectToDR went from '
            '1477 to 397 us a call and the derivative from 294 to 191, but the '
            'volume from 187 to 202.')

declare('esimd_unroll_fence',
        default=0,
        env='TF_ESIMD_UNROLL_FENCE',
        parse=parse_int,
        doc='The mask of an `esimd::fence` written between the trips of a loop '
            '`register_blocks` writes out, under the explicit vector only; 0 '
            'for none.  128 is vISA\'s software fence, a barrier to the '
            'scheduler that emits nothing.')

declare('sycl_split_broadcast',
        default=2,
        env='TF_SYCL_SPLIT_BROADCAST',
        parse=parse_int,
        doc='Under SPMD SYCL with a stated sub-group (oneAPI on Intel), how many '
            'multiplications a sub-group may hold for their broadcasts to be '
            'split: one `group_broadcast` per multiplication, each from a lane '
            'the whole sub-group agrees on, and a select by the work-item\'s '
            'own (`tensorforge::split_broadcast`, '
            '`tensorforge_device/sycl_spmd.h`).  Beyond it, and at 0, a '
            'broadcast reads at an index per work-item (`select_from_group`), '
            'which IGC writes as an indirect register access.  On the Max 1550 '
            '(IGC 2.41) two a sub-group -- eight lanes a multiplication -- '
            'split ran the order-4 kernels of eight fused simulations up to '
            '62 % faster than the default lanes, where the indirect read had '
            'gained only in `localFluxAll` in double precision; four a '
            'sub-group split lost to the default almost everywhere, at four '
            'broadcasts and three selects a read.')

declare('spmd_block_runs',
        rule=lambda target: target.prefs.spmd_block_runs,
        env='TF_SPMD_BLOCK_RUNS',
        parse=parse_bool,
        doc='Under SPMD SYCL with a stated sub-group (oneAPI on Intel), read a '
            'run of a multiplication\'s reads -- of one global buffer, at '
            'offsets one sub-group apart, a lane each -- as one sub-group '
            'block read (`sub_group::load`, `tensorforge::subGroupRun`, '
            '`pir.block_reads`): work-item `l` gets element `l` of each read '
            'in order, where every read was a message of its own.  Only where '
            'the multiplication is the sub-group, for up to eight reads and '
            'a power of two of them, inside the buffer\'s own extent.  The '
            'explicit vector reads its runs as one block message already.  '
            'On under DPC++\'s SPMD lowering for Intel (`oneapi`), where it '
            'was measured: 0.95x the time over 41 SeisSol kernels on the Max '
            '1550, every result bit-identical (`Preferences.spmd_block_runs`).')

declare('sycl_dpas',
        default=False,
        env='TF_SYCL_DPAS',
        parse=parse_bool_or_auto,
        doc='Compute single-precision products of a batch-constant operator '
            'with DPAS under DPC++\'s SPMD lowering on the Max 1550 (pvc), '
            'sixteen lanes a multiplication (`primitives/intel_spmd.py`): the '
            'operator as the instruction\'s B, its rows over the lanes, the '
            'element\'s operand as A, eight output columns an instruction, '
            'each product three TF32 ones (3xTF32, about F32\'s accuracy).  '
            'The accumulator is the lowering\'s own register image -- lane '
            'a row, the columns in registers -- so a product handed on to '
            'the next is not laid out again; the element\'s operand is, by '
            'one exchange of the sub-group per pair of columns and lane slot.  '
            'Needs IGC 2.x: the call is the OpenCL builtin '
            '`intel_sub_group_tf32_tf32_matrix_mad_k8`, which IGC 1.0.15468 '
            'builds without a word and leaves undefined, so that the kernel '
            'stops at its first launch; the generator asks the `ocloc` on '
            'PATH and stops with the reason instead '
            '(`intel_spmd.resolved`).  A probe on SNG2 (icpx 2025.2.1, IGC '
            '2.41.5) put a chain of four dense 56 x 56 products on '
            'register-resident data 30 to 38 % ahead of the FMA path, and a '
            'single product, which is bound by memory, even.  On SeisSol\'s '
            'kernels in single precision (Max 1550, batch 262144, ns an '
            'element, the FMA path against DPAS, with operator copies, the '
            'default under it): elastic order 6 derivative 27.2 / 19.9, '
            '`localFluxAll` 25.5 / 17.6, `neighboringFlux[0]` 11.2 / 7.3, '
            '`volume` 16.5 / 12.8; order 8 150.0 / 125.6, 81.3 / 72.0, 20.3 '
            '/ 15.4, 62.1 / 32.9; viscoelastic order 6 derivative 340.5 / '
            '556.3, `fluxLocalAll` 90.2 / 59.1, `neighborFluxExt[0]` 11.0 / '
            '12.3, `volumeExt` 23.2 / 13.2; order 8 1843 / 3896, 335.8 / '
            '174.0, 38.4 / 31.0, 87.9 / 79.2.  Off by default.  `auto` lets '
            '`autotune=lanes` decide per kernel, with the fence '
            '(`tuning._dpas_candidates`); without that rule it builds the '
            'FMA path.')

declare('sycl_dpas_min_depth',
        default=16,
        env='TF_SYCL_DPAS_MIN_DEPTH',
        parse=parse_int,
        doc='Contraction steps below which a product stays off the SPMD DPAS '
            'path (`sycl_dpas`).  The instruction contracts eight steps, and '
            'the element\'s operand is laid out for it whatever the '
            'operator holds, so a short contraction pays the layout for '
            'little arithmetic: SeisSol\'s 9 x 9 star products would be '
            'seven sixteenths padding.')

declare('sycl_dpas_fence',
        default=False,
        env='TF_SYCL_DPAS_FENCE',
        parse=parse_bool,
        doc='Under `sycl_dpas`, read each operator tile one ahead of its '
            'products and put a fence of the sub-group between one tile\'s '
            'products and the next (`Lexic.pin_fence`, which IGC moves no load '
            'across).  IGC otherwise reads a product\'s tiles as early as it '
            'can, which costs registers -- and while the fragments were '
            '`sycl::vec`, spills: the fence took SeisSol\'s order-6 kernels in '
            'single precision from 1 746 - 8 554 spill references to 750 - '
            '5 531.  In OpenCL\'s vectors it is a question per kernel (Max '
            '1550, IGC 2.41.5, batch 262144, ns an element without and with '
            'it): elastic order 6 derivative 19.9 / 28.9, `localFluxAll` '
            '17.6 / 16.5, `neighboringFlux[0]` 7.3 / 7.4, `volume` 12.8 / '
            '13.5; order 8 125.6 / 118.6, 72.0 / 77.2, 15.4 / 16.9, 32.9 / '
            '44.1; viscoelastic order 6 `fluxLocalAll` 59.1 / 54.1, '
            '`volumeExt` 13.2 / 13.4; order 8 174.0 / 246.6, 79.2 / 76.5 -- '
            'off, which `sycl_dpas=auto` under `autotune=lanes` answers per '
            'kernel.')

declare('lead_split',
        rule=lambda target: (2 if (target.hw.vendor == 'intel'
                                   and target.backend == 'oneapi'
                                   and not target.explicit_simd) else 1),
        env='TF_LEAD_SPLIT',
        parse=parse_int,
        doc='Blocks of a second destination axis a multiplication spreads over '
            'the lanes its lead leaves idle (a two-level lead): lane `t` holds '
            'lead row `t % R` and the rows `b, b + B, ...` of axis 1, `b = t // '
            'R`, where `R` is the lead extent and `B` this number.  Where a '
            'section runs more lanes than an operation\'s lead fills -- the '
            'eight simulations of a fused build in the sixteen lanes '
            '`localFluxAll` runs -- every lane works, and a lane holds `1/B` '
            'of what it held before.  The lanes are the section\'s: a section '
            'whose lead fills its lanes is left alone, since two '
            'multiplications a sub-group fill it already '
            '(`Options.sycl_split_broadcast`).  A product that contracts over '
            'the split axis reads the other blocks by exchange '
            '(`Lexic.exchange_xor`: SPMD SYCL).  1 is off; 2 is what is built '
            '(`MultilinearBuilder._split_axes`), and the default under DPC++\'s '
            'SPMD lowering: on the Max 1550 (IGC 2.41.5) the s8 kernels of '
            'SeisSol\'s elastic system at orders 3 and 4 ran 0.79 times their '
            'time without it in geometric mean at a batch of 262144, '
            '`localFluxAll` 0.27-0.38 and `neighboringFlux` 0.52-0.81 '
            '(e3-d +9 %), the rest unchanged (sng2 560412).  Not in a '
            'multiplication narrower than the vector unit, whose sub-group '
            'another one fills: the ten single-simulation kernels that led '
            'with four rows in eight lanes ran up to 21 % slower split '
            '(sng2 560417).')

declare('stream_shards',
        rule=lambda target: target.prefs.stream_shards,
        parse=parse_float,
        doc='The share of the shards\' budget (`preload_shards`) that is a room '
            'for what of the batch-constant operands does not fit for the '
            'whole batch loop: copied in there by the block for every element, '
            'a product at a time, each ahead of the product that reads it '
            '(`pir.shards._rest`); the block is then one group on the same '
            'elements.  0 holds what fits and leaves the rest in global memory.  '
            'Only where the operators are more than the block\'s shared memory '
            '(or `preload_shard_budget`); 0.25 by default on AMD '
            '(`Target.prefs`).')

declare('stage_members',
        rule=lambda target: target.prefs.stage_members,
        parse=parse_bool,
        doc='In a merged run (`merge_variants`), copy each iteration\'s '
            'batch-constant operand into a block-shared buffer and read it '
            'there, rather than every multiplication reading it from global '
            'memory.\n'
            'The operators a merged run walks are exactly the ones `preload_globals` '
            'cannot stage -- its members are selected per iteration and are read '
            'from global memory -- so the block copies the current one '
            'cooperatively, behind a block barrier and in front of one, and its '
            'multiplications share it.  Needs every multiplication of the block '
            'on the same trips through the batch loop: the block is one group '
            'of `stage_group` multiplications.\n'
            'On by default on AMD (`Target.prefs`): on the MI300A SeisSol\'s '
            'merged `localFluxAll` took 13.2 ns an element instead of 19.6 at '
            'order 6 in double precision, 7.0 instead of 8.0 in single, and '
            '54.9 instead of 75.2 at order 8.  Elsewhere unmeasured.')

declare('split_member_copies',
        rule=lambda target: target.prefs.split_member_copies,
        parse=parse_bool,
        doc='Under `stage_members`, split the copy of a trip\'s members: its '
            'stores -- and the two block barriers around them -- just ahead '
            'of the first statement of the trip that reads the copy, and its '
            'loads up to `split_member_lead` ahead of that (`pir.shards`).  '
            'What the trip computes in between goes on while the loads are in '
            'flight: SeisSol\'s `localFluxAll` multiplies `I` by the face\'s '
            '`AplusT` first, which is no member.  Where a trip has little '
            'ahead of its first read (`shards.SPLIT_COVER`), the copy stays '
            'whole.\n'
            'On by default on AMD (`Target.prefs`): on the MI300A '
            '`localFluxAll` took 6.77 ns an element instead of 6.96 at order 6 '
            'in single precision, 13.04 instead of 13.12 in double, 24.81 '
            'instead of 25.06 at order 8 in single and 48.16 instead of 47.88 '
            'in double; at order 4 the same.  Elsewhere unmeasured.')

declare('split_member_lead',
        default=192,
        parse=parse_int,
        doc='How far ahead of the first read of the copy a split member copy '
            '(`split_member_copies`) issues its loads at most, in the units '
            'the emitter counts code in (`issue_cover`); 0 is the head of the '
            'trip.  Further ahead than a load\'s latency needs, the loads only '
            'hold their registers: SeisSol\'s `localFluxAll` at order 8 in '
            'single precision on the MI300A, with 478 of them between the '
            'head and the first read, took 25.46 ns an element with its loads '
            'at the head and 24.81 at most 96 or 192 ahead, where the copy '
            'whole took 25.06.')

declare('rotate_slots',
        default=0,
        parse=parse_int,
        doc='Buffers a merged run\'s carried temporaries turn through, or 0 '
            'for one more than the distance the run carries over.\n'
            'A run that selects a temporary per iteration reaches it through '
            'a table of addresses, and a buffer per member would put all of '
            'them live at once -- 35 temporaries worth 89 kB a '
            'multiplication on SeisSol\'s anelastic derivative, which the '
            'allocator aliases down to 8 when the run is written out.  The '
            'members take the buffers in turn instead, which needs as many '
            'as the carry is deep: the value a pass writes is read '
            '`distance` passes later and never after, so `distance + 1` '
            'suffice and the fifth derivative is written where the third '
            'was.\n'
            'Fewer than that is a trade, not an error, and 1 is the far end '
            'of it: a pass then reads and writes one buffer, which is '
            'correct wherever the read of a trip completes before its write '
            '-- a multiplication accumulates in registers, so it does -- and '
            'costs the barrier between them.  On the order 6 anelastic '
            'derivative it took the block from 80 kB of shared memory to '
            '53 kB for two lines of code.  More than `distance + 1` buys '
            'nothing and is allowed only because refusing it would be a '
            'rule with no case behind it.')

declare('carry_trips',
        default=True,
        parse=parse_bool,
        doc='Keep in registers what the trips of a merged run hand each other '
            'through memory: an accumulator loaded at the head of a trip and '
            'stored at its tail is loaded once ahead of the loop and stored '
            'once behind it.  Only where every lane is shown to read back what '
            'it stored, and nothing else in the loop touches the buffer '
            '(`pir.carry`).  Every merged run hands its accumulators over '
            'that way (`Generator._emit_closed_loop`).')

declare('forward_across',
        rule=lambda target: target.prefs.placement.read_across,
        parse=parse_bool,
        doc='Read a load of what one other lane stored from that lane\'s '
            'register, handed across the lanes, where the lane holds the '
            'register still and every lane of the multiplication runs the '
            'load -- within one exchange of the lanes (`pir.forward`).  A '
            'broadcast instead of a store, a barrier and a load, and nothing '
            'of the arena.\n'
            'The vendor table decides by default -- AMD yes, NVIDIA no: an '
            'operand NVIDIA stages through shared memory, one element of each '
            'lane read by every lane, would become a shuffle per element.')

declare('keep_own',
        rule=lambda target: target.prefs.placement.keep_own,
        parse=parse_bool,
        doc='Keep a buffer of the multiplication\'s own shared memory in '
            'registers where every lane reads of it only what it wrote '
            'itself: one register of a lane\'s own per element it reaches, '
            'over the statements that touch the buffer, and the arena '
            'without it (`pir.carry.keep_own`).  Shared memory is for what '
            'lanes hand each other; a buffer no lane hands on costs a store, '
            'a load and its bytes of the arena, and registers instead.\n'
            'The vendor table decides by default -- AMD and NVIDIA yes, '
            'Intel no.')

declare('forward_memory',
        default=True,
        parse=parse_bool,
        doc='Take a load from where its own lane\'s store took the value: a '
            'result stored to memory and loaded back reads the registers it '
            'was stored from, where they hold it still, and a register image '
            'copied into another is read where the copy read it.  A store '
            'nothing reads then goes.  Asked lane by lane and trip by trip '
            '(`pir.forward`); an element\'s mask is taken to hold where the '
            'body keeps its contract (`elementmask`).')

declare('amd_setprio',
        default='',
        env='TF_AMD_SETPRIO',
        doc='On AMD, wave priorities (`s_setprio`) in the kernel body '
            '(`pir.priority`): `loads` raises a wave while it issues a '
            'transfer, `compute` while it does not, `alternate` every other '
            'wave for the whole kernel; empty for none.')

declare('buffer_transfers',
        default=False,
        parse=parse_bool,
        doc='On AMD, read and write global memory through buffer instructions '
            'where the base is the same on every lane of the wave -- a '
            'batch-constant operator, an element\'s binding where a '
            'multiplication is a whole number of waves -- and the access 4, '
            '8 or 16 bytes of 32- or 64-bit numbers with no cache hint '
            '(`pir.buffers`).  A buffer access past its resource reads zero '
            'and writes nothing, so a guard is an offset past it: no select, '
            'no clamped address and no branch around a store.  With '
            '`amd_async_copies` on gfx90a, gfx94x and gfx950, the LDS-DMA '
            'staging copies of such a base read through its resource as well '
            '(`buffer_load_dword ... lds`), their guards still branches: a '
            'lane held off by its offset would write zero into the LDS.  Off '
            'until measured: the guarded loads of 48 kernels patched to this '
            'form ran 1.6 % faster on MI300A at 64 lanes in the geometric '
            'mean, from 12.6 % faster to 23.3 % slower (2026-11-06).')

declare('pin_loads',
        rule=lambda target: target.prefs.pin_loads,
        parse=lambda text: 'auto' if text.strip() == 'auto' else int(text),
        doc='On NVIDIA, hold ptxas to the body\'s order: no chain inlined '
            'into one expression, and a `__syncwarp()` -- a NOP in converged '
            'code, and a point no memory access is moved across -- in front of '
            'every this many loads.  ptxas otherwise schedules a body as it '
            'likes, whatever order the text has, and its demand follows the '
            'count of `pir.pressure` only loosely (66 + 0.83 x, 29 registers '
            'off on average, on sm_90).  0 is off, and the default off NVIDIA, '
            'where it does nothing; `\'auto\'`, the default on NVIDIA '
            '(`Preferences.pin_loads`), fences every 32 '
            'where ptxas would spill the body by that estimate -- and under '
            '`autotune=lanes` where ptxas does spill it, 192 B of spill '
            'traffic a lane or more, and pinned spills at most half of that '
            '(`tuning.PIN_SPILL`).  Whole-warp multiplications only.')

declare('reassociate_products',
        rule=lambda target: target.prefs.reassociate_products,
        parse=parse_bool,
        doc='Write ``T = X S; Q (+)= A T`` as ``U = A X; Q (+)= U S`` where '
            'both are plain matrix products, T is a temporary nothing else '
            'reads and the operator A has no more rows in its box than X: '
            'yateto puts the per-element product first, and in the other '
            'order it runs over the operator\'s rows -- 24 behind a face '
            'matrix instead of 56 -- and the operator products of a step all '
            'read the same X (`generators.reassociate`).  On by default on '
            'AMD (`Preferences.reassociate_products`).')

declare('amd_fused_transpose',
        rule=lambda target: target.prefs.amd_fused_transpose,
        parse=parse_bool,
        doc='On gfx9, transpose the 4x4 quads of the lane-batched MFMA path '
            'with `transpose4x4b32Fused`: eight `v_cndmask_b32_dpp` with the '
            'lane mask in VCC -- the assembly `hip.h` kept in a comment, now '
            'one block behind one wait for the DPP hazard -- instead of eight '
            'DPP moves and eight selects.  MI300A, SeisSol F32: derivative '
            '-3 to -16 %, localFluxAll -4 to -15 % over 16 to 64 lanes.  On '
            'by default on AMD (`Preferences.amd_fused_transpose`); off gfx9 '
            'nothing reads it.')

declare('amd_matrix_k4',
        default=False,
        parse=parse_matrix_k4,
        doc='On CDNA -- where the instruction is: from gfx908 on '
            '(`mai-insts`) in single precision, from gfx90a on '
            '(`gfx90a-insts`) in double and for the emulation --, compute a '
            'product of a batch-constant operator '
            'and a per-element operand at 16 lanes a multiplication through a '
            '16x16x4 instruction -- `v_mfma_f64_16x16x4f64` in double '
            'precision, `v_mfma_f32_16x16x4f32` in single in place of the '
            'lane-batched `4x4x1` tile -- the four multiplications of a wave '
            'stacked along the instruction\'s M (`primitives.amd.stacked`): '
            'its D lands in the lanes and slots the register image already '
            'has, and the columns past the last four go through the DPP '
            'chain.  `transpose` builds the per-element fragment out of the '
            'image\'s registers with lane gathers, `shard` reads it out of the '
            'shared memory the operand already sits in, and `true` takes the '
            'shard where the operand is there and the gathers where it is '
            'not.  Bit-equal on MI300A to the DPP chain in double precision '
            'and to the lane-batched tile in single.  There, at sixteen lanes '
            'in double precision against the chain: order 6 `volume` -8 % '
            'with the gathers and the ninth column padded, -20 % with the '
            'products chained besides (`mma_d_layout`) and the operators left '
            'in global memory, which leaves room for two blocks a CU -- '
            'preloaded, they and the shard copies leave room for four '
            'multiplications a block, +61 % -- `localFluxAll` -7 to -10 % '
            'with `true`, the time derivative -24 % padded; order 4 '
            '`localFluxAll` -11 %; order 8 behind the chain at 32 lanes '
            'throughout.  In single precision against the lane-batched tile, '
            'chained: `volume` -13 to -16 % at order 4, -4 % at order 6 and '
            '-5 % at order 8 against 32 lanes, `localFluxAll` -10 % at order '
            '6 with the gathers; unchained within 5 % of it either way.\n'
            'At 32 and 64 lanes a multiplication spans two or four of the '
            'instruction\'s quarters, each taking four more columns of an '
            'issue, and the rows of an issue\'s tile go back to their lanes by '
            'lane gathers (`stacked._relayout`) -- or stay in D, where the '
            'products are chained.  There, chained, on the shard and against '
            'the chain at the same lanes: order 8 `volume` -10 % at 32 lanes '
            'in single precision, +15 % at 32 and -12 % at 64 in double, where '
            '32 lanes of the chain stay the fastest build; order 6 `volume` in '
            'double -22 % at 32 lanes, still behind 16.  Off by default.')

declare('amd_matrix_k4_pad',
        default=False,
        parse=parse_bool,
        doc='Under `amd_matrix_k4`, take the columns past the last whole tile '
            'of four into one more tile of the instruction, padded with zeros, '
            'instead of through the DPP chain: SeisSol\'s ninth column then '
            'costs half again the issues and no chain beside them, which reads '
            'the operator a second time, a row and a step at a time.  MI300A, '
            'order 6 in double precision: the time derivative 15.2 -> 11.1 ns '
            'an element (no spill left), `volume` -7 %; at order 4 '
            '`localFluxAll` and the derivative give back what the unpadded '
            'tiles gained.  Off.')

declare('amd_matrix_k4_emulate',
        default=False,
        parse=parse_bool,
        doc='Under `amd_matrix_k4`, emulate a single-precision product in BF16 '
            'terms: both operands split into three, six term products on '
            '`v_mfma_f32_16x16x16bf16_1k` -- the F32 tile\'s M, N and D, sixteen '
            'steps an issue -- into the F32 accumulator (`primitives.amd.'
            'stacked._emit_emulated`).  Sixteen lanes a multiplication.  With '
            '`prepare_operands` the operator is split on the host, k-quads of '
            'its three terms (`amd.prepared_order`), and every reader that '
            'multiplies its floats adds them back.  As accurate as the F32 '
            'paths in SeisSol (1e-7 in norm) and slower than both on MI300A: '
            'the `volume` +13 % at order 4, +45 % at order 6 and +74 % at '
            'order 8 against the fastest F32 build -- the BF16 instruction '
            'does six times the F32 one\'s work an issue, and six products '
            'and the splits take that back.')

declare('amd_matrix_k4_pipeline',
        default=0,
        env='TF_AMD_MATRIX_K4_PIPELINE',
        doc='Under `amd_matrix_k4`, read the fragments of the next step of the '
            'contraction before the issues of this one (1), and interleave '
            'those reads with the issues by scheduling groups (2, '
            '`__builtin_amdgcn_sched_group_barrier`, as CK does); 0 reads a '
            'step\'s fragments in front of its own issues.  The order-6 '
            '`volume` in double precision on MI300A waited on dependencies '
            '49 % of its cycles and on issue 30 %, its matrix core busy 23 %; '
            'at 1 the dependency waits fall from 8.7e9 to 3.7e9 cycles and '
            'the issue waits from 5.3e9 to 3.1e9, but the second set of '
            'fragments halves the waves in flight.  Order 6 at sixteen '
            'lanes: `volume` -7 % on the shard at 1 and +5 % with the '
            'gathers, `neighboringFlux` -2 to -7 % at 2, the time derivative '
            '+1 % at 1 and +28 % at 2.')

declare('amd_moved_broadcast',
        default=False,
        parse=parse_bool,
        doc='On CDNA (gfx90a, gfx942, gfx950), take the row share of a DPP '
            'product as a move of its own -- `movdpp16`, a builtin -- and '
            'plain FMAs behind it, where the moved value feeds four products '
            'or more (`select.MATERIALIZE_FROM`), instead of the modifier on '
            'every FMA.  The fused form is inline assembly there, one wait in '
            'front of every block, and nothing can be scheduled into a block: '
            'four steps of three 56x56 by 56x9 products on MI300A took 40.1 ns '
            'an element fused in double precision and 28.1 with the moves (one '
            'per product and row), 26.3 against 21.3 in single.')

declare('amd_row_lookahead',
        default=-1,
        env='TF_AMD_ROW_LOOKAHEAD',
        parse=parse_int,
        doc='How many rows ahead the fused DPP chain in the row order '
            '(`codegen._fused_rows`) reads its lead operand.  0 reads each '
            '`A(i, k)` at its own row, behind the pins that end the row before '
            '(`tensorforge::pin`, volatile assembly, which orders memory as '
            'well), so an LDS read waits its whole latency every row.  Each row '
            'ahead costs the slots\' registers once.  -1 leaves it to the target '
            '(`codegen.AUTO_LOOKAHEAD`): two rows on CDNA at 32 lanes in double '
            'precision, where MI300A measured -8 to -11 % at order 8 and no '
            'kernel slower, none elsewhere.')

declare('amd_moved_pins',
        default=False,
        parse=parse_bool,
        doc='With `amd_moved_broadcast`, pin the accumulators of the column '
            'order\'s chain (`hfma`) after every row of products, as the row '
            'order does (`_fused_rows`): a row\'s FMAs ahead of the next row\'s '
            'moves, and each accumulator in a VGPR at its pin.  Moved, SeisSol\'s '
            'order-6 `localFluxAll` in double precision at 16 lanes spilled '
            '812 B on gfx942 and ran four times as long: LLVM had the moves '
            'ahead and the accumulators in AGPRs.')

declare('amd_dpp_waits',
        default='needed',
        env='TF_AMD_DPP_WAITS',
        parse=parse_dpp_waits,
        doc='Where the fused DPP FMAs of CDNA 2 and 3 -- inline assembly, a '
            'block of up to 32 products the compiler\'s hazard recognizer does '
            'not look into (`primitives.amd.dpp_block`) -- wait for the '
            'hazards around them.  `needed`: two wait states in front of '
            'every block, where the compiler may put anything; behind a block '
            'only in a kernel that issues a matrix instruction, the one reader '
            'of a block\'s result hipcc does not wait for by itself; and a run '
            'kept whole where it broadcasts an accumulator it wrote, with a '
            'wait inside only where that write is fewer than two wait states '
            'back.  `always`: two wait states in front of and behind every '
            'block, and a run cut in front of such a broadcast, as before.  '
            'Over SeisSol\'s elastic kernels at orders 4 to 8 and viscoelastic '
            'ones at 4 and 6, the double-precision kernels lose half their '
            'waits (13444 to 6722 on gfx90a, gfx942 and gfx950) and the '
            'single-precision ones, which issue MFMAs, none; registers and '
            'occupancy stay, every result on MI300A is the same bit for bit, '
            'and the double-precision kernels there take 0.8 % less time at 16 '
            'lanes and 0.7 % at 32 (geometric means; the order-8 `volume` '
            '-2.1 and -2.8 %, `localFluxAll` -0.7 and -2.7 %, the viscoelastic '
            'order-6 `volumeExt` -2.8 % at 16; none more than 1.6 % slower, '
            'less than the single-precision kernels, unchanged, move between '
            'two builds of different names).')

declare('amd_dpp_block_operands',
        default=0,
        env='TF_AMD_DPP_BLOCK_OPERANDS',
        parse=parse_int,
        doc='Under `amd_dpp_waits=needed`, let a run of fused DPP products go '
            'on past 32 as one block while it reads and writes no more '
            'distinct operands than this, one wait in front of the run instead '
            'of one every 32 products; 0 cuts every run at 32.  A block holds '
            'all its '
            'operands in registers at once.  At 44, as many as the largest '
            'blocks of 32 in SeisSol\'s kernels take, another 616 of their 16770 '
            'waits go, but the registers move in 22 of the 80 kernels on '
            'gfx942, and the occupancy drops in two (`neighboringFlux` at 16 '
            'lanes, order 4 in double precision 4 to 3 waves, order 8 in '
            'single 2 to 1); on MI300A the double-precision kernels gain 0.4 % '
            'at 16 lanes and 0.2 % at 32 over `needed`, within what the '
            'unchanged kernels move between builds.  Off.')

declare('reread_over_budget',
        default=True,
        parse=parse_bool,
        doc='Where a body does not fit the register file, build it once more '
            'with every operand loaded from memory taken over only within the '
            'operation that loaded it, and keep that build where it needs '
            'fewer registers.  `pir.load_cse` otherwise hands a value read '
            'once to every later read of the same location, which is free '
            'where the registers are there and a spill where they are not: '
            'SeisSol\'s elastic time derivative read its three star matrices '
            'once and held them through every step, 2 KB a lane, and from '
            'order 4 in double precision spilled 1.8 to 2.3 KB on sm_90 '
            'where it had not before the barriers moved behind the passes '
            '(668337a8).  The register file is the target\'s '
            '(`max_reg_per_thread`), against all live values where wave-'
            'uniform ones share it (NVIDIA, Intel), and against the lane-'
            'varying ones and the scalar file apart where they do not '
            '(AMD, `max_scalar_reg_per_wave`).')

declare('stage_group',
        default=8,
        doc='Multiplications a block holds under `stage_members`, and so how '
            'many share one staged copy.  Rounded up to whole waves.')

declare('prepare_operands',
        default=True,
        parse=parse_bool,
        doc='Let the backend re-encode a batch-constant operand for the '
            'instruction that reads it -- store it in fragment order, split it '
            'into parts, or both.\n'
            'What preparing *means* is the primitive\'s business; this only '
            'says whether it may.  On `local_flux` at batch 8192 the fragment '
            'order alone was worth 16%, and 27% with the TF32 split on top.  '
            'Only `Addressing.NONE` operands are eligible at all, and only '
            'where a path asks: over SeisSol\'s 119 kernels it arranges eight '
            'operands on CDNA, for 3.7% more stored, and none at all on '
            'sm_120 or on RDNA, where neither the matrix path nor the SIMT '
            'interleave takes.  So on most targets this costs nothing and '
            'changes nothing.\n'
            'On by default, and what that turns over is worth stating.  '
            'Preparing moves work onto whoever fills the buffer: the kernel '
            'reads the order `layout_offerings()` describes, and a caller '
            'that never collects it writes the tensor\'s own order into a '
            'buffer the kernel reads as another -- the right size, the right '
            'numbers, the wrong places.  yateto keeps that collection out of '
            'its interface version on purpose, so a caller that does not '
            'collect is not detectable before the fact.  What makes the '
            'default honest is that it is detectable *after* one kernel, and '
            '`YatetoFrontend` stops there rather than going on '
            '(`_claim_check`).\n'
            'On the FMA path it stores such an operand in the SIMT interleave '
            '(`Tensor.simt_interleave`): each lane\'s rows side by side in '
            '16-byte groups, read with one aligned vector load.  Offered where '
            'the rows fill the lanes -- `local_flux` at 8 lanes, where it cut '
            'the global loads of the rolled merged kernel and made it 4 % faster.')

declare('pack_operands',
        default='fit',
        env='TF_PACK_OPERANDS',
        doc='Store a batch-constant operand without the bounding-box cells '
            'its own numbers never fill, and hold a read of them off the '
            'lanes concerned (`matrix.packing`).\n'
            'The box is the smallest *rectangle* around the non-zeros, and '
            "for SeisSol's operators that is a third larger than the "
            'non-zeros themselves: 20627 cells hold 11545 numbers.  Three '
            'forms, in increasing order of what addressing them costs: '
            '`rows` drops the rows no column fills at all (13963 cells, one '
            'pair of numbers for the whole operand and no table); `runs` '
            'stores each column between its own first and last non-zero '
            '(12341, a start and an offset per column, still the row plus a '
            'constant); `bits` stores the non-zeros alone (11545, a bitset '
            'per column and a masked population count on every read).  '
            '`auto` takes the smallest form the reads can address, and '
            '`fit` the loosest one that still lets the block stage what it '
            'wanted to -- `rows` first, and a tighter form only where the '
            'images would otherwise not fit shared memory, which is what '
            'the saving is for.\n'
            "`fit` by default, which is the unpadded form everywhere there "
            'is room and a tighter one only where the block would otherwise '
            'not stage what it wanted to.  It moves work onto whoever fills '
            'the buffer -- the host writes the slots the kernel reads, '
            'through the layout offering -- but that is the contract '
            '`prepare_operands` already imposes, so this adds no promise '
            'the caller was not making.\n'
            'Free, measured against a control row that builds the same '
            'kernel twice so the run can say what it is worth: on sm_120 '
            '`rows` came out -0.62 % where the control came out -0.63 %, '
            'and on gfx1150 +0.01 % against +0.09 %.  What does move is '
            'register pressure -- 1154 VGPRs to 1025 over sixteen gfx1150 '
            'kernels -- and the third of the operand storage that is the '
            'point.  `bits` is the one form that costs: +3.7 % against the '
            'control on sm_120, +15 % registers and 88 bytes of spill, '
            'which is why `fit` reaches it last and `auto` is not the '
            'default.\n'
            'Rounding the span to the lane count instead, so the predicate '
            'could be dropped, loses the whole gain and then some: at 32 '
            'lanes it comes out 25 % *larger* than the box, because a span '
            'of 25 rows rounds back up to 32.  So the predicate is the '
            'point and not an accident of the form.\n'
            'The per-column numbers are usually not data.  A packed slot '
            "is the column's start and offset applied to the row, and the "
            'column of an unrolled reduction is known when the read is '
            'built -- so they are immediates, and nothing is fetched.  '
            'Where the column is a loop variable they have to be read from '
            'somewhere, and `pack_tables` is where: a kernel parameter, '
            'which is the one place a uniform index is already a broadcast. '
            'Without it such a reader holds every reader of the operand to '
            'a form that needs no numbers, which in practice is `rows` -- '
            "worth 20 points of SeisSol's order-8 operand storage (-17.1 % "
            'against the box, -37.3 % with the table) and 11 at order 7.  '
            'Below order 7 nothing reaches it: the reductions are short of '
            '`k_unroll_max` and roll only when asked.\n'
            'Offered where `A` is read by coordinate: the generic nest and '
            "AMD's DPP chain.  The matrix paths are out, because they read "
            'fragments of an order they prepared for themselves and a '
            'packed operand has no fragments.\n'
            'A lane that holds several adjacent rows (`lead_vectorize`, '
            "AMD's width-2 DPP) is served too: `rows` and `runs` round "
            'their runs out to whole groups of that width, which over the '
            'SeisSol operators gives up 40.3 % at one row a lane, 38.7 % at '
            'two and 34.1 % at four.  `bits` has no such form -- its slot '
            'is a population count, so two adjacent rows are two adjacent '
            'slots only where both are stored.\n'
            'Against the orders that *grow* the buffer -- the SIMT '
            'interleave and the vendor fragment order, which buy a wide '
            "load with padding (a k-quad order of SeisSol's operators is "
            '33 % larger than the box at 32 lanes) -- a named form here '
            'takes the operand and `auto`/`fit` leave the measured win in '
            'place, packing only what it declines.')

declare('pack_rolled',
        default=False,
        parse=parse_bool,
        doc='Pack an operand (`pack_operands`) that a rolled reduction reads '
            '-- `k_roll`, or one `k_unroll_max` rolls.  Off by default: a read '
            'of a packed operand costs nothing where its column is a constant, '
            'and a trip of its own in a real loop -- the column\'s numbers from '
            'a table (`pack_tables`), the lane range compared, the read held '
            'off the lanes outside it, and the operand kept out of the SIMT '
            'interleave, which reads a lane\'s rows as one vector.  SeisSol '
            'elastic at order 8, single precision, sm_120: dense against '
            'packed, localFluxAll and derivative -12 to -13 %; GH200 at orders '
            '7 and 8 likewise.  What packing saves is storage, which is what to '
            'turn this on for where an operand no longer fits.')

declare('pack_tables',
        default='auto',
        parse=parse_str,
        env='TF_PACK_TABLES',
        doc='Let a packed operand be read at a column the kernel computes, '
            'by putting the per-column numbers in a kernel parameter.\n'
            'A rolled reduction hands its column over as a loop variable, '
            "so a packing's start, offset and length cannot be immediates "
            'in the address.  A uniform packing needs none of them -- its '
            'slot is affine in the column -- and anything else needs them '
            'read at run time or not at all; without this the offer falls '
            'back to the next looser form, which for a rolled reader has '
            'always meant `rows`.\n'
            'Param space, not a constant bank: one copy for the whole '
            'grid, read-only, broadcast to the lanes, and already '
            "allocated.  `__grid_constant__` on CUDA, the kernarg segment "
            'on AMD, where a column every lane agrees on makes it a scalar '
            'load.  Not `__constant__`, which is a second allocation the '
            'launcher would have to fill, and not a function-local array, '
            'whose dynamic index lands in per-thread memory '
            '(`TableForm`).\n'
            'The interface does not learn about it: the launcher declares '
            'the array as a local and passes it, so a caller written '
            'against the kernel before the table existed calls it '
            'unchanged.\n'
            'That space is small where it is smallest -- 4 kB on sm_86 and '
            'gfx942, 2 kB on PVC -- and three tables of a 96-column '
            'operator come to 3.4 kB, so `auto` puts a table that no '
            'longer fits at namespace scope in the generated file instead '
            '(`module`), which costs no argument space, no shared memory '
            'and no interface.  On sm_86 that is exactly the order-8 '
            "derivative's third operator: without the fallback it drops "
            'back to `rows` and the config comes out at -35.0 % against '
            'the box instead of -37.3 %.  `param` and `module` name one '
            'home for every table; `off` goes back to the looser form.\n'
            "SeisSol's order-8 derivative is what reaches this -- five "
            'readers of one operand, of which the later ones roll, so one '
            'rolled reader held seven operands at `rows` for all of them.')

declare('launch_control',
        default=False,
        parse=parse_bool,
        doc='Traverse the batch through Blackwell\'s cluster launch control '
            'instead of a grid-stride loop.\n'
            '`clusterlaunchcontrol.try_cancel` asks the launcher not to launch '
            'a CTA that has not started, and hands the caller its id, so a '
            'resident block drains the grid without an occupancy query and the '
            'tail is the hardware\'s problem.  Needs `sm_100` or above; the '
            'CCCL wrappers gate on `__CUDA_ARCH__ >= 1000` and a lower target '
            'fails at link time with a named symbol.\n'
            'Off by default, and the reason is measured rather than cautious. '
            'On an sm_120 part the queue costs 11-28% against the grid-stride '
            'loop at every batch size from 600 to 262144, and the split says '
            'why.  Timing the same traversal with one block barrier per '
            'element added accounts for 7-15% of that on its own; the queue '
            'adds about two points on top.  The grid-stride loop needs no '
            'block barrier because the rows of a block hold independent '
            'elements, and the hand-off needs one because every thread has to '
            'read the response before the slot is reused.  So the switch '
            'becomes interesting for a configuration that already '
            'synchronizes block-wide, and for work whose cost varies per '
            'element -- not for the row-independent, uniform-cost kernels '
            'this generator emits today.\n'
            'With `enable_wrap_loads` a transfer is moved ahead for the element '
            'the queue hands out next, read off it at the top of the '
            'iteration, which needs `launch_control_depth` 2 or more.  A copy '
            'into shared memory stays where it is: the queried loop carries '
            'no completion across its back edge.')

declare('launch_control_depth',
        default=1,
        parse=parse_int,
        doc='Cancel requests kept in flight per block.\n'
            'One hides the queue\'s own latency behind the body.  Two is the '
            'first depth at which the *next* element\'s index is known at the '
            'top of the iteration, which is what a data prefetch needs -- at '
            'depth one it arrives at the bottom, with no body left to overlap '
            'a transfer with.  Beyond two a block reserves elements it has not '
            'started, which costs at the tail: depth 4 measured worse than '
            'depth 2 in every configuration.')

declare('resident_grid',
        default=0.0,
        parse=parse_float,
        doc='Size the grid-stride traversal\'s grid to this many times what '
            'the device holds at once, so that a work-group loops over its '
            'share of the batch; 0 leaves it to the target '
            '(`Target.bounds_grid`): what the occupancy query answers on CUDA '
            'and HIP, and on SYCL a grid that covers the batch in one round.\n'
            'SYCL has no occupancy query, and what the device holds there is '
            'counted in `max_compute_units` -- one work-group per vector engine, '
            '1024 on a Data Center GPU Max 1550.  A grid that loops is what '
            'lets `enable_wrap_loads` and `prefetch_data` cover the next '
            'element with the current one.  See `batch_parts` for the order '
            'the elements are visited in, which on a two-stack GPU decides '
            'whether looping costs anything at all.')

declare('batch_parts',
        rule=lambda target: 2 if target.sycl else 1,
        parse=parse_int,
        doc='Cut the batch into this many contiguous parts, and give each '
            'part a contiguous share of the grid that strides over it alone; '
            '0 makes every work-group its own part, a contiguous run of '
            'elements.  1, the grid stride over the whole batch; 2 under '
            'SYCL by default, 1 elsewhere.  A grid that covers the batch in '
            'one round makes every row one trip whatever the parts, and is '
            'left as it is.\n'
            'For a GPU of several stacks driven as one device.  Under implicit '
            'scaling the driver hands the first half of the work-groups to one '
            'stack and the second half to the other, and places the first '
            'half of an allocation in the first stack\'s memory: a grid that '
            'covers the batch in one round reads its own stack\'s memory '
            'only.  A grid that strides over the whole batch reads the other '
            'stack\'s every second round, and on a Max 1550 that is 3.4 to '
            '3.9x slower than one round (2.7-3.0 ns against 0.80 per element '
            'of a 2 kB streaming kernel); in two parts it is 3 % slower, and '
            'one part per work-group 8-25 %.  SeisSol\'s elastic kernels at '
            'orders 4 and 6 on a grid of what the device holds '
            '(`resident_grid=1`): 1.69x the one round\'s time in geometric '
            'mean striding over the batch, 1.02x in two parts.  On one stack '
            'two parts cost nothing measurable, one part per work-group up to '
            '25 %.  Only the grid-stride traversal, and not where sections '
            'share it rotated or a loop drives groups of rows.')

declare('merge_variants',
        default='auto',
        parse=parse_bool_or_auto,
        doc='Macro-op merging: state a repeated run of the descriptor list once '
            'and bind its varying operands to a counter.  One switch covers both '
            'the rewrite and the emission, so that a rolled list cannot be '
            'expanded again on the way out.\n'
            '`auto`, the default, merges where the kernel written out takes more '
            'than `merge_icache_fraction` of the instruction cache, as many runs '
            'as it takes to fit (`Generator._auto_merge`); `1` merges every run, '
            '`0` none.  On sm_120 `local_flux` (its four faces merged) computes '
            'the same checksum as the expanded list and runs 5 % faster at 32 '
            'lanes, 12 % at 16 and 24 % at 8, from a quarter to a half of the '
            'code; on GB200 22 to 35 % (package 4).  With `k_roll` as well '
            'it gained less, the merged rolled body taking more registers.')

declare('merge_icache_fraction',
        default=0.25,
        parse=parse_float,
        rule=lambda target: 0.5 if str(target.hw.model) == 'gfx90a' else 0.25,
        doc='How much of the instruction cache a kernel written out may take '
            'before `merge_variants=auto` merges its repeated runs.  Generous on '
            'purpose: merging measured faster wherever it was tried, and the '
            'code size is an estimate that runs up to +89 % over what the '
            'compiler emits at the 90th percentile (`analysis.icache`).  Half '
            'on gfx90a, the 16 kB it merged at while its cache was stated as '
            'twice the 32 KB it is: merged from 8 kB, the elastic '
            '`localFluxAll` at orders 3 and 4, 12 to 15 kB, ran 13 to 18 % '
            'slower on MI210.  Where the target states no instruction cache '
            '(Intel), `auto` merges nothing.')

declare('merge_min_count',
        default=3,
        doc='Shortest run worth merging.  Three rather than two because two '
            'contributions are cheaper written out than a counter and a select '
            'per operand, and because a pair of same-shaped operations is the '
            'commonest accidental run.')

declare('merge_max_arity',
        default=None,
        parse=parse_optional_int,
        doc='Largest operand count a merged run may carry, or None for no cap.')

declare('k_roll',
        default=0,
        env='TF_K_ROLL',
        rule=lambda target: 1 if target.hw.vendor == 'generic' else 0,
        doc='Roll the reduction of a multilinear product into a real loop over '
            'groups of `k_width` steps, with `#pragma unroll <k_roll>` on it.  '
            '0 unrolls it completely in the generator.  Only where '
            'every operand the reduction indexes lives in memory (global, '
            'batch or shared) -- a register image indexed at runtime would go '
            'to local memory -- the reduction is dense, and its extent divides '
            'into whole groups.  For kernels whose fully unrolled body no '
            'longer fits the instruction cache.  1 on the target `generic`, '
            'whose one lane per multiplication would otherwise unroll every '
            'product of a kernel whole (a hundred times the code of a GPU '
            'target for SeisSol).')

declare('pin_shared',
        default=True,
        parse=parse_bool,
        env='TF_PIN_SHARED',
        doc='Whether loads from shared memory count toward the fences of '
            '`pin_loads`, or only loads from global memory do.  A fence is '
            'there to keep the compiler from issuing a global load so far '
            'ahead that its register is held for long; fenced, a shared load '
            'can no longer be reused or moved across the fence either.  '
            'SeisSol\'s order-8 elastic derivative in double precision on '
            'sm_90: fenced on all loads 239 fences, 42408 SASS instructions, '
            '12070 shared loads, 680 B spilled; on global loads only 52, '
            '29648, 8029 and 308 B; unpinned 27984 instructions and 1020 B.')

declare('k_unroll_max',
        default=64,
        env='TF_K_UNROLL_MAX',
        doc='Most reduction steps the generator unrolls whole when `k_roll` '
            'is not set.  A longer reduction is rolled as `k_roll` would roll '
            'it, by the largest divisor of its step count up to this one, '
            'where it can be (see `k_roll`); 0 unrolls every reduction whole.  '
            'Above the 56 of `local_flux`: at 120, eight lanes of it were '
            '121k lines of CUDA that cicc had not finished after twelve '
            'minutes, and rolled at 56 the same kernel was the fastest FFMA '
            'variant on GB200.')

declare('stage_operands',
        default='off',
        env='TF_STAGE_OPERANDS',
        parse=parse_str,
        doc='On NVIDIA, stage a multiplication\'s sources in shared memory '
            'instead of loading them into registers: `all` for every '
            'multiplication, `narrow` for one narrower than the warp, `stream` '
            'for an operation that sums over no axis, `off`.  '
            'The copy moves sixteen bytes a lane, asynchronously, where a '
            'register load moves one number a lane; the destination keeps its '
            'registers.  Staged is only a source no operation of the section '
            'stores and no other reads -- the copy would read what other lanes '
            'stored, or be what the next reader finds for the tensor.  An '
            'experiment.')

declare('k_unroll_max_tail',
        default=0,
        env='TF_K_UNROLL_MAX_TAIL',
        parse=parse_int,
        doc='Steps a reduction that `k_unroll_max` rolls may leave over.  The '
            'count is then the largest up to `k_unroll_max` whose trips leave '
            'at most this many steps: the loop runs whole trips, and the steps '
            'left over are written out behind it.  0 takes a count that '
            'divides the steps, or, where none does, leaves the rest to the '
            'compiler\'s remainder loop.  SeisSol\'s order-5 reduction of 35 '
            'steps is rolled by 7 at a cap of 16; at a tail of 2 by 11, with '
            'its last two steps written out.')

declare('fission',
        rule=lambda target: ('auto' if (target.hw.vendor == 'intel'
                                        and target.backend == 'oneapi'
                                        and not target.explicit_simd)
                             else 0),
        env='TF_FISSION',
        parse=lambda text: 'auto' if text.strip() == 'auto' else int(text),
        doc='Run the batch this many times in one kernel, a piece of the '
            'operation list each time, cut where nothing but the kernel\'s own '
            'operands is live (`generator.fission_cuts`) -- SeisSol\'s '
            'derivative between two of its derivatives, say -- so that each '
            'piece\'s code is what the instruction cache has to hold.  Every '
            'block takes the same elements in each piece, so a block barrier '
            'orders them (`BlockBarrierDescr`), and the pieces share one '
            'element count.  What crosses a cut goes through memory: the '
            'piece after reads what the one before stored.  0 or 1 cuts '
            'nothing; a list with too few such points stays whole.  `auto` '
            'cuts where the whole kernel\'s code is past one and a half '
            'times the instruction cache, into as few pieces as keep the '
            'largest under the share of the cache the part was measured at '
            '(`generator.FISSION_PIECE`); on a vendor whose parts state no '
            'cache, in two past the code measured there '
            '(`generator.FISSION_CODE`), and nowhere else.  `auto` by default '
            'on Intel under DPC++\'s SPMD lowering: SeisSol\'s order-7 and '
            'order-8 kernels it cuts 14 % faster (Max 1550), 0 everywhere '
            'else.')

declare('min_blocks_per_sm',
        default=1,
        env='TF_MIN_BLOCKS_PER_SM',
        doc='NVIDIA only: the second argument of `__launch_bounds__`, the '
            'blocks an SM must be able to hold at once; 0 leaves it out.  One '
            'rather than none: with only the thread count, ptxas -O3 gave '
            'SeisSol\'s viscoelastic time derivative (order 6, single '
            'precision) 40 registers and 58 KB of spill stores, and it ran at '
            '7.3 us an element on sm_120; told one block, it took 255 and ran '
            'at 1.3 us.  HIP reads the second argument as warps per execution '
            'unit, so it is not passed there.')

declare('staged_block_threads',
        default=256,
        env='TF_STAGED_BLOCK_THREADS',
        parse=parse_int,
        doc='Threads in a block that holds a copy of operator data in shared '
            'memory -- preloaded operators or shards (`RegmaxBlockPolicy`); '
            'its multiplications share that one copy, and the shards may take '
            'what the blocks a compute unit keeps leave each '
            '(`Generator._shard_budget`).  On AMD the count also bounds the '
            'registers hipcc gives a lane: 512 threads are two waves a SIMD '
            'at the least, so 256 VGPRs and AGPRs on CDNA2 and 3, and 1024 are '
            'four, so 128.')

declare('amd_waves_per_eu',
        default=0,
        env='TF_AMD_WAVES_PER_EU',
        parse=parse_int,
        doc='AMD only: the second argument of HIP\'s `__launch_bounds__`, the '
            'waves a SIMD must be able to hold at once (`amdgpu_waves_per_eu`), '
            'which hipcc sizes the registers for -- on CDNA2 and 3 the file of '
            '512 VGPRs and AGPRs a lane, split in two at two waves and in four '
            'at four.  0 leaves it out: with the 256-thread blocks this '
            'generator launches, hipcc may then take the whole file at one '
            'wave a SIMD, where a dependent FMA issues at 64 to 73 % of its '
            'rate on MI300A.')

declare('identity_reduction',
        default=True,
        parse=parse_bool,
        env='TF_IDENTITY_REDUCTION',
        doc='Collapse a product whose batch-constant first operand is the '
            'identity pattern -- the description gives its numbers, and every '
            'row the lanes run along has its nonzeros at the reduction step '
            'equal to the row -- to the one product a lane needs, `A[i,i] '
            'B[i,j]`, its weight a literal the lane selects.  Computed the '
            'usual way it is the dense product, since a step is issued by '
            'every lane and only one holds a nonzero there.  SeisSol\'s face '
            'orientation `fP(1)` is this at every order: `neighboringFlux[16]` '
            'on sm_120 loses 19 % (o4), 15.5 % (o6) and 8 % (o8) of its executed '
            'flops, and nothing crosses a lane on AMD where the broadcast path '
            'issued one step per row.  Ones are the second operand as it is, '
            'ones and zeros a select of it.  Needs `skip_known_zeros`.')

declare('ell_gather',
        rule=lambda target: target.prefs.ell_gather,
        parse=parse_int,
        env='TF_ELL_GATHER',
        doc='Most nonzeros a row of a batch-constant first operand may have '
            'for its product to be gathered rather than broadcast; 0 turns '
            'it off.  The lanes run along the rows of `A` in `C = A B`, so '
            'the usual nest issues every step of the reduction on every lane '
            'and only the lanes with a nonzero there need it.  Gathered, a '
            'lane takes its own nonzeros, in sliced ELL: each slot of lanes '
            'as many steps as its fullest row, and a step reads `B` at a '
            'row of its own in each lane -- from memory at a per-lane '
            'address, or from a register image across the lanes '
            '(`__shfl_sync`, `ds_bpermute`).  Columns and weights the lanes '
            'do not share come from a table in the generated file.  Taken '
            'only where it halves the steps the reduction takes with its '
            'known zeros left out -- all of them where it rolls.\n'
            '64 on NVIDIA, as many as `k_unroll_max` unrolls, where a gather '
            'is the instruction a broadcast is (a shared-memory read, '
            '`SHFL.IDX`): at order 8 on sm_120 `neighboringFlux` and '
            '`localFlux` ran in less than half the time once `rT` and `fMrT`, '
            '20 and 36 nonzeros a row against a rolled reduction of 120, were '
            'gathered.  3 elsewhere: on AMD a gather from a register image is a '
            '`ds_bpermute` against a broadcast fused into the FMA, and gathered '
            'at 8 or 32 the same kernels were 3 to 5 % faster at order 4 and 6 '
            'to 10 % slower at order 8 on gfx1150; Intel is unmeasured.  '
            'Needs `skip_known_zeros`.')

declare('k_segments',
        default=4,
        parse=parse_int,
        doc='At most how many loops a rolled reduction is cut into where the '
            'lane slots of its batch-constant operand start and stop having '
            'numbers.  Unrolled, a reduction leaves out every slot and step the '
            'operand is zero at (`skip_known_zeros`); rolled, every slot runs '
            'every step -- in the rolled products of SeisSol\'s order-8 '
            'derivative a third of the slot-steps at 32 lanes and over two '
            'fifths at 8 lie outside the steps where their slot has a number.  '
            'Cut, each segment leaves out the slots that are zero on all of '
            'it.  1 keeps one loop.')

declare('skip_known_zeros',
        default=True,
        parse=parse_bool,
        env='TF_SKIP_KNOWN_ZEROS',
        doc='Leave out a product whose batch-constant factor the description '
            'gives numbers for and which is zero at every cell the step reads '
            '-- in the per-lane loop, every row of the lane slot at that '
            'reduction index.  No load and no multiply-add.  SeisSol\'s '
            'volume kernel (order 6) reads kDivM dense, 16% of its cells '
            'nonzero, and its first 32-row slot needs 20 of 35 columns.  '
            'Exact for finite operands; a product of zero with an infinity '
            'or a NaN in the other factor is not formed.  It takes the '
            'description at its word: the buffer the caller passes has to hold '
            'the numbers the description gives -- as SeisSol\'s global '
            'matrices do, and as yateto\'s own sparse kernels assume -- and a '
            'caller that fills it with anything else gets the products of '
            'those numbers, not of its own.  Under the multi-element layout '
            '(`multi_element`), an operator step on the FMA units walks its '
            'operators\' nonzeros alone (`element_group.Terms`).')

declare('autotune',
        default='off',
        env='TF_AUTOTUNE',
        codegen=False,
        doc='Choose the lane geometry and the safe options per kernel by '
            'building candidates and ranking them (`generators.tuning`): '
            '`off`; `prefer`, measured preferences only and the default where '
            'none matches (`generators.preferences`); `static` (the build\'s '
            'own figures) or `compiled` (the target compiler\'s registers and '
            'spills; falls back to `static` where none is found), both of which '
            'take a matching preference first; or `lanes`, the lane count from '
            'eight up to the extent\'s power of two and merging, ranked on '
            'NVIDIA by the instructions an element executes with the '
            'compiler\'s spilling and its code past the instruction cache '
            'priced in (NVIDIA, and Intel under DPC++), and on AMD by the pipe '
            'bound with the spilling in it, stretched by the waves the part '
            'needs (`tuning.lanes_rule`; measured on MI300A and MI210, the '
            'default on other parts and where no compiler is found), which '
            'needs no GPU and asks the compiler about three to eight builds.  '
            'Only what is safe to ship unasked is '
            'turned: the lane count, a lead width of two where the target has '
            'a packed FP32 FMA, merging, rolling, and under `compiled` on AMD a '
            'stack of two elements and the width of the matrix instruction, '
            'whose register cost the static model does not see -- not preparing '
            'operands, which the host has to pack for, and not the matrix path.  '
            'A stack or a width the caller asks for stays as asked.  Not part '
            'of the kernel\'s identity: what it picks is, through the options '
            'and the geometry it builds with.')

declare('device',
        default='',
        env='TF_DEVICE',
        codegen=False,
        doc='Which device of the target architecture, where the architecture '
            'does not say: `mi300a` or `mi300x` for gfx942, say.  Read only by '
            'the measured preferences (`generators.preferences`), which name a '
            'device as `arch:variant` before `arch`.  What a preference then '
            'picks is part of the kernel; this name is not.')

declare('preferences',
        default='',
        env='TF_PREFERENCES_FILE',
        codegen=False,
        doc='Files of measured preferences to consult before the shipped one, '
            'separated by the path separator (`generators.preferences`).  '
            '`TF_PREFERENCES` names more, after these.')

declare('autotune_budget',
        default=24,
        env='TF_AUTOTUNE_BUDGET',
        codegen=False,
        doc='Most builds `autotune` spends on one kernel, the default one '
            'included.  A coordinate walk over the simple space of '
            '`local_flux` takes 10 to 20.')

declare('lanes_band',
        default=False,
        env='TF_LANES_BAND',
        parse=parse_bool,
        codegen=False,
        doc='Under `autotune=lanes` on NVIDIA, ask the compiler only about the '
            'candidates whose spilling the estimate cannot settle '
            '(`tuning.ASK_BAND`) and take the estimate\'s word for the rest.  '
            'Off: on GH200 the estimate dropped the fastest candidate of '
            'viscoelastic order-6 `volumeExt` (16 lanes, +33 % with the band\'s '
            '32), and asked in threads (`autotune_jobs`) the compiler costs '
            'little time.')

declare('lanes_occupancy_keep',
        default=0.0,
        env='TF_LANES_OCCUPANCY_KEEP',
        parse=parse_float,
        codegen=False,
        doc='Under `autotune=lanes` on NVIDIA, leave out a candidate whose '
            'resident warps a multiprocessor -- the blocks the compiler\'s '
            'registers admit, times the warps a block holds -- fall below this '
            'share of the default\'s; 0 leaves every one in.  Fewer '
            'instructions bought with half the warps lose where the kernel '
            'waits on memory: single-precision viscoelastic `neighbor` at eight '
            'lanes holds 224 and 255 registers (orders 5 and 6) against 117 at '
            '32, and ran 115 and 24 % slower on GH200.  '
            'Over the GH200 viscoelastic grid (17 kernels, 8 to 64 lanes) 0.75 '
            'took the rule\'s mean loss against the fastest candidate from 23.3 '
            'to 14.5 %, with `lanes_spill_floor` 512 to 8.9 % (worst 115 to '
            '44 %); over the elastic grid of four GPUs it moves nothing.')

declare('lanes_spill_floor',
        default=0,
        env='TF_LANES_SPILL_FLOOR',
        parse=parse_int,
        codegen=False,
        doc='Bytes a thread may spill that the lanes rule on NVIDIA still counts '
            'as none: the guard admits builds spilling up to this much, and the '
            'wider lane counts stay out unless the default spills more '
            '(`tuning.WIDE_SPILL`); 0 counts every byte.  A few hundred bytes '
            'stay in the L1: viscoelastic `volumeExt` in double precision spills '
            '260 and 488 bytes at 32 lanes (orders 5 and 6), and the 64 lanes '
            'the rule took to avoid that ran 42 and 36 % slower than 32 on '
            'GH200.')

declare('lanes_split_floor',
        default=True,
        env='TF_LANES_SPLIT_FLOOR',
        parse=parse_bool,
        codegen=False,
        doc='Let the lanes rule weigh lane counts below the vector unit where a '
            'sub-group can hold several multiplications with their broadcasts '
            'split (`Options.sycl_split_broadcast`; PVC under SPMD, where the '
            'floor is otherwise the 16-wide vector unit, '
            '`Preferences.lanes_down_to_wave`): eight lanes at the default '
            'split of two.  `tuning.lane_floor`.  On by default since the '
            'guard on spill references (`tuning.SPLIT_REF_GROWTH`): over 60 '
            'elastic kernels on the Max 1550 (orders 3-6, s1 and s8, IGC '
            '2.41.5, a batch of 262144) the rule with it runs 1.008 times the '
            'best of the default, the rule without it and the rule with it, '
            'in geometric mean, against 1.050 without it and 1.065 for the '
            'default; worst 1.32, where the rule without it is the same.')

declare('batch_dispatch',
        default=0.0,
        env='TF_BATCH_DISPATCH',
        parse=parse_float,
        doc='Under `autotune=lanes`, a second kernel for small batches, and a '
            'launcher that runs it below this many times the elements it holds '
            'at the waves a SIMD overlaps (`tuning.dispatch_overlap`, the '
            'multiprocessors asked of the runtime at the first launch); 0 '
            'builds one kernel.  At a batch the device '
            'holds at once a launch lasts as long as one wave\'s chain, and the '
            'candidate with the shortest one -- the pipe bound of a wave, '
            '`tuning.chain_cycles` -- is a different one from the throughput '
            'pick in most of SeisSol\'s kernels: over the paper-1106 sweeps (16 '
            'to 262144 elements, the default and the lanes pick of 88 kernels) '
            'the shorter chain was the faster launch at up to 64 elements in '
            '72 to 84 of 88 kernels on NVIDIA, 77 on MI300A and 70 on MI210; '
            'switching between the two left 1.2 to 2.4 % against the faster '
            'one over all batches (5.0 on MI210), where the lanes pick alone '
            'left 6.7 to 8.5 %.  On MI300A, with the small kernels the walk '
            'takes (16 to 32 and 64 lanes, elastic orders 4 and 6, '
            'viscoelastic order 4), 1.1 % against 9.7 %.  Built only where '
            'both kernels store the operands alike and the second asks for '
            'no argument the first is not given, on CUDA and HIP, and where '
            'the short chain is `tuning.DISPATCH_MARGIN` shorter.')

declare('autotune_jobs',
        default=1,
        env='TF_AUTOTUNE_JOBS',
        codegen=False,
        doc='Processes `autotune=lanes` generates its candidates in, and threads '
            'it asks the compiler in, at once.  The candidates are independent '
            'builds of one kernel, and generating them is most of what the walk '
            'costs (83 % on sm_90 over SeisSol\'s order-6 kernels, nvcc 12 %); 1 '
            'builds them one after the other, as before.  Forked, so Linux and '
            'macOS only; each process holds a build of its own.')

declare('autotune_cache',
        default='',
        env='TF_AUTOTUNE_CACHE',
        codegen=False,
        doc='A JSON file `autotune` keeps its picks in, keyed by the default '
            'build\'s source and the target, so that a kernel generated again '
            'costs one build.  Empty keeps them for the process only.')

declare('tensor_cores',
        default=None,
        env='TF_TENSOR_CORES',
        parse=parse_optional_bool,
        doc='Use the matrix instructions where the shape allows: NVIDIA\'s '
            '`mma.sync` (`primitives.nvidia`) and, under the explicit vector, '
            'Intel\'s DPAS (`primitives.intel`).  The SPMD lowering\'s DPAS '
            'has an option of its own (`sycl_dpas`), as AMD\'s stacked scheme '
            'has (`amd_matrix_k4`), and AMD\'s lane-batched MFMA none: it is '
            'taken wherever a tile fits.  None takes each path\'s '
            '`ENABLED`, the deployment switch, off for both; an option so '
            'that one build can ask for the path and the next one not, as a '
            'search over configurations has to -- flipping the module '
            'constant changes it for every build in the process.')

declare('multi_element',
        default='off',
        env='TF_MULTI_ELEMENT',
        parse=parse_multi_element,
        doc='Lay a kernel out over groups of elements rather than one element '
            'a multiplication where its operations allow it (the multi-element '
            'layout, `generators.element_group`): a block of E warps, one a '
            'multiplication, takes E consecutive elements together, stages '
            'their operands side by side in shared memory and the '
            'batch-constant operator once for the block.  `off` never; '
            '`force` wherever the fill rule finds a plan; `auto` where it '
            'finds one measured to pay -- none yet, so `auto` is `off` for '
            'now.')

declare('multi_element_fma',
        default='auto',
        env='TF_MULTI_ELEMENT_FMA',
        parse=parse_multi_element_fma,
        doc='The FMA form the multi-element layout (`multi_element`) runs a '
            'section in.  `warp`: a warp an element and a lane an output, the '
            'element\'s operands staged in shared memory (`fma`, `rows`), '
            'where the matrix tiling does not take a step.  `lanes`: a lane '
            'an element, every lane reading its own element through its own '
            'pointers and the operators\' nonzeros as literals, the outputs '
            'split over the warps the group\'s elements share (`lane`) -- '
            'ahead of the matrix tiling, wherever the section fits it, and '
            '`warp` where it does not.  `auto`: the target\'s rule, which is '
            '`warp` everywhere until the measurements say otherwise.')

declare('mma_prefetch',
        default=None,
        env='TF_MMA_PREFETCH',
        parse=parse_optional_int,
        doc='Steps ahead the matrix path loads a pre-ordered operand\'s '
            'fragments (0: at the step that uses them).  None takes '
            '`primitives.nvidia.PREFETCH`.  A search parameter: the second step '
            'hides more of the load latency and costs about forty registers '
            '(`local_flux` b56: 149 against 190 on sm_100a), which is a block '
            'per SM -- only a measurement says which of the two wins.')

declare('mma_prefetch_across',
        default=True,
        env='TF_MMA_PREFETCH_ACROSS',
        doc='Whether the matrix path loads the next block of rows\' first '
            'fragments during the last steps of this one (`mma_prefetch` '
            'steps ahead over the whole contraction), or starts every block '
            'cold.  Across made `mmaos` 10 % faster on GB200 and the merged '
            '`mmaosm` 7 % slower -- same instructions, L1 hits 95.2 % against '
            '93.1 % -- so it is a question per kernel, not a constant.')

declare('mma_d_layout',
        default=False,
        env='TF_MMA_D_LAYOUT',
        parse=parse_bool,
        doc='Keep the matrix path\'s accumulator in the instruction\'s D '
            'layout across the products that add into one output in global '
            'memory: the output is read into that layout once, before the '
            'first of them, and written once, after the last, each lane at '
            'the rows and columns its fragment holds -- with no epilogue tile '
            '(`ctile`) and no rows in between (`compute.dchain`).  NVIDIA, '
            'one multiplication per warp; on AMD the stacked tiles of '
            '`amd_matrix_k4`, whose D layout is the register image, every '
            'column in them (`amd_matrix_k4_pad` is implied); on Intel under '
            '`sycl_dpas` the DPAS accumulator, the register image too '
            '(`intel_spmd`), but not where the kernel reads the output '
            'again.  A product it does not reach keeps its epilogue, and on '
            'AMD and Intel so does a product on its own '
            '(`MatrixPaths.D_CHAIN_SINGLES`).  On MI300A at sixteen lanes, '
            'against the stacked tiles padded and unchained: order 6 '
            '`volume` -10 % in double precision and in single, '
            '`localFluxAll` -18 % in single; `neighboringFlux`, one product '
            'into `Q`, was 9 to 30 % slower as a chain of one.  On PVC the '
            'chains were slower but in the order-6 volume (Max 1550, '
            'single precision, batch 262144, ns an element without and with '
            'the chains): order 6 volume 12.8 / 12.4, `localFluxAll` 17.6 / '
            '23.8; order 8 volume 32.9 / 68.2, `localFluxAll` 72.0 / 106.1.  '
            'IGC spills the chained builds more (order-8 `localFluxAll` '
            '31 424 bytes against 8 576), and their products add into one '
            'accumulator one after another where on their own they do not '
            'wait for each other.')

declare('mma_d_layout_loops',
        default=False,
        env='TF_MMA_D_LAYOUT_LOOPS',
        parse=parse_bool,
        doc='Under `mma_d_layout`, let a chain span the trips of a merged run: '
            'the output read ahead of the loop, every trip issuing into the '
            'accumulators the loop carries, the output written behind it '
            '(`dchain.DChain.loop`; AMD\'s stacked tiles).  SeisSol\'s '
            '`localFluxAll`, its four faces one loop from order 6 on, on MI300A '
            'against the same builds with trips that add through their '
            'epilogues: double precision +6 to +11 % at order 6 and +12 to '
            '+14 % at order 8 (sixteen lanes, +57 % on the shard route), '
            'single precision -9 to +3 % and -3 to -6 % -- and nowhere better '
            'than the best build without it.  The chain reads `Q` ahead of '
            'the loop and waits for it, where the epilogues\' atomic adds '
            'wait for nothing.  Off.')

declare('mma_stage_b_once',
        default=False,
        env='TF_MMA_STAGE_B_ONCE',
        parse=parse_bool,
        doc='On NVIDIA\'s matrix path, stage every contraction step\'s B tile '
            'of a product at once, before its blocks of rows, instead of one '
            'step at a time for every block of 32 rows: two warp barriers a '
            'product instead of two a step and block, and B written once where '
            'the rows span several blocks.  The tile grows by the number of '
            'steps.  An experiment; off.')

declare('mma_stack_mults',
        default=False,
        env='TF_MMA_STACK_MULTS',
        parse=parse_bool,
        doc='Where several multiplications share a warp on NVIDIA\'s matrix '
            'path (fewer than 32 lanes each) and read one batch-constant A in '
            'a prepared order, stack them along N: column c of multiplication '
            'p is column p * width + c of one wide B, so an instruction '
            'serves the columns of several and no column is carried on the '
            'fragments.  An experiment; off.')

declare('mma_chunked_order',
        default=False,
        env='TF_MMA_CHUNKED_ORDER',
        parse=parse_bool,
        doc='Lay a prepared A operand out lane by lane per sixteen bytes '
            'rather than per tile: a lane\'s fragments of a tile in 16-byte '
            'chunks, the 32 lanes adjacent within each chunk.  A lane then '
            'reads a tile as several 16-byte accesses instead of one wide one, '
            'and every such access of a warp reads 512 adjacent bytes -- from '
            'shared memory without a bank conflict, from global memory in '
            'whole sectors.  Changes only the double-precision orders, whose '
            'eight fragments are 64 bytes a lane; a single-precision lane\'s '
            'fragments are one 16-byte chunk already.')

declare('mma_fragment_ld',
        default=False,
        env='TF_MMA_FRAGMENT_LD',
        parse=parse_bool,
        doc='With `tensor_cores`, give every temporary stored to shared memory '
            'the smallest leading dimension at or above its rows under which a '
            'B fragment of the matrix path is read without a bank conflict '
            '(`store.fragment_ld`): 60 for 56 rows, in either precision, '
            'where 56 is a two-way conflict on every read.  Not with '
            '`stage_row_bytes`, which pads the same dimension for wide '
            'accesses.')

declare('mma_operator_copies',
        default=None,
        env='TF_MMA_OPERATOR_COPIES',
        parse=parse_optional_bool,
        doc='Give each multiplication the matrix path may take that reads a '
            'batch-constant operator other operations read too a copy of it, '
            'laid out in its own fragment order and defined in the generated '
            'file (`Residence.MODULE`), where the description carries the '
            'numbers.  An order is the reader\'s window and fragment layout, '
            'so an operator with several readers is offered none and the '
            'matrix path stages it element by element (`_offer_order`): '
            'SeisSol\'s derivative reads each `kDivMT` five times.  The other '
            'readers keep the operator as it is passed.  On Intel the same for '
            'the SPMD lowering\'s DPAS (`sycl_dpas`), whose reader of an '
            'unordered operator splits every tile it reads and holds what it '
            'read for the next reader: None, the default, makes the copies '
            'there and nowhere else -- SeisSol\'s order-6 derivative in single '
            'precision ran 49.2 ns an element without them on the Max 1550 '
            'and 34.7 with them (IGC 2.41.5).')

declare('mma_presplit',
        default=False,
        env='TF_MMA_PRESPLIT',
        parse=parse_bool,
        doc='Store a batch-constant operand NVIDIA\'s TF32 path reads in '
            'fragment order as its two TF32 halves, so that no fragment is '
            'split after it is loaded.  Kept only where every reader of the '
            'operand is a matrix product.')

declare('mma_min_depth',
        default=None,
        env='TF_MMA_MIN_DEPTH',
        parse=parse_optional_int,
        doc='Contraction steps below which a product stays off NVIDIA\'s matrix '
            'path.  None takes `primitives.nvidia.MIN_DEPTH`, 16.  A search '
            'parameter and not a constant: the depth-9 star products of '
            'SeisSol\'s derivative are 44 % padding at k = 16, and yet o6 in '
            'double took 79.3 ns per element with them on the nest and 40.9 '
            'with them on DMMA (GH200) -- the nest spilled 2.8 kB, the matrix '
            'path 0.4 kB -- while the volume in either precision was 15 to 17 % '
            'slower with them on the matrix path.')

declare('matrix_columns',
        default=0,
        parse=parse_int,
        doc='Output columns one issue of the matrix instruction covers, where '
            'the target has entries of several widths for a product: CDNA\'s '
            'lane-batched F32 tiles, 4 and 16.  0 leaves it to the count of '
            'passes (`compute.ranking`), which takes the narrowest unless the '
            'columns fill the wider one as well; a width takes the entry of '
            'that width for every product it serves, and the count for the '
            'rest.  The count does not see what a narrow tile reads again: '
            'the lead operand once per block of columns, four times as often '
            'at 4 as at 16.  The static model on gfx942 puts the wide tile '
            '49 to 63 % ahead at order 8 of SeisSol\'s elastic kernels, where '
            'loads bind, and 55 to 98 % behind at order 4, where the '
            'arithmetic does -- a question per kernel (`generators.tuning`).')

declare('full_lane_tails',
        default=True,
        env='TF_FULL_LANE_TAILS',
        parse=parse_bool,
        doc='Whether the ragged end of a lead dimension computes on every '
            'lane, with only its memory accesses kept to the lanes that hold '
            'data.  Under ESIMD the guard `lead < 24` otherwise becomes a '
            '24-wide vector, and a 24-wide operation is issued as 16 + 8: '
            'local_flux on pvc, 11552 instructions against 6517 with the tail '
            'at 32, the ESIMD corpus 60568 against 50013.  Under SPMD it takes '
            'the branch around the tail block away: local_flux on sm_100 149 '
            'registers against 116, gfx942 and gfx1250 unchanged.  Only where '
            'the extra lanes are padding -- no lead origin shift, and the '
            'accumulator\'s register image exactly the loop\'s window, so that '
            'a slice of a larger image (theta) never has its neighboring rows '
            'overwritten.')

declare('lead_straddle',
        default=False,
        env='TF_LEAD_STRADDLE',
        parse=parse_bool,
        doc='With a lead dimension blocked wider than one element (a lead '
            'width of 2 or 4), the lane whose vector reaches past the end of a '
            'multiplication\'s rows computes the whole vector, as a full-lane '
            'tail does, instead of having its valid components peeled into '
            'scalar operations -- each a broadcast from the owning lane and a '
            'store guarded to it, at every step of every column.  Only where the '
            'accumulator is a register image of exactly the window, and every '
            'operand read along that dimension has storage for the rounded-up '
            'end, so that the extra component reads inside its own column and '
            'is never stored.  An experiment.')

declare('prefetch_data',
        default=False,
        env='TF_PREFETCH_DATA',
        parse=parse_bool,
        doc='Hint the next element\'s data where `enable_wrap_loads` would issue its '
            'transfer -- the tail of the loop body -- and leave the transfer '
            'where it is.\n'
            'The transfer itself moved costs a register image or a buffer '
            'that survives the back edge; a hint costs a pointer and a few '
            'messages, and changes no result.  One hint per cache span of '
            'each per-element source (`Target.prefetch_line_bytes`), for '
            '`PTR_BASED` and `STRIDED` operands; a batch-invariant one is '
            'already cached.  At `prefetch_level`.\n'
            'Off everywhere.  Under ESIMD, where one work-item per element '
            'leaves nothing else to cover the latency of the next one, it '
            'costs 6.6 % more instructions over the corpus, and measured on '
            'pvc it does not pay for them: the SeisSol elastic kernels (F32, '
            'batch 65536, the grid covering the batch in one round) ran 1.12x '
            '(order 4) and 1.03x (order 6) faster in geometric mean without '
            'the hints, single kernels up to 1.35x; nearly every fastest '
            'ESIMD configuration had it off.  The pointer hint at L1 '
            '(`enable_prefetch`, `prefetch_level=l1`) in its place was '
            'neutral, 0.99-1.03x.  Elsewhere never measured faster.')

declare('preload_register_share',
        default=0.0,
        env='TF_PRELOAD_REGISTER_SHARE',
        parse=float,
        doc='Under SPMD, the share of a lane\'s register file '
            '(`max_reg_per_thread`) that operand images staged in registers '
            'may take together; an operand that would push the images already '
            'resident past it is read in place instead.  0: no limit.  Under '
            'the explicit-SIMD lowering one work-item holds each image whole '
            'and the limit is the whole file per image, whatever this says.  '
            'chain_five_multiplies stages two 56 x 56 operators at 448 B a '
            'lane each: 256 VGPR and 428 B of scratch on gfx1150, 256 VGPR '
            'and 111 AGPR on gfx942.')

declare('lanes_per_mult',
        rule=lambda target: target.prefs.lanes_per_mult,
        env='TF_LANES',
        parse=parse_int,
        doc='Lanes one multiplication is spread over, instead of the deduced '
            'count.  0 keeps the deduction.  Only narrower counts are taken, '
            'and only powers of two, so that a warp still holds whole '
            'multiplications; the rows a lane covers grow to match, with the '
            'last ones padded.  A search parameter and not a default: on sm_120 '
            'eight lanes made `local_flux` 10 % faster and `chain_three` four '
            'times slower, because it spilled -- see `lanes.narrower`.  '
            'Except under Intel\'s SPMD lowering, where 16 is the default: a '
            'sub-group of 16 gives a lane twice the registers one of 32 does, '
            'and the SeisSol kernels spilled half as much and ran 0.76x '
            '(`Preferences.lanes_per_mult`).')

declare('wave_uniform_rows',
        default=True,
        parse=parse_bool,
        doc='Read the row of the block once, as the wave\'s, where every '
            'multiplication is a whole number of waves, and the element '
            'pointers addressed from it as well (AMD; `Lexic.first_lane`).  '
            'The compiler cannot tell that `threadIdx.y` is the same on every '
            'lane of a wave, since it does not know the block; told, it keeps '
            'the element index, the flags read and the base of every load of '
            'an element on the scalar unit.  Static ISA over the 117 kernels of '
            'SeisSol elastic o4/o6/o8, viscoelastic o6 and poroelastic o4 in '
            'single precision: 11 % fewer instructions in the median on '
            'gfx1150, where 32 lanes are a wave, and 5 % with '
            '`lanes_per_mult=64` on gfx942.  Where several multiplications '
            'share a wave nothing changes.  A switch to compare against, '
            'pending hardware numbers.')

declare('restrict_batch_arrays',
        default=True,
        parse=parse_bool,
        doc='Promise in the kernel\'s prototype that nothing writes the arrays '
            'of pointers and the flags while the kernel runs, and that nothing '
            'else reaches them: `const` all the way down and `__restrict__` '
            '(CUDA and HIP; `KernelParam.promise`).  The interface means that '
            'already, and the launcher keeps its types.  On AMD a pointer or a '
            'flag word the whole wave reads is then a scalar load through the '
            'scalar cache, and a wait for it no longer queues behind the '
            'vector loads in flight.  Static ISA over the SeisSol kernels of '
            '`wave_uniform_rows`: 3 to 4 % fewer vector loads on gfx1150 and '
            'with `lanes_per_mult=64` on gfx942, and the instruction count '
            'within half a percent.  Off is the prototype the interface '
            'spells.')

declare('wave_group_traversal',
        default=False,
        parse=parse_bool,
        doc='Drive the multiplications that share a wave through the batch '
            'loop together, a wave of rows at a time, wherever several share '
            'one (`BatchLoop._gen_grouped_ir`).  Without it that happens only '
            'where the body needs it: where the whole wave issues an '
            'instruction at once, a matrix fragment product, or where the '
            'target cannot synchronize part of a wave.  A row whose element is '
            'past the end or held back by its flag then runs the body on its '
            'group\'s element and writes nothing, instead of leaving the loop '
            'or skipping the body, so a group whose rows are all held back '
            'still computes.  What it buys is one body: the loop per row '
            'issues the transfers for the next element in both arms of its '
            'flag test.  Static ISA from clang 18 over the SeisSol kernels of '
            '`wave_uniform_rows`: on gfx942, where most multiplications take '
            '32 or 16 of the wave\'s 64 lanes, the 46 kernels it groups have '
            '7 % fewer instructions in the median and 29 % fewer in sum -- '
            '`accumulateIntegrals` and `stateToTransport` 40 to 59 % fewer, '
            'their bodies being little more than the transfers -- and 10 of '
            'them up to 2.5 % more; on gfx1150 the 8 kernels of 16 lanes, 6 % and '
            '13 %.  One kernel takes every register: the VTK volume projection '
            'at order 8, 511 VGPRs and 255 AGPRs against 313 and 57, since '
            'with no flag test around its reads LLVM takes the 480 of its '
            'constant operator out of the loop.  What the option is worth at '
            'run time is not measured.  A section that follows another '
            'without a barrier starts rotated and stays per row.')

declare('wave_uniform_group_rows',
        default=False,
        parse=parse_bool,
        doc='Where the multiplications of one wave take their trips through '
            'the batch loop as a group, read the row of each as the wave\'s, '
            'and with it its flag word and its element pointers: one copy per '
            'multiplication, each a scalar the whole wave holds, and every '
            'lane picks its own (AMD; `pir.rows`, `tensorforge::pick`).  The '
            'loop counts on the scalar unit, a pointer and a flag word are '
            'scalar loads through the scalar cache, and a wait for them does '
            'not queue behind the vector loads in flight -- for G '
            'multiplications to a wave, G of each, picked with a '
            '`v_cndmask` per 32-bit word.  Only where the group is the wave '
            'and one element a trip: not for a block-wide group '
            '(`stage_members`) and not under `stack_elements`.  Static ISA '
            'from clang 18 for gfx942 over the SeisSol kernels of '
            '`wave_uniform_rows`, all of them grouped '
            '(`wave_group_traversal`): 10 % fewer vector loads, 951 of 9100, '
            'and 7 VGPRs fewer in the median, for 13 % more instructions in '
            'the median and 9 % more in sum, every kernel longer -- 7 % for '
            'the 71 whose matrix products group the wave anyway, 12 % at two '
            'multiplications to a wave and 95 % at four.  What the scalar '
            'loads are worth at run time is not measured.')

declare('lead_vectorize',
        default=False,
        env='TF_LEAD_VEC',
        doc='Vectorize the lead dimension.  Off by default, and not out of doubt '
            'about the mechanism: it changes the thread count of every kernel, and '
            'the only instrument that can say whether that was a good idea is a '
            'register and occupancy measurement on real hardware.  The host oracle '
            'checks that the numbers still come out right; it cannot check that '
            'they come out faster.')

declare('lead_blocking',
        default=1,
        env='TF_LEAD_BLOCK',
        doc='Vectors per lane in the lead dimension.  1 keeps the arrangement the '
            'width alone produces; 2 is where the packed FMA starts paying for its '
            'own splat.  Separate from `lead_vectorize` because it is a *register* '
            'decision and the width is an instruction one -- they want separate '
            'measurements.')

declare('k_width',
        default=1,
        env='TF_K_WIDTH',
        doc='Reduction steps one body covers.  Independent of the lead width: it '
            'removes loads of the broadcast operand rather than instructions on '
            'the vectorized one, and it works with or without a lead width at all.')

declare('fma_interleave',
        rule=lambda target: target.prefs.fma_interleave,
        env='TF_FMA_INTERLEAVE',
        parse=parse_bool,
        doc='Write each step of the nest\'s accumulation as a statement of its '
            'own, where the nest puts it: the reduction outermost, every '
            'output a lane holds -- slot and column -- inside it, so that two '
            'FMAs in a row write different accumulators.  Off, the emitter '
            'folds an accumulator\'s single-use chain into one expression at '
            'its last use (`inline_pir_values`), and the text, and the IR the '
            'device compiler is handed, run the reduction innermost: one '
            'accumulator\'s FMAs back to back, each waiting for the one before '
            'it.  IGC keeps that order: on PVC 82 % of the accumulating `mad`s '
            'of `localFluxAll` at order 6 in double wait on the one right '
            'before them, and 0.3 % with this.  The same additions in the same '
            'order per accumulator, so the numbers do not change, and no '
            'register is added.  The products stay inlined into their '
            'accumulation, so a compiler that contracts only within one '
            'expression still forms the FMA.  Nothing changes where nothing '
            'is inlined (`inline_pir_values` off, or `pin_loads` pinning), '
            'nor for the matrix and DPP paths, which write their own order.  '
            'On by default under DPC++ on Intel (`Preferences.fma_interleave`).')

declare('fma_interleave_group',
        rule=lambda target: target.prefs.fma_interleave_group,
        env='TF_FMA_INTERLEAVE_GROUP',
        parse=parse_int,
        doc='Under `fma_interleave`, at most this many outputs a lane holds in '
            'flight: the reduction runs over a group of them to its end before '
            'the next group starts, so a group\'s partial sums are final '
            'before the next group\'s first step.  0 interleaves every output.  '
            'Where the groups are cut is `fma_interleave_cut`; a group is never '
            'smaller than one column, or one slot.  On PVC, in flight at once, '
            '`localFluxAll` at order 4 in double kept IGC\'s first attempt and '
            '17 KB of spill (2.5x the default\'s time); in groups, IGC '
            'compiles it again, as it does the default, without spill.  Eight '
            'by default under DPC++ on Intel '
            '(`Preferences.fma_interleave_group`).')

declare('fma_interleave_cut',
        default='auto',
        env='TF_FMA_INTERLEAVE_CUT',
        parse=parse_str,
        doc='Where `fma_interleave_group` cuts the outputs a lane holds: '
            '`columns`, a group every slot of some columns; `slots`, a group '
            'every column of some whole slots (at lead width 1); or `auto`, '
            'for each product: no cut where a lane holds at least '
            '`multilinear.AUTO_SLOT_CUT_FROM` slots of it, between columns '
            'below.  Between slots the first operand\'s values die with the '
            'group that reads them; between columns they are read once a '
            'group, or held across the groups where the compiler merges the '
            'reads, as IGC does.  The broadcast of the second operand\'s step, '
            'which a cut between slots repeats once a group, is under SPMD a '
            'register region IGC reads in place, not an instruction.  On the '
            'Max 1550 every output in flight was the fastest from four slots '
            '(order 6 and up), and groups of eight columns-wise below.')

declare('fma_partial_sums',
        default=1,
        env='TF_FMA_PARTIAL_SUMS',
        doc='Accumulate each output of the nest in this many partial sums, '
            'the steps of the reduction dealt to them in turn, and add them '
            'up behind the reduction.  For a lane that holds too few outputs '
            'for `fma_interleave` to hide an FMA\'s latency behind the '
            'others\'.  Each partial sum beyond the first is a register image '
            'as large as the accumulator, and the sum is reassociated: the '
            'result differs from the plain order by rounding.  Only a '
            'reduction written out whole (not one `k_roll` or `k_unroll_max` '
            'rolls, whose step is a run-time value), at `k_width` 1, and a '
            'sum; it implies `fma_interleave`, without which the compiler '
            'would get the partial chains one after the other again.  1 is '
            'off.')

declare('swizzle',
        default='decide',
        parse=parse_str,
        env='TF_SWIZZLE',
        doc='How a staged shared-memory window is permuted against bank '
            'conflicts.  `decide`, the default, builds the body with every '
            'window plain, reads the accesses it makes lane by lane, and '
            'builds it again with a permutation where one lowers the bank '
            'cycles by more than its own arithmetic costs (`pir.banks.propose`) '
            '-- which a window read the same in every lane, as the second '
            'operand of a product is, never does.  `volume` permutes every '
            'window whose volume admits it, as before the decision existed; '
            '`off` never permutes.  Tiles of the matrix paths keep the '
            'permutations they choose themselves.')

declare('stage_row_bytes',
        default=0,
        env='TF_STAGE_ROW_BYTES',
        doc='Pad the leading dimension of a staged shared-memory image up to a '
            'multiple of this many bytes.  0 copies the storage box as it is.\n'
            'The image is our own layout -- the tensor keeps its own in global '
            "memory -- so this costs shared memory, and the width of a copy "
            'into it from global memory: every access of that copy keeps a row '
            'whole, so rows of 9 floats are copied an element at a time.  What '
            'it buys is a row whose length a wide access divides: `k_width` 2 '
            "packs SeisSol's damage step only out of the buffers whose row is "
            'even, and every one of its 4216 refusals is a row of 125.  The pad '
            'is never read: a group that does not fit whole stays scalar.\n'
            'Under the explicit-vector lowering (ESIMD) only an image written '
            'out of registers is padded: a copy from global memory is a vector '
            'over the lanes from one address, which a longer row would split.')

declare('ir_debug',
        default='',
        env='TF_IR_DEBUG',
        codegen=False,
        doc='Diagnostics from the pseudo-IR.  Any non-empty value reports verifier '
            'findings and declined prefetches; a value containing `dump` also dumps '
            'each pass in the pipeline.')

declare('ir_stats',
        default=False,
        env='TF_IR_STATS',
        codegen=False,
        doc='Print node count and register pressure for every emitted body.')
