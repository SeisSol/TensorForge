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
        default=True,
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
            'a question applies to it.')

declare('enable_wrap_loads',
        default=False,
        doc='Prefetch across the back edge: issue each transfer for the next '
            'element at the tail of the current iteration, after the last '
            'instruction that touches its buffer -- where '
            '`enable_move_loads` would put it in the loop unrolled once.  One '
            'buffer copy, register and shared destinations alike; see '
            'backend/pir/wrap.py.')

declare('move_distance',
        default=1,
        doc='How many transfers a transfer is moved ahead by.  '
            '`enable_move_loads` lets one travel past this many earlier '
            'transfers before it stops (1: the one before it); '
            '`enable_wrap_loads` wraps the transfers whose move runs across '
            'the back edge, which are the first this many of the body.  A '
            'dependence stops a transfer whatever the distance.')

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
        default=False,
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
            '(`Generator._shard_budget`), or `preload_shard_budget`.  Off '
            'until it is measured: at order 8 in double precision no operator '
            'fits an AMD block whole, and that is what this is for.')

declare('preload_shard_budget',
        default=0,
        parse=parse_int,
        doc='Bytes of shared memory a block\'s shards (`preload_shards`) may '
            'take, up to what the block can take at all, in place of what '
            'keeps the blocks an SM holds -- which counts no registers, so '
            'where they bind first, as on AMD at order 8, a block can take '
            'more than that at no cost.  0 is that default.')

declare('stage_members',
        default=False,
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
            'of `stage_group` multiplications.')

declare('stage_group',
        default=8,
        doc='Multiplications a block holds under `stage_members`, and so how '
            'many share one staged copy.  Rounded up to whole waves.')

declare('prepare_operands',
        default=False,
        parse=parse_bool,
        doc='Let the backend re-encode a batch-constant operand for the '
            'instruction that reads it -- store it in fragment order, split it '
            'into parts, or both.\n'
            'A question and not a vendor rule because preparing an operand '
            'moves work onto whoever fills the buffer: a host that cannot run '
            'the packer cannot use the kernel, and only the caller knows '
            'whether it can.  What preparing *means* is the primitive\'s '
            'business; this only says whether it may.\n'
            'Off by default.  On `local_flux` at batch 8192 the fragment order '
            'alone was worth 16%, and 27% with the TF32 split on top -- but '
            'only `Addressing.NONE` operands are eligible at all, so a case '
            'with none of them pays the question with nothing.\n'
            'On the FMA path it stores such an operand in the SIMT interleave '
            '(`Tensor.simt_interleave`): each lane\'s rows side by side in '
            '16-byte groups, read with one aligned vector load.  Offered where '
            'the rows fill the lanes -- `local_flux` at 8 lanes, where it cut '
            'the global loads of the rolled merged kernel and made it 4 % faster.')

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
            'this generator emits today.')

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
            'code; on GB200 22 to 35 % (preferences.yml).  With `k_roll` as well '
            'it gained less, the merged rolled body taking more registers.')

declare('merge_icache_fraction',
        default=0.25,
        parse=parse_float,
        doc='How much of the instruction cache a kernel written out may take '
            'before `merge_variants=auto` merges its repeated runs.  Generous on '
            'purpose: merging measured faster wherever it was tried, and the '
            'code size is an estimate that runs up to +89 % over what the '
            'compiler emits at the 90th percentile (`analysis.icache`).  Where '
            'the target states no instruction cache (Intel), `auto` merges '
            'nothing.')

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

declare('early_writebacks',
        default=True,
        parse=parse_bool,
        env='TF_EARLY_WRITEBACKS',
        doc='Store a result that stayed in registers as soon as the rest of '
            'the section neither reads nor writes it, rather than with every '
            'other one at the section\'s end.  Held to the end, SeisSol\'s '
            'elastic time derivative (order 6) kept all six `dQ(k)` in '
            'registers through the whole kernel -- `dQ(0)` for 19,000 of its '
            '19,500 lines -- 72 registers of 255 that only waited for a store.')

declare('register_temporaries',
        default='all',
        env='TF_REGISTER_TEMPORARIES',
        doc='Which temporaries a pointwise operation -- elementwise, a '
            'reduction, a contraction to one value -- reads straight out of '
            'the register image its producer left them in, rather than having '
            'the image stored to its shared-memory buffer first and loading it '
            'back: `none`; `scalars`, the temporaries without axes, which are '
            'one value on every lane and need neither a buffer nor a barrier; '
            '`all`, arrays as well where the image spreads the lanes the way '
            'the buffer would.  SeisSol\'s damage step (order 4, single, 32 '
            'lanes) stored 398 scalars and loaded them 2061 times, each store '
            'followed by a barrier; arrays trade the shared buffer for '
            'registers that stay live until their last reader.  `all` by '
            'default, on measurement: over 70 corpus cases on sm_120 a '
            'geomean of 1 %, with `mixed/ew_then_ew` at +72 % and five cases '
            '4-6 % the other way, and SeisSol\'s damage step 6 % -- 11 % once '
            'its material part is a kernel of its own.  A kernel the trade '
            'does not suit says so through the tuner, which carries this '
            'knob (`tuning.simple_space`).')

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
            'those numbers, not of its own.')

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
            'take a matching preference first.  Only what is safe to ship unasked is '
            'turned: the lane count, a lead width of two where the target has '
            'a packed FP32 FMA, merging, and rolling -- not preparing operands, '
            'which the host has to pack for, and not the matrix path.  Not part '
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
        doc='Use the NVIDIA matrix instructions (`mma.sync`) where the shape '
            'allows.  None takes `primitives.nvidia.ENABLED`, the deployment '
            'switch, which is off; an option so that one build can ask for the '
            'path and the next one not, as a search over configurations has '
            'to -- flipping the module constant changes it for every build in '
            'the process.')

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
        default=0,
        env='TF_LANES',
        doc='Lanes one multiplication is spread over, instead of the deduced '
            'count.  0 keeps the deduction.  Only narrower counts are taken, '
            'and only powers of two, so that a warp still holds whole '
            'multiplications; the rows a lane covers grow to match, with the '
            'last ones padded.  A search parameter and not a default: on sm_120 '
            'eight lanes made `local_flux` 10 % faster and `chain_three` four '
            'times slower, because it spilled -- see `lanes.narrower`.')

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

declare('stage_row_bytes',
        default=0,
        env='TF_STAGE_ROW_BYTES',
        doc='Pad the leading dimension of a staged shared-memory image up to a '
            'multiple of this many bytes.  0 copies the storage box as it is.\n'
            'The image is our own layout -- the tensor keeps its own in global '
            "memory -- so this costs shared memory and nothing else.  What it "
            'buys is a row whose length a wide access divides: `k_width` 2 '
            "packs SeisSol's damage step only out of the buffers whose row is "
            'even, and every one of its 4216 refusals is a row of 125.  The pad '
            'is never read: a group that does not fit whole stays scalar.')

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
