# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
"""The generator's options: one place that declares them, one that resolves them.

An option is an *entry* and not a class attribute.  `declare` refuses a name it
already holds, so a second declaration of `wrap_distance` is an error at import
naming the option, rather than an assignment in a class body where the last one
silently wins.

Four layers answer for a value, nearest first:

1. what the caller passed to `Options`,
2. the option's own environment variable, where the declaration names one,
3. `TF_OPTIONS`, a comma-separated ``name=value`` list covering every option,
4. the vendor rule the declaration carries, or its plain default.

Layer 3 is what a caller that cannot reach the constructor uses -- the yateto
frontend builds its own context -- and layer 2 is for the few switches whose
spelling is already written down in tools and scripts.

`resolve` runs the layers against one hardware descriptor and returns a frozen
`ResolvedOptions`: one value per declared name, hashable, and carrying the
*delta* to what the same hardware would have produced had nothing been asked at
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
#: ``TF_OPTIONS=enable_pipeline=1,wrap_distance=2``.  A bare ``name`` means
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

  def base(self, hw) -> Any:
    """The value for this hardware when nobody asked for one."""
    return self.default if self.rule is None else self.rule(hw)

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

  def resolve(self, hw) -> 'ResolvedOptions':
    """Settle every declared option against this hardware descriptor."""
    from_env = _from_environment()
    asked = self.asked()
    values: Dict[str, Any] = {}
    delta: Dict[str, Any] = {}
    for name, opt in _REGISTRY.items():
      base = opt.base(hw)
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

  def as_dict(self) -> Dict[str, Any]:
    return dict(self._values)

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

declare('exact_contraction_length',
        default=False,
        doc='Cover the contraction range exactly rather than rounding it up to '
            'the lane count.')

declare('align_shr_mem',
        default=True,
        doc='Round every shared-memory allocation up to the access alignment.')

declare('enable_sync_block_opt',
        default=True,
        doc='Drop barriers a data-flow argument shows to be redundant.')

declare('enable_pipeline',
        default=False,
        doc='Software pipelining: advance the address computation of a transfer '
            'ahead of the iteration that consumes it.\n'
            'Off pending hardware numbers; correctness does not block it.')

declare('enable_multibuffer',
        default=False,
        doc='Rotate the shared-memory buffers on top of the advanced addresses. '
            'Needs `enable_pipeline`, since the rotation reads the advanced '
            'pointer, and is implemented for `pipeline_depth == 2` only -- see '
            'backend/opt/pipeline.py.')

declare('pipeline_depth',
        default=2,
        doc='Stages a rotating buffer holds.')

declare('enable_move_loads',
        default=True,
        parse=parse_bool,
        doc='Issue a load early and wait where the value is needed.\n'
            '`MoveLoads` splits a transfer from its `LoadWait` and walks the '
            'transfer up the stream to hide its latency, stopping where '
            'something between the two would write what the load reads.\n'
            'On by default and always has been -- this switch exists so the '
            'default can be *priced*, not because it is in doubt. A pass with '
            'no way to be turned off is a pass whose contribution nobody has '
            'measured, and the same argument that put `preload_globals` behind '
            'a question applies to it.')

declare('enable_wrap_loads',
        default=False,
        doc='Prefetch across the back edge: issue each transfer for the next '
            'element at the tail of the current iteration, after the last '
            'instruction that touches its buffer -- where MoveLoads would put '
            'it in the loop unrolled once.  One buffer copy, register and '
            'shared destinations alike; see backend/opt/wrap.py.')

declare('move_distance',
        default=1,
        doc='How many loads a transfer is moved ahead by.  `MoveLoads` lets a '
            'load travel past this many earlier loads before it stops (1: the '
            'one before it, as always); `WrapLoads` wraps the transfers whose '
            'move runs across the back edge, which are the first this many of '
            'the body.  A dependence stops a transfer whatever the distance.')

declare('wrap_distance',
        default=1,
        doc='Not read any more; superseded by `move_distance`.  `WrapLoads` '
            'placed register transfers this many compute slots ahead; it now '
            'places every transfer by dependence.  Still declared because '
            'callers pass it (the bench suite, the option tests).')

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

declare('preload_globals',
        rule=lambda hw: hw.vendor in ('amd',),
        parse=parse_bool,
        doc='Stage every `Addressing.NONE` operand into shared memory once per '
            'block, in the section prologue, instead of reading it from global '
            'inside the batch loop.\n'
            'A question and not a constant because the answer is a measurement '
            'nobody has taken on NVIDIA: the rule is "AMD only", so the whole '
            'NVIDIA path -- including the tensor-core one, where a batch-constant '
            'operand would also carry a batch-constant *conversion* -- has never '
            'been compared against its own alternative.  A benchmark cannot ask '
            'a question the generator cannot be asked.')

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
            'synchronises block-wide, and for work whose cost varies per '
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

declare('wide_bodies',
        default=True,
        env='TF_IR_WIDE',
        doc='One PIR body per loop body, rather than one per macro instruction.\n'
            'A pass sees a body.  Per macro instruction that means `RegisterAlloc`, '
            'the loader that fills the buffer and the multilinear that reads it are '
            'three separate bodies, and the only thing connecting them is the C++ '
            'name -- 60.7% of buffers in the corpus are named for that reason alone, '
            'against 10.3% per loop body (tools/buffer_spans.py).  Everything still '
            'needing a name here outlives one loop body: the shared arena, its '
            'scratch tail, and the tiles of the two cases that have two batch loops.\n'
            'So this is not primarily a code-quality switch -- the cross-instruction '
            'CSE win is 0.2% -- it is what makes the naming go away, and with it the '
            'reason `symbol.py` builds addresses as text.\n'
            '`TF_IR_WIDE=0` reaches it without touching a call site, because a '
            'setting that moves 71 of 108 generated outputs has to stay bisectable.')

declare('merge_variants',
        default=False,
        doc='Macro-op merging: state a repeated run of the descriptor list once '
            'and bind its varying operands to a counter.  One switch covers both '
            'the rewrite and the emission, so that a rolled list cannot be '
            'expanded again on the way out.\n'
            'Off by default.  On sm_120 `local_flux` (its four faces merged) '
            'computes the same checksum as the expanded list and runs 5 % faster '
            'at 32 lanes, 12 % at 16 and 24 % at 8, from a quarter to a half of '
            'the code; with `k_roll` as well it gained less, the merged rolled '
            'body taking more registers.')

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
        doc='Roll the reduction of a multilinear product into a real loop over '
            'groups of `k_width` steps, with `#pragma unroll <k_roll>` on it.  '
            '0 unrolls it completely in the generator, as always.  Only where '
            'every operand the reduction indexes lives in memory (global, '
            'batch or shared) -- a register image indexed at runtime would go '
            'to local memory -- the reduction is dense, and its extent divides '
            'into whole groups.  For kernels whose fully unrolled body no '
            'longer fits the instruction cache.')

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
            'to -- flipping the module constant changed it for every build in '
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
            'a slice of a larger image (theta) never has its neighbouring rows '
            'overwritten.')

declare('prefetch_data',
        rule=lambda hw: hw.explicit_simd,
        env='TF_PREFETCH_DATA',
        parse=parse_bool,
        doc='Hint the next element\'s data where `WrapLoads` would issue its '
            'transfer -- the tail of the loop body -- and leave the transfer '
            'where it is.\n'
            'The transfer itself moved costs a register image or a buffer '
            'that survives the back edge; a hint costs a pointer and a few '
            'messages, and changes no result.  One hint per cache span of '
            'each per-element source (`Lexic.prefetch_line_bytes`), for '
            '`PTR_BASED` and `STRIDED` operands; a batch-invariant one is '
            'already cached.  At `prefetch_level`.\n'
            'On under ESIMD: one work-item per element leaves nothing else to '
            'cover the latency of the next one, and the corpus pays 6.6 % '
            'instructions for it -- most in small kernels, which wait on '
            'memory anyway.  Elsewhere off until measured.')

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
        doc='Vectorise the lead dimension.  Off by default, and not out of doubt '
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
            'the vectorised one, and it works with or without a lead width at all.')

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
