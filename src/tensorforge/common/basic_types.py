# SPDX-FileCopyrightText: 2026 SeisSol Group
#
# SPDX-License-Identifier: MIT
import enum
import math


class DataFlowDirection(enum.Enum):
  SOURCE = 0
  SINK = 1
  SOURCESINK = 2

class Addressing(enum.Enum):
  NONE = 0
  STRIDED = 1
  PTR_BASED = 2
  SCALAR = 3

  def __str__(self):
    return self.addr2str(self)

  @classmethod
  def addr2ptr_type(cls, addr_type):
    map = {Addressing.NONE: '*',
           Addressing.STRIDED: '*',
           Addressing.PTR_BASED: '**',
           Addressing.SCALAR: ''}
    return map[addr_type]

  @classmethod
  def str2addr(cls, string):
    map = {'none': Addressing.NONE,
           'strided': Addressing.STRIDED,
           'pointer_based': Addressing.PTR_BASED,
           'scalar': Addressing.SCALAR}
    if string not in map:
      raise ValueError(f'arg must be either none, strided, pointer_based, or scalar; given: {string}')
    return map[string]

  @classmethod
  def addr2str(cls, addr):
    map = {Addressing.NONE: 'none',
           Addressing.STRIDED: 'strided',
           Addressing.PTR_BASED: 'pointer_based',
           Addressing.SCALAR: 'scalar'}
    return map[addr]

  def __str__(self):
    return self.addr2str(self)

  def to_pointer(self):
    return self.addr2ptr_type(self)

class StridedAddressing:
  def __init__(self, offset, stride=None):
    self.offset = offset
    self.stride = stride

  def to_pointer(self):
    return Addressing.STRIDED.to_pointer()

  def __req__(self, other):
    return other == Addressing.STRIDED

def _nonfinite(value: float):
  """C++ spelling for the values Python prints without a numeric literal.

  `INFINITY` and `NAN` are `<cmath>` macros, usable in device code on every
  backend the lexics target, and `INFINITY` converts exactly to double, so one
  spelling serves both float and double.
  """
  if math.isnan(value):
    return 'NAN'
  if math.isinf(value):
    return '-INFINITY' if value < 0 else 'INFINITY'
  return None


class Datatype(enum.Enum):
  F32 = 0
  F64 = 1
  F16 = 2
  BF16 = 3
  F128 = 4
  BOOL = 10
  I8 = 20
  I16 = 21
  I32 = 22
  I64 = 23
  # Unsigned, and only where a vendor signature demands it.  An `I32` does not
  # bind to a `uint32_t &`, so the kernel does not compile -- which is the only
  # reason this member exists.  Nothing in the generator computes with it.
  U32 = 32
  # The 19-bit E8M10 the matrix units multiply: 1 sign, 8 exponent, 10
  # mantissa, stored in 32 bits.  A storage type and not an arithmetic one --
  # nothing here computes with it, values are *converted into* it and handed
  # to an instruction.
  #
  # It earns a member because spelling it `F32` is a lie no front end catches.
  # `simd<float, 128>` and `simd<tf32, 128>` are both well-formed and both
  # accepted by `xmx::dpas`; only one is the instruction's operand, and the
  # difference is 13 mantissa bits computed on silently.
  TF32 = 33

  def size(self):
    if self == self.F32:
      return 4
    elif self == self.F64:
      return 8
    elif self == self.F128:
      return 16
    elif self == self.F16:
      return 2
    elif self == self.BF16:
      return 2
    elif self == self.BOOL:
      return 1 # ?
    elif self == self.I32:
      return 4
    elif self == self.U32:
      return 4
    elif self == self.TF32:
      return 4
    elif self == self.I8:
      return 1
    elif self == self.I16:
      return 2
    elif self == self.I64:
      return 8

  def __str__(self):
    return self.as_str(self)

  def literal(self, value):
    if self in (Datatype.F32, Datatype.F64, Datatype.F16, Datatype.BF16,
                Datatype.F128):
      spelling = _nonfinite(float(value))
      if spelling is not None:
        # `f'{float("inf"):.16}f'` is `inff`, which is not C++.  It only ever
        # showed up once an operator's *neutral element* reached here --
        # `MinOperator.neutral()` is `math.inf` -- so nothing caught it while
        # literals came from user data.
        if self == self.F16:
          return f'static_cast<__half>({spelling})'
        if self == self.BF16:
          return f'static_cast<__bfloat16>({spelling})'
        return spelling
    if self == self.F32:
      return f'{float(value):.16}f'
    elif self == self.F64:
      return f'{float(value):.16}'
    elif self == self.F16:
      return f'static_cast<__half>({float(value):.16})'
    elif self == self.BF16:
      return f'static_cast<__bfloat16>({float(value):.16})'
    elif self == self.F128:
      return f'{float(value):.32}q'
    elif self == self.BOOL:
      return 'true' if value else 'false'
    elif self == self.I8:
      return f'{int(value)}_i8'
    elif self == self.I16:
      return f'{int(value)}_i16'
    elif self == self.I32:
      return f'{int(value)}_i32'
    elif self == self.U32:
      return f'{int(value)}u'
    elif self == self.TF32:
      # Through a float: the type is a storage format, and every platform's
      # spelling of it converts from one.  There is no TF32 literal syntax to
      # write instead.
      return f'tensorforge::tf32({float(value):.16}f)'
    elif self == self.I64:
      return f'{int(value)}_i64'

  @classmethod
  def as_str(cls, fp):
    map = {Datatype.F32: 'float',
           Datatype.F64: 'double',
           Datatype.F128: '__float128',
           Datatype.F16: 'half',
           Datatype.BF16: 'bfloat16',
           Datatype.BOOL: 'bool',
           Datatype.I8: 'int8_t',
           Datatype.I16: 'int16_t',
           Datatype.I32: 'int32_t',
           Datatype.I64: 'int64_t',
           Datatype.U32: 'uint32_t',
           Datatype.TF32: 'tensorforge::tf32',}
    return map[fp]

  def ctype(self):
    return self.as_str(self)

  @classmethod
  def str2enum(cls, as_str: str):
    map = {'float': Datatype.F32,
           'double': Datatype.F64,
           'half': Datatype.F16,
           'bfloat16': Datatype.BF16,
           'quad': Datatype.F128,
           'bool': Datatype.BOOL,
           'int8_t': Datatype.I8,
           'int16_t': Datatype.I16,
           'int32_t': Datatype.I32,
           'int64_t': Datatype.I64,
           'uint32_t': Datatype.U32,
           'tensorforge::tf32': Datatype.TF32}
    return map[as_str]

  @classmethod
  def ytt2enum(cls, as_str: str):
    map = {'f32': Datatype.F32,
           'f64': Datatype.F64,
           'f16': Datatype.F16,
           'f128': Datatype.F128,
           'bf16': Datatype.BF16,
           'bool': Datatype.BOOL,
           'i8': Datatype.I8,
           'i16': Datatype.I16,
           'i32': Datatype.I32,
           'i64': Datatype.I64}
    return map[as_str]


class GeneralLexicon:
  BATCH_ID_NAME = 'batchId'
  NUM_ELEMENTS = 'numElements'
  EXTRA_OFFSET = '_extraOffset'
  STREAM_PTR_STR = 'streamPtr'
  FLAGS_NAME = 'flags'
  GLOBAL_MEM_PREFIX = 'glb_'
  TOTAL_SHR_MEM = 'totalShrMem'
  LOCAL_SHR_MEM = 'localShrMem'
  REG_NAME = 'reg'


class FlagMode(enum.Enum):
  """What the per-element flag mask costs a kernel that does not use one.

  The mask is a ``unsigned*`` parameter per section plus a guard around the
  loop body, and a kernel whose caller never skips an element pays for both.
  Which of the three shapes a kernel gets is decided by the frontend that
  builds it, from the attributes the caller passed at kernel creation:

  ``ABSENT``
      No parameter and no guard.  The caller did not ask for a mask, so
      naming one is a compile error rather than a silently ignored write.
  ``REQUIRED``
      Parameter without a default, dereferenced unconditionally.  The caller
      asked for the mask and therefore has to supply it.
  ``OPTIONAL``
      Parameter defaulting to ``nullptr``, guarded by a null check.  This is
      what a frontend that passes no attributes at all gets, so a caller that
      predates the attribute channel keeps working unchanged.
  """

  ABSENT = 'absent'
  REQUIRED = 'required'
  OPTIONAL = 'optional'

  @classmethod
  def from_attrs(cls, attrs):
    """The mode a kernel attribute dictionary asks for.

    ``None`` is not the empty dictionary here: it means the frontend has no
    attributes to give, which is answered with ``OPTIONAL``.  An empty
    dictionary comes from a frontend that does have them and left the mask
    out, which is ``ABSENT``.
    """
    if attrs is None:
      return cls.OPTIONAL
    return cls.REQUIRED if attrs.get(GeneralLexicon.FLAGS_NAME, False) else cls.ABSENT
