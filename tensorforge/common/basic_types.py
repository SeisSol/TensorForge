import enum


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
    elif self == self.I8:
      return 1
    elif self == self.I16:
      return 2
    elif self == self.I64:
      return 8

  def __str__(self):
    return self.as_str(self)

  def literal(self, value):
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
           Datatype.F16: 'int16_t',
           Datatype.I32: 'int32_t',
           Datatype.I64: 'int64_t',}
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
           'int64_t': Datatype.I64}
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
  ALPHA_SYMBOL_NAME = 'alpha'
  BETA_SYMBOL_NAME = 'beta'
  FLAGS_NAME = 'flags'
  GLOBAL_MEM_PREFIX = 'glb_'
  TOTAL_SHR_MEM = 'totalShrMem'
  LOCAL_SHR_MEM = 'localShrMem'
  SHR_MEM_REGION_PREFIX = 'shrRegion'
  REG_NAME = 'reg'
