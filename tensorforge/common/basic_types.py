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

class FloatingPointType(enum.Enum):
  FLOAT = 0
  DOUBLE = 1
  HALF = 2
  BFLOAT16 = 3
  BOOL = 10
  INT = 20
  UINT = 21
  LONG = 22
  ULONG = 23

  def size(self):
    if self == self.FLOAT:
      return 4
    elif self == self.DOUBLE:
      return 8
    elif self == self.HALF:
      return 2
    elif self == self.BFLOAT16:
      return 2
    elif self == self.BOOL:
      return 1 # ?
    elif self == self.INT:
      return 4
    elif self == self.UINT:
      return 4
    elif self == self.LONG:
      return 8
    elif self == self.ULONG:
      return 8

  def __str__(self):
    return self.as_str(self)
  
  def literal(self, value):
    if self == self.FLOAT:
      return f'{float(value):.16}f'
    elif self == self.DOUBLE:
      return f'{float(value):.16}'
    elif self == self.HALF:
      return f'static_cast<__half>({float(value):.16})'
    elif self == self.BFLOAT16:
      return f'static_cast<__bfloat16>({float(value):.16})'
    elif self == self.BOOL:
      return 'true' if value else 'false'
    elif self == self.INT:
      return f'{int(value)}'
    elif self == self.UINT:
      return f'{int(value)}u'
    elif self == self.LONG:
      return f'{int(value)}L'
    elif self == self.ULONG:
      return f'{int(value)}UL'

  @classmethod
  def as_str(cls, fp):
    map = {FloatingPointType.FLOAT: 'float',
           FloatingPointType.DOUBLE: 'double',
           FloatingPointType.HALF: 'half',
           FloatingPointType.BFLOAT16: 'bfloat16',
           FloatingPointType.BOOL: 'bool',
           FloatingPointType.INT: 'int',
           FloatingPointType.LONG: 'long',
           FloatingPointType.UINT: 'unsigned int',
           FloatingPointType.ULONG: 'unsigned long',}
    return map[fp]

  @classmethod
  def str2enum(cls, as_str: str):
    map = {'float': FloatingPointType.FLOAT,
           'double': FloatingPointType.DOUBLE,
           'half': FloatingPointType.HALF,
           'bfloat16': FloatingPointType.BFLOAT16,
           'bool': FloatingPointType.BOOL,
           'int': FloatingPointType.INT,
           'long': FloatingPointType.LONG,
           'unsigned int': FloatingPointType.UINT,
           'unsigned long': FloatingPointType.ULONG}
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
