from enum import Enum

class Datatype:
    pass

class BaseDatatype(Datatype, Enum):
    BOOL = 0
    F32 = 100
    F64 = 101
    F16 = 102
    BF16 = 103
    # TODO for smaller FP types, if ever needed
    I8 = 200
    I16 = 201
    I32 = 202
    I64 = 203
    U8 = 300
    U16 = 301
    U32 = 302
    U64 = 303
    PTROFFSET = 400
    PTRADDR = 401
    SIZE = 402

    def __repr__(self):
        return {
            BOOL: 'b',
            F32: 'f32',
            F64: 'f64',
            F16: 'f16',
            BF16: 'bf16',
            I8: 'i8',
            I16: 'i16',
            I32: 'i32',
            I64: 'i64',
            U8: 'u8',
            U16: 'u16',
            U32: 'u32',
            U64: 'u64',
            PTROFFSET: 'ptr_offset',
            PTRADDR: 'ptr_addr',
            SIZE: 'size',
        }[self]

    @classmethod
    def ytt2enum(cls, as_str: str):
        map = {'f32': cls.F32,
            'f64': cls.F64,
            'f16': cls.F16,
            'bf16': cls.BF16,
            'bool': cls.BOOL,
            'i8': cls.I8,
            'i16': cls.I16,
            'i32': cls.I32,
            'i64': cls.I64,
            'u8': cls.U8,
            'u16': cls.U16,
            'u32': cls.U32,
            'u64': cls.U64}
        return map[as_str]

class PointerDatatype(Datatype):
    def __init__(self, base):
        self.base = base
    
    def __repr__(self):
        return f'{self.base}*'

class VectorDatatype(Datatype):
    def __init__(self, base, count):
        self.base = base
        self.count = count
    
    def __repr__(self):
        return f'{self.base}[{self.count}]'
