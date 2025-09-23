from enum import Enum

class CppBaseDatatype(Enum):
    I8 = 0,
    I16 = 1,
    I32 = 2,
    I64 = 3,
    U8 = 4,
    U16 = 5,
    U32 = 6,
    U64 = 7,
    F32 = 10,
    F64 = 11,

    def cpp(self):
        return {
            I8: 'int8_t',
            I16: 'int16_t',
            I32: 'int32_t',
            I64: 'int64_t',
            U8: 'uint8_t',
            U16: 'uint16_t',
            U32: 'uint32_t',
            U64: 'uint64_t',
            F32: 'float',
            F64: 'double',
        }[self]
    
    def literal(self, value):
        return {
            I8: lambda x: f'static_cast<int8_t>({x}LL)',
            I16: lambda x: f'static_cast<int16_t>({x}LL)',
            I32: lambda x: f'static_cast<int32_t>({x}LL)',
            I64: lambda x: f'static_cast<int64_t>({x}LL)',
            U8: lambda x: f'static_cast<uint8_t>({x}ULL)',
            U16: lambda x: f'static_cast<uint16_t>({x}ULL)',
            U32: lambda x: f'static_cast<uint32_t>({x}ULL)',
            U64: lambda x: f'static_cast<uint64_t>({x}ULL)',
            F32: lambda x: f'{x:0.16}f',
            F64: lambda x: f'{x:0.16}',
        }[self](value)

class CppVectorDatatype:
    def __init__(self, base, count):
        self.base = base
        self.count = count
    
    def cpp(self):
        return 'TODO'
    
    def literal(self, value):
        pass

class CppConstant:
    def __init__(self, value, datatype):
        self.value = value
        self.datatype = datatype

    def cpp(self):
        return f'{self.datatype.literal(self.value)}'

    def contract(self):
        return self

    def constant(self):
        return True

class CppVariable:
    def __init__(self, value, datatype):
        self.value = value
        self.datatype = datatype

    def cpp(self):
        return f'{self.datatype.literal(self.value)}'

    def contract(self):
        return self

    def constant(self):
        return True

class CppUnaryOp(Enum):
    PLUS = 0
    MINUS = 1
    MUL = 2
    DIV = 3
    MOD = 4

class CppBinaryOp(Enum):
    PLUS = 0
    MINUS = 1
    MUL = 2
    DIV = 3
    MOD = 4

class CppExpression:
    def __init__(self, op, values):
        self.op = op
        self.values = values

    def datatype(self):
        return self.op.datatype(*[value.datatype() for value in self.values])

    def cpp(self):
        if self.constant():
            result = self.contract()
            return self.datatype().literal(result)
        else:
            return f'({self.value1.cpp()} {self.op} {self.value2.cpp()})'

    def contract(self):
        v = [value.contract() for value in self.values]
        return self.op.combine(*v)

    def constant(self):
        return all([value.constant() for value in self.values])

class CppFunctionCall:
    def __init__(self, function, args):
        self.name = function
        self.args = args

    def cpp(self):
        args = ', '.join([arg.cpp() for arg in self.args])
        return f'{self.name}({args})'

class CppFunctionDefinition:
    def __init__(self):
        pass

class CppBlock:
    def __init__(self, parts):
        self.parts = parts
    
    def cpp(self):
        return ';\n'.join([part.cpp() for part in self.parts])

class CppIf(CppBlock):
    def __init__(self, parts, condition):
        self.condition = condition
        super().__init__(parts)
    
    def cpp(self):
        return f"""if ({self.condition}) {{ {super().cpp()} }}"""

class CppFor(CppBlock):
    def __init__(self, parts, variable, start, end, step):
        self.variable = variable
        self.start = start
        self.end = end
        self.step = step
        super().__init__(parts)
    
    def cpp(self):
        return f"""for ({self.variable} = {self.start}; {self.variable} < {self.end}; {self.variable} += {self.step}) {{ {super().cpp()} }}"""

