
class Instruction:
    def __init__(self):
        pass

    def emit(self):
        pass

    def can_emit(self):
        pass

    def invar(self):
        return set()

    def outvar(self):
        return set()

    def inoutvar(self):
        return self.invar() | self.outvar()

    # TODO: keep that one here?
    def condition(self):
        return True
    
    def __repr__(self):
        return f'{type(self).__name__}: {",".join(str(var) for var in self.invar())} -> {",".join(str(var) for var in self.outvar())}'

class Block(Instruction):
    def __init__(self, instructions):
        self.instructions = instructions

    def emit(self):
        return [instr.emit() for instr in self.instructions]

    def can_emit(self):
        return all(instr.can_emit() for instr in self.instructions)

