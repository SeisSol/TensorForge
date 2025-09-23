from .logical.compute import *
from .logical.memory import *

class IRPass:
    def __init__(self):
        pass

    def run(self, ir):
        for instr in ir:
            self.apply(ir)
        return self

class IRTransform:
    def __init__(self):
        pass

    def run(self, ir):
        result = []
        for instr in ir:
            result += self.apply(ir)
        return result

class LivenessPass(IRPass):
    def __init__(self):
        self.liveset = []

    def apply(self, instr):
        pass

class LoadStore(IRTransform):
    def __init__(self, liveset):
        self.liveset = liveset

    def apply(self, instr, i):
        pass

class LowerLogicalCompute(IRTransform):
    # make logical compute to physical compute
    def apply(self, instr):
        expanded = []
        if isinstance(instr, Multilinear):
            for arg, tgt in zip(instr.args, instr.target):
                if len(tgt) > 0 and tgt[0] < 0:
                    expanded += [Broadcast(TODO, arg, [i for i in range(1, len(tgt))])]
            expanded += [TODO]
        return [instr]

class FuseBroadcasts(IRTransform):
    pass

def transformIR(ir):
    live = LivenessPass().run(ir).liveset
