# for most of these:
# load args
# execute
# store result

# TODO: handle memlayout changes more properly

from ..instruction import Instruction

class Permute(Instruction):
    def __init__(self, outtensor, tensor, permute):
        self.outtensor = outtensor
        self.permute = permute
        self.tensor = tensor

    def invar(self):
        return set([self.tensor])

    def outvar(self):
        return set([self.outtensor])

class Matmul(Instruction):
    def __init__(self, outtensor, opSum, opMul, tensor1, tensor2, tensorAcc):
        self.opSum = opSum
        self.opMul = opMul
        self.tensor1 = tensor1
        self.tensor2 = tensor2
        self.outer = outer
        self.tensorAcc = tensorAcc
        self.outtensor = outtensor

    def invar(self):
        return set([self.tensor1, self.tensor2, self.tensorAcc])

    # TODO: enforce tensor2 to be broadcasted if transposed.
    # i.e., make:
    # * load tensorAcc if needed; else zero tensorAcc
    # * load tensor1
    # * load-broadcast tensor2
    # * fma tensorAcc, tensor1, tensor2
    # * store outtensor

    def outvar(self):
        return set([self.outtensor])

class Multilinear(Instruction):
    def __init__(self, outtensor, opSum, opMul, args, target, tensorAcc):
        self.opSum = opSum
        self.opMul = opMul
        self.args = args
        self.target = target
        self.tensorAcc = tensorAcc
        self.outtensor = outtensor

    def invar(self):
        if self.tensorAcc:
            return set([self.tensorAcc] + list(self.args))
        else:
            return set(list(self.args))

    def outvar(self):
        return set([self.outtensor])

class Elementwise(Instruction):
    def __init__(self, outtensor, op, args):
        self.outtensor = outtensor
        self.op = op
        self.args = args
        # TODO: check that all args have the same dimensions

    def invar(self):
        return set(self.args)

    def outvar(self):
        return set([self.outtensor])

class Reduction(Instruction):
    def __init__(self, outtensor, op, tensor, dim):
        self.outtensor = outtensor
        self.op = op
        self.tensor = tensor
        # TODO: check that

    def invar(self):
        return set([self.tensor])

    def outvar(self):
        return set([self.outtensor])
