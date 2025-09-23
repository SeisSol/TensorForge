
from .context import Context

class ThreadConfig:
    def __init__(self, context: Context, threadcount, multiples=None):
        self.warpsize = context.get_vm().get_hw_descr().vec_unit_length
        self.threadcount = threadcount

        if multiples is None:
            self.multiples = 256 // self.threadcount
        else:
            self.multiples = multiples

    def subwarp(self):
        return self.threadcount < self.warpsize

    def inwarp(self):
        return self.threadcount <= self.warpsize
    
    def superwarp(self):
        return self.threadcount > self.warpsize

    def warps_per_multiple(self):
        return self.threadcount // self.warpsize

    def multiples_per_warp(self):
        return self.warpsize // self.threadcount
    
    def blocksize(self):
        return self.threadcount * self.multiples

    def warps(self):
        return self.blocksize() // self.warpsize
