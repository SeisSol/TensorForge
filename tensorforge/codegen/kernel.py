
class Kernel:
    def __init__(self, name, code, num_threads, shmem, args, gridsync, persistent):
        self.name = name
        self.code = code
        self.num_threads = num_threads
        self.shmem = shmem
        self.args = args
        self.gridsync = gridsync
        self.persistent = persistent

    def source(self):
        raise NotImplementedError()

    def launcher(self):
        raise NotImplementedError()
    
    def cpp(self):
        pass

class SingleSourceKernel:
    def cpp(self):
        return self.source() + '\n' + self.launcher()
