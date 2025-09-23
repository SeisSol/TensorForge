
class MemorySpace:
    def __repr__(self):
        return f'{type(self).__name__}'

class GlobalMemory(MemorySpace):
    pass

class SharedMemory(MemorySpace):
    pass

class RegisterMemory(MemorySpace):
    def __init__(self, lanes):
        self.lanes = lanes

class LocalMemory(MemorySpace):
    def __init__(self, lanes):
        self.lanes = lanes

class Logical(MemorySpace):
    pass

class Temporary(MemorySpace):
    pass
