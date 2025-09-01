from .instruction import Block

class Kernel:
    def __init__(self, root: Block):
        self.root = root

class GlobalBlock(Block):
    pass

class GridBlock(Block):
    def __init__(self):
        pass

class EntryBlock:
    def __init__(self, lanes):
        pass

class Loop(Block):
    pass
