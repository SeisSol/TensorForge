from .memory import MemorySpace
from ..type import Datatype
import numpy as np
from typing import Union

class TensorData:
    def __init__(self, datatype: Datatype, shape: Union[tuple, list], spp, values=None):
        self.shape = tuple(shape)
        self.spp = spp
        self.values = values
    
    def __repr__(self):
        return ''

class TensorAlloc:
    def __init__(self, name: str, data: TensorData, memoryspace: MemorySpace):
        self.name = name
        self.data = data
        self.memoryspace = memoryspace
    
    def __repr__(self):
        return f'{self.name}{"{"}{self.data} :: {self.memoryspace}{"}"}'

class TensorView:
    def __init__(self, alloc: TensorAlloc, bbox):
        self.alloc = alloc
        self.bbox = bbox
    
    def __repr__(self):
        return f'{self.alloc}[{self.bbox}]'

class Index:
    def __init__(self):
        pass

    def __repr__(self):
        return 'idx'
