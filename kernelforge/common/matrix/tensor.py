from abc import ABC, abstractmethod
from .spp import SparsityPattern, FullSPP
from .boundingbox import BoundingBox
from functools import reduce
from typing import List, Union
from ..basic_types import Addressing, FloatingPointType, DataFlowDirection
from kernelforge.common.exceptions import GenerationError

class Tensor:
    def __init__(self,
        shape: List[int],
        addressing: Addressing,
        bbox: Union[List[int], None]=None,
        alias: Union[str, None]=None,
        is_tmp: bool = False,
        spp: SparsityPattern = None,
        data: Union[List[float], None] = None):
        self.name = None
        self.alias = alias
        self.shape = shape
        self.is_tmp = is_tmp
        self.direction: Union[DataFlowDirection, None] = None
        self.data = data
        self.spp = spp

        if self.spp is None:
            self.spp = FullSPP(self.shape)

        if bbox is not None:
            self.bbox = bbox
        else:
            self.bbox = BoundingBox([0] * len(shape), shape)

        if isinstance(addressing, Addressing):
            self.addressing = addressing
            self.ptr_type = Addressing.addr2ptr_type(self.addressing)
        else:
            raise ValueError(f'Invalid matrix addressing type, given: {type(addressing)}')
        
        if self.addressing == Addressing.SCALAR:
            # allow higher-order tensors, if they're effectively a scalar anyways
            assert all(d == 1 for d in self.shape)

        # check whether bbox was given correctly
        if any(dimshape < dimsize for dimshape, dimsize in zip(self.shape, self.bbox.sizes())):
            raise GenerationError(f'Tensor {self} is smaller than bounding box {self.bbox}')

        if any(dimshape < dimupper for dimshape, dimupper in zip(self.shape, self.bbox.upper())):
            raise GenerationError(f'Bounding box {self.bbox} is smaller than tensor {self}')

    def set_data_flow_direction(self, direction: DataFlowDirection):
        if self.direction is None or self.direction == direction:
            self.direction = direction
        else:
            self.direction = DataFlowDirection.SOURCESINK

    def get_values(self):
        return self.data
    
    def value(self, index):
        return self.data[tuple(index)]

    def get_actual_shape(self):
        return self.bbox.sizes()

    def get_actual_volume(self):
        return reduce(lambda x,y:x*y, self.get_actual_shape(), 1)
    
    def get_real_shape(self):
        return self.shape

    def get_real_volume(self):
        return reduce(lambda x,y:x*y, self.get_real_shape(), 1)

    def get_offset_to_first_element(self):
        return '0' # self.bbox.first_element()

    def get_bbox(self):
        return self.bbox

    def _set_name(self, name):
        self.name = name

    def is_similar(self, other):
        is_similar = self.shape == other.shape
        is_similar &= self.addressing == other.addressing
        is_similar &= self.bbox == other.bbox
        return is_similar

    def is_same(self, other):
        return self.is_similar(other) and self.alias == other.alias and self.is_tmp == other.is_tmp

    def __str__(self):
        return self.name

    def gen_descr(self):
        return f'{self.name} {"×".join(str(d) for d in self.shape)}({"×".join(str(d) for d in self.bbox.sizes())}) {self.bbox} {self.addressing}'

    def density(self):
        return self.spp.count_nz() / self.get_actual_volume()
    
    def sparsity(self):
        return 1 - self.density()
    
    def __str__(self):
        return self.gen_descr()

class TensorWrapper:
    pass

class SubTensor(TensorWrapper):
    def __init__(self,
        tensor: Tensor,
        bbox: Union[BoundingBox, None] = None):
        self.tensor = tensor
        self.bbox = bbox

class FullTensor(TensorWrapper):
    def __init__(self, tensor: Tensor):
        self.tensor = tensor

