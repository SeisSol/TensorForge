from tensorforge.common.matrix.tensor import Tensor, SubTensor
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.spp import MaskSPP

class YatetoInterface:
  def __init__(self):
    pass

  @classmethod
  def deduce_bbox(cls, yateto_ranges, mem_layout, permute):
    """Converts yateto memory layout (bounding boxes) and ranges to GemmForge bounding boxes i.e.,
       a box is a list of rows and columns indices where the actual data is located within
       a memory patch and should be computed

    Args:
      yateto_ranges (set[loopRanges]): a range of rows and columns to operate on
      mem_layout (BoundingBox): memory layout given as yateto bounding box

    Returns:
      (list): bounding box in GemmForge format
    """
    # TODO: permute, mem_layout
    
    return BoundingBox([rng.start for rng in mem_layout], [rng.stop for rng in mem_layout])
  
  @classmethod
  def gen_matrix(cls,
                       yateto_ranges,
                       yateto_memory_layout_bbox,
                       addressing,
                       name,
                       is_tmp,
                       permute,
                       pattern=None,
                       values=None,
                       datatype=None):

    spp = None
    if pattern is not None:
      spp = MaskSPP(pattern)
    
    data = None
    if values is not None:
      data = values

    chainforge_bbox = cls.deduce_bbox(yateto_ranges=yateto_ranges,
                                      mem_layout=yateto_memory_layout_bbox,
                                      permute=permute)
    tensor = Tensor(shape=[yateto_ranges[i] for i in range(len(yateto_ranges))],
                    addressing=addressing,
                    bbox=chainforge_bbox,
                    alias=name,
                    is_tmp=is_tmp,
                    spp=spp,
                    data=data,
                    datatype=datatype)
    return tensor
    return SubTensor(tensor, chainforge_bbox)
