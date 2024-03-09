from kernelforge.common.matrix.tensor import Tensor
from kernelforge.common.matrix.boundingbox import BoundingBox

class YatetoInterface:
  def __init__(self):
    pass

  @classmethod
  def deduce_bbox(cls, yateto_ranges, mem_layout, transpose):
    """Converts yateto memory layout (bounding boxes) and ranges to GemmForge bounding boxes i.e.,
       a box is a list of rows and columns indices where the actual data is located within
       a memory patch and should be computed

    Args:
      yateto_ranges (set[loopRanges]): a range of rows and columns to operate on
      mem_layout (BoundingBox): memory layout given as yateto bounding box

    Returns:
      (list): bounding box in GemmForge format
    """
    # TODO: transpose, mem_layout
    
    return BoundingBox([rng.start for rng in yateto_ranges], [rng.stop for rng in yateto_ranges])
  
  @classmethod
  def gen_matrix(cls,
                       yateto_ranges,
                       yateto_memory_layout_bbox,
                       addressing,
                       name,
                       is_tmp,
                       transpose,
                       pattern=None,
                       values=None):

    chainforge_bbox = cls.deduce_bbox(yateto_ranges=yateto_ranges,
                                      mem_layout=yateto_memory_layout_bbox,
                                      transpose=transpose)
    
    return Tensor(shape=[yateto_memory_layout_bbox[i].size() for i in range(len(yateto_memory_layout_bbox))],
                    addressing=addressing,
                    bbox=BoundingBox([rng.start for rng in yateto_ranges], [rng.stop for rng in yateto_ranges]),
                    alias=name,
                    is_tmp=is_tmp)
