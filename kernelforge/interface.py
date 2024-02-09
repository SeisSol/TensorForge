from kernelforge.common.matrix.dense import DenseMatrix
from kernelforge.common.matrix.sparse import SparseMatrix

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
    if transpose:
      last, first = yateto_ranges
    else:
      first, last = yateto_ranges

    top_left_corner = (mem_layout[0].start, mem_layout[1].start)

    bbox = [first.start - top_left_corner[0],
            last.start - top_left_corner[1],
            first.stop - top_left_corner[0],
            last.stop - top_left_corner[1]]

    return bbox

  @classmethod
  def gen_dense_matrix(cls,
                       yateto_ranges,
                       yateto_memory_layout_bbox,
                       addressing,
                       name,
                       is_tmp,
                       transpose):

    chainforge_bbox = cls.deduce_bbox(yateto_ranges=yateto_ranges,
                                      mem_layout=yateto_memory_layout_bbox,
                                      transpose=transpose)

    return DenseMatrix(num_rows=yateto_memory_layout_bbox[0].size(),
                       num_cols=yateto_memory_layout_bbox[1].size(),
                       addressing=addressing,
                       bbox=chainforge_bbox,
                       alias=name,
                       is_tmp=is_tmp)
  
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
    if pattern is None:
        return DenseMatrix(num_rows=yateto_memory_layout_bbox[0].size(),
                        num_cols=yateto_memory_layout_bbox[1].size(),
                        addressing=addressing,
                        bbox=chainforge_bbox,
                        alias=name,
                        is_tmp=is_tmp)
    else:
        return SparseMatrix(num_rows=yateto_memory_layout_bbox[0].size(),
                        num_cols=yateto_memory_layout_bbox[1].size(),
                        addressing=addressing,
                        coordinates=pattern,
                        values=values,
                        bbox=chainforge_bbox,
                        alias=name,
                        is_tmp=is_tmp)
