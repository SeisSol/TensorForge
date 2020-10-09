from .matrix import DenseMatrix

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
      is_trans (bool): if true then a GemmForge bonding box needs to be transposed

    Returns:
      (list): bounding box in GemmForge format
    """
    if transpose:
      last, first = yateto_ranges
    else:
      first, last = yateto_ranges

    return [first.start - mem_layout[0].start,
            last.start - mem_layout[1].start,
            first.stop - mem_layout[0].start - 1,
            last.stop - mem_layout[1].start - 1]


  @classmethod
  def produce_dense_matrix(cls,
                           yateto_ranges,
                           yateto_memory_layout_bbox,
                           addressing,
                           transpose):

    gemmforge_bbox = cls.deduce_bbox(yateto_ranges=yateto_ranges,
                                     mem_layout=yateto_memory_layout_bbox,
                                     transpose=transpose)

    return DenseMatrix(num_rows=yateto_memory_layout_bbox[0].stop,
                       num_cols=yateto_memory_layout_bbox[1].stop,
                       addressing=addressing,
                       bbox=gemmforge_bbox,
                       transpose=transpose)