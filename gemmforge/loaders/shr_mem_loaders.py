from abc import ABC, abstractmethod
from .. import constructs
from .abstract_loader import AbstractShrMemLoader


class ExtendedPatchLoader(AbstractShrMemLoader):
  """A strategy which loads an entire matrix into shared memory.

  """

  def __init__(self, matrix, num_active_threads, load_and_transpose):
    super(ExtendedPatchLoader, self).__init__(matrix, num_active_threads, load_and_transpose)

    full_subvolume = (self.matrix.get_actual_num_cols() - 2) * self.matrix.num_rows
    cropped_subvolume = self.matrix.get_actual_num_rows() + self.matrix.num_rows
    self.shm_volume = cropped_subvolume + full_subvolume

    self.lid_dim = self.matrix.num_rows

  def compute_shared_mem_size(self):
    return self.shm_volume

  def generate_scr(self, file, in_symbol):

    file.Emptyline()
    file.Comment("using ExtendedPatchLoader")

    num_hops = int(self.shm_volume / self.num_active_threads)
    if num_hops > 0:
      if num_hops > self.manual_unroll_threshold:
        # load using a for-loop
        file.Pragma("unroll")
        with file.For("int i = 0; i < {}; ++i".format(num_hops)):

          shr_mem_index = "threadIdx.x + i * {}".format(self.num_active_threads)
          glob_mem_index = "threadIdx.x + i * {}".format(self.num_active_threads)
          file.Assignment("{}[{}]".format(self.out_symbol, shr_mem_index),
                          "{}[{}]".format(in_symbol, glob_mem_index))
      else:
        # load using manual loop unrolling
        for counter in range(num_hops):
          shr_mem_index = "threadIdx.x + {}".format(counter * self.num_active_threads)
          glob_mem_index = "threadIdx.x + {}".format(counter * self.num_active_threads)
          file.Assignment("{}[{}]".format(self.out_symbol, shr_mem_index),
                          "{}[{}]".format(in_symbol, glob_mem_index))

    # the last hop to fill shared mem with data
    if (self.shm_volume % self.num_active_threads) != 0:
      residue = self.shm_volume - num_hops * self.num_active_threads
      with file.If("threadIdx.x < {}".format(residue)):

        shr_mem_index = "threadIdx.x + {}".format(num_hops * self.num_active_threads)
        glob_mem_index = "threadIdx.x + {}".format(num_hops * self.num_active_threads)
        file.Assignment("{}[{}]".format(self.out_symbol, shr_mem_index),
                        "{}[{}]".format(in_symbol, glob_mem_index))


class ExactPatchLoader(AbstractShrMemLoader):
  """A strategy which loads only a necessary part of a matrix into shared memory.

  """

  def __init__(self, matrix, num_active_threads, load_and_transpose):
    super(ExactPatchLoader, self).__init__(matrix, num_active_threads, load_and_transpose)
    self.lid_dim = self.matrix.get_actual_num_rows()

  def compute_shared_mem_size(self):
      return self.matrix.get_actual_volume()

  def generate_scr(self, file, in_symbol):
    file.Emptyline()
    file.Comment("using ExactPatchLoader")

    file.Pragma("unroll")
    with file.For("int i = 0; i < {}; ++i".format(self.matrix.get_actual_num_cols())):

      num_hops = int(self.lid_dim / self.num_active_threads)
      if num_hops > 0:
        if num_hops > self.manual_unroll_threshold:
          # load using a for-loop
          file.Pragma("unroll")
          with file.For("int counter = 0; counter < {}; ++counter".format(num_hops)):
            shr_mem_index = "threadIdx.x + counter * {} + i * {}".format(self.num_active_threads,
                                                                         self.lid_dim)

            glob_mem_index = "threadIdx.x + counter * {} + i * {}".format(self.num_active_threads,
                                                                          self.matrix.num_rows)
            file.Assignment("{}[{}]".format(self.out_symbol, shr_mem_index),
                            "{}[{}]".format(in_symbol, glob_mem_index))
        else:
          # load using manual loop unrolling
          for counter in range(num_hops):
            offset = counter * self.num_active_threads
            shr_mem_index = "threadIdx.x + {} + i * {}".format(offset, self.lid_dim)

            glob_mem_index = "threadIdx.x + {} + i * {}".format(offset, self.matrix.num_rows)
            file.Assignment("{}[{}]".format(self.out_symbol, shr_mem_index),
                            "{}[{}]".format(in_symbol, glob_mem_index))

      # the last hop to fill shared mem with data
      if (self.lid_dim % self.num_active_threads) != 0:
        residue = self.lid_dim - num_hops * self.num_active_threads
        with file.If("threadIdx.x < {}".format(residue)):
          finial_offset = num_hops * self.num_active_threads
          shr_mem_index = "threadIdx.x + {} + i * {}".format(finial_offset, self.lid_dim)
          glob_mem_index = "threadIdx.x + {} + i * {}".format(finial_offset, self.matrix.num_rows)
          file.Assignment("{}[{}]".format(self.out_symbol, shr_mem_index),
                          "{}[{}]".format(in_symbol, glob_mem_index))
