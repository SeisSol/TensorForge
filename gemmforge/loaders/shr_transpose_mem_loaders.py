from abc import ABC, abstractmethod
from .. import constructs
from .abstract_loader import AbstractShrMemLoader


def _find_next_prime(number):
  factor = 2
  return _find_prime_in_range(number, factor * number)


def _find_prime_in_range(source, target):
  for number in range(source, target):
    for i in range(2, number):
      if number % i == 0:
        break
    else:
      return number


class ExtendedTransposePatchLoader(AbstractShrMemLoader):

  def __init__(self, matrix, num_active_threads, load_and_transpose):
    super(ExtendedTransposePatchLoader, self).__init__(matrix,
                                                       num_active_threads,
                                                       load_and_transpose)

    optimal_num_cols = _find_next_prime(self.matrix.get_actual_num_cols())
    self.shm_volume = optimal_num_cols * self.matrix.num_rows
    self.lid_dim = optimal_num_cols

  def compute_shared_mem_size(self):
    return self.shm_volume

  def generate_scr(self, file, in_symbol):

    file.Emptyline()
    file.Comment("using ExtendedTransposePatchLoader")
    num_hops = int(self.shm_volume / self.num_active_threads)
    file.VariableDeclaration("int", "Index")
    if num_hops > 0:
      if num_hops > self.manual_unroll_threshold:
        # load using a for-loop

        file.Pragma("unroll")
        with file.For("int i = 0; i < {}; ++i".format(num_hops)):

          file.Assignment("Index",
                          "threadIdx.x + i * {}".format(self.num_active_threads))

          shr_mem_index = "(Index % {}) * {} + Index / {}".format(self.matrix.num_rows,
                                                                  self.lid_dim,
                                                                  self.matrix.num_rows)

          glob_mem_index = "threadIdx.x + i * {}".format(self.num_active_threads)
          file.Assignment("{}[{}]".format(self.out_symbol, shr_mem_index),
                          "{}[{}]".format(in_symbol, glob_mem_index))
      else:
        # load using manual loop unrolling
        counter = 0

        while counter < num_hops:

          file.Assignment("Index",
                          "threadIdx.x + {}".format(counter * self.num_active_threads))
          shr_mem_index = "(Index % {}) * {} + Index / {}".format(self.matrix.num_rows,
                                                                  self.lid_dim,
                                                                  self.matrix.num_rows)

          glob_mem_index = "threadIdx.x + {}".format(counter * self.num_active_threads)
          file.Assignment("{}[{}]".format(self.out_symbol, shr_mem_index),
                          "{}[{}]".format(in_symbol, glob_mem_index))
          counter += 1

    # the last hop to fill shared mem with data
    if (self.shm_volume % self.num_active_threads) != 0:
      residue = self.shm_volume - num_hops * self.num_active_threads

      with file.If("threadIdx.x < {}".format(residue)):
        file.Assignment("Index",
                        "threadIdx.x + {}".format(num_hops * self.num_active_threads))

        shr_mem_index = "(Index % {}) * {} + Index / {}".format(self.matrix.num_rows,
                                                                self.lid_dim,
                                                                self.matrix.num_rows)

        glob_mem_index = "threadIdx.x + {}".format(num_hops * self.num_active_threads)
        file.Assignment("{}[{}]".format(self.out_symbol, shr_mem_index),
                        "{}[{}]".format(in_symbol, glob_mem_index))


class ExactTransposePatchLoader(AbstractShrMemLoader):
  """A strategy which loads only a necessary part of a matrix into shared memory.

  """

  def __init__(self, matrix, num_active_threads, load_and_transpose):
    super(ExactTransposePatchLoader, self).__init__(matrix, num_active_threads, load_and_transpose)

    optimal_num_cols = _find_next_prime(self.matrix.get_actual_num_cols())
    self.shm_volume = optimal_num_cols * self.matrix.num_rows
    self.lid_dim = optimal_num_cols

  def compute_shared_mem_size(self):
      return self.shm_volume

  def generate_scr(self, file, in_symbol):
    file.Emptyline()
    file.Comment("using ExactTransposePatchLoader")

    file.Pragma("unroll")
    with file.For("int i = 0; i < {}; ++i".format(self.matrix.get_actual_num_cols())):
      num_hops = int(self.matrix.get_actual_num_rows() / self.num_active_threads)
      if num_hops > 0:
        # load using a for-loop
        file.Pragma("unroll")
        with file.For("int Counter = 0; Counter < {}; ++Counter".format(num_hops)):
          thread_idx = "threadIdx.x + Counter * {}".format(self.num_active_threads,
                                                           self.matrix.get_actual_num_rows())

          file.VariableDeclaration("int", "Index",
                                   "{} + i * {}".format(thread_idx,
                                                        self.matrix.get_actual_num_rows()))

          shr_mem_index = "(Index % {}) * {} + Index / {}".format(self.matrix.get_actual_num_rows(),
                                                                  self.lid_dim,
                                                                  self.matrix.get_actual_num_rows())

          glob_mem_index = "{} + i * {}".format(thread_idx,
                                                self.matrix.num_rows)

          file.Assignment("{}[{}]".format(self.out_symbol, shr_mem_index),
                          "{}[{}]".format(in_symbol, glob_mem_index))

      # the last hop to fill shared mem with data
      if (self.matrix.get_actual_num_rows() % self.num_active_threads) != 0:
        residue = self.matrix.get_actual_num_rows() - num_hops * self.num_active_threads
        with file.If("threadIdx.x < {}".format(residue)):
          finial_offset = num_hops * self.num_active_threads
          thread_idx = "threadIdx.x + {}".format(finial_offset)

          file.VariableDeclaration("int", "Index",
                                   "{} + i * {}".format(thread_idx,
                                                        self.matrix.get_actual_num_rows()))

          shr_mem_index = "(Index % {}) * {} + Index / {}".format(self.matrix.get_actual_num_rows(),
                                                                  self.lid_dim,
                                                                  self.matrix.get_actual_num_rows())

          glob_mem_index = "{} + i * {}".format(thread_idx,
                                                self.matrix.num_rows)

          file.Assignment("{}[{}]".format(self.out_symbol, shr_mem_index),
                          "{}[{}]".format(in_symbol, glob_mem_index))
