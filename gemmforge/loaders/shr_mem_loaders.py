from abc import ABC, abstractmethod
from .. import constructs
from .abstract_loader import AbstractShrMemLoader

class ExtendedPatchLoader(AbstractShrMemLoader):
    """A strategy which loads an entire matrix into shared memory.

  """

    def __init__(self, matrix, num_active_threads, load_and_transpose, manufacturer):
        super(ExtendedPatchLoader, self).__init__(matrix, num_active_threads, manufacturer, load_and_transpose)

        full_subvolume = (self.matrix.get_actual_num_cols() - 2) * self.matrix.num_rows
        cropped_subvolume = self.matrix.get_actual_num_rows() + self.matrix.num_rows
        self.shm_volume = cropped_subvolume + full_subvolume

        self.lid_dim = self.matrix.num_rows
        # For better readability
        self.name_treadIdx_x = self.arch_lexic.get_thread_idx_x()

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

                    shr_mem_index = "{} + i * {}".format(self.name_treadIdx_x, self.num_active_threads)
                    glob_mem_index = "{} + i * {}".format(self.name_treadIdx_x, self.num_active_threads)
                    file.Assignment("{}[{}]".format(self.out_symbol, shr_mem_index),
                                    "{}[{}]".format(in_symbol, glob_mem_index))
            else:
                # load using manual loop unrolling
                for counter in range(num_hops):
                    shr_mem_index = "{} + {}".format(self.name_treadIdx_x, counter * self.num_active_threads)
                    glob_mem_index = "{} + {}".format(self.name_treadIdx_x, counter * self.num_active_threads)
                    file.Assignment("{}[{}]".format(self.out_symbol, shr_mem_index),
                                    "{}[{}]".format(in_symbol, glob_mem_index))

        # the last hop to fill shared mem with data
        if (self.shm_volume % self.num_active_threads) != 0:
            residue = self.shm_volume - num_hops * self.num_active_threads
            with file.If("{} < {}".format(self.name_treadIdx_x, residue)):
                shr_mem_index = "{} + {}".format(self.name_treadIdx_x, num_hops * self.num_active_threads)
                glob_mem_index = "{} + {}".format(self.name_treadIdx_x, num_hops * self.num_active_threads)
                file.Assignment("{}[{}]".format(self.out_symbol, shr_mem_index),
                                "{}[{}]".format(in_symbol, glob_mem_index))


class ExactPatchLoader(AbstractShrMemLoader):
    """A strategy which loads only a necessary part of a matrix into shared memory.

  """

    def __init__(self, matrix, num_active_threads, load_and_transpose, manufacturer):
        super(ExactPatchLoader, self).__init__(matrix, num_active_threads, manufacturer, load_and_transpose)
        self.lid_dim = self.matrix.get_actual_num_rows()

        self.name_treadIdx_x = None
        if manufacturer == "nvidia":
            self.name_treadIdx_x = "threadIdx.x"
        elif manufacturer == "amd":
            self.name_treadIdx_x = "hipThreadIdx_x"

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
                        shr_mem_index = "{} + counter * {} + i * {}".format(self.name_treadIdx_x,
                                                                            self.num_active_threads,
                                                                            self.lid_dim)

                        glob_mem_index = "{} + counter * {} + i * {}".format(self.name_treadIdx_x,
                                                                             self.num_active_threads,
                                                                             self.matrix.num_rows)
                        file.Assignment("{}[{}]".format(self.out_symbol, shr_mem_index),
                                        "{}[{}]".format(in_symbol, glob_mem_index))
                else:
                    # load using manual loop unrolling
                    for counter in range(num_hops):
                        offset = counter * self.num_active_threads
                        shr_mem_index = "{} + {} + i * {}".format(self.name_treadIdx_x, offset, self.lid_dim)

                        glob_mem_index = "{} + {} + i * {}".format(self.name_treadIdx_x, offset, self.matrix.num_rows)
                        file.Assignment("{}[{}]".format(self.out_symbol, shr_mem_index),
                                        "{}[{}]".format(in_symbol, glob_mem_index))

            # the last hop to fill shared mem with data
            if (self.lid_dim % self.num_active_threads) != 0:
                residue = self.lid_dim - num_hops * self.num_active_threads
                with file.If("{} < {}".format(self.name_treadIdx_x, residue)):
                    finial_offset = num_hops * self.num_active_threads
                    shr_mem_index = "{} + {} + i * {}".format(self.name_treadIdx_x, finial_offset, self.lid_dim)
                    glob_mem_index = "{} + {} + i * {}".format(self.name_treadIdx_x, finial_offset, self.matrix.num_rows)
                    file.Assignment("{}[{}]".format(self.out_symbol, shr_mem_index),
                                    "{}[{}]".format(in_symbol, glob_mem_index))
