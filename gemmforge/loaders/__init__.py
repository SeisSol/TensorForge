from .stub_shr_mem_loader import StubLoader
from .shr_mem_loaders import ExtendedPatchLoader, ExactPatchLoader
from .shr_transpose_mem_loaders import ExtendedTransposePatchLoader, ExactTransposePatchLoader
from math import ceil


def shm_mem_factory(vm, matrix, num_active_threads, load_and_transpose,):
    # Use an extended loader if the tail of a active threads can touch the next column
    # Otherwise, use an exact one
    num_loads_per_column = ceil(matrix.get_actual_num_rows() / num_active_threads) * num_active_threads
    if matrix.num_rows > num_loads_per_column:
        if load_and_transpose:
            return ExactTransposePatchLoader(vm, matrix, num_active_threads, load_and_transpose)
        else:
            return ExactPatchLoader(vm, matrix, num_active_threads, load_and_transpose)
    else:
        if load_and_transpose:
            return ExtendedTransposePatchLoader(vm, matrix, num_active_threads, load_and_transpose)
        else:
            return ExtendedPatchLoader(vm, matrix, num_active_threads, load_and_transpose)
