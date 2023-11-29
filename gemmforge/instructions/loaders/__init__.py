from gemmforge.symbol_table import SymbolType
from gemmforge.exceptions import InternalError
from .shr_mem_loaders import ExtendedPatchLoader, ExactPatchLoader
from .shr_transpose_mem_loaders import ExtendedTransposePatchLoader, ExactTransposePatchLoader
from math import ceil


def shm_mem_loader_factory(vm, dest, src, shr_mem, num_threads, load_and_transpose=False):
  params = {'vm': vm,
            'dest': dest,
            'src': src,
            'shr_mem': shr_mem,
            'num_threads': num_threads,
            'load_and_transpose': load_and_transpose}

  num_loads_per_column = ceil(src.data_view.rows / num_threads) * num_threads

  if src.data_view.lead_dim > num_loads_per_column:
    if load_and_transpose:
      return ExactTransposePatchLoader(**params)
    else:
      return ExactPatchLoader(**params)
  else:
    if load_and_transpose:
      return ExtendedTransposePatchLoader(**params)
    else:
      return ExtendedPatchLoader(**params)
