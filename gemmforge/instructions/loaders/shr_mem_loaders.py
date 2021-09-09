from abc import ABC, abstractmethod
from gemmforge import constructs
from .abstract_loader import AbstractShrMemLoader
from gemmforge.symbol_table import SymbolType, Symbol, DataView
from copy import deepcopy


class ExtendedPatchLoader(AbstractShrMemLoader):
  """A strategy which loads an entire matrix into shared memory
  including padding (caused by a difference between lead_dim and
  number of rows)
  """

  def __init__(self, **kwargs):
    super(ExtendedPatchLoader, self).__init__(**kwargs)

    data_view = self._src.data_view
    full_subvolume = (data_view.columns - 2) * data_view.lead_dim
    cropped_subvolume = data_view.rows + data_view.lead_dim
    self._shm_volume = cropped_subvolume + full_subvolume
    self._dest.data_view = deepcopy(self._src.data_view)

  def gen_code(self, writer):
    super(ExtendedPatchLoader, self).gen_code(writer)
    writer("// using ExtendedPatchLoader")

    with writer.Scope():

      thread_idx_x = self._lexic.thread_idx_x
      num_hops = int(self._shm_volume / self._num_threads)
      if num_hops > 0:
        if num_hops > self._manual_unroll_threshold:
          # load using a for-loop
          writer.Pragma("unroll")
          with writer.For(f'int i = 0; i < {num_hops}; ++i'):
            shr_mem_addr = f'{thread_idx_x} + i * {self._num_threads}'
            glb_mem_addr = f'{thread_idx_x} + i * {self._num_threads}'

            self._assign(writer, shr_mem_addr, glb_mem_addr)
        else:
          # load using manual loop unrolling
          for counter in range(num_hops):
            shr_mem_addr = f'{thread_idx_x} + {self._num_threads * counter}'
            glb_mem_addr = f'{thread_idx_x} + {self._num_threads * counter}'

            self._assign(writer, shr_mem_addr, glb_mem_addr)

      # the last hop to fill shared mem with data
      if (self._shm_volume % self._num_threads) != 0:
        residue = self._shm_volume - num_hops * self._num_threads
        with writer.If(f'{thread_idx_x} < {residue}'):
          shr_mem_addr = f'{thread_idx_x} + {num_hops * self._num_threads}'
          glb_mem_addr = f'{thread_idx_x} + {num_hops * self._num_threads}'

          self._assign(writer, shr_mem_addr, glb_mem_addr)


class ExactPatchLoader(AbstractShrMemLoader):
  """A strategy which loads only a necessary part of a matrix into shared memory.

  """

  def __init__(self, **kwargs):
    super(ExactPatchLoader, self).__init__(**kwargs)
    data_view = self._src.data_view
    self._shm_volume = data_view.rows * data_view.columns

    self._dest.data_view = deepcopy(self._src.data_view)
    self._dest.data_view.lead_dim = data_view.rows

  def gen_code(self, writer):
    super(ExactPatchLoader, self).gen_code(writer)
    writer("// using ExactPatchLoader")

    with writer.Scope():

      src_data_view = self._src.data_view
      dest_data_view = self._dest.data_view
      thread_idx_x = self._lexic.thread_idx_x
      writer.Pragma("unroll")
      with writer.For(f'int i = 0; i < {src_data_view.columns}; ++i'):
        num_hops = int(dest_data_view.lead_dim / self._num_threads)
        if num_hops > 0:
          if num_hops > self._manual_unroll_threshold:
            writer.Pragma("unroll")
            with writer.For(f'int counter = 0; counter < {num_hops}; ++counter'):
              shr_mem_addr = f'{thread_idx_x}'
              shr_mem_addr += f' + counter * {self._num_threads} + i {dest_data_view.lead_dim}'

              glb_mem_addr = f'{thread_idx_x}'
              glb_mem_addr += f' + counter * {self._num_threads} + i {src_data_view.lead_dim}'

              self._assign(writer, shr_mem_addr, glb_mem_addr)
          else:
            for counter in range(num_hops):
              offset = counter * self._num_threads
              shr_mem_addr = f'{thread_idx_x} + {offset} + i * {dest_data_view.lead_dim}'
              glb_mem_addr = f'{thread_idx_x} + {offset} + i * {src_data_view.lead_dim}'
              self._assign(writer, shr_mem_addr, glb_mem_addr)

        # the last hop to fill shared mem with data
        if (dest_data_view.lead_dim % self._num_threads) != 0:
          residue = dest_data_view.lead_dim - num_hops * self._num_threads
          with writer.If(f'{thread_idx_x} < {residue}'):
            finial_offset = num_hops * self._num_threads
            shr_mem_addr = f'{thread_idx_x} + {finial_offset} + i * {dest_data_view.lead_dim}'
            glb_mem_addr = f'{thread_idx_x} + {finial_offset} + i * {src_data_view.lead_dim}'
            self._assign(writer, shr_mem_addr, glb_mem_addr)
