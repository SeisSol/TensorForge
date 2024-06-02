from kernelforge.backend.writer import Writer
from kernelforge.backend.symbol import DataView
from .abstract_loader import AbstractShrMemLoader, ShrMemLoaderType
from kernelforge.common.matrix.boundingbox import BoundingBox


class ExtendedPatchLoader(AbstractShrMemLoader):
  """A strategy which loads an entire matrix into shared memory.
  """

  def __init__(self, **kwargs):
    super(ExtendedPatchLoader, self).__init__(**kwargs)
    self._shm_volume = self._tensor.ssp.count_nz()

    src_bbox = self._tensor.get_bbox()
    self._src.data_view = DataView(shape=self._tensor.shape,
                                   permute=None,
                                   bbox=src_bbox)

    dst_bbox = BoundingBox([0] * len(self._tensor.shape), self._tensor.bbox.sizes())
    self._dest.data_view = DataView(shape=self._tensor.shape,
                                    permute=None,
                                    bbox=dst_bbox)

  def get_loader_type(self):
    return ShrMemLoaderType.NOT_TRANSPOSED

  def gen_code(self, writer: Writer):
    super(ExtendedPatchLoader, self).gen_code(writer)
    writer.Comment(f'loading {self._src.name} to {self._dest.name} (extended)')

    # TODO: Nvidia extensions? (i.e. direct copy from main to shared memory)

    src_offset = self._src.data_view.get_offset()
    src_offset_str = f'{src_offset} + ' if src_offset else ''

    num_hops = self._shm_volume // self._num_threads
    num_hops2 = self._shm_volume // (self._num_threads * 2)
    num_hops4 = self._shm_volume // (self._num_threads * 4)

    end1 = num_hops
    end2 = num_hops2 * 2
    end4 = num_hops4 * 4

    # TODO: check when still possible
    # TODO: allow init (i.e. src_offset % 4 == 3 => float1, float4, [...])
    #if src_offset % 4 == 0:
    #  self._write_hop(writer, src_offset_str, 0, end4, 4)
    #  self._write_hop(writer, src_offset_str, end4, end2, 2)
    #  self._write_hop(writer, src_offset_str, end2, end1, 1)
    #elif src_offset % 2 == 0:
    #  self._write_hop(writer, src_offset_str, 0, end2, 2)
    #  self._write_hop(writer, src_offset_str, end2, end1, 1)
    #else:
    #  self._write_hop(writer, src_offset_str, 0, end1, 1)

    self._write_hop(writer, src_offset_str, 0, end1, 1)

    # the last hop to fill shared mem with data
    if (self._shm_volume % self._num_threads) != 0:
      residue = self._shm_volume - num_hops * self._num_threads
      with writer.If(f'{self._vm.get_lexic().thread_idx_x} < {residue}'):
        index = f'{self._vm.get_lexic().thread_idx_x} + {num_hops * self._num_threads}'
        lhs = f'{self._dest.name}[{index}]'
        rhs = f'{self._src.name}[{src_offset_str}{index}]'
        writer(f'{lhs} = {rhs};')

  def _write_hop(self, writer, src_offset_str, start, end, increment):
    if end > start:
      if increment > 1:
        vectortype = self._vm.get_lexic().get_fptype(self._context.fp_type, increment)
        typeprefix = f'*({vectortype}*)&'
      else:
        typeprefix = ''
      if (end - start) / increment > self._manual_unroll_threshold:
        # load using a for-loop
        writer.insert_pragma_unroll()
        with writer.For(f'int i = {start}; i < {end}; i += {increment}'):
          index = f'{increment} * {self._vm.get_lexic().thread_idx_x} + i * {self._num_threads}'
          lhs = f'{typeprefix}{self._dest.name}[{index}]'
          rhs = f'{typeprefix}{self._src.name}[{src_offset_str}{index}]'
          writer(f'{lhs} = {rhs};')
      else:
        # load using manual loop unrolling
        for counter in range(start, end, increment):
          index = f'{increment} * {self._vm.get_lexic().thread_idx_x} + {counter * self._num_threads}'
          lhs = f'{typeprefix}{self._dest.name}[{index}]'
          rhs = f'{typeprefix}{self._src.name}[{src_offset_str}{index}]'
          writer(f'{lhs} = {rhs};')

  def __str__(self):
    return f'{self._dest.name} = load_g2s_ext {self._shr_mem.name}, {self._src.name};'


class ExactPatchLoader(AbstractShrMemLoader):
  """A strategy which loads only a necessary part of a matrix into shared memory.
  """

  def __init__(self, **kwargs):
    super(ExactPatchLoader, self).__init__(**kwargs)
    self._shm_volume = self._tensor.get_actual_volume()
    self._src.data_view = DataView(shape = self._tensor.shape,
                                   permute = None,
                                   bbox = self._tensor.get_bbox())

    self._dest.data_view = DataView(shape=self._tensor.get_bbox().sizes(),
                                    permute=None)

  def get_loader_type(self):
    return ShrMemLoaderType.NOT_TRANSPOSED

  def gen_code(self, writer: Writer):
    super(ExactPatchLoader, self).gen_code(writer)
    writer(f'// loading {self._src.name} to {self._dest.name} (exact)')

    num_data_rows = self._src.data_view.get_dim_size(0)
    src_offset = self._src.data_view.get_offset()
    src_offset = f'{src_offset} + ' if src_offset else ''

    writer.insert_pragma_unroll()
    with writer.For(f'int i = 0; i < {self._src.data_view.get_dim_size(1)}; ++i'):

      num_hops = int(num_data_rows / self._num_threads)
      if num_hops > 0:
        if num_hops > self._manual_unroll_threshold:

          writer.insert_pragma_unroll()
          with writer.For(f'int counter = 0; counter < {num_hops}; ++counter'):
            shr_mem_index = f'{self._vm.get_lexic().thread_idx_x} + '
            shr_mem_index += f'counter * {self._num_threads} + i * {self._dest.data_view.get_lead_dim()}'
            lhs = f'{self._dest.name}[{shr_mem_index}]'

            glob_mem_index = f'{self._vm.get_lexic().thread_idx_x} + '
            glob_mem_index += f'counter * {self._num_threads} + i * {self._src.data_view.get_lead_dim()}'
            rhs = f'{self._src.name}[{src_offset}{glob_mem_index}]'
            writer(f'{lhs} = {rhs};')
        else:
          for counter in range(num_hops):
            shr_mem_index = f'{self._vm.get_lexic().thread_idx_x} + '
            shr_mem_index += f'{counter * self._num_threads} + i * {self._dest.data_view.get_lead_dim()}'
            lhs = f'{self._dest.name}[{shr_mem_index}]'

            glob_mem_index = f'{self._vm.get_lexic().thread_idx_x} + '
            glob_mem_index += f'{counter * self._num_threads} + i * {self._src.data_view.get_lead_dim()}'
            rhs = f'{self._src.name}[{src_offset}{glob_mem_index}]'
            writer(f'{lhs} = {rhs};')

      # the last hop to fill shared mem with data
      if (num_data_rows % self._num_threads) != 0:
        residue = num_data_rows - num_hops * self._num_threads
        with writer.If(f'{self._vm.get_lexic().thread_idx_x} < {residue}'):
          finial_offset = num_hops * self._num_threads
          shr_mem_index = f'{self._vm.get_lexic().thread_idx_x} + {finial_offset} + i * {self._dest.data_view.get_lead_dim()}'
          lhs = f'{self._dest.name}[{shr_mem_index}]'

          glb_mem_index = f'{self._vm.get_lexic().thread_idx_x} + {finial_offset} + i * {self._src.data_view.get_lead_dim()}'
          rhs = f'{self._src.name}[{src_offset}{glb_mem_index}]'
          writer(f'{lhs} = {rhs};')

  def __str__(self):
    return f'{self._dest.name} = load_g2s {self._shr_mem.name}, {self._src.name};'
