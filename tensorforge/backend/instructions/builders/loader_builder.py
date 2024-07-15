from tensorforge.common.context import Context
from tensorforge.backend.scopes import Scopes, Symbol
from tensorforge.common.matrix.matrix import Matrix
from tensorforge.backend.symbol import SymbolType, DataView
from tensorforge.backend.instructions.memory.load import GlbToShrLoader
from tensorforge.common.exceptions import InternalError
from tensorforge.common.basic_types import GeneralLexicon
from .abstract_builder import AbstractBuilder
from .allocator_builder import ShrMemAllocBuilder
from tensorforge.backend.instructions.ptr_manip import GetElementPtr
from tensorforge.backend.data_types import ShrMemObject

class GlobalLoaderBuilder(AbstractBuilder):
  def __init__(self, context: Context, scopes: Scopes, shrmem_obj: ShrMemObject, num_threads):
    super(GlobalLoaderBuilder, self).__init__(context, scopes)
    self.shrmem_obj = shrmem_obj
    self.num_threads = num_threads

  def build(self, src: Symbol):
    self._reset()

    assert src.stype == SymbolType.Batch

    predest = Symbol(name=f'ptr_{GeneralLexicon.GLOBAL_MEM_PREFIX}{src.name}',
                    stype=SymbolType.Global,
                    obj=src.obj)
    predest.data_view = DataView(shape=src.obj.shape, permute=None, bbox=src.obj.get_bbox())
    
    self._scopes.add_symbol(predest)
    self._instructions.append(GetElementPtr(self._context, src, predest, True))

    dest = Symbol(name=f'{GeneralLexicon.GLOBAL_MEM_PREFIX}{src.name}',
                    stype=SymbolType.SharedMem,
                    obj=src.obj)

    self._scopes.add_symbol(dest)

    shrmem_symbol = self._scopes.get_symbol(self.shrmem_obj)
    loader = GlbToShrLoader(context=self._context, src=predest, dest=dest, shr_mem=shrmem_symbol, num_threads=self.num_threads, permute=None, blockwide=True, max_load_offset=0)

    self._instructions.append(loader)

    offset = self.shrmem_obj.alloc_global(loader.compute_shared_mem_size())
    loader.set_shr_mem_offset(offset, True, True)

