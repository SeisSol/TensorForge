from ..common import *
from ..cache import GpuRoutineGenerator
from ..common import BatchedOperationsAux
from tensorforge.common.basic_types import DataFlowDirection



class CopyScaleAddGenerator(object):
  def __init__(self, arch, descr):
    self._arch = arch
    self._descr = descr

  def _formatTerm(self, alpha, term):
    """Generate a sub-string of a term for a source code which is going to be used
    inside of the inner most for-loop

    Args:
      alpha (Union[Scalar, float]): TODO
      term (IndexedTensorDescription): TODO

    Returns:

    Examples:
      >>> from tensorforge.memory import DenseMemoryLayout
      >>> from tensorforge.ast.indices import Indices
      >>> from tensorforge.codegen.common import IndexedTensorDescription
      >>> from tensorforge.aspp import dense
      >>> from tensorforge.codegen.copyscaleadd.generic import Generic
      >>> tensor_shape = (5, 6)
      >>> layout = DenseMemoryLayout(shape=tensor_shape)
      >>> indices = Indices(indexNames='ij', shape=tensor_shape)
      >>> description = IndexedTensorDescription(name='A', \
                                                 indices=indices, \
                                                 memoryLayout=layout, \
                                                 eqspp=dense(shape=tensor_shape))
      >>> obj = Generic(arch='dummy', descr=description)
      >>> obj._formatTerm(alpha=3, term=description)
      '3 * A[1*i + 5*j]'
    """

    prefix = ''
    if alpha == 0.0:
      return ''

    if alpha == 1.0:
      prefix = term.name
    else:
      prefix = '{} * {}'.format(alpha, term.name)

    return '{}[{}]'.format(prefix, term.memoryLayout.addressString(term.indices))

  def generate(self, cpp, routineCache):
    """Generates a tensor equation of a form: B = beta * B + alpha * A
    Args:
      cpp (IO): a file stream
      routineCache:

    Returns:

    """
    d = self._descr  # type: copyscaleadd.Description
    m = d.loopRanges[d.result.indices[0]]
    n = d.loopRanges[d.result.indices[1]]
    alpha = d.alpha

    aux = BatchedOperationsAux(self._arch.typename)
    matrix_a = tensorforge.interface.YatetoInterface.gen_matrix((m, n),
                                                        d.term.memoryLayout.bbox(),
                                                        addressing=aux.deduce_addresing(d.term),
                                                        transpose=False,
                                                        is_tmp=False,
                                                        name='A')

    matrix_b = tensorforge.interface.YatetoInterface.gen_matrix((m, n),
                                                        d.result.memoryLayout.bbox(),
                                                        addressing=aux.deduce_addresing(d.result),
                                                        transpose=False,
                                                        is_tmp=False,
                                                        name='B')

    matrix_a.set_data_flow_direction(DataFlowDirection.SOURCE)
    matrix_b.set_data_flow_direction(DataFlowDirection.SINK)

    try:
      vm = tensorforge.common.context.Context(self._arch.name, self._arch.backend, fp_type=tensorforge.common.basic_types.FloatingPointType.str2enum(self._arch.typename))
      forge_generator = tensorforge.generators.generator.Generator([
        tensorforge.generators.descriptions.CSADescr(False, matrix_a, matrix_b, alpha, d.beta, False, False)
      ], vm)
      forge_generator.register()
      routine_name = forge_generator.get_base_name()

      args = [str(alpha),
              aux.deduce_arg(d.term, as_const=True),
              aux.deduce_arg(d.result),
              BatchedOperationsAux.NUM_ELEMENTS_NAME,
              BatchedOperationsAux.FLAGS_NAME,
              BatchedOperationsAux.STREAM_PTR_NAME]
      cpp("launcher_{}({});".format(routine_name, ', '.join(args)))

      routineCache.addRoutine(routine_name, TensorForgeWriter(forge_generator, vm.get_vm().get_headers()))

    except tensorforge.common.exceptions.GenerationError as err:
      print("ERROR: {}".format(err))
      raise err

    return m.size() * n.size()


class TensorForgeWriter(GpuRoutineGenerator):
  def __init__(self, forge_generator, headers):
    self._generator = forge_generator
    self._basename = forge_generator.get_base_name()
    self._headers = headers

  def __eq__(self, other):
    if isinstance(other, TensorForgeWriter):
      return self._basename == other._basename
    else:
      return False

  def header(self, cpp):
    cpp.includes(self._headers)

  def __call__(self, routineName, fileName):
    self._generator.generate()
    declaration = self._generator.get_header()
    launcher = self._generator.get_launcher()
    kernel = self._generator.get_kernel()

    with open(fileName, "a") as file:
      file.write(kernel)
      file.write(launcher)

    return declaration
