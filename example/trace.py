import sys
sys.path.append('..')
from tensorforge.common.matrix.boundingbox import BoundingBox
from tensorforge.common.matrix.tensor import Tensor, SubTensor
from tensorforge.common.context import Context
from tensorforge.generators.descriptions import MultilinearDescr
from tensorforge.common.basic_types import FloatingPointType, Addressing
from tensorforge.generators.generator import Generator


# Q += tr(A)

mat_a = SubTensor(Tensor([32, 32], Addressing.PTR_BASED, BoundingBox([0,10], [32,30])), BoundingBox([0,10], [32,30]))

mat_b = SubTensor(Tensor([], Addressing.PTR_BASED, BoundingBox([0], [32])), BoundingBox([0], [32]))

gemm_list = [MultilinearDescr(mat_b, [mat_a], [[-1, -1]], [[0, 1]])]

context = Context(arch='sm_60',
                  backend='cuda',
                  fp_type=FloatingPointType.FLOAT)

generator = Generator(gemm_list, context)
generator.generate()

with_output = True
if with_output:
  print(generator.get_header())
  print(generator.default_generate_call_site())
  print()
  print(generator.get_launcher())
  print()
  print(generator.get_kernel())
