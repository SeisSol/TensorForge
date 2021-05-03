from .. import constructs
from io import StringIO
import math
import hashlib

from ..arch_lexic import arch_lexic_factory
from gemmforge.abstract_generator import AbstractGenerator


class ExactInitializer(AbstractGenerator):

    def team_index_str(self):
        # return "(threadIdx.z + blockDim.z * blockIdx.x)"
        return f"({self.arch_lexic.get_thread_idx_z()} + {self.arch_lexic.get_block_dim_z()} * {self.arch_lexic.get_block_idx_x()})"

    def __init__(self, init_value, matrix, arch, precision):
        super(ExactInitializer, self).__init__(arch, precision)
        self.arch_lexic = arch_lexic_factory(arch.manufacturer)
        self.init_value = init_value
        self.matrix = matrix
        self.matrix._set_name('A')
        self.matrix._set_mutability(True)
        self._matrices = [self.matrix]

    def generate(self, base_name=None):
        self.base_name = base_name if base_name is not None else self._generate_base_name()

        self._check()
        self._analyze()

        self._generate_kernel()
        self._generate_header()
        self._generate_launcher()

    def _check(self):
        pass

    def _analyze(self):
        lid_dim_length = self.matrix.get_actual_num_rows()

        # we use active threads to add a single column
        num_vector_units_required = math.ceil(lid_dim_length / self.arch.vec_unit_length)
        self.num_compute_threads = lid_dim_length
        self.num_active_threads = num_vector_units_required * self.arch.vec_unit_length

        total_num_threas_per_op = self.num_active_threads * self.matrix.get_actual_num_cols()

        self.max_num_regs_per_thread = 10
        mults_wrt_num_regs = self.arch.max_reg_per_block / (total_num_threas_per_op * self.max_num_regs_per_thread)
        self.num_mult_per_block = max(int(mults_wrt_num_regs / self.arch.max_block_per_sm), 1)

    def _generate_kernel(self):
        global_symbols = {self.matrix.name: f'Glob{self.matrix.name}'}
        src = StringIO()
        with constructs.Cpp(src) as file:
            total_num_threas_per_op = self.num_active_threads * self.matrix.get_actual_num_cols()
            max_num_threads_per_block = total_num_threas_per_op * self.num_mult_per_block
            kernel_bounds = [max_num_threads_per_block]
            with self.arch_lexic.kernel_definition(file, kernel_bounds, self.base_name, self._get_func_params()):
                with file.If("{} < {}".format(self.team_index_str(), AbstractGenerator.NUM_ELEMENTS_STR)):
                    # declare ptrs for correct matrices
                    file.VariableDeclaration("{}*".format(self.precision),
                                             global_symbols[self.matrix.name],
                                             self._get_global_matrix_ptr(self.matrix))

                    # assign initial value to a matrix element
                    with file.If(
                            "{} < {}".format(self.arch_lexic.get_thread_idx_x(), self.matrix.get_actual_num_rows())):
                        file.Assignment(f'{global_symbols[self.matrix.name]}[{self.arch_lexic.get_thread_idx_x()}]',
                                        f'{self.init_value}')

            self._kernel = src.getvalue()

    def _generate_launcher(self):
        src = StringIO()
        with constructs.Cpp(src) as file:
            with file.Function(self.base_name, self._get_launcher_params()):
                file.VariableDeclaration(self.arch_lexic.kernel_range_object(), self._get_block_dim_spec())
                file.VariableDeclaration(self.arch_lexic.kernel_range_object(), self._get_grid_dim_spec())

                self.arch_lexic.get_stream_via_pointer(file, "stream", AbstractGenerator.STREAM_PTR_STR)
                file.Expression(self.arch_lexic.get_launch_code(self.base_name,
                                                                "Grid",
                                                                "Block",
                                                                "stream",
                                                                self._get_func_args()))
                err = self.arch_lexic.check_error()
                if err is not None:
                    file.Expression(err)

            self._launcher = src.getvalue()

    def _generate_header(self):
        src = StringIO()
        with constructs.Cpp(src) as file:
            file.FunctionDeclaration(self.base_name,  self._get_launcher_params(with_defaults=True))
            content = src.getvalue()
        self._header = content

    def _generate_base_name(self):
        dim = f'm{self.matrix.get_actual_num_rows()}_{self.matrix.num_rows}'
        addressing = f'{self.matrix.addressing[0]}'

        result = hashlib.md5(f'{self.init_value}_{self.matrix.__str__()}'.encode())
        md5encoding = result.hexdigest()

        return "initialize_{}_{}_{}".format(dim,
                                            addressing,
                                            md5encoding[:AbstractGenerator.ENCODING_LENGTH])

    def _get_func_params(self):
        base_params = super(ExactInitializer, self)._get_func_params()
        if isinstance(self.init_value, float):
            return base_params
        else:
            return f'{self.precision} {self.init_value}, {base_params}'

    def _get_launcher_params(self, with_defaults=False):
        base_params = super(ExactInitializer, self)._get_launcher_params(with_defaults)
        if isinstance(self.init_value, float):
            return base_params
        else:
            return f'{self.precision} {self.init_value}, {base_params}'

    def _get_func_args(self):
        base_args = super(ExactInitializer, self)._get_func_args()
        if isinstance(self.init_value, float):
            return base_args
        else:
            return f'{self.init_value}, {base_args}'

    def _get_block_dim_spec(self):
        super(ExactInitializer, self)._get_block_dim_spec()
        return f'Block({self.num_active_threads}, {self.matrix.get_actual_num_cols()}, {self.num_mult_per_block})'

    def _get_grid_dim_spec(self):
        super(ExactInitializer, self)._get_grid_dim_spec()
        num_blocks = "({0} + {1} - 1) / {1}".format(AbstractGenerator.NUM_ELEMENTS_STR,
                                                    self.num_mult_per_block)
        return f'Grid({num_blocks}, 1, 1)'

    def _get_global_matrix_ptr(self, matrix):
        extra_offset_symbol = self._generate_extra_offset_symbol(matrix)
        offset_to_row = f'{self.arch_lexic.get_thread_idx_y()} * {matrix.num_rows}'

        if matrix.addressing == "strided":
            main_offset = "{} * {}".format(self.team_index_str(), matrix.get_real_volume())
            sub_offset = matrix.get_offset_to_first_element()
            return "&{}[{} + {} + {} + {}]".format(matrix.name,
                                                   extra_offset_symbol,
                                                   main_offset,
                                                   sub_offset,
                                                   offset_to_row)

        elif matrix.addressing == "pointer_based":
            main_offset = self.team_index_str()
            sub_offset = matrix.get_offset_to_first_element()
            return "&{}[{}][{} + {} + {}]".format(matrix.name,
                                                  main_offset,
                                                  extra_offset_symbol,
                                                  sub_offset,
                                                  offset_to_row)

        else:
            sub_offset = matrix.get_offset_to_first_element()
            return "&{}[{} + {} + {}]".format(matrix.name, sub_offset, extra_offset_symbol, offset_to_row)

    def func_call(self, args):
        return f'{self.base_name}({", ".join(args)});'
