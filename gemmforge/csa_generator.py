from . import constructs
from io import StringIO
from .exceptions import GenerationError
from .abstract_gemmlike_generator import GemmLikeGenerator
from .abstract_generator import AbstractGenerator as Generator
from .initializers import initializer_factory, StubInitializer
from .arch_lexic import arch_lexic_factory
import math
import hashlib
from copy import deepcopy



class CsaGenerator(GemmLikeGenerator):
    """ Copy-Add-Scale Generator: B = beta * B + alpha * A, where alpha is a real number
  and beta is either 1.0 or 0.0
  """

    def __init__(self, arch, precision):
        super(CsaGenerator, self).__init__(arch, precision)
        self._mat_b_initializer = None
        self.arch_lexic = arch_lexic_factory(arch.manufacturer)
        # For better readability of the remaining code
        self.TEAM_INDEX_STR = self.arch_lexic.get_tid_counter(self.arch_lexic.get_thread_idx_z(),
                                                              self.arch_lexic.get_block_dim_z(),
                                                              self.arch_lexic.get_block_idx_x())
        self.name_threadIdx_x = self.arch_lexic.get_thread_idx_x()

    def generate(self, mat_a, mat_b, alpha, beta, base_name=None):
        self.mat_a = mat_a
        self.mat_a._set_name('A')
        self.mat_a._set_mutability(False)

        self.mat_b = mat_b
        self.mat_b._set_name('B')
        self.mat_b._set_mutability(True)

        self._matrices = [self.mat_a, self.mat_b]

        self.alpha = alpha
        self.beta = beta

        self.base_name = base_name if base_name is not None else self._generate_base_name()

        if self.beta == 0.0:
            self._mat_b_initializer = initializer_factory(self.beta,
                                                          deepcopy(self.mat_b),
                                                          self.arch,
                                                          self.precision)
        else:
            self._mat_b_initializer = StubInitializer(self.arch, self.precision)
        self._mat_b_initializer.generate()

        self._check()
        self._analyze()

        self._generate_kernel()
        self._generate_header()
        self._generate_launcher()

    def _check(self):
        try:

            if self.mat_a.transpose:
                raise GenerationError("Cannot generate a copy-add-scale op. "
                                      "Matrix A is transpose")

            if self.mat_b.transpose:
                raise GenerationError("Cannot generate a copy-add-scale op. "
                                      "Matrix B is transpose")

            is_inside = lambda rst_range, term_range: term_range[0] >= rst_range[0] and term_range[1] <= rst_range[1]
            make_range = lambda bbox, dim: [bbox[dim], bbox[dim + 2]]

            for dim in range(2):
                if not is_inside(make_range(self.mat_b.bbox, dim), make_range(self.mat_a.bbox, dim)):
                    raise GenerationError(f"Cannot generate a copy-add-scale op. "
                                          "Data of mat. A (term) in not inside of mat. B (result)")

        except GenerationError as error:
            matrices = {"A": self.mat_a, "B": self.mat_b}
            for name in matrices:
                print("Matrix {}:".format(name))
                print(matrices[name])
                print("=" * 80)
            raise error

    def _analyze(self):
        lid_dim_length = self.mat_a.get_actual_num_rows()

        # we use active threads to add a single column
        num_vector_units_required = math.ceil(lid_dim_length / self.arch.vec_unit_length)
        self.num_compute_threads = lid_dim_length
        self.num_active_threads = num_vector_units_required * self.arch.vec_unit_length

        total_num_threas_per_op = self.num_active_threads * self.mat_a.get_actual_num_cols()

        self.max_num_regs_per_thread = 10
        mults_wrt_num_regs = self.arch.max_reg_per_block / (total_num_threas_per_op * self.max_num_regs_per_thread)
        self.num_mult_per_block = max(int(mults_wrt_num_regs / self.arch.max_block_per_sm), 1)

    def _generate_kernel(self):
        glob_symbols = {}
        for matrix in [self.mat_a, self.mat_b]:
            glob_symbols[matrix.name] = "GlobMat{}".format(matrix.name)

        src = StringIO()
        with constructs.Cpp(src) as file:
            total_num_threas_per_op = self.num_active_threads * self.mat_a.get_actual_num_cols()
            max_num_threads_per_block = total_num_threas_per_op * self.num_mult_per_block
            kernel_bounds = [max_num_threads_per_block]
            with self.arch_lexic.kernel_definition(file, kernel_bounds, self.base_name, self._get_func_params()):
                with file.If("{} < {}".format(self.TEAM_INDEX_STR, Generator.NUM_ELEMENTS_STR)):

                    # declare ptrs for correct matrices
                    file.VariableDeclaration("const {}*".format(self.precision),
                                             glob_symbols[self.mat_a.name],
                                             self._get_global_matrix_ptr(self.mat_a))

                    # we need to change bbox of matrix because it may not
                    # coincidence with bbox of matrix a (bbox of is larger than bbox of a)
                    view = deepcopy(self.mat_b)
                    view.bbox = self.mat_a.bbox

                    file.VariableDeclaration("{}*".format(self.precision),
                                             glob_symbols[self.mat_b.name],
                                             self._get_global_matrix_ptr(view))

                    with file.If("{} < {}".format(self.name_threadIdx_x, self.mat_a.get_actual_num_rows())):
                        if self.beta == 0.0:
                            file.Assignment(f'{glob_symbols[self.mat_b.name]}'
                                            f'[{self.name_threadIdx_x}]',
                                            f'Scale * {glob_symbols[self.mat_a.name]}'
                                            f'[{self.name_threadIdx_x}]')
                        elif self.beta == 1.0:
                            file.Accumulate(f'{glob_symbols[self.mat_b.name]}'
                                            f'[{self.name_threadIdx_x}]',
                                            f'Scale * {glob_symbols[self.mat_a.name]}'
                                            f'[{self.name_threadIdx_x}]')
                        elif self.beta == -1.0:
                            file.Deaccumulate(f'{glob_symbols[self.mat_b.name]}'
                                              f'[{self.name_threadIdx_x}]',
                                              f'Scale * {glob_symbols[self.mat_a.name]}'
                                              f'[{self.name_threadIdx_x}]')
                        else:
                            rhs = f'Scale * {glob_symbols[self.mat_a.name]}' \
                                  f'[{self.name_threadIdx_x}]' \
                                  f' + {self.beta}' \
                                  f' * {glob_symbols[self.mat_b.name]}[{self.name_threadIdx_x}] '
                            file.Assignment(left=f'{glob_symbols[self.mat_b.name]}[{self.name_threadIdx_x}]', right=rhs)

            self._kernel = src.getvalue()
            self._kernel += self._mat_b_initializer.get_kernel()

    def _generate_launcher(self):
        self._launcher = self._mat_b_initializer.get_launcher()

        src = StringIO()
        with constructs.Cpp(src) as file:
            with file.Function(self.base_name, self._get_launcher_params()):
                # prepare arguments for the initializer of matrix "b"
                initializer_args = [self.mat_b.name,
                                    self._generate_extra_offset_symbol(self.mat_b),
                                    "NumElements",
                                    Generator.STREAM_PTR_STR]

                # pass an additional argument if "beta" is known at run-time only
                if not isinstance(self.beta, float):
                    initializer_args.insert(0, self.beta)

                # call the initializer. Note: the initializer can be a stub i.e. will do nothing
                file("{}".format(self._mat_b_initializer.func_call(initializer_args)))

                file.VariableDeclaration(self.arch_lexic.kernel_range_object(), self._get_block_dim_spec())
                file.VariableDeclaration(self.arch_lexic.kernel_range_object(), self._get_grid_dim_spec())

                self.arch_lexic.get_stream_via_pointer(file, "streamPtr", Generator.STREAM_PTR_STR)
                file.Expression(self.arch_lexic.get_launch_code(self.base_name,
                                                                "Grid",
                                                                "Block",
                                                                "stream",
                                                                self._get_func_args()))
                err = self.arch_lexic.check_error()
                if err is not None:
                    file.Expression(err)
            self._launcher += src.getvalue()

    def _generate_header(self):
        src = StringIO()
        with constructs.Cpp(src) as file:
            file.FunctionDeclaration(self.base_name, self._get_launcher_params(with_defaults=True))
            content = src.getvalue()
        self._header = content

    def _generate_base_name(self):
        dim1 = f'm{self.mat_a.get_actual_num_rows()}_{self.mat_b.num_rows}'
        dim2 = f'n{self.mat_a.get_actual_num_cols()}_{self.mat_b.num_rows}'
        dims = f'{dim1}_{dim2}'
        addressing = f'{self.mat_b.addressing[0]}{self.mat_a.addressing[0]}'

        constants = f'{self.alpha}_{self.beta}'
        result = hashlib.md5(f'{constants}_{self.mat_a.__str__()}{self.mat_b.__str__()}'.encode())
        md5encoding = result.hexdigest()

        prefix = 's' if self.precision == "float" else "d"
        return "{}copyAddScale_{}_{}_{}".format(prefix,
                                                dims,
                                                addressing,
                                                md5encoding[:Generator.ENCODING_LENGTH])

    def _get_func_params(self):
        return f'{self.precision} Scale, {super(CsaGenerator, self)._get_func_params()}'

    def _get_launcher_params(self, with_defaults=False):
        return f'{self.precision} Scale, {super(CsaGenerator, self)._get_launcher_params(with_defaults)}'

    def _get_func_args(self):
        return f'Scale, {super(CsaGenerator, self)._get_func_args()}'

    def _get_block_dim_spec(self):
        super(CsaGenerator, self)._get_block_dim_spec()
        return f'Block({self.num_active_threads}, {self.mat_a.get_actual_num_cols()}, {self.num_mult_per_block})'

    def _get_grid_dim_spec(self):
        super(CsaGenerator, self)._get_grid_dim_spec()
        num_blocks = "({0} + {1} - 1) / {1}".format(Generator.NUM_ELEMENTS_STR,
                                                    self.num_mult_per_block)
        return f'Grid({num_blocks}, 1, 1)'

    def _get_global_matrix_ptr(self, matrix):
        extra_offset_symbol = self._generate_extra_offset_symbol(matrix)
        offset_to_row = f'{self.arch_lexic.get_thread_idx_y()} * {matrix.num_rows}'

        if matrix.addressing == "strided":
            main_offset = "{} * {}".format(self.TEAM_INDEX_STR, matrix.get_real_volume())
            sub_offset = matrix.get_offset_to_first_element()
            return "&{}[{} + {} + {} + {}]".format(matrix.name,
                                                   extra_offset_symbol,
                                                   main_offset,
                                                   sub_offset,
                                                   offset_to_row)

        elif matrix.addressing == "pointer_based":
            main_offset = self.TEAM_INDEX_STR
            sub_offset = matrix.get_offset_to_first_element()
            return "&{}[{}][{} + {} + {}]".format(matrix.name,
                                                  main_offset,
                                                  extra_offset_symbol,
                                                  sub_offset,
                                                  offset_to_row)

        else:
            sub_offset = matrix.get_offset_to_first_element()
            return "&{}[{} + {} + {}]".format(matrix.name, sub_offset, extra_offset_symbol, offset_to_row)
