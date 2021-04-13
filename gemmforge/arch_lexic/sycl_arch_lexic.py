from .abstract_arch_lexic import AbstractArchLexic


class SyclArchLexic(AbstractArchLexic):

    def __init__(self):
        AbstractArchLexic.__init__(self)
        self.threadIdx_y = "item.get_local_id(1)"
        self.threadIdx_x = "item.get_local_id(0)"
        self.threadIdx_z = "item.get_local_id(2)"
        self.blockIdx_x = "item.get_group().get_id(0)"
        self.blockDim_y = "item.get_group().get_local_range(1)"
        self.blockDim_z = "item.get_group().get_local_range(2)"
        self.stream_name = "cl::sycl::queue"

    def get_launch_code(self, func_name, grid, block, stream, func_params):
        return f"kernel_{func_name}({stream}, {grid}, {block}, {func_params})"

    def declare_shared_memory_inline(self, name, precision, size):
        return None

    def kernel_definition(self, file, kernel_bounds, base_name, params, precision=None, total_shared_mem_size=None):
        localmem = None

        if total_shared_mem_size is not None and precision is not None:
            localmem = "cl::sycl::accessor<{}, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> Scratch ({}, cgh);".format(precision, total_shared_mem_size)

        return file.SyclKernel(base_name, params, kernel_bounds, localmem)

    def sync_threads(self):
        return "item.barrier()"

    def kernel_range_object(self):
        return "cl::sycl::range<3>"

    def get_stream_via_pointer(self, file, stream_name, pointer_name):
        with file.If(f"{pointer_name} == nullptr"):
            file.Expression("throw std::invalid_argument(\"stream may not be null!\")")

        stream_obj = f'static_cast<{self.get_stream_name()} *>({pointer_name})'
        file(f'{self.get_stream_name()} *stream = {stream_obj};')

    def check_error(self):
        return None
