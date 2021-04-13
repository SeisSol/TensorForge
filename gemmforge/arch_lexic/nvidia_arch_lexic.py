from .abstract_arch_lexic import AbstractArchLexic


class NvidiaArchLexic(AbstractArchLexic):

    def __init__(self):
        AbstractArchLexic.__init__(self)
        self.threadIdx_y = "threadIdx.y"
        self.threadIdx_x = "threadIdx.x"
        self.threadIdx_z = "threadIdx.z"
        self.blockIdx_x = "blockIdx.x"
        self.blockDim_y = "blockDim.y"
        self.blockDim_z = "blockDim.z"
        self.stream_name = "cudaStream_t"

    def get_launch_code(self, func_name, grid, block, stream, func_params):
        return "kernel_{}<<<{},{},0,{}>>>({})".format(func_name, grid, block, stream, func_params)

    def declare_shared_memory_inline(self, name, precision, size):
        return f"__shared__ {precision} {name}[{size}]"

    def kernel_definition(self, file, kernel_bounds, base_name, params, precision=None, total_shared_mem_size=None):
        return file.CudaKernel(base_name, params, kernel_bounds)

    def sync_threads(self):
        return "__syncthreads()"

    def kernel_range_object(self):
        return "dim3"

    def get_stream_via_pointer(self, file, stream_name, pointer_name):
        if_stream_exists = f'({pointer_name} != nullptr)'
        stream_obj = f'static_cast<{self.get_stream_name()}>({pointer_name})'
        file(f'{self.get_stream_name()} stream = {if_stream_exists} ? {stream_obj} : 0;')

    def check_error(self):
        return "CHECK_ERR"
