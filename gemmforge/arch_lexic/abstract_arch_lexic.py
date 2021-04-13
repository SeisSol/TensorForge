from abc import ABC, abstractmethod


class AbstractArchLexic(ABC):
    """
    You can use this abstract class to add a dictionary for any manufacturer for variables like e.g. threadIdx.x for
    CUDA that are used by the generators and loaders
    """

    def __init__(self):
        self.threadIdx_x = None
        self.threadIdx_y = None
        self.threadIdx_z = None
        self.blockDim_y = None
        self.blockDim_z = None
        self.blockIdx_x = None
        self.stream_name = None

    def get_tid_counter(self, thread_id, block_dim, block_id):
        return "(" + thread_id + " + " + block_dim + " * " + block_id + ")"

    def get_thread_idx_x(self):
        return self.threadIdx_x

    def get_thread_idx_y(self):
        return self.threadIdx_y

    def get_thread_idx_z(self):
        return self.threadIdx_z

    def get_block_dim_y(self):
        return self.blockDim_y

    def get_block_dim_z(self):
        return self.blockDim_z

    def get_block_idx_x(self):
        return self.blockIdx_x

    def get_stream_name(self):
        return self.stream_name

    @abstractmethod
    def get_launch_code(self, func_name, grid, block, stream, func_params):
        pass

    @abstractmethod
    def declare_shared_memory_inline(self, name, precision, size):
        pass

    @abstractmethod
    def kernel_definition(self, file, kernel_bounds, base_name, params, precision=None, total_shared_mem_size=None):
        pass

    @abstractmethod
    def sync_threads(self):
        pass

    @abstractmethod
    def kernel_range_object(self):
        pass

    @abstractmethod
    def get_stream_via_pointer(self, file, stream_name, pointer_name):
        pass

    @abstractmethod
    def check_error(self):
        pass

