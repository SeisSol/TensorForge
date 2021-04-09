from abc import ABC, abstractmethod
from ..arch_lexic import arch_lexic_factory


class AbstractShrMemLoader(ABC):
    def __init__(self, matrix, num_active_threads, manufacturer, load_and_transpose=False):
        self.matrix = matrix
        self.num_active_threads = num_active_threads
        self.out_symbol = "ShrMat{}".format(self.matrix.name)
        self.load_and_transpose = load_and_transpose
        self.shm_size = None
        self.lid_dim = None
        self.manual_unroll_threshold = 3
        self.arch_lexic = arch_lexic_factory(manufacturer)

    @abstractmethod
    def compute_shared_mem_size(self):
        pass

    @abstractmethod
    def generate_scr(self, file, in_symbol):
        pass

    def get_lid_dim(self):
        return self.lid_dim

    def get_output_symbol(self):
        return self.out_symbol
