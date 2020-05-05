class Architecture:
    def __init__(self,
                 vec_unit_length,
                 max_local_mem_size_per_block,
                 max_num_threads,
                 max_reg_per_block,
                 max_block_per_sm):

        self.vec_unit_length = vec_unit_length
        self.max_local_mem_size_per_block = max_local_mem_size_per_block
        self.max_num_threads = max_num_threads
        self.max_reg_per_block = max_reg_per_block
        self.max_block_per_sm = max_block_per_sm

def produce(name):
    if name == "nvidia":
        arch = Architecture(vec_unit_length=32,
                            max_local_mem_size_per_block=49152,
                            max_num_threads=1024,
                            max_reg_per_block=65536,
                            max_block_per_sm=16)
        return arch
    else:
        raise ValueError("Unknown gpu architecture")