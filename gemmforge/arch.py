class Architecture:
    def __init__(self,
                 vec_unit_length,
                 max_local_mem_size_per_block,
                 max_num_threads,
                 max_reg_per_block,
                 max_threads_per_sm,
                 max_block_per_sm):

        self.vec_unit_length = vec_unit_length
        self.max_local_mem_size_per_block = max_local_mem_size_per_block
        self.max_num_threads = max_num_threads
        self.max_reg_per_block = max_reg_per_block
        self.max_threads_per_sm = max_threads_per_sm
        self.max_block_per_sm = max_block_per_sm

def produce(name, sub_name):
    KB = 1024
    if name == "nvidia":
      # from: https://en.wikipedia.org/wiki/CUDA
      nvidia_warp = 32
      max_reg_per_block = 64 * KB
      max_num_threads = 1024 # per block TODO: rename
      max_threads_per_sm = 2048
      max_block_per_sm = 32

      if sub_name in ['sm_60', 'sm_61', 'sm_62']:
        max_local_mem_size_per_block = 48 * KB
      elif sub_name == 'sm_70':
        max_local_mem_size_per_block = 96 * KB
      elif sub_name == 'sm_71':
        max_local_mem_size_per_block = 48 * KB
      elif sub_name == 'sm_75':
        max_block_per_sm = 16
        max_threads_per_sm = 1024
        max_local_mem_size_per_block = 64 * KB

      else:
        raise ValueError(f'Given nvidia SM model is not supported. Provided: {sub_name}')

      return Architecture(nvidia_warp,
                          max_local_mem_size_per_block,
                          max_num_threads,
                          max_reg_per_block,
                          max_threads_per_sm,
                          max_block_per_sm)
    else:
        raise ValueError('Unknown gpu architecture')
