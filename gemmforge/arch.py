class Architecture:
    def __init__(self,
                 vec_unit_length,
                 max_local_mem_size_per_block,
                 max_num_threads,
                 max_reg_per_block,
                 max_threads_per_sm,
                 max_block_per_sm,
                 name):
        self.vec_unit_length = vec_unit_length
        self.max_local_mem_size_per_block = max_local_mem_size_per_block
        self.max_num_threads = max_num_threads
        self.max_reg_per_block = max_reg_per_block
        self.max_threads_per_sm = max_threads_per_sm
        self.max_block_per_sm = max_block_per_sm
        self.manufacturer = name


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
                          max_block_per_sm,
                          name)

    elif name == "amd":
        
        if sub_name in ['gfx906']:
            #MI50
            # warpSize equals a wavefront for AMD (Page 31 from
            # https://www.olcf.ornl.gov/wp-content/uploads/2019/10/ORNL_Application_Readiness_Workshop-AMD_GPU_Basics.pdf)
            amd_wavefront = 64
            # Page 16 at https://developer.amd.com/wordpress/media/2017/08/Vega_Shader_ISA_28July2017.pdf
            max_reg_per_workgroup = 256 * KB
            # maxThreadsPerBlock
            max_num_threads = 1024
            # a sm at Nvidia corresponds to a CU at AMD (https://en.wikipedia.org/wiki/Graphics_Core_Next#CU_scheduler)
            # Block == Workgroup(Page 31 from
            # https://www.olcf.ornl.gov/wp-content/uploads/2019/10/ORNL_Application_Readiness_Workshop-AMD_GPU_Basics.pdf)
            # (https://rocmdocs.amd.com/en/latest/Programming_Guides/Opencl-optimization.html?#specific-guidelines-for-gcn-family-gpus)
            max_workgroup_per_cu = 40
            # 4 SIMD Vector Units, each 16-lane wide (https://en.wikipedia.org/wiki/Graphics_Core_Next#Compute_units)
            # Currently not used
            max_vec_units_per_cu = 64
            # sharedMemPerBlock
            max_local_mem_size_per_workgroup = 64 * KB
        elif sub_name in ['gfx908']:
            #MI100
            amd_wavefront = 64
            max_reg_per_workgroup = 512 * KB
            max_num_threads = 1024
            max_workgroup_per_cu = 40
            #Still 4 SIMD 16 Lanes Wide
            max_vec_units_per_cu = 64
            max_local_mem_size_per_workgroup = 64 * KB
        else:
            raise ValueError(f'Given amd CU model is not supported. Provided: {sub_name}')

        return Architecture(amd_wavefront,
                            max_local_mem_size_per_workgroup,
                            max_num_threads,
                            max_reg_per_workgroup,
                            max_vec_units_per_cu,
                            max_workgroup_per_cu,
                            name)

    else:
        raise ValueError('Unknown gpu architecture')
